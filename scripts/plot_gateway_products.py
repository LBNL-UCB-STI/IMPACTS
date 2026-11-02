"""Render existing classified traffic products using IMPACTS map primitives.

Read-only with respect to model products; all artifacts go under figures/.
No dispersion, fleet expansion, unit conversion, or exposure recalculation.
"""
from __future__ import annotations

import argparse
import gc
import json
import textwrap
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
import yaml

from impacts.pipeline.postprocess import _common as style
from impacts.pipeline.postprocess import step4_plot_concentrations as native
import matplotlib.pyplot as plt


POLLUTANTS = {'TotalPM25': 'Total PM₂.₅', 'BC': 'Black carbon', 'NO2': 'NO₂'}
FIELDS = [*POLLUTANTS, 'PrimaryPM25', 'SecondaryPM25',
          'aermod_PrimaryPM25_scaled', 'inmap_PrimaryPM25', 'inmap_SecondaryPM25']
MAIN = {'Passenger': 'Passenger', 'PortRelated': 'Port-related freight',
        'NonPort': 'Non-port-related freight'}
NOTE = 'Traffic ×10; no buses or power generation; partial ISRM source coverage'


def make_layout(bounds):
    """Same native regular-grid mapping as _grid_raster_layout, from streamed bounds."""
    bounds = np.asarray(bounds)
    if not len(bounds) or not np.isfinite(bounds).all():
        raise ValueError('Empty/nonfinite receptor geometry')
    widths = bounds[:, 2] - bounds[:, 0]
    heights = bounds[:, 3] - bounds[:, 1]
    if not np.allclose(widths, 100.) or not np.allclose(heights, 100.):
        raise ValueError('Expected unmodified 100-m receptor squares')
    xmin, ymin = bounds[:, :2].min(axis=0)
    cols = np.rint((bounds[:, 0] - xmin) / 100).astype(np.int32)
    rows = np.rint((bounds[:, 1] - ymin) / 100).astype(np.int32)
    nx, ny = int(cols.max()) + 1, int(rows.max()) + 1
    if len(np.unique(rows.astype(np.int64) * nx + cols)) != len(rows):
        raise ValueError('Duplicate raster receptor positions')
    return style._GridRasterLayout(rows, cols, (ny, nx),
                                   (float(xmin), float(xmin + nx*100), float(ymin), float(ymin + ny*100)))


def load_context(root, reference, *, manifest_path=None):
    inputs = yaml.safe_load((manifest_path or root/'oakland_port/preprocess_manifest.yaml').read_text())['inputs']
    region_path = Path(inputs['county_boundaries']['staged_path'])
    network_path = region_path.parent/'beam_osm_mapped.parquet'
    pf = pq.ParquetFile(reference)
    geo = json.loads(pf.schema_arrow.metadata[b'geo'])
    crs = geo['columns']['geometry']['crs']
    print('Reading county boundaries', flush=True)
    counties = gpd.read_file(region_path).to_crs(crs)
    name_col = next(c for c in ['County', 'NAME', 'NAMELSAD'] if c in counties)
    names = counties[name_col].str.replace(r'\s+County$', '', regex=True)
    counties = counties.loc[names.isin(native._SF_BAY_COUNTIES)].copy()
    if counties.empty:
        raise ValueError('No SF Bay county polygons')
    region = counties.geometry.union_all()
    shapely.prepare(region)
    masks, ids, selected_bounds = [], [], []
    print('Streaming receptor geometry', flush=True)
    for batch in pf.iter_batches(columns=['aermod_cell_id', 'geometry'], batch_size=131072):
        bb = shapely.bounds(shapely.from_wkb(batch['geometry'].to_numpy(zero_copy_only=False)))
        keep = shapely.contains_xy(region, (bb[:, 0]+bb[:, 2])/2, (bb[:, 1]+bb[:, 3])/2)
        masks.append(keep)
        ids.append(batch['aermod_cell_id'].to_numpy())
        selected_bounds.append(bb[keep])
    mask = np.concatenate(masks)
    layout = make_layout(np.concatenate(selected_bounds))
    print('Reading and clipping offline road network', flush=True)
    net = gpd.read_parquet(network_path, columns=['geometry']).drop_duplicates().to_crs(crs)
    net = net.loc[shapely.intersects(region, net.geometry.to_numpy())].copy()
    # Most roads are wholly inside the region: intersect only boundary-crossing
    # segments, preserving the same exact clipped geometry without repeated work.
    boundary_crossing = ~shapely.covers(region, net.geometry.to_numpy())
    net.loc[boundary_crossing, 'geometry'] = shapely.intersection(
        net.loc[boundary_crossing].geometry.to_numpy(), region)
    net = net.loc[~net.geometry.is_empty]
    return mask, np.concatenate(ids), layout, net, counties, {
        'region': str(region_path), 'network': str(network_path),
        'full_receptors': len(mask), 'sfbay_receptors': int(mask.sum()),
        'clip_rule': '100-m receptor centers inside nine-county union; no concentration interpolation',
    }


def load_values(path, mask, ids):
    pf = pq.ParquetFile(path)
    if pf.metadata.num_rows != len(ids):
        raise ValueError(f'Receptor count mismatch: {path}')
    selected = []
    offset = 0
    for batch in pf.iter_batches(columns=['aermod_cell_id', *FIELDS], batch_size=131072):
        end = offset + len(batch)
        np.testing.assert_array_equal(batch['aermod_cell_id'].to_numpy(), ids[offset:end])
        d = batch.to_pandas().drop(columns='aermod_cell_id')
        selected.append(d.loc[mask[offset:end]])
        offset = end
    data = pd.concat(selected, ignore_index=True)
    if not np.isfinite(data.to_numpy()).all() or data.to_numpy().min() < -1e-10:
        raise ValueError(f'Invalid concentrations: {path}')
    np.testing.assert_allclose(data.TotalPM25, data.PrimaryPM25 + data.SecondaryPM25, atol=1e-10)
    data = native._add_pm25_component_columns(data)
    np.testing.assert_allclose(data.TotalPM25, data.PM25_ThreeComponent_Total, atol=1e-10)
    return data


def map_axis(ax, data, column, layout, net, counties, vmax, title, fontsize=20):
    ax.set_aspect('equal')
    style._plot_raster_layer(ax, data, column, style.CMAP_GREEN_RED, layout=layout, vmax=vmax)
    style._add_network(ax, net)
    style._add_basemap(ax)
    lims = ax.get_xlim(), ax.get_ylim()
    counties.boundary.plot(ax=ax, color='#333333', linewidth=.4, alpha=.5, zorder=4)
    ax.set_xlim(lims[0]); ax.set_ylim(lims[1])
    ax.set_title(title, fontsize=fontsize, pad=16)
    ax.set_axis_off()


def positive_limit(data, cols):
    return native._shared_positive_vmax(data, cols, .99) or 1.


def finish(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path.name}', flush=True)


def single_map(data, column, title, name, layout, net, counties, vmax, path, dpi):
    fig, ax = plt.subplots(figsize=style.MAP_FIGSIZE)
    fig.canvas.draw_idle = lambda: None
    map_axis(ax, data, column, layout, net, counties, vmax,
             '\n'.join(textwrap.wrap(name, 65))+'\n'+title, style.MAP_TITLE_FONTSIZE)
    style._add_colorbar(fig, ax, style.CMAP_GREEN_RED, vmax, title+' (µg/m³)')
    fig.tight_layout(pad=.5)
    fig.text(.02, -.015, NOTE+'\nColor ceiling: positive-cell 99th percentile; higher values saturate.', fontsize=13)
    finish(fig, path, dpi)


def component_comparison(data, name, layout, net, counties, path, dpi):
    vmax = positive_limit(data, ['PrimaryPM25', 'SecondaryPM25'])
    fig, axes = plt.subplots(1, 2, figsize=(32, 18))
    fig.canvas.draw_idle = lambda: None
    for ax, col, title in zip(axes, ['PrimaryPM25', 'SecondaryPM25'], ['Primary PM₂.₅', 'Secondary PM₂.₅']):
        map_axis(ax, data, col, layout, net, counties, vmax, title, 25)
    fig.suptitle(name+' — Primary vs secondary PM₂.₅', fontsize=28)
    fig.subplots_adjust(right=.9, top=.92, bottom=.07, wspace=.02)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=style.CMAP_GREEN_RED)
    cbar = fig.colorbar(sm, cax=fig.add_axes([.92, .2, .018, .6]))
    cbar.set_label('PM₂.₅ (µg/m³)', fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    fig.text(.04, .02, NOTE+' | Shared positive-cell P99 color ceiling', fontsize=13)
    finish(fig, path, dpi)


def comparison_figures(groups, summary, layout, net, counties, load, destination, dpi, *, comparison_title=None):
    n = len(groups)
    ncols = min(3,n) if n <= 3 else 4
    nrows = (n+ncols-1)//ncols
    limits = {p: max(summary[g]['limits'][p] for g in groups) for p in POLLUTANTS}
    figures = {}
    for p in POLLUTANTS:
        fig, axes = plt.subplots(nrows, ncols, figsize=(8*ncols, 8.7*nrows), squeeze=False)
        # GeoPandas calls draw_idle after every overlay. Under the Agg backend
        # that redraws all preceding panels, giving quadratic rendering work.
        # savefig performs the full final draw; suppress only intermediate draws.
        fig.canvas.draw_idle = lambda: None
        figures[p] = (fig, axes.ravel())
    for i, group in enumerate(groups):
        print(f'Comparison panel {i+1}/{n}: {group}', flush=True)
        data = load(group)
        name = summary[group]['name']
        for p, (fig, axes) in figures.items():
            map_axis(axes[i], data, p, layout, net, counties, limits[p],
                     '\n'.join(textwrap.wrap(name, 31)), 17)
        del data
        gc.collect()
    for p, (fig, axes) in figures.items():
        for ax in axes[n:]:
            ax.set_axis_off()
        fig.suptitle(POLLUTANTS[p]+' — '+(comparison_title or ('Main traffic categories' if n == 3 else 'Port / airport freight groups')), fontsize=26)
        fig.subplots_adjust(left=.02, right=.91, top=.83 if n <= 3 else .90, bottom=.1 if n <= 3 else .045, wspace=.02, hspace=.12)
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, limits[p]), cmap=style.CMAP_GREEN_RED),
                            cax=fig.add_axes([.93, .25, .012, .5]))
        cbar.set_label(POLLUTANTS[p]+' (µg/m³)', fontsize=20)
        cbar.ax.tick_params(labelsize=15)
        fig.text(.03, .025 if n <= 3 else .012, NOTE+'\nCommon color ceiling = maximum of group positive-cell P99; higher values saturate.', fontsize=14)
        finish(fig, destination/f'{p.lower()}_common_scale.png', dpi)
    return limits


def pwc_chart(rows, title, path, dpi):
    labels = [r.get('gateway_name', MAIN.get(r['group'], r['group'])) for r in rows]
    labels = ['\n'.join(textwrap.wrap(s, 32)) for s in labels]
    height = max(5., len(rows)*.64)
    fig, axes = plt.subplots(1, 3, figsize=(22, height), sharey=True)
    y = np.arange(len(rows))
    for ax, (p, label) in zip(axes, POLLUTANTS.items()):
        values = np.array([r[p+'_PWC_ug_m3'] for r in rows])
        ax.barh(y, values, color=style.CMAP_GREEN_RED(.78))
        ax.set_title(label, fontsize=18)
        ax.set_xlabel('Population-weighted concentration (µg/m³)', fontsize=12)
        ax.set_xlim(0, values.max()*1.28)
        for yy, val in zip(y, values):
            ax.text(val+values.max()*.015, yy, f'{val:.6f}', va='center', fontsize=10)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y, labels, fontsize=11)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=20)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.text(.02, .02, NOTE+'; annual population-weighted contributions, not grid maxima.', fontsize=11)
    finish(fig, path, dpi)


def run(root, dpi=150, only=None, summaries_only=False):
    root = root.resolve()
    out = root/'figures'
    out.mkdir(exist_ok=True)
    results = json.loads((root/'results_summary.json').read_text())
    gateways = results['gateway_groups']
    sources = {g: root/f'products/main_categories/{g}_concentrations.parquet' for g in MAIN}
    sources.update({r['group']: root/f"products/{r['group']}_concentrations.parquet" for r in gateways})
    names = {**MAIN, **{r['group']: r['gateway_name'] for r in gateways}}
    signatures = {g: (p.stat().st_size, p.stat().st_mtime_ns) for g, p in sources.items()}
    print('Preparing shared SF Bay 100-m grid and offline network', flush=True)
    mask, ids, layout, net, counties, context = load_context(root, sources['Passenger'])
    print(context, flush=True)
    summary = {}
    chosen = [only] if only else list(sources)
    for g in chosen:
        print(f'Rendering {g}', flush=True)
        data = load_values(sources[g], mask, ids)
        folder = out/('main_categories' if g in MAIN else 'gateways')/g
        limits = {p: positive_limit(data, [p]) for p in POLLUTANTS}
        summary[g] = dict(name=names[g], source=str(sources[g]), limits=limits,
                          true_sfbay_maxima={p: float(data[p].max()) for p in POLLUTANTS})
        if summaries_only:
            if g in MAIN:
                summary[g]['component_shared_vmax'] = positive_limit(data, [c for c, _, _, _ in native._PM25_COMPONENT_LAYERS])
            del data
            gc.collect()
            continue
        for p, title in POLLUTANTS.items():
            single_map(data, p, title, names[g], layout, net, counties, limits[p], folder/f'{p.lower()}.png', dpi)
        if g in MAIN:
            component_comparison(data, names[g], layout, net, counties, folder/'primary_secondary_pm25.png', dpi)
            vmax = positive_limit(data, [c for c, _, _, _ in native._PM25_COMPONENT_LAYERS])
            summary[g]['component_shared_vmax'] = vmax
            for col, title, filename, _ in native._PM25_COMPONENT_LAYERS:
                single_map(data, col, title, names[g], layout, net, counties, vmax, folder/filename, dpi)
        del data
        gc.collect()
    if not only:
        load = lambda g: load_values(sources[g], mask, ids)
        scales = {}
        scales['main'] = comparison_figures(list(MAIN), summary, layout, net, counties, load, out/'summary/main_categories', dpi)
        # Sort by PM2.5 PWC, but retain every disjoint combination/unknown group.
        ordered = sorted(gateways, key=lambda r: r['TotalPM25_PWC_ug_m3'], reverse=True)
        scales['gateways'] = comparison_figures([r['group'] for r in ordered], summary, layout, net, counties, load, out/'summary/gateways', dpi)
        pwc_chart(results['main_categories'], 'Main traffic categories — population-weighted annual concentration', out/'summary/main_categories_pwc.png', dpi)
        pwc_chart(ordered, 'Port / airport freight — population-weighted annual concentration', out/'summary/gateway_pwc.png', dpi)
        assert signatures == {g: (p.stat().st_size, p.stat().st_mtime_ns) for g, p in sources.items()}
        from PIL import Image
        verified = []
        for path in out.rglob('*.png'):
            with Image.open(path) as im:
                size = im.size
                im.verify()
            verified.append(dict(file=str(path.relative_to(out)), pixels=size))
        if len(verified) != 74:
            raise ValueError(f'Expected 74 figures, found {len(verified)}')
        manifest = dict(status='complete', context=context, groups=summary, shared_scales=scales,
                        concentration_units='ug/m3', no2_label_basis='User-confirmed 2026-10-09; no numeric/unit conversion applied',
                        native_helpers='impacts.pipeline.postprocess._common and step4_plot_concentrations',
                        dpi=dpi, fleet_scale_already_applied=10, source_products_unchanged=True,
                        power_generation=False, transit=False, files=verified)
        (out/'figure_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (out/'README.txt').write_text(
            'IMPACTS classified traffic figures\n\n'
            'summary/main_categories: 3 side-by-side maps, shared scales across main categories.\n'
            'summary/gateways: all 14 disjoint gateway groups, shared scales per pollutant.\n'
            'summary/*_pwc.png: Bay Area population-weighted annual concentration contributions.\n'
            'main_categories/<group>: PM2.5, BC, NO2, primary/secondary comparison and four-component maps.\n'
            'gateways/<group>: PM2.5, BC and NO2 concentration maps.\n\n'
            'Units: ug/m3. NO2 label follows user confirmation; no numeric conversion applied.\n'
            'Traffic is already expanded 10x. No further scaling. No buses/power generation.\n'
            'Partial ISRM coverage retained. Vehicle/tour association, not port-bound trip segments.\n'
            '100-m grid cells retained inside the nine-county SF Bay region (cell-center selection).\n'
            'Individual maps: positive-cell 99th-percentile color ceiling; higher values saturate.\n'
            'Comparison maps: shared ceiling, maximum of member-group positive-cell P99 values.\n'
            'PM2.5 component maps share a color ceiling within each category.\n'
            'Color ceilings are not data maxima; both are recorded in figure_manifest.json.\n'
            'Offline road network and county boundaries; no remote API tiles or plant markers.\n'
            'Existing numerical results were not changed.\n', encoding='utf-8')
        print(f'COMPLETE: {len(verified)} verified PNGs in {out}', flush=True)


def refresh_gateway_summary(root, dpi=150):
    """Layout-only refresh without rerendering the 66 individual group maps."""
    root = root.resolve()
    out = root/'figures'
    manifest_path = out/'figure_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    results = json.loads((root/'results_summary.json').read_text())
    mask, ids, layout, net, counties, _ = load_context(root, root/'products/main_categories/Passenger_concentrations.parquet')
    ordered = sorted(results['gateway_groups'], key=lambda r: r['TotalPM25_PWC_ug_m3'], reverse=True)
    load = lambda g: load_values(root/f'products/{g}_concentrations.parquet', mask, ids)
    comparison_figures([r['group'] for r in ordered], manifest['groups'], layout, net, counties,
                       load, out/'summary/gateways', dpi)
    from PIL import Image
    for entry in manifest['files']:
        with Image.open(out/entry['file']) as im:
            entry['pixels'] = list(im.size)
            im.verify()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('COMPLETE: refreshed gateway layout; all 74 PNGs verified', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--dpi', type=int, default=150)
    parser.add_argument('--only', help='Render one group for QA before the complete run')
    parser.add_argument('--summaries-only', action='store_true', help='Preserve completed single-group maps and rebuild summaries')
    parser.add_argument('--gateway-summary-only', action='store_true', help='Refresh only the three gateway summary maps')
    args = parser.parse_args()
    if args.gateway_summary_only:
        refresh_gateway_summary(args.root, args.dpi)
    else:
        run(args.root, args.dpi, args.only, args.summaries_only)
