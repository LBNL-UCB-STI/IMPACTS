"""Continue a saved BEAM charging run with hourly Cambium/eGRID allocation.

Uses the existing BREATHE allocator and IMPACTS downstream implementations.
All generated artifacts are placed in a new, explicit output directory.
"""
from __future__ import annotations

from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import argparse
import hashlib
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from impacts.pipeline.workflow.hourly_power import select_sfbay_plants

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / 'examples/pilates/impacts/impacts_output'
BREATHE = BREATHE_ROOT
BASE = OUTPUTS / 'impacts-postsim--sfbay--base--20260901-165739'
DEMAND = OUTPUTS / 'beam_charging_fc07_v260419_it3'
POLLUTANTS = ['NOx', 'SOx', 'CO2', 'CH4', 'N2O', 'CO2e', 'PM2_5', 'NH3', 'VOC']
BAY_COUNTY_FIPS = {'001', '013', '041', '055', '075', '081', '085', '095', '097'}




def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def dispatch(out):
    from impacts.pipeline.workflow.hourly_power import dispatch_hourly
    dispatch_hourly(Path(out) / 'power_generation',
        hourly_demand_file=DEMAND / 'sfbay_hourly_charging_demand.parquet',
        charging_summary_file=DEMAND / 'beam_charging_summary.json', breathe_root=BREATHE, electricity_year=2040, balancing_area='p9', cambium_scenario='LowRECost')


def power_isrm(out, isrm_store, power_method='matrix', source_layer_index=1):
    if power_method != 'matrix':
        raise ValueError('Only hourly-dispatch emissions and matrix multiplication are supported.')
    from impacts.pipeline.workflow.hourly_power import compute_hourly_power_isrm
    return compute_hourly_power_isrm(Path(out) / 'power_generation',
        inmap_grid_path=str(BASE / 'preprocess/inmap_study_area.parquet'),
        counties_path=str(BASE / 'preprocess/county_boundaries.gpkg'),
        isrm_store=isrm_store, source_layer_index=source_layer_index)


def integrate(out):
    import geopandas as gpd
    from impacts.pipeline.workflow.step4_prepare_exposure import run
    from impacts.manifest.schema import PipelineConfig
    manifest = yaml.safe_load((BASE / 'pipeline_manifest.yaml').read_text())
    preprocess = yaml.safe_load((BASE / 'preprocess_manifest.yaml').read_text())
    (out / 'exposure').mkdir(exist_ok=True)
    run(pipeline=PipelineConfig.from_dict(manifest['pipeline']), raw_dir=out / 'exposure',
        inmap_concentrations_path=manifest['outputs']['beam_inmap_concentrations'],
        aermod_concentrations_path=manifest['outputs']['beam_aermod_concentrations'],
        power_generation_concentrations_path=str(out / 'power_generation/power_generation_isrm_concentrations.parquet'),
        manifest_inputs=preprocess['inputs'])


def postprocess(out):
    import gc
    from impacts.pipeline.postprocess.step4_plot_concentrations import run as plot
    from impacts.pipeline.postprocess.step5_plot_exposure import run as exposure_plot
    from impacts.pipeline.postprocess.step8_human_exposure import run as human
    settings = yaml.safe_load((ROOT / 'examples/pilates/settings.yaml').read_text())
    analysis = settings['impacts']['analysis']['human_exposure']
    conc = str(out / 'exposure/beam_concentration_distribution.parquet')
    plants = str(out / 'power_generation/plant_allocation/plant_emissions_approx.csv')
    if (out / 'power_generation/plant_emissions_used_for_concentration.csv').exists():
        plants = str(out / 'power_generation/plant_emissions_used_for_concentration.csv')
    products = plot(concentration_path=conc,
                    network_path=str(BASE / 'preprocess/beam_osm_mapped.parquet'),
                    power_plants_path=plants, pm25_component_region_path=analysis['ces_tracts_file'],
                    output_dir=out / 'postprocess/concentrations')
    save_json(out / 'concentration_plot_manifest.json', products)
    gc.collect()
    products = exposure_plot(population_path=str(out / 'exposure/beam_population_counts.parquet'),
                             concentration_path=conc,
                             network_path=str(BASE / 'preprocess/beam_osm_mapped.parquet'),
                             inmap_cells_path=str(BASE / 'concentrations/beam_inmap_concentrations.parquet'),
                             output_dir=out / 'postprocess/exposure')
    save_json(out / 'exposure_plot_manifest.json', products)
    gc.collect()
    products = human(population_path=str(out / 'exposure/beam_population_counts.parquet'),
                     concentration_path=conc, ces_tracts_path=analysis['ces_tracts_file'],
                     sensitive_facilities_path=analysis['sensitive_facilities_file'],
                     schools_layer=analysis['schools_layer'], ltc_layer=analysis['ltc_layer'],
                     output_dir=out / 'postprocess/human_exposure', power_plants_path=plants)
    save_json(out / 'human_exposure_manifest.json', products)
    subprocess.run([sys.executable, str(ROOT / 'scripts/export_original_style_human_exposure_tables.py'),
                    '--input', str(out / 'postprocess/human_exposure/human_exposure_ces_tracts.parquet'),
                    '--output-dir', str(out / 'postprocess/human_exposure/tables')], check=True)


def verify(out):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(out / 'exposure/beam_concentration_distribution.parquet')
    fields = ['PrimaryPM25', 'SecondaryPM25', 'TotalPM25', 'inmap_PrimaryPM25',
              'inmap_SecondaryPM25', 'aermod_PrimaryPM25', 'power_PrimaryPM25',
              'power_SecondaryPM25', 'inmap_only_PrimaryPM25', 'inmap_only_TotalPM25',
              'BC', 'inmap_BC', 'aermod_BC', 'NO2', 'inmap_NO2', 'aermod_NO2']
    maxima = {f: 0.0 for f in ['TotalPM25', 'PrimaryPM25', 'SecondaryPM25', 'power_PrimaryPM25', 'power_SecondaryPM25', 'BC', 'NO2']}
    for batch in pf.iter_batches(columns=fields):
        df = batch.to_pandas()
        assert np.isfinite(df.to_numpy()).all()
        assert (df.to_numpy() >= 0).all()
        np.testing.assert_allclose(df.PrimaryPM25, df.inmap_PrimaryPM25 + df.aermod_PrimaryPM25 + df.power_PrimaryPM25)
        np.testing.assert_allclose(df.SecondaryPM25, df.inmap_SecondaryPM25 + df.power_SecondaryPM25)
        np.testing.assert_allclose(df.TotalPM25, df.PrimaryPM25 + df.SecondaryPM25)
        np.testing.assert_allclose(df.inmap_only_TotalPM25, df.inmap_only_PrimaryPM25 + df.inmap_SecondaryPM25)
        np.testing.assert_allclose(df.BC, df.inmap_BC + df.aermod_BC)
        np.testing.assert_allclose(df.NO2, df.inmap_NO2 + df.aermod_NO2)
        assert (df.inmap_only_PrimaryPM25 + 1e-10 >= df.inmap_PrimaryPM25).all()
        for field in maxima:
            maxima[field] = max(maxima[field], float(df[field].max()))
    selected = pd.read_csv(out / 'power_generation/plant_emissions_used_for_concentration.csv').set_index('ORISPL')
    original = pd.read_csv(out / 'power_generation/plant_allocation/plant_emissions_approx.csv').set_index('ORISPL')
    pd.testing.assert_frame_equal(selected, original.loc[selected.index])
    power = pd.read_parquet(out / 'power_generation/power_generation_isrm_concentrations.parquet')
    np.testing.assert_allclose(power.power_TotalPM25, power.power_PrimaryPM25 + power.power_SecondaryPM25)
    save_json(out / 'concentration_validation.json', {
        'rows': pf.metadata.num_rows, 'component_identities_passed': True,
        'finite_nonnegative': True, 'inmap_only_retains_local_primary': True,
        'selected_plants_not_rescaled_or_reallocated': True,
        'max_concentration_ug_m3': maxima,
        'power_total_pm25_max_ug_m3': float(power.power_TotalPM25.max()),
        'inmap_only_fields_scope': 'traffic-only full ISRM, without local-to-local removal',
        'final_fields_scope': 'traffic nonlocal ISRM + traffic AERMOD + selected power plants',
        'power_NO2_and_BC': 'not modeled; power NOx contributes to secondary PM2.5 (pNO3)',
    })


def audit(out):
    import pyarrow.parquet as pq
    folder = out / 'power_generation'
    summary = json.loads((folder / 'power_generation_summary.json').read_text())
    assert summary['hourly_annual_totals_match']
    demand = pd.read_parquet(folder / 'sfbay_hourly_charging_demand.parquet')
    hourly = pd.read_csv(folder / 'plant_allocation/hourly_dispatch_summary.csv')
    assert len(hourly) == len(demand) == 8760
    np.testing.assert_allclose(hourly.enduse_MWh, demand.charging_demand_MWh)
    np.testing.assert_allclose(hourly.allocated_MWh + hourly.unallocated_MWh, hourly.grid_MWh)
    plants = pd.read_csv(folder / 'plant_allocation/plant_emissions_approx.csv')
    for pollutant in POLLUTANTS:
        np.testing.assert_allclose(hourly[f'{pollutant}_kg'].sum(), plants[f'{pollutant}_kg'].sum())
    sources = [DEMAND / 'sfbay_hourly_charging_demand.parquet',
               BREATHE / 'scripts/allocate_power_plants_approx.py',
               BREATHE / 'tmp/cambium_check/hourly_balancingArea/Cambium22_LowRECost_hourly_p9_2040.csv']
    report = {'checks_passed': True, 'hours': len(hourly),
              'plant_hour_rows': pq.ParquetFile(folder / 'plant_allocation/plant_hourly_generation_emissions.parquet').metadata.num_rows,
              'emitting_plants': int(plants[[f'{p}_kg' for p in ['NOx', 'SOx', 'PM2_5', 'NH3', 'VOC']]].sum(axis=1).gt(0).sum()),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    save_json(out / 'dispatch_validation.json', report)
    print(json.dumps(report, indent=2), flush=True)


def validate_products(out):
    """Check generated artifacts and embedded map payload without a browser."""
    from html.parser import HTMLParser
    from PIL import Image
    products = {}
    for name in ['concentration_plot_manifest', 'exposure_plot_manifest', 'human_exposure_manifest']:
        items = json.loads((out / f'{name}.json').read_text())
        if not items:
            raise ValueError(f'Empty product manifest: {name}')
        products.update({f'{name}/{k}': v for k, v in items.items()})
    products['power_only_map'] = str(out / 'power_generation/power_generation_isrm_total_pm25.png')
    for key, value in products.items():
        path = Path(value)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing/empty product {key}: {path}')
        if path.suffix.lower() == '.png':
            with Image.open(path) as img:
                img.verify()
    tables = out / 'postprocess/human_exposure/tables'
    for name in ['CES25_Direct_Final_tracts.csv', 'CES25_Direct_Final_Summary.csv', 'CES25_Direct_Final_Comparison.csv']:
        frame = pd.read_csv(tables / name)
        if frame.empty:
            raise ValueError(f'Empty exposure table: {name}')
        products[name] = str(tables / name)

    class IframeParser(HTMLParser):
        srcdoc = None

        def handle_starttag(self, tag, attrs):
            if tag == 'iframe':
                self.srcdoc = dict(attrs).get('srcdoc')

    parser = IframeParser()
    parser.feed(Path(products['human_exposure_manifest/webmap']).read_text(encoding='utf-8'))
    if not parser.srcdoc:
        raise ValueError('Original web-map iframe is missing.')
    payload = {}
    for key in ['DATA', 'RAW', 'POWER_PLANTS']:
        prefix = f'const {key} = '
        start = parser.srcdoc.index(prefix) + len(prefix)
        payload[key], _ = json.JSONDecoder().raw_decode(parser.srcdoc[start:])
    selected = pd.read_csv(out / 'power_generation/plant_emissions_used_for_concentration.csv')
    assert {p['id'] for p in payload['POWER_PLANTS']} == set(selected.ORISPL.astype(str))
    assert set(payload['RAW']['rasters']) == {'impm', 'aepm', 'imbc', 'fbc', 'imno2', 'fno2', 'dpm', 'dbc', 'dno2'}
    assert payload['RAW']['sourceCellMetres'] == 100
    assert all(payload['RAW']['rasters'].values())
    save_json(out / 'product_validation.json', {
        'artifact_checks_passed': True, 'products': products,
        'web_plant_markers': len(payload['POWER_PLANTS']),
        'web_layers': sorted(payload['RAW']['rasters']),
        'web_source_cell_meters': 100,
        'web_raster_display_size': [payload['RAW']['width'], payload['RAW']['height']],
        'browser_interaction_test': 'Not performed: browser runtime reports no available browser.',
    })


def main():
    global DEMAND
    # Windows redirected streams can default to GBK, which cannot encode PM2.5
    # subscripts used by the existing progress bars. Never fail a scientific run
    # solely because a log label is not representable in the console encoding.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['dispatch', 'power_isrm', 'integrate', 'postprocess', 'verify', 'remaining', 'audit', 'validate_products'])
    parser.add_argument('--isrm-store', default='s3://inmap-model/isrm_v1.2.1.zarr/')
    parser.add_argument('--power-method', choices=['matrix'], default='matrix')
    parser.add_argument('--source-layer-index', type=int, choices=[0, 1], default=1)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--demand-dir', type=Path, default=DEMAND,
                        help='Saved hourly charging outputs; leaves previous demand untouched.')
    parser.add_argument('--allow-incomplete-charging-scenario', action='store_true',
                        help='Explicitly run supplied-load diagnostics with recorded unmet truck energy; NOT a fully feasible electrification scenario.')
    args = parser.parse_args()
    DEMAND = args.demand_dir.resolve()
    charging_summary = DEMAND / 'beam_charging_summary.json'
    if charging_summary.exists():
        charging_qa = json.loads(charging_summary.read_text(encoding='utf-8'))
        if charging_qa.get('ready_for_full_electrification_pipeline') is False and not args.allow_incomplete_charging_scenario:
            parser.error('Truck charging has unmet energy: resolve feasibility or explicitly define a partial-electrification scenario before dispatch.')
    if args.power_method != 'matrix':
        parser.error('Annual-energy concentration scaling is disabled: use new hourly-dispatch plant emissions with an ISRM matrix.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    stages = ['power_isrm', 'integrate', 'postprocess', 'verify', 'validate_products'] if args.stage == 'remaining' else [args.stage]
    status_path = args.output_dir / 'continuation_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    summary_path = args.output_dir / 'power_generation/power_generation_summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text()).get('hourly_annual_totals_match'):
        status['dispatch'] = 'completed'
    for stage in ['power_isrm', 'integrate', 'postprocess', 'verify']:
        status.setdefault(stage, 'pending')
    status.update({'charging_input': str(DEMAND), 'traffic_baseline': str(BASE),
                   'incomplete_charging_scenario_explicitly_allowed': args.allow_incomplete_charging_scenario,
                   'output_dir': str(args.output_dir.resolve()),
                   'isrm_store': args.isrm_store, 'power_method': args.power_method,
                   'source_layer_index': args.source_layer_index})
    for stage in stages:
        status[stage] = 'running'
        save_json(status_path, status)
        try:
            if stage == 'power_isrm':
                power_isrm(args.output_dir.resolve(), args.isrm_store, args.power_method, args.source_layer_index)
            else:
                globals()[stage](args.output_dir.resolve())
        except Exception as exc:
            status[stage] = 'failed'
            status['error'] = f'{type(exc).__name__}: {exc}'
            save_json(status_path, status)
            raise
        status[stage] = 'completed'
        if 'failed' not in status.values():
            status.pop('error', None)
        save_json(status_path, status)


if __name__ == '__main__':
    main()
