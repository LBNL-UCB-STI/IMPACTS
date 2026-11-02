"""Postprocess Step 4 — Plot concentration maps.

Generates concentration PNGs for TotalPM25, BC, and NO2, a shared-scale
Primary/Secondary PM2.5 comparison map, and four PM2.5 component maps showing
the local AERMOD + nonlocal InMAP + secondary InMAP decomposition.

Standalone usage::

    python -m impacts.pipeline.postprocess.step4_plot_concentrations /path/to/output_dir
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...common import log_step_banner
from ._common import (
    CMAP_BC,
    CMAP_NO2,
    CMAP_PM25,
    MAP_COLORBAR_LABEL_FONTSIZE,
    MAP_COLORBAR_LABELPAD,
    MAP_COLORBAR_TICK_FONTSIZE,
    MAP_DPI,
    MAP_FIGSIZE,
    MAP_TITLE_FONTSIZE,
    _advance_progress,
    _add_basemap,
    _add_colorbar,
    _add_network,
    _add_power_plants,
    _close_progress,
    _grid_raster_layout,
    _map_progress,
    _plot_raster_layer,
    _set_progress_task,
)

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# (column, display_title, colormap, vmax_percentile)
_LAYERS = [
    ("TotalPM25",     "Total PM₂.₅ (μg/m³)",     CMAP_PM25, 0.99),
    ("BC",            "Black Carbon (μg/m³)",      CMAP_BC,   0.99),
    ("NO2",           "NO₂ (μg/m³)",               CMAP_NO2,  0.99),
]
_PRIMARY_SECONDARY_COLUMNS = ["PrimaryPM25", "SecondaryPM25"]
_PM25_COMPONENT_SOURCE_COLUMNS = [
    "aermod_PrimaryPM25_scaled",
    "inmap_PrimaryPM25",
    "inmap_SecondaryPM25",
]
_PM25_COMPONENT_LAYERS = [
    (
        "PM25_Local_AERMOD",
        "Primary PM₂.₅ Local Dispersion (AERMOD)",
        "primary_pm25_local_aermod.png",
        "primary_pm25_local_aermod_map",
    ),
    (
        "PM25_Nonlocal_InMAP",
        "Primary PM₂.₅ Nonlocal Dispersion (InMAP)",
        "primary_pm25_nonlocal_inmap.png",
        "primary_pm25_nonlocal_inmap_map",
    ),
    (
        "PM25_Secondary_InMAP",
        "Secondary PM₂.₅ (InMAP)",
        "secondary_pm25_inmap.png",
        "secondary_pm25_inmap_map",
    ),
    (
        "PM25_ThreeComponent_Total",
        "Total PM₂.₅ (AERMOD Local + InMAP Nonlocal + InMAP Secondary)",
        "total_pm25_three_component.png",
        "total_pm25_three_component_map",
    ),
]
_SF_BAY_COUNTIES = {
    "Alameda", "Contra Costa", "Marin", "Napa", "San Francisco",
    "San Mateo", "Santa Clara", "Solano", "Sonoma",
}
_LOCAL_NONLOCAL_COMPONENT_LAYERS = {
    pollutant: [
        (f"{pollutant}_Local_AERMOD", f"{label} Local Dispersion (AERMOD)", f"{pollutant.lower()}_local_aermod.png"),
        (f"{pollutant}_Nonlocal_InMAP", f"{label} Nonlocal Dispersion (InMAP)", f"{pollutant.lower()}_nonlocal_inmap.png"),
        (f"{pollutant}_TwoComponent_Total", f"Total {label} (AERMOD Local + InMAP Nonlocal)", f"{pollutant.lower()}_two_component_total.png"),
    ]
    for pollutant, label in [("BC", "BC"), ("NO2", "NO₂")]
}
_SUPERSEDED_OUTPUT_GLOBS = (
    "primarypm25.png",
    "secondarypm25.png",
    "concentration_delta.parquet",
    "delta_*.png",
)


def _remove_superseded_outputs(output_dir: Path) -> None:
    for pattern in _SUPERSEDED_OUTPUT_GLOBS:
        for path in output_dir.glob(pattern):
            if not path.is_file():
                continue
            path.unlink()
            logger.info("  Removed superseded concentration plot output → %s", path)


def _shared_positive_vmax(conc_gdf, columns: list[str], vmax_q: float) -> float | None:
    values = [
        conc_gdf[column].dropna()
        for column in columns
        if column in conc_gdf.columns
    ]
    if not values:
        return None
    positive = [series[series > 0] for series in values]
    positive = [series for series in positive if not series.empty]
    if not positive:
        return None
    return float(max(series.quantile(vmax_q) for series in positive))


def _add_pm25_component_columns(conc_gdf):
    required = _PM25_COMPONENT_SOURCE_COLUMNS
    missing = [column for column in required if column not in conc_gdf.columns]
    if missing:
        raise ValueError(f"PM2.5 component plots require concentration columns: {missing}")
    result = conc_gdf.copy()
    result["PM25_Local_AERMOD"] = result["aermod_PrimaryPM25_scaled"]
    result["PM25_Nonlocal_InMAP"] = result["inmap_PrimaryPM25"]
    result["PM25_Secondary_InMAP"] = result["inmap_SecondaryPM25"]
    result["PM25_ThreeComponent_Total"] = (
        result["PM25_Local_AERMOD"]
        + result["PM25_Nonlocal_InMAP"]
        + result["PM25_Secondary_InMAP"]
    )
    return result


def _add_local_nonlocal_component_columns(conc_gdf, pollutant):
    """Build traffic-only BC/NO2 decomposition, never substitute InMAP-only."""
    import numpy as np

    if pollutant not in _LOCAL_NONLOCAL_COMPONENT_LAYERS:
        raise ValueError(f"Unsupported local/nonlocal pollutant: {pollutant}")
    required = [f"aermod_{pollutant}", f"inmap_{pollutant}"]
    missing = [column for column in required if column not in conc_gdf]
    if missing:
        raise ValueError(f"{pollutant} component plots require concentration columns: {missing}")
    values = conc_gdf[required].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < -1e-10):
        raise ValueError(f"Invalid {pollutant} component concentrations")
    result = conc_gdf.copy()
    result[f"{pollutant}_Local_AERMOD"] = result[required[0]]
    result[f"{pollutant}_Nonlocal_InMAP"] = result[required[1]]
    result[f"{pollutant}_TwoComponent_Total"] = result[required[0]] + result[required[1]]
    return result


def _clip_pm25_components_to_sfbay(conc_gdf, net_gdf, region_path: str | None):
    """Clip the four decomposition maps to the nine-county SF Bay Area."""
    if not region_path:
        return conc_gdf, net_gdf
    import geopandas as gpd
    import shapely

    path = Path(region_path)
    if path.suffix.lower() in {".parquet", ".geoparquet"}:
        region = gpd.read_parquet(path)
    else:
        region = gpd.read_file(path)
    county_column = next(
        (column for column in ("County", "NAME", "NAMELSAD") if column in region.columns),
        None,
    )
    if county_column is None:
        raise ValueError(
            f"SF Bay component-map region must contain County, NAME, or NAMELSAD: {path}"
        )
    county_names = region[county_column].astype(str).str.replace(
        r"\s+County$", "", regex=True
    )
    region = region.loc[county_names.isin(_SF_BAY_COUNTIES), [county_column, "geometry"]]
    region = region.to_crs(conc_gdf.crs)
    if region.empty:
        raise ValueError(f"SF Bay component-map region contains none of the nine Bay Area counties: {path}")
    region_union = region.geometry.union_all()
    # The AERMOD receptor layer can contain millions of 100 m polygons.  Test
    # their cell centers with Shapely's vectorized predicate instead of doing
    # millions of polygon/polygon intersections.
    cell_bounds = shapely.bounds(conc_gdf.geometry.array)
    center_x = (cell_bounds[:, 0] + cell_bounds[:, 2]) / 2
    center_y = (cell_bounds[:, 1] + cell_bounds[:, 3]) / 2
    within_region = shapely.contains_xy(region_union, center_x, center_y)
    clipped_concentrations = conc_gdf.loc[within_region].copy()
    clipped_network = net_gdf.loc[net_gdf.geometry.intersects(region_union)].copy()
    if clipped_concentrations.empty:
        raise ValueError(f"No concentration cells intersect the SF Bay component-map region: {path}")
    return clipped_concentrations, clipped_network


def _plot_one(conc_gdf, net_gdf, plants, layout, column: str, title: str, cmap,
              vmax_q: float, out_path: Path, *, vmax_override: float | None = None,
              scenario_note: str | None = None) -> Optional[str]:
    if column not in conc_gdf.columns:
        logger.warning("  Column %s not in concentration data, skipping.", column)
        return None

    vals = conc_gdf[column].dropna()
    nonzero = vals[vals > 0]
    if nonzero.empty:
        logger.warning("  Column %s has no positive values, skipping.", column)
        return None

    vmax = float(vmax_override) if vmax_override is not None else float(nonzero.quantile(vmax_q))
    logger.info("  %s  vmax=%.4f → %s", column, vmax, out_path.name)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)
    ax.set_aspect("equal")
    _plot_raster_layer(
        ax,
        conc_gdf,
        column,
        cmap,
        layout=layout,
        vmax=vmax,
    )
    _add_network(ax, net_gdf)
    _add_basemap(ax, crs=conc_gdf.crs)
    _add_power_plants(ax, plants)
    _add_colorbar(fig, ax, cmap, vmax, title)
    ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    if scenario_note:
        fig.text(0.02, -0.015, scenario_note, fontsize=11, ha="left", va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def _plot_primary_secondary_comparison(conc_gdf, net_gdf, plants, layout, out_path: Path,
                                      scenario_note: str | None = None) -> Optional[str]:
    columns = _PRIMARY_SECONDARY_COLUMNS
    missing = [column for column in columns if column not in conc_gdf.columns]
    if missing:
        logger.warning("  Columns %s not in concentration data, skipping PM₂.₅ comparison.", missing)
        return None

    vmax = _shared_positive_vmax(conc_gdf, columns, 0.99)
    if vmax is None:
        logger.warning("  PrimaryPM25 and SecondaryPM25 have no positive values, skipping PM₂.₅ comparison.")
        return None

    logger.info("  Primary/Secondary PM₂.₅ shared vmax=%.4f → %s", vmax, out_path.name)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(MAP_FIGSIZE[0] * 1.85, MAP_FIGSIZE[1] * 0.95),
        dpi=MAP_DPI,
    )
    for ax, column, title in zip(
        axes,
        columns,
        ["Primary PM₂.₅", "Secondary PM₂.₅"],
        strict=True,
    ):
        ax.set_aspect("equal")
        _plot_raster_layer(
            ax,
            conc_gdf,
            column,
            CMAP_PM25,
            layout=layout,
            vmax=vmax,
        )
        _add_network(ax, net_gdf)
        _add_basemap(ax, crs=conc_gdf.crs)
        _add_power_plants(ax, plants)
        ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
        ax.set_axis_off()

    fig.suptitle("Primary vs Secondary PM₂.₅", fontsize=MAP_TITLE_FONTSIZE + 4, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.9, 0.96], pad=0.5)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=CMAP_PM25, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("PM₂.₅ (μg/m³)", fontsize=MAP_COLORBAR_LABEL_FONTSIZE, labelpad=MAP_COLORBAR_LABELPAD)
    cbar.ax.tick_params(labelsize=MAP_COLORBAR_TICK_FONTSIZE, pad=6)
    cbar.ax.yaxis.get_offset_text().set_fontsize(MAP_COLORBAR_TICK_FONTSIZE)
    if scenario_note:
        fig.text(0.02, -0.015, scenario_note, fontsize=11, ha="left", va="top")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def run(
    *,
    concentration_path: str,
    network_path: str,
    power_plants_path: str | None = None,
    pm25_component_region_path: str | None = None,
    output_dir: Path,
    scenario_note: str | None = None,
) -> dict[str, str]:
    """Render concentration maps for all pollutant variables.

    Parameters
    ----------
    concentration_path:
        Path to ``beam_concentration_distribution.parquet``.
    network_path:
        Path to ``beam_osm_mapped.parquet``.
    output_dir:
        Directory where PNG files are written.
    """
    import geopandas as gpd
    from ._common import _load_emitting_power_plants

    log_step_banner("Postprocess Step 4", "Plot Concentrations", logger=logger)
    output_dir = Path(output_dir)
    _remove_superseded_outputs(output_dir)

    concentration_columns = list(
        dict.fromkeys(
            [
                "geometry",
                *[column for column, _, _, _ in _LAYERS],
                *_PRIMARY_SECONDARY_COLUMNS,
                *_PM25_COMPONENT_SOURCE_COLUMNS,
            ]
        )
    )
    logger.info("Loading concentration data …")
    conc_gdf = gpd.read_parquet(concentration_path, columns=concentration_columns)
    conc_gdf = _add_pm25_component_columns(conc_gdf)
    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()
    plants = _load_emitting_power_plants(power_plants_path, conc_gdf.crs)

    logger.info("Building native grid raster layout …")
    layout = _grid_raster_layout(conc_gdf)
    component_gdf, component_net_gdf = _clip_pm25_components_to_sfbay(
        conc_gdf, net_gdf, pm25_component_region_path
    )
    component_layout = _grid_raster_layout(component_gdf)

    outputs: dict[str, str] = {}
    progress = _map_progress(len(_LAYERS) + 1 + len(_PM25_COMPONENT_LAYERS), "Postprocess Step 4")
    try:
        for column, title, cmap, vmax_q in _LAYERS:
            _set_progress_task(progress, column, step_label="Postprocess Step 4")
            result = _plot_one(conc_gdf, net_gdf, plants, layout, column, title, cmap, vmax_q,
                               output_dir / f"{column.lower()}.png", scenario_note=scenario_note)
            if result is not None:
                outputs[f"{column.lower()}_map"] = result
            _advance_progress(progress)
        _set_progress_task(progress, "Primary/Secondary PM2.5", step_label="Postprocess Step 4")
        result = _plot_primary_secondary_comparison(
            conc_gdf,
            net_gdf,
            plants,
            layout,
            output_dir / "primary_secondary_pm25.png",
            scenario_note=scenario_note,
        )
        if result is not None:
            outputs["primary_secondary_pm25_map"] = result
        _advance_progress(progress)
        component_columns = [column for column, _, _, _ in _PM25_COMPONENT_LAYERS]
        component_vmax = _shared_positive_vmax(component_gdf, component_columns, 0.99)
        for column, title, filename, output_key in _PM25_COMPONENT_LAYERS:
            _set_progress_task(progress, title, step_label="Postprocess Step 4")
            result = _plot_one(
                component_gdf,
                component_net_gdf,
                None,
                component_layout,
                column,
                title,
                CMAP_PM25,
                0.99,
                output_dir / filename,
                vmax_override=component_vmax,
                scenario_note=scenario_note,
            )
            if result is not None:
                outputs[output_key] = result
            _advance_progress(progress)
    finally:
        _close_progress(progress)

    logger.info("Postprocess Step 4 complete: %d maps written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 4 from a pipeline output directory using manifest-resolved paths."""
    from ._common import pipeline_outputs

    output_dir = Path(output_dir)
    outs = pipeline_outputs(output_dir)
    conc_path = outs.get("beam_concentration_distribution") or str(
        output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )
    net_path = str(output_dir / "preprocess" / "beam_osm_mapped.parquet")
    return run(
        concentration_path=conc_path,
        network_path=net_path,
        output_dir=output_dir / "postprocess" / "concentrations",
    )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step4_plot_concentrations",
        description="Plot concentration maps from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path,
                        help="Path to the main pipeline output folder.")
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir))
