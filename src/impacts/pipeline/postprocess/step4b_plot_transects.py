"""Postprocess Step 4b — Plot concentration transects.

Plots median concentration vs. distance-from-road for each pollutant, with
separate series for AERMOD and InMAP cells and a vertical dashed line marking
the approximate AERMOD domain boundary.

Standalone usage::

    python -m impacts.pipeline.postprocess.step4b_plot_transects /path/to/output_dir
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ...common import log_step_banner
from ._common import (
    CHART_AXIS_LABEL_FONTSIZE,
    CHART_LEGEND_FONTSIZE,
    CHART_SUPTITLE_FONTSIZE,
    CHART_TICK_LABEL_FONTSIZE,
    CHART_TITLE_FONTSIZE,
    PLOT_DPI,
    _advance_progress,
    _close_progress,
    _set_progress_task,
    _step_progress,
)

logger = logging.getLogger(__name__)

_TRANSECT_BIN_M = 100.0
_TRANSECT_MAX_KM = 10.0
_AERMOD_BOUNDARY_KM = 3.0

_HAS_AERMOD_FLAG = "has_aermod_primarypm25"
_CONCENTRATION_COLUMNS = ["TotalPM25", "PrimaryPM25", "SecondaryPM25", "BC", "NO2"]

# (column, y-axis label, split_by_source)
_PANEL_SPECS = [
    ("PrimaryPM25", "Primary PM₂.₅ (μg/m³)", True),
    ("TotalPM25",   "Total PM₂.₅ (μg/m³)",   True),
    ("BC",          "Black Carbon (μg/m³)",    False),
    ("NO2",         "NO₂ (μg/m³)",             False),
]

_STALE_OUTPUTS = [
    "concentration_transects.png",
    "concentration_transects_table.parquet",
]


def _remove_stale_outputs(output_dir: Path) -> None:
    for filename in _STALE_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()
            logger.info("  Removed stale Step 4b output → %s", path)


def _compute_road_distances(conc_gdf, net_gdf) -> pd.Series:
    import shapely

    net_geoms = net_gdf.geometry.values
    centroids = conc_gdf.geometry.centroid.values
    logger.info("  Building road network STRtree (%d segments) …", len(net_geoms))
    tree = shapely.STRtree(net_geoms)
    logger.info("  Querying nearest road segment per cell (%d cells) …", len(centroids))
    nearest_idx = tree.nearest(centroids)
    distances = shapely.distance(centroids, net_geoms[nearest_idx])
    return pd.Series(distances, index=conc_gdf.index)


def _assign_distance_bins(distances_m: np.ndarray) -> np.ndarray:
    max_m = _TRANSECT_MAX_KM * 1000.0
    edges = np.arange(0.0, max_m + _TRANSECT_BIN_M, _TRANSECT_BIN_M)
    bin_centers_km = (edges[:-1] + _TRANSECT_BIN_M / 2.0) / 1000.0
    idx = np.searchsorted(edges, distances_m, side="right") - 1
    in_range = (distances_m >= 0.0) & (distances_m < max_m)
    result = np.where(in_range, bin_centers_km[np.clip(idx, 0, len(bin_centers_km) - 1)], np.nan)
    return result


def _build_transect_table(conc_gdf, distances_m: pd.Series) -> pd.DataFrame:
    rows: dict = {
        "distance_km": _assign_distance_bins(distances_m.to_numpy(dtype="float64")),
    }
    for col in _CONCENTRATION_COLUMNS:
        if col in conc_gdf.columns:
            rows[col] = conc_gdf[col].to_numpy(dtype="float64")
    if _HAS_AERMOD_FLAG in conc_gdf.columns:
        rows[_HAS_AERMOD_FLAG] = conc_gdf[_HAS_AERMOD_FLAG].to_numpy(dtype=bool)
    return pd.DataFrame(rows).dropna(subset=["distance_km"])


def _median_by_distance(df: pd.DataFrame, column: str, mask: Optional[pd.Series] = None) -> pd.Series:
    sub = df if mask is None else df.loc[mask]
    return sub.groupby("distance_km")[column].median()


def _plot_transects(transect_df: pd.DataFrame, output_dir: Path) -> Optional[str]:
    import matplotlib.pyplot as plt

    has_flag = _HAS_AERMOD_FLAG in transect_df.columns
    available = [(col, lbl, split) for col, lbl, split in _PANEL_SPECS if col in transect_df.columns]
    if not available:
        logger.warning("  No concentration columns found in transect table; skipping plot.")
        return None

    ncols = 2
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows), dpi=PLOT_DPI)
    axes_flat = np.array(axes).flatten()

    for i, (column, ylabel, split) in enumerate(available):
        ax = axes_flat[i]

        all_series = _median_by_distance(transect_df, column)
        ax.plot(all_series.index, all_series.values, color="black", linewidth=1.8, label="All cells", zorder=3)

        if split and has_flag:
            aermod_mask = transect_df[_HAS_AERMOD_FLAG].eq(True)
            inmap_mask = ~transect_df[_HAS_AERMOD_FLAG]

            aermod_series = _median_by_distance(transect_df, column, aermod_mask)
            inmap_series = _median_by_distance(transect_df, column, inmap_mask)

            if not aermod_series.empty:
                ax.plot(
                    aermod_series.index, aermod_series.values,
                    color="#d7301f", linewidth=1.2, linestyle="--", alpha=0.85,
                    label="AERMOD cells", zorder=2,
                )
            if not inmap_series.empty:
                ax.plot(
                    inmap_series.index, inmap_series.values,
                    color="#2166ac", linewidth=1.2, linestyle=":", alpha=0.85,
                    label="InMAP cells", zorder=2,
                )

        ax.axvline(_AERMOD_BOUNDARY_KM, color="gray", linewidth=0.9, linestyle="--", alpha=0.55, zorder=1)
        ax.set_xlim(0.0, _TRANSECT_MAX_KM)
        ax.set_xlabel("Distance from road (km)", fontsize=CHART_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=CHART_AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=CHART_TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=CHART_LEGEND_FONTSIZE, loc="best")
        ax.set_title(ylabel.split(" (")[0], fontsize=CHART_TITLE_FONTSIZE)

    for j in range(len(available), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Concentration vs. Distance from Road (median per 100 m bin)", fontsize=CHART_SUPTITLE_FONTSIZE)
    fig.tight_layout()

    out_path = output_dir / "concentration_transects.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def run(
    *,
    concentration_path: str,
    network_path: str,
    output_dir: Path,
) -> dict[str, str]:
    """Render concentration transect plots.

    Parameters
    ----------
    concentration_path:
        Path to ``beam_concentration_distribution.parquet``.
    network_path:
        Path to ``beam_osm_mapped.parquet``.
    output_dir:
        Directory where output files are written.
    """
    import geopandas as gpd
    import pyarrow.parquet as pq

    log_step_banner("Postprocess Step 4b", "Plot Concentration Transects", logger=logger)
    output_dir = Path(output_dir)
    _remove_stale_outputs(output_dir)

    wanted_cols = ["geometry", _HAS_AERMOD_FLAG, *_CONCENTRATION_COLUMNS]
    file_cols = set(pq.read_schema(concentration_path).names)
    read_cols = [c for c in wanted_cols if c in file_cols]

    outputs: dict[str, str] = {}
    progress = _step_progress(4, "Postprocess Step 4b", unit="step")
    try:
        _set_progress_task(progress, "Load data", step_label="Postprocess Step 4b")
        logger.info("Loading concentration data …")
        conc_gdf = gpd.read_parquet(concentration_path, columns=read_cols)
        logger.info("Loading network …")
        net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()
        _advance_progress(progress)

        _set_progress_task(progress, "Compute distances", step_label="Postprocess Step 4b")
        logger.info("Computing cell distances to road network …")
        distances_m = _compute_road_distances(conc_gdf, net_gdf)
        _advance_progress(progress)

        _set_progress_task(progress, "Build table", step_label="Postprocess Step 4b")
        logger.info("Building transect table …")
        transect_df = _build_transect_table(conc_gdf, distances_m)
        table_path = output_dir / "concentration_transects_table.parquet"
        table_path.parent.mkdir(parents=True, exist_ok=True)
        transect_df.to_parquet(table_path, index=False)
        logger.info("  Saved transect table → %s", table_path)
        outputs["transect_table"] = str(table_path)
        _advance_progress(progress)

        _set_progress_task(progress, "Plot", step_label="Postprocess Step 4b")
        result = _plot_transects(transect_df, output_dir)
        if result is not None:
            outputs["concentration_transects"] = result
        _advance_progress(progress)
    finally:
        _close_progress(progress)

    logger.info("Postprocess Step 4b complete: %d outputs written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 4b from a pipeline output directory using manifest-resolved paths."""
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
        output_dir=output_dir / "postprocess" / "concentration_transects",
    )


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step4b_plot_transects",
        description="Plot concentration transects from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path, help="Path to the main pipeline output folder.")
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir))
