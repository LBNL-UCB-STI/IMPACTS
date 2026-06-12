"""Postprocess Step 6b — Plot delta concentration transects.

Plots median concentration delta vs. distance-from-road for PrimaryPM25 and
TotalPM25, with separate series for AERMOD and InMAP cells and a vertical
dashed line marking the approximate AERMOD domain boundary.

Standalone usage::

    python -m impacts.pipeline.postprocess.step6b_plot_delta_transects \
        /path/to/output_dir \
        --delta-table /path/to/concentration_delta.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ...common import log_step_banner
from ._common import (
    CHART_AXIS_LABEL_FONTSIZE,
    CHART_LEGEND_FONTSIZE,
    CHART_TICK_LABEL_FONTSIZE,
    CHART_TITLE_FONTSIZE,
    PLOT_DPI,
)
from .step4b_plot_transects import (
    _AERMOD_BOUNDARY_KM,
    _HAS_AERMOD_FLAG,
    _TRANSECT_MAX_KM,
    _assign_distance_bins,
    _compute_road_distances,
)

logger = logging.getLogger(__name__)

_DELTA_PANELS = [
    ("PrimaryPM25_delta", "Primary PM₂.₅ delta (μg/m³)"),
    ("TotalPM25_delta",   "Total PM₂.₅ delta (μg/m³)"),
]

_STALE_OUTPUTS = [
    "delta_transects.png",
    "delta_transects_table.parquet",
]


def _remove_stale_outputs(output_dir: Path) -> None:
    for filename in _STALE_OUTPUTS:
        path = output_dir / filename
        if path.exists():
            path.unlink()
            logger.info("  Removed stale Step 6b output → %s", path)


def _build_delta_transect_table(
    conc_gdf,
    delta_df: pd.DataFrame,
    distances_m: pd.Series,
) -> pd.DataFrame:
    rows: dict = {
        "aermod_cell_id": conc_gdf["aermod_cell_id"].to_numpy(),
        "distance_km": _assign_distance_bins(distances_m.to_numpy(dtype="float64")),
    }
    if _HAS_AERMOD_FLAG in conc_gdf.columns:
        rows[_HAS_AERMOD_FLAG] = conc_gdf[_HAS_AERMOD_FLAG].to_numpy(dtype=bool)

    base_df = pd.DataFrame(rows).dropna(subset=["distance_km"])

    delta_cols = ["aermod_cell_id"] + [col for col, _ in _DELTA_PANELS if col in delta_df.columns]
    return base_df.merge(delta_df[delta_cols], on="aermod_cell_id", how="left")


def _plot_delta_transects(transect_df: pd.DataFrame, output_dir: Path) -> Optional[str]:
    import matplotlib.pyplot as plt

    has_flag = _HAS_AERMOD_FLAG in transect_df.columns
    available = [(col, lbl) for col, lbl in _DELTA_PANELS if col in transect_df.columns]
    if not available:
        logger.warning("  No delta columns found in transect table; skipping delta transect plot.")
        return None

    ncols = len(available)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), dpi=PLOT_DPI)
    if ncols == 1:
        axes = [axes]

    for ax, (column, ylabel) in zip(axes, available):
        valid = transect_df.dropna(subset=[column])
        all_series = valid.groupby("distance_km")[column].median()

        ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="-", alpha=0.45, zorder=1)
        ax.axvline(_AERMOD_BOUNDARY_KM, color="gray", linewidth=0.9, linestyle="--", alpha=0.55, zorder=1)
        ax.plot(all_series.index, all_series.values, color="black", linewidth=1.8, label="All cells", zorder=3)

        if has_flag:
            aermod_mask = valid[_HAS_AERMOD_FLAG].eq(True)
            inmap_mask = ~valid[_HAS_AERMOD_FLAG]

            aermod_series = valid.loc[aermod_mask].groupby("distance_km")[column].median()
            inmap_series = valid.loc[inmap_mask].groupby("distance_km")[column].median()

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

        ax.set_xlim(0.0, _TRANSECT_MAX_KM)
        ax.set_xlabel("Distance from road (km)", fontsize=CHART_AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(ylabel, fontsize=CHART_AXIS_LABEL_FONTSIZE)
        ax.tick_params(labelsize=CHART_TICK_LABEL_FONTSIZE)
        ax.legend(fontsize=CHART_LEGEND_FONTSIZE, loc="upper right")
        ax.set_title(ylabel.split(" (")[0], fontsize=CHART_TITLE_FONTSIZE)

    fig.suptitle(
        "Concentration Delta vs. Distance from Road (median per 100 m bin)",
        fontsize=CHART_TITLE_FONTSIZE + 2,
    )
    fig.tight_layout()

    out_path = output_dir / "delta_transects.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def run(
    *,
    concentration_path: str,
    delta_table_path: str,
    network_path: str,
    output_dir: Path,
) -> dict[str, str]:
    """Render delta concentration transect plots.

    Parameters
    ----------
    concentration_path:
        Path to ``beam_concentration_distribution.parquet`` (for geometry and AERMOD flag).
    delta_table_path:
        Path to ``concentration_delta.parquet`` from Step 6.
    network_path:
        Path to ``beam_osm_mapped.parquet``.
    output_dir:
        Directory where output files are written.
    """
    import geopandas as gpd
    import pyarrow.parquet as pq

    log_step_banner("Postprocess Step 6b", "Plot Delta Concentration Transects", logger=logger)
    output_dir = Path(output_dir)
    _remove_stale_outputs(output_dir)

    wanted_geom_cols = ["geometry", "aermod_cell_id", _HAS_AERMOD_FLAG]
    file_cols = set(pq.read_schema(concentration_path).names)
    read_cols = [c for c in wanted_geom_cols if c in file_cols]

    logger.info("Loading concentration geometry …")
    conc_gdf = gpd.read_parquet(concentration_path, columns=read_cols)

    logger.info("Loading delta table …")
    delta_cols_wanted = ["aermod_cell_id"] + [col for col, _ in _DELTA_PANELS]
    delta_file_cols = set(pq.read_schema(delta_table_path).names)
    delta_read_cols = [c for c in delta_cols_wanted if c in delta_file_cols]
    delta_df = pd.read_parquet(delta_table_path, columns=delta_read_cols)

    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()

    logger.info("Computing cell distances to road network …")
    distances_m = _compute_road_distances(conc_gdf, net_gdf)

    logger.info("Building delta transect table …")
    transect_df = _build_delta_transect_table(conc_gdf, delta_df, distances_m)

    outputs: dict[str, str] = {}
    table_path = output_dir / "delta_transects_table.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    transect_df.to_parquet(table_path, index=False)
    logger.info("  Saved delta transect table → %s", table_path)
    outputs["delta_transect_table"] = str(table_path)

    result = _plot_delta_transects(transect_df, output_dir)
    if result is not None:
        outputs["delta_transects"] = result

    logger.info("Postprocess Step 6b complete: %d outputs written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(
    output_dir: Path,
    delta_table_path: Optional[str] = None,
) -> dict[str, str]:
    """Run Step 6b from a pipeline output directory using manifest-resolved paths."""
    from ._common import pipeline_outputs

    output_dir = Path(output_dir)
    outs = pipeline_outputs(output_dir)
    conc_path = outs.get("beam_concentration_distribution") or str(
        output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )
    net_path = str(output_dir / "preprocess" / "beam_osm_mapped.parquet")
    if delta_table_path is None:
        delta_table_path = str(
            output_dir / "postprocess" / "delta_concentrations" / "concentration_delta.parquet"
        )
    return run(
        concentration_path=conc_path,
        delta_table_path=delta_table_path,
        network_path=net_path,
        output_dir=output_dir / "postprocess" / "delta_transects",
    )


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step6b_plot_delta_transects",
        description="Plot delta concentration transects from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path, help="Path to the main pipeline output folder.")
    parser.add_argument(
        "--delta-table",
        default=None,
        help=(
            "Path to concentration_delta.parquet from Step 6. "
            "Defaults to {output_dir}/postprocess/delta_concentrations/concentration_delta.parquet."
        ),
    )
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir), delta_table_path=args.delta_table)
