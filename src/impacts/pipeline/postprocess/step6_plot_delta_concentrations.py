"""Postprocess Step 6 — Plot delta concentration maps.

Compares the current run's ``beam_concentration_distribution.parquet`` against
an external baseline run and writes signed deltas as ``current - baseline``.
Zero is white, decreases are blue/green, and increases are yellow/orange/red.

Standalone usage::

    python -m impacts.pipeline.postprocess.step6_plot_delta_concentrations \
        /path/to/output_dir --delta-baseline-concentration-distribution /path/to/baseline.parquet
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...common import (
    log_step_banner,
    _duckdb_scan_expression,
    configure_duckdb_connection,
)
from ._common import (
    MAP_COLORBAR_LABEL_FONTSIZE,
    MAP_COLORBAR_LABELPAD,
    MAP_COLORBAR_TICK_FONTSIZE,
    MAP_DPI,
    MAP_FIGSIZE,
    MAP_SUPTITLE_FONTSIZE,
    MAP_TITLE_FONTSIZE,
    _advance_progress,
    _add_basemap,
    _add_colorbar,
    _add_network,
    _close_progress,
    _grid_raster_layout,
    _map_progress,
    _plot_raster_layer,
    _set_progress_task,
)

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

DELTA_COLUMNS = ["TotalPM25", "PrimaryPM25", "SecondaryPM25", "BC", "NO2"]
_DELTA_SCALAR_LAYERS = [
    ("TotalPM25", "Total PM₂.₅ delta (μg/m³)"),
    ("BC", "Black Carbon delta (μg/m³)"),
    ("NO2", "NO₂ delta (μg/m³)"),
]
_PRIMARY_SECONDARY_COLUMNS = ["PrimaryPM25", "SecondaryPM25"]
_DELTA_OUTPUT_FILES = [
    "concentration_delta.parquet",
    "delta_totalpm25.png",
    "delta_bc.png",
    "delta_no2.png",
    "delta_primary_secondary_pm25.png",
]

DELTA_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "impacts_delta_decrease_increase",
    [
        (0.00, "#2166ac"),  # large decrease
        (0.24, "#67a9cf"),
        (0.42, "#d1f0e8"),
        (0.50, "#ffffff"),  # no change
        (0.58, "#fff7bc"),
        (0.76, "#fdae61"),
        (1.00, "#a50026"),  # large increase
    ],
)


def _remove_stale_outputs(output_dir: Path) -> None:
    for filename in _DELTA_OUTPUT_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
            logger.info("  Removed stale Step 6 output → %s", path)


def delta_norm(values: pd.Series) -> mcolors.TwoSlopeNorm | None:
    finite = pd.to_numeric(values, errors="coerce").replace([float("inf"), float("-inf")], pd.NA).dropna()
    nonzero = finite.loc[finite.ne(0.0)]
    if nonzero.empty:
        return None
    vmax = float(nonzero.abs().quantile(0.99))
    if vmax <= 0.0:
        vmax = float(nonzero.abs().max())
    return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)


def _shared_delta_norm(delta_gdf, columns: list[str]) -> mcolors.TwoSlopeNorm | None:
    values = [delta_gdf[f"{column}_delta"] for column in columns if f"{column}_delta" in delta_gdf.columns]
    if not values:
        return None
    return delta_norm(pd.concat(values, ignore_index=True))


def _build_concentration_delta(conc_gdf, delta_baseline_concentration_path: str | Path, *, output_dir: Path):
    import duckdb
    import geopandas as gpd

    key = "aermod_cell_id"
    current_required = [key, "inmap_cell_id", *DELTA_COLUMNS]

    missing_current = [col for col in current_required if col not in conc_gdf.columns]
    if missing_current:
        raise ValueError(
            "Current beam_concentration_distribution is missing columns required for delta baseline comparison: "
            f"{missing_current}"
        )
    if conc_gdf[key].duplicated().any():
        raise ValueError(
            "Current beam_concentration_distribution has duplicate aermod_cell_id values; "
            f"found {int(conc_gdf[key].duplicated().sum())} duplicate rows."
        )

    logger.info("  Current : %d rows", len(conc_gdf))
    logger.info("  Baseline: %s", Path(delta_baseline_concentration_path).resolve())
    current_df = pd.DataFrame(conc_gdf[current_required])

    logger.info("  Current column stats (before join):")
    for col in DELTA_COLUMNS:
        col_data = current_df[col].dropna()
        logger.info(
            "    %-15s mean=%.6f  min=%.6f  max=%.6f  n=%d",
            col, float(col_data.mean()), float(col_data.min()), float(col_data.max()), len(col_data),
        )

    baseline_scan = _duckdb_scan_expression(delta_baseline_concentration_path)

    con = duckdb.connect()
    try:
        configure_duckdb_connection(con, working_dir=output_dir, show_progress=False, profile="balanced")

        dupes = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT {key}) FROM {baseline_scan}").fetchone()[0]
        if dupes:
            raise ValueError(
                "Delta baseline beam_concentration_distribution has duplicate aermod_cell_id values; "
                f"found {int(dupes)} duplicate rows."
            )

        con.register("current_tbl", current_df)
        delta_exprs = ",\n            ".join(
            f"c.{col} - b.{col} AS {col}_delta, c.{col} AS {col}_current, b.{col} AS {col}_baseline"
            for col in DELTA_COLUMNS
        )

        logger.info("  Joining and computing deltas via DuckDB …")
        result_df = con.execute(f"""
            SELECT c.{key}, c.inmap_cell_id,
                {delta_exprs}
            FROM current_tbl c
            LEFT JOIN {baseline_scan} b ON c.{key} = b.{key}
        """).df()
    finally:
        con.close()

    missing_baseline = int(result_df[f"{DELTA_COLUMNS[0]}_baseline"].isna().sum())
    if missing_baseline:
        logger.warning(
            "  Baseline is missing %d current aermod cells; deltas will be null there.",
            missing_baseline,
        )

    logger.info("  Per-column delta stats (current - baseline):")
    for col in DELTA_COLUMNS:
        c_data = result_df[f"{col}_current"].dropna()
        b_data = result_df[f"{col}_baseline"].dropna()
        d_data = result_df[f"{col}_delta"].dropna()
        logger.info(
            "    %-15s current_mean=%.6f  baseline_mean=%.6f  delta_mean=%.6f  delta_min=%.6f  delta_max=%.6f  n_matched=%d",
            col,
            float(c_data.mean()) if len(c_data) else float("nan"),
            float(b_data.mean()) if len(b_data) else float("nan"),
            float(d_data.mean()) if len(d_data) else float("nan"),
            float(d_data.min()) if len(d_data) else float("nan"),
            float(d_data.max()) if len(d_data) else float("nan"),
            len(b_data),
        )

    logger.info("  Delta ready (%d rows) — reattaching geometry …", len(result_df))
    return gpd.GeoDataFrame(
        conc_gdf[["aermod_cell_id", "geometry"]].merge(result_df, on=key, how="left"),
        geometry="geometry",
        crs=conc_gdf.crs,
    )


def _plot_delta_map(delta_gdf, net_gdf, layout, column: str, title: str, out_path: Path) -> Optional[str]:
    delta_col = f"{column}_delta"
    norm = delta_norm(delta_gdf[delta_col])
    if norm is None:
        logger.warning("  %s has no non-zero delta values, skipping delta map.", column)
        return None

    logger.info("  %s delta  |vmax|=%.4f → %s", column, float(norm.vmax), out_path.name)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)
    ax.set_aspect("equal")
    _plot_raster_layer(
        ax,
        delta_gdf,
        delta_col,
        DELTA_CMAP,
        layout=layout,
        norm=norm,
    )
    _add_network(ax, net_gdf)
    _add_basemap(ax, crs=delta_gdf.crs)
    _add_colorbar(fig, ax, DELTA_CMAP, float(norm.vmax), title, norm=norm)
    ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def _plot_primary_secondary_delta_comparison(delta_gdf, net_gdf, layout, out_path: Path) -> Optional[str]:
    norm = _shared_delta_norm(delta_gdf, _PRIMARY_SECONDARY_COLUMNS)
    if norm is None:
        logger.warning("  PrimaryPM25 and SecondaryPM25 have no non-zero deltas, skipping comparison map.")
        return None

    logger.info("  Primary/Secondary PM₂.₅ delta shared |vmax|=%.4f → %s", float(norm.vmax), out_path.name)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(MAP_FIGSIZE[0] * 1.85, MAP_FIGSIZE[1] * 0.95),
        dpi=MAP_DPI,
    )
    for ax, column, title in zip(
        axes,
        _PRIMARY_SECONDARY_COLUMNS,
        ["Primary PM₂.₅ delta", "Secondary PM₂.₅ delta"],
        strict=True,
    ):
        ax.set_aspect("equal")
        _plot_raster_layer(
            ax,
            delta_gdf,
            f"{column}_delta",
            DELTA_CMAP,
            layout=layout,
            norm=norm,
        )
        _add_network(ax, net_gdf)
        _add_basemap(ax, crs=delta_gdf.crs)
        ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
        ax.set_axis_off()

    fig.suptitle("Primary vs Secondary PM₂.₅ Delta", fontsize=MAP_SUPTITLE_FONTSIZE, y=0.98)
    # Reserve 10% on the right for the colorbar and 4% on top for suptitle, then
    # place the colorbar in a manually-sized axis to prevent overlap with the right panel.
    fig.tight_layout(rect=[0, 0, 0.9, 0.96], pad=0.5)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=DELTA_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("PM₂.₅ delta (current - baseline, μg/m³)", fontsize=MAP_COLORBAR_LABEL_FONTSIZE, labelpad=MAP_COLORBAR_LABELPAD)
    cbar.ax.tick_params(labelsize=MAP_COLORBAR_TICK_FONTSIZE, pad=6)
    cbar.ax.yaxis.get_offset_text().set_fontsize(MAP_COLORBAR_TICK_FONTSIZE)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def run(
    *,
    concentration_path: str,
    delta_baseline_concentration_path: str,
    network_path: str,
    output_dir: Path,
) -> dict[str, str]:
    """Render concentration delta maps using ``current - baseline``."""
    import geopandas as gpd

    log_step_banner("Postprocess Step 6", "Plot Delta Concentrations", logger=logger)
    output_dir = Path(output_dir)
    _remove_stale_outputs(output_dir)

    logger.info("Step 6 paths:")
    logger.info("  current  (subtracted from): %s", Path(concentration_path).resolve())
    logger.info("  baseline (subtracted by)  : %s", Path(delta_baseline_concentration_path).resolve())
    logger.info("  output_dir                : %s", output_dir.resolve())

    concentration_columns = ["geometry", "aermod_cell_id", "inmap_cell_id", *DELTA_COLUMNS]
    logger.info("Loading current concentration data …")
    conc_gdf = gpd.read_parquet(concentration_path, columns=concentration_columns)
    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()
    logger.info("Building concentration delta table …")
    delta_gdf = _build_concentration_delta(conc_gdf, delta_baseline_concentration_path, output_dir=output_dir)
    logger.info("Building native grid raster layout …")
    layout = _grid_raster_layout(delta_gdf)

    table_path = output_dir / "concentration_delta.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    delta_gdf.drop(columns="geometry").to_parquet(table_path, index=False)
    logger.info("  Saved → %s", table_path)

    outputs: dict[str, str] = {"delta_table": str(table_path)}
    progress = _map_progress(len(_DELTA_SCALAR_LAYERS) + 1, "Postprocess Step 6")
    try:
        for column, title in _DELTA_SCALAR_LAYERS:
            _set_progress_task(progress, f"Delta {column}", step_label="Postprocess Step 6")
            result = _plot_delta_map(delta_gdf, net_gdf, layout, column, title, output_dir / f"delta_{column.lower()}.png")
            if result is not None:
                outputs[f"delta_{column.lower()}_map"] = result
            _advance_progress(progress)

        _set_progress_task(progress, "Primary/Secondary PM2.5 delta", step_label="Postprocess Step 6")
        result = _plot_primary_secondary_delta_comparison(
            delta_gdf,
            net_gdf,
            layout,
            output_dir / "delta_primary_secondary_pm25.png",
        )
        if result is not None:
            outputs["delta_primary_secondary_pm25_map"] = result
        _advance_progress(progress)
    finally:
        _close_progress(progress)

    logger.info("Postprocess Step 6 complete: %d outputs written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(output_dir: Path, delta_baseline_concentration_path: str) -> dict[str, str]:
    """Run Step 6 from a pipeline output directory using manifest-resolved paths."""
    from ._common import pipeline_outputs

    output_dir = Path(output_dir)
    outs = pipeline_outputs(output_dir)
    conc_path = outs.get("beam_concentration_distribution") or str(
        output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )
    net_path = str(output_dir / "preprocess" / "beam_osm_mapped.parquet")
    return run(
        concentration_path=conc_path,
        delta_baseline_concentration_path=delta_baseline_concentration_path,
        network_path=net_path,
        output_dir=output_dir / "postprocess" / "delta_concentrations",
    )


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step6_plot_delta_concentrations",
        description="Plot delta concentration maps from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path, help="Path to the main pipeline output folder.")
    parser.add_argument(
        "--delta-baseline-concentration-distribution",
        required=True,
        help=(
            "Baseline beam_concentration_distribution.parquet. "
            "Delta outputs are computed as current output minus this baseline file."
        ),
    )
    args = parser.parse_args()

    run_from_output_dir(
        Path(args.output_dir),
        delta_baseline_concentration_path=args.delta_baseline_concentration_distribution,
    )
