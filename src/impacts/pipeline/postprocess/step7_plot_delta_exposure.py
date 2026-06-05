"""Postprocess Step 7 — Plot delta exposure maps.

Uses Step 6 concentration deltas plus current-run population to show changes in
population-weighted PM2.5 exposure by InMAP cell.

Standalone usage::

    python -m impacts.pipeline.postprocess.step7_plot_delta_exposure /path/to/output_dir
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...common import log_step_banner
from ._common import (
    MAP_DPI,
    MAP_FIGSIZE,
    MAP_TITLE_FONTSIZE,
    _advance_progress,
    _add_basemap,
    _add_colorbar,
    _add_network,
    _close_progress,
    _map_progress,
    _padded_extent_from_bounds,
    _set_progress_task,
)
from .step6_plot_delta_concentrations import DELTA_CMAP
from .step6_plot_delta_concentrations import delta_norm

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

_DELTA_EXPOSURE_OUTPUT_FILES = [
    "pwc_inmap_delta.png",
    "exposure_burden_delta.png",
    "bivariate_delta.png",
    "exposure_delta_inmap.parquet",
]

# key: (delta_class, pop_class), delta_class 0=decrease 1=near-zero 2=increase
_DELTA_BV_COLORS: dict[tuple[int, int], str] = {
    (0, 0): "#d7eef8",
    (0, 1): "#82c7d6",
    (0, 2): "#2c7fb8",
    (1, 0): "#ffffff",
    (1, 1): "#e6e6e6",
    (1, 2): "#bdbdbd",
    (2, 0): "#fff3b0",
    (2, 1): "#fdae61",
    (2, 2): "#d73027",
}


def _remove_stale_outputs(output_dir: Path) -> None:
    for filename in _DELTA_EXPOSURE_OUTPUT_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
            logger.info("  Removed stale Step 7 output → %s", path)


def _merge_population_concentration_delta(pop_gdf, delta_df):
    merged = pop_gdf[["aermod_cell_id", "person_count"]].merge(
        delta_df[["aermod_cell_id", "inmap_cell_id", "TotalPM25_delta"]],
        on="aermod_cell_id",
        how="inner",
    )
    merged["person_count"] = pd.to_numeric(merged["person_count"], errors="coerce").fillna(0.0)
    merged["TotalPM25_delta"] = pd.to_numeric(merged["TotalPM25_delta"], errors="coerce")
    merged = merged.loc[merged["person_count"].gt(0) & merged["TotalPM25_delta"].notna()].copy()
    merged["pm25_exposure_burden_delta"] = merged["TotalPM25_delta"] * merged["person_count"]
    return merged


def _aggregate_delta_exposure_to_inmap(pop_gdf, delta_df, inmap_gdf):
    merged = _merge_population_concentration_delta(pop_gdf, delta_df)
    merged["inmap_cell_id"] = pd.to_numeric(merged["inmap_cell_id"], errors="coerce")
    merged = merged.loc[merged["inmap_cell_id"].notna()].copy()
    if merged.empty:
        empty = inmap_gdf.iloc[0:0][["inmap_cell_id", "geometry"]].copy()
        empty["population"] = pd.Series(dtype="float64")
        empty["exposure_burden_delta"] = pd.Series(dtype="float64")
        empty["pwc_pm25_delta"] = pd.Series(dtype="float64")
        return empty

    grouped = (
        merged.groupby("inmap_cell_id", dropna=False)
        .agg(
            population=("person_count", "sum"),
            exposure_burden_delta=("pm25_exposure_burden_delta", "sum"),
        )
        .reset_index()
    )
    grouped["pwc_pm25_delta"] = grouped["exposure_burden_delta"] / grouped["population"]

    inmap = inmap_gdf.copy()
    inmap["inmap_cell_id"] = pd.to_numeric(inmap["inmap_cell_id"], errors="coerce")
    return inmap[["inmap_cell_id", "geometry"]].merge(grouped, on="inmap_cell_id", how="inner")


def _plot_inmap_delta_scalar(
    inmap_delta_gdf,
    net_gdf,
    *,
    column: str,
    title: str,
    colorbar_label: str,
    out_path: Path,
) -> Optional[str]:
    norm = delta_norm(inmap_delta_gdf[column])
    if norm is None:
        logger.warning("  No non-zero values for %s, skipping.", column)
        return None

    plot_gdf = inmap_delta_gdf.loc[inmap_delta_gdf[column].notna(), [column, "geometry"]].copy()
    logger.info("  %s  cells=%d |vmax|=%.4f → %s", column, len(plot_gdf), float(norm.vmax), out_path.name)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)
    ax.set_aspect("equal")
    plot_gdf.plot(
        ax=ax,
        column=column,
        cmap=DELTA_CMAP,
        norm=norm,
        edgecolor="#555555",
        linewidth=0.18,
        rasterized=True,
        zorder=2,
    )
    xmin, xmax, ymin, ymax = _padded_extent_from_bounds(plot_gdf.total_bounds)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _add_network(ax, net_gdf)
    _add_basemap(ax, crs=plot_gdf.crs)
    _add_colorbar(fig, ax, DELTA_CMAP, float(norm.vmax), colorbar_label, norm=norm)

    ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def _tertile_classes(values: pd.Series) -> pd.Series:
    if len(values) < 3 or values.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return pd.qcut(values.rank(method="first"), q=3, labels=[0, 1, 2]).astype(int)


def _delta_direction_classes(values: pd.Series) -> pd.Series:
    finite_abs = values.abs().replace([float("inf"), float("-inf")], pd.NA).dropna()
    threshold = float(finite_abs.quantile(0.99)) * 0.02 if not finite_abs.empty else 0.0
    classes = pd.Series(np.ones(len(values), dtype=int), index=values.index)
    classes.loc[values < -threshold] = 0
    classes.loc[values > threshold] = 2
    return classes


def _add_delta_bivariate_legend(ax) -> None:
    ax_leg = ax.inset_axes([0.02, 0.02, 0.22, 0.20])
    for r in range(3):
        for c in range(3):
            ax_leg.add_patch(
                mpatches.Rectangle((c, r), 1, 1, color=_DELTA_BV_COLORS[(r, c)], linewidth=0)
            )
    ax_leg.set_xlim(0, 3)
    ax_leg.set_ylim(0, 3)
    ax_leg.set_xticks([1.5])
    ax_leg.set_xticklabels(["Population →"], fontsize=18)
    ax_leg.xaxis.set_ticks_position("top")
    ax_leg.xaxis.set_label_position("top")
    ax_leg.set_yticks([0.5, 1.5, 2.5])
    ax_leg.set_yticklabels(["Decrease", "No change", "Increase"], fontsize=16, rotation=90, va="center")
    ax_leg.tick_params(length=0, pad=6)
    ax_leg.set_facecolor("none")
    for spine in ax_leg.spines.values():
        spine.set_visible(False)


def _plot_delta_bivariate(inmap_delta_gdf, net_gdf, out_path: Path) -> Optional[str]:
    merged = inmap_delta_gdf[
        ["population", "pwc_pm25_delta", "geometry"]
    ].loc[
        inmap_delta_gdf["population"].gt(0) & inmap_delta_gdf["pwc_pm25_delta"].notna()
    ].copy()
    if merged.empty:
        logger.warning("  No populated cells with PM2.5 delta, skipping bivariate delta map.")
        return None

    merged["delta_class"] = _delta_direction_classes(merged["pwc_pm25_delta"])
    merged["pop_class"] = _tertile_classes(merged["population"])
    merged["bv_class"] = merged["delta_class"] * 3 + merged["pop_class"]

    colors_9 = [_DELTA_BV_COLORS[(r, c)] for r in range(3) for c in range(3)]
    cmap_bv = ListedColormap(colors_9)
    norm_bv = BoundaryNorm(np.arange(10) - 0.5, 9)

    logger.info("  Bivariate delta InMAP map  %d populated cells → %s", len(merged), out_path.name)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)
    ax.set_aspect("equal")
    merged.plot(
        ax=ax,
        column="bv_class",
        cmap=cmap_bv,
        norm=norm_bv,
        edgecolor="#555555",
        linewidth=0.18,
        rasterized=True,
        zorder=2,
    )
    xmin, xmax, ymin, ymax = _padded_extent_from_bounds(merged.total_bounds)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    _add_network(ax, net_gdf)
    _add_basemap(ax, crs=inmap_delta_gdf.crs)
    _add_delta_bivariate_legend(ax)

    ax.set_title(
        "Population-weighted PM₂.₅ delta × population by InMAP cell",
        fontsize=MAP_TITLE_FONTSIZE,
        pad=16,
    )
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def run(
    *,
    population_path: str,
    concentration_delta_path: str,
    network_path: str,
    inmap_cells_path: str | None = None,
    output_dir: Path,
) -> dict[str, str]:
    """Render exposure delta maps from Step 6 concentration deltas."""
    import geopandas as gpd

    log_step_banner("Postprocess Step 7", "Plot Delta Exposure", logger=logger)
    output_dir = Path(output_dir)
    _remove_stale_outputs(output_dir)

    logger.info("Loading population …")
    pop_gdf = gpd.read_parquet(population_path, columns=["aermod_cell_id", "person_count", "geometry"])
    logger.info("Loading concentration delta …")
    delta_df = pd.read_parquet(concentration_delta_path, columns=["aermod_cell_id", "inmap_cell_id", "TotalPM25_delta"])
    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()
    if not inmap_cells_path:
        logger.warning("Skipping Step 7: InMAP cell geometries are required for delta exposure maps.")
        return {}
    logger.info("Loading InMAP cell geometries …")
    inmap_gdf = gpd.read_parquet(inmap_cells_path)[["inmap_cell_id", "geometry"]].drop_duplicates()

    logger.info("Aligning map layers to population CRS …")
    if net_gdf.crs != pop_gdf.crs:
        net_gdf = net_gdf.to_crs(pop_gdf.crs)
    if inmap_gdf.crs != pop_gdf.crs:
        inmap_gdf = inmap_gdf.to_crs(pop_gdf.crs)
    logger.info("Aggregating delta exposure metrics to InMAP cells …")
    inmap_delta_gdf = _aggregate_delta_exposure_to_inmap(pop_gdf, delta_df, inmap_gdf)
    inmap_delta_gdf = inmap_delta_gdf.loc[
        inmap_delta_gdf["population"].gt(0)
        & inmap_delta_gdf["pwc_pm25_delta"].notna()
    ].copy()
    if inmap_delta_gdf.empty:
        logger.warning("Skipping Step 7: no populated InMAP cells with PM2.5 delta.")
        return {}

    table_path = output_dir / "exposure_delta_inmap.parquet"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    inmap_delta_gdf.drop(columns="geometry").to_parquet(table_path, index=False)
    logger.info("  Saved → %s", table_path)

    outputs: dict[str, str] = {"delta_exposure_table": str(table_path)}
    with logging_redirect_tqdm():
        progress = _map_progress(3, "Postprocess Step 7")
        try:
            _set_progress_task(progress, "pwc delta inmap", step_label="Postprocess Step 7")
            result = _plot_inmap_delta_scalar(
                inmap_delta_gdf,
                net_gdf,
                column="pwc_pm25_delta",
                title="Population-weighted PM₂.₅ delta by InMAP cell",
                colorbar_label="Population-weighted PM₂.₅ delta (current - baseline, μg/m³)",
                out_path=output_dir / "pwc_inmap_delta.png",
            )
            if result:
                outputs["pwc_inmap_delta_map"] = result
            _advance_progress(progress)

            _set_progress_task(progress, "exposure burden delta", step_label="Postprocess Step 7")
            result = _plot_inmap_delta_scalar(
                inmap_delta_gdf,
                net_gdf,
                column="exposure_burden_delta",
                title="PM₂.₅ exposure burden delta by InMAP cell",
                colorbar_label="PM₂.₅ exposure burden delta (person·μg/m³)",
                out_path=output_dir / "exposure_burden_delta.png",
            )
            if result:
                outputs["exposure_burden_delta_map"] = result
            _advance_progress(progress)

            _set_progress_task(progress, "bivariate delta", step_label="Postprocess Step 7")
            result = _plot_delta_bivariate(inmap_delta_gdf, net_gdf, output_dir / "bivariate_delta.png")
            if result:
                outputs["bivariate_delta_map"] = result
            _advance_progress(progress)
        finally:
            _close_progress(progress)

    logger.info("Postprocess Step 7 complete: %d outputs written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(output_dir: Path, concentration_delta_path: str | None = None) -> dict[str, str]:
    """Run Step 7 from a pipeline output directory using manifest-resolved paths."""
    from ._common import pipeline_outputs

    output_dir = Path(output_dir)
    outs = pipeline_outputs(output_dir)
    pop_path = outs.get("beam_population_counts") or str(
        output_dir / "exposure" / "beam_population_counts.parquet"
    )
    net_path = str(output_dir / "preprocess" / "beam_osm_mapped.parquet")
    delta_path = concentration_delta_path or str(
        output_dir / "postprocess" / "delta_concentrations" / "concentration_delta.parquet"
    )
    inmap_cells_path = outs.get("beam_inmap_concentrations") or str(
        output_dir / "concentrations" / "beam_inmap_concentrations.parquet"
    )
    if not Path(inmap_cells_path).exists():
        inmap_cells_path = str(output_dir / "preprocess" / "inmap_grid.parquet")
    if not Path(inmap_cells_path).exists():
        inmap_cells_path = None
    return run(
        population_path=pop_path,
        concentration_delta_path=delta_path,
        network_path=net_path,
        inmap_cells_path=inmap_cells_path,
        output_dir=output_dir / "postprocess" / "delta_exposure",
    )


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step7_plot_delta_exposure",
        description="Plot delta exposure maps from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path, help="Path to the main pipeline output folder.")
    parser.add_argument(
        "--concentration-delta",
        help="Optional Step 6 concentration_delta.parquet path.",
    )
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir), concentration_delta_path=args.concentration_delta)
