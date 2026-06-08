"""Postprocess Step 5 — Plot exposure maps.

Generates three InMAP-cell exposure maps:

1. **Population count** — persons summed by InMAP cell.
2. **Population-weighted concentration (PWC)** — Σ(TotalPM25 × persons) / Σ(persons)
   by InMAP cell; unit stays μg/m³.
3. **Bivariate map** — each InMAP cell colored by both PWC (one axis) and
   population (other axis) using a 3×3 palette; dark purple = high on both.

Standalone usage::

    python -m impacts.pipeline.postprocess.step5_plot_exposure /path/to/output_dir
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ...common import log_step_banner, configure_duckdb_connection
from ._common import (
    CMAP_PM25,
    CMAP_POP,
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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bivariate color table — Stevens (2015) DivergingBlue × DivergingRed scheme
# key: (conc_class, pop_class), classes 0=low 1=mid 2=high
# ---------------------------------------------------------------------------
_BV_COLORS: dict[tuple[int, int], str] = {
    (0, 0): "#e8e8e8",  # low conc, low pop  → light gray
    (0, 1): "#ace4e4",  # low conc, mid pop  → light teal
    (0, 2): "#5ac8c8",  # low conc, high pop → teal
    (1, 0): "#dfb0d6",  # mid conc, low pop  → light lavender
    (1, 1): "#a5b3cc",  # mid conc, mid pop  → blue-gray
    (1, 2): "#5698b9",  # mid conc, high pop → blue
    (2, 0): "#c85a5a",  # high conc, low pop → red
    (2, 1): "#b05ba5",  # high conc, mid pop → magenta
    (2, 2): "#574b9f",  # high conc, high pop → dark purple (hotspot)
}


# ---------------------------------------------------------------------------
# InMAP exposure aggregation
# ---------------------------------------------------------------------------

def _aggregate_exposure_to_inmap(pop_gdf, conc_gdf, inmap_gdf, *, output_dir: Path):
    import duckdb

    pop_df = pd.DataFrame(pop_gdf[["aermod_cell_id", "person_count"]])
    conc_df = pd.DataFrame(conc_gdf[["aermod_cell_id", "inmap_cell_id", "TotalPM25"]])

    con = duckdb.connect()
    try:
        configure_duckdb_connection(con, working_dir=output_dir, show_progress=False, profile="balanced")
        con.register("pop_tbl", pop_df)
        con.register("conc_tbl", conc_df)

        result_df = con.execute("""
            SELECT
                TRY_CAST(c.inmap_cell_id AS BIGINT) AS inmap_cell_id,
                SUM(TRY_CAST(p.person_count AS DOUBLE)) AS population,
                SUM(TRY_CAST(c.TotalPM25 AS DOUBLE) * TRY_CAST(p.person_count AS DOUBLE)) AS exposure_burden
            FROM pop_tbl p
            INNER JOIN conc_tbl c ON p.aermod_cell_id = c.aermod_cell_id
            WHERE TRY_CAST(p.person_count AS DOUBLE) > 0
              AND TRY_CAST(c.TotalPM25 AS DOUBLE) IS NOT NULL
              AND TRY_CAST(c.inmap_cell_id AS BIGINT) IS NOT NULL
            GROUP BY c.inmap_cell_id
        """).df()
    finally:
        con.close()

    if result_df.empty:
        empty = inmap_gdf.iloc[0:0][["inmap_cell_id", "geometry"]].copy()
        empty["population"] = pd.Series(dtype="float64")
        empty["exposure_burden"] = pd.Series(dtype="float64")
        empty["pwc_pm25"] = pd.Series(dtype="float64")
        return empty

    result_df["pwc_pm25"] = result_df["exposure_burden"] / result_df["population"]

    inmap = inmap_gdf.copy()
    inmap["inmap_cell_id"] = pd.to_numeric(inmap["inmap_cell_id"], errors="coerce")
    return inmap[["inmap_cell_id", "geometry"]].merge(result_df, on="inmap_cell_id", how="inner")


# ---------------------------------------------------------------------------
# InMAP map rendering
# ---------------------------------------------------------------------------

def _plot_inmap_scalar(
    inmap_exposure_gdf,
    net_gdf,
    *,
    column: str,
    title: str,
    colorbar_label: str,
    cmap,
    out_path: Path,
    gamma: float = 1.0,
) -> Optional[str]:
    values = inmap_exposure_gdf[column].dropna()
    positive = values[values > 0]
    if positive.empty:
        logger.warning("  No positive values for %s, skipping.", column)
        return None

    plot_gdf = inmap_exposure_gdf.loc[inmap_exposure_gdf[column].gt(0), [column, "geometry"]].copy()
    vmax = float(positive.quantile(0.99))
    logger.info("  %s  cells=%d vmax=%.4f → %s", column, len(plot_gdf), vmax, out_path.name)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE, dpi=MAP_DPI)
    ax.set_aspect("equal")
    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax)
    plot_gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
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
    _add_colorbar(fig, ax, cmap, vmax, colorbar_label, norm=norm)

    ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Plot 3 — bivariate map (PWC × population)
# ---------------------------------------------------------------------------

def _add_bivariate_legend(ax) -> None:
    """Draw a 3×3 color grid inset in the bottom-left corner."""
    ax_leg = ax.inset_axes([0.02, 0.02, 0.20, 0.20])
    for r in range(3):       # concentration axis (rows, bottom=low)
        for c in range(3):   # population axis (columns, left=low)
            ax_leg.add_patch(
                mpatches.Rectangle((c, r), 1, 1, color=_BV_COLORS[(r, c)],
                                   linewidth=0)
            )
    ax_leg.set_xlim(0, 3)
    ax_leg.set_ylim(0, 3)
    ax_leg.set_xticks([1.5])
    ax_leg.set_xticklabels(["Population →"], fontsize=18)
    ax_leg.xaxis.set_ticks_position("top")
    ax_leg.xaxis.set_label_position("top")
    ax_leg.set_yticks([1.5])
    ax_leg.set_yticklabels(["PM₂.₅ →"], fontsize=18, rotation=90, va="center")
    ax_leg.tick_params(length=0, pad=6)
    ax_leg.set_facecolor("none")
    for spine in ax_leg.spines.values():
        spine.set_visible(False)


def _tertile_classes(values: pd.Series) -> pd.Series:
    if len(values) < 3 or values.nunique(dropna=True) < 2:
        return pd.Series(np.zeros(len(values), dtype=int), index=values.index)
    return pd.qcut(values.rank(method="first"), q=3, labels=[0, 1, 2]).astype(int)


def _plot_bivariate(inmap_exposure_gdf, net_gdf, out_path: Path) -> Optional[str]:
    from matplotlib.colors import BoundaryNorm, ListedColormap

    merged = inmap_exposure_gdf[
        ["population", "pwc_pm25", "geometry"]
    ].loc[
        inmap_exposure_gdf["population"].gt(0) & inmap_exposure_gdf["pwc_pm25"].gt(0)
    ].copy()
    if merged.empty:
        logger.warning("  No cells with both population and concentration, skipping bivariate map.")
        return None

    # Bin each variable into 3 quantile classes (0=low, 1=mid, 2=high)
    merged["conc_class"] = _tertile_classes(merged["pwc_pm25"])
    merged["pop_class"] = _tertile_classes(merged["population"])

    # Map (conc_class, pop_class) → integer 0–8
    merged["bv_class"] = merged["conc_class"] * 3 + merged["pop_class"]

    colors_9 = [_BV_COLORS[(r, c)] for r in range(3) for c in range(3)]
    cmap_bv = ListedColormap(colors_9)
    norm_bv = BoundaryNorm(np.arange(10) - 0.5, 9)

    logger.info("  Bivariate InMAP map  %d populated cells → %s", len(merged), out_path.name)

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
    _add_basemap(ax, crs=inmap_exposure_gdf.crs)
    _add_bivariate_legend(ax)

    ax.set_title(
        "Population-weighted PM₂.₅ × population by InMAP cell",
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    *,
    population_path: str,
    concentration_path: str,
    network_path: str,
    inmap_cells_path: str | None = None,
    output_dir: Path,
) -> dict[str, str]:
    """Render all exposure maps.

    Parameters
    ----------
    population_path:
        Path to ``beam_population_counts.parquet``.
    concentration_path:
        Path to ``beam_concentration_distribution.parquet``.
    network_path:
        Path to ``beam_osm_mapped.parquet``.
    inmap_cells_path:
        Path to InMAP cell geometries, preferably ``beam_inmap_concentrations.parquet``
        or the staged ``inmap_grid.parquet``.
    output_dir:
        Directory where PNG files are written.
    """
    import geopandas as gpd

    log_step_banner("Postprocess Step 5", "Plot Exposure", logger=logger)
    output_dir = Path(output_dir)

    logger.info("Loading population …")
    pop_gdf = gpd.read_parquet(population_path, columns=["aermod_cell_id", "person_count", "geometry"])
    logger.info("Loading concentration …")
    conc_gdf = pd.read_parquet(concentration_path, columns=["aermod_cell_id", "inmap_cell_id", "TotalPM25"])
    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()
    if not inmap_cells_path:
        logger.warning("Skipping Step 5: InMAP cell geometries are required for exposure maps.")
        return {}
    logger.info("Loading InMAP cell geometries …")
    inmap_gdf = gpd.read_parquet(inmap_cells_path)[["inmap_cell_id", "geometry"]].drop_duplicates()

    logger.info("Aligning map layers to population CRS …")
    if net_gdf.crs != pop_gdf.crs:
        net_gdf = net_gdf.to_crs(pop_gdf.crs)
    if inmap_gdf.crs != pop_gdf.crs:
        inmap_gdf = inmap_gdf.to_crs(pop_gdf.crs)
    logger.info("Aggregating exposure metrics to InMAP cells …")
    inmap_exposure_gdf = _aggregate_exposure_to_inmap(pop_gdf, conc_gdf, inmap_gdf, output_dir=output_dir)
    inmap_exposure_gdf = inmap_exposure_gdf.loc[
        inmap_exposure_gdf["population"].gt(0)
        & inmap_exposure_gdf["pwc_pm25"].notna()
    ].copy()
    if inmap_exposure_gdf.empty:
        logger.warning("Skipping Step 5: no populated InMAP cells with PM2.5 concentration.")
        return {}

    outputs: dict[str, str] = {}

    progress = _map_progress(3, "Postprocess Step 5")
    try:
        _set_progress_task(progress, "population inmap", step_label="Postprocess Step 5")
        result = _plot_inmap_scalar(
            inmap_exposure_gdf,
            net_gdf,
            column="population",
            title="Population by InMAP cell",
            colorbar_label="Population per InMAP cell",
            cmap=CMAP_POP,
            out_path=output_dir / "population.png",
            gamma=0.55,
        )
        if result:
            outputs["population_map"] = result
        _advance_progress(progress)

        _set_progress_task(progress, "pwc inmap", step_label="Postprocess Step 5")
        result = _plot_inmap_scalar(
            inmap_exposure_gdf,
            net_gdf,
            column="pwc_pm25",
            title="Population-weighted PM₂.₅ concentration by InMAP cell",
            colorbar_label="Population-weighted PM₂.₅ (μg/m³)",
            cmap=CMAP_PM25,
            out_path=output_dir / "pwc_inmap.png",
            gamma=0.65,
        )
        if result:
            outputs["pwc_inmap_map"] = result
        _advance_progress(progress)

        _set_progress_task(progress, "bivariate", step_label="Postprocess Step 5")
        result = _plot_bivariate(inmap_exposure_gdf, net_gdf, output_dir / "bivariate.png")
        if result:
            outputs["bivariate_map"] = result
        _advance_progress(progress)
    finally:
        _close_progress(progress)

    logger.info("Postprocess Step 5 complete: %d maps written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 5 from a pipeline output directory using manifest-resolved paths."""
    from ._common import pipeline_outputs

    output_dir = Path(output_dir)
    outs = pipeline_outputs(output_dir)
    conc_path = outs.get("beam_concentration_distribution") or str(
        output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )
    pop_path = outs.get("beam_population_counts") or str(
        output_dir / "exposure" / "beam_population_counts.parquet"
    )
    net_path = str(output_dir / "preprocess" / "beam_osm_mapped.parquet")
    inmap_cells_path = outs.get("beam_inmap_concentrations") or str(
        output_dir / "concentrations" / "beam_inmap_concentrations.parquet"
    )
    if not Path(inmap_cells_path).exists():
        inmap_cells_path = str(output_dir / "preprocess" / "inmap_grid.parquet")
    if not Path(inmap_cells_path).exists():
        inmap_cells_path = None
    return run(
        population_path=pop_path,
        concentration_path=conc_path,
        network_path=net_path,
        inmap_cells_path=inmap_cells_path,
        output_dir=output_dir / "postprocess" / "exposure",
    )


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step5_plot_exposure",
        description="Plot exposure maps from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path,
                        help="Path to the main pipeline output folder.")
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir))
