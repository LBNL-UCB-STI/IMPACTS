"""Postprocess Step 5 — Plot exposure maps.

Generates three maps:

1. **Population count** — persons per 100 m cell (green gradient).
2. **Population-weighted concentration (PWC)** — Σ(TotalPM25 × persons) / Σ(persons)
   aggregated to county level; unit stays μg/m³.
3. **Bivariate map** — each 100 m cell colored by both TotalPM25 (one axis) and
   person count (other axis) using a 3×3 palette; dark purple = high on both.

Standalone usage::

    python -m impacts.pipeline.postprocess.step5_plot_exposure /path/to/output_dir
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ...common import log_step_banner
from ._common import (
    CMAP_PM25,
    CMAP_POP,
    _add_basemap,
    _add_colorbar,
    _add_network,
)

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
# Plot 1 — population count
# ---------------------------------------------------------------------------

def _plot_population(pop_wm, net_wm, out_path: Path) -> Optional[str]:
    vals = pop_wm["person_count"].dropna()
    nonzero = vals[vals > 0]
    if nonzero.empty:
        logger.warning("  No positive population counts, skipping.")
        return None

    vmax = float(nonzero.quantile(0.99))
    logger.info("  population  vmax=%.1f → %s", vmax, out_path.name)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_aspect("equal")
    pop_wm.loc[nonzero.index, ["person_count", "geometry"]].plot(
        ax=ax, column="person_count", cmap=CMAP_POP,
        vmin=0, vmax=vmax, edgecolor="none", rasterized=True,
    )
    _add_network(ax, net_wm)
    _add_basemap(ax)
    _add_colorbar(fig, ax, CMAP_POP, vmax, "Population per 100 m cell")
    ax.set_title("Population per 100 m cell", fontsize=13, pad=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Plot 2 — population-weighted concentration by county
# ---------------------------------------------------------------------------

def _compute_pwc(pop_gdf, conc_gdf, counties_gdf) -> "gpd.GeoDataFrame":
    """Join cells to counties and compute PWC = Σ(PM25×pop) / Σ(pop) per county."""
    import geopandas as gpd

    merged = pop_gdf[["aermod_cell_id", "person_count", "geometry"]].merge(
        conc_gdf[["aermod_cell_id", "TotalPM25"]], on="aermod_cell_id", how="inner"
    )
    merged = merged[merged["person_count"] > 0].copy()

    # Spatial join cell centroids → counties (faster than polygon overlay)
    centroids = merged.copy()
    centroids["geometry"] = merged.geometry.centroid
    joined = centroids.sjoin(
        counties_gdf[["NAME", "geometry"]], how="left", predicate="within"
    )

    pwc = (
        joined.groupby("NAME")
        .apply(
            lambda g: (g["TotalPM25"] * g["person_count"]).sum() / g["person_count"].sum()
        )
        .rename("pwc_pm25")
        .reset_index()
    )

    return counties_gdf[["NAME", "geometry"]].merge(pwc, on="NAME", how="left")


def _plot_pwc_county(pop_gdf, conc_gdf, counties_gdf, net_wm, out_path: Path) -> Optional[str]:
    logger.info("  Computing population-weighted concentration by county …")
    pwc_gdf = _compute_pwc(pop_gdf, conc_gdf, counties_gdf)

    pwc_wm = pwc_gdf.to_crs(epsg=3857)
    vmax = float(pwc_wm["pwc_pm25"].dropna().max())
    logger.info("  PWC choropleth  vmax=%.4f μg/m³ → %s", vmax, out_path.name)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_aspect("equal")

    pwc_wm.plot(
        ax=ax, column="pwc_pm25", cmap=CMAP_PM25,
        vmin=0, vmax=vmax, edgecolor="#555555", linewidth=0.5, rasterized=True,
        missing_kwds={"color": "#dddddd", "edgecolor": "#555555"},
    )
    _add_network(ax, net_wm)
    _add_basemap(ax)
    _add_colorbar(fig, ax, CMAP_PM25, vmax, "Population-weighted PM₂.₅ (μg/m³)")

    # County name labels at centroid
    for _, row in pwc_wm.iterrows():
        if pd.notna(row["pwc_pm25"]):
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.text(cx, cy, row["NAME"], ha="center", va="center",
                    fontsize=7, color="#222222", fontweight="bold")

    ax.set_title("Population-weighted PM₂.₅ concentration by county (μg/m³)", fontsize=12, pad=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


# ---------------------------------------------------------------------------
# Plot 3 — bivariate map (concentration × population)
# ---------------------------------------------------------------------------

def _add_bivariate_legend(ax) -> None:
    """Draw a 3×3 color grid inset in the bottom-left corner."""
    ax_leg = ax.inset_axes([0.02, 0.02, 0.14, 0.14])
    for r in range(3):       # concentration axis (rows, bottom=low)
        for c in range(3):   # population axis (columns, left=low)
            ax_leg.add_patch(
                mpatches.Rectangle((c, r), 1, 1, color=_BV_COLORS[(r, c)],
                                   linewidth=0)
            )
    ax_leg.set_xlim(0, 3)
    ax_leg.set_ylim(0, 3)
    ax_leg.set_xticks([1.5])
    ax_leg.set_xticklabels(["Population →"], fontsize=5.5)
    ax_leg.xaxis.set_ticks_position("top")
    ax_leg.xaxis.set_label_position("top")
    ax_leg.set_yticks([1.5])
    ax_leg.set_yticklabels(["PM₂.₅ →"], fontsize=5.5, rotation=90, va="center")
    ax_leg.tick_params(length=0, pad=1)
    ax_leg.set_facecolor("none")
    for spine in ax_leg.spines.values():
        spine.set_visible(False)


def _plot_bivariate(pop_gdf, conc_gdf, net_wm, out_path: Path) -> Optional[str]:
    import geopandas as gpd
    from matplotlib.colors import BoundaryNorm, ListedColormap

    merged = pop_gdf[["aermod_cell_id", "person_count", "geometry"]].merge(
        conc_gdf[["aermod_cell_id", "TotalPM25"]], on="aermod_cell_id", how="inner"
    )
    merged = merged[(merged["person_count"] > 0) & (merged["TotalPM25"] > 0)].copy()
    if merged.empty:
        logger.warning("  No cells with both population and concentration, skipping bivariate map.")
        return None

    # Bin each variable into 3 quantile classes (0=low, 1=mid, 2=high)
    merged["conc_class"] = pd.qcut(merged["TotalPM25"], q=3, labels=[0, 1, 2]).astype(int)
    merged["pop_class"]  = pd.qcut(merged["person_count"], q=3, labels=[0, 1, 2]).astype(int)

    # Map (conc_class, pop_class) → integer 0–8
    merged["bv_class"] = merged["conc_class"] * 3 + merged["pop_class"]

    colors_9 = [_BV_COLORS[(r, c)] for r in range(3) for c in range(3)]
    cmap_bv = ListedColormap(colors_9)
    norm_bv = BoundaryNorm(np.arange(10) - 0.5, 9)

    bv_gdf = gpd.GeoDataFrame(merged[["bv_class", "geometry"]], crs=pop_gdf.crs)
    bv_wm = bv_gdf.to_crs(epsg=3857)

    logger.info("  Bivariate map  %d populated cells → %s", len(bv_wm), out_path.name)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.set_aspect("equal")
    bv_wm.plot(ax=ax, column="bv_class", cmap=cmap_bv, norm=norm_bv,
               edgecolor="none", rasterized=True)
    _add_network(ax, net_wm)
    _add_basemap(ax)
    _add_bivariate_legend(ax)

    ax.set_title("PM₂.₅ concentration × population density", fontsize=13, pad=10)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
    county_boundaries_path: str,
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
    county_boundaries_path:
        Path to staged county boundaries (GeoPackage or GeoJSON).
    output_dir:
        Directory where PNG files are written.
    """
    import geopandas as gpd

    log_step_banner("Postprocess Step 5", "Plot Exposure", logger=logger)
    output_dir = Path(output_dir)

    logger.info("Loading population …")
    pop_gdf = gpd.read_parquet(population_path)
    logger.info("Loading concentration …")
    conc_gdf = gpd.read_parquet(concentration_path)
    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()
    logger.info("Loading county boundaries …")
    counties_gdf = gpd.read_file(county_boundaries_path)[["NAME", "geometry"]]

    logger.info("Reprojecting network to Web Mercator …")
    net_wm = net_gdf.to_crs(epsg=3857)
    del net_gdf

    outputs: dict[str, str] = {}

    result = _plot_population(pop_gdf.to_crs(epsg=3857), net_wm, output_dir / "population.png")
    if result:
        outputs["population_map"] = result

    result = _plot_pwc_county(pop_gdf, conc_gdf, counties_gdf, net_wm,
                              output_dir / "pwc_county.png")
    if result:
        outputs["pwc_county_map"] = result

    result = _plot_bivariate(pop_gdf, conc_gdf, net_wm, output_dir / "bivariate.png")
    if result:
        outputs["bivariate_map"] = result

    logger.info("Postprocess Step 5 complete: %d maps written to %s", len(outputs), output_dir)
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 5 from a pipeline output directory using manifest-resolved paths."""
    from impacts.postprocessor import _resolve_county_boundaries_path

    from ._common import pipeline_outputs, settings_path_from_output_dir

    output_dir = Path(output_dir)
    outs = pipeline_outputs(output_dir)
    conc_path = outs.get("beam_concentration_distribution") or str(
        output_dir / "exposure" / "beam_concentration_distribution.parquet"
    )
    pop_path = outs.get("beam_population_counts") or str(
        output_dir / "exposure" / "beam_population_counts.parquet"
    )
    net_path = str(output_dir / "preprocess" / "beam_osm_mapped.parquet")
    settings_path = settings_path_from_output_dir(output_dir)
    county_path = _resolve_county_boundaries_path(settings_path)
    return run(
        population_path=pop_path,
        concentration_path=conc_path,
        network_path=net_path,
        county_boundaries_path=str(county_path),
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
