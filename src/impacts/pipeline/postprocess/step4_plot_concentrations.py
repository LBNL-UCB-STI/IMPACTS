"""Postprocess Step 4 — Plot concentration maps.

Generates one PNG per concentration variable (TotalPM25, PrimaryPM25,
SecondaryPM25, BC, NO2) with an OSM basemap and road network overlay.

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
    MAP_DPI,
    MAP_FIGSIZE,
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

import matplotlib.pyplot as plt
from tqdm.contrib.logging import logging_redirect_tqdm

logger = logging.getLogger(__name__)

# (column, display_title, colormap, vmax_percentile)
_LAYERS = [
    ("TotalPM25",     "Total PM₂.₅ (μg/m³)",     CMAP_PM25, 0.99),
    ("PrimaryPM25",   "Primary PM₂.₅ (μg/m³)",   CMAP_PM25, 0.99),
    ("SecondaryPM25", "Secondary PM₂.₅ (μg/m³)", CMAP_PM25, 0.99),
    ("BC",            "Black Carbon (μg/m³)",      CMAP_BC,   0.99),
    ("NO2",           "NO₂ (μg/m³)",               CMAP_NO2,  0.99),
]


def _plot_one(conc_gdf, net_gdf, layout, column: str, title: str, cmap, vmax_q: float, out_path: Path) -> Optional[str]:
    if column not in conc_gdf.columns:
        logger.warning("  Column %s not in concentration data, skipping.", column)
        return None

    vals = conc_gdf[column].dropna()
    nonzero = vals[vals > 0]
    if nonzero.empty:
        logger.warning("  Column %s has no positive values, skipping.", column)
        return None

    vmax = float(nonzero.quantile(vmax_q))
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
    _add_colorbar(fig, ax, cmap, vmax, title)
    ax.set_title(title, fontsize=MAP_TITLE_FONTSIZE, pad=16)
    ax.set_axis_off()
    fig.tight_layout(pad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=MAP_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved → %s", out_path)
    return str(out_path)


def run(
    *,
    concentration_path: str,
    network_path: str,
    output_dir: Path,
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

    log_step_banner("Postprocess Step 4", "Plot Concentrations", logger=logger)
    output_dir = Path(output_dir)

    concentration_columns = ["geometry", *[column for column, _, _, _ in _LAYERS]]
    logger.info("Loading concentration data …")
    conc_gdf = gpd.read_parquet(concentration_path, columns=concentration_columns)
    logger.info("Loading network …")
    net_gdf = gpd.read_parquet(network_path)[["geometry"]].drop_duplicates()

    logger.info("Building native grid raster layout …")
    layout = _grid_raster_layout(conc_gdf)

    outputs: dict[str, str] = {}
    with logging_redirect_tqdm():
        progress = _map_progress(len(_LAYERS), "Postprocess Step 4")
        try:
            for column, title, cmap, vmax_q in _LAYERS:
                _set_progress_task(progress, column, step_label="Postprocess Step 4")
                result = _plot_one(conc_gdf, net_gdf, layout, column, title, cmap, vmax_q,
                                   output_dir / f"{column.lower()}.png")
                if result is not None:
                    outputs[f"{column.lower()}_map"] = result
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
