"""Shared helpers for the postprocess package.

Importing this module configures the matplotlib backend and MPLCONFIGDIR as a
side-effect, so all step files simply do ``from . import _common`` (or import
any symbol from here) before touching matplotlib.pyplot.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "impacts-matplotlib"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colormaps — (r, g, b, a) stops: transparent at low → opaque at high
# ---------------------------------------------------------------------------

CMAP_PM25 = mcolors.LinearSegmentedColormap.from_list("pm25", [
    (1.00, 1.00, 1.00, 0.00),  # transparent white
    (1.00, 1.00, 0.80, 0.35),  # soft cream
    (1.00, 0.92, 0.23, 0.65),  # yellow
    (1.00, 0.55, 0.00, 0.85),  # orange
    (0.75, 0.05, 0.05, 1.00),  # deep red
])

CMAP_BC = mcolors.LinearSegmentedColormap.from_list("bc", [
    (1.00, 1.00, 1.00, 0.00),
    (0.90, 0.85, 0.95, 0.35),
    (0.62, 0.42, 0.80, 0.65),
    (0.40, 0.12, 0.62, 0.85),
    (0.12, 0.00, 0.22, 1.00),
])

CMAP_NO2 = mcolors.LinearSegmentedColormap.from_list("no2", [
    (1.00, 1.00, 1.00, 0.00),
    (0.87, 0.98, 0.97, 0.35),
    (0.20, 0.78, 0.72, 0.65),
    (0.00, 0.48, 0.48, 0.85),
    (0.00, 0.18, 0.28, 1.00),
])

CMAP_POP = mcolors.LinearSegmentedColormap.from_list("population", [
    (1.00, 1.00, 1.00, 0.00),
    (0.88, 0.97, 0.88, 0.35),
    (0.38, 0.78, 0.38, 0.65),
    (0.00, 0.50, 0.10, 0.85),
    (0.00, 0.20, 0.05, 1.00),
])


# ---------------------------------------------------------------------------
# Map rendering primitives (used by step4 and step5)
# ---------------------------------------------------------------------------

def _add_basemap(ax) -> None:
    try:
        import contextily as ctx
        ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto", zorder=0)
    except Exception as exc:
        logger.warning("Could not add basemap (offline or contextily missing): %s", exc)


def _add_colorbar(fig, ax, cmap, vmax: float, label: str) -> None:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.65)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)


def _plot_scalar_layer(ax, gdf, column: str, cmap, vmax: float) -> None:
    threshold = max(vmax * 0.001, 1e-9)
    subset = gdf.loc[gdf[column] > threshold, [column, "geometry"]]
    if subset.empty:
        logger.warning("  No values above threshold for %s, skipping layer.", column)
        return
    subset.plot(ax=ax, column=column, cmap=cmap, vmin=0, vmax=vmax,
                edgecolor="none", rasterized=True)


def _add_network(ax, network_gdf) -> None:
    network_gdf.plot(ax=ax, color="#111111", linewidth=0.15, alpha=0.22, rasterized=True)


# ---------------------------------------------------------------------------
# String utilities shared across step files
# ---------------------------------------------------------------------------

def _normalize_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "target"


# ---------------------------------------------------------------------------
# Manifest-based path helpers (used by run_from_output_dir in all 5 steps)
# ---------------------------------------------------------------------------

def settings_path_from_output_dir(output_dir: Path) -> Path:
    """Walk pipeline_manifest → preprocess_manifest → settings_source."""
    from impacts.manifest.file_ops import load_structured_file
    from impacts.manifest.schema import PipelineManifest, PreprocessManifest

    pm = PipelineManifest.from_dict(
        load_structured_file(Path(output_dir) / "pipeline_manifest.yaml")
    ).to_dict()
    preprocess = PreprocessManifest.from_dict(
        load_structured_file(pm["preprocess_manifest_path"])
    ).to_dict()
    return Path(preprocess["settings_source"]).resolve()


def pipeline_outputs(output_dir: Path) -> dict:
    """Return the ``outputs`` section of ``pipeline_manifest.yaml``."""
    from impacts.manifest.file_ops import load_structured_file
    from impacts.manifest.schema import PipelineManifest

    return (
        PipelineManifest.from_dict(
            load_structured_file(Path(output_dir) / "pipeline_manifest.yaml")
        ).to_dict().get("outputs", {})
        or {}
    )
