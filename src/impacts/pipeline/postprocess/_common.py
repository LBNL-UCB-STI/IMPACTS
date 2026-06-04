"""Shared helpers for the postprocess package.

Importing any symbol from this module configures the matplotlib backend and
MPLCONFIGDIR before step files touch matplotlib.pyplot.
"""
from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "impacts-matplotlib"))

import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

MAP_CONTEXT_PADDING_FRACTION = 0.04
MAP_CONTEXT_MIN_PADDING = 2_000.0
PLOT_DPI = 300
CHART_TITLE_FONTSIZE = 18
CHART_AXIS_LABEL_FONTSIZE = 16
CHART_TICK_LABEL_FONTSIZE = 13
CHART_LEGEND_FONTSIZE = 13
MAP_FIGSIZE = (24, 24)
MAP_DPI = 400
MAP_TITLE_FONTSIZE = 28
MAP_BASEMAP_ZOOM_ADJUST = 1
MAP_COLORBAR_FRACTION = 0.04
MAP_COLORBAR_PAD = 0.025
MAP_COLORBAR_SHRINK = 0.74
MAP_COLORBAR_LABEL_FONTSIZE = 24
MAP_COLORBAR_TICK_FONTSIZE = 20
MAP_COLORBAR_LABELPAD = 18

# ---------------------------------------------------------------------------
# Colormaps — fully opaque, mild low values → saturated high values
# ---------------------------------------------------------------------------

CMAP_PM25 = mcolors.LinearSegmentedColormap.from_list("pm25", [
    (1.00, 1.00, 0.82, 1.00),  # mild yellow
    (1.00, 0.92, 0.36, 1.00),  # yellow
    (0.98, 0.62, 0.12, 1.00),  # orange
    (0.86, 0.20, 0.12, 1.00),  # red
    (0.45, 0.00, 0.05, 1.00),  # dark red
])

CMAP_BC = mcolors.LinearSegmentedColormap.from_list("bc", [
    (0.95, 0.93, 0.98, 1.00),
    (0.82, 0.74, 0.90, 1.00),
    (0.62, 0.42, 0.80, 1.00),
    (0.40, 0.12, 0.62, 1.00),
    (0.12, 0.00, 0.22, 1.00),
])

CMAP_NO2 = mcolors.LinearSegmentedColormap.from_list("no2", [
    (0.86, 0.97, 0.99, 1.00),
    (0.56, 0.86, 0.92, 1.00),
    (0.18, 0.66, 0.78, 1.00),
    (0.00, 0.38, 0.56, 1.00),
    (0.00, 0.12, 0.26, 1.00),
])

CMAP_POP = mcolors.LinearSegmentedColormap.from_list("population", [
    (0.91, 0.98, 0.90, 1.00),
    (0.68, 0.88, 0.62, 1.00),
    (0.34, 0.68, 0.30, 1.00),
    (0.04, 0.45, 0.16, 1.00),
    (0.00, 0.20, 0.05, 1.00),
])


# ---------------------------------------------------------------------------
# Map rendering primitives (used by step4 and step5)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _GridRasterLayout:
    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    extent: tuple[float, float, float, float]

    @property
    def padded_extent(self) -> tuple[float, float, float, float]:
        return _padded_extent(self.extent)


def _padded_extent(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = extent
    x_padding = max((xmax - xmin) * MAP_CONTEXT_PADDING_FRACTION, MAP_CONTEXT_MIN_PADDING)
    y_padding = max((ymax - ymin) * MAP_CONTEXT_PADDING_FRACTION, MAP_CONTEXT_MIN_PADDING)
    return (
        xmin - x_padding,
        xmax + x_padding,
        ymin - y_padding,
        ymax + y_padding,
    )


def _padded_extent_from_bounds(bounds) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = (float(value) for value in bounds)
    return _padded_extent((minx, maxx, miny, maxy))


def _add_basemap(ax, *, crs=None) -> None:
    try:
        import contextily as ctx
        ctx.add_basemap(
            ax,
            source=ctx.providers.CartoDB.PositronNoLabels,
            zoom="auto",
            zoom_adjust=MAP_BASEMAP_ZOOM_ADJUST,
            crs=crs,
            zorder=0,
        )
    except Exception as exc:
        logger.warning("Could not add basemap (offline or contextily missing): %s", exc)


def _add_colorbar(fig, ax, cmap, vmax: float, label: str, *, norm=None) -> None:
    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm if norm is not None else plt.Normalize(vmin=0, vmax=vmax),
    )
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        fraction=MAP_COLORBAR_FRACTION,
        pad=MAP_COLORBAR_PAD,
        shrink=MAP_COLORBAR_SHRINK,
    )
    cbar.set_label(label, fontsize=MAP_COLORBAR_LABEL_FONTSIZE, labelpad=MAP_COLORBAR_LABELPAD)
    cbar.ax.tick_params(labelsize=MAP_COLORBAR_TICK_FONTSIZE, pad=6)
    cbar.ax.yaxis.get_offset_text().set_fontsize(MAP_COLORBAR_TICK_FONTSIZE)


def _style_chart_axes(ax, *, legend: bool = True) -> None:
    ax.title.set_fontsize(CHART_TITLE_FONTSIZE)
    ax.xaxis.label.set_fontsize(CHART_AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(CHART_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=CHART_TICK_LABEL_FONTSIZE)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=CHART_LEGEND_FONTSIZE)


def _grid_raster_layout(gdf) -> _GridRasterLayout:
    bounds = gdf.geometry.bounds
    widths = (bounds["maxx"] - bounds["minx"]).to_numpy(dtype="float64")
    heights = (bounds["maxy"] - bounds["miny"]).to_numpy(dtype="float64")
    valid = np.flatnonzero((widths > 0) & (heights > 0))
    if len(valid) == 0:
        raise ValueError("Cannot rasterize grid: no non-empty geometries found.")

    sample = valid[: min(len(valid), 1000)]
    cell_width = float(np.median(widths[sample]))
    cell_height = float(np.median(heights[sample]))
    xmin = float(bounds["minx"].min())
    ymin = float(bounds["miny"].min())

    cols = np.rint((bounds["minx"].to_numpy(dtype="float64") - xmin) / cell_width).astype(np.int32)
    rows = np.rint((bounds["miny"].to_numpy(dtype="float64") - ymin) / cell_height).astype(np.int32)
    ncols = int(cols.max()) + 1
    nrows = int(rows.max()) + 1
    extent = (
        xmin,
        xmin + ncols * cell_width,
        ymin,
        ymin + nrows * cell_height,
    )
    return _GridRasterLayout(rows=rows, cols=cols, shape=(nrows, ncols), extent=extent)


def _rasterize_grid_values(
    values,
    layout: _GridRasterLayout,
    *,
    threshold: float | None = None,
    dtype: str = "float32",
) -> np.ma.MaskedArray:
    arr = np.full(layout.shape, np.nan, dtype=dtype)
    value_arr = np.asarray(values, dtype=dtype)
    valid = np.isfinite(value_arr)
    if threshold is not None:
        valid &= value_arr > threshold
    valid &= (
        (layout.rows >= 0)
        & (layout.rows < layout.shape[0])
        & (layout.cols >= 0)
        & (layout.cols < layout.shape[1])
    )
    arr[layout.rows[valid], layout.cols[valid]] = value_arr[valid]
    return np.ma.masked_invalid(arr)


def _plot_raster_layer(
    ax,
    gdf,
    column: str,
    cmap,
    *,
    layout: _GridRasterLayout,
    vmax: float | None = None,
    threshold: float | None = None,
    norm=None,
) -> None:
    raster = _rasterize_grid_values(gdf[column], layout, threshold=threshold)
    if raster.count() == 0:
        logger.warning("  No raster cells to plot for %s.", column)
        return
    plot_cmap = cmap.copy()
    plot_cmap.set_bad((1, 1, 1, 0))
    ax.imshow(
        raster,
        extent=layout.extent,
        origin="lower",
        cmap=plot_cmap,
        vmin=None if norm is not None else 0,
        vmax=None if norm is not None else vmax,
        norm=norm,
        interpolation="nearest",
        zorder=2,
    )
    xmin, xmax, ymin, ymax = layout.padded_extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def _add_network(ax, network_gdf) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    network_gdf.plot(ax=ax, color="#111111", linewidth=0.15, alpha=0.22, rasterized=True, zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def _step_progress(total: int, desc: str, *, unit: str = "task"):
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
        file=sys.stdout,
        disable=not (logger.isEnabledFor(logging.INFO) and sys.stdout.isatty()),
    )


def _map_progress(total: int, desc: str):
    return _step_progress(total, desc, unit="map")


def _set_progress_task(progress, label: str, *, step_label: str) -> None:
    if progress.disable:
        logger.info("%s progress: %s", step_label, label)
        return
    progress.set_postfix_str(label)


def _advance_progress(progress) -> None:
    progress.update(1)


def _close_progress(progress) -> None:
    progress.close()


# ---------------------------------------------------------------------------
# String utilities shared across step files
# ---------------------------------------------------------------------------

def _normalize_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "target"


def _duckdb_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# Manifest-based path helpers (used by run_from_output_dir in all 5 steps)
# ---------------------------------------------------------------------------

def settings_path_from_output_dir(output_dir: Path) -> Path:
    """Walk pipeline_manifest → preprocess_manifest → settings_source."""
    from impacts.postprocessor import _resolve_settings_path

    output_dir = Path(output_dir)
    return _resolve_settings_path(output_dir / "pipeline_manifest.yaml", output_root=output_dir)


def pipeline_outputs(output_dir: Path) -> dict:
    """Return the ``outputs`` section of ``pipeline_manifest.yaml``."""
    from impacts.postprocessor import _localized_pipeline_manifest

    output_dir = Path(output_dir)
    return _localized_pipeline_manifest(
        output_dir / "pipeline_manifest.yaml",
        output_root=output_dir,
    ).get("outputs", {}) or {}
