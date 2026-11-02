"""Generate the original-style interactive human-exposure web map."""
from __future__ import annotations

import base64
import html
import io
import json
import logging
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import Transformer
from shapely.geometry import mapping

from ._common import CMAP_GREEN_RED, _grid_raster_layout

logger = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).with_name("assets") / "human_exposure_webmap.html"
_CES_GROUPS = {"0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"}
_RASTER_FIELDS = {
    "impm": "inmap_only_TotalPM25",
    "aepm": "TotalPM25",
    "imbc": "inmap_only_BC",
    "fbc": "BC",
    "imno2": "inmap_only_NO2",
    "fno2": "NO2",
}
_DELTA_RASTER_FIELDS = {
    "dpm": ("inmap_only_TotalPM25", "TotalPM25"),
    "dbc": ("inmap_only_BC", "BC"),
    "dno2": ("inmap_only_NO2", "NO2"),
}


def _number(value: Any, digits: int = 6) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(numeric, digits) if math.isfinite(numeric) else 0.0


def _replace_json_constant(template: str, name: str, value: Any) -> str:
    prefix = f"const {name} = "
    start = template.index(prefix) + len(prefix)
    _, length = json.JSONDecoder().raw_decode(template[start:])
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return template[:start] + payload + template[start + length:]


def _wrap_original_artifact(inner_html: str) -> str:
    """Restore the exact outer iframe layout used by the original web map."""
    escaped = html.escape(inner_html, quote=True)
    return f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'unsafe-eval' 'wasm-unsafe-eval' blob: data: https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://esm.sh https://fonts.bunny.net https://fonts.googleapis.com https://fonts.gstatic.com https://unpkg.com; style-src 'unsafe-inline' blob: data: https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://esm.sh https://fonts.bunny.net https://fonts.googleapis.com https://fonts.gstatic.com https://unpkg.com; img-src blob: data: https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://esm.sh https://fonts.bunny.net https://fonts.googleapis.com https://fonts.gstatic.com https://unpkg.com; font-src blob: data: https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://esm.sh https://fonts.bunny.net https://fonts.googleapis.com https://fonts.gstatic.com https://unpkg.com; media-src blob: data: https://cdnjs.cloudflare.com https://cdn.jsdelivr.net https://esm.sh https://fonts.bunny.net https://fonts.googleapis.com https://fonts.gstatic.com https://unpkg.com; worker-src blob:; connect-src blob: data:; frame-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Bay Area Direct Final Pm25 Bc No2 Map</title>
<style>:root{{color-scheme:light dark;background:light-dark(rgb(255 255 255), rgb(24 24 24))}}html,body{{margin:0}}body{{box-sizing:border-box;padding:1rem;background:inherit}}iframe{{display:block;width:100%;max-width:736px;height:calc(100vh - 2rem);margin:0 auto;border:0}}</style>
</head>
<body>
<iframe sandbox="allow-scripts" referrerpolicy="no-referrer" title="Bay Area Direct Final Pm25 Bc No2 Map" srcdoc="{escaped}"></iframe>
</body>
</html>
'''


def _tract_data(path: str) -> dict[str, Any]:
    frame = gpd.read_parquet(path)
    frame = frame.loc[
        frame["CES_Group"].isin(_CES_GROUPS)
        & pd.to_numeric(frame["ModelPopulation"], errors="coerce").fillna(0).gt(0)
    ].to_crs(4326)
    frame.geometry = frame.geometry.simplify(0.00004, preserve_topology=True)
    features = []
    for row in frame.itertuples(index=False):
        geometry = mapping(row.geometry)
        aepm = _number(row.Final_TotalPM25_PW)
        impm = _number(row.InMAP_TotalPM25_PW)
        fbc = _number(row.Final_BC_PW)
        imbc = _number(row.InMAP_BC_PW)
        fno2 = _number(row.Final_NO2_PW)
        imno2 = _number(row.InMAP_NO2_PW)
        features.append({
            "type": "Feature",
            "geometry": geometry,
            "properties": {
                "id": str(getattr(row, "TractID", getattr(row, "Tract", ""))),
                "county": str(getattr(row, "County", "")),
                "group": str(row.CES_Group),
                "aepm": aepm,
                "impm": impm,
                "dpm": _number(impm - aepm),
                "fbc": fbc,
                "imbc": imbc,
                "dbc": _number(imbc - fbc),
                "fno2": fno2,
                "imno2": imno2,
                "dno2": _number(imno2 - fno2),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _facility_data(path: str, *, name_field: str, weight_field: str) -> list[dict[str, Any]]:
    frame = gpd.read_parquet(path)
    frame = frame.loc[frame["CES_Group"].isin(_CES_GROUPS)].to_crs(4326)
    points = frame.geometry
    result = []
    for row, point in zip(frame.itertuples(index=False), points):
        if point is None or point.is_empty:
            continue
        aepm = _number(row.Final_TotalPM25)
        impm = _number(row.InMAP_TotalPM25)
        fbc = _number(row.Final_BC)
        imbc = _number(row.InMAP_BC)
        fno2 = _number(row.Final_NO2)
        imno2 = _number(row.InMAP_NO2)
        result.append({
            "lon": _number(point.x, 5),
            "lat": _number(point.y, 5),
            "name": str(getattr(row, name_field, "")),
            "county": str(getattr(row, "County", "")),
            "group": str(row.CES_Group),
            "weight": _number(getattr(row, weight_field, 0), 2),
            "aepm": aepm,
            "impm": impm,
            "dpm": _number(impm - aepm),
            "fbc": fbc,
            "imbc": imbc,
            "dbc": _number(imbc - fbc),
            "fno2": fno2,
            "imno2": imno2,
            "dno2": _number(imno2 - fno2),
        })
    return result


def _power_plant_data(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    frame = pd.read_csv(path)
    pollutant_columns = [column for column in ("PM2_5_kg", "NOx_kg", "SOx_kg", "NH3_kg", "VOC_kg") if column in frame]
    if pollutant_columns:
        emitting = frame[pollutant_columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1).gt(0)
        frame = frame.loc[emitting]
    result = []
    for row in frame.itertuples(index=False):
        result.append({
            "id": str(getattr(row, "ORISPL", "")),
            "name": str(getattr(row, "PNAME", "")),
            "lon": _number(getattr(row, "LON", 0), 6),
            "lat": _number(getattr(row, "LAT", 0), 6),
            "generation": _number(getattr(row, "added_generation_MWh", 0), 12),
            "technology": str(getattr(row, "marginal_technologies", "")),
            **{column: _number(getattr(row, column, 0), 12) for column in pollutant_columns},
        })
    return result


def _maximum_composite(values: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                       source_shape: tuple[int, int], max_dimension: int = 960) -> np.ndarray:
    source_rows, source_cols = source_shape
    scale = min(1.0, max_dimension / max(source_rows, source_cols))
    height = max(1, int(round(source_rows * scale)))
    width = max(1, int(round(source_cols * scale)))
    target_rows = np.minimum((rows.astype(np.float64) * height / source_rows).astype(np.int32), height - 1)
    target_cols = np.minimum((cols.astype(np.float64) * width / source_cols).astype(np.int32), width - 1)
    raster = np.full(height * width, -np.inf, dtype=np.float32)
    valid = np.isfinite(values)
    np.maximum.at(raster, target_rows[valid] * width + target_cols[valid], values[valid].astype(np.float32))
    raster = raster.reshape(height, width)
    raster[~np.isfinite(raster)] = np.nan
    return raster


def _absolute_maximum_composite(values: np.ndarray, rows: np.ndarray, cols: np.ndarray,
                                source_shape: tuple[int, int], max_dimension: int = 960) -> np.ndarray:
    """Downsample signed values while retaining the largest absolute response."""
    source_rows, source_cols = source_shape
    scale = min(1.0, max_dimension / max(source_rows, source_cols))
    height = max(1, int(round(source_rows * scale)))
    width = max(1, int(round(source_cols * scale)))
    target_rows = np.minimum((rows.astype(np.float64) * height / source_rows).astype(np.int32), height - 1)
    target_cols = np.minimum((cols.astype(np.float64) * width / source_cols).astype(np.int32), width - 1)
    target = target_rows * width + target_cols
    valid = np.isfinite(values)
    positive = np.full(height * width, -np.inf, dtype=np.float32)
    negative = np.full(height * width, np.inf, dtype=np.float32)
    np.maximum.at(positive, target[valid], values[valid].astype(np.float32))
    np.minimum.at(negative, target[valid], values[valid].astype(np.float32))
    positive[~np.isfinite(positive)] = np.nan
    negative[~np.isfinite(negative)] = np.nan
    choose_negative = np.nan_to_num(np.abs(negative), nan=-1) > np.nan_to_num(np.abs(positive), nan=-1)
    raster = np.where(choose_negative, negative, positive).reshape(height, width)
    raster[~np.isfinite(raster)] = np.nan
    return raster


def _png_data_url(raster: np.ndarray, cap: float) -> str:
    normalized = np.clip(np.nan_to_num(raster, nan=0.0) / cap, 0, 1) if cap > 0 else np.zeros_like(raster)
    rgba = (CMAP_GREEN_RED(normalized) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(raster) & (raster > 0), 222, 0).astype(np.uint8)
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _signed_png_data_url(raster: np.ndarray, cap: float) -> str:
    normalized = np.clip((np.nan_to_num(raster, nan=0.0) + cap) / (2 * cap), 0, 1) if cap > 0 else np.full_like(raster, 0.5)
    # RdBu maps negative differences to red, zero to white, and positive to blue.
    from matplotlib import pyplot as plt
    rgba = (plt.get_cmap("RdBu")(normalized) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(raster), 222, 0).astype(np.uint8)
    image = Image.fromarray(np.flipud(rgba), mode="RGBA")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _raw_rasters(path: str) -> dict[str, Any]:
    columns = list(dict.fromkeys(["geometry", *_RASTER_FIELDS.values(),
                                  *(field for pair in _DELTA_RASTER_FIELDS.values() for field in pair)]))
    frame = gpd.read_parquet(path, columns=columns)
    layout = _grid_raster_layout(frame)
    composites: dict[str, np.ndarray] = {}
    for key, column in _RASTER_FIELDS.items():
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        composites[key] = _maximum_composite(values, layout.rows, layout.cols, layout.shape)
    for key, (inmap_column, enhanced_column) in _DELTA_RASTER_FIELDS.items():
        inmap = pd.to_numeric(frame[inmap_column], errors="coerce").to_numpy(dtype=np.float64)
        enhanced = pd.to_numeric(frame[enhanced_column], errors="coerce").to_numpy(dtype=np.float64)
        composites[key] = _absolute_maximum_composite(
            inmap - enhanced, layout.rows, layout.cols, layout.shape
        )
    caps = {}
    for left, right in (("impm", "aepm"), ("imbc", "fbc"), ("imno2", "fno2")):
        quantiles = []
        for key in (left, right):
            positive = composites[key][np.isfinite(composites[key]) & (composites[key] > 0)]
            quantiles.append(float(np.quantile(positive, 0.995)) if positive.size else 0.0)
        caps[left] = caps[right] = max(quantiles)
    for key in _DELTA_RASTER_FIELDS:
        finite = np.abs(composites[key][np.isfinite(composites[key])])
        caps[key] = float(np.quantile(finite, 0.995)) if finite.size else 0.0
    transformer = Transformer.from_crs(frame.crs, 4326, always_xy=True)
    xmin, xmax, ymin, ymax = layout.extent
    corners = [transformer.transform(x, y) for x, y in ((xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax))]
    lons, lats = zip(*corners)
    first = next(iter(composites.values()))
    return {
        "bounds": [_number(min(lons), 5), _number(min(lats), 5), _number(max(lons), 5), _number(max(lats), 5)],
        "width": int(first.shape[1]),
        "height": int(first.shape[0]),
        "sourceCellMetres": 100,
        "composite": "maximum",
        "displayCapQuantile": 0.995,
        "scaleMode": "shared within pollutant",
        "caps": {key: _number(value) for key, value in caps.items()},
        "positivePixels": {key: int(np.count_nonzero(np.isfinite(value) & (value > 0))) for key, value in composites.items()},
        "rasters": {
            key: (_signed_png_data_url(value, caps[key]) if key in _DELTA_RASTER_FIELDS else _png_data_url(value, caps[key]))
            for key, value in composites.items()
        },
    }


def run(*, concentration_path: str, tracts_path: str, schools_path: str, ltc_path: str,
        output_dir: Path, power_plants_path: str | None = None) -> str:
    """Write a self-contained copy of the original interactive exposure map."""
    if not _TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Human-exposure web-map template not found: {_TEMPLATE_PATH}")
    logger.info("Generating original-style human-exposure web map")
    data = {
        "tracts": _tract_data(tracts_path),
        "schools": _facility_data(schools_path, name_field="SchoolName", weight_field="Enrollment"),
        "ltc": _facility_data(ltc_path, name_field="FacilityName", weight_field="TotalLicensedBeds"),
    }
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    template = _replace_json_constant(template, "DATA", data)
    template = _replace_json_constant(template, "RAW", _raw_rasters(concentration_path))
    template = _replace_json_constant(template, "POWER_PLANTS", _power_plant_data(power_plants_path))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "BayArea_Interactive_Direct_Final_PM25_BC_NO2_Power.html"
    output_path.write_text(_wrap_original_artifact(template), encoding="utf-8")
    logger.info("Human-exposure web map written to %s", output_path)
    return str(output_path)
