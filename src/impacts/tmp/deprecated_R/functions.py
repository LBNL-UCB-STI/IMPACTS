import datetime as dt
import os
import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError as exc:  # pragma: no cover - optional dependency
    gpd = None
    box = None
    _GEO_IMPORT_ERROR = exc
else:
    _GEO_IMPORT_ERROR = None

try:
    import pyproj
except ImportError as exc:  # pragma: no cover - optional dependency
    pyproj = None
    _PROJ_IMPORT_ERROR = exc
else:
    _PROJ_IMPORT_ERROR = None


proj_baaqmd = "+proj=lcc +lat_1=60 +lat_2=30 +lon_0=-120.5 +lat_0=37 +datum=NAD83"
Ncol = 164
Nrow = 224
x0 = -220000
y0 = -16000
dx = 1000
dy = 1000
dt_seconds = 3600
x1 = x0 + Ncol * dx
y1 = y0 + Nrow * dy


def strlen_fit(value: str, length: int = 16) -> str:
    if len(value) < length:
        return f"{value:<{length}}"
    return value[:length]


def CR_XY(col: Iterable[float], row: Iterable[float]) -> pd.DataFrame:
    col_arr = np.asarray(col, dtype=float)
    row_arr = np.asarray(row, dtype=float)
    x = (col_arr - 1) * dx + x0 + dx / 2
    y = (row_arr - 1) * dy + y0 + dy / 2
    return pd.DataFrame({"x": x, "y": y})


def XY_CR(x: Iterable[float], y: Iterable[float]) -> pd.DataFrame:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    col = (x_arr - x0) / dx
    row = (y_arr - y0) / dy
    return pd.DataFrame({"COL": col, "ROW": row})


def _ensure_proj():
    if pyproj is None:
        raise ImportError("pyproj is required for coordinate transforms") from _PROJ_IMPORT_ERROR


def LCCXY_LonLat(lccx: Iterable[float], lccy: Iterable[float]) -> pd.DataFrame:
    _ensure_proj()
    transformer = pyproj.Transformer.from_crs(proj_baaqmd, "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(lccx, lccy)
    return pd.DataFrame({"lon": lon, "lat": lat})


def LonLat_LCCXY(lon: Iterable[float], lat: Iterable[float]) -> pd.DataFrame:
    _ensure_proj()
    transformer = pyproj.Transformer.from_crs("EPSG:4326", proj_baaqmd, always_xy=True)
    lccx, lccy = transformer.transform(lon, lat)
    return pd.DataFrame({"lccx": lccx, "lccy": lccy})


def get_wknd_dates(year: int) -> Iterable[dt.date]:
    start = dt.date(year, 1, 1)
    days = 365 if year % 4 else 366
    dates = [start + dt.timedelta(days=offset) for offset in range(days)]
    return [date for date in dates if date.weekday() >= 5]


def clearspace():
    return None


def _ensure_geopandas():
    if gpd is None:
        raise ImportError("geopandas is required for spatial operations") from _GEO_IMPORT_ERROR


def _load_counties_from_path(path: str) -> "gpd.GeoDataFrame":
    _ensure_geopandas()
    return gpd.read_file(path)


def _filter_bay_area_counties(counties: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
    county_names = {
        "alameda",
        "contra costa",
        "marin",
        "napa",
        "san francisco",
        "san mateo",
        "santa clara",
        "solano",
        "sonoma",
    }

    if "region" in counties.columns and "subregion" in counties.columns:
        filtered = counties[
            (counties["region"].str.lower() == "california")
            & (counties["subregion"].str.lower().isin(county_names))
        ]
        return filtered

    if "STATE_NAME" in counties.columns and "NAME" in counties.columns:
        filtered = counties[
            (counties["STATE_NAME"].str.lower() == "california")
            & (counties["NAME"].str.lower().isin(county_names))
        ]
        return filtered

    raise ValueError("County dataset missing expected columns for filtering")


def get_bounding_box() -> "gpd.GeoSeries":
    _ensure_geopandas()
    if box is None:
        raise ImportError("shapely is required for bounding box creation") from _GEO_IMPORT_ERROR

    bbox_env = os.getenv("IMPACTS_BOUNDING_BOX")
    if bbox_env:
        parts = [float(part) for part in bbox_env.split(",")]
        if len(parts) != 4:
            raise ValueError("IMPACTS_BOUNDING_BOX must be minx,miny,maxx,maxy")
        minx, miny, maxx, maxy = parts
        return gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326")

    counties_path = os.getenv("IMPACTS_COUNTIES_PATH")
    if not counties_path:
        raise ValueError("Set IMPACTS_BOUNDING_BOX or IMPACTS_COUNTIES_PATH to compute bounding_box")

    counties = _load_counties_from_path(counties_path)
    filtered = _filter_bay_area_counties(counties)
    filtered = filtered.to_crs("EPSG:4326")
    minx, miny, maxx, maxy = filtered.total_bounds
    minx -= 0.02
    miny -= 0.02
    maxx += 0.02
    maxy += 0.02
    return gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326")


def map_to_grid(
    output_dir: str,
    grid_source_grid: str,
    grid: "gpd.GeoDataFrame",
    bounding_box: "gpd.GeoSeries",
) -> Optional[pd.DataFrame]:
    _ensure_geopandas()
    start = time.time()

    shp_path = os.path.join(output_dir, f"grid_{grid_source_grid}_geopoint.shp")
    dat = gpd.read_file(shp_path)
    if dat.empty:
        return None

    pdat = dat.to_crs("EPSG:4326")
    pgrid = grid.to_crs("EPSG:4326")

    sf_grid = gpd.clip(pgrid, bounding_box)
    sf_dat = gpd.clip(pdat, bounding_box)
    if sf_grid.empty or sf_dat.empty:
        return None

    psf_grid = sf_grid.to_crs(grid.crs)
    psf_dat = sf_dat.copy()
    psf_dat["gobid"] = np.arange(1, len(psf_dat) + 1)
    psf_dat = psf_dat.to_crs(grid.crs)

    kk = gpd.overlay(psf_dat, psf_grid, how="intersection")
    if kk.empty:
        return None

    kk = kk[kk.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if kk.empty:
        return None

    kk["area"] = kk.geometry.area

    grouped = (
        kk.drop(columns="geometry")
        .groupby("grid")
        .apply(lambda frame: np.sum(frame["NOx"] * frame["area"]) / np.sum(frame["area"]))
        .reset_index(name="NOx")
    )
    grouped = grouped.sort_values("grid")

    res = pd.DataFrame([grouped["NOx"].to_numpy()], columns=grouped["grid"].to_numpy())
    res.index = [grid_source_grid]

    _ = time.time() - start
    return res


def read_rdata(path: str) -> Dict[str, object]:
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyreadr is required to read .RData files") from exc
    return pyreadr.read_r(path)


def write_rdata(path: str, objects: Dict[str, object]) -> None:
    try:
        import pyreadr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("pyreadr is required to write .RData files") from exc
    pyreadr.write_rdata(path, objects)
