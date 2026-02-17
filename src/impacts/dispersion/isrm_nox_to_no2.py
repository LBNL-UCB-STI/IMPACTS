"""GRID NOx-to-NO2 pipeline.

Converts InMAP NOx source-receptor matrix outputs into a NOx-to-NO2 GRID
for the SF Bay Area, using CMAQ NO2/NOx ratios as a scaling factor.

Steps:
  0. convert_cmaq_polygon  - Turn CMAQ NO2/NOx ratio into GRID-gridded polygons
  1. generate_xwalk        - Read InMAP NOx output and build GOB-to-GRID crosswalk
  2. cmaq_ratio_to_grid    - Map CMAQ NO2/NOx ratio onto GRID grid
  3. nox_to_no2_grid       - Convert NOx-to-NOx GRID into NOx-to-NO2 GRID
"""

import os
import re
import time
from multiprocessing import Pool
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError as exc:
    raise ImportError("geopandas and shapely are required to run this pipeline") from exc


INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "output")


# ---------------------------------------------------------------------------
# BAAQMD / CMAQ grid constants
# ---------------------------------------------------------------------------

PROJ_BAAQMD = "+proj=lcc +lat_1=60 +lat_2=30 +lon_0=-120.5 +lat_0=37 +datum=NAD83"
NCOL = 164
NROW = 224
X0 = -220000
Y0 = -16000
DX = 1000  # 1km grid resolution
DY = 1000

SF_BAY_COUNTIES = {
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


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_rdata(path: str) -> Dict[str, object]:
    """Read an .RData file and return a dict of objects (for legacy input data)."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("pyreadr is required to read .RData files") from exc
    return pyreadr.read_r(path)


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def cr_xy(col: Iterable[float], row: Iterable[float]) -> pd.DataFrame:
    """Convert BAAQMD grid col/row to projected LCC x/y (cell centers)."""
    col_arr = np.asarray(col, dtype=float)
    row_arr = np.asarray(row, dtype=float)
    x = (col_arr - 1) * DX + X0 + DX / 2
    y = (row_arr - 1) * DY + Y0 + DY / 2
    return pd.DataFrame({"x": x, "y": y})


def grid_polygons_from_centers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create 1km x 1km grid polygons from center-point coordinates."""
    half_dx = DX / 2
    half_dy = DY / 2
    geometry = [
        box(x - half_dx, y - half_dy, x + half_dx, y + half_dy)
        for x, y in zip(df["x"], df["y"])
    ]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=PROJ_BAAQMD)


# ---------------------------------------------------------------------------
# Bounding box
# ---------------------------------------------------------------------------

def _filter_bay_area_counties(counties: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter a counties GeoDataFrame to the 9 SF Bay Area counties."""
    if "region" in counties.columns and "subregion" in counties.columns:
        return counties[
            (counties["region"].str.lower() == "california")
            & (counties["subregion"].str.lower().isin(SF_BAY_COUNTIES))
        ]
    if "STATE_NAME" in counties.columns and "NAME" in counties.columns:
        return counties[
            (counties["STATE_NAME"].str.lower() == "california")
            & (counties["NAME"].str.lower().isin(SF_BAY_COUNTIES))
        ]
    raise ValueError("County dataset missing expected columns for filtering")


def get_bounding_box() -> gpd.GeoSeries:
    """Return SF Bay Area bounding box from env var or counties shapefile."""
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

    counties = gpd.read_file(counties_path)
    filtered = _filter_bay_area_counties(counties).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = filtered.total_bounds
    minx -= 0.02
    miny -= 0.02
    maxx += 0.02
    maxy += 0.02
    return gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326")


# ---------------------------------------------------------------------------
# GOB-to-GRID mapping (used by step 1)
# ---------------------------------------------------------------------------

def map_to_grid(
    output_dir: str,
    grid_source_grid: str,
    grid: gpd.GeoDataFrame,
    bounding_box: gpd.GeoSeries,
) -> Optional[pd.DataFrame]:
    """Map a single InMAP GOB output to GRID grid via area-weighted averaging."""
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


# ---------------------------------------------------------------------------
# Step 0: Convert CMAQ polygon
# ---------------------------------------------------------------------------

def convert_cmaq_polygon(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Load CMAQ NO2/NOx ratio and write BAAQMD grid as a GeoJSON file."""
    rdata = read_rdata(os.path.join(input_dir, "cmaqtestNO2_ratio.RData"))
    if "pdat" not in rdata:
        raise KeyError("Expected 'pdat' in cmaqtestNO2_ratio.RData")
    pdat = rdata["pdat"].copy()

    coords = cr_xy(pdat["col"], pdat["row"])
    pdat[["x", "y"]] = coords

    gdf = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]])
    gdf = gdf.drop(columns=["value"])

    gdf.to_file(os.path.join(output_dir, "baaqmd.geojson"), driver="GeoJSON")


# ---------------------------------------------------------------------------
# Step 1: Generate crosswalk GOB <-> GRID (parallel)
# ---------------------------------------------------------------------------

_GRID = None
_BOUNDING_BOX = None
_DATDIR = None


def _init_worker(grid_path: str, bbox_wkb: bytes, datdir: str) -> None:
    """Initializer for each multiprocessing worker."""
    global _GRID, _BOUNDING_BOX, _DATDIR
    _GRID = gpd.read_file(grid_path)
    _BOUNDING_BOX = gpd.GeoSeries.from_wkb([bbox_wkb], crs="EPSG:4326")
    _DATDIR = datdir


def _map_one(grid_source_grid: str) -> Optional[pd.DataFrame]:
    """Map a single GRID source grid (worker function)."""
    return map_to_grid(_DATDIR, grid_source_grid, _GRID, _BOUNDING_BOX)


def _extract_ids(files: Iterable[str]) -> List[str]:
    """Extract sorted unique GRID IDs from filenames like grid_00843_*."""
    ids = []
    for name in files:
        match = re.match(r"^grid_(\d+)", name)
        if match:
            ids.append(match.group(1))
    return sorted(set(ids))


def generate_xwalk(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR, n_workers: int = 6) -> None:
    """Build NOx-to-NOx GRID crosswalk from InMAP geopoint outputs."""
    datdir = os.path.join(input_dir, "sfbay_grid_geopoints_inmap_1.9.6")
    grid_path = os.path.join(input_dir, "grid_polygon", "grid_polygon.shp")
    bounding_box = get_bounding_box()

    idlist = _extract_ids(os.listdir(datdir))
    bbox_wkb = bounding_box.iloc[0].wkb

    results = []
    with Pool(processes=n_workers, initializer=_init_worker,
              initargs=(grid_path, bbox_wkb, datdir)) as pool:
        for res in pool.map(_map_one, idlist):
            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError("No results produced from map_to_grid")

    res_df = pd.concat(results)
    res_df.to_parquet(os.path.join(output_dir, "nox_nox_grid.parquet"))


# ---------------------------------------------------------------------------
# Step 2: CMAQ ratio to GRID grid
# ---------------------------------------------------------------------------

def cmaq_ratio_to_grid(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Intersect CMAQ NO2/NOx ratio grid with GRID and compute area-weighted averages."""
    grid = gpd.read_file(os.path.join(input_dir, "grid_polygon", "grid_polygon.shp"))

    rdata = read_rdata(os.path.join(input_dir, "cmaqtestNO2_ratio.RData"))
    if "pdat" not in rdata:
        raise KeyError("Expected 'pdat' in cmaqtestNO2_ratio.RData")
    pdat = rdata["pdat"].copy()

    pdat["col"] = pdat["x"]
    pdat["row"] = pdat["y"]
    coords = cr_xy(pdat["col"], pdat["row"])
    pdat[["x", "y"]] = coords

    mm = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]])
    mm = mm.drop(columns=["value"])
    mm.to_file(os.path.join(output_dir, "baaqmd.geojson"), driver="GeoJSON")

    mm = mm.to_crs(grid.crs)
    kk = gpd.overlay(mm, grid, how="intersection")
    kk["area"] = kk.geometry.area

    merged = kk.merge(pdat, on=["col", "row"], how="left")
    grouped = (
        merged.drop(columns="geometry")
        .groupby("grid")
        .apply(lambda frame: (frame["value"] * frame["area"]).sum() / frame["area"].sum())
        .reset_index(name="value")
    )

    no2ratio = grouped.rename(columns={"value": "NO2_NOx_ratio"})
    no2ratio.to_parquet(os.path.join(output_dir, "no2_nox_ratio_grid.parquet"), index=False)


# ---------------------------------------------------------------------------
# Step 3: NOx to NO2 GRID
# ---------------------------------------------------------------------------

def nox_to_no2_grid(output_dir: str = OUTPUT_DIR) -> None:
    """Multiply NOx GRID by NO2/NOx ratio to produce NOx-to-NO2 GRID."""
    res = pd.read_parquet(os.path.join(output_dir, "nox_nox_grid.parquet"))

    no2ratio = pd.read_parquet(os.path.join(output_dir, "no2_nox_ratio_grid.parquet"))

    res.index = pd.to_numeric(res.index, errors="coerce").astype(int)
    res.columns = [int(col) for col in res.columns]

    s_grid = res.index.to_numpy()
    s_grid = s_grid[(s_grid > 3) & (s_grid != 3554)]
    res = res.loc[s_grid, s_grid]

    if 843 not in no2ratio["grid"].astype(int).to_numpy():
        extra = pd.DataFrame({"grid": [843], "NO2_NOx_ratio": [0.94]})
        no2ratio = pd.concat([no2ratio, extra], ignore_index=True)

    dat = res.transpose()
    dat["grid"] = dat.index.astype(int)
    dat = dat.merge(no2ratio, on="grid", how="left")

    cols = [col for col in dat.columns if isinstance(col, int) and 843 <= col <= 3706]
    dat[cols] = dat[cols].multiply(dat["NO2_NOx_ratio"], axis=0)

    res_dat = dat[cols].transpose()
    res_dat.columns = dat["grid"].to_numpy()

    res_dat.to_parquet(os.path.join(output_dir, "nox_to_no2_grid.parquet"))


# ---------------------------------------------------------------------------
# Step registry and main
# ---------------------------------------------------------------------------

STEPS = {
    "convert_cmaq_polygon": lambda i, o: convert_cmaq_polygon(i, o),
    "generate_xwalk": lambda i, o: generate_xwalk(i, o),
    "cmaq_ratio_to_grid": lambda i, o: cmaq_ratio_to_grid(i, o),
    "nox_to_no2_grid": lambda i, o: nox_to_no2_grid(o),
}

DEFAULT_STEP_ORDER = list(STEPS.keys())


def run_pipeline(
    steps: Optional[List[str]] = None,
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Run the specified steps (or all steps) in order."""
    if steps is None:
        steps = DEFAULT_STEP_ORDER
    for step_name in steps:
        print(f"Running step: {step_name}")
        STEPS[step_name](input_dir, output_dir)
        print(f"Completed step: {step_name}")


if __name__ == "__main__":
    run_pipeline()