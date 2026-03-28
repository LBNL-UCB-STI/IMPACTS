"""One-off GRID NOx-to-NO2 preprocessing pipeline.

Converts InMAP NOx source-receptor matrix outputs into a NOx-to-NO2 GRID
for the SF Bay Area, using CMAQ NO2/NOx ratios as a scaling factor.

Steps:
  0. convert_cmaq_polygon  - Turn CMAQ NO2/NOx ratio into GRID-gridded polygons
  1. generate_xwalk        - Read InMAP NOx output and build GOB-to-GRID crosswalk
  2. cmaq_ratio_to_grid    - Map CMAQ NO2/NOx ratio onto GRID grid
  3. nox_to_no2_grid       - Convert NOx-to-NOx GRID into NOx-to-NO2 GRID
"""

from __future__ import annotations

import argparse
import os
import re
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError as exc:
    raise ImportError("geopandas and shapely are required to run this pipeline") from exc

from .rdata_util import rdata_to_parquet


INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "output")
REGIONAL_NOX_NOX_MATRIX_NAME = "nox_nox_regional_grid_matrix.parquet"
REGIONAL_NO2_RATIO_NAME = "no2_nox_regional_grid_ratio.parquet"
REGIONAL_NOX_TO_NO2_MATRIX_NAME = "nox_to_no2_regional_grid_matrix.parquet"


PROJ_BAAQMD = "+proj=lcc +lat_1=60 +lat_2=30 +lon_0=-120.5 +lat_0=37 +datum=NAD83"
X0 = -220000
Y0 = -16000
DX = 1000
DY = 1000

def _prepare_rdata_inputs(input_dir: str) -> Path:
    """Convert known RData inputs to parquet and return the preferred CMAQ ratio path."""
    input_root = Path(input_dir).resolve()
    parquet_candidates = sorted(input_root.glob("cmaqtestNO2_ratio*.parquet"))
    if parquet_candidates:
        return parquet_candidates[0]

    rdata_path = input_root / "cmaqtestNO2_ratio.RData"
    if not rdata_path.exists():
        raise FileNotFoundError(
            f"Expected either {rdata_path.name} or a converted cmaqtestNO2_ratio parquet in {input_root}"
        )

    written = rdata_to_parquet(rdata_path, input_root)
    parquet_candidates = [path for path in written if path.suffix == ".parquet"]
    if not parquet_candidates:
        raise ValueError(f"Converting {rdata_path} produced no parquet outputs")

    pdat_candidates = [path for path in parquet_candidates if path.stem.endswith("__pdat") or path.stem == "cmaqtestNO2_ratio"]
    return pdat_candidates[0] if pdat_candidates else parquet_candidates[0]


def _load_cmaq_ratio_table(input_dir: str) -> pd.DataFrame:
    """Load the CMAQ ratio table, converting the source RData once if needed."""
    ratio_path = _prepare_rdata_inputs(input_dir)
    table = pd.read_parquet(ratio_path)
    if "col" not in table.columns and "x" in table.columns:
        table["col"] = table["x"]
    if "row" not in table.columns and "y" in table.columns:
        table["row"] = table["y"]

    required = {"col", "row", "value"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"CMAQ ratio table {ratio_path} is missing required columns {sorted(missing)}; "
            f"available columns: {list(table.columns)}"
    )
    return table


def _read_rdata_frame(path: str | Path, object_name: str | None = None) -> pd.DataFrame:
    """Read a named data frame-like object from an RData file."""
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("pyreadr is required to read .RData files") from exc

    result = pyreadr.read_r(str(path))
    if object_name is not None:
        if object_name not in result:
            raise KeyError(f"Expected '{object_name}' in {path}; found {list(result.keys())}")
        value = result[object_name]
    else:
        value = next(iter(result.values()))

    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"Expected a data frame in {path}, got {type(value)}")
    return value.copy()


def _normalize_square_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a source-receptor matrix to integer row and column ids."""
    matrix = df.copy()
    if matrix.shape[1] == 1 and matrix.columns.tolist() == [None]:
        raise ValueError("Legacy NOx matrix could not be parsed as a square matrix")

    matrix.index = pd.to_numeric(pd.Index(matrix.index), errors="coerce")
    matrix.columns = pd.to_numeric(pd.Index(matrix.columns), errors="coerce")
    matrix = matrix.loc[matrix.index.notna(), matrix.columns.notna()].copy()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _load_legacy_no2_ratio(input_dir: str) -> pd.DataFrame:
    """Load a precomputed NO2/NOx ratio table from legacy inputs when present."""
    legacy_path = Path(input_dir).resolve() / "sfb.no2ratio_isrmGRID.RData"
    if not legacy_path.exists():
        raise FileNotFoundError(legacy_path)

    table = _read_rdata_frame(legacy_path, "no2ratio")
    if "grid" not in table.columns and "isrm" in table.columns:
        table["grid"] = table["isrm"]
    required = {"grid", "NO2_NOx_ratio"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Legacy NO2 ratio table {legacy_path} is missing required columns {sorted(missing)}; "
            f"available columns: {list(table.columns)}"
        )
    table = table[["grid", "NO2_NOx_ratio"]].copy()
    table["grid"] = pd.to_numeric(table["grid"], errors="coerce").astype("Int64")
    table["NO2_NOx_ratio"] = pd.to_numeric(table["NO2_NOx_ratio"], errors="coerce")
    return table.dropna(subset=["grid", "NO2_NOx_ratio"]).assign(grid=lambda df: df["grid"].astype(int))


def _load_precomputed_no2_matrix(input_dir: str) -> pd.DataFrame | None:
    """Load a direct precomputed NOx-to-NO2 matrix when available."""
    input_root = Path(input_dir).resolve()
    parquet_path = input_root / REGIONAL_NOX_TO_NO2_MATRIX_NAME
    if parquet_path.exists():
        return _normalize_square_matrix(pd.read_parquet(parquet_path))

    legacy_rdata = input_root / "NOx_to_NO2_ISRM.RData"
    if legacy_rdata.exists():
        return _normalize_square_matrix(_read_rdata_frame(legacy_rdata, "res.dat"))

    return None


def cr_xy(col: Iterable[float], row: Iterable[float]) -> pd.DataFrame:
    """Convert BAAQMD grid col/row to projected LCC x/y cell centers."""
    col_arr = np.asarray(col, dtype=float)
    row_arr = np.asarray(row, dtype=float)
    x = (col_arr - 1) * DX + X0 + DX / 2
    y = (row_arr - 1) * DY + Y0 + DY / 2
    return pd.DataFrame({"x": x, "y": y})


def grid_polygons_from_centers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Create 1 km grid polygons from center-point coordinates."""
    half_dx = DX / 2
    half_dy = DY / 2
    geometry = [box(x - half_dx, y - half_dy, x + half_dx, y + half_dy) for x, y in zip(df["x"], df["y"])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=PROJ_BAAQMD)


def get_bounding_box(grid_path: str | Path) -> gpd.GeoSeries:
    """Return a padded bounding box derived from the provided grid geometry."""
    grid = gpd.read_file(grid_path).to_crs("EPSG:4326")
    minx, miny, maxx, maxy = grid.total_bounds
    minx -= 0.02
    miny -= 0.02
    maxx += 0.02
    maxy += 0.02
    return gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs="EPSG:4326")


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


def convert_cmaq_polygon(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Load CMAQ NO2/NOx ratio and write BAAQMD grid as a GeoJSON file."""
    pdat = _load_cmaq_ratio_table(input_dir).copy()

    coords = cr_xy(pdat["col"], pdat["row"])
    pdat[["x", "y"]] = coords

    gdf = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]])
    gdf = gdf.drop(columns=["value"])
    gdf.to_file(os.path.join(output_dir, "baaqmd.geojson"), driver="GeoJSON")


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
    """Map a single GRID source grid worker task."""
    return map_to_grid(_DATDIR, grid_source_grid, _GRID, _BOUNDING_BOX)


def _extract_ids(files: Iterable[str]) -> List[str]:
    """Extract sorted unique GRID ids from filenames like grid_00843_*."""
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
    if not os.path.isdir(datdir) or not os.path.exists(grid_path):
        precomputed_no2 = _load_precomputed_no2_matrix(input_dir)
        if precomputed_no2 is not None:
            tqdm.write("Skipping generate_xwalk: direct NOx-to-NO2 matrix found in input-dir")
            return
        raise FileNotFoundError(
            "generate_xwalk requires both sfbay_grid_geopoints_inmap_1.9.6/ and grid_polygon/grid_polygon.shp "
            f"under {input_dir}"
        )

    bounding_box = get_bounding_box(grid_path)

    idlist = _extract_ids(os.listdir(datdir))
    bbox_wkb = bounding_box.iloc[0].wkb

    results = []
    with Pool(processes=n_workers, initializer=_init_worker, initargs=(grid_path, bbox_wkb, datdir)) as pool:
        for res in pool.map(_map_one, idlist):
            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError("No results produced from map_to_grid")

    res_df = pd.concat(results)
    res_df.to_parquet(os.path.join(output_dir, REGIONAL_NOX_NOX_MATRIX_NAME))


def cmaq_ratio_to_grid(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Intersect CMAQ NO2/NOx ratio grid with GRID and compute area-weighted averages."""
    grid_path = Path(input_dir).resolve() / "grid_polygon" / "grid_polygon.shp"
    legacy_ratio_path = Path(input_dir).resolve() / "sfb.no2ratio_isrmGRID.RData"
    if not grid_path.exists() and legacy_ratio_path.exists():
        _load_legacy_no2_ratio(input_dir).to_parquet(
            os.path.join(output_dir, REGIONAL_NO2_RATIO_NAME),
            index=False,
        )
        return

    grid = gpd.read_file(grid_path)

    pdat = _load_cmaq_ratio_table(input_dir).copy()

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
    no2ratio.to_parquet(os.path.join(output_dir, REGIONAL_NO2_RATIO_NAME), index=False)


def nox_to_no2_grid(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    """Multiply NOx GRID by NO2/NOx ratio to produce NOx-to-NO2 GRID."""
    direct_matrix = _load_precomputed_no2_matrix(input_dir)
    if direct_matrix is not None:
        direct_matrix.to_parquet(os.path.join(output_dir, REGIONAL_NOX_TO_NO2_MATRIX_NAME))
        return

    res = pd.read_parquet(os.path.join(output_dir, REGIONAL_NOX_NOX_MATRIX_NAME))
    no2ratio = pd.read_parquet(os.path.join(output_dir, REGIONAL_NO2_RATIO_NAME))

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
    res_dat.to_parquet(os.path.join(output_dir, REGIONAL_NOX_TO_NO2_MATRIX_NAME))


STEPS = {
    "convert_cmaq_polygon": lambda i, o: convert_cmaq_polygon(i, o),
    "generate_xwalk": lambda i, o: generate_xwalk(i, o),
    "cmaq_ratio_to_grid": lambda i, o: cmaq_ratio_to_grid(i, o),
    "nox_to_no2_grid": lambda i, o: nox_to_no2_grid(i, o),
}

DEFAULT_STEP_ORDER = list(STEPS.keys())


def run_pipeline(
    steps: Optional[List[str]] = None,
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Run the specified preprocessing steps in order."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if steps is None:
        steps = DEFAULT_STEP_ORDER
    progress = tqdm(steps, desc="NOx->NO2 preprocessing", unit="step", dynamic_ncols=True)
    for step_name in progress:
        progress.set_postfix_str(step_name)
        STEPS[step_name](input_dir, output_dir)
    progress.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.utils.isrm_nox_to_no2",
        description="Run the one-off NOx-to-NO2 preprocessing utility.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing GRID, geopoints, and CMAQ inputs")
    parser.add_argument("--output-dir", required=True, help="Directory where generated artifacts will be written")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    return 0


__all__ = [
    "DEFAULT_STEP_ORDER",
    "STEPS",
    "build_parser",
    "cmaq_ratio_to_grid",
    "convert_cmaq_polygon",
    "generate_xwalk",
    "get_bounding_box",
    "main",
    "map_to_grid",
    "nox_to_no2_grid",
    "run_pipeline",
]


if __name__ == "__main__":
    raise SystemExit(main())
