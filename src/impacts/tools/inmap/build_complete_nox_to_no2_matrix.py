"""Build a workflow-ready full-domain NOx-to-NO2 matrix.

This utility:
1. Runs the one-off NOx/NO2 preprocessing pipeline.
2. Reads the generated sparse/regional ``nox_to_no2_regional_grid_matrix.parquet``.
3. Writes a sparse full-domain NOx->NO2 artifact using the ISRM source/receptor shape.

The result is a compressed ``.npz`` file with:
- ``source_ids``: sparse source row ids
- ``receptor_ids``: sparse receptor column ids
- ``values``: NOx->NO2 coefficients
- ``source_dim`` / ``receptor_dim``: full ISRM shape metadata
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import time
from multiprocessing import Pool
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box
from tqdm import tqdm

from .rdata_conversion import rdata_to_parquet


DEFAULT_OUTPUT_NAME = "nox_to_no2_full_isrm_matrix.npz"
DEFAULT_WRITE_CHUNK_ROWS = 256
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


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(name).strip())
    return cleaned or "object"


def _read_rdata_frame(path: str | Path, object_name: str | None = None) -> pd.DataFrame:
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
    matrix = df.copy()
    if matrix.shape[1] == 1 and matrix.columns.tolist() == [None]:
        raise ValueError("Legacy NOx matrix could not be parsed as a square matrix")

    matrix.index = pd.to_numeric(pd.Index(matrix.index), errors="coerce")
    matrix.columns = pd.to_numeric(pd.Index(matrix.columns), errors="coerce")
    matrix = matrix.loc[matrix.index.notna(), matrix.columns.notna()].copy()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _prepare_rdata_inputs(input_dir: str) -> Path:
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

    pdat_candidates = [
        path for path in parquet_candidates if path.stem.endswith("__pdat") or path.stem == "cmaqtestNO2_ratio"
    ]
    return pdat_candidates[0] if pdat_candidates else parquet_candidates[0]


def _load_cmaq_ratio_table(input_dir: str) -> pd.DataFrame:
    ratio_path = _prepare_rdata_inputs(input_dir)
    table = pd.read_parquet(ratio_path)
    required = {"col", "row", "value"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"CMAQ ratio table {ratio_path} is missing required columns {sorted(missing)}; "
            f"available columns: {list(table.columns)}"
        )
    return table

def _load_precomputed_no2_matrix(input_dir: str) -> pd.DataFrame | None:
    input_root = Path(input_dir).resolve()
    parquet_path = input_root / REGIONAL_NOX_TO_NO2_MATRIX_NAME
    if parquet_path.exists():
        return _normalize_square_matrix(pd.read_parquet(parquet_path))
    return None


def cr_xy(col: Iterable[float], row: Iterable[float]) -> pd.DataFrame:
    col_arr = np.asarray(col, dtype=float)
    row_arr = np.asarray(row, dtype=float)
    x = (col_arr - 1) * DX + X0 + DX / 2
    y = (row_arr - 1) * DY + Y0 + DY / 2
    return pd.DataFrame({"x": x, "y": y})


def grid_polygons_from_centers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    half_dx = DX / 2
    half_dy = DY / 2
    geometry = [box(x - half_dx, y - half_dy, x + half_dx, y + half_dy) for x, y in zip(df["x"], df["y"])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=PROJ_BAAQMD)


def get_bounding_box(grid_path: str | Path) -> gpd.GeoSeries:
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


_GRID = None
_BOUNDING_BOX = None
_DATDIR = None


def _init_worker(grid_path: str, bbox_wkb: bytes, datdir: str) -> None:
    global _GRID, _BOUNDING_BOX, _DATDIR
    _GRID = gpd.read_file(grid_path)
    _BOUNDING_BOX = gpd.GeoSeries.from_wkb([bbox_wkb], crs="EPSG:4326")
    _DATDIR = datdir


def _map_one(grid_source_grid: str) -> Optional[pd.DataFrame]:
    return map_to_grid(_DATDIR, grid_source_grid, _GRID, _BOUNDING_BOX)


def _extract_ids(files: Iterable[str]) -> List[str]:
    ids = []
    for name in files:
        match = re.match(r"^grid_(\d+)", name)
        if match:
            ids.append(match.group(1))
    return sorted(set(ids))


def convert_cmaq_polygon(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    pdat = _load_cmaq_ratio_table(input_dir).copy()
    coords = cr_xy(pdat["col"], pdat["row"])
    pdat[["x", "y"]] = coords
    gdf = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]])
    gdf = gdf.drop(columns=["value"])
    gdf.to_file(os.path.join(output_dir, "baaqmd.geojson"), driver="GeoJSON")


def generate_xwalk(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR, n_workers: int = 6) -> None:
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
    pd.concat(results).to_parquet(os.path.join(output_dir, REGIONAL_NOX_NOX_MATRIX_NAME))


def cmaq_ratio_to_grid(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
    grid_path = Path(input_dir).resolve() / "grid_polygon" / "grid_polygon.shp"
    grid = gpd.read_file(grid_path)
    pdat = _load_cmaq_ratio_table(input_dir).copy()
    coords = cr_xy(pdat["col"], pdat["row"])
    pdat[["x", "y"]] = coords
    mm = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]]).drop(columns=["value"])
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
    grouped["NO2_NOx_ratio"] = grouped["value"]
    grouped = grouped[["grid", "NO2_NOx_ratio"]]
    grouped.to_parquet(os.path.join(output_dir, REGIONAL_NO2_RATIO_NAME), index=False)


def nox_to_no2_grid(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR) -> None:
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
        no2ratio = pd.concat([no2ratio, pd.DataFrame({"grid": [843], "NO2_NOx_ratio": [0.94]})], ignore_index=True)
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
STEP_ORDER = list(STEPS.keys())


def run_pipeline(
    steps: Optional[List[str]] = None,
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    resolved_steps = steps or STEP_ORDER
    progress = tqdm(resolved_steps, desc="NOx->NO2 preprocessing", unit="step", dynamic_ncols=True)
    for step_name in progress:
        progress.set_postfix_str(step_name)
        STEPS[step_name](input_dir, output_dir)
    progress.close()


def _load_isrm_store(isrm_url: str):
    import zarr

    if isrm_url.startswith("s3://"):
        try:
            import s3fs  # noqa: F401
        except ImportError as exc:
            raise ImportError("s3fs is required to read ISRM zarr from s3:// URLs") from exc
        return zarr.open(
            isrm_url,
            mode="r",
            storage_options={"anon": True, "client_kwargs": {"region_name": "us-east-2"}},
        )

    return zarr.open(isrm_url, mode="r")


def _read_transfer_matrix(path: str | Path) -> pd.DataFrame:
    matrix = pd.read_parquet(path)
    matrix.index = pd.to_numeric(pd.Index(matrix.index), errors="coerce")
    matrix.columns = pd.to_numeric(pd.Index(matrix.columns), errors="coerce")
    matrix = matrix.loc[matrix.index.notna(), matrix.columns.notna()].copy()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _extract_sparse_triplets(
    matrix: pd.DataFrame,
    *,
    chunk_rows: int = DEFAULT_WRITE_CHUNK_ROWS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_chunks: list[np.ndarray] = []
    receptor_chunks: list[np.ndarray] = []
    value_chunks: list[np.ndarray] = []

    progress = tqdm(
        total=matrix.shape[0],
        desc="Extracting sparse NOx->NO2 matrix",
        unit="row",
        dynamic_ncols=True,
    )
    try:
        for start in range(0, matrix.shape[0], chunk_rows):
            stop = min(start + chunk_rows, matrix.shape[0])
            chunk = matrix.iloc[start:stop]
            values = chunk.to_numpy()
            row_idx, col_idx = np.nonzero(values)
            if row_idx.size:
                chunk_index = chunk.index.to_numpy(dtype=np.int64, copy=False)
                chunk_columns = chunk.columns.to_numpy(dtype=np.int64, copy=False)
                source_chunks.append(chunk_index[row_idx])
                receptor_chunks.append(chunk_columns[col_idx])
                value_chunks.append(values[row_idx, col_idx].astype(np.float64, copy=False))
            progress.update(stop - start)
    finally:
        progress.close()

    if not source_chunks:
        empty_i64 = np.array([], dtype=np.int64)
        empty_f64 = np.array([], dtype=np.float64)
        return empty_i64, empty_i64.copy(), empty_f64

    return (
        np.concatenate(source_chunks),
        np.concatenate(receptor_chunks),
        np.concatenate(value_chunks),
    )


def _write_sparse_npz(
    *,
    output_path: str | Path,
    source_ids: np.ndarray,
    receptor_ids: np.ndarray,
    values: np.ndarray,
    source_dim: int,
    receptor_dim: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        source_ids=source_ids.astype(np.int64, copy=False),
        receptor_ids=receptor_ids.astype(np.int64, copy=False),
        values=values.astype(np.float64, copy=False),
        source_dim=np.array(source_dim, dtype=np.int64),
        receptor_dim=np.array(receptor_dim, dtype=np.int64),
    )


def build_complete_matrix(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    isrm_zarr: str,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_dir=str(input_root), output_dir=str(output_root))

    sparse_matrix_path = output_root / REGIONAL_NOX_TO_NO2_MATRIX_NAME
    if not sparse_matrix_path.exists():
        raise FileNotFoundError(f"Expected {sparse_matrix_path} after running NOx/NO2 preprocessing")

    if not str(isrm_zarr).strip():
        raise ValueError("isrm_zarr must be provided explicitly. Use --isrm-zarr and point it to the ISRM zarr store.")
    sr = _load_isrm_store(str(isrm_zarr))
    source_dim = int(sr["SOA"].shape[1])
    receptor_dim = int(sr["SOA"].shape[2])

    sparse_matrix = _read_transfer_matrix(sparse_matrix_path)
    source_ids, receptor_ids, values = _extract_sparse_triplets(sparse_matrix)

    output_path = output_root / output_name
    _write_sparse_npz(
        output_path=output_path,
        source_ids=source_ids,
        receptor_ids=receptor_ids,
        values=values,
        source_dim=source_dim,
        receptor_dim=receptor_dim,
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.inmap.build_complete_nox_to_no2_matrix",
        description="Build a full-domain workflow-ready NOx-to-NO2 ISRM matrix.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing NOx/NO2 preprocessing inputs")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written")
    parser.add_argument(
        "--isrm-zarr",
        required=True,
        help="Local path or s3:// URL for the ISRM zarr store.",
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Filename for the sparse full-domain matrix. Default: {DEFAULT_OUTPUT_NAME}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = build_complete_matrix(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        isrm_zarr=args.isrm_zarr,
        output_name=args.output_name,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
