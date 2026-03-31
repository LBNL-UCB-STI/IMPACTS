"""Step 3 — AERMOD concentration workflow built from ASRV pattern kernels."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional
from typing import TypedDict

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_table
from ..common import read_vector
from ..config.defaults import annualization_days as default_annualization_days
from ..config.defaults import grams_per_short_ton
from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)

_DEFAULT_SITE = "LIVERMORE_2015"
_DEFAULT_URBAN_CLASS = 0
_DEFAULT_TEMPORAL = "CITYSTREET"
_DEFAULT_HEIGHT = 1.0
_SOURCE_POPULATION_COLUMN = "source_population"
_SOURCE_TEMPORAL_COLUMN = "source_temporal_class"
_SOURCE_HEIGHT_COLUMN = "source_release_height"

_KERNEL_RADIUS_METERS = 1000.0
_ASRV_SOURCE_RATE_G_PER_S_M2 = 0.1
_ASRV_ACTIVE_DAYS_PER_YEAR = default_annualization_days
_SOURCE_CHUNK_SIZE = 50_000


class _Kernel(TypedDict):
    dix: np.ndarray
    diy: np.ndarray
    response_per_ton: np.ndarray


def _step_label(step: str) -> str:
    return f"Step 3.{step}"


def _trace_frame(step: str, label: str, df: pd.DataFrame, *, key_cols: Optional[list[str]] = None) -> None:
    logger.info("%s trace %s shape=%s", _step_label(step), label, df.shape)
    preview = list(df.columns[:20])
    suffix = "" if len(df.columns) <= 20 else " ..."
    logger.info("%s trace %s columns(%d): %s%s", _step_label(step), label, len(df.columns), preview, suffix)
    if key_cols:
        present = [col for col in key_cols if col in df.columns]
        if present and not df.empty:
            logger.info(
                "%s trace %s sample_keys=%s",
                _step_label(step),
                label,
                df[present].head(5).to_dict(orient="records"),
            )


def _resolve_source_grid_id_column(pipeline: PipelineConfig, df: pd.DataFrame) -> str:
    if not pipeline.aermod_grid_id:
        raise ValueError("pipeline.aermod_grid_id must be configured before running AERMOD concentrations.")
    if pipeline.aermod_grid_id not in df.columns:
        raise ValueError(
            "AERMOD emissions input is missing a source grid id column. "
            f"Expected configured pipeline.aermod_grid_id='{pipeline.aermod_grid_id}'."
        )
    return pipeline.aermod_grid_id


def _resolve_target_grid_id_column(pipeline: PipelineConfig, gdf: gpd.GeoDataFrame) -> str:
    if not pipeline.aermod_grid_id:
        raise ValueError("pipeline.aermod_grid_id must be configured before running AERMOD concentrations.")
    if pipeline.aermod_grid_id in gdf.columns:
        return pipeline.aermod_grid_id
    raise ValueError(
        "AERMOD grid is missing an id column. "
        f"Expected configured pipeline.aermod_grid_id='{pipeline.aermod_grid_id}'."
    )


def _emissions_columns(df: pd.DataFrame, pipeline: PipelineConfig) -> list[str]:
    ordered = [f"tons_per_year_{pollutant}" for pollutant in list(pipeline.pollutants)]
    present = [col for col in ordered if col in df.columns]
    if not present:
        raise ValueError(
            "AERMOD emissions input does not contain any configured tons_per_year_* columns. "
            f"Expected one or more of: {ordered}"
        )
    return present


def _geometry_midpoints(geometry: gpd.GeoSeries) -> tuple[np.ndarray, np.ndarray]:
    bounds = geometry.bounds
    x = ((bounds["minx"].to_numpy(dtype=np.float64) + bounds["maxx"].to_numpy(dtype=np.float64)) * 0.5)
    y = ((bounds["miny"].to_numpy(dtype=np.float64) + bounds["maxy"].to_numpy(dtype=np.float64)) * 0.5)
    return x, y


def _prepare_source_emissions(
    *,
    emissions_gdf: gpd.GeoDataFrame,
    source_id_col: str,
    emissions_cols: list[str],
    grid_size_meters: float,
    origin_x: float,
    origin_y: float,
) -> pd.DataFrame:
    source = emissions_gdf.copy()
    if source.geometry.isna().any():
        raise ValueError("AERMOD emissions input contains missing geometry.")
    source_xm, source_ym = _geometry_midpoints(source.geometry)
    source_frame = pd.DataFrame({source_id_col: source[source_id_col].to_numpy(), "source_xm": source_xm, "source_ym": source_ym})
    pop_col = _SOURCE_POPULATION_COLUMN if _SOURCE_POPULATION_COLUMN in source.columns else None
    temporal_col = _SOURCE_TEMPORAL_COLUMN if _SOURCE_TEMPORAL_COLUMN in source.columns else None
    height_col = _SOURCE_HEIGHT_COLUMN if _SOURCE_HEIGHT_COLUMN in source.columns else None
    if pop_col:
        source_frame["source_population"] = pd.to_numeric(source[pop_col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if temporal_col:
        source_frame["source_temporal_class"] = source[temporal_col].astype("string").fillna("").to_numpy()
    if height_col:
        source_frame["source_release_height"] = pd.to_numeric(source[height_col], errors="coerce").fillna(_DEFAULT_HEIGHT).to_numpy(dtype=np.float64)
    for col in emissions_cols:
        source_frame[col] = pd.to_numeric(source[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    aggregation_select = ", ".join(
        [f"AVG(source_xm) AS source_xm", f"AVG(source_ym) AS source_ym"]
        + (["AVG(source_population) AS source_population"] if pop_col else [])
        + (["ANY_VALUE(source_temporal_class) AS source_temporal_class"] if temporal_col else [])
        + (["ANY_VALUE(source_release_height) AS source_release_height"] if height_col else [])
        + [f"SUM({col}) AS {col}" for col in emissions_cols]
    )
    con = duckdb.connect(database=":memory:")
    try:
        con.register("source_frame", source_frame)
        source = con.execute(
            f"""
            SELECT
                "{source_id_col}" AS "{source_id_col}",
                {aggregation_select}
            FROM source_frame
            GROUP BY 1
            """
        ).df()
    finally:
        con.close()
    source["source_ix"] = np.rint((source["source_xm"] - origin_x) / grid_size_meters).astype(int)
    source["source_iy"] = np.rint((source["source_ym"] - origin_y) / grid_size_meters).astype(int)
    if "source_population" in source.columns:
        source["source_population"] = pd.to_numeric(source["source_population"], errors="coerce").fillna(0.0)
    if "source_temporal_class" in source.columns:
        source["source_temporal_class"] = source["source_temporal_class"].astype("string").str.strip().replace("", pd.NA)
    if "source_release_height" in source.columns:
        source["source_release_height"] = pd.to_numeric(source["source_release_height"], errors="coerce").fillna(_DEFAULT_HEIGHT)
    return source


def _prepare_target_grid(
    *,
    target_grid: gpd.GeoDataFrame,
    target_grid_path: Optional[str],
    outputs_dir: Path,
    target_id_col: str,
    target_epsg: int,
    grid_size_meters: float,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, float, float]:
    target = target_grid.copy()
    if target.geometry.isna().any():
        raise ValueError("AERMOD target grid contains missing geometry.")
    cache_path = (
        _target_index_cache_path(
            outputs_dir=outputs_dir,
            grid_path=target_grid_path,
            target_epsg=target_epsg,
            target_id_col=target_id_col,
        )
        if target_grid_path
        else None
    )
    target_index: Optional[pd.DataFrame] = None
    if cache_path and cache_path.exists():
        target_index = pd.read_parquet(cache_path)
        target = target.merge(target_index[[target_id_col, "target_xm", "target_ym", "target_ix", "target_iy"]], on=target_id_col, how="left")
    else:
        target_xm, target_ym = _geometry_midpoints(target.geometry)
        target["target_xm"] = target_xm
        target["target_ym"] = target_ym
        x0 = float(target["target_xm"].min())
        y0 = float(target["target_ym"].min())
        target["target_ix"] = np.rint((target["target_xm"] - x0) / grid_size_meters).astype(int)
        target["target_iy"] = np.rint((target["target_ym"] - y0) / grid_size_meters).astype(int)
        target["target_key"] = (
            target["target_ix"].to_numpy(dtype=np.int64) * np.int64(int(target["target_iy"].max()) + 1)
            + target["target_iy"].to_numpy(dtype=np.int64)
        )
        target_index = target[[target_id_col, "target_xm", "target_ym", "target_ix", "target_iy", "target_key"]].copy()
        if cache_path:
            target_index.to_parquet(cache_path, index=False)
    x0 = float(target["target_xm"].min())
    y0 = float(target["target_ym"].min())
    if target_index is None:
        target_index = target[[target_id_col, "target_xm", "target_ym", "target_ix", "target_iy"]].copy()
    if "target_key" not in target_index.columns:
        stride = int(target["target_iy"].max()) + 1
        target_index["target_key"] = (
            target_index["target_ix"].to_numpy(dtype=np.int64) * np.int64(stride)
            + target_index["target_iy"].to_numpy(dtype=np.int64)
        )
    return target, target_index, x0, y0


def _load_asrv_patterns(path: str) -> gpd.GeoDataFrame:
    candidate = Path(path)
    if candidate.suffix.lower() == ".parquet":
        try:
            return gpd.read_parquet(
                candidate,
                columns=[
                    "Concentration",
                    "Distance",
                    "DataSet_ID",
                    "Emissions",
                    "Height",
                    "Urban_Rural",
                    "geometry",
                ],
            )
        except Exception:
            frame = read_table(str(candidate))
            if {"Longitude", "Latitude"}.issubset(frame.columns):
                return gpd.GeoDataFrame(
                    frame,
                    geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
                    crs="EPSG:4326",
                )
            raise
    return read_vector(str(candidate))


def _load_emissions_input(path: str, *, pipeline: PipelineConfig) -> gpd.GeoDataFrame:
    candidate = Path(path)
    if candidate.suffix.lower() == ".parquet":
        if not pipeline.aermod_grid_id:
            raise ValueError("pipeline.aermod_grid_id must be configured before loading AERMOD emissions input.")
        emissions_cols = [f"tons_per_year_{pollutant}" for pollutant in list(pipeline.pollutants)]
        requested = [
            pipeline.aermod_grid_id,
            *emissions_cols,
            _SOURCE_POPULATION_COLUMN,
            _SOURCE_TEMPORAL_COLUMN,
            _SOURCE_HEIGHT_COLUMN,
        ]
        requested = list(dict.fromkeys(requested + ["geometry"]))
        try:
            return gpd.read_parquet(candidate, columns=requested)
        except Exception:
            return gpd.read_parquet(candidate)
    return read_vector(str(candidate))


def _load_vector_subset(path: str, *, columns: Optional[list[str]] = None) -> gpd.GeoDataFrame:
    candidate = Path(path)
    if candidate.suffix.lower() == ".parquet":
        requested = None if columns is None else list(dict.fromkeys(columns + ["geometry"]))
        return gpd.read_parquet(candidate, columns=requested)
    gdf = read_vector(path)
    if columns is None:
        return gdf
    keep = [col for col in columns if col in gdf.columns]
    if "geometry" not in keep:
        keep.append("geometry")
    return gdf[keep].copy()


def _build_default_pattern_key() -> str:
    return f"{_DEFAULT_SITE}__{_DEFAULT_URBAN_CLASS}__{_DEFAULT_TEMPORAL}__{_DEFAULT_HEIGHT:g}"


def _pattern_keys_from_raw_frame(patterns_df: pd.DataFrame) -> pd.Series:
    return (
        patterns_df["DataSet_ID"].astype("string").str.strip()
        + "__"
        + pd.to_numeric(patterns_df["Urban_Rural"], errors="coerce").astype("Int64").astype("string")
        + "__"
        + patterns_df["Emissions"].astype("string").str.strip()
        + "__"
        + pd.to_numeric(patterns_df["Height"], errors="coerce").map(lambda value: f"{float(value):g}" if pd.notna(value) else None).astype("string")
    )


def _classify_urban(population: pd.Series) -> pd.Series:
    values = pd.to_numeric(population, errors="coerce").fillna(0.0)
    return pd.Series(
        np.where(values < 1000, 0, np.where(values < 10000, 1000, 10000)),
        index=population.index,
        dtype="int64",
    )


def _asrv_cache_key(*, path: str, target_epsg: int, grid_size_meters: float, pattern_keys: list[str]) -> str:
    stat = Path(path).stat()
    payload = json.dumps(
        {
            "path": str(Path(path).resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "target_epsg": int(target_epsg),
            "grid_size_meters": round(float(grid_size_meters), 6),
            "kernel_radius_meters": _KERNEL_RADIUS_METERS,
            "pattern_keys": pattern_keys,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _kernel_cache_paths(
    *,
    raw_dir: Path,
    path: str,
    target_epsg: int,
    grid_size_meters: float,
    pattern_keys: list[str],
) -> tuple[Path, Path]:
    cache_key = _asrv_cache_key(
        path=path,
        target_epsg=target_epsg,
        grid_size_meters=grid_size_meters,
        pattern_keys=pattern_keys,
    )
    return (
        raw_dir / f"aermod_asrv_kernels_{cache_key}.parquet",
        raw_dir / f"aermod_asrv_patterns_{cache_key}.parquet",
    )


def _target_index_cache_path(*, outputs_dir: Path, grid_path: str, target_epsg: int, target_id_col: str) -> Path:
    stat = Path(grid_path).stat()
    payload = json.dumps(
        {
            "path": str(Path(grid_path).resolve()),
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
            "target_epsg": int(target_epsg),
            "target_id_col": target_id_col,
        },
        sort_keys=True,
    )
    cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return outputs_dir / f"aermod_target_index_{cache_key}.parquet"


def _project_asrv_patterns(patterns_gdf: gpd.GeoDataFrame, *, target_epsg: int) -> pd.DataFrame:
    projected = patterns_gdf.to_crs(epsg=target_epsg).copy()
    projected["Grid_X"] = projected.geometry.x.astype(float)
    projected["Grid_Y"] = projected.geometry.y.astype(float)
    projected["Concentration"] = pd.to_numeric(projected["Concentration"], errors="coerce")
    projected["Distance"] = pd.to_numeric(projected.get("Distance"), errors="coerce")
    projected["Height"] = pd.to_numeric(projected["Height"], errors="coerce")
    projected["Urban_Rural"] = pd.to_numeric(projected["Urban_Rural"], errors="coerce")
    projected["DataSet_ID"] = projected["DataSet_ID"].astype("string").str.strip()
    projected["Emissions"] = projected["Emissions"].astype("string").str.strip()
    projected = projected.dropna(
        subset=["Grid_X", "Grid_Y", "Concentration", "Height", "Urban_Rural", "DataSet_ID", "Emissions"]
    ).copy()
    projected["pattern_key"] = _pattern_keys_from_raw_frame(pd.DataFrame(projected.drop(columns="geometry")))
    return pd.DataFrame(projected.drop(columns="geometry"))


def _normalize_projected_asrv_patterns(projected: pd.DataFrame, *, grid_size_meters: float) -> pd.DataFrame:
    projected = projected.copy()
    grid_area = float(grid_size_meters) * float(grid_size_meters)
    annual_tons_equivalent = (
        _ASRV_SOURCE_RATE_G_PER_S_M2
        * grid_area
        * _ASRV_ACTIVE_DAYS_PER_YEAR
        * 24.0
        * 3600.0
        / grams_per_short_ton
    )
    projected["Concentration"] = projected["Concentration"] / annual_tons_equivalent
    return projected


def _resolve_site_reference(patterns_df: pd.DataFrame) -> pd.DataFrame:
    site_ref = (
        patterns_df.groupby("DataSet_ID", dropna=False)[["Grid_X", "Grid_Y"]]
        .mean()
        .rename(columns={"Grid_X": "centroid_x", "Grid_Y": "centroid_y"})
        .reset_index()
    )
    site_pts = patterns_df[["DataSet_ID", "Grid_X", "Grid_Y"]].merge(site_ref, on="DataSet_ID", how="left")
    site_pts["dist_to_centroid"] = np.sqrt(
        (site_pts["Grid_X"] - site_pts["centroid_x"]) ** 2 + (site_pts["Grid_Y"] - site_pts["centroid_y"]) ** 2
    )
    site_pts = site_pts.sort_values(["DataSet_ID", "dist_to_centroid"])
    return (
        site_pts.groupby("DataSet_ID", dropna=False)
        .head(1)
        .rename(columns={"Grid_X": "site_xm", "Grid_Y": "site_ym"})[["DataSet_ID", "site_xm", "site_ym"]]
        .reset_index(drop=True)
    )


def _assign_source_pattern_keys(
    *,
    source_df: pd.DataFrame,
    site_reference: pd.DataFrame,
    available_pattern_keys: set[str],
) -> pd.DataFrame:
    result = source_df.copy()
    site_x = site_reference["site_xm"].to_numpy(dtype=np.float64)
    site_y = site_reference["site_ym"].to_numpy(dtype=np.float64)
    source_x = result["source_xm"].to_numpy(dtype=np.float64)[:, None]
    source_y = result["source_ym"].to_numpy(dtype=np.float64)[:, None]
    nearest_idx = np.argmin((source_x - site_x[None, :]) ** 2 + (source_y - site_y[None, :]) ** 2, axis=1)
    result["nearest_site"] = site_reference["DataSet_ID"].to_numpy()[nearest_idx]
    if "source_population" in result.columns:
        result["selected_urban"] = _classify_urban(result["source_population"])
    else:
        result["selected_urban"] = _DEFAULT_URBAN_CLASS
    if "source_temporal_class" in result.columns:
        temporal = result["source_temporal_class"].fillna(_DEFAULT_TEMPORAL).astype(str).str.strip()
        result["selected_temporal"] = temporal.where(temporal != "", _DEFAULT_TEMPORAL)
    else:
        result["selected_temporal"] = _DEFAULT_TEMPORAL
    if "source_release_height" in result.columns:
        result["selected_height"] = pd.to_numeric(result["source_release_height"], errors="coerce").fillna(_DEFAULT_HEIGHT)
    else:
        result["selected_height"] = _DEFAULT_HEIGHT
    result["pattern_key_raw"] = (
        result["nearest_site"].astype(str)
        + "__"
        + result["selected_urban"].astype(int).astype(str)
        + "__"
        + result["selected_temporal"].astype(str)
        + "__"
        + result["selected_height"].map(lambda value: f"{float(value):g}")
    )
    result["pattern_key"] = result["pattern_key_raw"]

    same_site_default = (
        result["nearest_site"].astype(str)
        + "__"
        + str(_DEFAULT_URBAN_CLASS)
        + "__"
        + _DEFAULT_TEMPORAL
        + "__"
        + f"{_DEFAULT_HEIGHT:g}"
    )
    default_pattern = _build_default_pattern_key()

    missing_mask = ~result["pattern_key"].isin(available_pattern_keys)
    result.loc[missing_mask, "pattern_key"] = same_site_default.loc[missing_mask]
    missing_mask = ~result["pattern_key"].isin(available_pattern_keys)
    result.loc[missing_mask, "pattern_key"] = default_pattern
    if (~result["pattern_key"].isin(available_pattern_keys)).any():
        raise ValueError("Some source cells still have no matched ASRV pattern after fallback.")
    return result
    site_pts = patterns_df[["DataSet_ID", "Grid_X", "Grid_Y"]].merge(site_ref, on="DataSet_ID", how="left")
    site_pts["dist_to_centroid"] = np.sqrt(
        (site_pts["Grid_X"] - site_pts["centroid_x"]) ** 2 + (site_pts["Grid_Y"] - site_pts["centroid_y"]) ** 2
    )
    site_pts = site_pts.sort_values(["DataSet_ID", "dist_to_centroid"])
    return (
        site_pts.groupby("DataSet_ID", dropna=False)
        .head(1)
        .rename(columns={"Grid_X": "site_xm", "Grid_Y": "site_ym"})[["DataSet_ID", "site_xm", "site_ym"]]
        .reset_index(drop=True)
    )


def _build_kernel_for_pattern(pattern_df: pd.DataFrame, *, grid_size_meters: float) -> _Kernel:
    ordered = pattern_df.copy()
    if "Distance" in ordered.columns:
        ordered["_distance_rank"] = ordered["Distance"].abs()
        ordered = ordered.sort_values("_distance_rank")
    center = ordered.iloc[0]
    kernel = pattern_df.copy()
    kernel["dx"] = kernel["Grid_X"] - float(center["Grid_X"])
    kernel["dy"] = kernel["Grid_Y"] - float(center["Grid_Y"])
    kernel["dist"] = np.sqrt(kernel["dx"] ** 2 + kernel["dy"] ** 2)
    kernel = kernel.loc[kernel["dist"] <= _KERNEL_RADIUS_METERS].copy()
    kernel["dix"] = np.rint(kernel["dx"] / grid_size_meters).astype(int)
    kernel["diy"] = np.rint(kernel["dy"] / grid_size_meters).astype(int)
    kernel = (
        kernel.groupby(["dix", "diy"], dropna=False)["Concentration"]
        .mean()
        .reset_index()
        .rename(columns={"Concentration": "response_per_ton"})
    )
    return {
        "dix": kernel["dix"].to_numpy(dtype=np.int32),
        "diy": kernel["diy"].to_numpy(dtype=np.int32),
        "response_per_ton": kernel["response_per_ton"].to_numpy(dtype=np.float64),
    }


def _build_kernel_library(patterns_df: pd.DataFrame, *, grid_size_meters: float) -> dict[str, _Kernel]:
    kernels: dict[str, _Kernel] = {}
    for pattern_key, pattern_df in patterns_df.groupby("pattern_key", dropna=False):
        kernels[str(pattern_key)] = _build_kernel_for_pattern(pattern_df, grid_size_meters=grid_size_meters)
    return kernels


def _serialize_kernel_library(kernel_library: dict[str, _Kernel]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern_key, kernel in kernel_library.items():
        if kernel["response_per_ton"].size == 0:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "pattern_key": pattern_key,
                    "dix": kernel["dix"],
                    "diy": kernel["diy"],
                    "response_per_ton": kernel["response_per_ton"],
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["pattern_key", "dix", "diy", "response_per_ton"])
    return pd.concat(frames, ignore_index=True)


def _deserialize_kernel_library(kernel_df: pd.DataFrame) -> dict[str, _Kernel]:
    kernels: dict[str, _Kernel] = {}
    if kernel_df.empty:
        return kernels
    for pattern_key, frame in kernel_df.groupby("pattern_key", dropna=False):
        kernels[str(pattern_key)] = {
            "dix": frame["dix"].to_numpy(dtype=np.int32),
            "diy": frame["diy"].to_numpy(dtype=np.int32),
            "response_per_ton": frame["response_per_ton"].to_numpy(dtype=np.float64),
        }
    return kernels


def _load_or_build_kernel_library(
    *,
    raw_dir: Path,
    asrv_patterns_file: str,
    asrv_patterns_epsg: Optional[int],
    target_epsg: int,
    grid_size_meters: float,
    requested_pattern_keys: list[str],
    projected_patterns_df: pd.DataFrame,
) -> tuple[dict[str, _Kernel], Optional[pd.DataFrame]]:
    kernels_path, normalized_patterns_path = _kernel_cache_paths(
        raw_dir=raw_dir,
        path=asrv_patterns_file,
        target_epsg=target_epsg,
        grid_size_meters=grid_size_meters,
        pattern_keys=requested_pattern_keys,
    )
    if kernels_path.exists():
        kernel_df = pd.read_parquet(kernels_path)
        logger.info("%s reusing cached ASRV kernels from %s", _step_label("3"), kernels_path)
        patterns_df = pd.read_parquet(normalized_patterns_path) if normalized_patterns_path.exists() else None
        return _deserialize_kernel_library(kernel_df), patterns_df

    filtered_patterns_df = projected_patterns_df.loc[projected_patterns_df["pattern_key"].isin(requested_pattern_keys)].copy()
    patterns_df = _normalize_projected_asrv_patterns(filtered_patterns_df, grid_size_meters=grid_size_meters)
    kernel_library = _build_kernel_library(patterns_df, grid_size_meters=grid_size_meters)
    kernels_path.parent.mkdir(parents=True, exist_ok=True)
    _serialize_kernel_library(kernel_library).to_parquet(kernels_path, index=False)
    patterns_df.to_parquet(normalized_patterns_path, index=False)
    logger.info("%s cached ASRV kernels at %s", _step_label("3"), kernels_path)
    return kernel_library, patterns_df


def _apply_kernels(
    *,
    source_df: pd.DataFrame,
    target_index: pd.DataFrame,
    target_id_col: str,
    emissions_cols: list[str],
    kernel_library: dict[str, _Kernel],
) -> pd.DataFrame:
    target_key_index = pd.Index(target_index["target_key"].to_numpy(dtype=np.int64))
    target_ids = target_index[target_id_col].to_numpy(dtype=np.int64)
    result_arrays = {
        f"concentration_{col.removeprefix('tons_per_year_')}": np.zeros(len(target_index), dtype=np.float64)
        for col in emissions_cols
    }
    stride = int(target_index["target_iy"].max()) + 1 if not target_index.empty else 1
    for pattern_key, source_group in source_df.groupby("pattern_key", dropna=False):
        kernel = kernel_library.get(str(pattern_key))
        if kernel is None or kernel["response_per_ton"].size == 0:
            raise ValueError(f"No AERMOD ASRV kernel available for pattern {pattern_key}")
        for start in range(0, len(source_group), _SOURCE_CHUNK_SIZE):
            chunk = source_group.iloc[start : start + _SOURCE_CHUNK_SIZE]
            source_ix = chunk["source_ix"].to_numpy(dtype=np.int64)
            source_iy = chunk["source_iy"].to_numpy(dtype=np.int64)
            source_values = {
                f"concentration_{col.removeprefix('tons_per_year_')}": chunk[col].to_numpy(dtype=np.float64)
                for col in emissions_cols
            }
            for dix, diy, response in zip(kernel["dix"], kernel["diy"], kernel["response_per_ton"], strict=True):
                keys = (source_ix + np.int64(dix)) * np.int64(stride) + (source_iy + np.int64(diy))
                positions = target_key_index.get_indexer(keys)
                valid_mask = positions >= 0
                if not valid_mask.any():
                    continue
                valid_positions = positions[valid_mask]
                for name, values in source_values.items():
                    np.add.at(result_arrays[name], valid_positions, values[valid_mask] * float(response))
    if not result_arrays:
        return pd.DataFrame(columns=[target_id_col])
    concentrations = pd.DataFrame({target_id_col: target_ids})
    for name, values in result_arrays.items():
        concentrations[name] = values
    return concentrations.reset_index(drop=True)


def _write_outputs(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, index=False)
    gdf.to_file(output_path.with_suffix(".gpkg"), driver="GPKG")


def _attach_concentrations(
    *,
    target_grid: gpd.GeoDataFrame,
    concentrations_df: pd.DataFrame,
    target_id_col: str,
) -> gpd.GeoDataFrame:
    result = target_grid.copy()
    concentration_cols = [col for col in concentrations_df.columns if col != target_id_col]
    if not concentration_cols:
        for col in concentration_cols:
            result[col] = 0.0
        return result
    lookup = concentrations_df.set_index(target_id_col)
    target_ids = result[target_id_col].to_numpy()
    for col in concentration_cols:
        result[col] = lookup[col].reindex(target_ids, fill_value=0.0).to_numpy(dtype=np.float64)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=target_grid.crs)


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    emissions_input_path: str,
    emissions_input_gdf: Optional[gpd.GeoDataFrame] = None,
    target_grid_gdf: Optional[gpd.GeoDataFrame] = None,
) -> tuple[gpd.GeoDataFrame, np.ndarray, Path]:
    if not pipeline.aermod_enabled:
        raise ValueError("AERMOD concentration step was called but pipeline.aermod_enabled is false.")
    if not pipeline.asrv_patterns_file:
        raise ValueError(
            "asrv_patterns_file must be configured to run AERMOD concentrations. "
            "Set impacts.dispersions.aermod.asrv_patterns_file in settings.yaml."
        )
    if not pipeline.aermod_grid_path:
        raise ValueError("aermod_grid_path must be configured before running AERMOD concentrations.")

    log_step_banner("Step 3", "Compute AERMOD Concentrations", logger=logger)

    log_substep_banner("3.0", "load source emissions and target grid", logger=logger)
    emissions_gdf = (
        emissions_input_gdf.copy()
        if emissions_input_gdf is not None
        else _load_emissions_input(emissions_input_path, pipeline=pipeline)
    )
    target_grid = (
        target_grid_gdf.copy()
        if target_grid_gdf is not None
        else _load_vector_subset(pipeline.aermod_grid_path)
    )
    if emissions_gdf.crs is not None:
        emissions_gdf = emissions_gdf.to_crs(epsg=int(pipeline.output_epsg))
    if target_grid.crs is not None:
        target_grid = target_grid.to_crs(epsg=int(pipeline.output_epsg))
    source_id_col = _resolve_source_grid_id_column(pipeline, emissions_gdf)
    target_id_col = _resolve_target_grid_id_column(pipeline, target_grid)
    emissions_cols = _emissions_columns(emissions_gdf, pipeline)
    if emissions_input_gdf is None:
        keep_cols = [source_id_col] + emissions_cols
        emissions_gdf = emissions_gdf[keep_cols + ["geometry"]].copy()
    if target_grid_gdf is None:
        target_grid = target_grid[[target_id_col, "geometry"]].copy()
    _trace_frame("0", "aermod_source_emissions", pd.DataFrame(emissions_gdf.drop(columns="geometry")), key_cols=[source_id_col])

    log_substep_banner("3.1", "prepare local AERMOD grid indices", logger=logger)
    grid_size_meters = float(pipeline.grid_size_meters)
    target_grid, target_index, origin_x, origin_y = _prepare_target_grid(
        target_grid=target_grid,
        target_grid_path=None if target_grid_gdf is not None else pipeline.aermod_grid_path,
        outputs_dir=raw_dir,
        target_id_col=target_id_col,
        target_epsg=int(pipeline.output_epsg),
        grid_size_meters=grid_size_meters,
    )
    source_df = _prepare_source_emissions(
        emissions_gdf=emissions_gdf,
        source_id_col=source_id_col,
        emissions_cols=emissions_cols,
        grid_size_meters=grid_size_meters,
        origin_x=origin_x,
        origin_y=origin_y,
    )
    logger.info(
        "%s prepared %d source cells over %d target cells at %.2fm grid spacing",
        _step_label("1"),
        len(source_df),
        len(target_grid),
        grid_size_meters,
    )

    log_substep_banner("3.2", "load and normalize ASRV pattern library", logger=logger)
    patterns_gdf = _load_asrv_patterns(pipeline.asrv_patterns_file)
    if pipeline.asrv_patterns_epsg is not None and patterns_gdf.crs is None:
        patterns_gdf = patterns_gdf.set_crs(epsg=int(pipeline.asrv_patterns_epsg))
    if patterns_gdf.crs is None:
        raise ValueError(
            "AERMOD ASRV patterns file is missing CRS metadata and no asrv_patterns_epsg was provided."
        )
    projected_patterns_df = _project_asrv_patterns(patterns_gdf, target_epsg=int(pipeline.output_epsg))
    site_reference = _resolve_site_reference(projected_patterns_df)
    available_pattern_keys = set(projected_patterns_df["pattern_key"].dropna().astype(str).tolist())
    source_df = _assign_source_pattern_keys(
        source_df=source_df,
        site_reference=site_reference,
        available_pattern_keys=available_pattern_keys,
    )
    requested_pattern_keys = sorted(set(source_df["pattern_key"].dropna().astype(str).tolist()))
    kernel_library, patterns_df = _load_or_build_kernel_library(
        raw_dir=raw_dir,
        asrv_patterns_file=pipeline.asrv_patterns_file,
        asrv_patterns_epsg=pipeline.asrv_patterns_epsg,
        target_epsg=int(pipeline.output_epsg),
        grid_size_meters=grid_size_meters,
        requested_pattern_keys=requested_pattern_keys,
        projected_patterns_df=projected_patterns_df,
    )
    if patterns_df is not None:
        _trace_frame("2", "asrv_patterns", patterns_df, key_cols=["DataSet_ID", "pattern_key"])
        _trace_frame("2", "asrv_site_reference", site_reference, key_cols=["DataSet_ID"])

    log_substep_banner("3.3", "build ASRV kernels", logger=logger)
    logger.info(
        "%s using %d ASRV pattern(s) across %d source cells",
        _step_label("3"),
        len(requested_pattern_keys),
        len(source_df),
    )

    log_substep_banner("3.4", "accumulate receptor concentrations", logger=logger)
    concentrations_df = _apply_kernels(
        source_df=source_df,
        target_index=target_index,
        target_id_col=target_id_col,
        emissions_cols=emissions_cols,
        kernel_library=kernel_library,
    )
    _trace_frame("4", "aermod_concentrations", concentrations_df, key_cols=[target_id_col])

    log_substep_banner("3.5", "write concentration outputs", logger=logger)
    output_path = raw_dir / "beam_aermod_concentrations.parquet"
    result_gdf = _attach_concentrations(
        target_grid=target_grid,
        concentrations_df=concentrations_df,
        target_id_col=target_id_col,
    )
    _write_outputs(result_gdf, output_path)
    logger.info("%s AERMOD concentrations → %s", _step_label("5"), output_path)
    return result_gdf, target_grid[target_id_col].to_numpy(dtype=int), output_path
