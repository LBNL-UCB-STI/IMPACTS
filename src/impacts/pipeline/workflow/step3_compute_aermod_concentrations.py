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
from scipy.signal import fftconvolve

from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import _table_available_columns
from ...common import _should_show_duckdb_progress_bar
from ...common import configure_duckdb_connection
from ...common import read_table
from ...common import read_vector
from ...config.defaults import aermod_active_days_per_year as default_aermod_active_days_per_year
from ...config.defaults import grams_per_short_ton
from ...manifest.schema import PipelineConfig
from . import _step_label

logger = logging.getLogger(__name__)

_AERMOD_SOURCE_ID_COLUMN = "aermod_cell_id"
_SOURCE_TEMPORAL_COLUMN = "source_temporal_class"
_SOURCE_HEIGHT_COLUMN = "source_release_height"
_SOURCE_URBAN_COLUMN = "source_urban_class"
_AERMOD_SUPPORT_COLUMNS = {
    "PrimaryPM25": "has_aermod_primarypm25",
    "BC": "has_aermod_bc",
    "NO2": "has_aermod_no2",
}
_AERMOD_SUPPORT_COLUMN_SET = set(_AERMOD_SUPPORT_COLUMNS.values())

# Maps canonical pollutant names (from emissions columns) to output concentration column names.
# Mirrors the InMAP naming convention used in step 2 so impacts steps work uniformly.
_POLLUTANT_TO_CONCENTRATION_COLUMN: dict[str, str] = {
    "PM25": "PrimaryPM25",
    "BC": "BC",
    "NOx": "NO2",
    "NH3": "pNH4",
    "SOx": "pSO4",
    "ROG": "SOA",
}

_ASRV_SOURCE_RATE_G_PER_S_M2 = 0.1
_ASRV_ACTIVE_DAYS_PER_YEAR = default_aermod_active_days_per_year


class _Kernel(TypedDict):
    dix: np.ndarray
    diy: np.ndarray
    response_per_ton: np.ndarray


def _trace_frame(step: str, label: str, df: pd.DataFrame, *, key_cols: Optional[list[str]] = None) -> None:
    logger.info("%s trace %s shape=%s", _step_label(f"3.{step}"), label, df.shape)
    preview = list(df.columns[:20])
    suffix = "" if len(df.columns) <= 20 else " ..."
    logger.info("%s trace %s columns(%d): %s%s", _step_label(f"3.{step}"), label, len(df.columns), preview, suffix)
    if key_cols:
        present = [col for col in key_cols if col in df.columns]
        if present and not df.empty:
            logger.info(
                "%s trace %s sample_keys=%s",
                _step_label(f"3.{step}"),
                label,
                df[present].head(5).to_dict(orient="records"),
            )


def _resolve_source_grid_id_column(df: pd.DataFrame) -> str:
    if _AERMOD_SOURCE_ID_COLUMN not in df.columns:
        raise ValueError(
            "AERMOD emissions input is missing a source grid id column. "
            f"Expected '{_AERMOD_SOURCE_ID_COLUMN}'."
        )
    return _AERMOD_SOURCE_ID_COLUMN


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
    ordered = [f"tons_per_year_{pollutant}_aermod_allocated" for pollutant in list(pipeline.pollutants)]
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
    outputs_dir: Path,
) -> pd.DataFrame:
    source = emissions_gdf
    if source.geometry.isna().any():
        raise ValueError("AERMOD emissions input contains missing geometry.")
    required_source_class_cols = [_SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN, _SOURCE_URBAN_COLUMN]
    missing_source_class_cols = [col for col in required_source_class_cols if col not in source.columns]
    if missing_source_class_cols:
        raise ValueError(
            "AERMOD emissions input is missing required source class columns: "
            f"{missing_source_class_cols}"
        )
    temporal = source[_SOURCE_TEMPORAL_COLUMN].astype("string").str.strip()
    heights = pd.to_numeric(source[_SOURCE_HEIGHT_COLUMN], errors="coerce")
    urban = pd.to_numeric(source[_SOURCE_URBAN_COLUMN], errors="coerce")
    invalid_source_class = (
        temporal.isna()
        | temporal.eq("")
        | heights.isna()
        | ~np.isfinite(heights)
        | heights.le(0.0)
        | urban.isna()
    )
    if invalid_source_class.any():
        sample = source.loc[
            invalid_source_class,
            [source_id_col, _SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN, _SOURCE_URBAN_COLUMN],
        ].head(10).to_dict(orient="records")
        raise ValueError(
            "AERMOD emissions input contains invalid source class values. "
            f"sample={sample}"
        )
    source_xm, source_ym = _geometry_midpoints(source.geometry)
    source_frame = pd.DataFrame({source_id_col: source[source_id_col].to_numpy(), "source_xm": source_xm, "source_ym": source_ym})
    source_frame[_SOURCE_TEMPORAL_COLUMN] = temporal.to_numpy()
    source_frame[_SOURCE_HEIGHT_COLUMN] = heights.to_numpy(dtype=np.float64)
    source_frame[_SOURCE_URBAN_COLUMN] = urban.to_numpy(dtype=np.int64)
    for col in emissions_cols:
        source_frame[col] = pd.to_numeric(source[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    source_class_cols = required_source_class_cols
    aggregation_select = ", ".join(
        ["AVG(source_xm) AS source_xm", "AVG(source_ym) AS source_ym"]
        + [f'"{col}" AS "{col}"' for col in source_class_cols]
        + [f"SUM({col}) AS {col}" for col in emissions_cols]
    )
    group_by_cols = ", ".join([f'"{source_id_col}"'] + [f'"{col}"' for col in source_class_cols])
    con = duckdb.connect()
    show_progress = _should_show_duckdb_progress_bar()
    try:
        configure_duckdb_connection(con, working_dir=outputs_dir, show_progress=show_progress, profile="balanced")
        con.register("source_frame", source_frame)
        source = con.execute(
            f"""
            SELECT
                "{source_id_col}" AS "{source_id_col}",
                {aggregation_select}
            FROM source_frame
            GROUP BY {group_by_cols}
            """
        ).df()
    finally:
        con.close()
    source["source_ix"] = np.rint((source["source_xm"] - origin_x) / grid_size_meters).astype(int)
    source["source_iy"] = np.rint((source["source_ym"] - origin_y) / grid_size_meters).astype(int)
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
        target_index = target[[target_id_col, "target_xm", "target_ym", "target_ix", "target_iy", "target_key"]]
        if cache_path:
            target_index.to_parquet(cache_path, index=False)
    x0 = float(target["target_xm"].min())
    y0 = float(target["target_ym"].min())
    if target_index is None:
        target_index = target[[target_id_col, "target_xm", "target_ym", "target_ix", "target_iy"]]
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
        available = set(_table_available_columns(str(candidate)))
        required = {"Concentration", "Distance", "DataSet_ID", "Emissions", "Height", "Urban_Rural"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"AERMOD ASRV patterns parquet is missing required columns: {missing}")
        if "geometry" in available:
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
        if {"Longitude", "Latitude"}.issubset(available):
            frame = read_table(str(candidate))
            return gpd.GeoDataFrame(
                frame,
                geometry=gpd.points_from_xy(frame["Longitude"], frame["Latitude"]),
                crs="EPSG:4326",
            )
        raise ValueError(
            "AERMOD ASRV patterns parquet must include either geometry or Longitude/Latitude columns."
        )
    return read_vector(str(candidate))


def _load_emissions_input(path: str, *, pipeline: PipelineConfig) -> gpd.GeoDataFrame:
    candidate = Path(path)
    if candidate.suffix.lower() == ".parquet":
        available = set(_table_available_columns(str(candidate)))
        required = {_AERMOD_SOURCE_ID_COLUMN, _SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN, _SOURCE_URBAN_COLUMN, "geometry"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                "AERMOD emissions input parquet is missing required source columns: "
                f"{missing}"
            )
        emissions_cols = [
            f"tons_per_year_{pollutant}_aermod_allocated"
            for pollutant in list(pipeline.pollutants)
            if f"tons_per_year_{pollutant}_aermod_allocated" in available
        ]
        requested = list(dict.fromkeys([_AERMOD_SOURCE_ID_COLUMN, *emissions_cols, _SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN, _SOURCE_URBAN_COLUMN, "geometry"]))
        return gpd.read_parquet(candidate, columns=requested)
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
    return gdf[keep]


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
    projected = patterns_gdf.to_crs(epsg=target_epsg)
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
    )
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
    pipeline: PipelineConfig,
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

    required = [_SOURCE_URBAN_COLUMN, _SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN]
    missing = [col for col in required if col not in result.columns]
    if missing:
        raise ValueError(f"AERMOD source rows are missing required source class columns: {missing}")
    urban_series = pd.to_numeric(result[_SOURCE_URBAN_COLUMN], errors="coerce")
    temporal_series = result[_SOURCE_TEMPORAL_COLUMN].astype("string").str.strip()
    height_series = pd.to_numeric(result[_SOURCE_HEIGHT_COLUMN], errors="coerce")
    invalid = (
        urban_series.isna()
        | temporal_series.isna()
        | temporal_series.eq("")
        | height_series.isna()
        | ~np.isfinite(height_series)
        | height_series.le(0.0)
    )
    if invalid.any():
        sample = result.loc[
            invalid,
            ["nearest_site", _SOURCE_URBAN_COLUMN, _SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN],
        ].head(10).to_dict(orient="records")
        raise ValueError(f"AERMOD source rows have invalid source class values: sample={sample}")
    result["pattern_key_raw"] = (
        result["nearest_site"].astype(str)
        + "__"
        + urban_series.astype("Int64").astype(str)
        + "__"
        + temporal_series.astype(str)
        + "__"
        + height_series.map(lambda v: f"{float(v):g}")
    )
    result["pattern_key"] = result["pattern_key_raw"]
    missing_mask = ~result["pattern_key"].isin(available_pattern_keys)
    if missing_mask.any():
        missing_keys = sorted(result.loc[missing_mask, "pattern_key"].astype(str).unique().tolist())
        raise ValueError(
            "AERMOD ASRV pattern library is missing exact source class patterns. "
            f"Missing keys: {missing_keys[:10]}"
        )

    return result


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
        logger.info("%s reusing cached ASRV kernels from %s", _step_label("3.3"), kernels_path)
        patterns_df = pd.read_parquet(normalized_patterns_path) if normalized_patterns_path.exists() else None
        return _deserialize_kernel_library(kernel_df), patterns_df

    filtered_patterns_df = projected_patterns_df.loc[projected_patterns_df["pattern_key"].isin(requested_pattern_keys)]
    patterns_df = _normalize_projected_asrv_patterns(filtered_patterns_df, grid_size_meters=grid_size_meters)
    kernel_library = _build_kernel_library(patterns_df, grid_size_meters=grid_size_meters)
    kernels_path.parent.mkdir(parents=True, exist_ok=True)
    _serialize_kernel_library(kernel_library).to_parquet(kernels_path, index=False)
    patterns_df.to_parquet(normalized_patterns_path, index=False)
    logger.info("%s cached ASRV kernels at %s", _step_label("3.3"), kernels_path)
    return kernel_library, patterns_df


def _apply_kernels(
    *,
    source_df: pd.DataFrame,
    target_index: pd.DataFrame,
    target_id_col: str,
    emissions_cols: list[str],
    kernel_library: dict[str, _Kernel],
) -> pd.DataFrame:
    def _output_col(emissions_col: str) -> str:
        pollutant = emissions_col.removeprefix("tons_per_year_").removesuffix("_aermod_allocated")
        return _POLLUTANT_TO_CONCENTRATION_COLUMN.get(pollutant, f"concentration_{pollutant}")

    output_cols = [_output_col(col) for col in emissions_cols]
    target_ids = target_index[target_id_col].to_numpy(dtype=np.int64)
    target_ix = target_index["target_ix"].to_numpy(dtype=np.int64)
    target_iy = target_index["target_iy"].to_numpy(dtype=np.int64)

    ix_min = int(target_ix.min())
    iy_min = int(target_iy.min())
    nx = int(target_ix.max()) - ix_min + 1
    ny = int(target_iy.max()) - iy_min + 1

    result_arrays = {col: np.zeros(len(target_index), dtype=np.float64) for col in output_cols}
    support_arrays = {col: np.zeros(len(target_index), dtype=bool) for col in output_cols}

    for pattern_key, source_group in source_df.groupby("pattern_key", dropna=False):
        kernel = kernel_library.get(str(pattern_key))
        if kernel is None or kernel["response_per_ton"].size == 0:
            raise ValueError(f"No AERMOD ASRV kernel available for pattern {pattern_key}")

        dix = kernel["dix"].astype(np.int64)
        diy = kernel["diy"].astype(np.int64)
        dix_max = int(np.abs(dix).max())
        diy_max = int(np.abs(diy).max())

        # Build 2D kernel array centred at (dix_max, diy_max)
        kernel_2d = np.zeros((2 * dix_max + 1, 2 * diy_max + 1), dtype=np.float64)
        kernel_2d[dix + dix_max, diy + diy_max] = kernel["response_per_ton"]
        kernel_support_2d = np.zeros((2 * dix_max + 1, 2 * diy_max + 1), dtype=np.float64)
        kernel_support_2d[dix + dix_max, diy + diy_max] = 1.0

        # Source grid positions
        si = source_group["source_ix"].to_numpy(dtype=np.int64) - ix_min
        sj = source_group["source_iy"].to_numpy(dtype=np.int64) - iy_min
        valid_src = (si >= 0) & (si < nx) & (sj >= 0) & (sj < ny)

        # Target positions in the fftconvolve 'full' output array
        # output[i,j] accumulates contributions to receptor at (ix_min + i - dix_max, iy_min + j - diy_max)
        ti = target_ix - ix_min + dix_max
        tj = target_iy - iy_min + diy_max
        in_bounds = (ti >= 0) & (ti < nx + 2 * dix_max) & (tj >= 0) & (tj < ny + 2 * diy_max)

        for emis_col, out_col in zip(emissions_cols, output_cols):
            emissions_grid = np.zeros((nx, ny), dtype=np.float64)
            source_values = source_group[emis_col].to_numpy(dtype=np.float64)[valid_src]
            np.add.at(
                emissions_grid,
                (si[valid_src], sj[valid_src]),
                source_values,
            )
            conc_full = fftconvolve(emissions_grid, kernel_2d, mode="full")
            result_arrays[out_col][in_bounds] += conc_full[ti[in_bounds], tj[in_bounds]]
            source_support_grid = np.zeros((nx, ny), dtype=np.float64)
            positive_src = source_values > 0.0
            if positive_src.any():
                np.add.at(
                    source_support_grid,
                    (si[valid_src][positive_src], sj[valid_src][positive_src]),
                    1.0,
                )
                support_full = fftconvolve(source_support_grid, kernel_support_2d, mode="full")
                # Support is an integer-count convolution. FFT roundoff can leave tiny
                # non-zero values far from any modeled source, so round back to the
                # intended counts before thresholding.
                support_arrays[out_col][in_bounds] |= np.rint(support_full[ti[in_bounds], tj[in_bounds]]) > 0

    if not result_arrays:
        return pd.DataFrame(columns=[target_id_col])
    concentrations = pd.DataFrame({target_id_col: target_ids})
    for name, values in result_arrays.items():
        support_col = _AERMOD_SUPPORT_COLUMNS.get(name)
        if support_col:
            supported = support_arrays[name]
            concentrations[name] = np.where(supported, values, 0.0)
            concentrations[support_col] = supported
        else:
            concentrations[name] = values
    # TotalPM25 is primary PM2.5 dispersion only. BC is a separate primary pollutant and
    # is NOT included here — secondary species come from InMAP and are merged in step 4.
    if "PrimaryPM25" in concentrations.columns:
        concentrations["TotalPM25"] = concentrations["PrimaryPM25"]
    return concentrations.reset_index(drop=True)


def _compute_no2_from_isrm_matrix(
    *,
    concentrations_gdf: gpd.GeoDataFrame,
    target_id_col: str,
    isrm_matrix_path: str,
) -> gpd.GeoDataFrame:
    """Convert attached AERMOD-dispersed NOx concentration to NO2 using ISRM column-sum ratios.

    ratio[j] = sum_i M[i,j]  (column sum of the sparse NOx→NO2 ISRM matrix)
    NO2_100m[r] = NOx_conc_kernel[r] × ratio[inmap_cell_id(r)]

    Step 3 attaches the raw AERMOD-dispersed NOx field to the full target grid first.
    We then post-multiply only cells explicitly marked as AERMOD-backed via has_aermod_no2.
    """
    if "NO2" not in concentrations_gdf.columns:
        logger.warning("%s NO2 column not found in concentrations; skipping ISRM ratio step", _step_label("3.5"))
        return concentrations_gdf

    if "inmap_cell_id" not in concentrations_gdf.columns:
        logger.warning("%s target_grid missing inmap_cell_id; skipping ISRM ratio step", _step_label("3.5"))
        return concentrations_gdf

    if "has_aermod_no2" not in concentrations_gdf.columns:
        logger.warning("%s has_aermod_no2 mask missing from concentrations; skipping ISRM ratio step", _step_label("3.5"))
        return concentrations_gdf

    data = np.load(isrm_matrix_path)
    receptor_ids = data["receptor_ids"].astype(np.int64)
    values = data["values"].astype(np.float64)
    receptor_dim = int(data["receptor_dim"]) if "receptor_dim" in data else int(receptor_ids.max()) + 1

    # Column sums: ratio[j] = Σ_i M[i,j] = NO2/NOx conversion factor for ISRM cell j
    col_ratio = np.zeros(receptor_dim, dtype=np.float64)
    np.add.at(col_ratio, receptor_ids, values)
    col_ratio = np.asarray(col_ratio, dtype=np.float64)

    target_ids = concentrations_gdf[target_id_col].to_numpy()
    inmap_ids = pd.to_numeric(concentrations_gdf["inmap_cell_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    valid = (inmap_ids >= 0) & (inmap_ids < receptor_dim)
    aermod_backed = concentrations_gdf["has_aermod_no2"].fillna(False).astype(bool).to_numpy()
    apply_mask = valid & aermod_backed
    cell_ratio = np.zeros(len(target_ids), dtype=np.float64)
    cell_ratio[apply_mask] = col_ratio[inmap_ids[apply_mask]]
    cell_ratio = np.asarray(cell_ratio, dtype=np.float64)

    result = concentrations_gdf.copy()
    result["NO2"] = np.multiply(
        pd.to_numeric(result["NO2"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        cell_ratio,
        dtype=np.float64,
    )
    logger.info(
        "%s NO2 computed from ISRM column-sum ratios: %d / %d cells have ratio > 0",
        _step_label("3.5"),
        int((cell_ratio > 0).sum()),
        len(target_ids),
    )
    return result


def _write_outputs(gdf: gpd.GeoDataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(output_path, index=False)
    gdf.to_file(output_path.with_suffix(".gpkg"), driver="GPKG")


def _validate_aermod_concentrations_schema(
    concentrations_df: pd.DataFrame,
    *,
    target_id_col: str,
) -> None:
    if target_id_col not in concentrations_df.columns:
        raise ValueError(
            "AERMOD concentrations table is missing its target id column. "
            f"Expected '{target_id_col}'."
        )
    target_ids = pd.Series(concentrations_df[target_id_col])
    duplicate_ids = target_ids.loc[target_ids.notna() & target_ids.duplicated(keep=False)].drop_duplicates().tolist()
    if duplicate_ids:
        raise ValueError(
            "AERMOD concentrations table must contain at most one row per "
            f"{target_id_col}; duplicate ids: {duplicate_ids[:10]}"
        )

    concentration_cols = [
        col for col in concentrations_df.columns
        if col != target_id_col and col not in _AERMOD_SUPPORT_COLUMN_SET
    ]
    for col in concentration_cols:
        if not pd.api.types.is_numeric_dtype(concentrations_df[col]):
            raise TypeError(
                "AERMOD concentration column has a non-numeric dtype. "
                f"Column '{col}' has dtype {concentrations_df[col].dtype}."
            )

    for col in _AERMOD_SUPPORT_COLUMN_SET.intersection(concentrations_df.columns):
        if pd.api.types.is_numeric_dtype(concentrations_df[col]):
            values = pd.Series(concentrations_df[col]).dropna()
            invalid = values[~values.isin([0, 1, 0.0, 1.0, True, False])]
            if not invalid.empty:
                raise TypeError(
                    "AERMOD support mask contains numeric values outside boolean 0/1. "
                    f"Column '{col}' sample_invalid={invalid.head(5).tolist()}."
                )


def _attach_concentrations(
    *,
    target_grid: gpd.GeoDataFrame,
    concentrations_df: pd.DataFrame,
    target_id_col: str,
) -> gpd.GeoDataFrame:
    _validate_aermod_concentrations_schema(concentrations_df, target_id_col=target_id_col)
    result = target_grid.copy()
    concentration_cols = [col for col in concentrations_df.columns if col != target_id_col]
    if not concentration_cols:
        return result
    lookup = concentrations_df.set_index(target_id_col)
    target_ids = result[target_id_col].to_numpy()
    for col in concentration_cols:
        if col in _AERMOD_SUPPORT_COLUMN_SET:
            result[col] = lookup[col].reindex(target_ids, fill_value=False).to_numpy(dtype=bool)
        else:
            result[col] = lookup[col].reindex(target_ids, fill_value=0.0).to_numpy(dtype=np.float64)
    return gpd.GeoDataFrame(result, geometry="geometry", crs=target_grid.crs)


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    cache_dir: Optional[Path] = None,
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
    source_id_col = _resolve_source_grid_id_column(emissions_gdf)
    target_id_col = _resolve_target_grid_id_column(pipeline, target_grid)
    emissions_cols = _emissions_columns(emissions_gdf, pipeline)
    if emissions_input_gdf is None:
        source_class_cols = [
            col
            for col in (_SOURCE_TEMPORAL_COLUMN, _SOURCE_HEIGHT_COLUMN, _SOURCE_URBAN_COLUMN)
            if col in emissions_gdf.columns
        ]
        keep_cols = [source_id_col] + source_class_cols + emissions_cols
        emissions_gdf = emissions_gdf[keep_cols + ["geometry"]]
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
        outputs_dir=raw_dir,
    )
    logger.info(
        "%s prepared %d source cells over %d target cells at %.2fm grid spacing",
        _step_label("3.1"),
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
        pipeline=pipeline,
        site_reference=site_reference,
        available_pattern_keys=available_pattern_keys,
    )
    requested_pattern_keys = sorted(set(source_df["pattern_key"].dropna().astype(str).tolist()))
    effective_cache_dir = cache_dir if cache_dir is not None else raw_dir
    effective_cache_dir.mkdir(parents=True, exist_ok=True)
    kernel_library, patterns_df = _load_or_build_kernel_library(
        raw_dir=effective_cache_dir,
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
        _step_label("3.3"),
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

    result_gdf = _attach_concentrations(
        target_grid=target_grid,
        concentrations_df=concentrations_df,
        target_id_col=target_id_col,
    )

    log_substep_banner("3.5", "compute NO2 from ISRM NO2/NOx column-sum ratios", logger=logger)
    _no2_matrix_path = pipeline.isrm_nox_to_no2_ratios_file
    if _no2_matrix_path and "NO2" in result_gdf.columns:
        result_gdf = _compute_no2_from_isrm_matrix(
            concentrations_gdf=result_gdf,
            target_id_col=target_id_col,
            isrm_matrix_path=_no2_matrix_path,
        )
        _trace_frame("5", "aermod_concentrations_with_isrm_no2", pd.DataFrame(result_gdf.drop(columns="geometry", errors="ignore")), key_cols=[target_id_col])
    elif "NO2" in result_gdf.columns:
        logger.info(
            "%s no ISRM NO2/NOx matrix configured (isrm_nox_to_no2_ratios_file); "
            "NO2 column will not be produced",
            _step_label("3.5"),
        )
        result_gdf = result_gdf.drop(columns=["NO2", "has_aermod_no2"], errors="ignore")
    else:
        logger.info("%s NO2 column absent from kernel output; skipping ratio step", _step_label("3.5"))

    log_substep_banner("3.6", "write concentration outputs", logger=logger)
    output_path = raw_dir / "beam_aermod_concentrations.parquet"
    _write_outputs(result_gdf, output_path)
    logger.info("%s AERMOD concentrations → %s", _step_label("3.6"), output_path)
    return result_gdf, target_grid[target_id_col].to_numpy(dtype=int), output_path
