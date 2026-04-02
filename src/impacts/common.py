"""Shared maintained helpers for IMPACTS preprocess, settings-driven workflow, and postprocess code.

This module is intentionally limited to cross-cutting helpers that are reused by
multiple maintained modules. It is organized by concern:

1. Config and local-path validation
2. Input discovery helpers
3. Table/vector IO
4. Preprocess staging and grid preparation
5. Shared table-shaping helpers
6. Skims preparation and annualization
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
import time
from typing import Any
from typing import Dict
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from tqdm import tqdm

from .config.defaults import annualization_days as default_annualization_days
from .config.defaults import chunk_size as default_chunk_size
from .config.defaults import grams_per_short_ton
from .config.defaults import meters_per_mile as _METERS_PER_MILE
from .config.defaults import pollutants as default_prepared_pollutants
from .consist_artifacts import log_input_reference
from .consist_artifacts import resolve_logged_path
from .manifest.file_ops import file_entry
from .manifest.file_ops import is_remote_path
from .manifest.file_ops import parquet_available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config and local-path validation
# ---------------------------------------------------------------------------


def _normalized_stage_label(label: str, *, logger: logging.Logger, is_substep: bool = False) -> str:
    text = str(label).strip()
    upper = text.upper()
    if upper.startswith("PREPROCESS STEP"):
        return upper
    if upper.startswith("STEP"):
        return f"WORKFLOW {upper}"
    if is_substep:
        prefix = "PREPROCESS STEP" if ".preprocessing." in logger.name else "WORKFLOW STEP"
        return f"{prefix} {text}"
    return text.upper()


def log_step_banner(step_label: str, title: str, *, logger: logging.Logger) -> None:
    logger.info("")
    logger.info("========== %s: %s ==========", _normalized_stage_label(step_label, logger=logger), title.upper())


def log_substep_banner(substep_label: str, title: str, *, logger: logging.Logger) -> None:
    logger.info("----- %s: %s -----", _normalized_stage_label(substep_label, logger=logger, is_substep=True), title)

def parse_epsg(value: Any) -> int:
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(
            "geography.local_crs (output EPSG) must be set in the settings file. "
            "Example: local_crs: 26910"
        )
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if ":" in text:
        _, _, suffix = text.rpartition(":")
        text = suffix
    return int(text)


def required_local_path(path: Optional[str], label: str) -> str:
    if not path:
        raise ValueError(f"Missing required config path: {label}")
    if is_remote_path(path):
        raise ValueError(f"{label} must be a local path during preprocessing: {path}")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return str(resolved)


def optional_local_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if is_remote_path(path):
        return path
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Configured path not found: {path}")
    return str(resolved)


# ---------------------------------------------------------------------------
# Input discovery
# ---------------------------------------------------------------------------

def find_first_matching(root: str, pattern: str) -> Optional[str]:
    if not root or is_remote_path(root):
        return None
    path = Path(root)
    if not path.exists():
        return None
    if path.is_file():
        return str(path)
    matches = sorted(candidate for candidate in path.glob(pattern) if candidate.is_file())
    if matches:
        return str(matches[0])
    recursive_matches = sorted(candidate for candidate in path.rglob(pattern) if candidate.is_file())
    if recursive_matches:
        return str(recursive_matches[0])
    return None


def find_preferred_file(root: str, names: list[str]) -> Optional[str]:
    path = Path(root)
    if not path.exists():
        return None
    for name in names:
        direct = path / name
        if direct.exists():
            return str(direct)
    for name in names:
        matches = sorted(candidate for candidate in path.rglob(name) if candidate.is_file())
        if matches:
            return str(matches[0])
    return None


def find_latest_iters_dir(root: str) -> Optional[Path]:
    path = Path(root)
    if not path.exists():
        return None
    candidates = [candidate for candidate in path.rglob("ITERS") if candidate.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.stat().st_mtime)


def find_latest_iteration_events(iters_dir: Path) -> Optional[Path]:
    patterns = [
        ("*.events.parquet", r"/it\.(\d+)/(\d+)\.events\.parquet$"),
        ("*.events.csv.gz",  r"/it\.(\d+)/(\d+)\.events\.csv\.gz$"),
    ]
    latest_iter = -1
    latest_events: Optional[Path] = None
    for glob_pattern, regex_pattern in patterns:
        for candidate in iters_dir.glob(f"it.*/{glob_pattern}"):
            match = re.search(regex_pattern, str(candidate))
            if not match:
                continue
            dir_iter = int(match.group(1))
            file_iter = int(match.group(2))
            if dir_iter != file_iter:
                continue
            if dir_iter > latest_iter:
                latest_iter = dir_iter
                latest_events = candidate
    return latest_events


def find_latest_iteration_skims(iters_dir: Path) -> Optional[Path]:
    patterns = [
        ("*.skimsEmissionsTotals.csv.gz", r"/it\.(\d+)/(\d+)\.skimsEmissionsTotals\.csv\.gz$"),
        ("*.skimsEmissions.csv.gz", r"/it\.(\d+)/(\d+)\.skimsEmissions\.csv\.gz$"),
        ("*.skimsEmissions.parquet", r"/it\.(\d+)/(\d+)\.skimsEmissions\.parquet$"),
    ]
    latest_iter = -1
    latest_skims: Optional[Path] = None
    for glob_pattern, regex_pattern in patterns:
        for candidate in iters_dir.glob(f"it.*/{glob_pattern}"):
            match = re.search(regex_pattern, str(candidate))
            if not match:
                continue
            dir_iter = int(match.group(1))
            file_iter = int(match.group(2))
            if dir_iter != file_iter:
                continue
            if dir_iter > latest_iter:
                latest_iter = dir_iter
                latest_skims = candidate
    return latest_skims


def resolve_beam_network_local_path(root: str) -> str:
    resolved = required_local_path(root, "BEAM output root")
    network_path = find_preferred_file(resolved, ["network.csv.gz", "network.parquet"])
    if network_path:
        return network_path
    latest_iters_dir = find_latest_iters_dir(resolved)
    if latest_iters_dir:
        run_root = latest_iters_dir.parent
        for candidate in [run_root / "network.csv.gz", run_root / "network.parquet"]:
            if candidate.exists():
                return str(candidate)
    raise FileNotFoundError(f"No BEAM network file found under configured BEAM output root: {resolved}")


def resolve_emissions_skims_local_path(root: str) -> str:
    resolved = required_local_path(root, "BEAM output root")
    latest_iters_dir = find_latest_iters_dir(resolved)
    if latest_iters_dir:
        latest_skims = find_latest_iteration_skims(latest_iters_dir)
        if latest_skims:
            return str(latest_skims)
    for pattern in ("*.skimsEmissionsTotals*.csv.gz", "*.skimsEmissions*.csv.gz", "*.skimsEmissions*.parquet"):
        match = find_first_matching(resolved, pattern)
        if match:
            return match
    raise FileNotFoundError(f"No BEAM skims emissions file found under configured BEAM output root: {resolved}")


def resolve_latest_events_local_path(root: str) -> Optional[str]:
    """Return the latest events file under root, or None if not found."""
    resolved = Path(root)
    if not resolved.exists():
        return None
    latest_iters_dir = find_latest_iters_dir(str(resolved))
    if latest_iters_dir:
        latest_events = find_latest_iteration_events(latest_iters_dir)
        if latest_events:
            return str(latest_events)
    return None


def resolve_osm_pbf_local_path(path: Optional[str]) -> Optional[str]:
    resolved = optional_local_path(path)
    if not resolved or is_remote_path(resolved):
        return resolved
    resolved_path = Path(resolved)
    if resolved_path.is_file():
        return str(resolved_path)
    match = find_first_matching(str(resolved_path), "*.osm.pbf")
    if match:
        return match
    raise FileNotFoundError(f"No .osm.pbf file found under configured path: {resolved}")


def infer_vector_epsg(path: Optional[str]) -> Optional[int]:
    if not path or is_remote_path(path):
        return None
    target = Path(path)
    if not target.exists():
        return None
    try:
        if target.suffix.lower() == ".parquet":
            gdf = gpd.read_parquet(target)
        else:
            gdf = gpd.read_file(target)
    except Exception:
        return None
    if gdf.crs is None:
        return None
    try:
        return gdf.crs.to_epsg()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table and vector IO
# ---------------------------------------------------------------------------

def read_vector(path: str) -> gpd.GeoDataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return gpd.read_parquet(target)
    return gpd.read_file(target)


def write_vector(gdf: gpd.GeoDataFrame, path: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        gdf.to_parquet(target, index=False)
    else:
        gdf.to_file(target)


def read_network_lines(path: str, *, target_epsg: Optional[int] = None) -> gpd.GeoDataFrame:
    target = Path(path)
    lower = target.name.lower()
    line_cols = ["fromLocationX", "fromLocationY", "toLocationX", "toLocationY"]
    if lower.endswith(".csv.gz"):
        network = pd.read_csv(target, compression="gzip", usecols=line_cols)
        geometry = [
            LineString([(fx, fy), (tx, ty)])
            for fx, fy, tx, ty in zip(
                pd.to_numeric(network["fromLocationX"], errors="coerce"),
                pd.to_numeric(network["fromLocationY"], errors="coerce"),
                pd.to_numeric(network["toLocationX"], errors="coerce"),
                pd.to_numeric(network["toLocationY"], errors="coerce"),
            )
        ]
        gdf = gpd.GeoDataFrame(network, geometry=geometry, crs=f"EPSG:{int(target_epsg)}" if target_epsg else None)
    elif lower.endswith(".csv"):
        network = pd.read_csv(target, usecols=line_cols)
        geometry = [
            LineString([(fx, fy), (tx, ty)])
            for fx, fy, tx, ty in zip(
                pd.to_numeric(network["fromLocationX"], errors="coerce"),
                pd.to_numeric(network["fromLocationY"], errors="coerce"),
                pd.to_numeric(network["toLocationX"], errors="coerce"),
                pd.to_numeric(network["toLocationY"], errors="coerce"),
            )
        ]
        gdf = gpd.GeoDataFrame(network, geometry=geometry, crs=f"EPSG:{int(target_epsg)}" if target_epsg else None)
    else:
        gdf = read_vector(path)
        if target_epsg is not None and gdf.crs is not None:
            gdf = gdf.to_crs(epsg=int(target_epsg))
        return gdf

    if gdf.crs is None:
        raise ValueError(f"Could not determine CRS for network lines from {path}")
    if target_epsg is not None:
        gdf = gdf.to_crs(epsg=int(target_epsg))
    return gdf.loc[gdf.geometry.notna()].copy()


# ---------------------------------------------------------------------------
# Preprocess staging and grid preparation
# ---------------------------------------------------------------------------

def generate_fishnet_from_bounds(
    *,
    bounds: tuple[float, float, float, float],
    mask_gdf: gpd.GeoDataFrame,
    cell_size: float,
    target_path: str,
    target_epsg: int,
    cell_id_col: str,
) -> tuple[str, str]:
    target = Path(target_path)
    if target.exists():
        target.unlink()
    started = time.perf_counter()
    minx, miny, maxx, maxy = (float(v) for v in bounds)
    start_x = cell_size * int(minx // cell_size)
    start_y = cell_size * int(miny // cell_size)
    end_x = cell_size * int(-(-maxx // cell_size))
    end_y = cell_size * int(-(-maxy // cell_size))

    xs = np.arange(start_x, end_x, cell_size, dtype=float)
    ys = np.arange(start_y, end_y, cell_size, dtype=float)
    total_cells = int(len(xs) * len(ys))
    logger.info(
        "Preprocess: generating AERMOD fishnet candidates at %.0fm resolution over %d x %d cells (%d total)",
        cell_size,
        len(xs),
        len(ys),
        total_cells,
    )

    x_grid, y_grid = np.meshgrid(xs, ys)
    x0 = x_grid.ravel()
    y0 = y_grid.ravel()
    geometries = shapely.box(x0, y0, x0 + cell_size, y0 + cell_size)

    fishnet = gpd.GeoDataFrame(
        {cell_id_col: np.arange(total_cells, dtype=int)},
        geometry=geometries,
        crs=f"EPSG:{int(target_epsg)}",
    )

    mask_gdf = mask_gdf.to_crs(epsg=target_epsg)
    mask_union = mask_gdf.geometry.union_all() if hasattr(mask_gdf.geometry, "union_all") else mask_gdf.geometry.unary_union
    filter_started = time.perf_counter()
    keep_mask = np.zeros(len(fishnet), dtype=bool)
    progress = tqdm(
        total=len(fishnet),
        desc="Filtering AERMOD fishnet",
        unit="cell",
        dynamic_ncols=True,
        leave=True,
    )
    try:
        for start in range(0, len(fishnet), default_chunk_size):
            stop = min(start + default_chunk_size, len(fishnet))
            keep_mask[start:stop] = fishnet.geometry.iloc[start:stop].intersects(mask_union).to_numpy()
            progress.update(stop - start)
    finally:
        progress.close()
    fishnet = fishnet.loc[keep_mask].copy()
    logger.info(
        "Preprocess: filtered fishnet candidates to the staged InMAP footprint in %.2fs → %d rows kept",
        time.perf_counter() - filter_started,
        len(fishnet),
    )
    write_vector(fishnet, target_path)
    logger.info(
        "Preprocess: wrote generated AERMOD fishnet in %.2fs → %d rows at %s",
        time.perf_counter() - started,
        len(fishnet),
        target_path,
    )
    return target_path, cell_id_col


def constrain_grid_to_network(
    *,
    grid_path: str,
    network_path: str,
    grid_id_col: str,
    target_epsg: int,
    output_path: str,
) -> str:
    started = time.perf_counter()
    target = Path(output_path)
    if target.exists():
        logger.info("Preprocess: reusing network-constrained grid %s", target)
        return str(target)

    grid = read_vector(grid_path)
    if grid.crs is None:
        raise ValueError(f"Grid constraint input is missing CRS: {grid_path}")
    grid = grid.to_crs(epsg=int(target_epsg))
    if grid_id_col not in grid.columns:
        raise ValueError(
            f"Expected grid id column '{grid_id_col}' in {grid_path}. Available columns: {list(grid.columns)}"
        )

    network = read_network_lines(network_path, target_epsg=int(target_epsg))
    if network.crs is None:
        raise ValueError(f"Network constraint input is missing CRS: {network_path}")

    joined = gpd.sjoin(
        grid[[grid_id_col, "geometry"]],
        network[["geometry"]],
        how="inner",
        predicate="intersects",
    )
    keep_ids = (
        pd.to_numeric(joined[grid_id_col], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    constrained = grid[pd.to_numeric(grid[grid_id_col], errors="coerce").isin(keep_ids)].copy()
    write_vector(constrained, str(target))
    logger.info(
        "Preprocess: constrained %s to %d network-intersecting cells in %.2fs → %s",
        grid_id_col,
        len(constrained),
        time.perf_counter() - started,
        target,
    )
    return str(target)


def assign_grid_cells_to_zones(
    *,
    grid_path: str,
    zone_path: str,
    zone_id_col: str,
    output_col: str,
    target_epsg: int,
) -> str:
    grid = read_vector(grid_path)
    zones = read_vector(zone_path)
    if zone_id_col not in zones.columns:
        raise ValueError(
            f"Zone mapping expected column '{zone_id_col}' in {zone_path}. "
            f"Available columns: {list(zones.columns)}"
        )
    if grid.crs is None:
        raise ValueError(f"Grid mapping input is missing CRS: {grid_path}")
    if zones.crs is None:
        raise ValueError(f"Zone mapping input is missing CRS: {zone_path}")
    grid = grid.to_crs(epsg=target_epsg)
    zones = zones.to_crs(epsg=target_epsg)

    grid_index_col = grid.index.name or "grid_index"
    zone_lookup = zones[[zone_id_col, "geometry"]].copy()
    zone_lookup = zone_lookup.rename(columns={"geometry": "_zone_geometry"})

    representative_points = gpd.GeoDataFrame(
        {grid_index_col: grid.index.to_numpy(dtype=int)},
        geometry=grid.geometry.representative_point(),
        crs=grid.crs,
    )
    joined = gpd.sjoin(
        representative_points,
        zones[[zone_id_col, "geometry"]],
        how="left",
        predicate="within",
    )
    duplicate_matches = joined.duplicated(subset=[grid_index_col], keep=False)
    unique_point_matches = joined.loc[~duplicate_matches, [grid_index_col, zone_id_col]].copy()
    mapped = unique_point_matches.rename(columns={zone_id_col: output_col})
    mapped[output_col] = pd.to_numeric(mapped[output_col], errors="coerce")
    unresolved_ids = set()
    unresolved_ids.update(
        mapped.loc[mapped[output_col].isna(), grid_index_col].astype(int).tolist()
    )
    if duplicate_matches.any():
        unresolved_ids.update(joined.loc[duplicate_matches, grid_index_col].astype(int).tolist())

    if unresolved_ids:
        overlap_grid = (
            grid.loc[grid.index.isin(sorted(unresolved_ids)), ["geometry"]]
            .reset_index()
            .rename(columns={grid.index.name or "index": grid_index_col})
        )
        overlap_candidates = gpd.sjoin(
            overlap_grid,
            zones[[zone_id_col, "geometry"]],
            how="left",
            predicate="intersects",
        )
        if not overlap_candidates.empty:
            overlap_candidates = overlap_candidates.merge(
                zone_lookup.reset_index().rename(columns={"index": "zone_index"}),
                how="left",
                left_on="index_right",
                right_on="zone_index",
            )
            overlap_zone_col = zone_id_col
            if overlap_zone_col not in overlap_candidates.columns:
                for candidate in (f"{zone_id_col}_x", f"{zone_id_col}_y"):
                    if candidate in overlap_candidates.columns:
                        overlap_zone_col = candidate
                        break
            overlap_candidates["_overlap_area"] = overlap_candidates.geometry.intersection(
                overlap_candidates["_zone_geometry"]
            ).area
            overlap_candidates[overlap_zone_col] = pd.to_numeric(overlap_candidates[overlap_zone_col], errors="coerce")
            overlap_candidates = overlap_candidates.sort_values(
                [grid_index_col, "_overlap_area", overlap_zone_col],
                ascending=[True, False, True],
            )
            best_overlap = (
                overlap_candidates.loc[
                    overlap_candidates[overlap_zone_col].notna() & overlap_candidates["_overlap_area"].gt(0),
                    [grid_index_col, overlap_zone_col],
                ]
                .drop_duplicates(subset=[grid_index_col], keep="first")
                .rename(columns={overlap_zone_col: output_col})
            )
            if not best_overlap.empty:
                best_overlap[output_col] = pd.to_numeric(best_overlap[output_col], errors="coerce")
                mapped = mapped.loc[~mapped[grid_index_col].isin(best_overlap[grid_index_col])].copy()
                mapped = pd.concat([mapped, best_overlap], ignore_index=True)
    unresolved_duplicates = mapped.duplicated(subset=[grid_index_col], keep=False)
    if unresolved_duplicates.any():
        sample = (
            mapped.loc[unresolved_duplicates, [grid_index_col, output_col]]
            .head(10)
            .to_dict(orient="records")
        )
        raise ValueError(
            f"Grid-to-zone mapping still produced multiple {output_col} matches for some cells after overlap resolution. "
            f"sample={sample}"
        )
    unmatched = mapped[output_col].isna()
    if unmatched.any():
        logger.info(
            "Preprocess: dropping %d generated AERMOD cells that do not map to a corresponding InMAP cell",
            int(unmatched.sum()),
        )
        mapped = mapped.loc[~unmatched].copy()
    mapped[output_col] = mapped[output_col].astype(int)
    grid = grid.merge(mapped, how="inner", left_index=True, right_on=grid_index_col).drop(columns=[grid_index_col])
    write_vector(grid, grid_path)
    logger.info(
        "Preprocess: assigned %s to %d generated AERMOD cells → %s",
        output_col,
        len(grid),
        grid_path,
    )
    return grid_path


def ensure_grid_cell_id(
    staged_path: str,
    cell_id_col: str,
    source_col: Optional[str] = None,
    output_path: Optional[str] = None,
) -> tuple[str, str]:
    gdf = read_vector(staged_path)
    if cell_id_col in gdf.columns:
        return staged_path, cell_id_col
    if source_col:
        if source_col not in gdf.columns:
            raise ValueError(
                f"Configured grid_id column '{source_col}' not found in {staged_path}. "
                f"Available columns: {list(gdf.columns)}"
            )
        gdf[cell_id_col] = gdf[source_col]
    else:
        gdf[cell_id_col] = range(len(gdf))
    normalized_path = output_path or staged_path
    if Path(normalized_path).suffix.lower() != ".parquet":
        normalized_path = str(Path(normalized_path).with_suffix(".parquet"))
    write_vector(gdf, normalized_path)
    return normalized_path, cell_id_col


def register_local_input(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    key: str,
    source_path: str,
    relative_target: str,
    optional: bool = False,
) -> str:
    source = str(Path(source_path).resolve())
    manifest_inputs[key] = file_entry(
        kind="local",
        path=source,
        staged_path=source,
        optional=optional,
    )
    return source


def register_managed_input(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    key: str,
    source_path: str,
    relative_target: str,
    artifact_key: Optional[str] = None,
    optional: bool = False,
    prefer_reference: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    if prefer_reference:
        logged_entry = log_input_reference(
            key=key,
            source_path=source_path,
            artifact_key=artifact_key,
            optional=optional,
            metadata=metadata,
        )
        if logged_entry is not None:
            manifest_inputs[key] = logged_entry
            return resolve_logged_path(logged_entry)
    return register_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key=key,
        source_path=source_path,
        relative_target=relative_target,
        optional=optional,
    )


def register_optional_input(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    key: str,
    source_path: Optional[str],
    relative_target: str,
) -> Optional[str]:
    if not source_path:
        return None
    if is_remote_path(source_path):
        manifest_inputs[key] = {
            "kind": "remote",
            "source_path": source_path,
            "staged_path": None,
            "optional": True,
            "exists": True,
        }
        return source_path
    return register_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key=key,
        source_path=source_path,
        relative_target=relative_target,
        optional=True,
    )


def resolve_manifest_input_path(entry: Dict[str, Any], *, label: str) -> str:
    try:
        return resolve_logged_path(entry)
    except Exception as exc:
        raise ValueError(f"Could not resolve input path for {label}") from exc


def prepared_table_target(input_root: Path, stem: str) -> Path:
    suffix = ".parquet" if parquet_available() else ".csv.gz"
    return input_root / "skims" / f"{stem}{suffix}"


# ---------------------------------------------------------------------------
# Shared table-shaping helpers
# ---------------------------------------------------------------------------

def read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(target)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(target, compression="gzip")
    if lower.endswith(".csv"):
        return pd.read_csv(target)
    raise ValueError(f"Unsupported table format: {target}")


def first_existing(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_county_fips(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)")[0]
    normalized = extracted.where(extracted.isna(), extracted.str.zfill(3))
    return normalized.astype("string")


def resolve_column_config(
    config: Optional[Dict[str, str]],
    defaults: Dict[str, str],
) -> Dict[str, str]:
    resolved = defaults.copy()
    if config:
        resolved.update({k: v for k, v in config.items() if v})
    return resolved


def parse_emissions_string(emissions: str) -> Dict[str, float]:
    if emissions is None or (isinstance(emissions, float) and pd.isna(emissions)):
        return {}
    txt = str(emissions).strip()
    if not txt:
        return {}
    out: Dict[str, float] = {}
    for part in txt.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        key, value = part.split(":", 1)
        try:
            out[key.strip()] = float(value)
        except ValueError:
            continue
    return out


def read_skims_emissions(
    path: str,
    pollutants: Optional[list[str]] = None,
    pollutants_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    dim_cols = ["hour", "linkId", "vehicleTypeId", "process", "travelTimeInSecond", "parkingDurationInSecond"]
    compact_cols = dim_cols + ["emissions", "observations", "iterations"]
    source_pollutants = [pollutants_map.get(p, p) for p in (pollutants or [])] if pollutants_map else (pollutants or [])
    cols = None if pollutants is None else compact_cols + [c for c in source_pollutants if c not in compact_cols]
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        import pyarrow.parquet as pq

        if cols is not None:
            available = set(pq.read_schema(path).names)
            cols = [c for c in cols if c in available] or None
        df = pd.read_parquet(target, columns=cols)
    elif lower.endswith(".csv.gz"):
        if cols is not None:
            header_cols = pd.read_csv(target, compression="gzip", nrows=0).columns.tolist()
            cols = [c for c in cols if c in header_cols] or None
        df = pd.read_csv(target, compression="gzip", usecols=cols if pollutants else None)
    elif lower.endswith(".csv"):
        if cols is not None:
            header_cols = pd.read_csv(target, nrows=0).columns.tolist()
            cols = [c for c in cols if c in header_cols] or None
        df = pd.read_csv(target, usecols=cols if pollutants else None)
    else:
        raise ValueError(f"Unsupported skims format: {target}. Use .csv, .csv.gz, or .parquet")

    if "emissions" not in df.columns:
        return df

    requested_pollutants = list(dict.fromkeys(pollutants or default_prepared_pollutants))
    parsed = df["emissions"].apply(parse_emissions_string)
    observations = pd.to_numeric(df.get("observations", 1.0), errors="coerce").fillna(0.0)
    normalized = df[[c for c in dim_cols if c in df.columns]].copy()
    normalized["observations"] = observations
    for pollutant in requested_pollutants:
        source_pollutant = pollutants_map.get(pollutant, pollutant) if pollutants_map else pollutant
        normalized[pollutant] = (
            parsed.apply(lambda values, p=source_pollutant: float(values.get(p, 0.0))) * observations
        )
    return normalized


def _totals_pollutant_columns(
    df: pd.DataFrame,
    required_pollutants: list[str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for pollutant in required_pollutants:
        if pollutant in df.columns:
            resolved[pollutant] = pollutant
    return resolved


def _group_and_sum_numeric(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    sum_cols: list[str],
    count_nonzero_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    prepared = df[group_cols + sum_cols].copy()
    for col in sum_cols:
        prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)
    grouped = prepared.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    if count_nonzero_cols:
        count_frame = df[group_cols].copy()
        for source_col, output_col in count_nonzero_cols.items():
            values = pd.to_numeric(df.get(source_col, 0.0), errors="coerce").fillna(0.0)
            count_frame[output_col] = values.ne(0).astype(int)
        count_outputs = list(count_nonzero_cols.values())
        counts = count_frame.groupby(group_cols, dropna=False)[count_outputs].sum().reset_index()
        grouped = grouped.merge(counts, on=group_cols, how="left")
    return grouped


# ---------------------------------------------------------------------------
# Skims preparation and annualization
# ---------------------------------------------------------------------------

def prepare_skims_for_grid_allocation(
    skims_path: str,
    output_path: str,
    *,
    group_cols: Optional[list[str]] = None,
    required_pollutants: Optional[list[str]] = None,
    pollutants_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    df = read_skims_emissions(
        skims_path,
        pollutants=required_pollutants,
        pollutants_map=pollutants_map,
    )
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    missing_group_cols = [col for col in prepared_group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Prepared skims missing required grouping columns: {missing_group_cols}")

    required = required_pollutants or default_prepared_pollutants
    totals_cols = _totals_pollutant_columns(df, required)
    prepared = df[prepared_group_cols + list(totals_cols.values())].copy()
    rename_map = {source: pollutant for pollutant, source in totals_cols.items()}
    prepared = prepared.rename(columns=rename_map)
    for pollutant in required:
        if pollutant not in prepared.columns:
            prepared[pollutant] = 0.0
    if "observations" in df.columns:
        prepared["observations"] = pd.to_numeric(df["observations"], errors="coerce").fillna(0.0)
    else:
        prepared["observations"] = 0.0
    aggregated = _group_and_sum_numeric(
        prepared,
        group_cols=prepared_group_cols,
        sum_cols=["observations"] + list(required),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        aggregated.to_parquet(out, index=False)
    elif out.name.lower().endswith(".csv.gz"):
        aggregated.to_csv(out, index=False, compression="gzip")
    else:
        raise ValueError("Prepared skims output must be .parquet or .csv.gz")
    return aggregated


_SKIMS_DIMENSION_COLS = {
    "linkId",
    "vehicleTypeId",
    "process",
    "totTrips",
    "totVMT",
    "roadCategory",
}


def annualize_prepared_skims_for_grid_allocation(
    prepared_skims_path: str,
    output_path: str,
    *,
    network_path: str,
    beam_length_col: str,
    group_cols: Optional[list[str]] = None,
    required_pollutants: Optional[list[str]] = None,
    annualization_days: float = default_annualization_days,
    population_sample: float = 1.0,
) -> pd.DataFrame:
    if annualization_days <= 0:
        raise ValueError(f"Annualization days must be positive, got {annualization_days}")
    if not 0 < population_sample <= 1:
        raise ValueError(f"population_sample must be in the interval (0, 1], got {population_sample}")

    prepared = read_table(prepared_skims_path)
    link_lengths = read_table(network_path)
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    required = required_pollutants or default_prepared_pollutants
    missing_group_cols = [col for col in prepared_group_cols if col not in prepared.columns]
    if missing_group_cols:
        raise ValueError(f"Annualized skims missing required grouping columns: {missing_group_cols}")

    if "linkId" not in link_lengths.columns or beam_length_col not in link_lengths.columns:
        raise ValueError(
            f"Link lengths table must include 'linkId' and '{beam_length_col}'."
        )
    network_cols = ["linkId", beam_length_col]
    if "attributeOrigType" in link_lengths.columns:
        network_cols.append("attributeOrigType")
    link_lengths = link_lengths[network_cols].copy()
    link_lengths[beam_length_col] = pd.to_numeric(link_lengths[beam_length_col], errors="coerce").fillna(0.0)
    prepared = prepared.merge(link_lengths, how="left", on="linkId")
    prepared[beam_length_col] = pd.to_numeric(prepared[beam_length_col], errors="coerce").fillna(0.0)
    if "attributeOrigType" in prepared.columns:
        prepared = prepared.rename(columns={"attributeOrigType": "roadCategory"})
    scale_factor = 1.0 / population_sample

    retained_dim_cols = [col for col in prepared.columns if col in _SKIMS_DIMENSION_COLS and col not in prepared_group_cols]
    out = prepared[prepared_group_cols + retained_dim_cols].copy()
    prepared_observations = pd.to_numeric(prepared.get("observations", 0.0), errors="coerce").fillna(0.0)
    out["totTrips"] = prepared_observations * scale_factor * annualization_days
    out["totVMT"] = out["totTrips"] * prepared[beam_length_col] / _METERS_PER_MILE
    out = out.drop(columns=[col for col in [beam_length_col] if col in out.columns], errors="ignore")
    for pollutant in required:
        values = (
            pd.to_numeric(prepared[pollutant], errors="coerce").fillna(0.0)
            if pollutant in prepared.columns
            else pd.Series(np.zeros(len(prepared), dtype=float), index=prepared.index)
        )
        out[f"tons_per_year_{pollutant}"] = values * scale_factor * annualization_days / grams_per_short_ton

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        out.to_parquet(output, index=False)
    elif output.name.lower().endswith(".csv.gz"):
        out.to_csv(output, index=False, compression="gzip")
    else:
        raise ValueError("Annualized skims output must be .parquet or .csv.gz")
    return out


def stage_county_boundaries(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    state_fips: str,
    county_fips_codes: list[str],
    year: int,
    area_name: str,
    target_epsg: int,
) -> str:
    from osm_chordify.utils.data_collection import collect_geographic_boundaries

    county_gdf = collect_geographic_boundaries(
        state_fips_code=str(state_fips),
        county_fips_codes=[str(code) for code in county_fips_codes],
        year=int(year),
        area_name=str(area_name),
        geo_level="county",
        work_dir=str(input_root / "county"),
        target_epsg=int(target_epsg),
    )
    destination = input_root / "county" / "county_boundaries.gpkg"
    write_vector(county_gdf, str(destination))
    manifest_inputs["county_boundaries"] = file_entry(
        kind="local",
        path=str(destination),
        staged_path=str(destination),
        optional=False,
    )
    return str(destination)
