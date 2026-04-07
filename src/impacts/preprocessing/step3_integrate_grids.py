"""Step 3 (preprocess) — Network mapping and grid intersection helpers."""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional
from typing import Tuple

import geopandas as gpd
import numpy as np
import osm_chordify
import pandas as pd
import shapely.wkb

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_vector
from ..common import write_vector
from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)
_SOURCE_ROW_ID = "source_row_id"
_CANONICAL_INTERSECTION_COLUMNS = [
    "linkId",
    "countyfp",
    "aermod_cell_id",
    "inmap_cell_id",
    "county_zone_edge_proportion",
    "county_edge_link_length_m",
    "county_zone_link_length_m",
    "aermod_zone_edge_proportion",
    "aermod_edge_link_length_m",
    "aermod_zone_link_length_m",
    "inmap_zone_edge_proportion",
    "inmap_edge_link_length_m",
    "inmap_zone_link_length_m",
    "geometry",
]

def _load_geodataframe(path_or_gdf):
    if isinstance(path_or_gdf, gpd.GeoDataFrame):
        return path_or_gdf.copy()
    path = Path(path_or_gdf)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _save_geodataframe(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(out, index=False)
        return
    if suffix == ".geojson":
        gdf.to_file(out, driver="GeoJSON")
        return
    if suffix == ".gpkg":
        gdf.to_file(out, driver="GPKG")
        return
    gdf.to_file(out.with_suffix(".geojson"), driver="GeoJSON")


def _map_beam_network_to_osm(
    osm_path: str,
    network_path: str,
    output_path: str,
    network_osm_id_col: str = "attributeOrigId",
    output_epsg: int | None = None,
):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    mapped = osm_chordify.map_osm_with_beam_network(
        osm_path=osm_path,
        network_path=network_path,
        network_osm_id_col=network_osm_id_col,
        output_path=output_path,
    )
    if output_epsg is not None:
        mapped_gdf = _load_geodataframe(output_path).to_crs(epsg=output_epsg)
        _save_geodataframe(mapped_gdf, output_path)
        return mapped_gdf
    return mapped


def trace_and_filter_void_zone_rows(
    df: pd.DataFrame,
    *,
    zone_id_col: str,
    proportion_col: str,
    context: str,
) -> pd.DataFrame:
    if df.empty:
        logger.info("%s trace zone_intersection empty_result=True", context)
        return df
    if zone_id_col not in df.columns or proportion_col not in df.columns:
        if proportion_col in df.columns and zone_id_col not in df.columns:
            logger.info(
                "%s trace zone_intersection empty_result_without_zone_ids=True available_columns=%s",
                context,
                list(df.columns),
            )
            return df
        raise ValueError(
            f"{context} requires columns '{zone_id_col}' and '{proportion_col}'. "
            f"Available columns: {list(df.columns)}"
        )
    zone_ids = pd.to_numeric(df[zone_id_col], errors="coerce")
    proportions = pd.to_numeric(df[proportion_col], errors="coerce")
    real_hit_mask = zone_ids.notna() & proportions.gt(0)
    void_mask = zone_ids.notna() & ~real_hit_mask
    hit_zone_ids = np.sort(zone_ids.loc[real_hit_mask].astype(int).unique())
    void_zone_ids = np.sort(zone_ids.loc[void_mask].astype(int).unique())
    logger.info(
        "%s trace zone_intersection zone_id_col=%s real_hit_zones=%d void_zones=%d sample_void=%s",
        context,
        zone_id_col,
        int(hit_zone_ids.shape[0]),
        int(void_zone_ids.shape[0]),
        void_zone_ids[:10].tolist(),
    )
    if not void_mask.any():
        return df
    filtered = df.loc[~void_mask].copy()
    logger.info(
        "%s removed %d bbox-only zone rows after exact-intersection screening",
        context,
        int(void_mask.sum()),
    )
    return filtered


def _resolve_source_row_col(df: pd.DataFrame) -> str:
    for candidate in (
        _SOURCE_ROW_ID,
        f"edge_{_SOURCE_ROW_ID}",
    ):
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Expected source row id column '{_SOURCE_ROW_ID}' in columns: {list(df.columns)}")


def _resolve_staged_file(base: Path, names: tuple[str, ...], label: str) -> str:
    for name in names:
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    for name in names:
        matches = sorted(path for path in base.rglob(name) if path.is_file()) if base.exists() else []
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Step 1 could not find any of {names} under staged {label} directory: {base}")


def _resolve_mapped_network_path(input_root: Path) -> str:
    network_dir = input_root / "network"
    direct = network_dir / "beam_osm_mapped.parquet"
    if direct.exists():
        return str(direct)
    matches = sorted(candidate for candidate in network_dir.rglob("beam_osm_mapped.parquet") if candidate.is_file())
    if matches:
        return str(matches[0])
    raise FileNotFoundError(
        f"Step 3 could not find beam_osm_mapped.parquet under staged network directory: {network_dir}"
    )


def _resolve_county_path(input_root: Path) -> str:
    return _resolve_staged_file(
        input_root / "county",
        ("county_boundaries.gpkg", "county_boundaries.parquet", "county_boundaries.geojson", "county_boundaries.shp"),
        "county",
    )


def _resolve_staged_network_path(input_root: Path) -> str:
    network_dir = input_root / "network"
    if network_dir.exists():
        return _resolve_staged_file(
            network_dir,
            ("network.parquet", "network.csv.gz", "network.csv"),
            "network",
        )
    return _resolve_staged_file(
        input_root,
        ("network.parquet", "network.csv.gz", "network.csv"),
        "network",
    )


def _resolve_staged_osm_path(input_root: Path) -> str:
    osm_dir = input_root / "osm"
    search_roots = [osm_dir] if osm_dir.exists() else []
    search_roots.append(input_root)
    for root in search_roots:
        for name in ("network.osm.pbf", "sfbay-cbg5500-weakConn-network.osm.pbf"):
            candidate = root / name
            if candidate.exists():
                return str(candidate)
        matches = sorted(candidate for candidate in root.rglob("*.osm.pbf") if candidate.is_file())
        if matches:
            return str(matches[0])
    raise FileNotFoundError(f"Step 3 could not find any staged .osm.pbf file under: {osm_dir}")


def canonicalize_intersection_schema(df: pd.DataFrame) -> pd.DataFrame:
    canonical = df.copy()
    def _combine(target: str, candidates: tuple[str, ...]) -> None:
        result = None
        for col in candidates:
            if col not in canonical.columns:
                continue
            series = canonical[col]
            result = series if result is None else series.combine_first(result)
        if result is not None:
            canonical[target] = result

    _combine("linkId", ("edge_linkId", "linkId"))
    _combine("countyfp", ("countyfp", "county_COUNTYFP", "zone_COUNTYFP", "COUNTYFP"))
    _combine(
        "county_zone_edge_proportion",
        ("edge_county_zone_edge_proportion", "county_zone_edge_proportion"),
    )
    _combine(
        "county_edge_link_length_m",
        ("edge_county_edge_link_length_m", "county_edge_link_length_m"),
    )
    _combine(
        "county_zone_link_length_m",
        ("edge_county_zone_link_length_m", "county_zone_link_length_m"),
    )
    _combine("aermod_cell_id", ("edge_aermod_cell_id", "aermod_aermod_cell_id", "aermod_cell_id"))
    _combine("inmap_cell_id", ("edge_inmap_cell_id", "edge_inmap_inmap_cell_id", "aermod_inmap_cell_id", "inmap_cell_id"))
    _combine(
        "aermod_zone_edge_proportion",
        ("edge_aermod_zone_edge_proportion", "aermod_zone_edge_proportion"),
    )
    _combine(
        "aermod_edge_link_length_m",
        ("edge_aermod_edge_link_length_m", "aermod_edge_link_length_m"),
    )
    _combine(
        "aermod_zone_link_length_m",
        ("edge_aermod_zone_link_length_m", "aermod_zone_link_length_m"),
    )
    _combine(
        "inmap_zone_edge_proportion",
        ("edge_inmap_zone_edge_proportion", "inmap_zone_edge_proportion"),
    )
    _combine(
        "inmap_edge_link_length_m",
        ("edge_inmap_edge_link_length_m", "inmap_edge_link_length_m"),
    )
    _combine(
        "inmap_zone_link_length_m",
        ("edge_inmap_zone_link_length_m", "inmap_zone_link_length_m"),
    )

    for col in ("linkId", "aermod_cell_id", "inmap_cell_id"):
        if col in canonical.columns:
            canonical[col] = pd.to_numeric(canonical[col], errors="coerce")
    if "countyfp" in canonical.columns:
        canonical["countyfp"] = canonical["countyfp"].astype("string")
    if "geometry" in canonical.columns:
        sample = canonical["geometry"].dropna().head(1)
        if not sample.empty and isinstance(sample.iloc[0], (bytes, bytearray, memoryview)):
            canonical["geometry"] = canonical["geometry"].map(
                lambda value: shapely.wkb.loads(value) if isinstance(value, (bytes, bytearray, memoryview)) else value
            )

    ordered_cols = [col for col in _CANONICAL_INTERSECTION_COLUMNS if col in canonical.columns]
    if "geometry" not in ordered_cols and "geometry" in canonical.columns:
        ordered_cols.append("geometry")
    canonical = canonical[ordered_cols].copy()
    if "geometry" in canonical.columns:
        return gpd.GeoDataFrame(canonical, geometry="geometry", crs=getattr(df, "crs", None))
    return canonical


def _union_county_matches_with_unmatched(
    source: gpd.GeoDataFrame,
    matched: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if _SOURCE_ROW_ID not in source.columns:
        raise ValueError(f"Source rows must include {_SOURCE_ROW_ID}")

    matched_source_col = _resolve_source_row_col(matched)
    matched_ids = set(pd.to_numeric(matched[matched_source_col], errors="coerce").dropna().astype(int).tolist())
    unmatched = source.loc[~source[_SOURCE_ROW_ID].isin(matched_ids)].copy()

    county_cols = [c for c in matched.columns if c.startswith("county_") and c != "geometry"]
    for col in county_cols:
        if col not in unmatched.columns:
            dtype = matched[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                fill = np.nan
            else:
                fill = pd.NA
            unmatched[col] = pd.Series([fill] * len(unmatched), index=unmatched.index)

    matched_clean = matched.drop(columns=[matched_source_col], errors="ignore")
    unmatched_clean = unmatched.drop(columns=[_SOURCE_ROW_ID], errors="ignore")
    ordered_cols = list(dict.fromkeys(list(matched_clean.columns) + list(unmatched_clean.columns)))

    for col in ordered_cols:
        if col not in matched_clean.columns:
            matched_clean[col] = pd.Series([pd.NA] * len(matched_clean), index=matched_clean.index)
        if col not in unmatched_clean.columns:
            unmatched_clean[col] = pd.Series([pd.NA] * len(unmatched_clean), index=unmatched_clean.index)
    matched_clean = matched_clean[ordered_cols]
    unmatched_clean = unmatched_clean[ordered_cols]

    return gpd.GeoDataFrame(
        pd.concat([matched_clean, unmatched_clean], ignore_index=True),
        geometry="geometry",
        crs=matched.crs,
    )




def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    input_root: Path,
) -> Tuple[str, Optional[gpd.GeoDataFrame]]:
    """Map staged BEAM network to OSM and intersect it with enabled dispersion grids and county boundaries."""
    from osm_chordify.osm.intersect import intersect_road_network_with_zones

    log_step_banner("Preprocess Step 3", "Integrate Grids", logger=logger)
    grid_intersection_path = raw_dir / "beam_osm_zone_county_intersection.parquet"
    if grid_intersection_path.exists():
        logger.info("Step 3: reusing existing grid intersection %s", grid_intersection_path)
        return str(grid_intersection_path), None
    mapped_network_path = str((input_root / "network" / "beam_osm_mapped.parquet").resolve())
    mapped_network_gpkg = str(Path(mapped_network_path).with_suffix(".gpkg"))
    log_substep_banner("3.1", "map BEAM network to OSM", logger=logger)
    if Path(mapped_network_path).exists():
        logger.info("Step 3.1: reusing BEAM/OSM mapping %s", mapped_network_path)
    else:
        staged_network = _resolve_staged_network_path(input_root)
        staged_osm = _resolve_staged_osm_path(input_root)
        logger.info("Step 3.1: mapping BEAM network to OSM using %s", staged_osm)
        mapped_network = _map_beam_network_to_osm(
            osm_path=staged_osm,
            network_path=staged_network,
            output_path=mapped_network_path,
            network_osm_id_col=pipeline.beam_osm_id_col,
            output_epsg=int(pipeline.output_epsg),
        )
        mapped_network.to_file(mapped_network_gpkg, driver="GPKG")
        logger.info("Step 3.1 complete: wrote %s", mapped_network_path)
    mapped_network = _load_geodataframe(mapped_network_path)
    epsg = int(pipeline.output_epsg)

    network_with_zones = mapped_network
    if pipeline.inmap_enabled:
        log_substep_banner("3.2", "intersect with InMAP grid", logger=logger)
        logger.info("Step 3.2: intersecting with inMAP grid %s", pipeline.inmap_grid_path)
        network_with_zones = intersect_road_network_with_zones(
            mapped_network,
            epsg,
            pipeline.inmap_grid_path,
            output_epsg=epsg,
            prefilter_zones_to_network_bbox=True,
            zone_label="inmap",
        )
        network_with_zones = trace_and_filter_void_zone_rows(
            network_with_zones,
            zone_id_col="inmap_cell_id",
            proportion_col="inmap_zone_edge_proportion",
            context="Step 3.2",
        )
        logger.info("Step 3.2 complete: %d rows", len(network_with_zones))

    if pipeline.aermod_enabled:
        log_substep_banner("3.3", "intersect with AERMOD grid", logger=logger)
        logger.info("Step 3.3: intersecting line network with AERMOD grid %s", pipeline.aermod_grid_path)
        network_with_zones = intersect_road_network_with_zones(
            network_with_zones,
            epsg,
            pipeline.aermod_grid_path,
            output_epsg=epsg,
            prefilter_zones_to_network_bbox=True,
            zone_label="aermod",
        )
        logger.info("Step 3.3 complete: %d rows", len(network_with_zones))
    B = network_with_zones.reset_index(drop=True).copy()
    B[_SOURCE_ROW_ID] = range(len(B))

    log_substep_banner("3.4", "intersect with county boundaries", logger=logger)
    county_setup_started = time.perf_counter()
    county_path = _resolve_county_path(input_root)
    county_gdf = read_vector(county_path)
    if county_gdf.crs is not None:
        county_gdf = county_gdf.to_crs(epsg=epsg)
    logger.info(
        "Step 3.4: intersecting with county boundaries (%d polygons prepared in %.2fs)",
        len(county_gdf),
        time.perf_counter() - county_setup_started,
    )
    county_match_started = time.perf_counter()
    C_matched = intersect_road_network_with_zones(
        B, epsg, county_gdf, output_epsg=epsg, zone_label="county",
    )
    logger.info(
        "Step 3.4 complete: %d matched rows in %.2fs",
        len(C_matched),
        time.perf_counter() - county_match_started,
    )

    log_substep_banner("3.5", "recover unmatched county rows", logger=logger)
    logger.info("Step 3.5: identifying unmatched rows from county matches")
    unmatched_started = time.perf_counter()
    matched_source_col = _resolve_source_row_col(C_matched)
    C = _union_county_matches_with_unmatched(B, C_matched)
    logger.info(
        "Step 3.5 complete: %d unmatched rows in %.2fs",
        len(B) - len(C_matched[matched_source_col].drop_duplicates()),
        time.perf_counter() - unmatched_started,
    )

    persist_started = time.perf_counter()
    log_substep_banner("3.6", "write canonical intersection outputs", logger=logger)
    logger.info("Step 3.6: canonicalizing and writing intersection outputs")
    C_canonical = canonicalize_intersection_schema(C)
    C_canonical.to_parquet(grid_intersection_path, index=False)
    C_canonical.to_file(grid_intersection_path.with_suffix(".gpkg"), driver="GPKG")
    logger.info(
        "Step 3.6 complete: wrote outputs in %.2fs",
        time.perf_counter() - persist_started,
    )
    logger.info("Step 3 complete: %d total rows → %s", len(C_canonical), grid_intersection_path)

    return str(grid_intersection_path), C_canonical
