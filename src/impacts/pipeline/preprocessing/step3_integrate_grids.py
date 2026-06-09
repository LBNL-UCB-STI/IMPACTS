"""Step 3 (preprocess) — Network mapping and grid intersection helpers."""
from __future__ import annotations

from contextlib import contextmanager
import logging
import os
from pathlib import Path
import time
from typing import Iterator
from typing import Optional
from typing import Tuple

import geopandas as gpd
import numpy as np
import osm_chordify
import pandas as pd
import shapely.wkb
from shapely.geometry import LineString
from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_vector
from ...common import resolve_required_manifest_input
from ...manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)
_DEFAULT_OSM_CHORDIFY_TQDM_NCOLS = 79
_SOURCE_ROW_ID = "source_row_id"
_COUNTY_INTERSECTION_COLUMNS = [
    "linkId",
    "county_COUNTYFP",
    "county_proportion",
    "county_link_length_m",
    "geometry",
]
_INMAP_INTERSECTION_COLUMNS = [
    "linkId",
    "inmap_cell_id",
    "inmap_proportion",
    "inmap_link_length_m",
    "geometry",
]
_AERMOD_INTERSECTION_COLUMNS = [
    "linkId",
    "aermod_cell_id",
    "aermod_proportion",
    "aermod_link_length_m",
    "geometry",
]

def _osm_chordify_tqdm_ncols() -> int:
    raw_value = os.environ.get("IMPACTS_OSM_CHORDIFY_TQDM_NCOLS", "").strip()
    if raw_value:
        ncols = int(raw_value)
        if ncols < 40:
            raise ValueError("IMPACTS_OSM_CHORDIFY_TQDM_NCOLS must be at least 40.")
        return ncols
    return _DEFAULT_OSM_CHORDIFY_TQDM_NCOLS


@contextmanager
def _log_safe_osm_chordify_progress() -> Iterator[None]:
    import osm_chordify.osm.intersect as osm_intersect

    original_tqdm = osm_intersect.tqdm

    def _fixed_width_tqdm(*args, **kwargs):
        kwargs["dynamic_ncols"] = False
        kwargs.setdefault("ncols", _osm_chordify_tqdm_ncols())
        return original_tqdm(*args, **kwargs)

    osm_intersect.tqdm = _fixed_width_tqdm
    try:
        yield
    finally:
        osm_intersect.tqdm = original_tqdm


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


def _link_supports_car(value: object) -> bool:
    token = str("" if pd.isna(value) else value).strip()
    if not token:
        return False
    return "car" in {part.strip().lower() for part in token.split(";") if part.strip()}


def _build_synthetic_beam_links(
    *,
    network_path: str,
    mapped_network: gpd.GeoDataFrame,
    output_epsg: int,
) -> gpd.GeoDataFrame:
    network = pd.read_csv(network_path, compression="infer")
    required = {
        "linkId",
        "linkModes",
        "attributeOrigId",
        "fromLocationX",
        "fromLocationY",
        "toLocationX",
        "toLocationY",
        "linkLength",
    }
    missing = sorted(required - set(network.columns))
    if missing:
        raise ValueError(f"Network input missing required columns for synthetic BEAM links: {missing}")

    mapped_ids = set(pd.to_numeric(mapped_network["linkId"], errors="coerce").dropna().astype(int).tolist())
    candidates = network.loc[
        network["attributeOrigId"].isna()
        & network["linkModes"].map(_link_supports_car)
        & ~network["linkId"].isin(mapped_ids)
    ].copy()
    if candidates.empty:
        return mapped_network

    for col in ("fromLocationX", "fromLocationY", "toLocationX", "toLocationY", "linkLength"):
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates = candidates.loc[
        candidates[["fromLocationX", "fromLocationY", "toLocationX", "toLocationY"]].notna().all(axis=1)
    ].copy()
    if candidates.empty:
        return mapped_network

    candidates["geometry"] = [
        LineString([(float(from_x), float(from_y)), (float(to_x), float(to_y))])
        for from_x, from_y, to_x, to_y in zip(
            candidates["fromLocationX"].to_numpy(),
            candidates["fromLocationY"].to_numpy(),
            candidates["toLocationX"].to_numpy(),
            candidates["toLocationY"].to_numpy(),
        )
    ]
    candidates = gpd.GeoDataFrame(candidates, geometry="geometry", crs=f"EPSG:{output_epsg}")
    candidates["osm_id"] = pd.NA
    candidates["name"] = pd.NA
    candidates["highway"] = pd.NA
    candidates["waterway"] = pd.NA
    candidates["aerialway"] = pd.NA
    candidates["barrier"] = pd.NA
    candidates["man_made"] = pd.NA
    candidates["railway"] = pd.NA
    candidates["z_order"] = pd.NA
    candidates["other_tags"] = pd.NA
    candidates["edge_id"] = candidates["linkId"].map(lambda value: f"beam_synthetic_{int(value)}")
    candidates["edge_length"] = pd.to_numeric(candidates["linkLength"], errors="coerce").fillna(candidates.geometry.length)

    for col in mapped_network.columns:
        if col not in candidates.columns:
            candidates[col] = pd.NA
    candidates = candidates[mapped_network.columns].copy()
    augmented = pd.concat([mapped_network, candidates], ignore_index=True)
    logger.info(
        "Step 3.1 synthetic link fallback: appended %d BEAM car links without OSM origin to mapped network",
        len(candidates),
    )
    return gpd.GeoDataFrame(augmented, geometry="geometry", crs=mapped_network.crs)


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
    filtered = df.loc[~void_mask]
    logger.info(
        "%s removed %d bbox-only zone rows after exact-intersection screening",
        context,
        int(void_mask.sum()),
    )
    return filtered


def _resolve_source_row_col(df: pd.DataFrame) -> str:
    if _SOURCE_ROW_ID in df.columns:
        return _SOURCE_ROW_ID
    raise ValueError(f"Expected source row id column '{_SOURCE_ROW_ID}' in columns: {list(df.columns)}")


def _ensure_mapped_network_covers_car_beam_links(
    *,
    mapped_network_path: str,
    network_path: str,
    output_epsg: int,
    enabled: bool,
) -> gpd.GeoDataFrame:
    mapped_network = _load_geodataframe(mapped_network_path)
    if not enabled:
        return mapped_network
    augmented = _build_synthetic_beam_links(
        network_path=network_path,
        mapped_network=mapped_network,
        output_epsg=output_epsg,
    )
    if len(augmented) != len(mapped_network):
        _save_geodataframe(augmented, mapped_network_path)
        gpkg_path = str(Path(mapped_network_path).with_suffix(".gpkg"))
        _save_geodataframe(augmented, gpkg_path)
    return augmented


def _normalize_zone_intersection_schema(
    df: pd.DataFrame,
    *,
    zone_label: str,
) -> gpd.GeoDataFrame:
    canonical = df.copy()
    canonical["linkId"] = pd.to_numeric(canonical["linkId"], errors="coerce")

    if zone_label == "county":
        required_cols = _COUNTY_INTERSECTION_COLUMNS
        canonical["county_COUNTYFP"] = canonical["county_COUNTYFP"].astype("string")
        canonical["county_proportion"] = pd.to_numeric(canonical["county_proportion"], errors="coerce")
        canonical["county_link_length_m"] = pd.to_numeric(canonical["county_link_length_m"], errors="coerce")
    elif zone_label == "inmap":
        required_cols = _INMAP_INTERSECTION_COLUMNS
        canonical["inmap_cell_id"] = pd.to_numeric(canonical["inmap_cell_id"], errors="coerce")
        canonical["inmap_proportion"] = pd.to_numeric(canonical["inmap_proportion"], errors="coerce")
        canonical["inmap_link_length_m"] = pd.to_numeric(canonical["inmap_link_length_m"], errors="coerce")
    elif zone_label == "aermod":
        required_cols = _AERMOD_INTERSECTION_COLUMNS
        canonical["aermod_cell_id"] = pd.to_numeric(canonical["aermod_cell_id"], errors="coerce")
        canonical["aermod_proportion"] = pd.to_numeric(canonical["aermod_proportion"], errors="coerce")
        canonical["aermod_link_length_m"] = pd.to_numeric(canonical["aermod_link_length_m"], errors="coerce")
    else:
        raise ValueError(f"Unsupported zone_label {zone_label!r}")
    missing_cols = [col for col in required_cols if col not in canonical.columns]
    if missing_cols:
        raise ValueError(
            f"{zone_label} intersection must use the canonical schema. Missing columns: {missing_cols}"
        )
    ordered_cols = list(required_cols)

    if "geometry" in canonical.columns:
        sample = canonical["geometry"].dropna().head(1)
        if not sample.empty and isinstance(sample.iloc[0], (bytes, bytearray, memoryview)):
            canonical["geometry"] = canonical["geometry"].map(
                lambda value: shapely.wkb.loads(value) if isinstance(value, (bytes, bytearray, memoryview)) else value
            )
    if "geometry" in canonical.columns and "geometry" not in ordered_cols:
        ordered_cols.append("geometry")
    return gpd.GeoDataFrame(canonical[ordered_cols].copy(), geometry="geometry", crs=getattr(df, "crs", None))


def _union_county_matches_with_unmatched(
    source: gpd.GeoDataFrame,
    matched: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if _SOURCE_ROW_ID not in source.columns:
        raise ValueError(f"Source rows must include {_SOURCE_ROW_ID}")

    matched_source_col = _resolve_source_row_col(matched)
    matched_ids = set(pd.to_numeric(matched[matched_source_col], errors="coerce").dropna().astype(int).tolist())
    unmatched = source.loc[~source[_SOURCE_ROW_ID].isin(matched_ids)]

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


def _assign_fallback_counties(
    *,
    source_links: gpd.GeoDataFrame,
    county_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    fallback = source_links[["linkId", "geometry"]].copy()
    fallback["geometry"] = fallback.geometry.representative_point()
    joined = gpd.sjoin(
        fallback,
        county_gdf[["COUNTYFP", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")
    joined = joined.rename(columns={"COUNTYFP": "county_COUNTYFP"})
    missing = joined["county_COUNTYFP"].isna()
    if missing.any():
        nearest = gpd.sjoin_nearest(
            fallback.loc[missing, ["linkId", "geometry"]],
            county_gdf[["COUNTYFP", "geometry"]],
            how="left",
        ).drop(columns=["index_right", "distance"], errors="ignore")
        nearest = nearest.rename(columns={"COUNTYFP": "county_COUNTYFP"})
        joined.loc[missing, "county_COUNTYFP"] = nearest["county_COUNTYFP"].to_numpy()
    joined["county_COUNTYFP"] = joined["county_COUNTYFP"].astype("string")
    return pd.DataFrame(joined[["linkId", "county_COUNTYFP"]])


def _ensure_county_mass_conservation(
    *,
    source_links: gpd.GeoDataFrame,
    county_intersection: gpd.GeoDataFrame,
    county_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    if county_intersection.empty:
        return county_intersection

    result = county_intersection.copy()
    result["county_proportion"] = pd.to_numeric(
        result["county_proportion"], errors="coerce"
    ).fillna(0.0)
    result["county_link_length_m"] = pd.to_numeric(
        result["county_link_length_m"], errors="coerce"
    ).fillna(0.0)

    link_metrics = (
        result.groupby("linkId", dropna=False)
        .agg(
            county_prop_sum=("county_proportion", "sum"),
            county_zone_length_sum=("county_link_length_m", "sum"),
        )
        .reset_index()
    )
    affected = link_metrics.loc[link_metrics["county_prop_sum"] < 0.999999]
    if affected.empty:
        return result

    source = source_links[["linkId", "geometry"]].copy()
    source["source_edge_length_m"] = source.geometry.length
    fallback_links = source.loc[source["linkId"].isin(affected["linkId"])]
    fallback_counties = _assign_fallback_counties(source_links=fallback_links, county_gdf=county_gdf)

    affected = affected.merge(source[["linkId", "geometry", "source_edge_length_m"]], how="left", on="linkId")
    affected = affected.merge(
        fallback_counties.rename(columns={"county_COUNTYFP": "nearest_county_COUNTYFP"}),
        how="left",
        on="linkId",
    )
    affected["county_COUNTYFP"] = affected["nearest_county_COUNTYFP"].astype("string")
    affected["missing_share"] = (1.0 - affected["county_prop_sum"]).clip(lower=0.0)
    affected["missing_zone_length_m"] = (
        affected["source_edge_length_m"] - affected["county_zone_length_sum"]
    ).clip(lower=0.0)
    use_length_fallback = affected["missing_zone_length_m"].le(0.0) & affected["missing_share"].gt(0.0)
    affected.loc[use_length_fallback, "missing_zone_length_m"] = (
        affected.loc[use_length_fallback, "source_edge_length_m"]
        * affected.loc[use_length_fallback, "missing_share"]
    )

    synthetic = affected.loc[affected["missing_share"] > 0.0, [
        "linkId",
        "county_COUNTYFP",
        "missing_share",
        "missing_zone_length_m",
        "geometry",
    ]]
    synthetic = synthetic.rename(
        columns={
            "missing_share": "county_proportion",
            "missing_zone_length_m": "county_link_length_m",
        }
    )
    synthetic = synthetic.loc[synthetic["county_COUNTYFP"].notna()]
    if synthetic.empty:
        return result

    result = result.loc[
        ~(result["linkId"].isin(synthetic["linkId"])) | result["county_COUNTYFP"].notna()
    ].copy()
    combined = pd.concat([result, synthetic], ignore_index=True)
    logger.info(
        "Step 3.6 county conservation: synthesized %d fallback county rows across %d links",
        len(synthetic),
        synthetic["linkId"].nunique(),
    )
    return gpd.GeoDataFrame(combined, geometry="geometry", crs=county_intersection.crs)


def run(
    pipeline: PipelineConfig,
    input_root: Path,
    manifest_inputs: Optional[dict[str, object]] = None,
) -> Tuple[dict[str, Optional[str]], dict[str, Optional[gpd.GeoDataFrame]]]:
    """Map staged BEAM network to OSM and intersect it with county, InMAP, and AERMOD surfaces separately."""
    from osm_chordify.osm.intersect import intersect_road_network_with_zones

    log_step_banner("Preprocess Step 3", "Integrate Grids", logger=logger)
    county_intersection_path = input_root / "beam_osm_county_intersection.parquet"
    inmap_intersection_path = input_root / "beam_osm_inmap_intersection.parquet"
    aermod_intersection_path = input_root / "beam_osm_aermod_intersection.parquet"
    existing_paths = {
        "county": str(county_intersection_path) if county_intersection_path.exists() else None,
        "inmap": str(inmap_intersection_path) if pipeline.inmap_enabled and inmap_intersection_path.exists() else None,
        "aermod": str(aermod_intersection_path) if pipeline.aermod_enabled and aermod_intersection_path.exists() else None,
    }
    if manifest_inputs is None:
        raise ValueError("Step 3 requires manifest_inputs to resolve network and OSM inputs.")
    staged_network = resolve_required_manifest_input(manifest_inputs, key="network")
    if existing_paths["county"] and ((not pipeline.inmap_enabled) or existing_paths["inmap"]) and ((not pipeline.aermod_enabled) or existing_paths["aermod"]):
        logger.info(
            "Step 3: reusing existing separate intersections county=%s inmap=%s aermod=%s",
            existing_paths["county"],
            existing_paths["inmap"],
            existing_paths["aermod"],
        )
        return existing_paths, {"county": None, "inmap": None, "aermod": None}
    mapped_network_path = str((input_root / "beam_osm_mapped.parquet").resolve())
    log_substep_banner("3.1", "map BEAM network to OSM", logger=logger)
    if Path(mapped_network_path).exists():
        logger.info("Step 3.1: reusing BEAM/OSM mapping %s", mapped_network_path)
    else:
        staged_osm = resolve_required_manifest_input(manifest_inputs, key="osm_network")
        logger.info("Step 3.1: mapping BEAM network to OSM using %s", staged_osm)
        mapped_network = _map_beam_network_to_osm(
                osm_path=staged_osm,
                network_path=staged_network,
                output_path=mapped_network_path,
                network_osm_id_col=pipeline.beam_osm_id_col,
                output_epsg=int(pipeline.output_epsg),
            )
        mapped_network.to_file(str(Path(mapped_network_path).with_suffix(".gpkg")), driver="GPKG")
        logger.info("Step 3.1 complete: wrote %s", mapped_network_path)
    mapped_network = _ensure_mapped_network_covers_car_beam_links(
        mapped_network_path=mapped_network_path,
        network_path=staged_network,
        output_epsg=int(pipeline.output_epsg),
        enabled=bool(pipeline.include_non_osm_car_links),
    )
    epsg = int(pipeline.output_epsg)

    inmap_intersection: Optional[gpd.GeoDataFrame] = None
    if pipeline.inmap_enabled:
        log_substep_banner("3.2", "intersect with InMAP grid", logger=logger)
        logger.info("Step 3.2: intersecting with inMAP grid %s", pipeline.inmap_grid_path)
        with _log_safe_osm_chordify_progress():
            inmap_intersection = intersect_road_network_with_zones(
                mapped_network,
                epsg,
                pipeline.inmap_grid_path,
                output_epsg=epsg,
                prefilter_zones_to_network_bbox=True,
                zone_label="inmap",
            )
        inmap_intersection = trace_and_filter_void_zone_rows(
            inmap_intersection,
            zone_id_col="inmap_cell_id",
            proportion_col="inmap_proportion",
            context="Step 3.2",
        )
        inmap_intersection = _normalize_zone_intersection_schema(inmap_intersection, zone_label="inmap")
        inmap_intersection.to_parquet(inmap_intersection_path, index=False)
        inmap_intersection.to_file(inmap_intersection_path.with_suffix(".gpkg"), driver="GPKG")
        logger.info("Step 3.2 complete: %d rows → %s", len(inmap_intersection), inmap_intersection_path)

    aermod_intersection: Optional[gpd.GeoDataFrame] = None
    if pipeline.aermod_enabled:
        log_substep_banner("3.3", "intersect with AERMOD grid", logger=logger)
        logger.info("Step 3.3: intersecting line network with AERMOD grid %s", pipeline.aermod_grid_path)
        with _log_safe_osm_chordify_progress():
            aermod_intersection = intersect_road_network_with_zones(
                mapped_network,
                epsg,
                pipeline.aermod_grid_path,
                output_epsg=epsg,
                prefilter_zones_to_network_bbox=True,
                zone_label="aermod",
            )
        aermod_intersection = _normalize_zone_intersection_schema(aermod_intersection, zone_label="aermod")
        aermod_intersection.to_parquet(aermod_intersection_path, index=False)
        aermod_intersection.to_file(aermod_intersection_path.with_suffix(".gpkg"), driver="GPKG")
        logger.info("Step 3.3 complete: %d rows → %s", len(aermod_intersection), aermod_intersection_path)

    B = mapped_network.reset_index(drop=True)
    B[_SOURCE_ROW_ID] = range(len(B))

    log_substep_banner("3.4", "intersect with county boundaries", logger=logger)
    county_setup_started = time.perf_counter()
    county_path = resolve_required_manifest_input(manifest_inputs, key="county_boundaries")
    county_gdf = read_vector(county_path)
    if "COUNTYFP" not in county_gdf.columns:
        raise ValueError("County boundaries must include COUNTYFP.")
    county_gdf["COUNTYFP"] = county_gdf["COUNTYFP"].astype("string")
    if county_gdf.crs is not None:
        county_gdf = county_gdf.to_crs(epsg=epsg)
    logger.info(
        "Step 3.4: intersecting with county boundaries (%d polygons prepared in %.2fs)",
        len(county_gdf),
        time.perf_counter() - county_setup_started,
    )
    county_match_started = time.perf_counter()
    with _log_safe_osm_chordify_progress():
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
    log_substep_banner("3.6", "write canonical county intersection outputs", logger=logger)
    logger.info("Step 3.6: canonicalizing and writing county intersection outputs")
    county_intersection = _normalize_zone_intersection_schema(C, zone_label="county")
    county_intersection = _ensure_county_mass_conservation(
        source_links=B,
        county_intersection=county_intersection,
        county_gdf=county_gdf,
    )
    county_intersection.to_parquet(county_intersection_path, index=False)
    county_intersection.to_file(county_intersection_path.with_suffix(".gpkg"), driver="GPKG")
    logger.info(
        "Step 3.6 complete: wrote county outputs in %.2fs",
        time.perf_counter() - persist_started,
    )
    logger.info(
        "Step 3 complete: county=%s inmap=%s aermod=%s",
        county_intersection_path,
        inmap_intersection_path if pipeline.inmap_enabled else None,
        aermod_intersection_path if pipeline.aermod_enabled else None,
    )

    return (
        {
            "county": str(county_intersection_path),
            "inmap": str(inmap_intersection_path) if pipeline.inmap_enabled else None,
            "aermod": str(aermod_intersection_path) if pipeline.aermod_enabled else None,
        },
        {
            "county": county_intersection,
            "inmap": inmap_intersection,
            "aermod": aermod_intersection,
        },
    )
