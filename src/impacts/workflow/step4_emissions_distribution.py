"""Step 4 — Combined emissions distribution path."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import geopandas as gpd
import pandas as pd
import duckdb
import numpy as np

from ..config.defaults import annualization_days as default_annualization_days
from ..config.defaults import chunk_size as default_chunk_size
from ..config.defaults import county_correction_columns as default_county_correction_columns
from ..config.defaults import pollutants as default_prepared_pollutants
from ..config.defaults import grams_per_ton

from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)


def _load_grid_geometries(grid_path: str) -> gpd.GeoDataFrame:
    path = Path(grid_path)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _read_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(target)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(target, compression="gzip")
    if lower.endswith(".csv"):
        return pd.read_csv(target)
    raise ValueError(f"Unsupported table format: {target}")


def _first_existing(df: pd.DataFrame, candidates) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_column_config(config: Optional[Dict[str, str]], defaults: Dict[str, str]) -> Dict[str, str]:
    resolved = defaults.copy()
    if config:
        resolved.update({k: v for k, v in config.items() if v})
    return resolved


def _normalize_county_fips(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3)


def read_skims_emissions(
    path: str,
    pollutants: Optional[List[str]] = None,
) -> pd.DataFrame:
    dim_cols = ["linkId", "vehicleTypeId", "process"]
    cols = None if pollutants is None else dim_cols + [c for c in pollutants if c not in dim_cols]
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        import pyarrow.parquet as pq

        if cols is not None:
            available = set(pq.read_schema(path).names)
            cols = [c for c in cols if c in available] or None
        return pd.read_parquet(target, columns=cols)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(target, compression="gzip", usecols=cols if pollutants else None)
    if lower.endswith(".csv"):
        return pd.read_csv(target, usecols=cols if pollutants else None)
    raise ValueError(f"Unsupported skims format: {target}. Use .csv, .csv.gz, or .parquet")


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


def expand_emissions_columns(df: pd.DataFrame, emissions_col: str = "emissions") -> pd.DataFrame:
    parsed = df[emissions_col].apply(parse_emissions_string)
    expanded_pollutants = sorted({key for values in parsed for key in values.keys()})
    for pollutant in expanded_pollutants:
        df[f"em_{pollutant}"] = parsed.apply(lambda values, p=pollutant: float(values.get(p, 0.0)))
    return df


def _totals_pollutant_columns(
    df: pd.DataFrame,
    required_pollutants: List[str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for pollutant in required_pollutants:
        for candidate in (pollutant, f"em_{pollutant}", f"tons_per_year_{pollutant}"):
            if candidate in df.columns:
                resolved[pollutant] = candidate
                break
    return resolved


def prepare_skims_for_grid_allocation(
    skims_path: str,
    output_path: str,
    *,
    group_cols: Optional[List[str]] = None,
    required_pollutants: Optional[List[str]] = None,
) -> pd.DataFrame:
    df = read_skims_emissions(skims_path)
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    missing_group_cols = [col for col in prepared_group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Prepared skims missing required grouping columns: {missing_group_cols}")

    required = required_pollutants or default_prepared_pollutants
    if "emissions" in df.columns:
        df = expand_emissions_columns(df, emissions_col="emissions")
        observations_col = "observations"
        if observations_col not in df.columns:
            raise ValueError("Prepared skims require an observations column.")

        source_pollutant_cols = [f"em_{pollutant}" for pollutant in required]
        for col in source_pollutant_cols:
            if col not in df.columns:
                df[col] = 0.0

        prepared = df[prepared_group_cols + [observations_col] + source_pollutant_cols].copy()
        prepared[observations_col] = pd.to_numeric(prepared[observations_col], errors="coerce").fillna(0.0)
        rename_map = {f"em_{pollutant}": pollutant for pollutant in required}
        prepared = prepared.rename(columns=rename_map)
        pollutant_cols = list(rename_map.values())
        for col in pollutant_cols:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0) * prepared[observations_col]
        aggregated = prepared.groupby(prepared_group_cols, dropna=False)[pollutant_cols].sum().reset_index()
    else:
        totals_cols = _totals_pollutant_columns(df, required)
        prepared = df[prepared_group_cols + list(totals_cols.values())].copy()
        rename_map = {source: pollutant for pollutant, source in totals_cols.items()}
        prepared = prepared.rename(columns=rename_map)
        pollutant_cols = list(rename_map.values())
        for pollutant in required:
            if pollutant not in prepared.columns:
                prepared[pollutant] = 0.0
        pollutant_cols = list(required)
        for col in pollutant_cols:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)
        aggregated = prepared.groupby(prepared_group_cols, dropna=False)[pollutant_cols].sum().reset_index()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        aggregated.to_parquet(out, index=False)
    elif out.name.lower().endswith(".csv.gz"):
        aggregated.to_csv(out, index=False, compression="gzip")
    else:
        raise ValueError("Prepared skims output must be .parquet or .csv.gz")
    return aggregated


def annualize_prepared_skims_for_grid_allocation(
    prepared_skims_path: str,
    output_path: str,
    *,
    group_cols: Optional[List[str]] = None,
    required_pollutants: Optional[List[str]] = None,
    annualization_days: float = default_annualization_days,
) -> pd.DataFrame:
    if annualization_days <= 0:
        raise ValueError(f"Annualization days must be positive, got {annualization_days}")

    prepared = _read_table(prepared_skims_path)
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    required = required_pollutants or default_prepared_pollutants
    missing_group_cols = [col for col in prepared_group_cols if col not in prepared.columns]
    if missing_group_cols:
        raise ValueError(f"Annualized skims missing required grouping columns: {missing_group_cols}")

    out = prepared[prepared_group_cols].copy()
    for pollutant in required:
        source_col = _first_existing(prepared, [pollutant, f"em_{pollutant}", f"tons_per_year_{pollutant}"])
        values = (
            pd.to_numeric(prepared[source_col], errors="coerce").fillna(0.0)
            if source_col is not None
            else pd.Series(np.zeros(len(prepared), dtype=float), index=prepared.index)
        )
        out[f"tons_per_year_{pollutant}"] = values * annualization_days / 1_000_000.0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        out.to_parquet(output, index=False)
    elif output.name.lower().endswith(".csv.gz"):
        out.to_csv(output, index=False, compression="gzip")
    else:
        raise ValueError("Annualized skims output must be .parquet or .csv.gz")
    return out


_COUNTY_FIPS_CANDIDATES = ["county_zone_COUNTYFP", "county_COUNTYFP", "zone_COUNTYFP", "COUNTYFP", "countyfp"]
_VMT_PROCESSES = {"RUNEX", "PMBW", "PMTW", "PRDUST", "RUNLOSS"}
_TRIP_PROCESSES = {"HOTSOAK", "DIURN", "STREX"}


def apply_county_corrections(
    allocated_df: pd.DataFrame,
    corrections_path: str,
    *,
    correction_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    columns = _resolve_column_config(correction_columns, default_county_correction_columns)
    corrections = _read_table(corrections_path)
    fips_source_col = _first_existing(corrections, [columns["county_fips"]])
    if fips_source_col is None:
        raise ValueError(
            f"County corrections file must include county FIPS column: {columns['county_fips']}."
        )

    county_col = _first_existing(allocated_df, _COUNTY_FIPS_CANDIDATES)
    if county_col is None:
        raise ValueError(
            "Allocated DataFrame must include a county FIPS column. "
            f"Expected one of: {_COUNTY_FIPS_CANDIDATES}."
        )

    vmt_col = _first_existing(corrections, [columns["vmt_factor"]])
    trips_col = _first_existing(corrections, [columns["trips_factor"]])
    factors = corrections.copy()
    factors["_fips_norm"] = _normalize_county_fips(factors[fips_source_col])
    factors["_corr_vmt"] = (
        pd.to_numeric(factors[vmt_col], errors="coerce").fillna(1.0).replace(0.0, 1.0)
        if vmt_col
        else 1.0
    )
    factors["_corr_trips"] = (
        pd.to_numeric(factors[trips_col], errors="coerce").fillna(1.0).replace(0.0, 1.0)
        if trips_col
        else 1.0
    )
    factors = factors[["_fips_norm", "_corr_vmt", "_corr_trips"]].drop_duplicates("_fips_norm")

    result = allocated_df.copy()
    result["_fips_norm"] = _normalize_county_fips(result[county_col])
    result = result.merge(factors, how="left", on="_fips_norm")
    result["_corr_vmt"] = result["_corr_vmt"].fillna(1.0)
    result["_corr_trips"] = result["_corr_trips"].fillna(1.0)

    process_upper = result.get("process", pd.Series("", index=result.index)).astype(str).str.upper()
    factor_arr = np.ones(len(result), dtype=np.float32)
    factor_arr = np.where(process_upper.isin(_VMT_PROCESSES), result["_corr_vmt"].to_numpy(dtype=np.float32), factor_arr)
    factor_arr = np.where(process_upper.isin(_TRIP_PROCESSES), result["_corr_trips"].to_numpy(dtype=np.float32), factor_arr)

    allocated_cols = [c for c in result.columns if c.endswith("_allocated")]
    for col in allocated_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) * factor_arr

    return result.drop(columns=["_fips_norm", "_corr_vmt", "_corr_trips"])


_SKIMS_DIMENSION_COLS = {
    "linkId",
    "vehicleTypeId",
    "process",
    "hour",
    "observations",
    "iterations",
    "travelTimeInSecond",
    "parkingDurationInSecond",
}


def annualize_skims(
    skims_df: pd.DataFrame,
    pollutants: List[str],
    annualization_days: float,
) -> pd.DataFrame:
    if annualization_days <= 0:
        raise ValueError(f"annualization_days must be positive, got {annualization_days}")

    dim_cols = [c for c in skims_df.columns if c in _SKIMS_DIMENSION_COLS]
    out = skims_df[dim_cols].copy()
    factor = annualization_days / grams_per_ton
    for pollutant in pollutants:
        source_col = _first_existing(skims_df, [pollutant, f"em_{pollutant}", f"tons_per_year_{pollutant}"])
        if source_col is None:
            out[f"tons_per_year_{pollutant}"] = 0.0
            continue
        values = pd.to_numeric(skims_df[source_col], errors="coerce").fillna(0.0)
        if source_col.startswith("tons_per_year_"):
            out[f"tons_per_year_{pollutant}"] = values
        else:
            out[f"tons_per_year_{pollutant}"] = values * factor
    return out


def _save_grid_emissions(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    grid_path: str,
    output_epsg: int,
    output_stem: Path,
) -> None:
    grid_gdf = _load_grid_geometries(grid_path)
    if grid_gdf.crs is not None:
        grid_gdf = grid_gdf.to_crs(epsg=output_epsg)
    expected_grid_ids = set(pd.to_numeric(df[left_col], errors="coerce").dropna().astype(int).unique().tolist())
    joined = df.merge(
        grid_gdf[[right_col, "geometry"]],
        how="left",
        left_on=left_col,
        right_on=right_col,
    )
    if right_col != left_col:
        joined = joined.drop(columns=[right_col])
    geo = gpd.GeoDataFrame(joined, geometry="geometry", crs=grid_gdf.crs)
    actual_grid_ids = set(pd.to_numeric(geo[left_col], errors="coerce").dropna().astype(int).unique().tolist())
    if actual_grid_ids != expected_grid_ids:
        missing = sorted(expected_grid_ids - actual_grid_ids)[:10]
        extra = sorted(actual_grid_ids - expected_grid_ids)[:10]
        raise ValueError(
            f"Step 4 grid export mismatch for {left_col}: expected {len(expected_grid_ids)} grid ids, "
            f"got {len(actual_grid_ids)}. sample_missing={missing} sample_extra={extra}"
        )
    missing_geometry = int(geo.geometry.isna().sum())
    if missing_geometry:
        raise ValueError(
            f"Step 4 grid export missing geometry for {missing_geometry} rows in {output_stem}"
        )
    geo.to_parquet(Path(str(output_stem) + ".parquet"), index=False)
    geo.to_file(Path(str(output_stem) + ".gpkg"), driver="GPKG")


def _load_intersection_subset(path: str, columns: list[str]) -> pd.DataFrame:
    target = Path(path)
    frame = _read_table(target)
    return frame[columns].copy()


def _load_intersection_subset_or_df(
    *,
    path: str,
    columns: list[str],
    intersection_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if intersection_df is not None:
        return intersection_df[columns].copy()
    return _load_intersection_subset(path, columns)

def _step_label(step: str, zone_label: Optional[str] = None) -> str:
    suffix = f"[{zone_label}]" if zone_label else ""
    return f"Step 4.{step}{suffix}"


def _existing_output(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _reuse_existing_step4_outputs(raw_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    beam_emissions_for_inmap = _existing_output(raw_dir / "beam_emissions_for_inmap.parquet")
    if not beam_emissions_for_inmap:
        return None

    outputs = {
        "beam_emissions_for_aermod": _existing_output(raw_dir / "beam_emissions_for_aermod.parquet"),
        "beam_emissions_for_inmap": beam_emissions_for_inmap,
    }
    logger.info(
        "%s reusing existing emissions outputs; skipping Step 4 recomputation (inmap=%s, aermod=%s)",
        _step_label("0"),
        outputs["beam_emissions_for_inmap"],
        outputs["beam_emissions_for_aermod"],
    )
    return outputs


def _build_combined_grouped_table(
    *,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    required_cols = {
        "linkId",
        "countyfp",
        "aermod_srv_cell_id",
        "inmap_srm_cell_id",
        "aermod_zone_edge_proportion",
        "aermod_edge_link_length_m",
        "aermod_zone_link_length_m",
        "inmap_zone_edge_proportion",
        "inmap_edge_link_length_m",
        "inmap_zone_link_length_m",
    }
    intersection = _load_intersection_subset_or_df(
        path=intersection_path,
        columns=list(required_cols),
        intersection_df=intersection_df,
    )
    missing = [col for col in required_cols if col not in intersection.columns]
    if missing:
        raise ValueError(
            f"{_step_label('1')} requires canonical Step 3 columns. Missing: {missing}"
        )

    metric_cols = [
        "aermod_zone_edge_proportion",
        "aermod_edge_link_length_m",
        "aermod_zone_link_length_m",
        "inmap_zone_edge_proportion",
        "inmap_edge_link_length_m",
        "inmap_zone_link_length_m",
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("intersection_df", intersection)
        metric_select = ",\n                ".join([f"SUM(COALESCE(i.{col}, 0.0)) AS {col}" for col in metric_cols])
        grouped = con.execute(
            f"""
            SELECT
                i.linkId,
                i.aermod_srv_cell_id,
                i.inmap_srm_cell_id
                , i.countyfp
                {"," if metric_select else ""} {metric_select}
            FROM intersection_df AS i
            WHERE i.aermod_srv_cell_id IS NOT NULL OR i.inmap_srm_cell_id IS NOT NULL
            GROUP BY
                i.linkId,
                i.aermod_srv_cell_id,
                i.inmap_srm_cell_id
                , i.countyfp
            """
        ).df()
    finally:
        con.close()

    if grouped.empty:
        return None

    logger.info("%s BEAM mapping across grids rows=%d", _step_label("1"), len(grouped))
    return grouped


def _build_combined_allocated_table(
    *,
    grouped_df: pd.DataFrame,
    skims_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if grouped_df is None or grouped_df.empty:
        return None

    emission_cols = [
        c for c in skims_df.columns
        if c.startswith("tons_per_year_") and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    aermod_prop = "aermod_zone_edge_proportion"
    inmap_prop = "inmap_zone_edge_proportion"

    metric_cols = [
        col for col in grouped_df.columns
        if (col.startswith("aermod_") or col.startswith("inmap_")) and any(tag in col for tag in ("_proportion", "_length_m", "_surface_m2"))
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("grouped_df", grouped_df)
        con.register("skims_df", skims_df[["linkId", "vehicleTypeId", "process"] + emission_cols].copy())
        metric_select = ",\n                ".join([f"g.{col}" for col in metric_cols])
        aermod_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{aermod_prop} AS DOUBLE), 0.0) AS {col}_aermod_allocated"
            for col in emission_cols
        ]) if aermod_prop else ""
        inmap_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{inmap_prop} AS DOUBLE), 0.0) AS {col}_inmap_allocated"
            for col in emission_cols
        ]) if inmap_prop else ""
        extra_select = ",\n                ".join([part for part in [metric_select, aermod_alloc, inmap_alloc] if part])
        allocated = con.execute(
            f"""
            SELECT
                g.linkId,
                s.vehicleTypeId,
                s.process,
                g.aermod_srv_cell_id,
                g.inmap_srm_cell_id,
                g.countyfp
                {"," if extra_select else ""} {extra_select}
            FROM grouped_df AS g
            LEFT JOIN skims_df AS s
                ON g.linkId = s.linkId
            """
        ).df()
    finally:
        con.close()

    if allocated.empty:
        return None

    logger.info("%s BEAM emissions allocated across grids rows=%d", _step_label("2"), len(allocated))
    return allocated


def _split_zone_allocated(
    *,
    combined_df: pd.DataFrame,
    zone_label: str,
    cell_col: str,
) -> Optional[pd.DataFrame]:
    if combined_df is None or combined_df.empty or cell_col not in combined_df.columns:
        return None
    zone_metric_cols = [
        col for col in combined_df.columns
        if col.startswith(f"{zone_label}_") and col != cell_col
    ]
    zone_emission_cols = [col for col in combined_df.columns if col.startswith("tons_per_year_") and col.endswith(f"_{zone_label}_allocated")]
    keep_cols = [col for col in ["linkId", "vehicleTypeId", "process", cell_col] if col in combined_df.columns]
    if "countyfp" in combined_df.columns and "countyfp" not in keep_cols:
        keep_cols.append("countyfp")
    keep_cols.extend(zone_metric_cols)
    keep_cols.extend(zone_emission_cols)
    keep_cols = [col for i, col in enumerate(keep_cols) if col not in keep_cols[:i]]
    zone_df = combined_df[keep_cols].copy()
    zone_df = zone_df[zone_df[cell_col].notna()].copy()
    if zone_df.empty:
        return None

    sum_cols = [col for col in zone_df.columns if col not in {"linkId", "vehicleTypeId", "process", cell_col, "countyfp"}]
    group_cols = [col for col in ["linkId", "vehicleTypeId", "process", cell_col, "countyfp"] if col in zone_df.columns]
    zone_df = zone_df.groupby(group_cols, dropna=False)[sum_cols].sum().reset_index()
    return zone_df


def _build_combined_corrected_table(
    *,
    allocated_df: pd.DataFrame,
    pipeline: PipelineConfig,
) -> Optional[pd.DataFrame]:
    if allocated_df is None or allocated_df.empty:
        return None

    if pipeline.activity_corrections_path:
        logger.info(
            "%s applying county corrections from %s",
            _step_label("3"),
            pipeline.activity_corrections_path,
        )
        corrected = apply_county_corrections(
            allocated_df,
            corrections_path=pipeline.activity_corrections_path,
            correction_columns=pipeline.activity_corrections_columns or None,
        )
    else:
        corrected = allocated_df
        logger.info("%s no corrections configured; using allocated totals as-is", _step_label("3"))
    return corrected


def run(
    pipeline: PipelineConfig,
    raw_dir: Path,
    skims_df: pd.DataFrame,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Optional[str]]:
    reused = _reuse_existing_step4_outputs(raw_dir)
    if reused is not None:
        return reused

    combined_grouped_df = _build_combined_grouped_table(
        intersection_path=intersection_path,
        intersection_df=intersection_df,
    )
    combined_allocated_df = _build_combined_allocated_table(
        grouped_df=combined_grouped_df,
        skims_df=skims_df,
    )
    combined_corrected_df = _build_combined_corrected_table(
        allocated_df=combined_allocated_df,
        pipeline=pipeline,
    )

    beam_emissions_for_aermod_path = None
    if pipeline.aermod_grid_path and combined_grouped_df is not None:
        aermod_corrected_df = _split_zone_allocated(
            combined_df=combined_corrected_df,
            zone_label="aermod",
            cell_col="aermod_srv_cell_id",
        )
        if aermod_corrected_df is not None and not aermod_corrected_df.empty:
            beam_emissions_for_aermod_stem = raw_dir / "beam_emissions_for_aermod"
            _save_grid_emissions(
                aermod_corrected_df,
                left_col="aermod_srv_cell_id",
                right_col="srv_cell_id",
                grid_path=pipeline.aermod_grid_path,
                output_epsg=int(pipeline.output_epsg),
                output_stem=beam_emissions_for_aermod_stem,
            )
            beam_emissions_for_aermod_path = str(beam_emissions_for_aermod_stem) + ".parquet"
            logger.info(
                "%s BEAM emissions for AERMOD → %s",
                _step_label("4", "aermod"),
                beam_emissions_for_aermod_path,
            )

    inmap_corrected_df = _split_zone_allocated(
        combined_df=combined_corrected_df,
        zone_label="inmap",
        cell_col="inmap_srm_cell_id",
    )
    beam_emissions_for_inmap_path = None
    if inmap_corrected_df is not None and not inmap_corrected_df.empty:
        beam_emissions_for_inmap_stem = raw_dir / "beam_emissions_for_inmap"
        _save_grid_emissions(
            inmap_corrected_df,
            left_col="inmap_srm_cell_id",
            right_col="srm_cell_id",
            grid_path=pipeline.inmap_grid_path,
            output_epsg=int(pipeline.output_epsg),
            output_stem=beam_emissions_for_inmap_stem,
        )
        beam_emissions_for_inmap_path = str(beam_emissions_for_inmap_stem) + ".parquet"
        logger.info(
            "%s BEAM emissions for InMAP → %s",
            _step_label("5", "inmap"),
            beam_emissions_for_inmap_path,
        )

    return {
        "beam_emissions_for_aermod": beam_emissions_for_aermod_path,
        "beam_emissions_for_inmap": beam_emissions_for_inmap_path,
    }
