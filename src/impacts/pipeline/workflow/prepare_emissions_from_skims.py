"""Shared helpers for skims preparation and grid allocation inputs."""

from __future__ import annotations

import logging
import re
from pathlib import Path
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import duckdb
import pandas as pd

from ...common import configure_duckdb_connection
from ...common import _table_available_columns
from ...common import normalize_county_fips
from ...common import prepare_skims_for_grid_allocation
from ...common import prepared_table_target
from ...common import read_table
from ...common import resolve_required_manifest_input
from ...common import is_valid_parquet
from . import _step_label
from .annualization import annualize_prepared_skims_for_grid_allocation

logger = logging.getLogger(__name__)

_PASSENGER_CATEGORY_TOKENS = {
    "car",
    "bike",
    "body",
    "body-type-default",
    "body type default",
    "mediumdutypassenger",
    "rail-default",
    "rail default",
    "ferry-default",
    "ferry default",
    "tram-sf",
    "tram sf",
    "obus",
    "sbus",
    "ubus",
    "mcy",
    "motor coach",
}
_TRANSIT_CATEGORY_TOKENS = {
    "obus",
    "sbus",
    "ubus",
    "motor coach",
    "rail-default",
    "rail default",
    "ferry-default",
    "ferry default",
    "subway-default",
    "subway default",
    "tram-default",
    "tram default",
    "train-default",
    "train default",
}
_TRANSIT_VEHICLETYPE_PATTERN = re.compile(
    r"(^|[-_])(BUS|RAIL|FERRY|SUBWAY|TRAM|TRAIN|COACH)($|[-_])",
    re.IGNORECASE,
)
_FREIGHT_CATEGORY_PATTERN = re.compile(r"^(class\d|class\d+[a-z]?|mdv|ldt\d|hdt|t\d)", re.IGNORECASE)
_FREIGHT_CATEGORY_SUBSTRINGS = ("vocational", "tractor")


# ---------------------------------------------------------------------------
def _existing_output(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _existing_valid_skims_parquet(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    if is_valid_parquet(path):
        return str(path)
    logger.warning("Step 1: ignoring invalid cached skims parquet %s", path)
    try:
        path.unlink()
    except Exception:
        pass
    return None


def _validate_prepared_skims_schema(*, path: str, require_aermod_support: bool) -> None:
    if not require_aermod_support:
        return
    available = set(_table_available_columns(path))
    required = {"roadCategory", "source_release_height"}
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            "Prepared skims cache is stale for AERMOD processing and must be rebuilt. "
            f"Cached file {path} is missing required columns: {missing}"
        )


def _load_intersection_subset(path: str, columns: List[str]) -> pd.DataFrame:
    target = Path(path)
    lower = target.name.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(target, columns=columns)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(target, compression="gzip", usecols=columns)
    if lower.endswith(".csv"):
        return pd.read_csv(target, usecols=columns)
    frame = read_table(target)
    return frame[columns].copy()


def _load_intersection_subset_or_df(
    *,
    path: str,
    columns: List[str],
    intersection_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if intersection_df is not None:
        return intersection_df[columns].copy()
    return _load_intersection_subset(path, columns)


def _register_frame_or_scan(
    *,
    con: duckdb.DuckDBPyConnection,
    relation_name: str,
    path: Optional[str],
    columns: list[str],
    frame: Optional[pd.DataFrame],
) -> str:
    if frame is not None:
        con.register(relation_name, frame[columns].copy())
        return relation_name
    if not path:
        raise ValueError(f"Expected path or frame for DuckDB relation '{relation_name}'.")
    target = Path(path)
    if target.suffix.lower() != ".parquet":
        subset = read_table(target)[columns].copy()
        con.register(relation_name, subset)
        return relation_name
    quoted_path = str(target).replace("'", "''")
    projected = ", ".join(f'"{column}"' for column in columns)
    return f"(SELECT {projected} FROM read_parquet('{quoted_path}'))"


def _normalize_vehicle_type_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _row_is_transit_vehicle_type(row: pd.Series) -> bool:
    vehicle_type_id = _normalize_vehicle_type_token(row.get("vehicleTypeId"))
    if vehicle_type_id and _TRANSIT_VEHICLETYPE_PATTERN.search(vehicle_type_id):
        return True
    token = _normalize_vehicle_type_token(row.get("emfacVehicleCategory")).lower()
    if token in _TRANSIT_CATEGORY_TOKENS:
        return True
    return False


def _classify_vehicle_type_assignment(row: pd.Series, *, source_name: str) -> Optional[str]:
    if _row_is_transit_vehicle_type(row):
        return "transit"

    explicit_assignment = _normalize_vehicle_type_token(row.get("assignment_group")).lower()
    if explicit_assignment in {"passenger", "freight", "transit"}:
        return explicit_assignment

    source_lower = source_name.lower()
    if "vehicletypes--atlas--" in source_lower or "vehicletypes--passenger" in source_lower:
        return "passenger"
    if "vehicletypes--frism--" in source_lower or "vehicletypes--freight" in source_lower:
        return "freight"

    vehicle_type_id = _normalize_vehicle_type_token(row.get("vehicleTypeId"))
    if vehicle_type_id.startswith("pax-"):
        return "passenger"
    if vehicle_type_id.startswith("ft-"):
        return "freight"

    vehicle_use = _normalize_vehicle_type_token(row.get("vehicleUse"))
    vehicle_class = _normalize_vehicle_type_token(row.get("vehicleClass"))
    if vehicle_use or vehicle_class:
        return "freight"

    vehicle_category = _normalize_vehicle_type_token(row.get("vehicleCategory")).lower()
    emfac_vehicle_category = _normalize_vehicle_type_token(row.get("emfacVehicleCategory")).lower()
    if emfac_vehicle_category in _PASSENGER_CATEGORY_TOKENS:
        return "passenger"
    if _FREIGHT_CATEGORY_PATTERN.match(emfac_vehicle_category) or any(
        token in emfac_vehicle_category for token in _FREIGHT_CATEGORY_SUBSTRINGS
    ):
        return "freight"
    if vehicle_category in _PASSENGER_CATEGORY_TOKENS:
        return "passenger"
    if _FREIGHT_CATEGORY_PATTERN.match(vehicle_category) or any(
        token in vehicle_category for token in _FREIGHT_CATEGORY_SUBSTRINGS
    ):
        return "freight"
    return None


def _load_vehicle_types_table(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    passenger = read_table(passenger_vehicle_types_path).copy()
    freight = read_table(freight_vehicle_types_path).copy()
    passenger["assignment_group"] = "passenger"
    freight["assignment_group"] = "freight"
    vehicle_types = pd.concat([passenger, freight], ignore_index=True, sort=False)
    if "vehicleTypeId" not in vehicle_types.columns:
        raise ValueError("Vehicle types inputs must include vehicleTypeId.")
    vehicle_types["vehicleTypeId"] = vehicle_types["vehicleTypeId"].map(_normalize_vehicle_type_token)
    duplicate_ids = (
        vehicle_types.loc[vehicle_types["vehicleTypeId"].ne("") & vehicle_types["vehicleTypeId"].duplicated(), "vehicleTypeId"]
        .drop_duplicates()
        .tolist()
    )
    if duplicate_ids:
        raise ValueError(
            "Configured passenger and freight vehicle types files contain duplicate vehicleTypeId values: "
            f"{duplicate_ids[:10]}"
        )
    return vehicle_types


def _load_vehicle_type_assignments(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    vehicle_types = _load_vehicle_types_table(passenger_vehicle_types_path, freight_vehicle_types_path)
    if "vehicleTypeId" not in vehicle_types.columns:
        raise ValueError("Vehicle types input must include vehicleTypeId for passenger/freight skims filtering.")

    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_vehicle_type_token)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()
    prepared["assignment_group"] = prepared.apply(
        lambda row: _classify_vehicle_type_assignment(row, source_name="combined_vehicle_types"),
        axis=1,
    )
    assignments = (
        prepared[["vehicleTypeId", "assignment_group"]]
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .reset_index(drop=True)
    )
    duplicate_conflicts = (
        prepared[["vehicleTypeId", "assignment_group"]]
        .dropna(subset=["assignment_group"])
        .drop_duplicates()
        .groupby("vehicleTypeId", dropna=False)["assignment_group"]
        .nunique()
    )
    conflicting_ids = duplicate_conflicts[duplicate_conflicts.gt(1)].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Vehicle types input assigns the same vehicleTypeId to both passenger and freight: "
            f"{conflicting_ids[:10]}"
        )
    return assignments


def _load_vehicle_type_activity_lookup(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    vehicle_types = _load_vehicle_types_table(passenger_vehicle_types_path, freight_vehicle_types_path)
    if "vehicleTypeId" not in vehicle_types.columns:
        raise ValueError("Vehicle types input must include vehicleTypeId for activity correction lookup.")
    if "emfacId" not in vehicle_types.columns:
        raise ValueError(
            "Vehicle types input must include emfacId for inventory-based activity correction."
        )
    if "emfacResolvedModelYear" not in vehicle_types.columns:
        raise ValueError(
            "Vehicle types input must include emfacResolvedModelYear for inventory-based activity correction."
        )

    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_vehicle_type_token)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()
    prepared["assignment_group"] = prepared.apply(
        lambda row: _classify_vehicle_type_assignment(row, source_name="combined_vehicle_types"),
        axis=1,
    )
    prepared["emfacId"] = prepared["emfacId"].map(_normalize_vehicle_type_token)
    prepared["modelYear"] = prepared["emfacResolvedModelYear"].astype(str).str.strip()
    prepared["modelYear"] = prepared["modelYear"].where(
        prepared["modelYear"].ne("") & ~prepared["modelYear"].str.lower().eq("nan"),
        pd.NA,
    )
    prepared = prepared.loc[prepared["assignment_group"].notna()].copy()
    correction_eligible = prepared["assignment_group"].isin({"passenger", "freight"})
    prepared = prepared.loc[~correction_eligible | prepared["modelYear"].notna()].copy()
    duplicate_conflicts = (
        prepared[["vehicleTypeId", "assignment_group", "modelYear"]]
        .drop_duplicates()
        .groupby("vehicleTypeId", dropna=False)
        .agg(
            assignment_group_count=("assignment_group", "nunique"),
            model_year_count=("modelYear", "nunique"),
        )
    )
    conflicting_ids = duplicate_conflicts.loc[
        duplicate_conflicts["assignment_group_count"].gt(1)
        | duplicate_conflicts["model_year_count"].gt(1)
    ].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Vehicle types input has conflicting assignment or modelYear rows for vehicleTypeId values: "
            f"{conflicting_ids[:10]}"
        )
    return (
        prepared[["vehicleTypeId", "assignment_group", "modelYear"]]
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .reset_index(drop=True)
    )


def _resolve_assignment_filter_ids(
    *,
    passenger_vehicle_types_path: Optional[str],
    freight_vehicle_types_path: Optional[str],
    include_passenger: bool,
    include_freight: bool,
) -> tuple[Optional[set[str]], Optional[set[str]]]:
    if include_passenger and include_freight:
        return None, None
    if not include_passenger and not include_freight:
        raise ValueError("At least one of include_passenger or include_freight must be true.")
    if not passenger_vehicle_types_path or not freight_vehicle_types_path:
        raise ValueError(
            "Vehicle types input is required to filter prepared skims by passenger/freight assignment."
        )

    assignments = _load_vehicle_type_assignments(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
    )
    allowed_groups = set()
    if include_passenger:
        allowed_groups.add("passenger")
        allowed_groups.add("transit")
    if include_freight:
        allowed_groups.add("freight")
    allowed_ids = set(assignments.loc[assignments["assignment_group"].isin(allowed_groups), "vehicleTypeId"].tolist())
    known_ids = set(assignments["vehicleTypeId"].tolist())
    if not allowed_ids:
        raise ValueError(
            "No vehicleTypeId values in the vehicle types input match the requested passenger/freight filter."
        )
    return allowed_ids, known_ids


# ---------------------------------------------------------------------------
# substep 1.0 — cache check
# ---------------------------------------------------------------------------

def _reuse_existing_outputs(raw_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    beam_emissions_for_inmap = _existing_output(raw_dir / "beam_emissions_for_inmap.parquet")
    if not beam_emissions_for_inmap:
        return None

    outputs = {
        "beam_activity_totals": _existing_output(raw_dir / "beam_activity_totals.parquet"),
        "beam_activity_correction_factors": _existing_output(raw_dir / "beam_activity_correction_factors.parquet"),
        "beam_emissions_by_county_process": _existing_output(raw_dir / "beam_emissions_by_county_process.parquet"),
        "beam_emissions_for_aermod": _existing_output(raw_dir / "beam_emissions_for_aermod.parquet"),
        "beam_emissions_for_inmap": beam_emissions_for_inmap,
    }
    logger.info(
        "%s reusing existing emissions outputs; skipping recomputation (inmap=%s, aermod=%s)",
        _step_label("1.0"),
        outputs["beam_emissions_for_inmap"],
        outputs["beam_emissions_for_aermod"],
    )
    return outputs


# ---------------------------------------------------------------------------
# substep 1.1 — group each intersection surface independently
# ---------------------------------------------------------------------------

def _build_zone_grouped_table(
    *,
    intersection_path: Optional[str],
    intersection_df: Optional[pd.DataFrame],
    zone_label: str,
    scratch_dir: Path,
) -> Optional[pd.DataFrame]:
    if not intersection_path and intersection_df is None:
        return None
    zone_id_col = f"{zone_label}_cell_id" if zone_label != "county" else "county_COUNTYFP"
    proportion_col = f"{zone_label}_proportion"
    link_length_col = f"{zone_label}_link_length_m"
    required_cols = {"linkId", zone_id_col, proportion_col, link_length_col}
    if intersection_df is not None:
        missing = [col for col in required_cols if col not in intersection_df.columns]
    else:
        probe = _load_intersection_subset_or_df(
            path=str(intersection_path),
            columns=list(required_cols),
            intersection_df=intersection_df,
        )
        missing = [col for col in required_cols if col not in probe.columns]
    if missing:
        raise ValueError(
            f"{_step_label('1.1')} requires canonical {zone_label} intersection columns. Missing: {missing}"
        )

    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(
            con,
            working_dir=scratch_dir,
            show_progress=False,
            profile="memory_heavy",
        )
        source = _register_frame_or_scan(
            con=con,
            relation_name=f"{zone_label}_intersection",
            path=intersection_path,
            columns=list(required_cols),
            frame=intersection_df,
        )
        grouped = con.execute(
            f"""
            SELECT
                "linkId" AS "linkId",
                "{zone_id_col}" AS "{zone_id_col}",
                SUM(COALESCE(TRY_CAST("{proportion_col}" AS DOUBLE), 0.0)) AS "{proportion_col}",
                SUM(COALESCE(TRY_CAST("{link_length_col}" AS DOUBLE), 0.0)) AS "{link_length_col}"
            FROM {source}
            GROUP BY 1, 2
            """
        ).fetchdf()
    finally:
        con.close()
    if grouped.empty:
        return None
    logger.info("%s BEAM %s mapping rows=%d", _step_label("1.1"), zone_label, len(grouped))
    return grouped


# ---------------------------------------------------------------------------
# substep 1.2 — allocate skims emissions to one surface
# ---------------------------------------------------------------------------

def _build_zone_allocated_table(
    *,
    grouped_df: Optional[pd.DataFrame],
    skims_df: pd.DataFrame,
    zone_label: str,
    scratch_dir: Path,
    step_id: str = "1.2",
) -> Optional[pd.DataFrame]:
    if grouped_df is None or grouped_df.empty:
        return None

    emission_cols = [
        c for c in skims_df.columns
        if c.startswith("tons_per_year_") and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    activity_cols = [
        c for c in ("totVMT", "totTrips")
        if c in skims_df.columns and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    zone_id_col = f"{zone_label}_cell_id" if zone_label != "county" else "county_COUNTYFP"
    proportion_col = f"{zone_label}_proportion"
    link_length_col = f"{zone_label}_link_length_m"

    has_road_category = "roadCategory" in skims_df.columns
    has_release_height = "source_release_height" in skims_df.columns
    extra_cols = (["roadCategory"] if has_road_category else []) + (["source_release_height"] if has_release_height else [])
    merge_cols = ["linkId"] + extra_cols + ["vehicleTypeId", "process"] + activity_cols + emission_cols
    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(
            con,
            working_dir=scratch_dir,
            show_progress=True,
            profile="memory_heavy",
        )
        con.register("grouped_df", grouped_df)
        con.register("skims_df", skims_df[merge_cols].copy())
        value_selects = [
            f"""
            COALESCE(TRY_CAST(s."{col}" AS DOUBLE), 0.0) * COALESCE(TRY_CAST(g."{proportion_col}" AS DOUBLE), 0.0)
            AS "{col}_{zone_label}_allocated"
            """.strip()
            for col in activity_cols + emission_cols
        ]
        extra_selects = ""
        if has_road_category:
            extra_selects += ',\n                trim(CAST(s."roadCategory" AS VARCHAR)) AS "roadCategory"'
        if has_release_height:
            extra_selects += ',\n                COALESCE(TRY_CAST(s."source_release_height" AS DOUBLE), 1.0) AS "source_release_height"'
        allocated = con.execute(
            f"""
            SELECT
                g."linkId" AS "linkId",
                trim(CAST(s."vehicleTypeId" AS VARCHAR)) AS "vehicleTypeId",
                trim(CAST(s."process" AS VARCHAR)) AS "process",
                g."{zone_id_col}" AS "{zone_id_col}",
                COALESCE(TRY_CAST(g."{proportion_col}" AS DOUBLE), 0.0) AS "{proportion_col}",
                COALESCE(TRY_CAST(g."{link_length_col}" AS DOUBLE), 0.0) AS "{link_length_col}"{extra_selects},
                {", ".join(value_selects)}
            FROM grouped_df AS g
            LEFT JOIN skims_df AS s
                ON g."linkId" = s."linkId"
            WHERE trim(CAST(s."vehicleTypeId" AS VARCHAR)) <> ''
              AND trim(CAST(s."process" AS VARCHAR)) <> ''
            """
        ).fetchdf()
    finally:
        con.close()
    if allocated.empty:
        return None
    logger.info("%s BEAM emissions allocated across %s rows=%d", _step_label(step_id), zone_label, len(allocated))
    return allocated


def _build_source_activity_totals(skims_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if not {"countyfp", "totVMT", "totTrips"}.issubset(skims_df.columns):
        return None
    activity = skims_df[["countyfp", "totVMT", "totTrips"]].copy()

    activity["countyfp"] = normalize_county_fips(activity["countyfp"])
    activity["totVMT"] = pd.to_numeric(activity["totVMT"], errors="coerce").fillna(0.0)
    activity["totTrips"] = pd.to_numeric(activity["totTrips"], errors="coerce").fillna(0.0)
    grouped = activity.groupby("countyfp", dropna=False).agg(
        totVMT=("totVMT", "sum"),
        totTrips=("totTrips", "sum"),
    ).reset_index()
    zero_null_mask = grouped["countyfp"].isna() & grouped["totVMT"].eq(0.0) & grouped["totTrips"].eq(0.0)
    if zero_null_mask.any():
        grouped = grouped.loc[~zero_null_mask].reset_index(drop=True)
    return grouped


def resolve_prepared_skims_path(input_root: Path) -> Optional[str]:
    candidate = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    return _existing_valid_skims_parquet(candidate)


def resolve_prepared_grouped_skims_path(input_root: Path) -> Optional[str]:
    candidate = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    return _existing_valid_skims_parquet(candidate)


def _resolve_staged_skims_input_path(manifest_inputs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if not manifest_inputs:
        return None
    for key in ["emissions_skims_input", "skims_from_events"]:
        if key in manifest_inputs:
            return resolve_required_manifest_input(manifest_inputs, key=key)
    return None


def prepare_staged_skims_for_processing(
    *,
    input_root: Path,
    skims_input_source: str,
    network_path: str,
    passenger_vehicle_types_path: Optional[str],
    freight_vehicle_types_path: Optional[str],
    beam_length_col: str,
    prepared_skims_group_cols: list[str],
    pollutants: list[str],
    pollutants_map: Dict[str, str],
    vehicle_category_metadata_file: str,
    annualization_days: dict[str, float],
    population_sample: float,
    transit_sample: float,
    include_passenger: bool,
    include_freight: bool,
    require_aermod_support: bool = False,
) -> pd.DataFrame:
    started = time.perf_counter()
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    prepared_skims_existing = _existing_valid_skims_parquet(prepared_skims_path)
    if prepared_skims_existing:
        _validate_prepared_skims_schema(
            path=prepared_skims_existing,
            require_aermod_support=require_aermod_support,
        )
        logger.info("Step 1: reusing prepared skims %s", prepared_skims_path)
        return read_table(prepared_skims_path)

    prepared_grouped_skims_path = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    prepared_grouped_existing = _existing_valid_skims_parquet(prepared_grouped_skims_path)
    if prepared_grouped_existing:
        logger.info("Step 1: reusing grouped prepared skims %s", prepared_grouped_skims_path)
    else:
        group_started = time.perf_counter()
        allowed_vehicle_type_ids, known_vehicle_type_ids = _resolve_assignment_filter_ids(
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            include_passenger=include_passenger,
            include_freight=include_freight,
        )
        prepare_skims_for_grid_allocation(
            skims_path=skims_input_source,
            output_path=str(prepared_grouped_skims_path),
            group_cols=list(prepared_skims_group_cols),
            required_pollutants=list(pollutants),
            pollutants_map=dict(pollutants_map),
            allowed_vehicle_type_ids=allowed_vehicle_type_ids,
            known_vehicle_type_ids=known_vehicle_type_ids,
        )
        logger.info(
            "Step 1: prepared grouped skims in %.2fs -> %s",
            time.perf_counter() - group_started,
            prepared_grouped_skims_path,
        )

    annualize_started = time.perf_counter()
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        network_path=network_path,
        beam_length_col=beam_length_col,
        group_cols=list(prepared_skims_group_cols),
        required_pollutants=list(pollutants),
        vehicle_category_metadata_file=vehicle_category_metadata_file,
        annualization_days=annualization_days,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        population_sample=float(population_sample),
        transit_sample=float(transit_sample),
    )
    logger.info(
        "Step 1: annualized prepared skims in %.2fs -> %s",
        time.perf_counter() - annualize_started,
        prepared_skims_path,
    )
    logger.info("Step 1: total skims preparation time %.2fs", time.perf_counter() - started)
    return read_table(prepared_skims_path)


def load_or_prepare_skims_df(
    *,
    input_root: Path,
    intersection_path: str,
    beam_length_col: str,
    prepared_skims_group_cols: list[str],
    pollutants: list[str],
    pollutants_map: Dict[str, str],
    vehicle_category_metadata_file: str,
    annualization_days: dict[str, float],
    population_sample: float,
    transit_sample: float,
    include_passenger: bool,
    include_freight: bool,
    manifest_inputs: Optional[Dict[str, Any]] = None,
    require_aermod_support: bool = False,
) -> pd.DataFrame:
    prepared_path = resolve_prepared_skims_path(input_root)
    if prepared_path:
        _validate_prepared_skims_schema(
            path=prepared_path,
            require_aermod_support=require_aermod_support,
        )
        logger.info("Step 1: using prepared skims %s", prepared_path)
        return read_table(prepared_path)

    if manifest_inputs is None:
        raise ValueError("Step 1 requires manifest_inputs to resolve inputs.network.")
    network_path = resolve_required_manifest_input(manifest_inputs, key="network")
    passenger_vehicle_types_path = resolve_required_manifest_input(manifest_inputs, key="passenger_vehicle_types_input")
    freight_vehicle_types_path = resolve_required_manifest_input(manifest_inputs, key="freight_vehicle_types_input")
    skims_input_source = _resolve_staged_skims_input_path(manifest_inputs)
    if skims_input_source:
        logger.info("Step 1: preparing skims input %s", skims_input_source)
        return prepare_staged_skims_for_processing(
            input_root=input_root,
            skims_input_source=skims_input_source,
            network_path=network_path,
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            beam_length_col=beam_length_col,
            prepared_skims_group_cols=prepared_skims_group_cols,
            pollutants=pollutants,
            pollutants_map=pollutants_map,
            vehicle_category_metadata_file=vehicle_category_metadata_file,
            annualization_days=annualization_days,
            population_sample=population_sample,
            transit_sample=transit_sample,
            include_passenger=include_passenger,
            include_freight=include_freight,
            require_aermod_support=require_aermod_support,
        )

    from .prepare_emissions_from_events import build_staged_skims_from_events

    events_skims_path = build_staged_skims_from_events(
        input_root=input_root,
        network_path=network_path,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        intersection_path=intersection_path,
        manifest_inputs=manifest_inputs,
    )
    if not events_skims_path:
        raise FileNotFoundError(
            "Could not find skims or events input under the staged input tree "
            f"{input_root}, including {input_root / 'skims'} and {input_root / 'events'}."
        )

    return prepare_staged_skims_for_processing(
        input_root=input_root,
        skims_input_source=events_skims_path,
        network_path=network_path,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        beam_length_col=beam_length_col,
        prepared_skims_group_cols=prepared_skims_group_cols,
        pollutants=pollutants,
        pollutants_map=pollutants_map,
        vehicle_category_metadata_file=vehicle_category_metadata_file,
        annualization_days=annualization_days,
        population_sample=population_sample,
        transit_sample=transit_sample,
        include_passenger=include_passenger,
        include_freight=include_freight,
        require_aermod_support=require_aermod_support,
    )
