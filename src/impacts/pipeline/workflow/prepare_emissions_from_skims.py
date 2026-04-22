"""Shared helpers for skims preparation and grid allocation inputs."""

from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from ...common import normalize_county_fips
from ...common import prepare_skims_for_grid_allocation
from ...common import prepared_table_target
from ...common import read_table
from ...common import resolve_required_manifest_input
from ...common import register_managed_input
from ...manifest.file_ops import file_entry
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
_EMFAC_MODEL_YEAR_GROUP_PATTERN = re.compile(r"^(pre\d+|\d{4}to\d{4}|post\d+)")


# ---------------------------------------------------------------------------
def _existing_output(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _load_intersection_subset(path: str, columns: List[str]) -> pd.DataFrame:
    target = Path(path)
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


def _normalize_vehicle_type_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _row_is_transit_vehicle_type(row: pd.Series) -> bool:
    vehicle_type_id = _normalize_vehicle_type_token(row.get("vehicleTypeId"))
    if vehicle_type_id and _TRANSIT_VEHICLETYPE_PATTERN.search(vehicle_type_id):
        return True
    for candidate in ("emfacVehicleCategory", "vehicleCategory"):
        token = _normalize_vehicle_type_token(row.get(candidate)).lower()
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

    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_vehicle_type_token)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()
    prepared["assignment_group"] = prepared.apply(
        lambda row: _classify_vehicle_type_assignment(row, source_name="combined_vehicle_types"),
        axis=1,
    )
    prepared["emfacId"] = prepared["emfacId"].map(_normalize_vehicle_type_token)
    prepared["modelYear"] = prepared["emfacId"].str.extract(_EMFAC_MODEL_YEAR_GROUP_PATTERN, expand=False)
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


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
        return
    if path.name.lower().endswith(".csv.gz"):
        frame.to_csv(path, index=False, compression="gzip")
        return
    raise ValueError(f"Unsupported table output format: {path}")


def _filter_prepared_skims_by_assignment(
    prepared: pd.DataFrame,
    *,
    passenger_vehicle_types_path: Optional[str],
    freight_vehicle_types_path: Optional[str],
    include_passenger: bool,
    include_freight: bool,
) -> pd.DataFrame:
    if include_passenger and include_freight:
        return prepared
    if not include_passenger and not include_freight:
        raise ValueError("At least one of include_passenger or include_freight must be true.")
    if not passenger_vehicle_types_path or not freight_vehicle_types_path:
        raise ValueError(
            "Vehicle types input is required to filter prepared skims by passenger/freight assignment."
        )
    if "vehicleTypeId" not in prepared.columns:
        raise ValueError("Prepared skims must include vehicleTypeId for passenger/freight filtering.")

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
    if not allowed_ids:
        raise ValueError(
            "No vehicleTypeId values in the vehicle types input match the requested passenger/freight filter."
        )

    result = prepared.copy()
    result["vehicleTypeId"] = result["vehicleTypeId"].map(_normalize_vehicle_type_token)
    observed_ids = set(result["vehicleTypeId"].dropna().tolist())
    missing_ids = sorted(observed_ids - set(assignments["vehicleTypeId"].tolist()))
    if missing_ids:
        raise ValueError(
            "Could not assign some skim vehicleTypeId values to passenger or freight using "
            f"the configured passenger/freight vehicle types files: sample={missing_ids[:10]}"
        )

    filtered = result.loc[result["vehicleTypeId"].isin(allowed_ids)].copy()
    logger.info(
        "%s filtered prepared skims by assignment (include_passenger=%s, include_freight=%s): %d -> %d rows",
        _step_label("1.0"),
        include_passenger,
        include_freight,
        len(result),
        len(filtered),
    )
    return filtered


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
) -> Optional[pd.DataFrame]:
    if not intersection_path and intersection_df is None:
        return None
    zone_id_col = f"{zone_label}_cell_id" if zone_label != "county" else "countyfp"
    proportion_col = f"{zone_label}_zone_edge_proportion"
    edge_length_col = f"{zone_label}_edge_link_length_m"
    zone_length_col = f"{zone_label}_zone_link_length_m"
    required_cols = {"linkId", zone_id_col, proportion_col, edge_length_col, zone_length_col}
    intersection = _load_intersection_subset_or_df(
        path=str(intersection_path),
        columns=list(required_cols),
        intersection_df=intersection_df,
    )
    missing = [col for col in required_cols if col not in intersection.columns]
    if missing:
        raise ValueError(
            f"{_step_label('1.1')} requires canonical {zone_label} intersection columns. Missing: {missing}"
        )
    grouped = intersection.copy()
    for col in [proportion_col, edge_length_col, zone_length_col]:
        grouped[col] = pd.to_numeric(grouped[col], errors="coerce").fillna(0.0)
    group_cols = ["linkId", zone_id_col]
    grouped = grouped.groupby(group_cols, dropna=False)[[proportion_col, edge_length_col, zone_length_col]].sum().reset_index()
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
    zone_id_col = f"{zone_label}_cell_id" if zone_label != "county" else "countyfp"
    proportion_col = f"{zone_label}_zone_edge_proportion"
    edge_length_col = f"{zone_label}_edge_link_length_m"
    zone_length_col = f"{zone_label}_zone_link_length_m"

    merge_cols = ["linkId", "vehicleTypeId", "process"] + activity_cols + emission_cols
    allocated = grouped_df.merge(skims_df[merge_cols].copy(), how="left", on="linkId")
    allocated["vehicleTypeId"] = allocated["vehicleTypeId"].map(_normalize_vehicle_type_token)
    allocated["process"] = allocated["process"].map(_normalize_vehicle_type_token)
    allocated = allocated.loc[allocated["vehicleTypeId"].ne("") & allocated["process"].ne("")].copy()
    if allocated.empty:
        return None

    proportion = pd.to_numeric(allocated[proportion_col], errors="coerce").fillna(0.0)
    for col in activity_cols:
        allocated[f"{col}_{zone_label}_allocated"] = pd.to_numeric(allocated[col], errors="coerce").fillna(0.0) * proportion
    for col in emission_cols:
        allocated[f"{col}_{zone_label}_allocated"] = pd.to_numeric(allocated[col], errors="coerce").fillna(0.0) * proportion

    keep_cols = ["linkId", "vehicleTypeId", "process", zone_id_col, proportion_col, edge_length_col, zone_length_col]
    keep_cols.extend([f"{col}_{zone_label}_allocated" for col in activity_cols + emission_cols])
    allocated = allocated[keep_cols].copy()
    logger.info("%s BEAM emissions allocated across %s rows=%d", _step_label("1.2"), zone_label, len(allocated))
    return allocated


def _build_source_activity_totals(skims_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if {"countyfp", "distanceMiles_county", "tripCount_county"}.issubset(skims_df.columns):
        activity = skims_df[["countyfp", "distanceMiles_county", "tripCount_county"]].copy()
        vmt_col = "distanceMiles_county"
        trip_col = "tripCount_county"
    elif {"countyfp", "totVMT", "totTrips"}.issubset(skims_df.columns):
        activity = skims_df[["countyfp", "totVMT", "totTrips"]].copy()
        vmt_col = "totVMT"
        trip_col = "totTrips"
    else:
        return None

    activity["countyfp"] = normalize_county_fips(activity["countyfp"])
    activity[vmt_col] = pd.to_numeric(activity[vmt_col], errors="coerce").fillna(0.0)
    activity[trip_col] = pd.to_numeric(activity[trip_col], errors="coerce").fillna(0.0)
    grouped = activity.groupby("countyfp", dropna=False).agg(
        totVMT=(vmt_col, "sum"),
        totTrips=(trip_col, "sum"),
    ).reset_index()
    zero_null_mask = grouped["countyfp"].isna() & grouped["totVMT"].eq(0.0) & grouped["totTrips"].eq(0.0)
    if zero_null_mask.any():
        grouped = grouped.loc[~zero_null_mask].reset_index(drop=True)
    return grouped


def resolve_prepared_skims_path(input_root: Path) -> Optional[str]:
    candidate = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    if candidate.exists():
        return str(candidate)
    return None


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
    annualization_days_or_file: float | str,
    population_sample: float,
    transit_sample: float,
    include_passenger: bool,
    include_freight: bool,
) -> pd.DataFrame:
    prepared_grouped_skims_path = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    prepared_grouped = prepare_skims_for_grid_allocation(
        skims_path=skims_input_source,
        output_path=str(prepared_grouped_skims_path),
        group_cols=list(prepared_skims_group_cols),
        required_pollutants=list(pollutants),
        pollutants_map=dict(pollutants_map),
    )
    prepared_grouped = _filter_prepared_skims_by_assignment(
        prepared_grouped,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        include_passenger=include_passenger,
        include_freight=include_freight,
    )
    _write_frame(prepared_grouped, prepared_grouped_skims_path)
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        network_path=network_path,
        beam_length_col=beam_length_col,
        group_cols=list(prepared_skims_group_cols),
        required_pollutants=list(pollutants),
        annualization_days_or_file=annualization_days_or_file,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        population_sample=float(population_sample),
        transit_sample=float(transit_sample),
    )
    return read_table(prepared_skims_path)


def load_or_prepare_skims_df(
    *,
    input_root: Path,
    intersection_path: str,
    beam_length_col: str,
    prepared_skims_group_cols: list[str],
    pollutants: list[str],
    pollutants_map: Dict[str, str],
    annualization_days_or_file: float | str,
    population_sample: float,
    transit_sample: float,
    include_passenger: bool,
    include_freight: bool,
    manifest_inputs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    prepared_path = resolve_prepared_skims_path(input_root)
    if prepared_path:
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
            annualization_days_or_file=annualization_days_or_file,
            population_sample=population_sample,
            transit_sample=transit_sample,
            include_passenger=include_passenger,
            include_freight=include_freight,
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
        annualization_days_or_file=annualization_days_or_file,
        population_sample=population_sample,
        transit_sample=transit_sample,
        include_passenger=include_passenger,
        include_freight=include_freight,
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def prepare_skims_inputs(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    processing,
    skims_input_source: Optional[str],
    network_path: str,
    passenger_vehicle_types_path: Optional[str],
    freight_vehicle_types_path: Optional[str],
    intersection_path: Optional[str] = None,
) -> dict[str, Any]:
    event_inputs: Optional[Dict[str, Any]] = None
    if not skims_input_source:
        if not intersection_path:
            raise ValueError(
                "intersection_path is required when deriving skims from events."
            )
        logger.info("Step 1: no skims artifact found; building skims from local events")
        from .prepare_emissions_from_events import prepare_events_inputs

        event_inputs = prepare_events_inputs(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            network_path=network_path,
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            intersection_path=intersection_path,
            beam_length_col=processing.beam_length_col,
            prepared_skims_group_cols=list(processing.prepared_skims_group_cols),
            pollutants=list(processing.pollutants),
            pollutants_map=dict(processing.pollutants_map),
            annualization_days_or_file=processing.annualization_days_or_file,
            population_sample=float(processing.population_sample),
            transit_sample=float(processing.transit_sample),
            include_passenger=bool(processing.include_passenger),
            include_freight=bool(processing.include_freight),
        )
        if event_inputs is None:
            raise FileNotFoundError(
                "No skimsEmissions or events file found under the staged input tree."
            )
        skims_input_source = event_inputs["skims_path"]

    staged_skims_input = register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="emissions_skims_input",
        source_path=skims_input_source,
        relative_target=f"skims/{Path(skims_input_source).name}",
        optional=True,
        prefer_reference=True,
        metadata={
            "workflow_step": "step1_prepare_skims_inputs",
            "artifact_family": "emissions_skims_input",
        },
    )

    prepared_grouped_skims_path = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    canonical_pollutants = list(processing.pollutants)
    prepared_grouped = prepare_skims_for_grid_allocation(
        skims_path=staged_skims_input,
        output_path=str(prepared_grouped_skims_path),
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=canonical_pollutants,
        pollutants_map=dict(processing.pollutants_map),
    )
    prepared_grouped = _filter_prepared_skims_by_assignment(
        prepared_grouped,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        include_passenger=bool(processing.include_passenger),
        include_freight=bool(processing.include_freight),
    )
    _write_frame(prepared_grouped, prepared_grouped_skims_path)
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        network_path=network_path,
        beam_length_col=processing.beam_length_col,
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=canonical_pollutants,
        annualization_days_or_file=processing.annualization_days_or_file,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        population_sample=float(processing.population_sample),
        transit_sample=float(processing.transit_sample),
    )
    manifest_inputs["prepared_skims_grouped"] = file_entry(
        kind="local",
        path=str(prepared_grouped_skims_path),
        staged_path=str(prepared_grouped_skims_path),
        optional=True,
    )
    manifest_inputs["prepared_skims_input"] = file_entry(
        kind="local",
        path=str(prepared_skims_path),
        staged_path=str(prepared_skims_path),
        optional=True,
    )
    skims_df = event_inputs["skims_df"] if event_inputs is not None else read_table(prepared_skims_path)
    source_activity_df = (
        event_inputs["activity_df"]
        if event_inputs is not None
        else _build_source_activity_totals(skims_df)
    )
    return {
        "staged_skims_input": str(staged_skims_input),
        "prepared_grouped_skims_path": str(prepared_grouped_skims_path),
        "prepared_skims_path": str(prepared_skims_path),
        "skims_df": skims_df,
        "activity_path": None if event_inputs is None else event_inputs["activity_path"],
        "activity_df": source_activity_df,
    }
