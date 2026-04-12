"""Fleet Step 2: assign EMFAC ids to passenger bus and bike vehicle types.

Substeps:
2.1 Assign one EMFAC id to each passenger bus vehicle type.
2.2 Assign one EMFAC id to each passenger bike vehicle type.
2.3 Add an empty emfacId column to passenger other vehicle types.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import pandas as pd

from impacts.fleet.config import read_table
from impacts.fleet.config import resolve_workflow_path


_EMFAC_KEY_COLUMNS = ["vehicleCategory", "fuel", "modelYear"]


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


def _sanitize_emfac_component(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", _normalize_text(value)).strip("_")
    return token.replace("_", "")


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    return (
        f"{_sanitize_emfac_component(model_year)}"
        f"{_sanitize_emfac_component(vehicle_category)}"
        f"{_sanitize_emfac_component(fuel)}"
    )


def _model_year_sort_key(model_year: object) -> tuple[int, int, int]:
    token = _normalize_lower(model_year)
    range_match = re.fullmatch(r"(\d{4})to(\d{4})", token)
    if range_match is not None:
        start_year = int(range_match.group(1))
        end_year = int(range_match.group(2))
        return (1, end_year, start_year)
    pre_match = re.fullmatch(r"pre(\d{4})", token)
    if pre_match is not None:
        cutoff_year = int(pre_match.group(1))
        return (0, cutoff_year - 1, 0)
    post_match = re.fullmatch(r"post(\d{4})", token)
    if post_match is not None:
        cutoff_year = int(post_match.group(1))
        return (2, 9999, cutoff_year + 1)
    year_match = re.fullmatch(r"(\d{4})", token)
    if year_match is not None:
        year_value = int(year_match.group(1))
        return (1, year_value, year_value)
    return (0, -1, -1)


def _build_valid_emfac_candidates(config: dict[str, Any]) -> pd.DataFrame:
    emfac_config = config["emfac"]
    rates = read_table(
        emfac_config["rates_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["activity_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS + ["population", "total_vmt"],
    )
    fleet = read_table(
        emfac_config["fleet_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()

    activity_agg = (
        activity.groupby(_EMFAC_KEY_COLUMNS, dropna=False, as_index=False)[["population", "total_vmt"]]
        .sum()
    )
    valid_keys = rates.merge(fleet, on=_EMFAC_KEY_COLUMNS, how="inner")
    candidates = activity_agg.merge(valid_keys, on=_EMFAC_KEY_COLUMNS, how="inner")
    if candidates.empty:
        raise ValueError("No valid EMFAC passenger candidates remain after intersecting rates, activity, and fleet inputs")
    candidates["emfacId"] = candidates.apply(
        lambda row: _build_emfac_id(
            vehicle_category=row["vehicleCategory"],
            fuel=row["fuel"],
            model_year=row["modelYear"],
        ),
        axis=1,
    )
    return candidates


def _load_emfac_class_targets(config: dict[str, Any]) -> dict[str, str]:
    class_map_path = config["mapping"]["emfac_beam_class_map"]
    class_map = read_table(class_map_path, dtype=None)
    for column_name in ["group", "emfac", "beam"]:
        _require_column(class_map, column_name, "EMFAC BEAM class mapping file")

    prepared = class_map.copy()
    prepared["group_key"] = prepared["group"].apply(_normalize_lower)
    prepared["beam_key"] = prepared["beam"].apply(_normalize_text)
    matched = prepared[prepared["beam_key"].isin(["Bike", "MediumDutyPassenger"])].copy()

    bike_match = matched[(matched["group_key"] == "passenger") & (matched["beam_key"] == "Bike")]
    bus_match = matched[(matched["group_key"] == "transit") & (matched["beam_key"] == "MediumDutyPassenger")]
    if bike_match.empty:
        raise ValueError("EMFAC class mapping file is missing the passenger Bike mapping")
    if bus_match.empty:
        raise ValueError("EMFAC class mapping file is missing the transit MediumDutyPassenger mapping")

    return {
        "bike": _normalize_text(bike_match.iloc[0]["emfac"]),
        "bus": _normalize_text(bus_match.iloc[0]["emfac"]),
    }


def _load_emfac_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    fuel_map_path = config["mapping"]["emfac_beam_fuel_map"]
    fuel_map = read_table(fuel_map_path, dtype=None)
    for column_name in ["group", "emfac", "beam_primary", "beam_secondary"]:
        _require_column(fuel_map, column_name, "EMFAC BEAM fuel mapping file")

    prepared = fuel_map.copy()
    prepared["group_key"] = prepared["group"].apply(_normalize_lower)
    prepared["beam_primary_key"] = prepared["beam_primary"].apply(_normalize_lower)
    prepared["beam_secondary_key"] = prepared["beam_secondary"].apply(_normalize_lower)
    prepared["emfac_fuel"] = prepared["emfac"].apply(_normalize_text)
    return prepared


def _matched_emfac_fuels(
    *,
    beam_group: str,
    primary_fuel: object,
    secondary_fuel: object,
    fuel_map: pd.DataFrame,
) -> list[str]:
    group_key = _normalize_lower(beam_group)
    primary_key = _normalize_lower(primary_fuel)
    secondary_key = _normalize_lower(secondary_fuel)

    candidates = fuel_map[
        (fuel_map["group_key"] == group_key)
        & (fuel_map["beam_primary_key"] == primary_key)
    ].copy()
    if candidates.empty:
        return []

    matched = candidates[
        (candidates["beam_secondary_key"] == "")
        | (candidates["beam_secondary_key"] == "any")
        | (candidates["beam_secondary_key"] == secondary_key)
    ]["emfac_fuel"].drop_duplicates()
    return sorted(matched.astype(str).tolist())


def _select_best_emfac_candidate(
    *,
    vehicle_type_id: str,
    emfac_vehicle_category: str,
    emfac_fuels: list[str],
    candidates: pd.DataFrame,
) -> pd.Series:
    matched = candidates[
        (candidates["vehicleCategory"].astype(str) == str(emfac_vehicle_category))
        & (candidates["fuel"].astype(str).isin(emfac_fuels))
    ].copy()
    if matched.empty:
        raise ValueError(
            "No EMFAC candidate available for "
            f"vehicleTypeId={vehicle_type_id}, vehicleCategory={emfac_vehicle_category}, fuels={emfac_fuels}"
        )

    year_keys = matched["modelYear"].apply(_model_year_sort_key)
    matched["modelYearTier"] = year_keys.apply(lambda key: key[0])
    matched["modelYearEnd"] = year_keys.apply(lambda key: key[1])
    matched["modelYearStart"] = year_keys.apply(lambda key: key[2])
    matched = matched.sort_values(
        by=["population", "total_vmt", "modelYearTier", "modelYearEnd", "modelYearStart", "fuel", "vehicleCategory"],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    )
    return matched.iloc[0]


def _assign_emfac_ids_to_vehicle_types(
    *,
    vehicle_types: pd.DataFrame,
    beam_group: str,
    emfac_vehicle_category: str,
    fuel_map: pd.DataFrame,
    emfac_candidates: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(vehicle_types, "vehicleTypeId", "Passenger vehicle types")
    _require_column(vehicle_types, "primaryFuelType", "Passenger vehicle types")
    _require_column(vehicle_types, "secondaryFuelType", "Passenger vehicle types")

    prepared = vehicle_types.copy()
    assignments: list[str] = []
    for row in prepared.itertuples(index=False):
        emfac_fuels = _matched_emfac_fuels(
            beam_group=beam_group,
            primary_fuel=getattr(row, "primaryFuelType"),
            secondary_fuel=getattr(row, "secondaryFuelType"),
            fuel_map=fuel_map,
        )
        if not emfac_fuels:
            raise ValueError(
                "No EMFAC fuel mapping available for "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId')}, "
                f"primaryFuelType={getattr(row, 'primaryFuelType')}, "
                f"secondaryFuelType={getattr(row, 'secondaryFuelType')}"
            )
        selected = _select_best_emfac_candidate(
            vehicle_type_id=str(getattr(row, "vehicleTypeId")),
            emfac_vehicle_category=emfac_vehicle_category,
            emfac_fuels=emfac_fuels,
            candidates=emfac_candidates,
        )
        assignments.append(str(selected["emfacId"]))

    prepared["emfacId"] = assignments
    duplicate_vehicle_type_ids = prepared["vehicleTypeId"][prepared["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Duplicate vehicleTypeId values encountered while assigning EMFAC ids:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    return prepared


def _write_vehicle_types(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _ensure_empty_emfac_id_column(vehicle_types: pd.DataFrame) -> pd.DataFrame:
    prepared = vehicle_types.copy()
    prepared["emfacId"] = ""
    return prepared


def run_step2(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 2: assign EMFAC ids to passenger bus and bike vehicle types."""
    config = workflow["config"]

    emfac_candidates = _build_valid_emfac_candidates(config)
    emfac_class_targets = _load_emfac_class_targets(config)
    emfac_fuel_map = _load_emfac_fuel_mapping(config)

    bus_file = workflow.get("built_passenger_bus_vehicle_types_file")
    bike_file = workflow.get("built_passenger_bike_vehicle_types_file")
    other_file = workflow.get("built_passenger_other_vehicle_types_file")
    if not bus_file or not bike_file or not other_file:
        raise ValueError("Step 2 requires passenger bus, bike, and other vehicle-type files from Step 1")

    print("=== Step 2.1: assign emfacId to passenger bus vehicle types ===")
    bus_vehicle_types = read_table(str(bus_file), dtype=None)
    bus_with_emfac = _assign_emfac_ids_to_vehicle_types(
        vehicle_types=bus_vehicle_types,
        beam_group="transit",
        emfac_vehicle_category=emfac_class_targets["bus"],
        fuel_map=emfac_fuel_map,
        emfac_candidates=emfac_candidates,
    )
    bus_output_file = _write_vehicle_types(bus_with_emfac, str(bus_file))

    print("=== Step 2.2: assign emfacId to passenger bike vehicle types ===")
    bike_vehicle_types = read_table(str(bike_file), dtype=None)
    bike_with_emfac = _assign_emfac_ids_to_vehicle_types(
        vehicle_types=bike_vehicle_types,
        beam_group="passenger",
        emfac_vehicle_category=emfac_class_targets["bike"],
        fuel_map=emfac_fuel_map,
        emfac_candidates=emfac_candidates,
    )
    bike_output_file = _write_vehicle_types(bike_with_emfac, str(bike_file))

    print("=== Step 2.3: add empty emfacId to passenger other vehicle types ===")
    other_vehicle_types = read_table(str(other_file), dtype=None)
    other_with_emfac = _ensure_empty_emfac_id_column(other_vehicle_types)
    other_output_file = _write_vehicle_types(other_with_emfac, str(other_file))

    workflow["emfac_passenger_candidates"] = emfac_candidates
    workflow["built_passenger_bus_vehicle_types"] = bus_with_emfac
    workflow["built_passenger_bus_vehicle_types_file"] = bus_output_file
    workflow["built_passenger_bike_vehicle_types"] = bike_with_emfac
    workflow["built_passenger_bike_vehicle_types_file"] = bike_output_file
    workflow["built_passenger_other_vehicle_types"] = other_with_emfac
    workflow["built_passenger_other_vehicle_types_file"] = other_output_file
    return workflow
