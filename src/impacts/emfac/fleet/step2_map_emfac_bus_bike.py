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

from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path


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
    emfac_config = config["activities"]
    rates = read_table(
        emfac_config["passenger_rates_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()
    activity = read_table(
        emfac_config["passenger_activity_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS + ["population_vehicles", "total_vmt_vehicle_miles_per_year"],
    )
    fleet = read_table(
        emfac_config["passenger_fleet_file"],
        dtype=None,
        columns=_EMFAC_KEY_COLUMNS,
    )[_EMFAC_KEY_COLUMNS].drop_duplicates()

    activity_agg = (
        activity.groupby(_EMFAC_KEY_COLUMNS, dropna=False, as_index=False)[
            ["population_vehicles", "total_vmt_vehicle_miles_per_year"]
        ]
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


def _load_emfac_category_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = read_table(config["mappings"]["emfac_category_fuel_mapping_file"], dtype=None)
    for column_name in [
        "group",
        "emfac_vehicle_category",
        "emfac_fuel",
        "beam_category",
        "adopt_fuel",
    ]:
        _require_column(frame, column_name, "EMFAC category fuel mapping file")
    prepared = frame.copy()
    prepared["group"] = prepared["group"].apply(_normalize_lower)
    prepared["emfac_vehicle_category"] = prepared["emfac_vehicle_category"].apply(_normalize_text)
    prepared["emfac_fuel"] = prepared["emfac_fuel"].apply(_normalize_text)
    prepared["beam_category"] = prepared["beam_category"].apply(_normalize_text)
    prepared["adopt_fuel_key"] = prepared["adopt_fuel"].apply(_normalize_lower)
    return prepared


def _load_step2_emfac_mapping_slice(
    *,
    config: dict[str, Any],
    beam_category: str,
    label: str,
) -> tuple[str, pd.DataFrame]:
    mapping = _load_emfac_category_fuel_mapping(config)
    mapping = mapping[
        (mapping["group"] == "passenger")
        & (mapping["beam_category"] == str(beam_category))
    ].copy()
    if mapping.empty:
        raise ValueError(
            "EMFAC category fuel mapping file is missing the "
            f"{label} passenger mapping slice for beam_category={beam_category}"
        )

    emfac_categories = mapping["emfac_vehicle_category"].drop_duplicates().astype(str).tolist()
    if len(emfac_categories) != 1:
        raise ValueError(
            "EMFAC category fuel mapping file defines multiple passenger EMFAC categories for "
            f"{label} beam_category={beam_category}: {emfac_categories}"
        )

    return emfac_categories[0], mapping


def _normalize_to_fastsim_adopt_fuel(*, fuel_domain: str, adopt_fuel: object) -> str:
    token = _normalize_lower(adopt_fuel)
    if fuel_domain == "ldv":
        mapping = {
            "gasoline": "conv",
            "diesel": "conv",
            "biodiesel": "conv",
            "cng": "conv",
            "naturalgas": "conv",
            "electricity": "ev",
            "hydrogen": "fuelcell",
            "electricity+gasoline": "phev",
            "electricity+diesel": "phev",
        }
        return mapping.get(token, token)
    mapping = {
        "diesel": "diesel",
        "gasoline": "gasoline",
        "biodiesel": "diesel",
        "cng": "naturalgas",
        "naturalgas": "naturalgas",
        "electricity": "electricity",
        "hydrogen": "hydrogen",
        "electricity+gasoline": "electricity",
        "electricity+diesel": "electricity",
    }
    return mapping.get(token, token)


def _matched_emfac_fuels(
    *,
    fuel_domain: str,
    emfac_vehicle_category: str,
    adopt_fuel: object,
    category_fuel_map: pd.DataFrame,
) -> list[str]:
    adopt_fuel_key = _normalize_to_fastsim_adopt_fuel(
        fuel_domain=_normalize_lower(fuel_domain),
        adopt_fuel=adopt_fuel,
    )
    candidates = category_fuel_map[
        (category_fuel_map["emfac_vehicle_category"] == str(emfac_vehicle_category))
        & (category_fuel_map["adopt_fuel_key"] == adopt_fuel_key)
    ].copy()
    if candidates.empty:
        return []
    matched = candidates["emfac_fuel"].drop_duplicates()
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
        by=[
            "population_vehicles",
            "total_vmt_vehicle_miles_per_year",
            "modelYearTier",
            "modelYearEnd",
            "modelYearStart",
            "fuel",
            "vehicleCategory",
        ],
        ascending=[False, False, False, False, False, True, True],
        kind="mergesort",
    )
    return matched.iloc[0]


def _assign_emfac_ids_to_vehicle_types(
    *,
    vehicle_types: pd.DataFrame,
    fuel_domain: str,
    emfac_vehicle_category: str,
    category_fuel_map: pd.DataFrame,
    emfac_candidates: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(vehicle_types, "vehicleTypeId", "Passenger vehicle types")
    _require_column(vehicle_types, "adopt_fuel", "Passenger vehicle types")

    prepared = vehicle_types.copy()
    assignments: list[str] = []
    for row in prepared.itertuples(index=False):
        emfac_fuels = _matched_emfac_fuels(
            fuel_domain=fuel_domain,
            emfac_vehicle_category=emfac_vehicle_category,
            adopt_fuel=getattr(row, "adopt_fuel"),
            category_fuel_map=category_fuel_map,
        )
        if not emfac_fuels:
            raise ValueError(
                "No EMFAC fuel mapping available for "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId')}, "
                f"adopt_fuel={getattr(row, 'adopt_fuel')}"
            )
        selected = _select_best_emfac_candidate(
            vehicle_type_id=str(getattr(row, "vehicleTypeId")),
            emfac_vehicle_category=emfac_vehicle_category,
            emfac_fuels=emfac_fuels,
            candidates=emfac_candidates,
        )
        assignments.append(str(selected["emfacId"]))

    prepared["emfacId"] = assignments
    prepared["emfacVehicleCategory"] = str(emfac_vehicle_category)
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


def _run_step2_substep_assign_bus(
    *,
    bus_file: str,
    emfac_candidates: pd.DataFrame,
    bus_emfac_category: str,
    bus_category_fuel_map: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    bus_vehicle_types = read_table(str(bus_file), dtype=None)
    bus_with_emfac = _assign_emfac_ids_to_vehicle_types(
        vehicle_types=bus_vehicle_types,
        fuel_domain="mhdv",
        emfac_vehicle_category=bus_emfac_category,
        category_fuel_map=bus_category_fuel_map,
        emfac_candidates=emfac_candidates,
    )
    return bus_with_emfac, _write_vehicle_types(bus_with_emfac, str(bus_file))


def _run_step2_substep_assign_bike(
    *,
    bike_file: str,
    emfac_candidates: pd.DataFrame,
    bike_emfac_category: str,
    bike_category_fuel_map: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    bike_vehicle_types = read_table(str(bike_file), dtype=None)
    bike_with_emfac = _assign_emfac_ids_to_vehicle_types(
        vehicle_types=bike_vehicle_types,
        fuel_domain="ldv",
        emfac_vehicle_category=bike_emfac_category,
        category_fuel_map=bike_category_fuel_map,
        emfac_candidates=emfac_candidates,
    )
    return bike_with_emfac, _write_vehicle_types(bike_with_emfac, str(bike_file))


def _run_step2_substep_prepare_other(*, other_file: str) -> tuple[pd.DataFrame, str]:
    other_vehicle_types = read_table(str(other_file), dtype=None)
    other_with_emfac = _ensure_empty_emfac_id_column(other_vehicle_types)
    return other_with_emfac, _write_vehicle_types(other_with_emfac, str(other_file))


def run_step2(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 2: assign EMFAC ids to passenger bus and bike vehicle types."""
    config = workflow["config"]

    emfac_candidates = _build_valid_emfac_candidates(config)
    bus_emfac_category, bus_category_fuel_map = _load_step2_emfac_mapping_slice(
        config=config,
        beam_category="MediumDutyPassenger",
        label="bus",
    )
    bike_emfac_category, bike_category_fuel_map = _load_step2_emfac_mapping_slice(
        config=config,
        beam_category="Bike",
        label="bike",
    )

    bus_file = workflow.get("built_passenger_bus_vehicle_types_file")
    bike_file = workflow.get("built_passenger_bike_vehicle_types_file")
    other_file = workflow.get("built_passenger_other_vehicle_types_file")
    if not bus_file or not bike_file or not other_file:
        raise ValueError("Step 2 requires passenger bus, bike, and other vehicle-type files from Step 1")

    print("=== Step 2.1: assign emfacId to passenger bus vehicle types ===")
    bus_with_emfac, bus_output_file = _run_step2_substep_assign_bus(
        bus_file=str(bus_file),
        emfac_candidates=emfac_candidates,
        bus_emfac_category=bus_emfac_category,
        bus_category_fuel_map=bus_category_fuel_map,
    )

    print("=== Step 2.2: assign emfacId to passenger bike vehicle types ===")
    bike_with_emfac, bike_output_file = _run_step2_substep_assign_bike(
        bike_file=str(bike_file),
        emfac_candidates=emfac_candidates,
        bike_emfac_category=bike_emfac_category,
        bike_category_fuel_map=bike_category_fuel_map,
    )

    print("=== Step 2.3: add empty emfacId to passenger other vehicle types ===")
    other_with_emfac, other_output_file = _run_step2_substep_prepare_other(other_file=str(other_file))

    workflow["emfac_passenger_candidates"] = emfac_candidates
    workflow["built_passenger_bus_vehicle_types"] = bus_with_emfac
    workflow["built_passenger_bus_vehicle_types_file"] = bus_output_file
    workflow["built_passenger_bike_vehicle_types"] = bike_with_emfac
    workflow["built_passenger_bike_vehicle_types_file"] = bike_output_file
    workflow["built_passenger_other_vehicle_types"] = other_with_emfac
    workflow["built_passenger_other_vehicle_types_file"] = other_output_file
    return workflow
