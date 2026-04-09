"""Fleet Step 1: prepare passenger fleet inputs.

Substeps:
1.1 Read the configured passenger vehicles and vehicle types files.
1.2 Split vehicle types into car, bus, bike, and other groups.
1.3 Keep only car vehicle types that are referenced by the vehicles file.
1.4 Apply mapped curb weights and add empty emissions references.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from impacts.fleet.config import read_table

_EMISSIONS_PLACEHOLDER_COLUMNS = {
    "emfacId": "",
    "emissionsRatesFile": "",
}


def _read_csv(path_like: str) -> pd.DataFrame:
    return read_table(path_like)


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _prepare_vehicle_types(vehicles: pd.DataFrame, vehicle_types: pd.DataFrame) -> pd.DataFrame:
    used_vehicle_type_ids = pd.Index(vehicles["vehicleTypeId"].astype(str).unique())
    prepared = vehicle_types[vehicle_types["vehicleTypeId"].astype(str).isin(used_vehicle_type_ids)].copy()
    return prepared


def _split_vehicle_types(vehicle_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    category = vehicle_types["vehicleCategory"].astype(str)
    vehicle_type_id = vehicle_types["vehicleTypeId"].astype(str)
    bus_mask = category.eq("MediumDutyPassenger") & vehicle_type_id.str.contains("BUS-", case=False, na=False)
    return {
        "car": vehicle_types[category.eq("Car")].copy(),
        "bus": vehicle_types[bus_mask].copy(),
        "bike": vehicle_types[category.eq("Bike")].copy(),
        "other": vehicle_types[~(category.eq("Car") | bus_mask | category.eq("Bike"))].copy(),
    }


def _apply_curb_weight_mapping(
    frame: pd.DataFrame,
    curb_weight_mapping: pd.DataFrame,
    *,
    section_name: str,
) -> pd.DataFrame:
    prepared = frame.copy()
    mapping = curb_weight_mapping.copy()
    _require_column(mapping, "vehicleTypeId", "Curb weight mapping file")
    _require_column(mapping, "curbWeightInKg", "Curb weight mapping file")
    if "curbWeightSource" not in mapping.columns:
        mapping["curbWeightSource"] = ""
    mapping = mapping[["vehicleTypeId", "curbWeightInKg", "curbWeightSource"]].copy()
    mapping["vehicleTypeId"] = mapping["vehicleTypeId"].astype(str)
    mapping["curbWeightInKg"] = mapping["curbWeightInKg"].astype(str)
    mapping["curbWeightSource"] = mapping["curbWeightSource"].fillna("").astype(str)
    prepared = prepared.merge(mapping, on="vehicleTypeId", how="left")
    missing = prepared[prepared["curbWeightInKg"].isna() | prepared["curbWeightInKg"].eq("")]["vehicleTypeId"].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            "Missing curbWeightInKg mapping for vehicleTypeId values:\n"
            + "\n".join(missing.astype(str).tolist())
        )
    prepared["curbWeightInKg"] = prepared["curbWeightInKg"].astype(str)
    if not prepared.empty:
        print(f"\nPassenger vehicle-type curb weight estimates for {section_name}:")
        reason_counts = prepared["curbWeightSource"].replace("", "unspecified mapping source").value_counts()
        for reason, count in reason_counts.items():
            print(f"  - {reason}: {count}")
    prepared.drop(columns=["curbWeightSource"], inplace=True)
    return prepared


def _add_emissions_placeholders(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column_name, default_value in _EMISSIONS_PLACEHOLDER_COLUMNS.items():
        if column_name not in prepared.columns:
            prepared[column_name] = default_value
        else:
            prepared[column_name] = prepared[column_name].fillna("").astype(str)
    return prepared


def run_step1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 1: load passenger fleet inputs, split vehicle types, and prepare car types."""
    config = workflow["config"]
    vehicles = _read_csv(config["atlas"]["vehicles_file"])
    vehicle_types = _read_csv(config["atlas"]["vehicles_types_file"])
    curb_weight_mapping = _read_csv(config["atlas"]["curb_weight_mapping_file"])

    _require_column(vehicles, "vehicleTypeId", "Passenger vehicles file")
    _require_column(vehicle_types, "vehicleTypeId", "Passenger vehicle types file")

    sections = _split_vehicle_types(vehicle_types)
    sections["car"] = _prepare_vehicle_types(vehicles, sections["car"])
    sections = {
        name: _add_emissions_placeholders(
            _apply_curb_weight_mapping(frame, curb_weight_mapping, section_name=name)
        )
        for name, frame in sections.items()
    }
    prepared_vehicle_types = pd.concat(
        [sections["car"], sections["bus"], sections["bike"], sections["other"]],
        ignore_index=True,
    )

    workflow["source_pax_vehicles"] = vehicles
    workflow["source_pax_vehicle_types"] = vehicle_types
    workflow["prepared_pax_vehicles"] = vehicles.copy()
    workflow["prepared_pax_car_vehicle_types"] = sections["car"]
    workflow["prepared_pax_bus_vehicle_types"] = sections["bus"]
    workflow["prepared_pax_bike_vehicle_types"] = sections["bike"]
    workflow["prepared_pax_other_vehicle_types"] = sections["other"]
    workflow["prepared_pax_vehicle_types"] = prepared_vehicle_types
    return workflow
