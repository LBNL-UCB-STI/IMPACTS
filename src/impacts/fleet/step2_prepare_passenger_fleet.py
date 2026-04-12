"""Fleet Step 2: prepare passenger fleet inputs.

Substeps:
2.1 Read the configured passenger vehicles and the built vehicle-types table, then split vehicle types into sections.
2.2 Check whether passenger vehicleTypeIds are already a subset of car vehicle types.
2.3 Build the newer car vehicle-types file, recalculate probabilities, and store it.
2.4 Apply emissions placeholders to the prepared sections.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.fleet.config import read_table
from impacts.fleet.config import resolve_workflow_path

_EMISSIONS_PLACEHOLDER_COLUMNS = {
    "emfacId": "",
    "emissionsRatesFile": "",
}


def _read_csv(path_like: str) -> pd.DataFrame:
    return read_table(path_like)


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _normalize_bodytype(value: object) -> str:
    token = str(value).strip().lower()
    mapping = {
        "car": "car",
        "suv": "suv",
        "pickup": "pickup",
        "truck": "pickup",
        "van": "van",
        "minvan": "van",
    }
    return mapping.get(token, token)


def _normalize_atlas_fuel(value: object) -> str:
    token = str(value).strip().lower()
    mapping = {
        "conv": "conv",
        "ice": "conv",
        "gas": "conv",
        "gasoline": "conv",
        "diesel": "conv",
        "hybrid": "hybrid",
        "phev": "phev",
        "ev": "ev",
        "aev": "ev",
    }
    return mapping.get(token, token)


def _vehicle_type_id_from_fastsim_token(token: object) -> str:
    stem = str(token)
    if stem.endswith(".csv.gz"):
        stem = stem[:-7]
    elif stem.endswith(".csv"):
        stem = stem[:-4]
    if stem.endswith("_lookup_table"):
        stem = stem[:-13]
    match = re.match(r"^(?P<year>\d{4})_(?P<fuel>[^_]+)_(?P<name>.+)$", stem)
    if match is None:
        return stem
    name = match.group("name")
    if name.endswith("_Charge_Depleting"):
        name = name[: -len("_Charge_Depleting")]
    elif name.endswith("_Charge_Sustaining"):
        name = name[: -len("_Charge_Sustaining")]
    return f"{match.group('year')}_{name}"


def _build_fastsim_vehicle_type_mapping(
    fastsim_bodytype_mapping: pd.DataFrame,
    fastsim_adoptfuel_mapping: pd.DataFrame,
) -> pd.DataFrame:
    body_col = "bodytype" if "bodytype" in fastsim_bodytype_mapping.columns else "body_type"
    if body_col not in fastsim_bodytype_mapping.columns:
        raise ValueError("FASTSim bodytype mapping file is missing 'body_type'")
    id_col = "vehicleTypeId" if "vehicleTypeId" in fastsim_bodytype_mapping.columns else "vehicle_id"
    if id_col not in fastsim_bodytype_mapping.columns:
        raise ValueError("FASTSim bodytype mapping file is missing 'vehicle_id'")

    bodytypes = fastsim_bodytype_mapping[[id_col, body_col]].copy()
    bodytypes["vehicleTypeId"] = bodytypes[id_col].apply(_vehicle_type_id_from_fastsim_token)
    bodytypes["bodytype"] = bodytypes[body_col].astype(str)
    bodytypes = bodytypes[["vehicleTypeId", "bodytype"]].drop_duplicates()

    adoptfuel = fastsim_adoptfuel_mapping[["vehicleTypeId", "adopt_fuel"]].copy()
    adoptfuel["vehicleTypeId"] = adoptfuel["vehicleTypeId"].astype(str)
    adoptfuel["adopt_fuel"] = adoptfuel["adopt_fuel"].fillna("").astype(str)
    adoptfuel = adoptfuel[adoptfuel["adopt_fuel"].ne("")]
    adoptfuel = adoptfuel.drop_duplicates()

    prepared = bodytypes.merge(adoptfuel, on="vehicleTypeId", how="inner")
    extracted_year = (
        prepared["vehicleTypeId"]
        .astype(str)
        .str.extract(r"^(?P<leading_year>\d{4})|(?P<trailing_year>\d{4})$")
        .bfill(axis=1)
        .iloc[:, 0]
    )
    prepared["modelyear"] = pd.to_numeric(extracted_year, errors="coerce")
    return prepared[["vehicleTypeId", "bodytype", "modelyear", "adopt_fuel"]].drop_duplicates()


def _vehicle_types_are_subset_of_car_types(
    used_vehicle_type_ids: pd.Index,
    car_vehicle_types: pd.DataFrame,
) -> bool:
    car_vehicle_type_ids = pd.Index(car_vehicle_types["vehicleTypeId"].astype(str).unique())
    missing_vehicle_type_ids = used_vehicle_type_ids.difference(car_vehicle_type_ids)
    return missing_vehicle_type_ids.empty


def _filter_car_vehicle_types_to_atlas_year(
    car_vehicle_types: pd.DataFrame,
    atlas_year: Any,
) -> pd.DataFrame:
    prepared = car_vehicle_types.copy()
    _require_column(prepared, "vehicleTypeId", "Passenger car vehicle types")
    extracted_year = (
        prepared["vehicleTypeId"]
        .astype(str)
        .str.extract(r"^(?P<leading_year>\d{4})|(?P<trailing_year>\d{4})$")
        .bfill(axis=1)
        .iloc[:, 0]
    )
    prepared["modelyear"] = pd.to_numeric(extracted_year, errors="coerce")
    return prepared[prepared["modelyear"].le(pd.to_numeric(atlas_year, errors="coerce"))].copy()


def _assign_vehicle_types_to_vehicles(
    vehicles: pd.DataFrame,
    vehicle_type_mapping: pd.DataFrame,
    car_vehicle_types: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(vehicle_type_mapping, "vehicleTypeId", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "bodytype", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "modelyear", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "adopt_fuel", "Vehicle type mapping file")
    _require_column(car_vehicle_types, "vehicleTypeId", "Passenger car vehicle types")
    _require_column(car_vehicle_types, "sampleProbabilityWithinCategory", "Passenger car vehicle types")
    _require_column(vehicles, "bodytype", "Passenger vehicles file")
    _require_column(vehicles, "modelyear", "Passenger vehicles file")
    _require_column(vehicles, "adopt_fuel", "Passenger vehicles file")

    vehicles_prepared = vehicles.copy()
    vehicles_prepared["bodytype_key"] = vehicles_prepared["bodytype"].apply(_normalize_bodytype)
    vehicles_prepared["atlas_fuel_key"] = vehicles_prepared["adopt_fuel"].apply(_normalize_atlas_fuel)
    vehicles_prepared["modelyear"] = pd.to_numeric(vehicles_prepared["modelyear"], errors="coerce")
    missing_modelyear = vehicles_prepared["modelyear"].isna()
    extracted_vehicle_year = (
        vehicles_prepared.loc[missing_modelyear, "vehicleTypeId"]
        .astype(str)
        .str.extract(r"^(?P<leading_year>\d{4})|(?P<trailing_year>\d{4})$")
        .bfill(axis=1)
        .iloc[:, 0]
    )
    vehicles_prepared.loc[missing_modelyear, "modelyear"] = pd.to_numeric(extracted_vehicle_year, errors="coerce")

    mapping_prepared = vehicle_type_mapping.copy()
    mapping_prepared["bodytype_key"] = mapping_prepared["bodytype"].apply(_normalize_bodytype)
    mapping_prepared["atlas_fuel_key"] = mapping_prepared["adopt_fuel"].apply(_normalize_atlas_fuel)
    mapping_prepared["modelyear"] = pd.to_numeric(mapping_prepared["modelyear"], errors="coerce")

    car_probabilities = car_vehicle_types[["vehicleTypeId", "sampleProbabilityWithinCategory"]].copy()
    car_probabilities["vehicleTypeId"] = car_probabilities["vehicleTypeId"].astype(str)
    car_probabilities["carProbability"] = pd.to_numeric(
        car_probabilities["sampleProbabilityWithinCategory"],
        errors="coerce",
    ).fillna(0.0)
    car_probabilities = car_probabilities[["vehicleTypeId", "carProbability"]]

    mapping_prepared["mappingVehicleTypeId"] = mapping_prepared["vehicleTypeId"].astype(str)
    car_probabilities["carVehicleTypeId"] = car_probabilities["vehicleTypeId"].astype(str)
    candidates = mapping_prepared.merge(
        car_probabilities,
        left_on="vehicleTypeId",
        right_on="carVehicleTypeId",
        how="inner",
    )
    candidates["combinedProbability"] = candidates["carProbability"]

    if candidates.empty:
        raise ValueError("No candidate vehicle types remain after merging FASTSim mapping files with car vehicle types")

    rng = np.random.default_rng(0)
    assigned_vehicle_type_ids = pd.Series(index=vehicles_prepared.index, dtype="object")

    group_columns = ["bodytype_key", "atlas_fuel_key", "modelyear"]
    for group_key, vehicle_group in vehicles_prepared.groupby(group_columns, dropna=False):
        candidate_group = candidates[
            (candidates["bodytype_key"] == group_key[0])
            & (candidates["atlas_fuel_key"] == group_key[1])
            & (candidates["modelyear"] == group_key[2])
        ][["carVehicleTypeId", "combinedProbability"]].copy()
        if candidate_group.empty:
            raise ValueError(
                "No mapped car vehicle types available for passenger vehicles group "
                f"bodytype={group_key[0]}, adopt_fuel={group_key[1]}, modelyear={group_key[2]}"
            )
        probability_sum = candidate_group["combinedProbability"].sum()
        if probability_sum <= 0:
            candidate_group["combinedProbability"] = 1.0 / len(candidate_group)
        else:
            candidate_group["combinedProbability"] = candidate_group["combinedProbability"] / probability_sum
        assigned_vehicle_type_ids.loc[vehicle_group.index] = rng.choice(
            candidate_group["carVehicleTypeId"].to_numpy(),
            size=len(vehicle_group),
            p=candidate_group["combinedProbability"].to_numpy(),
        )

    original_vehicle_type_ids = vehicles_prepared["vehicleTypeId"].astype(str).str.replace("_", "", regex=False)
    assigned_vehicle_type_ids = assigned_vehicle_type_ids.astype(str).str.replace("_", "", regex=False)
    vehicles_prepared["vehicleTypeId"] = assigned_vehicle_type_ids + "_" + original_vehicle_type_ids
    return vehicles_prepared.drop(columns=["bodytype_key", "atlas_fuel_key"])


def _build_assigned_car_vehicle_types(
    prepared_vehicles: pd.DataFrame,
    car_vehicle_types: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(prepared_vehicles, "vehicleTypeId", "Prepared passenger vehicles")
    assigned_pairs = prepared_vehicles[["vehicleTypeId"]].copy()
    assigned_pairs["vehicleTypeId"] = assigned_pairs["vehicleTypeId"].astype(str)
    assigned_pairs["assignedCarVehicleTypeId"] = assigned_pairs["vehicleTypeId"].str.split("_", n=1).str[0]
    assigned_pairs = assigned_pairs.drop_duplicates()

    prepared = assigned_pairs.merge(
        car_vehicle_types,
        left_on="assignedCarVehicleTypeId",
        right_on="vehicleTypeId",
        how="left",
    )
    missing = prepared[prepared["vehicleTypeId_y"].isna()]["assignedCarVehicleTypeId"].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            "Missing assigned car vehicle types for vehicleTypeId values:\n"
            + "\n".join(missing.astype(str).tolist())
        )

    prepared = prepared.rename(columns={"vehicleTypeId_x": "combinedVehicleTypeId"})
    prepared = prepared.drop(columns=["vehicleTypeId_y", "assignedCarVehicleTypeId"])
    prepared = prepared.rename(columns={"combinedVehicleTypeId": "vehicleTypeId"})
    return prepared


def _parse_sample_probability_string(prob_string: object) -> tuple[object, object, object, object]:
    if pd.isna(prob_string) or prob_string == "":
        return None, None, None, None

    cleaned = str(prob_string).replace(" ", "").lower()
    income_match = re.search(r"income\|([^:]+):([0-9.]+)", cleaned)
    ridehail_match = re.search(r"ridehail\|([^:]+):([0-9.]+)", cleaned)

    income_bin = income_match.group(1) if income_match else None
    income_prob = float(income_match.group(2)) if income_match else None
    ridehail_bin = ridehail_match.group(1) if ridehail_match else None
    ridehail_prob = float(ridehail_match.group(2)) if ridehail_match else None
    return income_bin, income_prob, ridehail_bin, ridehail_prob


def _create_sample_probability_string(
    income_bin: object,
    income_prob: object,
    ridehail_bin: object,
    ridehail_prob: object,
) -> str:
    if income_bin is None and income_prob is None and ridehail_bin is None and ridehail_prob is None:
        return ""

    parts = []
    if income_bin is not None and income_prob is not None:
        parts.append(f"income|{income_bin}:{float(income_prob):.6f}")
    if ridehail_bin is not None and ridehail_prob is not None:
        parts.append(f"ridehail|{ridehail_bin}:{float(ridehail_prob):.6f}")
    return "; ".join(parts)


def _recalculate_vehicle_type_probabilities(vehicle_types: pd.DataFrame) -> pd.DataFrame:
    _require_column(vehicle_types, "vehicleCategory", "Passenger car vehicle types")
    _require_column(vehicle_types, "sampleProbabilityWithinCategory", "Passenger car vehicle types")
    _require_column(vehicle_types, "sampleProbabilityString", "Passenger car vehicle types")

    prepared = vehicle_types.copy()
    parsed_data = prepared["sampleProbabilityString"].apply(_parse_sample_probability_string)
    prepared["income_bin"] = parsed_data.apply(lambda x: x[0])
    prepared["income_prop"] = parsed_data.apply(lambda x: x[1])
    prepared["ridehail_bin"] = parsed_data.apply(lambda x: x[2])
    prepared["ridehail_prop"] = parsed_data.apply(lambda x: x[3])
    prepared["sampleProbabilityWithinCategory"] = pd.to_numeric(
        prepared["sampleProbabilityWithinCategory"],
        errors="coerce",
    ).fillna(0.0)

    category_totals = prepared.groupby("vehicleCategory")["sampleProbabilityWithinCategory"].transform("sum")
    positive_category_totals = category_totals > 0
    prepared.loc[positive_category_totals, "sampleProbabilityWithinCategory"] = (
        prepared.loc[positive_category_totals, "sampleProbabilityWithinCategory"] / category_totals.loc[positive_category_totals]
    )

    for category in prepared["vehicleCategory"].dropna().unique():
        category_mask = prepared["vehicleCategory"] == category
        for income_bin in prepared.loc[category_mask, "income_bin"].dropna().unique():
            mask = category_mask & (prepared["income_bin"] == income_bin)
            prob_sum = prepared.loc[mask, "income_prop"].sum()
            if prob_sum > 0:
                prepared.loc[mask, "income_prop"] = prepared.loc[mask, "income_prop"] / prob_sum
        for ridehail_bin in prepared.loc[category_mask, "ridehail_bin"].dropna().unique():
            mask = category_mask & (prepared["ridehail_bin"] == ridehail_bin)
            prob_sum = prepared.loc[mask, "ridehail_prop"].sum()
            if prob_sum > 0:
                prepared.loc[mask, "ridehail_prop"] = prepared.loc[mask, "ridehail_prop"] / prob_sum

    prepared["sampleProbabilityWithinCategory"] = prepared["sampleProbabilityWithinCategory"].map(lambda x: f"{float(x):.6f}")
    prepared["sampleProbabilityString"] = prepared.apply(
        lambda row: _create_sample_probability_string(
            row["income_bin"],
            row["income_prop"],
            row["ridehail_bin"],
            row["ridehail_prop"],
        ),
        axis=1,
    )
    return prepared.drop(columns=["income_bin", "income_prop", "ridehail_bin", "ridehail_prop"])


def _write_new_vehicle_types_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = Path(resolve_workflow_path(output_dir)) / "vehicleTypes--atlas--step1-prepared.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


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


def _add_emissions_placeholders(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column_name, default_value in _EMISSIONS_PLACEHOLDER_COLUMNS.items():
        if column_name not in prepared.columns:
            prepared[column_name] = default_value
        else:
            prepared[column_name] = prepared[column_name].fillna("").astype(str)
    return prepared


def run_step2(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 2: load passenger fleet inputs, split vehicle types, and prepare car types."""
    config = workflow["config"]

    print("\n=== Step 2.1: read built vehicle types and separate/store vehicle types by section ===")
    vehicles = _read_csv(config["atlas"]["vehicles_file"])
    vehicle_types = workflow.get("built_vehicle_types")
    if vehicle_types is None:
        vehicle_types = _read_csv(config["beam"]["vehicle_types_file"])
    else:
        vehicle_types = vehicle_types.copy()
    prepared_vehicles = vehicles.copy()
    prepared_vehicle_types_file = ""

    _require_column(vehicles, "vehicleTypeId", "Passenger vehicles file")
    _require_column(vehicle_types, "vehicleTypeId", "Passenger vehicle types file")

    sections = _split_vehicle_types(vehicle_types)

    print("\n=== Step 2.2: read vehicles_file and check vehicleTypeIds are a subset of car vehicle types ===")
    used_vehicle_type_ids = pd.Index(vehicles["vehicleTypeId"].astype(str).unique())
    vehicle_type_ids_are_subset = _vehicle_types_are_subset_of_car_types(used_vehicle_type_ids, sections["car"])
    if vehicle_type_ids_are_subset:
        sections["car"] = sections["car"][
            sections["car"]["vehicleTypeId"].astype(str).isin(used_vehicle_type_ids)
        ].copy()
        prepared_vehicle_types = sections["car"].copy()
    else:
        sections["car"] = _filter_car_vehicle_types_to_atlas_year(
            sections["car"],
            config["atlas"]["year"],
        )
        fastsim_bodytype_mapping = _read_csv(config["beam"]["fastsim_bodytype_xwalk_file"])
        fastsim_adoptfuel_mapping = _read_csv(config["beam"]["fastsim_adoptfuel_mapping_file"])
        vehicle_type_mapping = _build_fastsim_vehicle_type_mapping(
            fastsim_bodytype_mapping,
            fastsim_adoptfuel_mapping,
        )
        prepared_vehicles = _assign_vehicle_types_to_vehicles(
            vehicles,
            vehicle_type_mapping,
            sections["car"],
        )
        prepared_vehicle_types = _build_assigned_car_vehicle_types(
            prepared_vehicles,
            sections["car"],
        )

    sections = {
        name: _add_emissions_placeholders(frame)
        for name, frame in sections.items()
    }
    print("\n=== Step 2.3: recalculate probabilities for the newer car vehicle types file ===")
    prepared_vehicle_types = _recalculate_vehicle_type_probabilities(prepared_vehicle_types)
    if not vehicle_type_ids_are_subset:
        prepared_vehicle_types_file = _write_new_vehicle_types_file(
            prepared_vehicle_types,
            config["output"],
        )

    workflow["source_pax_vehicles"] = vehicles
    workflow["source_pax_vehicle_types"] = vehicle_types
    workflow["prepared_pax_vehicles"] = prepared_vehicles
    workflow["prepared_pax_car_vehicle_types"] = prepared_vehicle_types.copy()
    workflow["prepared_pax_bus_vehicle_types"] = sections["bus"]
    workflow["prepared_pax_bike_vehicle_types"] = sections["bike"]
    workflow["prepared_pax_other_vehicle_types"] = sections["other"]
    workflow["prepared_pax_vehicle_types"] = prepared_vehicle_types
    workflow["prepared_pax_vehicle_types_file"] = prepared_vehicle_types_file
    return workflow
