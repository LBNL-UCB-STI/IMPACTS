"""Fleet Step 1: build BEAM passenger vehicle types for the target ATLAS year.

Substeps:
1.1 Load beam.vehicle_types_file, inspect encoded model years, and filter or rebuild
    vehicle types for <= atlas.year.
1.2 Load ATLAS vehicles, households, and persons, then calculate fleetShare and
    representative income bins for unique ATLAS bodytype/adopt_fuel/modelyear rows.
1.3 Map built FASTSim vehicle types to ATLAS rows and expand the final Step 1
    vehicle types using combined ids.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.fleet.config import read_table
from impacts.fleet.config import resolve_workflow_path


def _read_csv(path_like: str, *, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    return read_table(path_like, columns=columns)


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _extract_model_year_from_vehicle_type_id(series: pd.Series) -> pd.Series:
    extracted = (
        series.astype(str)
        .str.extract(r"^(?P<leading_year>\d{4})|(?P<trailing_year>\d{4})$")
        .bfill(axis=1)
        .iloc[:, 0]
    )
    return pd.to_numeric(extracted, errors="coerce")


def _normalize_bodytype(value: object) -> str:
    token = str(value).strip().lower()
    mapping = {
        "car": "car",
        "suv": "suv",
        "pickup": "pickup",
        "truck": "pickup",
        "van": "minvan",
        "minvan": "minvan",
    }
    return mapping.get(token, token)


def _combine_vehicle_type_ids(fastsim_vehicle_type_id: object, atlas_vehicle_type_id: object) -> str:
    fastsim_token = str(fastsim_vehicle_type_id).replace("_", "")
    atlas_token = str(atlas_vehicle_type_id).replace("_", "")
    return f"{fastsim_token}_{atlas_token}"


def _capitalize_token(value: object) -> str:
    token = str(value)
    return token[:1].upper() + token[1:] if token else token


def _build_atlas_vehicle_type_ids(
    bodytypes: pd.Series,
    adopt_fuels: pd.Series,
    modelyears: pd.Series,
) -> pd.Series:
    body = bodytypes.astype(str).str.strip()
    body = body.str[:1].str.upper() + body.str[1:]
    fuel = adopt_fuels.astype(str).str.strip()
    fuel = fuel.str[:1].str.upper() + fuel.str[1:]
    years = pd.to_numeric(modelyears, errors="coerce")
    year_tokens = years.map(lambda value: str(int(value)) if pd.notna(value) else str(value))
    return body + "_" + fuel + "_" + year_tokens


def _write_new_vehicle_types_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = Path(resolve_workflow_path(output_dir)) / "vehicleTypes--beam--step1-built.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _write_vehicle_type_crosswalk_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = Path(resolve_workflow_path(output_dir)) / "vehicle_type_atlas_crosswalk.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _build_atlas_vehicle_type_targets(vehicles: pd.DataFrame) -> pd.DataFrame:
    _require_column(vehicles, "bodytype", "ATLAS vehicles file")
    _require_column(vehicles, "modelyear", "ATLAS vehicles file")
    _require_column(vehicles, "adopt_fuel", "ATLAS vehicles file")

    prepared = vehicles[["bodytype", "modelyear", "adopt_fuel"]].copy()
    prepared["atlasVehicleTypeId"] = _build_atlas_vehicle_type_ids(
        prepared["bodytype"],
        prepared["adopt_fuel"],
        prepared["modelyear"],
    )
    grouped = (
        prepared.groupby(["atlasVehicleTypeId", "bodytype", "modelyear", "adopt_fuel"], dropna=False)
        .size()
        .reset_index(name="vehicleCount")
    )
    duplicate_atlas_vehicle_type_ids = grouped["atlasVehicleTypeId"].duplicated(keep=False)
    if duplicate_atlas_vehicle_type_ids.any():
        duplicate_rows = grouped.loc[duplicate_atlas_vehicle_type_ids, ["atlasVehicleTypeId", "bodytype", "modelyear", "adopt_fuel", "vehicleCount"]]
        raise ValueError(
            "Duplicate atlasVehicleTypeId values were generated in Step 1.2:\n"
            + duplicate_rows.to_string(index=False)
        )
    total = grouped["vehicleCount"].sum()
    prepared = grouped.copy()
    prepared["fleetShare"] = prepared["vehicleCount"] / total if total > 0 else 0.0
    return prepared[
        [
            "atlasVehicleTypeId",
            "bodytype",
            "modelyear",
            "adopt_fuel",
            "vehicleCount",
            "fleetShare",
        ]
    ]


def _format_income_bin(min_value: object, max_value: object) -> str:
    if pd.isna(min_value) or pd.isna(max_value):
        return "all"
    minimum = float(pd.to_numeric(pd.Series([min_value]), errors="coerce").iloc[0])
    maximum = float(pd.to_numeric(pd.Series([max_value]), errors="coerce").iloc[0])
    if minimum >= 100.0 or maximum >= 100.0:
        return "100-9999"
    lower = int(round(minimum))
    upper = int(round(maximum))
    return f"{lower}-{upper}"


def _validate_income_bins(income_bins: list[object] | None) -> list[float] | None:
    if income_bins in (None, []):
        return None
    numeric_bins = [float(pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]) for value in income_bins]
    if any(pd.isna(value) for value in numeric_bins):
        raise ValueError("atlas.income_bins must contain only numeric interval edges")
    if len(numeric_bins) < 2:
        raise ValueError("atlas.income_bins must contain at least two interval edges")
    if any(left >= right for left, right in zip(numeric_bins[:-1], numeric_bins[1:])):
        raise ValueError("atlas.income_bins must be strictly increasing")
    return numeric_bins


def _format_configured_income_bin_labels(income_bins: list[float]) -> list[str]:
    return [f"{int(lower)}-{int(upper)}" for lower, upper in zip(income_bins[:-1], income_bins[1:])]


def _lowest_income_bin_label(income_bins: list[object] | None) -> str:
    normalized_income_bins = _validate_income_bins(income_bins)
    if normalized_income_bins is not None:
        return _format_configured_income_bin_labels(normalized_income_bins)[0]
    return "0-30"


def _build_atlas_income_bin_targets(
    vehicles: pd.DataFrame,
    households: pd.DataFrame,
    persons: pd.DataFrame,
    income_bins: list[object] | None = None,
) -> pd.DataFrame:
    _require_column(vehicles, "household_id", "ATLAS vehicles file")
    _require_column(persons, "household_id", "ATLAS persons file")

    prepared_households = households.reset_index().copy()
    _require_column(prepared_households, "household_id", "ATLAS households file")
    _require_column(prepared_households, "income_segment", "ATLAS households file")
    _require_column(prepared_households, "income_in_thousands", "ATLAS households file")

    prepared_households["household_id"] = pd.to_numeric(prepared_households["household_id"], errors="coerce").astype("Int64")
    prepared_households["income_segment"] = pd.to_numeric(prepared_households["income_segment"], errors="coerce")
    prepared_households["income_in_thousands"] = pd.to_numeric(
        prepared_households["income_in_thousands"],
        errors="coerce",
    )
    valid_person_household_ids = pd.to_numeric(persons["household_id"], errors="coerce").astype("Int64")
    prepared_households = prepared_households[
        prepared_households["household_id"].isin(valid_person_household_ids.dropna().unique())
    ].copy()

    prepared_vehicles = vehicles[["household_id", "bodytype", "modelyear", "adopt_fuel"]].copy()
    prepared_vehicles["household_id"] = pd.to_numeric(prepared_vehicles["household_id"], errors="coerce").astype("Int64")
    prepared_vehicles["atlasVehicleTypeId"] = _build_atlas_vehicle_type_ids(
        prepared_vehicles["bodytype"],
        prepared_vehicles["adopt_fuel"],
        prepared_vehicles["modelyear"],
    )

    merged = prepared_vehicles.merge(
        prepared_households[["household_id", "income_segment", "income_in_thousands"]],
        on="household_id",
        how="left",
    )
    merged = merged.dropna(subset=["income_segment"]).copy()
    if merged.empty:
        raise ValueError("No ATLAS vehicles could be matched to ATLAS households with income_segment")

    normalized_income_bins = _validate_income_bins(income_bins)
    if normalized_income_bins is not None:
        merged = merged.dropna(subset=["income_in_thousands"]).copy()
        if merged.empty:
            raise ValueError("No ATLAS vehicles could be matched to ATLAS households with income_in_thousands")
        merged["incomeBin"] = pd.cut(
            merged["income_in_thousands"],
            bins=normalized_income_bins,
            labels=_format_configured_income_bin_labels(normalized_income_bins),
            include_lowest=True,
            right=True,
        )
        merged = merged.dropna(subset=["incomeBin"]).copy()
        income_counts = (
            merged.groupby(["atlasVehicleTypeId", "incomeBin"], dropna=False)
            .agg(
                incomeVehicleCount=("household_id", "size"),
                representativeIncomeK=("income_in_thousands", "mean"),
            )
            .reset_index()
        )
        representative = (
            income_counts.sort_values(["atlasVehicleTypeId", "incomeVehicleCount", "incomeBin"], ascending=[True, False, True])
            .drop_duplicates(subset=["atlasVehicleTypeId"], keep="first")
            .copy()
        )
        income_totals = (
            representative.groupby("incomeBin", dropna=False)["incomeVehicleCount"]
            .sum()
            .reset_index(name="incomeBinTotalVehicleCount")
        )
        representative = representative.merge(income_totals, on="incomeBin", how="left")
    else:
        income_counts = (
            merged.groupby(["atlasVehicleTypeId", "income_segment"], dropna=False)
            .agg(
                incomeVehicleCount=("household_id", "size"),
                incomeMinK=("income_in_thousands", "min"),
                incomeMaxK=("income_in_thousands", "max"),
                representativeIncomeK=("income_in_thousands", "mean"),
            )
            .reset_index()
        )
        representative = (
            income_counts.sort_values(["atlasVehicleTypeId", "incomeVehicleCount", "income_segment"], ascending=[True, False, True])
            .drop_duplicates(subset=["atlasVehicleTypeId"], keep="first")
            .copy()
        )
        income_totals = (
            representative.groupby("income_segment", dropna=False)["incomeVehicleCount"]
            .sum()
            .reset_index(name="incomeBinTotalVehicleCount")
        )
        representative = representative.merge(income_totals, on="income_segment", how="left")
        representative["incomeBin"] = representative.apply(
            lambda row: _format_income_bin(row["incomeMinK"], row["incomeMaxK"]),
            axis=1,
        )
    representative["incomeBin"] = representative["incomeBin"].astype(str)
    representative["incomeProbability"] = representative["incomeVehicleCount"] / representative["incomeBinTotalVehicleCount"]
    return representative[
        ["atlasVehicleTypeId", "incomeBin", "incomeVehicleCount", "incomeProbability", "representativeIncomeK"]
    ]


def _affordability_weight(
    msrp_usd: object,
    representative_income_k: object,
    target_ratio: float = 0.35,
    sigma: float = 0.15,
) -> float:
    msrp = pd.to_numeric(pd.Series([msrp_usd]), errors="coerce").iloc[0]
    income_k = pd.to_numeric(pd.Series([representative_income_k]), errors="coerce").iloc[0]
    if (
        pd.isna(msrp)
        or pd.isna(income_k)
        or float(msrp) <= 0
        or float(income_k) <= 0
        or sigma <= 0
    ):
        return 0.0
    income_usd = float(income_k) * 1000.0
    ratio = float(msrp) / income_usd
    z_score = (ratio - target_ratio) / sigma
    return float(np.exp(-0.5 * z_score**2))


def _build_vehicle_type_atlas_crosswalk(
    built_vehicle_types: pd.DataFrame,
    vehicle_type_mapping: pd.DataFrame,
    atlas_vehicle_type_targets: pd.DataFrame,
    fuel_mapping: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(vehicle_type_mapping, "vehicleTypeId", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "bodytype", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "modelyear", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "primaryFuelType", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "secondaryFuelType", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "msrp_usd", "Vehicle type mapping file")
    _require_column(fuel_mapping, "atlas_adopt_fuel", "FASTSim ATLAS fuel mapping file")
    _require_column(fuel_mapping, "fastsim_primary_fuel", "FASTSim ATLAS fuel mapping file")
    _require_column(fuel_mapping, "fastsim_secondary_fuel", "FASTSim ATLAS fuel mapping file")
    _require_column(fuel_mapping, "weight", "FASTSim ATLAS fuel mapping file")

    prepared_mapping = vehicle_type_mapping[
        ["vehicleTypeId", "bodytype", "modelyear", "primaryFuelType", "secondaryFuelType", "msrp_usd"]
    ].copy()
    prepared_mapping["vehicleTypeId"] = prepared_mapping["vehicleTypeId"].astype(str)
    prepared_mapping["modelyear"] = pd.to_numeric(prepared_mapping["modelyear"], errors="coerce")
    prepared_mapping["bodytype_norm"] = prepared_mapping["bodytype"].apply(_normalize_bodytype)
    prepared_mapping["primaryFuelType"] = prepared_mapping["primaryFuelType"].astype(str).str.lower()
    prepared_mapping["secondaryFuelType"] = prepared_mapping["secondaryFuelType"].fillna("").astype(str).str.lower()

    prepared_built_vehicle_types = built_vehicle_types[["vehicleTypeId"]].copy()
    prepared_built_vehicle_types["vehicleTypeId"] = prepared_built_vehicle_types["vehicleTypeId"].astype(str)

    built_mapping = prepared_built_vehicle_types.merge(prepared_mapping, on="vehicleTypeId", how="left")
    missing = built_mapping[
        built_mapping["bodytype"].isna()
        | built_mapping["primaryFuelType"].isna()
        | built_mapping["secondaryFuelType"].isna()
        | built_mapping["modelyear"].isna()
    ][
        "vehicleTypeId"
    ].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            "Missing vehicle-type mapping rows for built vehicleTypeId values:\n"
            + "\n".join(missing.astype(str).tolist())
        )

    atlas_keys = atlas_vehicle_type_targets.copy()
    atlas_keys["modelyear"] = pd.to_numeric(atlas_keys["modelyear"], errors="coerce")
    atlas_keys["bodytype_norm"] = atlas_keys["bodytype"].apply(_normalize_bodytype)
    atlas_keys["representativeIncomeK"] = pd.to_numeric(
        atlas_keys["representativeIncomeK"],
        errors="coerce",
    )
    prepared_fuel_mapping = fuel_mapping[
        ["atlas_adopt_fuel", "fastsim_primary_fuel", "fastsim_secondary_fuel", "weight"]
    ].copy()
    prepared_fuel_mapping["atlas_adopt_fuel"] = prepared_fuel_mapping["atlas_adopt_fuel"].astype(str)
    prepared_fuel_mapping["fastsim_primary_fuel"] = prepared_fuel_mapping["fastsim_primary_fuel"].astype(str).str.lower()
    prepared_fuel_mapping["fastsim_secondary_fuel"] = prepared_fuel_mapping["fastsim_secondary_fuel"].fillna("").astype(str).str.lower()
    prepared_fuel_mapping["weight"] = pd.to_numeric(prepared_fuel_mapping["weight"], errors="coerce").fillna(0.0)

    crosswalk_rows: list[pd.DataFrame] = []
    for atlas_vehicle_type_id, atlas_group in atlas_keys.groupby("atlasVehicleTypeId", sort=True, dropna=False):
        representative_row = (
            atlas_group.sort_values(["vehicleCount", "modelyear"], ascending=[False, False]).iloc[0]
        )
        candidate_frames: list[pd.DataFrame] = []

        for row in atlas_group.itertuples(index=False):
            row_fuel_mapping = prepared_fuel_mapping[
                prepared_fuel_mapping["atlas_adopt_fuel"] == str(row.adopt_fuel)
            ].copy()
            if row_fuel_mapping.empty:
                raise ValueError(
                    "No FASTSim fuel mapping configured for atlas adopt_fuel "
                    f"{row.adopt_fuel}"
                )

            candidates = built_mapping[
                built_mapping["bodytype_norm"] == row.bodytype_norm
            ].copy()
            candidates = candidates.merge(
                row_fuel_mapping,
                left_on=["primaryFuelType", "secondaryFuelType"],
                right_on=["fastsim_primary_fuel", "fastsim_secondary_fuel"],
                how="inner",
            )
            if candidates.empty:
                continue

            candidates["yearDistance"] = (candidates["modelyear"] - row.modelyear).abs()
            candidates["yearWeight"] = 1.0 / (1.0 + candidates["yearDistance"])
            candidates["affordabilityWeight"] = candidates["msrp_usd"].apply(
                lambda value: _affordability_weight(value, row.representativeIncomeK)
            )
            candidates["candidateWeight"] = (
                pd.to_numeric(candidates["weight"], errors="coerce").fillna(0.0)
                * pd.to_numeric(candidates["yearWeight"], errors="coerce").fillna(0.0)
                * pd.to_numeric(candidates["affordabilityWeight"], errors="coerce").fillna(0.0)
                * float(row.vehicleCount)
            )
            candidate_frames.append(
                candidates[["vehicleTypeId", "candidateWeight"]].copy()
            )

        if not candidate_frames:
            raise ValueError(
                "No FASTSim candidates available for atlas vehicle type "
                f"{atlas_vehicle_type_id}"
            )

        candidates = pd.concat(candidate_frames, ignore_index=True)
        candidates = candidates.groupby("vehicleTypeId", as_index=False)["candidateWeight"].sum()
        candidates["candidateWeight"] = pd.to_numeric(candidates["candidateWeight"], errors="coerce").fillna(0.0)
        candidates = candidates.sort_values(["vehicleTypeId"]).reset_index(drop=True)

        if len(candidates) == 1:
            selected_vehicle_type_id = str(candidates.iloc[0]["vehicleTypeId"])
        else:
            weights = pd.to_numeric(candidates["candidateWeight"], errors="coerce").fillna(0.0)
            weight_sum = weights.sum()
            probabilities = None if weight_sum <= 0 else (weights / weight_sum).to_numpy()
            selected_vehicle_type_id = str(candidates.iloc[rng.choice(len(candidates), p=probabilities)]["vehicleTypeId"])
        crosswalk_rows.append(
            pd.DataFrame(
                [
                    {
                        "atlasVehicleTypeId": str(atlas_vehicle_type_id),
                        "fastsimVehicleTypeId": selected_vehicle_type_id,
                        "vehicleTypeId": _combine_vehicle_type_ids(selected_vehicle_type_id, atlas_vehicle_type_id),
                        "bodytype": representative_row["bodytype"],
                        "modelyear": int(representative_row["modelyear"]) if pd.notna(representative_row["modelyear"]) else representative_row["modelyear"],
                        "adopt_fuel": representative_row["adopt_fuel"],
                        "fleetShare": float(representative_row["fleetShare"]) if pd.notna(representative_row["fleetShare"]) else 0.0,
                        "incomeBin": str(representative_row["incomeBin"]),
                        "incomeProbability": float(representative_row["incomeProbability"]) if pd.notna(representative_row["incomeProbability"]) else 0.0,
                    }
                ]
            )
        )

    crosswalk = pd.concat(crosswalk_rows, ignore_index=True)
    crosswalk = crosswalk[
        [
            "atlasVehicleTypeId",
            "fastsimVehicleTypeId",
            "vehicleTypeId",
            "bodytype",
            "modelyear",
            "adopt_fuel",
            "fleetShare",
            "incomeBin",
            "incomeProbability",
        ]
    ].drop_duplicates()
    if crosswalk["atlasVehicleTypeId"].duplicated().any():
        duplicate_values = crosswalk.loc[crosswalk["atlasVehicleTypeId"].duplicated(), "atlasVehicleTypeId"].drop_duplicates().tolist()
        raise ValueError(
            "Duplicate atlasVehicleTypeId values were generated in Step 1.3:\n"
            + "\n".join(duplicate_values)
        )
    if crosswalk["vehicleTypeId"].duplicated().any():
        duplicate_values = crosswalk.loc[crosswalk["vehicleTypeId"].duplicated(), "vehicleTypeId"].drop_duplicates().tolist()
        raise ValueError(
            "Duplicate final vehicleTypeId values were generated in Step 1.3:\n"
            + "\n".join(duplicate_values)
        )

    return crosswalk.sort_values(
        ["atlasVehicleTypeId", "bodytype", "adopt_fuel", "modelyear", "vehicleTypeId"]
    ).reset_index(drop=True)


def _expand_vehicle_types_by_crosswalk(
    built_vehicle_types: pd.DataFrame,
    vehicle_type_atlas_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(vehicle_type_atlas_crosswalk, "fastsimVehicleTypeId", "Vehicle type atlas crosswalk")
    _require_column(vehicle_type_atlas_crosswalk, "vehicleTypeId", "Vehicle type atlas crosswalk")

    templates = built_vehicle_types.copy()
    templates["fastsimVehicleTypeId"] = templates["vehicleTypeId"].astype(str)

    expanded = vehicle_type_atlas_crosswalk[["fastsimVehicleTypeId", "vehicleTypeId"]].drop_duplicates().merge(
        templates,
        on="fastsimVehicleTypeId",
        how="left",
    )
    missing = expanded[expanded["vehicleTypeId_y"].isna()]["fastsimVehicleTypeId"].drop_duplicates()
    if not missing.empty:
        raise ValueError(
            "Missing built vehicle type templates for FASTSim vehicleTypeId values:\n"
            + "\n".join(missing.astype(str).tolist())
        )

    expanded = expanded.drop(columns=["vehicleTypeId_y"])
    expanded = expanded.rename(columns={"vehicleTypeId_x": "vehicleTypeId"})
    expanded = expanded.drop(columns=["fastsimVehicleTypeId"])
    return expanded.drop_duplicates().reset_index(drop=True)


def _round_fleet_share(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    _require_column(frame, "fleetShare", "Vehicle type atlas crosswalk")
    prepared = frame.copy()
    prepared["fleetShare"] = pd.to_numeric(prepared["fleetShare"], errors="coerce").fillna(0.0)
    prepared["fleetShare"] = prepared["fleetShare"].round(digits)
    if not prepared.empty:
        remainder = round(1.0 - prepared.iloc[:-1]["fleetShare"].sum(), digits)
        prepared.iloc[-1, prepared.columns.get_loc("fleetShare")] = remainder
    return prepared


def _round_income_probability(frame: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    _require_column(frame, "incomeBin", "Vehicle type atlas crosswalk")
    _require_column(frame, "incomeProbability", "Vehicle type atlas crosswalk")
    prepared = frame.copy()
    prepared["incomeProbability"] = pd.to_numeric(prepared["incomeProbability"], errors="coerce").fillna(0.0)
    prepared["incomeBin"] = prepared["incomeBin"].astype(str)
    rounded_groups: list[pd.DataFrame] = []
    for _, group in prepared.groupby("incomeBin", sort=False, dropna=False):
        rounded = group.copy()
        rounded["incomeProbability"] = rounded["incomeProbability"].round(digits)
        if not rounded.empty:
            remainder = round(1.0 - rounded.iloc[:-1]["incomeProbability"].sum(), digits)
            rounded.iloc[-1, rounded.columns.get_loc("incomeProbability")] = remainder
        rounded_groups.append(rounded)
    return pd.concat(rounded_groups, ignore_index=True) if rounded_groups else prepared


def _create_probability_string(income_bin: str, income_probability: float, ridehail_probability: float) -> str:
    return f"ridehail | all:{ridehail_probability:.6f}; income | {income_bin}:{income_probability:.6f}"


def _apply_vehicle_type_probabilities_from_crosswalk(
    built_vehicle_types: pd.DataFrame,
    vehicle_type_atlas_crosswalk: pd.DataFrame,
    income_bins: list[object] | None = None,
) -> pd.DataFrame:
    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(vehicle_type_atlas_crosswalk, "vehicleTypeId", "Vehicle type atlas crosswalk")
    _require_column(vehicle_type_atlas_crosswalk, "fleetShare", "Vehicle type atlas crosswalk")
    _require_column(vehicle_type_atlas_crosswalk, "incomeBin", "Vehicle type atlas crosswalk")
    _require_column(vehicle_type_atlas_crosswalk, "incomeProbability", "Vehicle type atlas crosswalk")

    prepared = built_vehicle_types.copy()
    probabilities = vehicle_type_atlas_crosswalk[
        ["vehicleTypeId", "fleetShare", "incomeBin", "incomeProbability"]
    ].drop_duplicates().copy()
    probabilities["vehicleTypeId"] = probabilities["vehicleTypeId"].astype(str)
    probabilities["fleetShare"] = pd.to_numeric(probabilities["fleetShare"], errors="coerce").fillna(0.0)
    probabilities["incomeBin"] = probabilities["incomeBin"].astype(str)
    probabilities["incomeProbability"] = pd.to_numeric(probabilities["incomeProbability"], errors="coerce").fillna(0.0)

    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].astype(str)
    prepared = prepared.merge(probabilities, on="vehicleTypeId", how="left")
    prepared["fleetShare"] = prepared["fleetShare"].fillna(0.0)
    prepared["incomeBin"] = prepared["incomeBin"].fillna(_lowest_income_bin_label(income_bins))
    prepared["incomeProbability"] = prepared["incomeProbability"].fillna(0.0)
    prepared["sampleProbabilityWithinCategory"] = prepared["fleetShare"].map(lambda value: f"{float(value):.6f}")
    prepared["sampleProbabilityString"] = prepared.apply(
        lambda row: _create_probability_string(
            row["incomeBin"],
            float(row["incomeProbability"]),
            float(row["fleetShare"]),
        ),
        axis=1,
    )
    return prepared.drop(columns=["fleetShare", "incomeBin", "incomeProbability"])


def _strip_known_suffixes(filename: str) -> str:
    token = filename
    if token.endswith(".csv.gz"):
        token = token[:-7]
    elif token.endswith(".csv"):
        token = token[:-4]
    if token.endswith("_lookup_table"):
        token = token[:-13]
    return token


def _vehicle_type_id_from_fastsim_token(token: object) -> str:
    stem = _strip_known_suffixes(str(token))
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
    built_vehicle_types: pd.DataFrame,
    fastsim_bodytype_xwalk: pd.DataFrame,
) -> pd.DataFrame:
    body_col = "bodytype" if "bodytype" in fastsim_bodytype_xwalk.columns else "body_type"
    if body_col not in fastsim_bodytype_xwalk.columns:
        raise ValueError("FASTSim bodytype xwalk file is missing 'body_type'")
    id_col = "vehicleTypeId" if "vehicleTypeId" in fastsim_bodytype_xwalk.columns else "vehicle_id"
    if id_col not in fastsim_bodytype_xwalk.columns:
        raise ValueError("FASTSim bodytype xwalk file is missing 'vehicle_id'")
    _require_column(fastsim_bodytype_xwalk, "msrp_usd", "FASTSim bodytype xwalk file")

    bodytypes = fastsim_bodytype_xwalk[[id_col, body_col, "msrp_usd"]].copy()
    bodytypes["vehicleTypeId"] = bodytypes[id_col].apply(_vehicle_type_id_from_fastsim_token)
    bodytypes["bodytype"] = bodytypes[body_col].astype(str)
    bodytypes["msrp_usd"] = pd.to_numeric(bodytypes["msrp_usd"], errors="coerce")
    bodytypes = bodytypes[["vehicleTypeId", "bodytype", "msrp_usd"]].drop_duplicates()

    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(built_vehicle_types, "primaryFuelType", "Built vehicle types")
    _require_column(built_vehicle_types, "secondaryFuelType", "Built vehicle types")

    fuels = built_vehicle_types[["vehicleTypeId", "primaryFuelType", "secondaryFuelType"]].copy()
    fuels["vehicleTypeId"] = fuels["vehicleTypeId"].astype(str)
    fuels["primaryFuelType"] = fuels["primaryFuelType"].astype(str).str.lower()
    fuels["secondaryFuelType"] = fuels["secondaryFuelType"].fillna("").astype(str).str.lower()
    fuels = fuels.drop_duplicates()

    mapping = bodytypes.merge(fuels, on="vehicleTypeId", how="inner")
    mapping["modelyear"] = _extract_model_year_from_vehicle_type_id(mapping["vehicleTypeId"])
    mapping = mapping[
        ["vehicleTypeId", "bodytype", "modelyear", "primaryFuelType", "secondaryFuelType", "msrp_usd"]
    ].drop_duplicates()
    return mapping.sort_values(["vehicleTypeId", "bodytype", "primaryFuelType", "secondaryFuelType"]).reset_index(drop=True)


def _normalize_primary_fuel(token: str) -> str:
    token = token.strip().lower()
    mapping = {
        "electric": "electricity",
        "electricity": "electricity",
        "gasoline": "gasoline",
        "diesel": "diesel",
        "hydrogen": "hydrogen",
        "cng": "naturalgas",
        "naturalgas": "naturalgas",
    }
    return mapping.get(token, token)


def _parse_fastsim_lookup_filename(path: Path) -> dict[str, str] | None:
    token = _strip_known_suffixes(path.name)
    match = re.match(r"^(?P<year>\d{4})_(?P<fuel>[^_]+)_(?P<name>.+)$", token)
    if match is None:
        return None

    year = match.group("year")
    fuel = match.group("fuel")
    name = match.group("name")
    charge_mode = ""
    if name.endswith("_Charge_Depleting"):
        name = name[: -len("_Charge_Depleting")]
        charge_mode = "Charge_Depleting"
    elif name.endswith("_Charge_Sustaining"):
        name = name[: -len("_Charge_Sustaining")]
        charge_mode = "Charge_Sustaining"

    return {
        "vehicleTypeId": f"{year}_{name}",
        "modelyear": year,
        "fuel": _normalize_primary_fuel(fuel),
        "charge_mode": charge_mode,
        "file_name": path.name,
    }


def _select_template_row(vehicle_types: pd.DataFrame, primary_fuel: str, secondary_fuel: str) -> pd.Series:
    primary = vehicle_types.get("primaryFuelType", pd.Series(index=vehicle_types.index, dtype="object")).astype(str).str.lower()
    secondary = vehicle_types.get("secondaryFuelType", pd.Series(index=vehicle_types.index, dtype="object")).astype(str).str.lower()
    category = vehicle_types.get("vehicleCategory", pd.Series(index=vehicle_types.index, dtype="object")).astype(str)

    mask = primary.eq(primary_fuel) & secondary.eq(secondary_fuel) & category.eq("Car")
    if mask.any():
        return vehicle_types.loc[mask].iloc[0].copy()
    raise ValueError(
        "No Car template row matches FASTSim fuels "
        f"primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
    )


def _build_vehicle_types_from_fastsim_folder(
    vehicle_types: pd.DataFrame,
    fastsim_data_folder: str,
    atlas_year: Any,
) -> pd.DataFrame:
    folder = Path(resolve_workflow_path(fastsim_data_folder))
    files = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and (path.name.endswith(".csv") or path.name.endswith(".csv.gz"))
    )
    parsed_rows = [row for row in (_parse_fastsim_lookup_filename(path) for path in files) if row is not None]
    if not parsed_rows:
        raise ValueError(f"No FASTSim lookup files were found under {folder}")

    parsed = pd.DataFrame(parsed_rows)
    parsed["modelyear"] = pd.to_numeric(parsed["modelyear"], errors="coerce")
    parsed = parsed[parsed["modelyear"].le(pd.to_numeric(atlas_year, errors="coerce"))].copy()
    if parsed.empty:
        raise ValueError(f"No FASTSim lookup files at or below atlas.year={atlas_year} were found under {folder}")

    new_rows: list[pd.Series] = []
    for vehicle_type_id, group in parsed.groupby("vehicleTypeId", sort=True):
        charge_depleting = group[group["charge_mode"].eq("Charge_Depleting")]
        charge_sustaining = group[group["charge_mode"].eq("Charge_Sustaining")]
        plain = group[group["charge_mode"].eq("")]

        if not charge_depleting.empty:
            primary_fuel = charge_depleting.iloc[0]["fuel"]
            secondary_fuel = charge_sustaining.iloc[0]["fuel"] if not charge_sustaining.empty else ""
            template = _select_template_row(vehicle_types, primary_fuel, secondary_fuel)
            template["primaryFuelType"] = primary_fuel
            template["primaryVehicleEnergyFile"] = charge_depleting.iloc[0]["file_name"]
            template["secondaryFuelType"] = secondary_fuel
            template["secondaryVehicleEnergyFile"] = charge_sustaining.iloc[0]["file_name"] if not charge_sustaining.empty else ""
        else:
            primary_fuel = plain.iloc[0]["fuel"] if not plain.empty else charge_sustaining.iloc[0]["fuel"]
            template = _select_template_row(vehicle_types, primary_fuel, "")
            template["primaryFuelType"] = primary_fuel
            template["primaryVehicleEnergyFile"] = plain.iloc[0]["file_name"] if not plain.empty else charge_sustaining.iloc[0]["file_name"]
            template["secondaryFuelType"] = ""
            template["secondaryVehicleEnergyFile"] = ""

        template["vehicleTypeId"] = vehicle_type_id
        new_rows.append(template)

    prepared = pd.DataFrame(new_rows).reset_index(drop=True)
    return prepared


def run_step1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 1: build BEAM vehicle types for the target ATLAS year."""
    config = workflow["config"]
    rng = np.random.default_rng(int(config["seed"]))

    print("=== Step 1.1: build vehicle types for atlas.year from beam inputs ===")
    vehicle_types = _read_csv(
        config["beam"]["vehicle_types_file"],
        columns=[
            "vehicleTypeId",
            "curbWeightInKg",
            "seatingCapacity",
            "standingRoomCapacity",
            "lengthInMeter",
            "primaryFuelType",
            "primaryFuelConsumptionInJoulePerMeter",
            "primaryFuelCapacityInJoule",
            "primaryVehicleEnergyFile",
            "secondaryFuelType",
            "secondaryFuelConsumptionInJoulePerMeter",
            "secondaryVehicleEnergyFile",
            "secondaryFuelCapacityInJoule",
            "automationLevel",
            "maxVelocity",
            "passengerCarUnit",
            "rechargeLevel2RateLimitInWatts",
            "rechargeLevel3RateLimitInWatts",
            "vehicleCategory",
            "sampleProbabilityWithinCategory",
            "sampleProbabilityString",
        ],
    )
    _require_column(vehicle_types, "vehicleTypeId", "BEAM vehicle types file")
    atlas_year = pd.to_numeric(config["atlas"]["year"], errors="coerce")

    encoded_years = _extract_model_year_from_vehicle_type_id(vehicle_types["vehicleTypeId"])
    has_year_below_atlas_year = encoded_years.notna().any() and bool((encoded_years < atlas_year).any())
    if has_year_below_atlas_year:
        prepared_vehicle_types = vehicle_types.copy()
        prepared_vehicle_types["modelyear"] = encoded_years
        prepared_vehicle_types = prepared_vehicle_types[prepared_vehicle_types["modelyear"].le(atlas_year)].copy()
    else:
        fastsim_data_folder = config["beam"].get("fastsim_data_folder")
        if fastsim_data_folder in (None, ""):
            raise ValueError(
                "beam.fastsim_data_folder is required when beam.vehicle_types_file does not contain model years at or below atlas.year"
            )
        prepared_vehicle_types = _build_vehicle_types_from_fastsim_folder(
            vehicle_types,
            fastsim_data_folder,
            atlas_year,
        )
        prepared_vehicle_types["modelyear"] = _extract_model_year_from_vehicle_type_id(prepared_vehicle_types["vehicleTypeId"])

    fastsim_bodytype_xwalk = _read_csv(config["beam"]["fastsim_bodytype_xwalk_file"])
    vehicle_type_mapping = _build_fastsim_vehicle_type_mapping(
        prepared_vehicle_types,
        fastsim_bodytype_xwalk,
    )

    print("=== Step 1.2: load atlas vehicles, households, and persons and calculate fleetShare by income bin ===")
    atlas_vehicles = _read_csv(
        config["atlas"]["vehicles_file"],
        columns=["household_id", "bodytype", "modelyear", "adopt_fuel"],
    )
    atlas_households = _read_csv(
        config["atlas"]["households_file"],
        columns=["household_id", "income_segment", "income_in_thousands"],
    )
    atlas_persons = _read_csv(
        config["atlas"]["persons_file"],
        columns=["household_id"],
    )
    atlas_vehicle_type_targets = _build_atlas_vehicle_type_targets(atlas_vehicles)
    atlas_income_bin_targets = _build_atlas_income_bin_targets(
        atlas_vehicles,
        atlas_households,
        atlas_persons,
        config["atlas"].get("income_bins"),
    )
    atlas_vehicle_type_targets = atlas_vehicle_type_targets.merge(
        atlas_income_bin_targets,
        on="atlasVehicleTypeId",
        how="left",
    )
    atlas_vehicle_type_targets["incomeBin"] = atlas_vehicle_type_targets["incomeBin"].fillna(
        _lowest_income_bin_label(config["atlas"].get("income_bins"))
    )
    atlas_vehicle_type_targets["incomeProbability"] = pd.to_numeric(
        atlas_vehicle_type_targets["incomeProbability"],
        errors="coerce",
    ).fillna(0.0)
    print("=== Step 1.3: create beam-to-atlas crosswalk file ===")
    fastsim_atlas_fuel_mapping = _read_csv(config["beam"]["fastsim_atlas_fuel_mapping_file"])
    vehicle_type_atlas_crosswalk = _build_vehicle_type_atlas_crosswalk(
        prepared_vehicle_types,
        vehicle_type_mapping,
        atlas_vehicle_type_targets,
        fastsim_atlas_fuel_mapping,
        rng,
    )
    vehicle_type_atlas_crosswalk = _round_fleet_share(vehicle_type_atlas_crosswalk)
    vehicle_type_atlas_crosswalk = _round_income_probability(vehicle_type_atlas_crosswalk)
    prepared_vehicle_types = _expand_vehicle_types_by_crosswalk(
        prepared_vehicle_types,
        vehicle_type_atlas_crosswalk,
    )
    prepared_vehicle_types = _apply_vehicle_type_probabilities_from_crosswalk(
        prepared_vehicle_types,
        vehicle_type_atlas_crosswalk,
        config["atlas"].get("income_bins"),
    )
    prepared_vehicle_types_file = _write_new_vehicle_types_file(prepared_vehicle_types, config["output"])
    vehicle_type_atlas_crosswalk_file = _write_vehicle_type_crosswalk_file(
        vehicle_type_atlas_crosswalk,
        config["output"],
    )

    workflow["source_beam_vehicle_types"] = vehicle_types
    workflow["source_atlas_vehicles"] = atlas_vehicles
    workflow["source_atlas_households"] = atlas_households
    workflow["source_atlas_persons"] = atlas_persons
    workflow["source_beam_vehicle_bodytype_xwalk"] = fastsim_bodytype_xwalk
    workflow["source_beam_fuel_mapping"] = fastsim_atlas_fuel_mapping
    workflow["source_beam_vehicle_type_mapping"] = vehicle_type_mapping
    workflow["built_vehicle_types"] = prepared_vehicle_types
    workflow["built_vehicle_types_file"] = prepared_vehicle_types_file
    workflow["atlas_vehicle_type_targets"] = atlas_vehicle_type_targets
    workflow["atlas_income_bin_targets"] = atlas_income_bin_targets
    workflow["vehicle_type_atlas_crosswalk"] = vehicle_type_atlas_crosswalk
    workflow["vehicle_type_atlas_crosswalk_file"] = vehicle_type_atlas_crosswalk_file
    return workflow
