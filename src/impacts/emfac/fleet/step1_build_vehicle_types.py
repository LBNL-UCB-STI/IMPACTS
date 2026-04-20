"""Fleet Step 1: build passenger vehicle types for the target ATLAS year.

Substeps:
1.1 Load fastsim.passenger_vehicle_types_file, inspect encoded model years, and
    filter or rebuild vehicle types for <= atlas.year.
1.2 Load ATLAS vehicles, households, and persons, then calculate fleetShare and
    representative income bins for unique ATLAS bodytype/adopt_fuel/modelyear rows.
1.3 Map built FASTSim vehicle types to ATLAS rows and expand the final Step 1
    vehicle types using combined ids.
1.4 Build the freight vehicle-type population from FRISM carriers and tours.
1.5 Filter FRISM freight vehicle types to the population and attach all-bucket probabilities.
1.6 Assign FASTSim energy files to freight vehicle types.
1.7 Extract non-car passenger vehicle types from the source passenger vehicle-types file.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path


def _read_csv(path_like: str, *, columns: list[str] | tuple[str, ...] | None = None) -> pd.DataFrame:
    return read_table(path_like, columns=columns)


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_lower(value: object) -> str:
    return _normalize_text(value).lower()


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
    return f"{fastsim_token}--{atlas_token}"


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


def _step1_tmp_dir(output_dir: str) -> Path:
    return Path(resolve_workflow_path(output_dir)) / "_tmp"


def _normalize_energy_file_path(value: Any) -> str:
    token = str(value or "").strip()
    if not token:
        return ""
    normalized = token.replace("\\", "/").lstrip("/")
    if normalized.startswith("fuel/"):
        return normalized
    return f"fuel/{normalized}"


def _normalize_energy_file_columns(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column_name in ["primaryVehicleEnergyFile", "secondaryVehicleEnergyFile"]:
        if column_name in prepared.columns:
            prepared[column_name] = prepared[column_name].map(_normalize_energy_file_path)
    return prepared


def _attach_adopt_fuel_column(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    _require_column(prepared, "primaryFuelType", "Vehicle types file")
    _require_column(prepared, "secondaryFuelType", "Vehicle types file")
    prepared["adopt_fuel"] = [
        _compose_adopt_fuel(primary_fuel, secondary_fuel)
        for primary_fuel, secondary_fuel in zip(
            prepared["primaryFuelType"],
            prepared["secondaryFuelType"],
        )
    ]
    return prepared


def _write_new_vehicle_types_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / "vehicleTypes--passenger-car.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _write_vehicle_type_crosswalk_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / "vehicleTypeCrosswalk--fastsim-atlas.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _write_new_freight_vehicle_types_file(frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / "vehicleTypes--freight.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _write_passenger_section_vehicle_types_file(section_name: str, frame: pd.DataFrame, output_dir: str) -> str:
    target = _step1_tmp_dir(output_dir) / f"vehicleTypes--passenger-{section_name}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    return str(target)


def _remove_stale_step1_output_files(output_dir: str) -> None:
    root = Path(resolve_workflow_path(output_dir))
    tmp_root = _step1_tmp_dir(output_dir)
    for name in [
        "vehicleTypes--beam--step1-built.csv",
        "vehicle_type_atlas_crosswalk.csv",
        "freightVehicleTypePopulation--step1.csv",
        "vehicleTypes--passenger-car--step1-built.csv",
        "vehicleTypeCrosswalk--passenger-atlas--step1.csv",
        "vehicleTypePopulation--freight--step1.csv",
        "vehicleTypePopulation--freight.csv",
        "vehicleTypes--freight--step1-built.csv",
        "vehicleTypes--passenger-noncar--step1-built.csv",
        "vehicleTypes--passenger-noncar.csv",
        "vehicleTypes--passenger-bus.csv",
        "vehicleTypes--passenger-bike.csv",
        "vehicleTypes--passenger-other.csv",
        "vehicleTypes--passenger-car.csv",
        "vehicleTypeCrosswalk--fastsim-atlas.csv",
        "vehicleTypes--freight.csv",
    ]:
        for base_dir in (root, tmp_root):
            target = base_dir / name
            if target.exists():
                target.unlink()


def _align_vehicle_types_to_source_schema(
    frame: pd.DataFrame,
    source_columns: list[str],
    *,
    frame_name: str,
) -> pd.DataFrame:
    missing = [column for column in source_columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing source vehicle-types columns:\n" + "\n".join(missing)
        )
    return frame.loc[:, source_columns].copy()


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


def _build_freight_vehicle_type_population(
    carriers: pd.DataFrame,
    tours: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(carriers, "tourId", "FRISM carriers file")
    _require_column(carriers, "vehicleTypeId", "FRISM carriers file")
    _require_column(tours, "tourId", "FRISM tours file")

    prepared_carriers = carriers[["tourId", "vehicleTypeId"]].copy()
    prepared_carriers["tourId"] = prepared_carriers["tourId"].astype(str)
    prepared_carriers["vehicleTypeId"] = prepared_carriers["vehicleTypeId"].astype(str)

    prepared_tours = tours[["tourId"]].copy()
    prepared_tours["tourId"] = prepared_tours["tourId"].astype(str)

    matched = prepared_carriers.merge(
        prepared_tours.drop_duplicates(),
        on="tourId",
        how="inner",
    )
    if matched.empty:
        raise ValueError("No FRISM carrier rows could be matched to FRISM tours on tourId")

    grouped = (
        matched.groupby("vehicleTypeId", dropna=False)
        .size()
        .reset_index(name="vehicleCount")
        .sort_values(["vehicleCount", "vehicleTypeId"], ascending=[False, True])
        .reset_index(drop=True)
    )
    total = grouped["vehicleCount"].sum()
    grouped["fleetShare"] = grouped["vehicleCount"] / total if total > 0 else 0.0
    return grouped[["vehicleTypeId", "vehicleCount", "fleetShare"]]


def _create_catch_all_income_probability_string(probability: float) -> str:
    return f"income | 0-999999:{float(probability):.6f}"


def _normalize_freight_vehicle_type_id(value: object) -> str:
    normalized = re.sub(r"^ft[-_]+", "", str(value).strip(), flags=re.IGNORECASE)
    tokens = [token for token in re.split(r"[-_]+", normalized) if token]
    if not tokens:
        return ""
    return "".join(token[:1].upper() + token[1:].lower() for token in tokens)


def _map_freight_beam_vehicle_category(vehicle_type_id: object, vehicle_category: object, vehicle_class: object) -> str:
    normalized_id = _normalize_freight_vehicle_type_id(vehicle_type_id)
    prefix_map = {
        "Ld1": "Class12aVocational",
        "Ld3": "Class2b3Vocational",
        "Mdv": "Class456Vocational",
        "Hdt": "Class78Tractor",
        "Hdv": "Class78Vocational",
    }
    for prefix, beam_vehicle_category in prefix_map.items():
        if normalized_id.startswith(prefix):
            return beam_vehicle_category

    vehicle_category_token = _normalize_text(vehicle_category)
    if vehicle_category_token in {
        "Class12aVocational",
        "Class2b3Vocational",
        "Class456Vocational",
        "Class78Tractor",
        "Class78Vocational",
    }:
        return vehicle_category_token

    vehicle_class_key = _normalize_lower(vehicle_class)
    class_text_map = {
        "class 1&2a vocational": "Class12aVocational",
        "class 2&b3 vocational": "Class2b3Vocational",
        "class 4-6 vocational": "Class456Vocational",
        "class 7&8 tractor": "Class78Tractor",
        "class 7&8 vocational": "Class78Vocational",
    }
    mapped = class_text_map.get(vehicle_class_key)
    if mapped:
        return mapped
    raise ValueError(
        "Could not determine freight BEAM vehicle category for vehicleTypeId="
        f"{vehicle_type_id}, vehicleCategory={vehicle_category}, vehicleClass={vehicle_class}"
    )


def _build_freight_vehicle_types_from_population(
    freight_vehicle_types: pd.DataFrame,
    freight_vehicle_type_population: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(freight_vehicle_types, "vehicleTypeId", "FRISM freight vehicle types file")
    _require_column(freight_vehicle_type_population, "vehicleTypeId", "Freight vehicle type population")
    _require_column(freight_vehicle_type_population, "fleetShare", "Freight vehicle type population")

    prepared_types = freight_vehicle_types.copy()
    prepared_types["vehicleTypeId"] = prepared_types["vehicleTypeId"].astype(str)

    population = freight_vehicle_type_population[["vehicleTypeId", "fleetShare"]].drop_duplicates().copy()
    population["vehicleTypeId"] = population["vehicleTypeId"].astype(str)
    population["fleetShare"] = pd.to_numeric(population["fleetShare"], errors="coerce").fillna(0.0)

    prepared = prepared_types.merge(population, on="vehicleTypeId", how="inner")
    if prepared.empty:
        raise ValueError("No FRISM freight vehicle types intersect with the freight vehicle-type population")

    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_freight_vehicle_type_id)
    duplicate_vehicle_type_ids = prepared["vehicleTypeId"][prepared["vehicleTypeId"].duplicated()].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Freight vehicleTypeId normalization produced duplicates:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    prepared["sampleProbabilityWithinCategory"] = prepared["fleetShare"].map(lambda value: f"{float(value):.6f}")
    prepared["sampleProbabilityString"] = prepared["fleetShare"].map(_create_catch_all_income_probability_string)
    return prepared.drop(columns=["fleetShare"]).drop_duplicates().reset_index(drop=True)


def _load_frism_atlas_bodytype_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read_csv(config["mappings"]["atlas_frism_xwalk_file"])
    _require_column(frame, "body_type", "FRISM ATLAS bodytype crosswalk file")
    weight_columns = [column for column in frame.columns if column != "body_type"]
    if not weight_columns:
        raise ValueError("FRISM ATLAS bodytype crosswalk file has no freight vehicle-category columns")
    prepared = frame.melt(
        id_vars=["body_type"],
        value_vars=weight_columns,
        var_name="freight_beam_category",
        value_name="weight",
    )
    prepared["body_type"] = prepared["body_type"].apply(_normalize_bodytype)
    prepared["freight_beam_category"] = prepared["freight_beam_category"].astype(str)
    prepared["weight"] = pd.to_numeric(prepared["weight"], errors="coerce").fillna(0.0)
    return prepared[prepared["weight"].gt(0)].reset_index(drop=True)


def _load_atlas_passenger_category_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read_csv(config["mappings"]["atlas_emfac_xwalk_file"])
    _require_column(frame, "body_type", "ATLAS EMFAC crosswalk file")
    _require_column(frame, "passenger_beam_category", "ATLAS EMFAC crosswalk file")
    prepared = frame[["body_type", "passenger_beam_category"]].drop_duplicates().copy()
    prepared["body_type"] = prepared["body_type"].apply(_normalize_bodytype)
    prepared["passenger_beam_category"] = prepared["passenger_beam_category"].apply(_normalize_bodytype)
    prepared = prepared[prepared["body_type"].ne("") & prepared["passenger_beam_category"].ne("")]
    conflicting = (
        prepared.groupby("body_type", dropna=False)["passenger_beam_category"]
        .nunique()
        .reset_index(name="passenger_category_count")
    )
    conflicting = conflicting[conflicting["passenger_category_count"].gt(1)]["body_type"]
    if not conflicting.empty:
        raise ValueError(
            "ATLAS EMFAC crosswalk file maps one body_type to multiple passenger BEAM categories:\n"
            + "\n".join(conflicting.astype(str).tolist())
        )
    return prepared.drop_duplicates(subset=["body_type"], keep="first").reset_index(drop=True)


def _load_fastsim_registry(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read_csv(config["mappings"]["fastsim_category_fuel_mapping_file"])
    for column_name in [
        "passenger_beam_category",
        "freight_beam_category",
        "atlas_fuel",
        "frism_fuel",
        "fastsim_relative_path",
        "fastsim_fuel_type",
        "msrp_usd",
    ]:
        _require_column(frame, column_name, "FASTSim category fuel mapping file")
    prepared = frame.copy()
    prepared["passenger_beam_category"] = prepared["passenger_beam_category"].fillna("").astype(str).str.strip()
    prepared["freight_beam_category"] = prepared["freight_beam_category"].fillna("").astype(str).str.strip()
    prepared["atlas_fuel"] = prepared["atlas_fuel"].fillna("").astype(str).str.strip().str.lower()
    prepared["frism_fuel"] = prepared["frism_fuel"].fillna("").astype(str).str.strip().str.lower()
    prepared["fastsim_relative_path"] = prepared["fastsim_relative_path"].map(_normalize_energy_file_path)
    prepared["fastsim_fuel_type"] = prepared["fastsim_fuel_type"].astype(str).str.strip()
    prepared["msrp_usd"] = pd.to_numeric(prepared["msrp_usd"], errors="coerce")
    return prepared


def _build_adopt_fuel_keys(fuel_values: pd.Series) -> pd.Series:
    return fuel_values.str.split("|").map(
        lambda parts: ",".join([part.strip() for part in parts[1:] if str(part).strip()])
    )


def _expand_passenger_fastsim_mapping_rows(fastsim_registry: pd.DataFrame) -> pd.DataFrame:
    passenger_rows = fastsim_registry[fastsim_registry["passenger_beam_category"].ne("")].copy()
    passenger_rows["lookup_domain"] = "ldv"
    passenger_rows["vehicle_category"] = passenger_rows["passenger_beam_category"].map(_normalize_bodytype)
    passenger_rows["adopt_fuel"] = (
        passenger_rows["atlas_fuel"].str.split("|").str[0].fillna("").astype(str).str.strip()
    )
    passenger_rows["adopt_fuel_keys"] = _build_adopt_fuel_keys(passenger_rows["atlas_fuel"])
    return passenger_rows


def _expand_freight_fastsim_mapping_rows(fastsim_registry: pd.DataFrame) -> pd.DataFrame:
    freight_rows = fastsim_registry[fastsim_registry["freight_beam_category"].ne("")].copy()
    freight_rows["lookup_domain"] = np.where(
        freight_rows["freight_beam_category"].isin(["Class12aVocational", "Class2b3Vocational"]),
        "ldv",
        "mhdv",
    )
    freight_rows["vehicle_category"] = freight_rows["freight_beam_category"]
    freight_rows["adopt_fuel"] = (
        freight_rows["frism_fuel"].str.split("|").str[0].fillna("").astype(str).str.strip()
    )
    freight_rows["adopt_fuel_keys"] = _build_adopt_fuel_keys(freight_rows["frism_fuel"])
    return freight_rows


def _load_fastsim_category_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    fastsim_registry = _load_fastsim_registry(config)
    passenger_rows = _expand_passenger_fastsim_mapping_rows(fastsim_registry)
    freight_rows = _expand_freight_fastsim_mapping_rows(fastsim_registry)
    combined = pd.concat([passenger_rows, freight_rows], ignore_index=True, sort=False)
    return combined[
        [
            "lookup_domain",
            "passenger_beam_category",
            "freight_beam_category",
            "atlas_fuel",
            "frism_fuel",
            "vehicle_category",
            "adopt_fuel",
            "adopt_fuel_keys",
            "fastsim_relative_path",
            "fastsim_fuel_type",
            "msrp_usd",
        ]
    ].drop_duplicates().reset_index(drop=True)


def _build_fastsim_lookup_indexes(
    config: dict[str, Any],
    category_fuel_mapping: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "ldv": _build_ldv_fastsim_lookup_index(
            config["fastsim"]["ldv_fastsim_data_folder"],
            category_fuel_mapping,
        ),
        "mhdv": _build_mhdv_fastsim_lookup_index(
            config["fastsim"]["mhdv_fastsim_data_folder"],
            category_fuel_mapping,
        ),
    }


def _attach_fastsim_fuel_types(
    lookup_index: pd.DataFrame,
    *,
    lookup_domain: str,
    category_fuel_mapping: pd.DataFrame,
    frame_name: str,
) -> pd.DataFrame:
    scoped_ids = category_fuel_mapping[
        category_fuel_mapping["lookup_domain"] == lookup_domain
    ][
        [
            "fastsim_relative_path",
            "fastsim_fuel_type",
            "vehicle_category",
            "adopt_fuel",
            "adopt_fuel_keys",
            "msrp_usd",
        ]
    ].drop_duplicates().copy()
    prepared = lookup_index.merge(scoped_ids, on="fastsim_relative_path", how="left")
    prepared = prepared[prepared["fastsim_fuel_type"].notna()].copy()
    if prepared.empty:
        raise ValueError(
            f"{frame_name} has no rows that match the configured FASTSim category fuel mapping"
        )
    prepared["fastsim_fuel_type"] = prepared["fastsim_fuel_type"].astype(str)
    return prepared


def _build_ldv_fastsim_lookup_index(fastsim_data_folder: str, category_fuel_mapping: pd.DataFrame) -> pd.DataFrame:
    folder = Path(resolve_workflow_path(fastsim_data_folder))
    rows: list[dict[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
            continue
        parsed = _parse_fastsim_lookup_filename(path)
        if parsed is None:
            continue
        parsed["fastsim_relative_path"] = _normalize_energy_file_path(f"{folder.name}/{path.name}")
        rows.append(parsed)
    if not rows:
        raise ValueError(f"No LDV FASTSim lookup files were found under {folder}")
    return _attach_fastsim_fuel_types(
        pd.DataFrame(rows),
        lookup_domain="ldv",
        category_fuel_mapping=category_fuel_mapping,
        frame_name="LDV FASTSim lookup index",
    )


def _build_mhdv_fastsim_lookup_index(fastsim_data_folder: str, category_fuel_mapping: pd.DataFrame) -> pd.DataFrame:
    folder = Path(resolve_workflow_path(fastsim_data_folder))
    rows: list[dict[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or not path.name.endswith(".csv"):
            continue
        stem = path.stem
        match = re.match(r"^(?P<representative>.+)_\((?P<token>[^,]+),_(?P<year>\d{4}),_.+\)$", stem)
        if match is None:
            continue
        rows.append(
            {
                "representative_stem": match.group("representative"),
                "modelyear": match.group("year"),
                "file_name": path.name,
                "fastsim_relative_path": _normalize_energy_file_path(f"{folder.name}/{path.name}"),
            }
        )
    if not rows:
        raise ValueError(f"No MHDV FASTSim lookup files were found under {folder}")
    return _attach_fastsim_fuel_types(
        pd.DataFrame(rows),
        lookup_domain="mhdv",
        category_fuel_mapping=category_fuel_mapping,
        frame_name="MHDV FASTSim lookup index",
    )


def _compose_adopt_fuel(primary_fuel: object, secondary_fuel: object) -> str:
    primary = _normalize_lower(primary_fuel)
    secondary = _normalize_lower(secondary_fuel)
    if secondary:
        return f"{primary}+{secondary}"
    return primary


def _adopt_fuel_matches(frame: pd.DataFrame, adopt_fuel: str) -> pd.Series:
    primary = frame["adopt_fuel"].astype(str).str.strip().str.lower()
    alternative = frame["adopt_fuel_keys"].fillna("").astype(str).str.strip().str.lower()
    target = str(adopt_fuel).strip().lower()
    alt_match = alternative.ne("") & alternative.str.split(",").map(lambda parts: target in [part.strip() for part in parts])
    return primary.eq(target) | alt_match


def _select_freight_fastsim_adopt_fuel(
    category_fuel_mapping: pd.DataFrame,
    *,
    lookup_domain: str,
    beam_vehicle_category: str,
    primary_fuel: object,
    secondary_fuel: object,
) -> str:
    frism_adopt_fuel = _compose_adopt_fuel(primary_fuel, secondary_fuel)
    candidates = category_fuel_mapping[
        (category_fuel_mapping["lookup_domain"] == lookup_domain)
        & (category_fuel_mapping["vehicle_category"] == beam_vehicle_category)
    ].copy()
    candidates = candidates[_adopt_fuel_matches(candidates, frism_adopt_fuel)].copy()
    if candidates.empty:
        raise ValueError(
            "No FRISM FASTSim category fuel mapping row found for "
            f"lookup_domain={lookup_domain}, vehicle_category={beam_vehicle_category}, adopt_fuel={frism_adopt_fuel}, "
            f"primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )
    if lookup_domain == "ldv":
        atlas_fuel = str(candidates.iloc[0]["atlas_fuel"]).strip().lower()
        if not atlas_fuel:
            raise ValueError(
                "Matched FRISM LDV FASTSim mapping row has no atlas_fuel for "
                f"vehicle_category={beam_vehicle_category}, adopt_fuel={frism_adopt_fuel}"
            )
        return atlas_fuel.split("|")[0].strip()
    return frism_adopt_fuel


def _select_ldv_fastsim_energy_files(
    *,
    body_type: str,
    adopt_fuel: str,
    ldv_lookup_index: pd.DataFrame,
) -> tuple[str, str]:
    candidates = ldv_lookup_index[
        ldv_lookup_index["vehicle_category"].astype(str).map(_normalize_bodytype).eq(body_type)
    ].copy()
    candidates = candidates[_adopt_fuel_matches(candidates, adopt_fuel)].copy()
    if candidates.empty:
        raise ValueError(f"No LDV FASTSim category mapping found for body_type={body_type}")

    if str(adopt_fuel).strip().lower() == "phev":
        primary_files = candidates[candidates["charge_mode"].astype(str).eq("Charge_Depleting")].copy()
        secondary_files = candidates[candidates["charge_mode"].astype(str).eq("Charge_Sustaining")].copy()
        if primary_files.empty or secondary_files.empty:
            raise ValueError(
                f"No LDV FASTSim PHEV lookup pair found for body_type={body_type}, adopt_fuel={adopt_fuel}"
            )
        primary_files["modelyear"] = pd.to_numeric(primary_files["modelyear"], errors="coerce")
        secondary_files["modelyear"] = pd.to_numeric(secondary_files["modelyear"], errors="coerce")
        paired = primary_files.merge(
            secondary_files[["vehicleTypeId", "modelyear", "fastsim_relative_path"]],
            on="vehicleTypeId",
            how="inner",
            suffixes=("_primary", "_secondary"),
        )
        if paired.empty:
            raise ValueError(
                f"No LDV FASTSim hybrid lookup pair shared the same representative vehicle for body_type={body_type}"
            )
        selected = paired.sort_values(
            ["modelyear_primary", "vehicleTypeId"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[0]
        return str(selected["fastsim_relative_path_primary"]), str(selected["fastsim_relative_path_secondary"])

    files = candidates[candidates["charge_mode"].astype(str).eq("")].copy()
    if files.empty:
        raise ValueError(
            f"No LDV FASTSim lookup file found for body_type={body_type}, adopt_fuel={adopt_fuel}"
        )
    files["modelyear"] = pd.to_numeric(files["modelyear"], errors="coerce")
    selected = files.sort_values(
        ["modelyear", "vehicleTypeId"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return str(selected["fastsim_relative_path"]), ""


def _select_mhdv_fastsim_energy_files(
    *,
    beam_vehicle_category: str,
    adopt_fuel: str,
    mhdv_lookup_index: pd.DataFrame,
) -> tuple[str, str]:
    candidates = mhdv_lookup_index[
        mhdv_lookup_index["vehicle_category"].astype(str).eq(beam_vehicle_category)
    ].copy()
    if candidates.empty:
        raise ValueError(
            f"No MHDV FASTSim category mapping found for vehicle_category={beam_vehicle_category}"
        )
    candidates = candidates[_adopt_fuel_matches(candidates, adopt_fuel)].copy()
    if candidates.empty:
        raise ValueError(
            f"No MHDV FASTSim lookup file found for vehicle_category={beam_vehicle_category}, adopt_fuel={adopt_fuel}"
        )
    representative_stems = set(candidates["representative_stem"].astype(str))
    files = mhdv_lookup_index[
        (mhdv_lookup_index["representative_stem"].astype(str).isin(representative_stems))
        & _adopt_fuel_matches(mhdv_lookup_index, adopt_fuel)
    ].copy()
    if files.empty:
        raise ValueError(
            f"No MHDV FASTSim lookup file found for vehicle_category={beam_vehicle_category}, adopt_fuel={adopt_fuel}"
        )
    files["modelyear"] = pd.to_numeric(files["modelyear"], errors="coerce")
    selected = files.sort_values(
        ["modelyear", "representative_stem"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return str(selected["fastsim_relative_path"]), ""


def _assign_freight_fastsim_energy_files(
    freight_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
    *,
    category_fuel_mapping: pd.DataFrame | None = None,
    lookup_indexes: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    prepared = freight_vehicle_types.copy()
    frism_atlas_map = _load_frism_atlas_bodytype_mapping(config)
    atlas_passenger_category_mapping = _load_atlas_passenger_category_mapping(config)
    atlas_passenger_category_lookup = dict(
        zip(
            atlas_passenger_category_mapping["body_type"].astype(str).map(_normalize_bodytype),
            atlas_passenger_category_mapping["passenger_beam_category"].astype(str).map(_normalize_bodytype),
        )
    )
    if category_fuel_mapping is None:
        category_fuel_mapping = _load_fastsim_category_fuel_mapping(config)
    if lookup_indexes is None:
        lookup_indexes = _build_fastsim_lookup_indexes(config, category_fuel_mapping)
    ldv_lookup_index = lookup_indexes["ldv"]
    mhdv_lookup_index = lookup_indexes["mhdv"]

    primary_paths: list[str] = []
    secondary_paths: list[str] = []
    for row in prepared.itertuples(index=False):
        beam_vehicle_category = _map_freight_beam_vehicle_category(
            getattr(row, "vehicleTypeId"),
            getattr(row, "vehicleCategory", ""),
            getattr(row, "vehicleClass", ""),
        )
        lookup_domain = "ldv" if beam_vehicle_category in {"Class12aVocational", "Class2b3Vocational"} else "mhdv"
        adopt_fuel = _select_freight_fastsim_adopt_fuel(
            category_fuel_mapping,
            lookup_domain=lookup_domain,
            beam_vehicle_category=beam_vehicle_category,
            primary_fuel=getattr(row, "primaryFuelType", ""),
            secondary_fuel=getattr(row, "secondaryFuelType", ""),
        )
        if lookup_domain == "ldv":
            bodytype_candidates = frism_atlas_map[
                frism_atlas_map["freight_beam_category"] == beam_vehicle_category
            ].copy()
            if bodytype_candidates.empty:
                raise ValueError(
                    f"No FRISM-ATLAS body_type mapping found for freight_beam_category={beam_vehicle_category}"
                )
            selected_body_type = (
                bodytype_candidates.sort_values(["weight", "body_type"], ascending=[False, True], kind="mergesort").iloc[0]["body_type"]
            )
            selected_body_type = atlas_passenger_category_lookup.get(
                _normalize_bodytype(selected_body_type),
                _normalize_bodytype(selected_body_type),
            )
            primary_path, secondary_path = _select_ldv_fastsim_energy_files(
                body_type=str(selected_body_type),
                adopt_fuel=adopt_fuel,
                ldv_lookup_index=ldv_lookup_index,
            )
        else:
            primary_path, secondary_path = _select_mhdv_fastsim_energy_files(
                beam_vehicle_category=beam_vehicle_category,
                adopt_fuel=adopt_fuel,
                mhdv_lookup_index=mhdv_lookup_index,
            )
        primary_paths.append(_normalize_energy_file_path(primary_path))
        secondary_paths.append(_normalize_energy_file_path(secondary_path))

    prepared["primaryVehicleEnergyFile"] = primary_paths
    prepared["secondaryVehicleEnergyFile"] = secondary_paths
    return prepared


def _split_passenger_vehicle_types(vehicle_types: pd.DataFrame) -> dict[str, pd.DataFrame]:
    _require_column(vehicle_types, "vehicleTypeId", "Passenger vehicle types file")
    _require_column(vehicle_types, "vehicleCategory", "Passenger vehicle types file")

    category = vehicle_types["vehicleCategory"].astype(str)
    vehicle_type_id = vehicle_types["vehicleTypeId"].astype(str)
    bus_mask = category.eq("MediumDutyPassenger") & vehicle_type_id.str.contains("BUS-", case=False, na=False)
    return {
        "car": vehicle_types[category.eq("Car")].copy(),
        "bus": vehicle_types[bus_mask].copy(),
        "bike": vehicle_types[category.eq("Bike")].copy(),
        "other": vehicle_types[~(category.eq("Car") | bus_mask | category.eq("Bike"))].copy(),
    }


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
    atlas_passenger_category_mapping: pd.DataFrame,
    category_fuel_mapping: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(vehicle_type_mapping, "vehicleTypeId", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "body_type", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "modelyear", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "primaryVehicleEnergyFile", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "secondaryVehicleEnergyFile", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "msrp_usd", "Vehicle type mapping file")
    _require_column(category_fuel_mapping, "lookup_domain", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "vehicle_category", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "adopt_fuel", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "fastsim_relative_path", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "fastsim_fuel_type", "FASTSim category fuel mapping file")

    prepared_mapping = vehicle_type_mapping[
        [
            "vehicleTypeId",
            "body_type",
            "modelyear",
            "primaryVehicleEnergyFile",
            "secondaryVehicleEnergyFile",
            "msrp_usd",
        ]
    ].copy()
    prepared_mapping["vehicleTypeId"] = prepared_mapping["vehicleTypeId"].astype(str)
    prepared_mapping["modelyear"] = pd.to_numeric(prepared_mapping["modelyear"], errors="coerce")
    prepared_mapping["bodytype_norm"] = prepared_mapping["body_type"].apply(_normalize_bodytype)
    prepared_mapping["primaryVehicleEnergyFile"] = prepared_mapping["primaryVehicleEnergyFile"].map(_normalize_energy_file_path)
    prepared_mapping["secondaryVehicleEnergyFile"] = prepared_mapping["secondaryVehicleEnergyFile"].map(_normalize_energy_file_path)

    fuel_type_lookup = category_fuel_mapping[["fastsim_relative_path", "fastsim_fuel_type"]].drop_duplicates().copy()
    fuel_type_lookup["fastsim_relative_path"] = fuel_type_lookup["fastsim_relative_path"].astype(str)
    fuel_type_lookup["fastsim_fuel_type"] = fuel_type_lookup["fastsim_fuel_type"].astype(str)
    primary_lookup = fuel_type_lookup.rename(
        columns={"fastsim_relative_path": "primaryVehicleEnergyFile", "fastsim_fuel_type": "primaryFuelTypeId"}
    )
    secondary_lookup = fuel_type_lookup.rename(
        columns={"fastsim_relative_path": "secondaryVehicleEnergyFile", "fastsim_fuel_type": "secondaryFuelTypeId"}
    )
    prepared_mapping = prepared_mapping.merge(primary_lookup, on="primaryVehicleEnergyFile", how="left")
    prepared_mapping = prepared_mapping.merge(secondary_lookup, on="secondaryVehicleEnergyFile", how="left")
    prepared_mapping["secondaryFuelTypeId"] = prepared_mapping["secondaryFuelTypeId"].fillna("")

    prepared_built_vehicle_types = built_vehicle_types[["vehicleTypeId"]].copy()
    prepared_built_vehicle_types["vehicleTypeId"] = prepared_built_vehicle_types["vehicleTypeId"].astype(str)

    built_mapping = prepared_built_vehicle_types.merge(prepared_mapping, on="vehicleTypeId", how="left")
    missing = built_mapping[
        built_mapping["body_type"].isna()
        | built_mapping["primaryFuelTypeId"].isna()
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
    atlas_category_mapping = atlas_passenger_category_mapping.copy()
    atlas_category_mapping["body_type"] = atlas_category_mapping["body_type"].apply(_normalize_bodytype)
    atlas_category_mapping["passenger_beam_category"] = atlas_category_mapping["passenger_beam_category"].apply(
        _normalize_bodytype
    )
    atlas_keys = atlas_keys.merge(
        atlas_category_mapping.rename(
            columns={
                "body_type": "bodytype_norm",
                "passenger_beam_category": "passenger_bodytype_norm",
            }
        ),
        on="bodytype_norm",
        how="left",
    )
    missing_bodytype_mapping = atlas_keys[atlas_keys["passenger_bodytype_norm"].isna()]["bodytype_norm"].drop_duplicates()
    if not missing_bodytype_mapping.empty:
        raise ValueError(
            "ATLAS body types are missing passenger BEAM category mappings:\n"
            + "\n".join(missing_bodytype_mapping.astype(str).tolist())
        )
    atlas_keys["representativeIncomeK"] = pd.to_numeric(
        atlas_keys["representativeIncomeK"],
        errors="coerce",
    )
    prepared_category_mapping = category_fuel_mapping[
        ["lookup_domain", "vehicle_category", "adopt_fuel", "adopt_fuel_keys", "fastsim_fuel_type"]
    ].drop_duplicates().copy()
    prepared_category_mapping["lookup_domain"] = prepared_category_mapping["lookup_domain"].astype(str).str.lower()
    prepared_category_mapping["vehicle_category"] = prepared_category_mapping["vehicle_category"].astype(str)
    prepared_category_mapping["adopt_fuel"] = prepared_category_mapping["adopt_fuel"].astype(str).str.lower()
    prepared_category_mapping["adopt_fuel_keys"] = prepared_category_mapping["adopt_fuel_keys"].fillna("").astype(str).str.lower()
    prepared_category_mapping["fastsim_fuel_type"] = prepared_category_mapping["fastsim_fuel_type"].astype(str).str.lower()

    crosswalk_rows: list[pd.DataFrame] = []
    for atlas_vehicle_type_id, atlas_group in atlas_keys.groupby("atlasVehicleTypeId", sort=True, dropna=False):
        representative_row = (
            atlas_group.sort_values(["vehicleCount", "modelyear"], ascending=[False, False]).iloc[0]
        )
        candidate_frames: list[pd.DataFrame] = []

        for row in atlas_group.itertuples(index=False):
            row_fuel_mapping = prepared_category_mapping[
                (prepared_category_mapping["lookup_domain"] == "ldv")
                & (prepared_category_mapping["vehicle_category"] == row.passenger_bodytype_norm)
            ].copy()
            row_fuel_mapping = row_fuel_mapping[_adopt_fuel_matches(row_fuel_mapping, str(row.adopt_fuel).lower())].copy()
            if row_fuel_mapping.empty:
                raise ValueError(
                    "No FASTSim category fuel mapping configured for atlas "
                    f"bodytype={row.bodytype_norm}, adopt_fuel={row.adopt_fuel}"
                )

            candidates = built_mapping[
                built_mapping["bodytype_norm"] == row.passenger_bodytype_norm
            ].copy()
            candidates = candidates.merge(
                row_fuel_mapping,
                left_on=["primaryFuelTypeId"],
                right_on=["fastsim_fuel_type"],
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
                pd.to_numeric(candidates["yearWeight"], errors="coerce").fillna(0.0)
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
    category_fuel_mapping: pd.DataFrame,
) -> pd.DataFrame:
    _require_column(category_fuel_mapping, "lookup_domain", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "vehicle_category", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "fastsim_relative_path", "FASTSim category fuel mapping file")
    _require_column(category_fuel_mapping, "msrp_usd", "FASTSim category fuel mapping file")

    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(built_vehicle_types, "primaryFuelType", "Built vehicle types")
    _require_column(built_vehicle_types, "secondaryFuelType", "Built vehicle types")
    _require_column(built_vehicle_types, "primaryVehicleEnergyFile", "Built vehicle types")
    _require_column(built_vehicle_types, "secondaryVehicleEnergyFile", "Built vehicle types")

    fuels = built_vehicle_types[
        [
            "vehicleTypeId",
            "primaryFuelType",
            "secondaryFuelType",
            "primaryVehicleEnergyFile",
            "secondaryVehicleEnergyFile",
        ]
    ].copy()
    fuels["vehicleTypeId"] = fuels["vehicleTypeId"].astype(str)
    fuels["primaryFuelType"] = fuels["primaryFuelType"].astype(str).str.lower()
    fuels["secondaryFuelType"] = fuels["secondaryFuelType"].fillna("").astype(str).str.lower()
    fuels["primaryVehicleEnergyFile"] = fuels["primaryVehicleEnergyFile"].map(_normalize_energy_file_path)
    fuels["secondaryVehicleEnergyFile"] = fuels["secondaryVehicleEnergyFile"].map(_normalize_energy_file_path)
    fuels = fuels.drop_duplicates()

    bodytypes = category_fuel_mapping[
        category_fuel_mapping["passenger_beam_category"].astype(str).str.strip().ne("")
    ][["fastsim_relative_path", "passenger_beam_category", "msrp_usd"]].drop_duplicates().copy()
    bodytypes["passenger_beam_category"] = bodytypes["passenger_beam_category"].astype(str).map(_normalize_bodytype)
    bodytypes["msrp_usd"] = pd.to_numeric(bodytypes["msrp_usd"], errors="coerce")

    mapping = fuels.merge(
        bodytypes,
        left_on="primaryVehicleEnergyFile",
        right_on="fastsim_relative_path",
        how="inner",
    )
    mapping["modelyear"] = _extract_model_year_from_vehicle_type_id(mapping["vehicleTypeId"])
    mapping = mapping[
        [
            "vehicleTypeId",
            "passenger_beam_category",
            "modelyear",
            "primaryFuelType",
            "secondaryFuelType",
            "primaryVehicleEnergyFile",
            "secondaryVehicleEnergyFile",
            "msrp_usd",
        ]
    ].drop_duplicates()
    mapping = mapping.rename(columns={"passenger_beam_category": "body_type"})
    return mapping.sort_values(["vehicleTypeId", "body_type", "primaryFuelType", "secondaryFuelType"]).reset_index(drop=True)


def _build_passenger_fastsim_mapping_context(
    prepared_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    fastsim_registry = _load_fastsim_registry(config)
    category_fuel_mapping = _load_fastsim_category_fuel_mapping(config)
    atlas_passenger_category_mapping = _load_atlas_passenger_category_mapping(config)
    vehicle_type_mapping = _build_fastsim_vehicle_type_mapping(
        prepared_vehicle_types,
        category_fuel_mapping,
    )
    return {
        "fastsim_registry": fastsim_registry,
        "fastsim_category_fuel_mapping": category_fuel_mapping,
        "atlas_passenger_category_mapping": atlas_passenger_category_mapping,
        "vehicle_type_mapping": vehicle_type_mapping,
    }


def _build_freight_fastsim_mapping_context(config: dict[str, Any]) -> dict[str, Any]:
    category_fuel_mapping = _load_fastsim_category_fuel_mapping(config)
    lookup_indexes = _build_fastsim_lookup_indexes(config, category_fuel_mapping)
    return {
        "fastsim_category_fuel_mapping": category_fuel_mapping,
        "fastsim_lookup_indexes": lookup_indexes,
    }


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
    def _fuel_relative_path(file_name: Any) -> str:
        return _normalize_energy_file_path(file_name)

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
            template["primaryVehicleEnergyFile"] = _fuel_relative_path(charge_depleting.iloc[0]["file_name"])
            template["secondaryFuelType"] = secondary_fuel
            template["secondaryVehicleEnergyFile"] = (
                _fuel_relative_path(charge_sustaining.iloc[0]["file_name"])
                if not charge_sustaining.empty else ""
            )
        else:
            primary_fuel = plain.iloc[0]["fuel"] if not plain.empty else charge_sustaining.iloc[0]["fuel"]
            template = _select_template_row(vehicle_types, primary_fuel, "")
            template["primaryFuelType"] = primary_fuel
            template["primaryVehicleEnergyFile"] = _fuel_relative_path(
                plain.iloc[0]["file_name"] if not plain.empty else charge_sustaining.iloc[0]["file_name"]
            )
            template["secondaryFuelType"] = ""
            template["secondaryVehicleEnergyFile"] = ""

        template["vehicleTypeId"] = vehicle_type_id
        new_rows.append(template)

    prepared = pd.DataFrame(new_rows).reset_index(drop=True)
    return prepared


def _run_step1_passenger_substeps(
    *,
    config: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, Any]:
    fastsim_config = config["fastsim"]
    print("=== Step 1.1: build vehicle types for atlas.year from FASTSim passenger inputs ===")
    vehicle_types = _read_csv(
        fastsim_config["passenger_vehicle_types_file"],
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
    _require_column(vehicle_types, "vehicleTypeId", "FASTSim passenger vehicle types file")
    atlas_year = pd.to_numeric(config["atlas"]["year"], errors="coerce")

    encoded_years = _extract_model_year_from_vehicle_type_id(vehicle_types["vehicleTypeId"])
    has_year_below_atlas_year = encoded_years.notna().any() and bool((encoded_years < atlas_year).any())
    if has_year_below_atlas_year:
        prepared_vehicle_types = vehicle_types.copy()
        prepared_vehicle_types["modelyear"] = encoded_years
        prepared_vehicle_types = prepared_vehicle_types[prepared_vehicle_types["modelyear"].le(atlas_year)].copy()
    else:
        fastsim_data_folder = fastsim_config.get("ldv_fastsim_data_folder")
        if fastsim_data_folder in (None, ""):
            raise ValueError(
                "fastsim.ldv_fastsim_data_folder is required when fastsim.passenger_vehicle_types_file does not contain model years at or below atlas.year"
            )
        prepared_vehicle_types = _build_vehicle_types_from_fastsim_folder(
            vehicle_types,
            fastsim_data_folder,
            atlas_year,
        )
        prepared_vehicle_types["modelyear"] = _extract_model_year_from_vehicle_type_id(prepared_vehicle_types["vehicleTypeId"])

    print("=== Step 1.1a: load FASTSim registry and build passenger match context ===")
    passenger_fastsim_context = _build_passenger_fastsim_mapping_context(
        prepared_vehicle_types,
        config,
    )
    fastsim_registry = passenger_fastsim_context["fastsim_registry"]
    fastsim_category_fuel_mapping = passenger_fastsim_context["fastsim_category_fuel_mapping"]
    atlas_passenger_category_mapping = passenger_fastsim_context["atlas_passenger_category_mapping"]
    vehicle_type_mapping = passenger_fastsim_context["vehicle_type_mapping"]

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

    print("=== Step 1.3: create fastsim-to-atlas crosswalk file ===")
    vehicle_type_atlas_crosswalk = _build_vehicle_type_atlas_crosswalk(
        prepared_vehicle_types,
        vehicle_type_mapping,
        atlas_vehicle_type_targets,
        atlas_passenger_category_mapping,
        fastsim_category_fuel_mapping,
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
    prepared_vehicle_types = _align_vehicle_types_to_source_schema(
        prepared_vehicle_types,
        list(vehicle_types.columns),
        frame_name="Built passenger vehicle types",
    )
    prepared_vehicle_types = _normalize_energy_file_columns(prepared_vehicle_types)
    prepared_vehicle_types = _attach_adopt_fuel_column(prepared_vehicle_types)
    prepared_vehicle_types_file = _write_new_vehicle_types_file(prepared_vehicle_types, config["output"])
    vehicle_type_atlas_crosswalk_file = _write_vehicle_type_crosswalk_file(
        vehicle_type_atlas_crosswalk,
        config["output"],
    )

    print("=== Step 1.4: extract non-car passenger vehicle types from the source passenger vehicle-types file ===")
    passenger_vehicle_type_sections = _split_passenger_vehicle_types(vehicle_types)
    built_passenger_bus_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["bus"],
        list(vehicle_types.columns),
        frame_name="Built passenger bus vehicle types",
    )
    built_passenger_bus_vehicle_types = _normalize_energy_file_columns(built_passenger_bus_vehicle_types)
    built_passenger_bus_vehicle_types = _attach_adopt_fuel_column(built_passenger_bus_vehicle_types)
    built_passenger_bike_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["bike"],
        list(vehicle_types.columns),
        frame_name="Built passenger bike vehicle types",
    )
    built_passenger_bike_vehicle_types = _normalize_energy_file_columns(built_passenger_bike_vehicle_types)
    built_passenger_bike_vehicle_types = _attach_adopt_fuel_column(built_passenger_bike_vehicle_types)
    built_passenger_other_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["other"],
        list(vehicle_types.columns),
        frame_name="Built passenger other vehicle types",
    )
    built_passenger_other_vehicle_types = _normalize_energy_file_columns(built_passenger_other_vehicle_types)
    built_passenger_other_vehicle_types = _attach_adopt_fuel_column(built_passenger_other_vehicle_types)
    built_passenger_bus_vehicle_types_file = _write_passenger_section_vehicle_types_file(
        "bus",
        built_passenger_bus_vehicle_types,
        config["output"],
    )
    built_passenger_bike_vehicle_types_file = _write_passenger_section_vehicle_types_file(
        "bike",
        built_passenger_bike_vehicle_types,
        config["output"],
    )
    built_passenger_other_vehicle_types_file = _write_passenger_section_vehicle_types_file(
        "other",
        built_passenger_other_vehicle_types,
        config["output"],
    )

    return {
        "source_fastsim_passenger_vehicle_types": vehicle_types,
        "source_atlas_vehicles": atlas_vehicles,
        "source_atlas_households": atlas_households,
        "source_atlas_persons": atlas_persons,
        "source_fastsim_registry": fastsim_registry,
        "source_fastsim_category_fuel_mapping": fastsim_category_fuel_mapping,
        "source_fastsim_passenger_vehicle_type_mapping": vehicle_type_mapping,
        "built_vehicle_types": prepared_vehicle_types,
        "built_vehicle_types_file": prepared_vehicle_types_file,
        "atlas_vehicle_type_targets": atlas_vehicle_type_targets,
        "atlas_income_bin_targets": atlas_income_bin_targets,
        "vehicle_type_atlas_crosswalk": vehicle_type_atlas_crosswalk,
        "vehicle_type_atlas_crosswalk_file": vehicle_type_atlas_crosswalk_file,
        "passenger_vehicle_type_sections": passenger_vehicle_type_sections,
        "built_passenger_bus_vehicle_types": built_passenger_bus_vehicle_types,
        "built_passenger_bus_vehicle_types_file": built_passenger_bus_vehicle_types_file,
        "built_passenger_bike_vehicle_types": built_passenger_bike_vehicle_types,
        "built_passenger_bike_vehicle_types_file": built_passenger_bike_vehicle_types_file,
        "built_passenger_other_vehicle_types": built_passenger_other_vehicle_types,
        "built_passenger_other_vehicle_types_file": built_passenger_other_vehicle_types_file,
    }


def _run_step1_freight_substeps(
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    fastsim_config = config["fastsim"]
    print("=== Step 1.5: build freight vehicle-type population from FRISM carriers and tours ===")
    frism_carriers = _read_csv(
        config["frism"]["carriers_file"],
        columns=["tourId", "vehicleTypeId"],
    )
    frism_tours = _read_csv(
        config["frism"]["tours_file"],
        columns=["tourId"],
    )
    freight_vehicle_type_population = _build_freight_vehicle_type_population(
        frism_carriers,
        frism_tours,
    )

    print("=== Step 1.6: filter freight vehicle types to the freight population and calculate probabilities ===")
    freight_vehicle_types = _read_csv(
        fastsim_config["freight_vehicle_types_file"],
    )
    freight_vehicle_type_population = _round_fleet_share(freight_vehicle_type_population)
    built_freight_vehicle_types = _build_freight_vehicle_types_from_population(
        freight_vehicle_types,
        freight_vehicle_type_population,
    )

    print("=== Step 1.7: assign FASTSim energy files to freight vehicle types ===")
    print("=== Step 1.7a: load FASTSim registry and build freight lookup context ===")
    freight_fastsim_context = _build_freight_fastsim_mapping_context(config)
    built_freight_vehicle_types = _assign_freight_fastsim_energy_files(
        built_freight_vehicle_types,
        config,
        category_fuel_mapping=freight_fastsim_context["fastsim_category_fuel_mapping"],
        lookup_indexes=freight_fastsim_context["fastsim_lookup_indexes"],
    )
    built_freight_vehicle_types = _normalize_energy_file_columns(built_freight_vehicle_types)
    built_freight_vehicle_types = _attach_adopt_fuel_column(built_freight_vehicle_types)
    built_freight_vehicle_types_file = _write_new_freight_vehicle_types_file(
        built_freight_vehicle_types,
        config["output"],
    )

    return {
        "source_frism_carriers": frism_carriers,
        "source_frism_tours": frism_tours,
        "freight_vehicle_type_population": freight_vehicle_type_population,
        "source_fastsim_freight_vehicle_types": freight_vehicle_types,
        "source_freight_fastsim_category_fuel_mapping": freight_fastsim_context["fastsim_category_fuel_mapping"],
        "built_freight_vehicle_types": built_freight_vehicle_types,
        "built_freight_vehicle_types_file": built_freight_vehicle_types_file,
    }


def run_step1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 1: build BEAM vehicle types for the target ATLAS year."""
    config = workflow["config"]
    rng = np.random.default_rng(int(config["seed"]))
    _remove_stale_step1_output_files(config["output"])
    workflow.update(_run_step1_passenger_substeps(config=config, rng=rng))
    workflow.update(_run_step1_freight_substeps(config=config))
    return workflow
