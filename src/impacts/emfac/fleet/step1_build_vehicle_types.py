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
    return Path(resolve_workflow_path(output_dir)) / "tmp"


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


def _map_freight_beam_class(vehicle_type_id: object, vehicle_category: object, vehicle_class: object) -> str:
    normalized_id = _normalize_freight_vehicle_type_id(vehicle_type_id)
    prefix_map = {
        "Ld1": "Class12aVocational",
        "Ld3": "Class2b3Vocational",
        "Mdv": "Class456Vocational",
        "Hdt": "Class78Tractor",
        "Hdv": "Class78Vocational",
    }
    for prefix, beam_class in prefix_map.items():
        if normalized_id.startswith(prefix):
            return beam_class

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
        "Could not determine freight BEAM class for vehicleTypeId="
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
    frame = _read_csv(config["mapping"]["frism_atlas_map"])
    _require_column(frame, "body_type", "FRISM ATLAS bodytype crosswalk file")
    weight_columns = [column for column in frame.columns if column != "body_type"]
    if not weight_columns:
        raise ValueError("FRISM ATLAS bodytype crosswalk file has no freight class columns")
    prepared = frame.melt(
        id_vars=["body_type"],
        value_vars=weight_columns,
        var_name="beamClass",
        value_name="weight",
    )
    prepared["body_type"] = prepared["body_type"].apply(_normalize_bodytype)
    prepared["beamClass"] = prepared["beamClass"].astype(str)
    prepared["weight"] = pd.to_numeric(prepared["weight"], errors="coerce").fillna(0.0)
    return prepared[prepared["weight"].gt(0)].reset_index(drop=True)


def _build_ldv_fastsim_lookup_index(fastsim_data_folder: str) -> pd.DataFrame:
    folder = Path(resolve_workflow_path(fastsim_data_folder))
    rows: list[dict[str, str]] = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or not (path.name.endswith(".csv") or path.name.endswith(".csv.gz")):
            continue
        parsed = _parse_fastsim_lookup_filename(path)
        if parsed is None:
            continue
        parsed["relative_path"] = f"{folder.name}/{path.name}"
        rows.append(parsed)
    if not rows:
        raise ValueError(f"No LDV FASTSim lookup files were found under {folder}")
    return pd.DataFrame(rows)


def _build_mhdv_fastsim_lookup_index(fastsim_data_folder: str) -> pd.DataFrame:
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
                "fastsim_token": match.group("token").strip().lower(),
                "modelyear": match.group("year"),
                "relative_path": f"{folder.name}/{path.name}",
            }
        )
    if not rows:
        raise ValueError(f"No MHDV FASTSim lookup files were found under {folder}")
    return pd.DataFrame(rows)


def _load_freight_fastsim_bodytype_xwalk(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read_csv(config["mapping"]["fastsim_frism_bodytype_xwalk_file"])
    for column_name in ["fastsim_id", "body_type"]:
        _require_column(frame, column_name, "FRISM FASTSim bodytype xwalk file")
    prepared = frame.copy()
    prepared["fastsim_id"] = prepared["fastsim_id"].astype(str)
    prepared["body_type"] = prepared["body_type"].astype(str)
    prepared["representative_stem"] = prepared["fastsim_id"].map(lambda value: str(value).split("_(", 1)[0])
    return prepared


def _load_freight_fastsim_fuel_mapping(config: dict[str, Any]) -> pd.DataFrame:
    frame = _read_csv(config["mapping"]["fastsim_frism_fuel_mapping_file"])
    for column_name in [
        "fastsim_source",
        "beam_primary_fuel",
        "beam_secondary_fuel",
        "fastsim_primary_token",
        "fastsim_secondary_token",
    ]:
        _require_column(frame, column_name, "FRISM FASTSim fuel mapping file")
    prepared = frame.copy()
    for column_name in [
        "fastsim_source",
        "beam_primary_fuel",
        "beam_secondary_fuel",
        "fastsim_primary_token",
        "fastsim_secondary_token",
    ]:
        prepared[column_name] = prepared[column_name].fillna("").astype(str).str.strip().str.lower()
    return prepared


def _select_freight_fastsim_fuel_mapping_row(
    fuel_mapping: pd.DataFrame,
    *,
    fastsim_source: str,
    primary_fuel: object,
    secondary_fuel: object,
) -> pd.Series:
    primary_key = _normalize_lower(primary_fuel)
    secondary_key = _normalize_lower(secondary_fuel)
    candidates = fuel_mapping[
        (fuel_mapping["fastsim_source"] == fastsim_source)
        & (fuel_mapping["beam_primary_fuel"] == primary_key)
    ].copy()
    if candidates.empty:
        raise ValueError(
            "No FRISM FASTSim fuel mapping row found for "
            f"fastsim_source={fastsim_source}, primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )

    def _secondary_match_score(value: str) -> float:
        token = str(value).strip().lower()
        if token == secondary_key:
            return 2.0
        if token == "any":
            return 1.0
        if token == "" and secondary_key == "":
            return 0.5
        return -1.0

    candidates["matchScore"] = candidates["beam_secondary_fuel"].map(_secondary_match_score)
    candidates = candidates[candidates["matchScore"].ge(0)].copy()
    if candidates.empty:
        raise ValueError(
            "No FRISM FASTSim fuel mapping row matched secondary fuel for "
            f"fastsim_source={fastsim_source}, primaryFuelType={primary_fuel}, secondaryFuelType={secondary_fuel}"
        )
    return candidates.sort_values(
        ["matchScore", "fastsim_secondary_token"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]


def _select_ldv_fastsim_energy_files(
    *,
    body_type: str,
    fuel_row: pd.Series,
    atlas_bodytype_xwalk: pd.DataFrame,
    ldv_lookup_index: pd.DataFrame,
) -> tuple[str, str]:
    _require_column(atlas_bodytype_xwalk, "fastsim_id", "FASTSim ATLAS bodytype xwalk file")
    _require_column(atlas_bodytype_xwalk, "body_type", "FASTSim ATLAS bodytype xwalk file")
    atlas_xwalk = atlas_bodytype_xwalk.copy()
    atlas_xwalk["body_type"] = atlas_xwalk["body_type"].apply(_normalize_bodytype)
    atlas_xwalk["fastsim_id"] = atlas_xwalk["fastsim_id"].astype(str)
    atlas_xwalk["vehicleTypeId"] = atlas_xwalk["fastsim_id"].apply(_vehicle_type_id_from_fastsim_token)

    candidates = atlas_xwalk[atlas_xwalk["body_type"] == body_type].copy()
    if candidates.empty:
        raise ValueError(f"No LDV FASTSim bodytype mapping found for body_type={body_type}")

    candidate_ids = set(candidates["vehicleTypeId"].astype(str))
    primary_token = str(fuel_row["fastsim_primary_token"]).strip().lower()
    secondary_token = str(fuel_row["fastsim_secondary_token"]).strip().lower()

    if secondary_token:
        primary_files = ldv_lookup_index[
            (ldv_lookup_index["vehicleTypeId"].astype(str).isin(candidate_ids))
            & (ldv_lookup_index["fuel"].astype(str).str.lower() == primary_token)
            & (ldv_lookup_index["charge_mode"] == "Charge_Depleting")
        ].copy()
        secondary_files = ldv_lookup_index[
            (ldv_lookup_index["vehicleTypeId"].astype(str).isin(candidate_ids))
            & (ldv_lookup_index["fuel"].astype(str).str.lower() == secondary_token)
            & (ldv_lookup_index["charge_mode"] == "Charge_Sustaining")
        ].copy()
        if primary_files.empty or secondary_files.empty:
            raise ValueError(
                f"No LDV FASTSim hybrid lookup pair found for body_type={body_type}, "
                f"primary token={primary_token}, secondary token={secondary_token}"
            )
        primary_files["modelyear"] = pd.to_numeric(primary_files["modelyear"], errors="coerce")
        secondary_files["modelyear"] = pd.to_numeric(secondary_files["modelyear"], errors="coerce")
        paired = primary_files.merge(
            secondary_files[["vehicleTypeId", "modelyear", "relative_path"]],
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
        return str(selected["relative_path_primary"]), str(selected["relative_path_secondary"])

    files = ldv_lookup_index[
        (ldv_lookup_index["vehicleTypeId"].astype(str).isin(candidate_ids))
        & (ldv_lookup_index["fuel"].astype(str).str.lower() == primary_token)
        & (ldv_lookup_index["charge_mode"] == "")
    ].copy()
    if files.empty:
        raise ValueError(
            f"No LDV FASTSim lookup file found for body_type={body_type}, primary token={primary_token}"
        )
    files["modelyear"] = pd.to_numeric(files["modelyear"], errors="coerce")
    selected = files.sort_values(
        ["modelyear", "vehicleTypeId"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return str(selected["relative_path"]), ""


def _select_mhdv_fastsim_energy_files(
    *,
    beam_class: str,
    fuel_row: pd.Series,
    frism_bodytype_xwalk: pd.DataFrame,
    mhdv_lookup_index: pd.DataFrame,
) -> tuple[str, str]:
    candidates = frism_bodytype_xwalk[frism_bodytype_xwalk["body_type"] == beam_class].copy()
    if candidates.empty:
        raise ValueError(f"No MHDV FASTSim bodytype mapping found for beam class {beam_class}")
    representative_stems = set(candidates["representative_stem"].astype(str))
    primary_token = str(fuel_row["fastsim_primary_token"]).strip().lower()
    files = mhdv_lookup_index[
        (mhdv_lookup_index["representative_stem"].astype(str).isin(representative_stems))
        & (mhdv_lookup_index["fastsim_token"].astype(str).str.lower() == primary_token)
    ].copy()
    if files.empty:
        raise ValueError(
            f"No MHDV FASTSim lookup file found for beam class={beam_class}, primary token={primary_token}"
        )
    files["modelyear"] = pd.to_numeric(files["modelyear"], errors="coerce")
    selected = files.sort_values(
        ["modelyear", "representative_stem"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    return str(selected["relative_path"]), ""


def _assign_freight_fastsim_energy_files(
    freight_vehicle_types: pd.DataFrame,
    config: dict[str, Any],
    atlas_bodytype_xwalk: pd.DataFrame,
) -> pd.DataFrame:
    prepared = freight_vehicle_types.copy()
    frism_atlas_map = _load_frism_atlas_bodytype_mapping(config)
    frism_bodytype_xwalk = _load_freight_fastsim_bodytype_xwalk(config)
    fuel_mapping = _load_freight_fastsim_fuel_mapping(config)
    ldv_lookup_index = _build_ldv_fastsim_lookup_index(config["fastsim"]["ldv_fastsim_data_folder"])
    mhdv_lookup_index = _build_mhdv_fastsim_lookup_index(config["fastsim"]["mhdv_fastsim_data_folder"])

    primary_paths: list[str] = []
    secondary_paths: list[str] = []
    for row in prepared.itertuples(index=False):
        beam_class = _map_freight_beam_class(
            getattr(row, "vehicleTypeId"),
            getattr(row, "vehicleCategory", ""),
            getattr(row, "vehicleClass", ""),
        )
        fastsim_source = "ldv" if beam_class in {"Class12aVocational", "Class2b3Vocational"} else "mhdv"
        fuel_row = _select_freight_fastsim_fuel_mapping_row(
            fuel_mapping,
            fastsim_source=fastsim_source,
            primary_fuel=getattr(row, "primaryFuelType", ""),
            secondary_fuel=getattr(row, "secondaryFuelType", ""),
        )
        if fastsim_source == "ldv":
            bodytype_candidates = frism_atlas_map[frism_atlas_map["beamClass"] == beam_class].copy()
            if bodytype_candidates.empty:
                raise ValueError(f"No FRISM ATLAS bodytype mapping found for beam class {beam_class}")
            selected_body_type = (
                bodytype_candidates.sort_values(["weight", "body_type"], ascending=[False, True], kind="mergesort").iloc[0]["body_type"]
            )
            primary_path, secondary_path = _select_ldv_fastsim_energy_files(
                body_type=str(selected_body_type),
                fuel_row=fuel_row,
                atlas_bodytype_xwalk=atlas_bodytype_xwalk,
                ldv_lookup_index=ldv_lookup_index,
            )
        else:
            primary_path, secondary_path = _select_mhdv_fastsim_energy_files(
                beam_class=beam_class,
                fuel_row=fuel_row,
                frism_bodytype_xwalk=frism_bodytype_xwalk,
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
    fuel_mapping: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    _require_column(built_vehicle_types, "vehicleTypeId", "Built vehicle types")
    _require_column(vehicle_type_mapping, "vehicleTypeId", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "body_type", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "modelyear", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "primaryFuelType", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "secondaryFuelType", "Vehicle type mapping file")
    _require_column(vehicle_type_mapping, "msrp_usd", "Vehicle type mapping file")
    _require_column(fuel_mapping, "atlas_adopt_fuel", "FASTSim ATLAS fuel mapping file")
    _require_column(fuel_mapping, "fastsim_primary_fuel", "FASTSim ATLAS fuel mapping file")
    _require_column(fuel_mapping, "fastsim_secondary_fuel", "FASTSim ATLAS fuel mapping file")
    _require_column(fuel_mapping, "weight", "FASTSim ATLAS fuel mapping file")

    prepared_mapping = vehicle_type_mapping[
        ["vehicleTypeId", "body_type", "modelyear", "primaryFuelType", "secondaryFuelType", "msrp_usd"]
    ].copy()
    prepared_mapping["vehicleTypeId"] = prepared_mapping["vehicleTypeId"].astype(str)
    prepared_mapping["modelyear"] = pd.to_numeric(prepared_mapping["modelyear"], errors="coerce")
    prepared_mapping["bodytype_norm"] = prepared_mapping["body_type"].apply(_normalize_bodytype)
    prepared_mapping["primaryFuelType"] = prepared_mapping["primaryFuelType"].astype(str).str.lower()
    prepared_mapping["secondaryFuelType"] = prepared_mapping["secondaryFuelType"].fillna("").astype(str).str.lower()

    prepared_built_vehicle_types = built_vehicle_types[["vehicleTypeId"]].copy()
    prepared_built_vehicle_types["vehicleTypeId"] = prepared_built_vehicle_types["vehicleTypeId"].astype(str)

    built_mapping = prepared_built_vehicle_types.merge(prepared_mapping, on="vehicleTypeId", how="left")
    missing = built_mapping[
        built_mapping["body_type"].isna()
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
    _require_column(fastsim_bodytype_xwalk, "body_type", "FASTSim bodytype xwalk file")
    _require_column(fastsim_bodytype_xwalk, "fastsim_id", "FASTSim bodytype xwalk file")
    _require_column(fastsim_bodytype_xwalk, "msrp_usd", "FASTSim bodytype xwalk file")

    bodytypes = fastsim_bodytype_xwalk[["fastsim_id", "body_type", "msrp_usd"]].copy()
    bodytypes["vehicleTypeId"] = bodytypes["fastsim_id"].apply(_vehicle_type_id_from_fastsim_token)
    bodytypes["body_type"] = bodytypes["body_type"].astype(str)
    bodytypes["msrp_usd"] = pd.to_numeric(bodytypes["msrp_usd"], errors="coerce")
    bodytypes = bodytypes[["vehicleTypeId", "body_type", "msrp_usd"]].drop_duplicates()

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
        ["vehicleTypeId", "body_type", "modelyear", "primaryFuelType", "secondaryFuelType", "msrp_usd"]
    ].drop_duplicates()
    return mapping.sort_values(["vehicleTypeId", "body_type", "primaryFuelType", "secondaryFuelType"]).reset_index(drop=True)


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


def run_step1(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 1: build BEAM vehicle types for the target ATLAS year."""
    config = workflow["config"]
    rng = np.random.default_rng(int(config["seed"]))
    _remove_stale_step1_output_files(config["output"])

    fastsim_config = config["fastsim"]
    mapping_config = config["mapping"]

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

    fastsim_bodytype_xwalk = _read_csv(mapping_config["fastsim_atlas_bodytype_xwalk_file"])
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
    print("=== Step 1.3: create fastsim-to-atlas crosswalk file ===")
    fastsim_atlas_fuel_mapping = _read_csv(mapping_config["fastsim_atlas_fuel_mapping_file"])
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
    prepared_vehicle_types = _align_vehicle_types_to_source_schema(
        prepared_vehicle_types,
        list(vehicle_types.columns),
        frame_name="Built passenger vehicle types",
    )
    prepared_vehicle_types = _normalize_energy_file_columns(prepared_vehicle_types)
    prepared_vehicle_types_file = _write_new_vehicle_types_file(prepared_vehicle_types, config["output"])
    vehicle_type_atlas_crosswalk_file = _write_vehicle_type_crosswalk_file(
        vehicle_type_atlas_crosswalk,
        config["output"],
    )
    print("=== Step 1.4: build freight vehicle-type population from FRISM carriers and tours ===")
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
    print("=== Step 1.5: filter freight vehicle types to the freight population and calculate probabilities ===")
    freight_vehicle_types = _read_csv(
        fastsim_config["freight_vehicle_types_file"],
    )
    freight_vehicle_type_population = _round_fleet_share(freight_vehicle_type_population)
    built_freight_vehicle_types = _build_freight_vehicle_types_from_population(
        freight_vehicle_types,
        freight_vehicle_type_population,
    )
    print("=== Step 1.6: assign FASTSim energy files to freight vehicle types ===")
    built_freight_vehicle_types = _assign_freight_fastsim_energy_files(
        built_freight_vehicle_types,
        config,
        fastsim_bodytype_xwalk,
    )
    built_freight_vehicle_types = _normalize_energy_file_columns(built_freight_vehicle_types)
    built_freight_vehicle_types_file = _write_new_freight_vehicle_types_file(
        built_freight_vehicle_types,
        config["output"],
    )
    print("=== Step 1.7: extract non-car passenger vehicle types from the source passenger vehicle-types file ===")
    passenger_vehicle_type_sections = _split_passenger_vehicle_types(vehicle_types)
    built_passenger_bus_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["bus"],
        list(vehicle_types.columns),
        frame_name="Built passenger bus vehicle types",
    )
    built_passenger_bus_vehicle_types = _normalize_energy_file_columns(built_passenger_bus_vehicle_types)
    built_passenger_bike_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["bike"],
        list(vehicle_types.columns),
        frame_name="Built passenger bike vehicle types",
    )
    built_passenger_bike_vehicle_types = _normalize_energy_file_columns(built_passenger_bike_vehicle_types)
    built_passenger_other_vehicle_types = _align_vehicle_types_to_source_schema(
        passenger_vehicle_type_sections["other"],
        list(vehicle_types.columns),
        frame_name="Built passenger other vehicle types",
    )
    built_passenger_other_vehicle_types = _normalize_energy_file_columns(built_passenger_other_vehicle_types)
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

    workflow["source_fastsim_passenger_vehicle_types"] = vehicle_types
    workflow["source_atlas_vehicles"] = atlas_vehicles
    workflow["source_atlas_households"] = atlas_households
    workflow["source_atlas_persons"] = atlas_persons
    workflow["source_fastsim_passenger_bodytype_xwalk"] = fastsim_bodytype_xwalk
    workflow["source_fastsim_passenger_fuel_mapping"] = fastsim_atlas_fuel_mapping
    workflow["source_fastsim_passenger_vehicle_type_mapping"] = vehicle_type_mapping
    workflow["built_vehicle_types"] = prepared_vehicle_types
    workflow["built_vehicle_types_file"] = prepared_vehicle_types_file
    workflow["atlas_vehicle_type_targets"] = atlas_vehicle_type_targets
    workflow["atlas_income_bin_targets"] = atlas_income_bin_targets
    workflow["vehicle_type_atlas_crosswalk"] = vehicle_type_atlas_crosswalk
    workflow["vehicle_type_atlas_crosswalk_file"] = vehicle_type_atlas_crosswalk_file
    workflow["source_frism_carriers"] = frism_carriers
    workflow["source_frism_tours"] = frism_tours
    workflow["freight_vehicle_type_population"] = freight_vehicle_type_population
    workflow["source_fastsim_freight_vehicle_types"] = freight_vehicle_types
    workflow["built_freight_vehicle_types"] = built_freight_vehicle_types
    workflow["built_freight_vehicle_types_file"] = built_freight_vehicle_types_file
    workflow["passenger_vehicle_type_sections"] = passenger_vehicle_type_sections
    workflow["built_passenger_bus_vehicle_types"] = built_passenger_bus_vehicle_types
    workflow["built_passenger_bus_vehicle_types_file"] = built_passenger_bus_vehicle_types_file
    workflow["built_passenger_bike_vehicle_types"] = built_passenger_bike_vehicle_types
    workflow["built_passenger_bike_vehicle_types_file"] = built_passenger_bike_vehicle_types_file
    workflow["built_passenger_other_vehicle_types"] = built_passenger_other_vehicle_types
    workflow["built_passenger_other_vehicle_types_file"] = built_passenger_other_vehicle_types_file
    return workflow
