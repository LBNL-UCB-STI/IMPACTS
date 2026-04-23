"""Fleet Step 5: attach EMFAC rates stores to passenger and freight vehicle types."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd

from impacts.emfac.config import resolve_workflow_path
from impacts.emfac.fleet.step2_map_emfac_bus_bike import _load_emfac_category_fuel_mapping
from impacts.emfac.fleet.step2_map_emfac_bus_bike import _matched_emfac_fuels


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def _sanitize_output_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str("" if pd.isna(value) else value).strip()).strip("-")


def _build_year_scenario_token(*, year: object, scenario: object) -> str:
    year_token = _sanitize_output_component(year)
    scenario_token = _sanitize_output_component(scenario)
    if year_token and scenario_token:
        return f"{year_token}-{scenario_token}"
    return year_token or scenario_token


def _build_vehicle_types_output_filename(*, source_name: str, year: object, scenario: object) -> str:
    scenario_token = _build_year_scenario_token(year=year, scenario=scenario)
    source_token = _sanitize_output_component(source_name)
    return f"vehicleTypes--{source_token}--{scenario_token}--EM.csv"


def _write_vehicle_types(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return str(output_path)


def _write_parquet(frame: pd.DataFrame, path_like: str) -> str:
    output_path = Path(resolve_workflow_path(path_like))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return str(output_path)


def _assign_rate_filepaths(vehicle_types: pd.DataFrame, relative_paths: dict[str, str]) -> pd.DataFrame:
    prepared = vehicle_types.copy()
    prepared["emfacId"] = prepared.get("emfacId", "").fillna("").astype(str)
    prepared["emissionsRatesFile"] = prepared["emfacId"].map(relative_paths).fillna("")
    return prepared


_SPECIAL_CATEGORY_FUEL_DOMAINS = {
    "UBUS": "mhdv",
    "MCY": "ldv",
}


def _build_fleet_average_emfac_id(*, vehicle_category: str, fuel: str) -> str:
    return f"fleetavg{_sanitize_output_component(vehicle_category)}{_sanitize_output_component(fuel)}"


def _aggregate_rate_frames_by_vmt_share(frames: list[pd.DataFrame], *, weights: list[float]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    working_frames: list[pd.DataFrame] = []
    for frame, weight in zip(frames, weights, strict=False):
        prepared = frame.copy()
        prepared["_fleet_weight"] = float(weight)
        working_frames.append(prepared)
    combined = pd.concat(working_frames, ignore_index=True, sort=False)
    rate_columns = [column for column in combined.columns if column.endswith("_gram")]
    if not rate_columns:
        raise ValueError("Rates store partitions do not include any '*_gram' pollutant rate columns to aggregate.")
    group_columns = [column for column in combined.columns if column not in {"_fleet_weight", "modelYear", *rate_columns}]
    aggregated = combined[group_columns].drop_duplicates().reset_index(drop=True)
    for column in rate_columns:
        values = pd.to_numeric(combined[column], errors="coerce")
        valid = values.notna() & combined["_fleet_weight"].notna() & (combined["_fleet_weight"] > 0)
        if not valid.any():
            aggregated[column] = pd.NA
            continue
        weighted = combined.loc[valid, group_columns].copy()
        weighted["_weighted_value"] = values.loc[valid] * combined.loc[valid, "_fleet_weight"]
        weighted["_fleet_weight"] = combined.loc[valid, "_fleet_weight"]
        weighted = (
            weighted.groupby(group_columns, dropna=False)[["_weighted_value", "_fleet_weight"]]
            .sum()
            .reset_index()
        )
        weighted[column] = weighted["_weighted_value"] / weighted["_fleet_weight"]
        aggregated = aggregated.merge(weighted[group_columns + [column]], on=group_columns, how="left")
    if "modelYear" in combined.columns:
        aggregated["modelYear"] = "fleetAvg"
    return aggregated


def _write_special_rates_partition(
    *,
    output_root: Path,
    store_root: Path,
    emfac_id: str,
    frame: pd.DataFrame,
) -> str:
    partition_dir = store_root / "dataset" / f"emfacId={emfac_id}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    output_path = partition_dir / f"{emfac_id}.parquet"
    frame.to_parquet(output_path, index=False)
    return str(output_path.relative_to(output_root))


def _build_special_category_rate_aliases(
    *,
    config: dict[str, Any],
    shared_rates_store: dict[str, Any],
) -> dict[tuple[str, str], str]:
    passenger_fleet_path = Path(str(config["activities"]["passenger_fleet_file"])).expanduser().resolve()
    passenger_fleet = pd.read_parquet(passenger_fleet_path)
    required = {"vehicleCategory", "fuel", "modelYear", "vmtShare"}
    missing = sorted(required - set(passenger_fleet.columns))
    if missing:
        raise ValueError(f"Passenger fleet file is missing required columns for special rate aliases: {missing}")
    relative_paths: dict[str, str] = shared_rates_store["relative_paths"]
    output_root = Path(str(config["output"])).expanduser().resolve()
    store_root = Path(str(shared_rates_store["store_root"])).expanduser().resolve()
    aliases: dict[tuple[str, str], str] = {}
    special_fleet = passenger_fleet.loc[
        passenger_fleet["vehicleCategory"].astype(str).isin(_SPECIAL_CATEGORY_FUEL_DOMAINS)
    ].copy()
    if special_fleet.empty:
        return aliases
    special_fleet["vehicleCategory"] = special_fleet["vehicleCategory"].astype(str)
    special_fleet["fuel"] = special_fleet["fuel"].astype(str)
    special_fleet["modelYear"] = special_fleet["modelYear"].astype(str)
    special_fleet["vmtShare"] = pd.to_numeric(special_fleet["vmtShare"], errors="coerce")
    for (vehicle_category, fuel), group in special_fleet.groupby(["vehicleCategory", "fuel"], dropna=False):
        valid = group.loc[group["vmtShare"].notna() & (group["vmtShare"] > 0)].copy()
        if valid.empty:
            continue
        frames: list[pd.DataFrame] = []
        weights: list[float] = []
        for row in valid.itertuples(index=False):
            source_emfac_id = f"{str(row.modelYear).strip()}{str(vehicle_category).strip()}{str(fuel).strip()}"
            relative_path = relative_paths.get(source_emfac_id)
            if not relative_path:
                raise FileNotFoundError(
                    "Special fleet-average rate alias requires an existing EMFAC rates partition for "
                    f"{source_emfac_id}"
                )
            frame = pd.read_parquet(output_root / relative_path)
            frames.append(frame)
            weights.append(float(row.vmtShare))
        aggregated = _aggregate_rate_frames_by_vmt_share(frames, weights=weights)
        alias_emfac_id = _build_fleet_average_emfac_id(vehicle_category=str(vehicle_category), fuel=str(fuel))
        relative_path = _write_special_rates_partition(
            output_root=output_root,
            store_root=store_root,
            emfac_id=alias_emfac_id,
            frame=aggregated,
        )
        relative_paths[alias_emfac_id] = relative_path
        aliases[(str(vehicle_category), str(fuel))] = relative_path
    return aliases


def _override_special_category_rate_paths(
    vehicle_types: pd.DataFrame,
    *,
    config: dict[str, Any],
    special_rate_aliases: dict[tuple[str, str], str],
) -> pd.DataFrame:
    if not special_rate_aliases:
        return vehicle_types
    prepared = vehicle_types.copy()
    _require_column(prepared, "adopt_fuel", "Vehicle types table")
    _require_column(prepared, "emfacVehicleCategory", "Vehicle types table")
    category_fuel_map = _load_emfac_category_fuel_mapping(config)
    override_paths: list[str] = []
    for row in prepared.itertuples(index=False):
        emfac_category = str(getattr(row, "emfacVehicleCategory", "")).strip()
        if emfac_category not in _SPECIAL_CATEGORY_FUEL_DOMAINS:
            override_paths.append(str(getattr(row, "emissionsRatesFile", "")))
            continue
        emfac_fuels = _matched_emfac_fuels(
            fuel_domain=_SPECIAL_CATEGORY_FUEL_DOMAINS[emfac_category],
            emfac_vehicle_category=emfac_category,
            adopt_fuel=getattr(row, "adopt_fuel", ""),
            category_fuel_map=category_fuel_map,
        )
        if not emfac_fuels:
            raise ValueError(
                "No EMFAC fuel mapping available for special fleet-average rate assignment: "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, emfacVehicleCategory={emfac_category}, "
                f"adopt_fuel={getattr(row, 'adopt_fuel', '')}"
            )
        unique_alias_paths = {
            special_rate_aliases[(emfac_category, fuel)]
            for fuel in emfac_fuels
            if (emfac_category, fuel) in special_rate_aliases
        }
        if not unique_alias_paths:
            raise ValueError(
                "No fleet-average EMFAC rates alias is available for special category/fuel assignment: "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, emfacVehicleCategory={emfac_category}, fuels={emfac_fuels}"
            )
        if len(unique_alias_paths) != 1:
            raise ValueError(
                "Ambiguous fleet-average EMFAC rates aliases for special category/fuel assignment: "
                f"vehicleTypeId={getattr(row, 'vehicleTypeId', '')}, emfacVehicleCategory={emfac_category}, fuels={emfac_fuels}"
            )
        override_paths.append(next(iter(unique_alias_paths)))
    prepared["emissionsRatesFile"] = override_paths
    return prepared


def _load_idle_time_fraction_lookup(metadata_path: str) -> dict[str, float]:
    metadata = pd.read_csv(Path(resolve_workflow_path(metadata_path)))
    required = {"emfac_vehicle_category", "idle_time_fraction"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(
            "Vehicle category metadata file is missing required columns for Fleet Step 5: "
            f"{missing}"
        )
    prepared = metadata[["emfac_vehicle_category", "idle_time_fraction"]].copy()
    prepared["emfac_vehicle_category"] = prepared["emfac_vehicle_category"].fillna("").astype(str).str.strip()
    prepared["idle_time_fraction"] = pd.to_numeric(prepared["idle_time_fraction"], errors="coerce")
    prepared = prepared.loc[
        prepared["emfac_vehicle_category"].ne("") & prepared["idle_time_fraction"].notna()
    ].copy()
    return (
        prepared.drop_duplicates(subset=["emfac_vehicle_category"], keep="first")
        .set_index("emfac_vehicle_category")["idle_time_fraction"]
        .to_dict()
    )


def _attach_idle_time_fraction(
    vehicle_types: pd.DataFrame,
    *,
    idle_time_fraction_lookup: dict[str, float],
) -> pd.DataFrame:
    prepared = vehicle_types.copy()
    _require_column(prepared, "emfacVehicleCategory", "Vehicle types table")
    categories = prepared["emfacVehicleCategory"].fillna("").astype(str).str.strip()
    prepared["idleTimeFraction"] = categories.map(idle_time_fraction_lookup)
    prepared.loc[categories.ne("") & prepared["idleTimeFraction"].isna(), "idleTimeFraction"] = 0.0
    return prepared


def _build_short_vehicle_type_ids(
    vehicle_types: pd.DataFrame,
    *,
    prefix: str,
    mapping_column: str = "mappingVehicleTypeId",
    source_column: str = "vehicleTypeId",
    hash_hex_length: int = 6,
    shorten_mask: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    prepared = vehicle_types.copy()
    if source_column not in prepared.columns:
        raise ValueError(f"Vehicle types table is missing required column '{source_column}'")

    prepared[mapping_column] = prepared[source_column].astype(str)
    if shorten_mask is None:
        effective_mask = pd.Series(True, index=prepared.index)
    else:
        effective_mask = pd.Series(shorten_mask, index=prepared.index).fillna(False).astype(bool)
    seen_short_to_long: dict[str, str] = {}
    long_to_short: dict[str, str] = {}
    short_ids: list[str] = []
    rewritten_ids: list[str] = []
    for should_shorten, mapping_vehicle_type_id in zip(
        effective_mask.tolist(),
        prepared[mapping_column].tolist(),
        strict=False,
    ):
        mapping_vehicle_type_id = str(mapping_vehicle_type_id)
        if should_shorten:
            digest = hashlib.sha256(mapping_vehicle_type_id.encode("utf-8")).hexdigest()
            rewritten_id = f"{prefix}-{digest[:hash_hex_length]}"
            existing = seen_short_to_long.get(rewritten_id)
            if existing is not None and existing != mapping_vehicle_type_id:
                raise ValueError(
                    "Short vehicleTypeId hash collision detected for "
                    f"{mapping_vehicle_type_id} and {existing} using prefix={prefix} length={hash_hex_length}"
                )
            seen_short_to_long[rewritten_id] = mapping_vehicle_type_id
            short_ids.append(rewritten_id)
        else:
            rewritten_id = mapping_vehicle_type_id
        long_to_short[mapping_vehicle_type_id] = rewritten_id
        rewritten_ids.append(rewritten_id)

    if len(set(short_ids)) != len(short_ids):
        raise ValueError(f"Generated duplicate short vehicleTypeId values for prefix={prefix}")

    prepared[source_column] = rewritten_ids
    return prepared, long_to_short


def _remap_output_vehicle_ids(
    frame: pd.DataFrame,
    *,
    id_mapping: dict[str, str],
    source_column: str = "vehicleTypeId",
) -> pd.DataFrame:
    prepared = frame.copy()
    if source_column not in prepared.columns:
        raise ValueError(f"Output table is missing required column '{source_column}'")
    original_ids = prepared[source_column].astype(str)
    prepared[source_column] = original_ids.map(id_mapping)
    missing = original_ids[prepared[source_column].isna()].drop_duplicates().tolist()
    if missing:
        raise ValueError(
            f"Could not remap {source_column} values to shortened ids:\n" + "\n".join(missing)
        )
    return prepared


def _build_passenger_vehicle_types_table(workflow: dict[str, Any]) -> pd.DataFrame:
    frames = [
        workflow["built_vehicle_types"].copy(),
        workflow["built_passenger_bus_vehicle_types"].copy(),
        workflow["built_passenger_bike_vehicle_types"].copy(),
        workflow["built_passenger_other_vehicle_types"].copy(),
    ]
    passenger_vehicle_types = pd.concat(frames, ignore_index=True)
    duplicate_vehicle_type_ids = passenger_vehicle_types["vehicleTypeId"][
        passenger_vehicle_types["vehicleTypeId"].duplicated()
    ].drop_duplicates()
    if not duplicate_vehicle_type_ids.empty:
        raise ValueError(
            "Passenger Step 5 generated duplicate vehicleTypeId values:\n"
            + "\n".join(duplicate_vehicle_type_ids.astype(str).tolist())
        )
    return passenger_vehicle_types


def _attach_rates_to_vehicle_types(
    *,
    vehicle_types: pd.DataFrame,
    relative_paths: dict[str, str],
    idle_time_fraction_lookup: dict[str, float],
    config: dict[str, Any],
    special_rate_aliases: dict[tuple[str, str], str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    prepared = vehicle_types.copy()
    prepared = _assign_rate_filepaths(prepared, relative_paths)
    prepared = _override_special_category_rate_paths(
        prepared,
        config=config,
        special_rate_aliases=special_rate_aliases,
    )
    prepared = _attach_idle_time_fraction(
        prepared,
        idle_time_fraction_lookup=idle_time_fraction_lookup,
    )
    return prepared


def _is_beam_passenger_car_vehicle_type(frame: pd.DataFrame) -> pd.Series:
    _require_column(frame, "vehicleCategory", "Passenger vehicle types file")
    return frame["vehicleCategory"].astype(str).eq("Car")


def _run_step5_substep_load_rates_store(
    workflow: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = workflow["config"]
    output_root = Path(str(config["output"])).expanduser().resolve()
    passenger_vehicle_types = _build_passenger_vehicle_types_table(workflow)
    freight_vehicle_types = workflow["built_freight_vehicle_types"].copy()
    passenger_emfac_ids = sorted(
        [emfac_id for emfac_id in passenger_vehicle_types.get("emfacId", pd.Series(dtype="string")).fillna("").astype(str).unique().tolist() if emfac_id]
    )
    freight_emfac_ids = sorted(
        [emfac_id for emfac_id in freight_vehicle_types.get("emfacId", pd.Series(dtype="string")).fillna("").astype(str).unique().tolist() if emfac_id]
    )
    shared_emfac_ids = sorted(set(passenger_emfac_ids) | set(freight_emfac_ids))
    activities_output_root = Path(str(config["activities"]["outputs"])).expanduser().resolve()
    frism_year = workflow["config"]["frism"]["year"]
    scenario_name = str(workflow["scenario"])
    store_root = activities_output_root / "emissions" / _build_year_scenario_token(year=frism_year, scenario=scenario_name)
    parquet_root = store_root / "dataset"
    duckdb_path = store_root / "dataset.duckdb"
    relative_paths: dict[str, str] = {}
    for partition_dir in sorted(parquet_root.glob("emfacId=*")):
        if not partition_dir.is_dir():
            continue
        emfac_id = partition_dir.name.removeprefix("emfacId=")
        parquet_path = partition_dir / f"{emfac_id}.parquet"
        if parquet_path.exists():
            relative_paths[emfac_id] = str(parquet_path.relative_to(output_root))
    missing_store_ids: list[str] = []
    for emfac_id in shared_emfac_ids:
        if emfac_id not in relative_paths:
            missing_store_ids.append(emfac_id)
    if missing_store_ids:
        preview = ", ".join(missing_store_ids[:10])
        raise FileNotFoundError(
            "EMFAC rates store is missing parquet partitions for required emfacId values: "
            f"{preview}{' ...' if len(missing_store_ids) > 10 else ''}"
        )
    if not duckdb_path.exists():
        raise FileNotFoundError(f"EMFAC rates store database not found: {duckdb_path}")
    metadata_path = config.get("vehicle_category_attributes_file")
    if not metadata_path:
        raise ValueError("Fleet Step 5 requires vehicle_category_attributes_file in the EMFAC config.")
    shared_rates_store = {
        "store_root": str(store_root),
        "parquet_root": str(parquet_root),
        "duckdb_path": str(duckdb_path),
        "relative_paths": relative_paths,
        "idle_time_fraction_lookup": _load_idle_time_fraction_lookup(str(metadata_path)),
    }
    shared_rates_store["special_rate_aliases"] = _build_special_category_rate_aliases(
        config=config,
        shared_rates_store=shared_rates_store,
    )
    return shared_rates_store, passenger_vehicle_types, freight_vehicle_types


def _run_step5_substep_attach_passenger_rates(
    *,
    workflow: dict[str, Any],
    shared_rates_store: dict[str, str],
    passenger_vehicle_types: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], str]:
    config = workflow["config"]
    output_root = Path(str(config["output"])).expanduser().resolve()
    atlas_year = workflow["config"]["atlas"]["year"]
    scenario_name = str(workflow["scenario"])
    passenger_vehicle_types_with_rates = _attach_rates_to_vehicle_types(
        vehicle_types=passenger_vehicle_types,
        relative_paths=shared_rates_store["relative_paths"],
        idle_time_fraction_lookup=shared_rates_store["idle_time_fraction_lookup"],
        config=config,
        special_rate_aliases=shared_rates_store["special_rate_aliases"],
    )
    passenger_vehicle_types_with_rates, passenger_id_mapping = _build_short_vehicle_type_ids(
        passenger_vehicle_types_with_rates,
        prefix="pax",
        shorten_mask=_is_beam_passenger_car_vehicle_type(passenger_vehicle_types_with_rates),
    )
    passenger_output_file = _write_vehicle_types(
        passenger_vehicle_types_with_rates,
        str(
            output_root
            / _build_vehicle_types_output_filename(
                source_name="atlas",
                year=atlas_year,
                scenario=scenario_name,
            )
        ),
    )
    return passenger_vehicle_types_with_rates, passenger_id_mapping, passenger_output_file


def _run_step5_substep_attach_freight_rates(
    *,
    workflow: dict[str, Any],
    shared_rates_store: dict[str, str],
    freight_vehicle_types: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str], str]:
    config = workflow["config"]
    output_root = Path(str(config["output"])).expanduser().resolve()
    frism_year = workflow["config"]["frism"]["year"]
    scenario_name = str(workflow["scenario"])
    freight_vehicle_types_with_rates = _attach_rates_to_vehicle_types(
        vehicle_types=freight_vehicle_types,
        relative_paths=shared_rates_store["relative_paths"],
        idle_time_fraction_lookup=shared_rates_store["idle_time_fraction_lookup"],
        config=config,
        special_rate_aliases=shared_rates_store["special_rate_aliases"],
    )
    freight_vehicle_types_with_rates, freight_id_mapping = _build_short_vehicle_type_ids(
        freight_vehicle_types_with_rates,
        prefix="ft",
    )
    freight_output_file = _write_vehicle_types(
        freight_vehicle_types_with_rates,
        str(
            output_root
            / _build_vehicle_types_output_filename(
                source_name="frism",
                year=frism_year,
                scenario=scenario_name,
            )
        ),
    )
    return freight_vehicle_types_with_rates, freight_id_mapping, freight_output_file


def _run_step5_substep_write_mapped_entities(
    *,
    workflow: dict[str, Any],
    passenger_id_mapping: dict[str, str],
    freight_id_mapping: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapped_passenger_vehicles = _remap_output_vehicle_ids(
        workflow["mapped_passenger_vehicles"],
        id_mapping=passenger_id_mapping,
    )
    _write_parquet(
        mapped_passenger_vehicles,
        workflow["mapped_passenger_vehicles_file"],
    )

    mapped_freight_carriers = _remap_output_vehicle_ids(
        workflow["mapped_freight_carriers"],
        id_mapping=freight_id_mapping,
    )
    _write_parquet(
        mapped_freight_carriers,
        workflow["mapped_freight_carriers_file"],
    )
    return mapped_passenger_vehicles, mapped_freight_carriers


def run_step5(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 5: attach the EMFAC rates store to passenger and freight vehicle types."""
    shared_rates_store, passenger_vehicle_types, freight_vehicle_types = _run_step5_substep_load_rates_store(workflow)

    print("=== Step 5.1: attach emfac rates to passenger vehicle types ===")
    passenger_vehicle_types_with_rates, passenger_id_mapping, passenger_output_file = _run_step5_substep_attach_passenger_rates(
        workflow=workflow,
        shared_rates_store=shared_rates_store,
        passenger_vehicle_types=passenger_vehicle_types,
    )

    print("=== Step 5.2: attach emfac rates to freight vehicle types ===")
    freight_vehicle_types_with_rates, freight_id_mapping, freight_output_file = _run_step5_substep_attach_freight_rates(
        workflow=workflow,
        shared_rates_store=shared_rates_store,
        freight_vehicle_types=freight_vehicle_types,
    )

    mapped_passenger_vehicles, mapped_freight_carriers = _run_step5_substep_write_mapped_entities(
        workflow=workflow,
        passenger_id_mapping=passenger_id_mapping,
        freight_id_mapping=freight_id_mapping,
    )

    workflow["passenger_vehicle_types_with_rates"] = passenger_vehicle_types_with_rates
    workflow["passenger_vehicle_types_with_rates_file"] = passenger_output_file
    workflow["freight_vehicle_types_with_rates"] = freight_vehicle_types_with_rates
    workflow["freight_vehicle_types_with_rates_file"] = freight_output_file
    workflow["mapped_passenger_vehicles"] = mapped_passenger_vehicles
    workflow["mapped_freight_carriers"] = mapped_freight_carriers
    workflow["emfac_rates_store"] = shared_rates_store
    return workflow
