"""Fleet Step 5: attach EMFAC rates stores to passenger and freight vehicle types."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

import pandas as pd

from impacts.emfac.config import resolve_workflow_path


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
) -> tuple[pd.DataFrame, dict[str, str]]:
    prepared = vehicle_types.copy()
    prepared = _assign_rate_filepaths(prepared, relative_paths)
    return prepared


def _is_beam_passenger_car_vehicle_type(frame: pd.DataFrame) -> pd.Series:
    _require_column(frame, "vehicleCategory", "Passenger vehicle types file")
    return frame["vehicleCategory"].astype(str).eq("Car")


def _run_step5_substep_load_rates_store(
    workflow: dict[str, Any],
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
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
    missing_store_ids: list[str] = []
    for emfac_id in shared_emfac_ids:
        parquet_path = parquet_root / f"emfacId={emfac_id}" / f"{emfac_id}.parquet"
        if parquet_path.exists():
            relative_paths[emfac_id] = str(parquet_path.relative_to(output_root))
        else:
            missing_store_ids.append(emfac_id)
    if missing_store_ids:
        preview = ", ".join(missing_store_ids[:10])
        raise FileNotFoundError(
            "EMFAC rates store is missing parquet partitions for required emfacId values: "
            f"{preview}{' ...' if len(missing_store_ids) > 10 else ''}"
        )
    if not duckdb_path.exists():
        raise FileNotFoundError(f"EMFAC rates store database not found: {duckdb_path}")
    shared_rates_store = {
        "store_root": str(store_root),
        "parquet_root": str(parquet_root),
        "duckdb_path": str(duckdb_path),
        "relative_paths": relative_paths,
    }
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
