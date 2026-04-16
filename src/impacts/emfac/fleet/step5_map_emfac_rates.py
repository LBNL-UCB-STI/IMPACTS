"""Fleet Step 5: attach EMFAC rates stores to passenger and freight vehicle types."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shutil
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path


_RATES_STRING_COLUMNS = {
    "county",
    "process",
    "roadCategory",
    "speedMph_timeMin",
}

_BEAM_RATES_COLUMNS = [
    "county",
    "process",
    "speedMph_timeMin",
    "roadCategory",
    "bc_gram",
    "ch4_gram",
    "co_gram",
    "co2_gram",
    "hc_gram",
    "nh3_gram",
    "n2o_gram",
    "nox_gram",
    "pm_gram",
    "pm10_gram",
    "pm25_gram",
    "rog_gram",
    "sox_gram",
    "tog_gram",
]


def _sanitize_emfac_component(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str("" if pd.isna(value) else value).strip()).strip("_")
    return token.replace("_", "")


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


def _build_emfac_id(*, vehicle_category: object, fuel: object, model_year: object) -> str:
    return (
        f"{_sanitize_emfac_component(model_year)}"
        f"{_sanitize_emfac_component(vehicle_category)}"
        f"{_sanitize_emfac_component(fuel)}"
    )


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


def _column_exists(con: duckdb.DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def _build_duckdb_database(*, parquet_root: Path, duckdb_path: Path) -> Path:
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_glob = (parquet_root / "**" / "*.parquet").as_posix()
    con = duckdb.connect(str(duckdb_path))
    try:
        con.execute("DROP TABLE IF EXISTS emfac_rates")
        con.execute(
            """
            CREATE TABLE emfac_rates AS
            SELECT *
            FROM read_parquet(?, hive_partitioning = true, union_by_name = true)
            """,
            [parquet_glob],
        )
        con.execute("CREATE INDEX IF NOT EXISTS emfac_rates_emfac_id_idx ON emfac_rates (emfacId)")
        if _column_exists(con, "emfac_rates", "county"):
            con.execute("CREATE INDEX IF NOT EXISTS emfac_rates_county_idx ON emfac_rates (county)")
        if _column_exists(con, "emfac_rates", "process"):
            con.execute("CREATE INDEX IF NOT EXISTS emfac_rates_process_idx ON emfac_rates (process)")
    finally:
        con.close()
    return duckdb_path


def _write_rates_store(
    *,
    emissions_rates: pd.DataFrame,
    emfac_ids: list[str],
    output_root: Path,
    scenario_token: str,
) -> dict[str, str]:
    store_root = output_root / "emissions" / scenario_token
    for stale_dir in [store_root / "passenger", store_root / "freight"]:
        if stale_dir.exists():
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)
            else:
                stale_dir.unlink()
    parquet_root = store_root / "dataset"
    duckdb_path = store_root / "dataset.duckdb"
    parquet_root.mkdir(parents=True, exist_ok=True)

    rates = emissions_rates[emissions_rates["emfacId"].isin(emfac_ids)].copy()
    for column_name in _RATES_STRING_COLUMNS.intersection(rates.columns):
        rates[column_name] = rates[column_name].astype("string")

    relative_paths: dict[str, str] = {}
    for emfac_id, frame in rates.groupby("emfacId", dropna=False):
        emfac_id_str = str(emfac_id)
        partition_dir = parquet_root / f"emfacId={emfac_id_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / f"{emfac_id_str}.parquet"
        parquet_columns = [column for column in _BEAM_RATES_COLUMNS if column in frame.columns]
        parquet_frame = frame[parquet_columns].reset_index(drop=True)
        table = pa.Table.from_pandas(parquet_frame, preserve_index=False)
        pq.write_table(table, output_path, compression="zstd")
        relative_paths[emfac_id_str] = str(output_path.relative_to(output_root))

    _build_duckdb_database(parquet_root=parquet_root, duckdb_path=duckdb_path)
    return {
        "store_root": str(store_root),
        "parquet_root": str(parquet_root),
        "duckdb_path": str(duckdb_path),
        "relative_paths": relative_paths,
    }


def _load_emfac_rates_with_ids(config: dict[str, Any]) -> pd.DataFrame:
    rates = read_table(config["activities"]["rates_file"], dtype=None)
    for column_name in ["vehicleCategory", "fuel", "modelYear"]:
        if column_name not in rates.columns:
            raise ValueError(f"EMFAC rates file is missing required column '{column_name}'")
    rates = rates.copy()
    rates["emfacId"] = rates.apply(
        lambda row: _build_emfac_id(
            vehicle_category=row["vehicleCategory"],
            fuel=row["fuel"],
            model_year=row["modelYear"],
        ),
        axis=1,
    )
    return rates


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


def run_step5(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 5: build passenger and freight EMFAC rates stores and attach file paths."""
    config = workflow["config"]
    output_root = Path(str(config["output"])).expanduser().resolve()
    emissions_rates = _load_emfac_rates_with_ids(config)
    passenger_vehicle_types = _build_passenger_vehicle_types_table(workflow)
    freight_vehicle_types = workflow["built_freight_vehicle_types"].copy()
    passenger_emfac_ids = sorted(
        [emfac_id for emfac_id in passenger_vehicle_types.get("emfacId", pd.Series(dtype="string")).fillna("").astype(str).unique().tolist() if emfac_id]
    )
    freight_emfac_ids = sorted(
        [emfac_id for emfac_id in freight_vehicle_types.get("emfacId", pd.Series(dtype="string")).fillna("").astype(str).unique().tolist() if emfac_id]
    )
    shared_emfac_ids = sorted(set(passenger_emfac_ids) | set(freight_emfac_ids))
    frism_year = workflow["config"]["frism"]["year"]
    atlas_year = workflow["config"]["atlas"]["year"]
    scenario_name = str(workflow["scenario"])
    shared_rates_store = _write_rates_store(
        emissions_rates=emissions_rates,
        emfac_ids=shared_emfac_ids,
        output_root=output_root,
        scenario_token=_build_year_scenario_token(year=frism_year, scenario=scenario_name),
    )

    print("=== Step 5.1: attach emfac rates to passenger vehicle types ===")
    passenger_vehicle_types_with_rates = _attach_rates_to_vehicle_types(
        vehicle_types=passenger_vehicle_types,
        relative_paths=shared_rates_store["relative_paths"],
    )
    passenger_vehicle_types_with_rates, passenger_id_mapping = _build_short_vehicle_type_ids(
        passenger_vehicle_types_with_rates,
        prefix="pax",
        # Only BEAM passenger car rows use the shortened id form.
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

    print("=== Step 5.2: attach emfac rates to freight vehicle types ===")
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

    workflow["passenger_vehicle_types_with_rates"] = passenger_vehicle_types_with_rates
    workflow["passenger_vehicle_types_with_rates_file"] = passenger_output_file
    workflow["freight_vehicle_types_with_rates"] = freight_vehicle_types_with_rates
    workflow["freight_vehicle_types_with_rates_file"] = freight_output_file
    workflow["mapped_passenger_vehicles"] = mapped_passenger_vehicles
    workflow["mapped_freight_carriers"] = mapped_freight_carriers
    workflow["emfac_rates_store"] = shared_rates_store
    return workflow
