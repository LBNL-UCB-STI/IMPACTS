"""Fleet Step 5: attach EMFAC rates stores to passenger and freight vehicle types."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from impacts.fleet.config import read_table
from impacts.fleet.config import resolve_workflow_path


_RATES_STRING_COLUMNS = {
    "county",
    "emfacId",
    "process",
    "roadCategory",
    "source_file",
    "speedMph_timeMin",
    "vehicleCategory",
    "fuel",
    "modelYear",
}


def _sanitize_emfac_component(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str("" if pd.isna(value) else value).strip()).strip("_")
    return token.replace("_", "")


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
    store_name: str,
) -> dict[str, str]:
    store_root = output_root / "emissions" / store_name
    parquet_root = store_root / "dataset"
    duckdb_path = store_root / "dataset.duckdb"
    parquet_root.mkdir(parents=True, exist_ok=True)

    rates = emissions_rates[emissions_rates["emfacId"].isin(emfac_ids)].copy()
    rates["source_file"] = rates["emfacId"].astype(str) + ".parquet"
    for column_name in _RATES_STRING_COLUMNS.intersection(rates.columns):
        rates[column_name] = rates[column_name].astype("string")

    relative_paths: dict[str, str] = {}
    for emfac_id, frame in rates.groupby("emfacId", dropna=False):
        emfac_id_str = str(emfac_id)
        partition_dir = parquet_root / f"emfacId={emfac_id_str}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / f"{emfac_id_str}.parquet"
        parquet_frame = frame.drop(columns=["emfacId"], errors="ignore").reset_index(drop=True)
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
    rates = read_table(config["emfac"]["rates_file"], dtype=None)
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
    emissions_rates: pd.DataFrame,
    output_root: Path,
    store_name: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    prepared = vehicle_types.copy()
    emfac_ids = sorted(prepared.get("emfacId", pd.Series(dtype="string")).fillna("").astype(str).unique().tolist())
    emfac_ids = [emfac_id for emfac_id in emfac_ids if emfac_id]
    if emfac_ids:
        rates_store = _write_rates_store(
            emissions_rates=emissions_rates,
            emfac_ids=emfac_ids,
            output_root=output_root,
            store_name=store_name,
        )
        prepared = _assign_rate_filepaths(prepared, rates_store["relative_paths"])
    else:
        rates_store = {
            "store_root": str(output_root / "emissions" / store_name),
            "parquet_root": str(output_root / "emissions" / store_name / "dataset"),
            "duckdb_path": str(output_root / "emissions" / store_name / "dataset.duckdb"),
            "relative_paths": {},
        }
        prepared["emissionsRatesFile"] = ""
    return prepared, rates_store


def run_step5(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 5: build passenger and freight EMFAC rates stores and attach file paths."""
    config = workflow["config"]
    output_root = Path(str(config["output"])).expanduser().resolve()
    emissions_rates = _load_emfac_rates_with_ids(config)

    print("=== Step 5.1: attach emfac rates to passenger vehicle types ===")
    passenger_vehicle_types = _build_passenger_vehicle_types_table(workflow)
    passenger_vehicle_types_with_rates, passenger_rates_store = _attach_rates_to_vehicle_types(
        vehicle_types=passenger_vehicle_types,
        emissions_rates=emissions_rates,
        output_root=output_root,
        store_name="passenger",
    )
    passenger_output_file = _write_vehicle_types(
        passenger_vehicle_types_with_rates,
        str(output_root / "vehicleTypes--passenger.csv"),
    )

    print("=== Step 5.2: attach emfac rates to freight vehicle types ===")
    freight_vehicle_types = workflow["built_freight_vehicle_types"].copy()
    freight_vehicle_types_with_rates, freight_rates_store = _attach_rates_to_vehicle_types(
        vehicle_types=freight_vehicle_types,
        emissions_rates=emissions_rates,
        output_root=output_root,
        store_name="freight",
    )
    freight_output_file = _write_vehicle_types(
        freight_vehicle_types_with_rates,
        str(output_root / "vehicleTypes--freight.csv"),
    )

    workflow["passenger_vehicle_types_with_rates"] = passenger_vehicle_types_with_rates
    workflow["passenger_vehicle_types_with_rates_file"] = passenger_output_file
    workflow["freight_vehicle_types_with_rates"] = freight_vehicle_types_with_rates
    workflow["freight_vehicle_types_with_rates_file"] = freight_output_file
    workflow["passenger_rates_store"] = passenger_rates_store
    workflow["freight_rates_store"] = freight_rates_store
    return workflow
