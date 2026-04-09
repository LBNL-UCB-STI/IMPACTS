"""Fleet Step 5: finalize mapped fleet outputs and the EMFAC rates store.

Substeps:
5.1 Resolve output file locations for mapped fleet artifacts.
5.2 Prepare the emissions-rate store output directory.
5.3 Write a DuckDB+Parquet EMFAC rates store and attach parquet paths to vehicle types.
5.4 Persist mapped carriers, passenger vehicles, and vehicle-type tables.
"""

import logging
import os
import os.path
import shutil
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from impacts.fleet.config import BeamClasses
from impacts.fleet.config import resolve_workflow_path
pd.set_option('display.max_columns', 20)

_RATES_STRING_COLUMNS = {
    "scenario",
    "emfacId",
    "county",
    "speed_mph_float_bins",
    "time_minutes_float_bins",
    "road_category",
    "process",
    "source_file",
}


def _resolve_freight_input_file(config, prefix):
    directory = Path(str(config["beam"]["freight_directory"])).expanduser().resolve()
    matches = sorted(directory.glob(f"{prefix}--*"))
    if not matches:
        raise FileNotFoundError(f"No file matching '{prefix}--*' found in {directory}")
    return matches[0]


def _mapped_output_path(path_like):
    source = Path(path_like)
    if source.suffix == ".gz" and source.name.endswith(".csv.gz"):
        stem = source.name[:-7]
        return str(source.with_name(f"{stem}--EM.csv.gz"))
    if source.suffix:
        stem = source.name[: -len(source.suffix)]
        return str(source.with_name(f"{stem}--EM{source.suffix}"))
    return str(source.with_name(f"{source.name}--EM"))


def _write_table(frame, path):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        frame.to_parquet(target, index=False)
        return
    compression = "gzip" if target.name.endswith(".csv.gz") else None
    frame.to_csv(target, index=False, compression=compression)


def _resolve_output_paths(scenario, config):
    output_root = Path(str(config["output"])).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    carriers_out_file = str(output_root / Path(_mapped_output_path(_resolve_freight_input_file(config, "carriers"))).name)
    ft_vehtypes_out_file = str(
        output_root / Path(_mapped_output_path(resolve_workflow_path(config["beam"]["ft_vehicle_types_file"]))).name
    )
    pax_vehtypes_out_file = str(
        output_root / Path(_mapped_output_path(resolve_workflow_path(config["beam"]["pax_vehicle_types_file"]))).name
    )
    vehicles_output = str(
        output_root / Path(_mapped_output_path(resolve_workflow_path(config["beam"]["pax_vehicles_file"]))).name
    )
    emissions_rates_dir = str(output_root / "emissions" / scenario)
    return carriers_out_file, ft_vehtypes_out_file, pax_vehtypes_out_file, vehicles_output, emissions_rates_dir


def _prepare_emissions_rate_directory(emissions_rates_dir):
    try:
        if os.path.exists(emissions_rates_dir):
            shutil.rmtree(emissions_rates_dir)
        os.makedirs(emissions_rates_dir, exist_ok=True)
        logging.info(f"Ready to write new data to the directory {emissions_rates_dir}")
    except Exception as e:
        logging.error(f"Failed to prepare directory {emissions_rates_dir}: {e}")


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


def _write_rates_store(emissions_rates: pd.DataFrame, emissions_rates_dir: str) -> dict[str, str]:
    output_root = Path(emissions_rates_dir).expanduser().resolve()
    parquet_root = output_root / "dataset"
    duckdb_path = output_root / "dataset.duckdb"
    parquet_root.mkdir(parents=True, exist_ok=True)

    rates = emissions_rates.copy()
    rates["emfacId"] = rates["emfacId"].astype("string")
    rates["source_file"] = rates["emfacId"].astype(str) + ".parquet"
    for column in _RATES_STRING_COLUMNS.intersection(rates.columns):
        rates[column] = rates[column].astype("string")

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
        "output_dir": str(output_root),
        "parquet_root": str(parquet_root),
        "duckdb_path": str(duckdb_path),
        "relative_paths": relative_paths,
    }


def _assign_rate_filepaths(vehtypes_with_emfac_id, relative_paths):
    for veh_type_id, emfac_id in vehtypes_with_emfac_id[["vehicleTypeId", "emfacId"]].itertuples(index=False):
        relative_rates_filepath = relative_paths.get(str(emfac_id))
        if relative_rates_filepath:
            vehtypes_with_emfac_id.loc[
                vehtypes_with_emfac_id["vehicleTypeId"] == veh_type_id,
                "emissionsRatesFile",
            ] = relative_rates_filepath


def _write_updated_vehicle_types(
    vehtypes_with_emfac_id,
    other_pax_vehicle_types,
    ft_vehtypes_out_file,
    pax_vehtypes_out_file,
):
    logging.info(f"Writing:\n{ft_vehtypes_out_file}\n{pax_vehtypes_out_file}")

    ft_freight_mask = vehtypes_with_emfac_id['vehicleCategory'].isin(BeamClasses.get_freight_classes())
    updated_ft_vehicle_types = vehtypes_with_emfac_id[ft_freight_mask].copy()
    updated_ft_vehicle_types.drop(['emfacId', 'oldVehicleTypeId', 'vehicleClass'], axis=1, inplace=True)
    updated_ft_vehicle_types.to_csv(ft_vehtypes_out_file, index=False)

    updated_pax_vehicle_types_others = other_pax_vehicle_types.copy()
    updated_pax_vehicle_types_others['emissionsRatesFile'] = ""
    updated_pax_vehicle_types = pd.concat(
        [vehtypes_with_emfac_id[~ft_freight_mask].copy(), other_pax_vehicle_types],
        axis=0,
    )
    updated_pax_vehicle_types.drop(['emfacId', 'oldVehicleTypeId', 'vehicleClass'], axis=1, inplace=True)
    updated_pax_vehicle_types.to_csv(pax_vehtypes_out_file, index=False)


# Step 5.3-5.4: build mapped BEAM outputs and attach emissions files

def run_step5(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 5: write mapped fleet outputs and the EMFAC emissions-rate store."""
    print("\n=== Map EMFAC To BEAM Population ===\n")
    carriers_out_file, ft_vehtypes_out_file, pax_vehtypes_out_file, vehicles_output, emissions_rates_dir = (
        _resolve_output_paths(workflow["scenario"], workflow["config"])
    )

    vehtypes_with_emfac_id = pd.concat(
        [workflow["new_ft_vehicle_types"], workflow["new_pax_vehicle_types"]],
        ignore_index=True,
    )
    vehtypes_with_emfac_id = vehtypes_with_emfac_id.fillna("")
    _prepare_emissions_rate_directory(emissions_rates_dir)
    rates_store = _write_rates_store(workflow["emfac_rates"], emissions_rates_dir)
    _assign_rate_filepaths(vehtypes_with_emfac_id, rates_store["relative_paths"])
    _write_updated_vehicle_types(
        vehtypes_with_emfac_id,
        workflow["other_pax_vehicle_types"],
        ft_vehtypes_out_file,
        pax_vehtypes_out_file,
    )
    _write_table(workflow["new_carriers"], carriers_out_file)
    _write_table(workflow["pax_vehicles"], vehicles_output)
    workflow["fleet_output_paths"] = {
        "carriers": carriers_out_file,
        "ft_vehicle_types": ft_vehtypes_out_file,
        "pax_vehicle_types": pax_vehtypes_out_file,
        "pax_vehicles": vehicles_output,
        "emissions_rates_dir": emissions_rates_dir,
        "emissions_rates_parquet_root": rates_store["parquet_root"],
        "emissions_rates_duckdb": rates_store["duckdb_path"],
    }
    return workflow

def print_unmapped(df, mapped_col, col_to_be_mapped):
    unmapped_classes = df[df[mapped_col].isna()][col_to_be_mapped].unique()
    if len(unmapped_classes) > 0:
        unmapped_classes_message = f"The following {col_to_be_mapped} were not mapped to {mapped_col}:\n"
        formatted_list = ""
        current_line = ""
        for vehicle_class in unmapped_classes:
            # Check if adding this class would exceed the line limit
            if len(current_line + vehicle_class) > 115:  # 115 to leave room for comma and space
                formatted_list += current_line.rstrip(", ") + "\n"
                current_line = vehicle_class + ", "
            else:
                current_line += vehicle_class + ", "
        # Add the last line
        if current_line:
            formatted_list += current_line.rstrip(", ")
        print(f"{unmapped_classes_message}{formatted_list}")
