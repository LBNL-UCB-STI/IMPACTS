from __future__ import annotations

import argparse
import logging
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger(__name__)

_STRING_COLUMNS = {
    "scenario",
    "emfacId",
    "county",
    "speed_mph_float_bins",
    "time_minutes_float_bins",
    "road_category",
    "process",
    "source_file",
}


def _iter_rate_csvs(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    files = sorted(p for p in root.rglob("*.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {root}")
    LOGGER.info("Found %s EMFAC rate CSV files under %s", len(files), root)
    return files


def _read_rate_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    emfac_id = path.stem

    if "emfacId" not in frame.columns:
        frame["emfacId"] = emfac_id
    else:
        frame["emfacId"] = frame["emfacId"].fillna(emfac_id)

    if "scenario" not in frame.columns:
        frame["scenario"] = path.parent.name

    frame["source_file"] = path.name
    for column in _STRING_COLUMNS.intersection(frame.columns):
        frame[column] = frame[column].astype("string")
    return frame


def _write_partitioned_parquet(
    csv_paths: list[Path],
    *,
    parquet_root: str | Path,
    compression: str,
) -> list[Path]:
    output_root = Path(parquet_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    total = len(csv_paths)
    for index, csv_path in enumerate(csv_paths, start=1):
        frame = _read_rate_csv(csv_path)
        emfac_id = str(frame["emfacId"].iloc[0])
        partition_dir = output_root / f"emfacId={emfac_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        output_path = partition_dir / f"{csv_path.stem}.parquet"

        parquet_frame = frame.drop(columns=["emfacId"], errors="ignore")
        table = pa.Table.from_pandas(parquet_frame, preserve_index=False)
        pq.write_table(table, output_path, compression=compression)
        written.append(output_path)
        if index == 1 or index == total or index % 25 == 0:
            LOGGER.info("Parquet conversion progress: %s/%s files written", index, total)

    return written


def _build_duckdb_database(
    *,
    parquet_root: str | Path,
    duckdb_path: str | Path,
) -> Path:
    db_path = Path(duckdb_path).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_glob = (Path(parquet_root).resolve() / "**" / "*.parquet").as_posix()

    LOGGER.info("Building DuckDB database at %s from Parquet dataset %s", db_path, Path(parquet_root).resolve())
    con = duckdb.connect(str(db_path))
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

    return db_path


def _column_exists(con: duckdb.DuckDBPyConnection, table_name: str, column_name: str) -> bool:
    rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return any(str(row[1]) == column_name for row in rows)


def build_emfac_rates_store(
    *,
    input_dir: str,
    output_dir: str,
    compression: str = "zstd",
) -> dict[str, object]:
    output_root = Path(output_dir).resolve()
    parquet_root = output_root / "dataset"
    duckdb_path = output_root / "dataset.duckdb"

    LOGGER.info("Starting EMFAC rates store build")
    LOGGER.info("Input directory: %s", Path(input_dir).resolve())
    LOGGER.info("Output directory: %s", output_root)
    LOGGER.info("Parquet compression: %s", compression)
    csv_paths = _iter_rate_csvs(input_dir)
    written_parquet = _write_partitioned_parquet(
        csv_paths,
        parquet_root=parquet_root,
        compression=compression,
    )
    built_duckdb = _build_duckdb_database(
        parquet_root=parquet_root,
        duckdb_path=duckdb_path,
    )
    result = {
        "input_file_count": len(csv_paths),
        "parquet_file_count": len(written_parquet),
        "output_dir": str(output_root),
        "parquet_root": str(parquet_root),
        "duckdb_path": str(built_duckdb),
    }
    LOGGER.info(
        "Completed EMFAC rates store build: files=%s parquet_root=%s duckdb=%s",
        result["parquet_file_count"],
        result["parquet_root"],
        result["duckdb_path"],
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.emfac.build_emfac_rates_store",
        description="Convert an EMFAC rates folder into a partitioned Parquet dataset and a DuckDB database.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing EMFAC rate CSV files.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory that will contain dataset/ and dataset.duckdb.",
    )
    parser.add_argument("--compression", default="zstd", help="Parquet compression codec. Default: zstd.")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    result = build_emfac_rates_store(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        compression=args.compression,
    )
    print(f"input_file_count={result['input_file_count']}")
    print(f"parquet_file_count={result['parquet_file_count']}")
    print(f"output_dir={result['output_dir']}")
    print(f"parquet_root={result['parquet_root']}")
    print(f"duckdb_path={result['duckdb_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
