from __future__ import annotations

import concurrent.futures
import gzip
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
EXPLICIT_SKIMS_POLLUTANTS = [
    "CH4",
    "CO",
    "CO2",
    "HC",
    "NH3",
    "NOx",
    "PM",
    "PM10",
    "PM25",
    "ROG",
    "SOx",
    "TOG",
]
COMPACT_SKIMS_COLUMNS = [
    "hour",
    "linkId",
    "vehicleTypeId",
    "process",
    "emissions",
    "travelTimeInSecond",
    "parkingDurationInSecond",
    "observations",
    "iterations",
]
SAMPLING_CHUNK_SIZE = 200_000

SKIMS_PARQUET_TYPES = {
    "hour": "BIGINT",
    "linkId": "BIGINT",
    "vehicleTypeId": "VARCHAR",
    "process": "VARCHAR",
    "emissions": "VARCHAR",
    "travelTimeInSecond": "DOUBLE",
    "parkingDurationInSecond": "DOUBLE",
    "observations": "BIGINT",
    "iterations": "BIGINT",
}

EVENTS_PARQUET_TYPES = {
    "person": "VARCHAR",
    "vehicle": "VARCHAR",
    "time": "DOUBLE",
    "type": "VARCHAR",
    "x": "DOUBLE",
    "y": "DOUBLE",
    "shiftStatus": "VARCHAR",
    "parkingTaz": "VARCHAR",
    "chargingPointType": "VARCHAR",
    "pricingModel": "VARCHAR",
    "parkingType": "VARCHAR",
    "locationY": "DOUBLE",
    "locationX": "DOUBLE",
    "parkingZoneId": "BIGINT",
    "price": "DOUBLE",
    "fuel": "DOUBLE",
    "duration": "DOUBLE",
    "vehicleType": "VARCHAR",
    "actType": "VARCHAR",
    "secondaryFuelLevel": "DOUBLE",
    "primaryFuelLevel": "DOUBLE",
    "link": "BIGINT",
    "facility": "VARCHAR",
    "legMode": "VARCHAR",
    "tripId": "VARCHAR",
    "departTime": "DOUBLE",
    "startX": "DOUBLE",
    "startY": "DOUBLE",
    "endX": "DOUBLE",
    "endY": "DOUBLE",
    "requireWheelchair": "VARCHAR",
    "reason": "VARCHAR",
    "emissions": "VARCHAR",
    "score": "DOUBLE",
    "cost": "DOUBLE",
    "driver": "VARCHAR",
    "mode": "VARCHAR",
    "incentive": "DOUBLE",
    "tollCost": "DOUBLE",
    "netCost": "DOUBLE",
    "currentTourMode": "VARCHAR",
    "arrivalTime": "DOUBLE",
    "departureTime": "DOUBLE",
    "capacity": "BIGINT",
    "linkTravelTime": "VARCHAR",
    "secondaryFuel": "DOUBLE",
    "secondaryFuelType": "VARCHAR",
    "primaryFuelType": "VARCHAR",
    "riders": "VARCHAR",
    "toStopIndex": "BIGINT",
    "fromStopIndex": "BIGINT",
    "tollPaid": "DOUBLE",
    "seatingCapacity": "BIGINT",
    "links": "VARCHAR",
    "numPassengers": "BIGINT",
    "length": "DOUBLE",
    "primaryFuel": "DOUBLE",
    "expectedMaximumUtility": "DOUBLE",
    "availableAlternatives": "VARCHAR",
    "location": "VARCHAR",
    "personalVehicleAvailable": "VARCHAR",
    "tourIndex": "BIGINT",
    "legModes": "VARCHAR",
    "legVehicleIds": "VARCHAR",
    "currentActivity": "VARCHAR",
    "nextActivity": "VARCHAR",
}


def _available_compact_workers() -> int:
    return max(1, int(os.cpu_count() or 1))


class ProgressReporter:
    def __init__(self, label: str, total_bytes: int):
        self.label = label
        self.total_bytes = max(0, total_bytes)
        self.start_time = time.time()
        self.last_report = 0.0
        self.closed = False

    def report(self, processed_bytes: int, extra: str = "") -> None:
        now = time.time()
        if now - self.last_report < 1.0 and processed_bytes < self.total_bytes:
            return
        self.last_report = now
        elapsed = now - self.start_time
        pct = 0.0
        eta_txt = "eta unknown"
        if self.total_bytes > 0:
            pct = min(100.0, 100.0 * processed_bytes / self.total_bytes)
            if processed_bytes > 0 and elapsed > 0:
                remaining = max(0.0, elapsed * (self.total_bytes - processed_bytes) / processed_bytes)
                eta_txt = f"eta {remaining:.1f}s"
        msg = (
            f"[{self.label}] {pct:5.1f}% "
            f"elapsed {elapsed:.1f}s "
            f"{eta_txt}"
        )
        if extra:
            msg = f"{msg} {extra}"
        print(f"\r{msg}", file=sys.stderr, end="", flush=True)

    def close(self, extra: str = "done") -> None:
        if self.closed:
            return
        self.report(self.total_bytes, extra=extra)
        print(file=sys.stderr, flush=True)
        self.closed = True


def _iter_csv_chunks(
    path: str | Path,
    *,
    chunksize: int,
    usecols=None,
    engine: Optional[str] = None,
    label: str,
    reporter: Optional[ProgressReporter] = None,
    progress_offset: int = 0,
):
    source = Path(path)
    total_bytes = source.stat().st_size
    local_reporter = reporter or ProgressReporter(label=label, total_bytes=total_bytes)
    lower = str(path).lower()

    if lower.endswith(".parquet"):
        parquet_file = pq.ParquetFile(source)
        total_rows = parquet_file.metadata.num_rows
        if total_rows == 0:
            local_reporter.report(progress_offset + total_bytes, extra="chunk_rows 0")
            if reporter is None:
                local_reporter.close()
            return
        processed_rows = 0
        for batch in parquet_file.iter_batches(batch_size=chunksize, columns=usecols):
            chunk = batch.to_pandas()
            processed_rows += len(chunk)
            processed = int(total_bytes * (processed_rows / total_rows))
            local_reporter.report(progress_offset + processed, extra=f"chunk_rows {len(chunk)}")
            yield chunk
        if reporter is None:
            local_reporter.close()
        return

    if _compression_for_path(path) == "gzip":
        with source.open("rb") as raw_handle:
            with gzip.open(raw_handle, mode="rt", newline="") as text_handle:
                reader = pd.read_csv(
                    text_handle,
                    usecols=usecols,
                    chunksize=chunksize,
                    engine=engine,
                )
                for chunk in reader:
                    local_reporter.report(progress_offset + raw_handle.tell(), extra=f"chunk_rows {len(chunk)}")
                    yield chunk
                if reporter is None:
                    local_reporter.close()
        return

    with source.open("r", newline="") as text_handle:
        reader = pd.read_csv(
            text_handle,
            usecols=usecols,
            chunksize=chunksize,
            engine=engine,
        )
        for chunk in reader:
            local_reporter.report(progress_offset + text_handle.tell(), extra=f"chunk_rows {len(chunk)}")
            yield chunk
        if reporter is None:
            local_reporter.close()


def _compression_for_path(path: str | Path) -> Optional[str]:
    return "gzip" if str(path).lower().endswith(".gz") else None


def _read_empty_like(path: str | Path) -> pd.DataFrame:
    lower = str(path).lower()
    if lower.endswith(".parquet"):
        parquet_file = pq.ParquetFile(path)
        return pd.DataFrame(columns=list(parquet_file.schema_arrow.names))
    return pd.read_csv(path, compression=_compression_for_path(path), nrows=0)


def _write_chunk(df: pd.DataFrame, output_path: str | Path, first_chunk: bool) -> None:
    compression = _compression_for_path(output_path)
    mode = "w" if first_chunk else "a"
    df.to_csv(
        output_path,
        index=False,
        mode=mode,
        header=first_chunk,
        compression=compression,
    )


def _convert_csv_to_parquet_with_schema(
    input_path: str | Path,
    output_path: str | Path,
    *,
    schema: dict[str, str],
    reporter: Optional[ProgressReporter] = None,
    progress_offset: int = 0,
) -> None:
    arrow_type_map = {
        "VARCHAR": pa.string(),
        "DOUBLE": pa.float64(),
        "BIGINT": pa.int64(),
    }
    arrow_schema = pa.schema([(column, arrow_type_map[dtype]) for column, dtype in schema.items()])
    writer: Optional[pq.ParquetWriter] = None
    target = Path(output_path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    try:
        for chunk in _iter_csv_chunks(
            input_path,
            chunksize=SAMPLING_CHUNK_SIZE,
            label=f"convert_parquet {target.name}",
            reporter=reporter,
            progress_offset=progress_offset,
        ):
            normalized = pd.DataFrame(index=chunk.index)
            for column, dtype in schema.items():
                source_series = chunk[column] if column in chunk.columns else pd.Series(pd.NA, index=chunk.index)
                if dtype == "VARCHAR":
                    normalized[column] = source_series.astype("string")
                elif dtype == "DOUBLE":
                    normalized[column] = pd.to_numeric(source_series, errors="coerce").astype("Float64")
                elif dtype == "BIGINT":
                    normalized[column] = pd.to_numeric(source_series, errors="coerce").astype("Int64")
                else:
                    raise ValueError(f"Unsupported parquet conversion dtype: {dtype}")

            table = pa.Table.from_pandas(normalized, schema=arrow_schema, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(target, arrow_schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _sample_output_target(output_path: str | Path, *, kind: str) -> tuple[Path, Optional[Path], Optional[tempfile.TemporaryDirectory]]:
    output = Path(output_path)
    if output.suffix.lower() != ".parquet":
        return output, None, None

    temp_dir = tempfile.TemporaryDirectory(prefix=f"sample_{kind}_")
    temp_output = Path(temp_dir.name) / f"sampled_{kind}.csv.gz"
    return temp_output, output, temp_dir


def _serialize_explicit_pollutants(row: pd.Series) -> str:
    parts = []
    for pollutant in EXPLICIT_SKIMS_POLLUTANTS:
        if pollutant not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[pollutant]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        value = float(value)
        if value != 0.0:
            parts.append(f"{pollutant}:{value}")
    return ";".join(parts)


def _is_explicit_skims_schema(df: pd.DataFrame) -> bool:
    return "emissionsProcess" in df.columns and any(col in df.columns for col in EXPLICIT_SKIMS_POLLUTANTS)


def _compact_explicit_skims_partition(df: pd.DataFrame) -> pd.DataFrame:
    compact = pd.DataFrame(index=df.index)
    compact["hour"] = df["hour"] if "hour" in df.columns else 0
    compact["linkId"] = df["linkId"] if "linkId" in df.columns else pd.NA
    compact["vehicleTypeId"] = df["vehicleTypeId"] if "vehicleTypeId" in df.columns else pd.NA
    compact["process"] = df["emissionsProcess"]
    compact["emissions"] = df.apply(_serialize_explicit_pollutants, axis=1)
    compact["travelTimeInSecond"] = 0.0
    compact["parkingDurationInSecond"] = 0.0
    compact["observations"] = df["observations"] if "observations" in df.columns else 0
    compact["iterations"] = df["iterations"] if "iterations" in df.columns else 0
    return compact[COMPACT_SKIMS_COLUMNS]


def _compact_explicit_skims_parallel(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    if df.empty:
        return _compact_explicit_skims_partition(df)
    workers = max(1, min(workers, len(df)))
    if workers == 1 or len(df) < 10_000:
        return _compact_explicit_skims_partition(df)

    partition_size = math.ceil(len(df) / workers)
    partitions = [df.iloc[i : i + partition_size].copy() for i in range(0, len(df), partition_size)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        compacted = list(executor.map(_compact_explicit_skims_partition, partitions))
    return pd.concat(compacted, ignore_index=True)


def _compact_skims_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "emissions" in df.columns and "process" in df.columns:
        compact = df.copy()
        for col in COMPACT_SKIMS_COLUMNS:
            if col not in compact.columns:
                if col in {"travelTimeInSecond", "parkingDurationInSecond"}:
                    compact[col] = 0.0
                elif col in {"observations", "iterations"}:
                    compact[col] = 0
                else:
                    compact[col] = pd.NA
        return compact[COMPACT_SKIMS_COLUMNS]

    return df


def _normalize_sampled_skims(
    sampled: pd.DataFrame,
    *,
    compact_workers: int,
) -> pd.DataFrame:
    if sampled.empty:
        return pd.DataFrame(columns=COMPACT_SKIMS_COLUMNS)
    if _is_explicit_skims_schema(sampled):
        return _compact_explicit_skims_parallel(sampled, compact_workers)
    return _compact_skims_if_needed(sampled)


def sample_skims_by_fraction(
    input_path: str | Path,
    output_path: str | Path,
    fraction: float,
    seed: int = 42,
    chunk_size: int = SAMPLING_CHUNK_SIZE,
    compact_workers: Optional[int] = None,
) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1].")

    compact_workers = max(1, int(compact_workers or _available_compact_workers()))
    rng = np.random.default_rng(seed)
    total_rows = 0
    kept_rows = 0
    first_chunk = True
    write_target, parquet_target, temp_dir = _sample_output_target(output_path, kind="skims")
    output = Path(write_target)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    sample_reporter = ProgressReporter(label="sample_skims", total_bytes=Path(input_path).stat().st_size)
    for chunk in _iter_csv_chunks(
        input_path,
        chunksize=chunk_size,
        label="sample_skims",
        reporter=sample_reporter,
    ):
        total_rows += len(chunk)
        mask = rng.random(len(chunk)) < fraction
        sampled = chunk.loc[mask].copy()
        sampled = _normalize_sampled_skims(
            sampled,
            compact_workers=compact_workers,
        )
        kept_rows += len(sampled)
        if sampled.empty:
            continue
        _write_chunk(sampled, output, first_chunk=first_chunk)
        first_chunk = False

    if first_chunk:
        empty = _read_empty_like(input_path)
        empty = _normalize_sampled_skims(
            empty,
            compact_workers=compact_workers,
        )
        _write_chunk(empty, output, first_chunk=True)
    sample_reporter.close()

    final_output = Path(output_path).resolve()
    if parquet_target is not None:
        final_output.parent.mkdir(parents=True, exist_ok=True)
        if final_output.exists():
            final_output.unlink()
        convert_reporter = ProgressReporter(label="convert_parquet sample_skims", total_bytes=output.stat().st_size)
        _convert_csv_to_parquet_with_schema(
            output,
            final_output,
            schema=SKIMS_PARQUET_TYPES,
            reporter=convert_reporter,
        )
        convert_reporter.close()
        temp_dir.cleanup()

    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(final_output),
        "fraction": fraction,
        "seed": seed,
        "total_rows": total_rows,
        "kept_rows": kept_rows,
    }


def sample_events_by_vehicle(
    input_path: str | Path,
    output_path: str | Path,
    fraction: float,
    seed: int = 42,
    vehicle_column: str = "vehicle",
    chunk_size: int = SAMPLING_CHUNK_SIZE,
) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1].")

    compression = _compression_for_path(input_path)
    write_target, parquet_target, temp_dir = _sample_output_target(output_path, kind="events")
    input_size = Path(input_path).stat().st_size
    scan_reporter = ProgressReporter(label="sample_events scan_vehicles", total_bytes=input_size)
    unique_vehicles = set()
    total_rows = 0
    for chunk in _iter_csv_chunks(
        input_path,
        usecols=[vehicle_column],
        chunksize=chunk_size,
        engine="python",
        label="sample_events",
        reporter=scan_reporter,
    ):
        if vehicle_column not in chunk.columns:
            raise ValueError(f"Column `{vehicle_column}` not found in events file.")
        series = chunk[vehicle_column].dropna().astype(str)
        unique_vehicles.update(series.tolist())
    scan_reporter.close()

    vehicles = np.array(list(unique_vehicles), dtype=object)
    if len(vehicles) == 0:
        raise ValueError("No vehicle ids found in events file.")

    rng = np.random.default_rng(seed)
    target_count = max(1, int(np.ceil(len(vehicles) * fraction)))
    selected = set(rng.choice(vehicles, size=target_count, replace=False).tolist())

    output = Path(write_target)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    first_chunk = True
    kept_rows = 0
    filter_reporter = ProgressReporter(label="sample_events filter_rows", total_bytes=input_size)
    for chunk in _iter_csv_chunks(
        input_path,
        chunksize=chunk_size,
        engine="python",
        label="sample_events",
        reporter=filter_reporter,
    ):
        total_rows += len(chunk)
        sampled = chunk[chunk[vehicle_column].astype(str).isin(selected)].copy()
        kept_rows += len(sampled)
        if sampled.empty:
            continue
        _write_chunk(sampled, output, first_chunk=first_chunk)
        first_chunk = False

    if first_chunk:
        empty = _read_empty_like(input_path)
        _write_chunk(empty, output, first_chunk=True)
    filter_reporter.close()

    final_output = Path(output_path).resolve()
    if parquet_target is not None:
        final_output.parent.mkdir(parents=True, exist_ok=True)
        if final_output.exists():
            final_output.unlink()
        convert_reporter = ProgressReporter(label="convert_parquet sample_events", total_bytes=output.stat().st_size)
        _convert_csv_to_parquet_with_schema(
            output,
            final_output,
            schema=EVENTS_PARQUET_TYPES,
            reporter=convert_reporter,
        )
        convert_reporter.close()
        temp_dir.cleanup()

    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(final_output),
        "fraction": fraction,
        "seed": seed,
        "vehicle_column": vehicle_column,
        "total_rows": total_rows,
        "unique_vehicles": int(len(vehicles)),
        "selected_vehicles": int(len(selected)),
        "kept_rows": kept_rows,
    }
