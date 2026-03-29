from __future__ import annotations

import concurrent.futures
import gzip
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
EXPLICIT_SKIMS_POLLUTANTS = [
    "CH4",
    "CO",
    "CO2",
    "HC",
    "NH3",
    "NOx",
    "PM",
    "PM10",
    "PM2_5",
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


def _available_compact_workers() -> int:
    return max(1, int(os.cpu_count() or 1))


class ProgressReporter:
    def __init__(self, label: str, total_bytes: int):
        self.label = label
        self.total_bytes = max(0, total_bytes)
        self.start_time = time.time()
        self.last_report = 0.0

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
        print(msg, file=sys.stderr, flush=True)


def _iter_csv_chunks(
    path: str | Path,
    *,
    chunksize: int,
    usecols=None,
    engine: Optional[str] = None,
    label: str,
):
    source = Path(path)
    total_bytes = source.stat().st_size
    reporter = ProgressReporter(label=label, total_bytes=total_bytes)

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
                    reporter.report(raw_handle.tell(), extra=f"chunk_rows {len(chunk)}")
                    yield chunk
                reporter.report(total_bytes, extra="done")
        return

    with source.open("r", newline="") as text_handle:
        reader = pd.read_csv(
            text_handle,
            usecols=usecols,
            chunksize=chunksize,
            engine=engine,
        )
        for chunk in reader:
            reporter.report(text_handle.tell(), extra=f"chunk_rows {len(chunk)}")
            yield chunk
        reporter.report(total_bytes, extra="done")


def _compression_for_path(path: str | Path) -> Optional[str]:
    return "gzip" if str(path).lower().endswith(".gz") else None


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


def _pollutant_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in EXPLICIT_SKIMS_POLLUTANTS if col in df.columns]


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
    if "emissionsProcess" in df.columns and "process" not in df.columns:
        df = df.rename(columns={"emissionsProcess": "process"})
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
    population_sample: float = 1.0,
) -> dict:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in the interval (0, 1].")
    if not 0 < population_sample <= 1:
        raise ValueError("population_sample must be in the interval (0, 1].")

    compact_workers = max(1, int(compact_workers or _available_compact_workers()))
    rng = np.random.default_rng(seed)
    total_rows = 0
    kept_rows = 0
    first_chunk = True
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    for chunk in _iter_csv_chunks(
        input_path,
        chunksize=chunk_size,
        label="sample_skims",
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
        empty = pd.read_csv(input_path, compression=_compression_for_path(input_path), nrows=0)
        empty = _normalize_sampled_skims(
            empty,
            compact_workers=compact_workers,
        )
        _write_chunk(empty, output, first_chunk=True)

    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(output.resolve()),
        "fraction": fraction,
        "population_sample": population_sample,
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
    unique_vehicles = set()
    total_rows = 0
    for chunk in _iter_csv_chunks(
        input_path,
        usecols=[vehicle_column],
        chunksize=chunk_size,
        engine="python",
        label="sample_events phase=scan_vehicles",
    ):
        if vehicle_column not in chunk.columns:
            raise ValueError(f"Column `{vehicle_column}` not found in events file.")
        series = chunk[vehicle_column].dropna().astype(str)
        unique_vehicles.update(series.tolist())

    vehicles = np.array(list(unique_vehicles), dtype=object)
    if len(vehicles) == 0:
        raise ValueError("No vehicle ids found in events file.")

    rng = np.random.default_rng(seed)
    target_count = max(1, int(np.ceil(len(vehicles) * fraction)))
    selected = set(rng.choice(vehicles, size=target_count, replace=False).tolist())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    first_chunk = True
    kept_rows = 0
    for chunk in _iter_csv_chunks(
        input_path,
        chunksize=chunk_size,
        engine="python",
        label="sample_events phase=filter_rows",
    ):
        total_rows += len(chunk)
        sampled = chunk[chunk[vehicle_column].astype(str).isin(selected)].copy()
        kept_rows += len(sampled)
        if sampled.empty:
            continue
        _write_chunk(sampled, output, first_chunk=first_chunk)
        first_chunk = False

    if first_chunk:
        empty = pd.read_csv(input_path, compression=compression, nrows=0)
        _write_chunk(empty, output, first_chunk=True)

    return {
        "input_path": str(Path(input_path).resolve()),
        "output_path": str(output.resolve()),
        "fraction": fraction,
        "seed": seed,
        "vehicle_column": vehicle_column,
        "total_rows": total_rows,
        "unique_vehicles": int(len(vehicles)),
        "selected_vehicles": int(len(selected)),
        "kept_rows": kept_rows,
    }
