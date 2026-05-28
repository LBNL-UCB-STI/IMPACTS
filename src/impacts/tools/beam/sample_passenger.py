from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_HOUSEHOLDS_FILE = "households.parquet"
DEFAULT_PERSONS_FILE = "persons.parquet"
DEFAULT_PLANS_FILE = "plans.parquet"
DEFAULT_VEHICLES_FILE = "vehicles.csv.gz"
DEFAULT_CHUNK_SIZE = 250_000


class ProgressBar:
    def __init__(self, label: str, total: int):
        self.label = label
        self.total = max(0, int(total))
        self.current = 0
        self.last_render = 0.0
        self.started = time.time()

    def update(self, increment: int) -> None:
        self.current += max(0, int(increment))
        now = time.time()
        if now - self.last_render < 0.25 and self.current < self.total:
            return
        self.last_render = now
        total = max(1, self.total)
        ratio = min(1.0, self.current / total)
        width = 24
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.started
        message = f"\r{self.label:<18} [{bar}] {ratio * 100:5.1f}% ({self.current:,}/{self.total:,}) {elapsed:5.1f}s"
        print(message, end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        self.last_render = 0.0
        self.update(0)
        print(file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample UrbanSim passenger population files while keeping households, persons, "
            "vehicles, and plans aligned on the same sampled households."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing UrbanSim households.parquet, persons.parquet, plans.parquet, and vehicles.csv.gz",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the sampled files will be written",
    )
    parser.add_argument(
        "--extra-vehicles-file",
        default=None,
        help=(
            "Optional second vehicles file to sample by the same sampled households. "
            "The sampled copy is written to the output directory using the source basename."
        ),
    )
    parser.add_argument(
        "--blocks-file",
        default=None,
        help=(
            "Optional blocks parquet file to filter by the sampled households' block_id values. "
            "The sampled copy is written to the output directory using the source basename."
        ),
    )
    parser.add_argument(
        "--sample-share",
        type=float,
        required=True,
        help="Share of households to sample, in (0, 1]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for household sampling",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Chunk size used when filtering large vehicles and plans files",
    )
    return parser.parse_args()


def _normalize_integral_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return text
    if not decimal_value.is_finite():
        return text
    if decimal_value == decimal_value.to_integral_value():
        return format(decimal_value.quantize(Decimal("1")), "f")
    return text


def _normalize_identifier_series(series: pd.Series) -> pd.Series:
    return series.map(_normalize_integral_identifier)


def _resolve_input_paths(input_dir: Path) -> dict[str, Path]:
    paths = {
        "households": input_dir / DEFAULT_HOUSEHOLDS_FILE,
        "persons": input_dir / DEFAULT_PERSONS_FILE,
        "plans": input_dir / DEFAULT_PLANS_FILE,
        "vehicles": input_dir / DEFAULT_VEHICLES_FILE,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required UrbanSim input files:\n" + "\n".join(missing))
    return paths


def _resolve_extra_vehicles_path(extra_vehicles_file: str | None) -> Path | None:
    if extra_vehicles_file is None:
        return None
    path = Path(extra_vehicles_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Extra vehicles file does not exist: {path}")
    return path


def _resolve_blocks_path(blocks_file: str | None) -> Path | None:
    if blocks_file is None:
        return None
    path = Path(blocks_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Blocks file does not exist: {path}")
    return path


def _validate_sample_share(sample_share: float) -> None:
    if not (0 < sample_share <= 1):
        raise ValueError("--sample-share must be in the interval (0, 1]")


def _sample_households(
    households: pd.DataFrame,
    *,
    sample_share: float,
    seed: int,
) -> pd.DataFrame:
    _validate_sample_share(sample_share)
    sampled = households.sample(frac=sample_share, random_state=seed)
    return sampled.sort_index()


def _load_sampled_households(
    households_path: Path,
    *,
    sample_share: float,
    seed: int,
) -> tuple[pd.DataFrame, set[str]]:
    households = pd.read_parquet(households_path)
    if households.index.name != "household_id":
        raise ValueError(
            f"Expected households index name 'household_id', found {households.index.name!r}"
        )
    sampled_households = _sample_households(
        households,
        sample_share=sample_share,
        seed=seed,
    )
    sampled_households.index = _normalize_identifier_series(sampled_households.index.to_series())
    sampled_households.index.name = "household_id"
    ProgressBar("households", len(households)).update(len(households))
    print(file=sys.stderr, flush=True)
    return sampled_households, set(sampled_households.index.astype(str))


def _load_sampled_persons(persons_path: Path, sampled_household_ids: set[str]) -> pd.DataFrame:
    persons = pd.read_parquet(persons_path)
    if persons.index.name != "person_id":
        raise ValueError(f"Expected persons index name 'person_id', found {persons.index.name!r}")
    if "household_id" not in persons.columns:
        raise ValueError("Persons file is missing required column 'household_id'")
    normalized_household_ids = _normalize_identifier_series(persons["household_id"])
    sampled_persons = persons.loc[normalized_household_ids.isin(sampled_household_ids)].copy()
    sampled_persons["household_id"] = _normalize_identifier_series(sampled_persons["household_id"])
    sampled_persons.index = _normalize_identifier_series(sampled_persons.index.to_series())
    sampled_persons.index.name = "person_id"
    ProgressBar("persons", len(persons)).update(len(persons))
    print(file=sys.stderr, flush=True)
    return sampled_persons


def _filter_vehicles(
    vehicles_path: Path,
    output_path: Path,
    sampled_household_ids: set[str],
    *,
    chunksize: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if vehicles_path.suffix == ".parquet":
        frame = pd.read_parquet(vehicles_path)
        if "household_id" not in frame.columns:
            raise ValueError("Vehicles file is missing required column 'household_id'")
        normalized_household_ids = _normalize_identifier_series(frame["household_id"])
        filtered = frame.loc[normalized_household_ids.isin(sampled_household_ids)].copy()
        progress = ProgressBar(f"vehicles {vehicles_path.name}", len(frame))
        progress.update(len(frame))
        progress.close()
        filtered.to_parquet(output_path, index=False)
        return len(filtered)

    rows_written = 0
    first_chunk = True
    total_bytes = vehicles_path.stat().st_size
    progress = ProgressBar(f"vehicles {vehicles_path.name}", total_bytes)
    with vehicles_path.open("rb") as raw_input:
        if vehicles_path.suffix == ".gz":
            reader_handle = gzip.open(raw_input, mode="rt", newline="")
        else:
            reader_handle = raw_input
        with reader_handle as input_handle:
            with gzip.open(output_path, mode="wt", newline="") as handle:
                for chunk in pd.read_csv(input_handle, chunksize=chunksize):
                    if "household_id" not in chunk.columns:
                        raise ValueError("Vehicles file is missing required column 'household_id'")
                    normalized_household_ids = _normalize_identifier_series(chunk["household_id"])
                    filtered = chunk.loc[normalized_household_ids.isin(sampled_household_ids)].copy()
                    if not filtered.empty:
                        filtered.to_csv(handle, index=False, header=first_chunk)
                        rows_written += len(filtered)
                        first_chunk = False
                    progress.update(raw_input.tell() - progress.current)
                if first_chunk:
                    pd.read_csv(vehicles_path, nrows=0).to_csv(handle, index=False)
    progress.close()
    return rows_written


def _filter_blocks(
    blocks_path: Path,
    output_path: Path,
    sampled_block_ids: set[str],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(blocks_path)
    if "block_id" not in frame.columns:
        raise ValueError("Blocks file is missing required column 'block_id'")
    normalized_block_ids = _normalize_identifier_series(frame["block_id"])
    filtered = frame.loc[normalized_block_ids.isin(sampled_block_ids)].copy()
    progress = ProgressBar(f"blocks {blocks_path.name}", len(frame))
    progress.update(len(frame))
    progress.close()
    filtered.to_parquet(output_path, index=False)
    return len(filtered)


def _filter_plans(
    plans_path: Path,
    output_path: Path,
    sampled_person_ids: set[str],
    *,
    chunksize: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    parquet_file = pq.ParquetFile(plans_path)
    progress = ProgressBar("plans", parquet_file.metadata.num_rows)
    try:
        for batch in parquet_file.iter_batches(batch_size=chunksize):
            table = pa.Table.from_batches([batch])
            frame = table.to_pandas()
            if "person_id" not in frame.columns:
                raise ValueError("Plans file is missing required column 'person_id'")
            normalized_person_ids = _normalize_identifier_series(frame["person_id"])
            filtered = frame.loc[normalized_person_ids.isin(sampled_person_ids)].copy()
            progress.update(len(frame))
            if filtered.empty:
                continue
            filtered["person_id"] = _normalize_identifier_series(filtered["person_id"])
            table = pa.Table.from_pandas(filtered, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            rows_written += table.num_rows
    finally:
        if writer is not None:
            writer.close()
        progress.close()
    if writer is None:
        empty_frame = pd.read_parquet(plans_path).iloc[0:0].copy()
        pq.write_table(pa.Table.from_pandas(empty_frame, preserve_index=False), output_path)
    return rows_written


def _write_summary(
    output_dir: Path,
    *,
    sampled_households: pd.DataFrame,
    sampled_persons: pd.DataFrame,
    vehicles_rows: int,
    extra_vehicles_rows: int | None,
    extra_vehicles_file: str | None,
    blocks_rows: int | None,
    blocks_file: str | None,
    plans_rows: int,
    sample_share: float,
    seed: int,
) -> None:
    summary = {
        "sample_share": sample_share,
        "seed": seed,
        "households": int(len(sampled_households)),
        "persons": int(len(sampled_persons)),
        "vehicles": int(vehicles_rows),
        "extra_vehicles": extra_vehicles_rows,
        "extra_vehicles_file": extra_vehicles_file,
        "blocks": blocks_rows,
        "blocks_file": blocks_file,
        "plans": int(plans_rows),
    }
    (output_dir / "sample-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _resolve_input_paths(input_dir)
    extra_vehicles_path = _resolve_extra_vehicles_path(args.extra_vehicles_file)
    blocks_path = _resolve_blocks_path(args.blocks_file)
    sampled_households, sampled_household_ids = _load_sampled_households(
        input_paths["households"],
        sample_share=args.sample_share,
        seed=args.seed,
    )
    sampled_persons = _load_sampled_persons(input_paths["persons"], sampled_household_ids)
    sampled_person_ids = set(sampled_persons.index.astype(str))
    sampled_block_ids: set[str] = set()
    if "block_id" in sampled_households.columns:
        sampled_block_ids = set(_normalize_identifier_series(sampled_households["block_id"]).astype(str))

    households_output = output_dir / DEFAULT_HOUSEHOLDS_FILE
    persons_output = output_dir / DEFAULT_PERSONS_FILE
    vehicles_output = output_dir / DEFAULT_VEHICLES_FILE
    plans_output = output_dir / DEFAULT_PLANS_FILE
    extra_vehicles_rows: int | None = None
    blocks_rows: int | None = None

    sampled_households.to_parquet(households_output)
    sampled_persons.to_parquet(persons_output)
    vehicles_rows = _filter_vehicles(
        input_paths["vehicles"],
        vehicles_output,
        sampled_household_ids,
        chunksize=args.chunksize,
    )
    plans_rows = _filter_plans(
        input_paths["plans"],
        plans_output,
        sampled_person_ids,
        chunksize=args.chunksize,
    )
    if extra_vehicles_path is not None:
        extra_vehicles_rows = _filter_vehicles(
            extra_vehicles_path,
            output_dir / extra_vehicles_path.name,
            sampled_household_ids,
            chunksize=args.chunksize,
        )
    if blocks_path is not None:
        blocks_rows = _filter_blocks(
            blocks_path,
            output_dir / blocks_path.name,
            sampled_block_ids,
        )
    _write_summary(
        output_dir,
        sampled_households=sampled_households,
        sampled_persons=sampled_persons,
        vehicles_rows=vehicles_rows,
        extra_vehicles_rows=extra_vehicles_rows,
        extra_vehicles_file=None if extra_vehicles_path is None else str(extra_vehicles_path),
        blocks_rows=blocks_rows,
        blocks_file=None if blocks_path is None else str(blocks_path),
        plans_rows=plans_rows,
        sample_share=args.sample_share,
        seed=args.seed,
    )

    print(f"Sampled households: {len(sampled_households)}")
    print(f"Sampled persons: {len(sampled_persons)}")
    print(f"Sampled vehicles: {vehicles_rows}")
    if extra_vehicles_rows is not None:
        print(f"Sampled extra vehicles: {extra_vehicles_rows}")
    if blocks_rows is not None:
        print(f"Sampled blocks: {blocks_rows}")
    print(f"Sampled plans: {plans_rows}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
