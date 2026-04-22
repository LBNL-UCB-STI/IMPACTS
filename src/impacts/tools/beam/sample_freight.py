from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd


DEFAULT_CARRIERS_FILE = "carriers--2018-Baseline.parquet"
DEFAULT_PAYLOADS_FILE = "payloads--2018-Baseline.parquet"
DEFAULT_TOURS_FILE = "tours--2018-Baseline.parquet"


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
        print(
            f"\r{self.label:<18} [{bar}] {ratio * 100:5.1f}% ({self.current:,}/{self.total:,}) {elapsed:5.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        self.last_render = 0.0
        self.update(0)
        print(file=sys.stderr, flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample BEAM freight outputs while keeping carriers, tours, and payloads aligned "
            "on the same sampled tour set."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing carriers, payloads, and tours parquet files",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the sampled freight files will be written",
    )
    parser.add_argument(
        "--extra-carriers-file",
        default=None,
        help=(
            "Optional second carriers parquet file to sample by the same sampled tour set. "
            "The sampled copy is written to the output directory using the source basename."
        ),
    )
    parser.add_argument(
        "--sample-share",
        type=float,
        required=True,
        help="Share of tours to sample, in (0, 1]",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for tour sampling",
    )
    return parser.parse_args()


def _validate_sample_share(sample_share: float) -> None:
    if not (0 < sample_share <= 1):
        raise ValueError("--sample-share must be in the interval (0, 1]")


def _resolve_input_paths(input_dir: Path) -> dict[str, Path]:
    paths = {
        "carriers": input_dir / DEFAULT_CARRIERS_FILE,
        "payloads": input_dir / DEFAULT_PAYLOADS_FILE,
        "tours": input_dir / DEFAULT_TOURS_FILE,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required freight input files:\n" + "\n".join(missing))
    return paths


def _resolve_extra_carriers_path(extra_carriers_file: str | None) -> Path | None:
    if extra_carriers_file is None:
        return None
    path = Path(extra_carriers_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Extra carriers file does not exist: {path}")
    return path


def _sample_tours(
    tours: pd.DataFrame,
    *,
    sample_share: float,
    seed: int,
) -> pd.DataFrame:
    _validate_sample_share(sample_share)
    if "tourId" not in tours.columns:
        raise ValueError("Tours file is missing required column 'tourId'")
    if tours["tourId"].isna().any():
        raise ValueError("Tours file contains null values in required column 'tourId'")
    if not tours["tourId"].astype(str).is_unique:
        raise ValueError("Tours file must have unique 'tourId' values")
    progress = ProgressBar("tours", len(tours))
    progress.update(len(tours))
    progress.close()
    sampled = tours.sample(frac=sample_share, random_state=seed)
    return sampled.sort_values("tourId").reset_index(drop=True)


def _filter_by_tour_id(frame: pd.DataFrame, frame_name: str, sampled_tour_ids: set[str]) -> pd.DataFrame:
    if "tourId" not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column 'tourId'")
    progress = ProgressBar(frame_name.lower().replace(" file", ""), len(frame))
    progress.update(len(frame))
    progress.close()
    return frame.loc[frame["tourId"].astype(str).isin(sampled_tour_ids)].copy()


def _write_summary(
    output_dir: Path,
    *,
    sample_share: float,
    seed: int,
    tours: pd.DataFrame,
    carriers: pd.DataFrame,
    extra_carriers: pd.DataFrame | None,
    extra_carriers_file: str | None,
    payloads: pd.DataFrame,
) -> None:
    summary = {
        "sample_share": sample_share,
        "seed": seed,
        "tours": int(len(tours)),
        "carriers": int(len(carriers)),
        "extra_carriers": None if extra_carriers is None else int(len(extra_carriers)),
        "extra_carriers_file": extra_carriers_file,
        "payloads": int(len(payloads)),
        "carrier_ids": int(carriers["carrierId"].nunique()) if "carrierId" in carriers.columns else None,
        "vehicle_ids": int(carriers["vehicleId"].nunique()) if "vehicleId" in carriers.columns else None,
    }
    (output_dir / "sample-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = _resolve_input_paths(input_dir)
    extra_carriers_path = _resolve_extra_carriers_path(args.extra_carriers_file)
    tours = pd.read_parquet(input_paths["tours"])
    sampled_tours = _sample_tours(
        tours,
        sample_share=args.sample_share,
        seed=args.seed,
    )
    sampled_tour_ids = set(sampled_tours["tourId"].astype(str))

    carriers = pd.read_parquet(input_paths["carriers"])
    payloads = pd.read_parquet(input_paths["payloads"])
    sampled_carriers = _filter_by_tour_id(carriers, "Carriers file", sampled_tour_ids)
    sampled_extra_carriers: pd.DataFrame | None = None
    if extra_carriers_path is not None:
        extra_carriers = pd.read_parquet(extra_carriers_path)
        sampled_extra_carriers = _filter_by_tour_id(extra_carriers, "Extra carriers file", sampled_tour_ids)
    sampled_payloads = _filter_by_tour_id(payloads, "Payloads file", sampled_tour_ids)

    sampled_carriers.to_parquet(output_dir / input_paths["carriers"].name, index=False)
    if sampled_extra_carriers is not None:
        sampled_extra_carriers.to_parquet(output_dir / extra_carriers_path.name, index=False)
    sampled_payloads.to_parquet(output_dir / input_paths["payloads"].name, index=False)
    sampled_tours.to_parquet(output_dir / input_paths["tours"].name, index=False)
    _write_summary(
        output_dir,
        sample_share=args.sample_share,
        seed=args.seed,
        tours=sampled_tours,
        carriers=sampled_carriers,
        extra_carriers=sampled_extra_carriers,
        extra_carriers_file=None if extra_carriers_path is None else str(extra_carriers_path),
        payloads=sampled_payloads,
    )

    print(f"Sampled tours: {len(sampled_tours)}")
    print(f"Sampled carriers: {len(sampled_carriers)}")
    if sampled_extra_carriers is not None:
        print(f"Sampled extra carriers: {len(sampled_extra_carriers)}")
    print(f"Sampled payloads: {len(sampled_payloads)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
