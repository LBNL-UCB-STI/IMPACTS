from __future__ import annotations

import argparse
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from impacts.utils.utils_sampling import sample_events_by_vehicle
from impacts.utils.utils_sampling import sample_skims_by_fraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample large BEAM events or skims files into examples/pilates/upstream."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_events = subparsers.add_parser("events", help="Sample events by vehicle id.")
    sample_events.add_argument("--input", required=True)
    sample_events.add_argument("--fraction", type=float, default=0.05)
    sample_events.add_argument("--seed", type=int, default=42)
    sample_events.add_argument("--vehicle-column", default="vehicle")
    sample_events.add_argument(
        "--output",
        default=str(EXAMPLE_DIR / "upstream" / "events_sample.csv.gz"),
    )

    sample_skims = subparsers.add_parser("skims", help="Sample skims by row fraction.")
    sample_skims.add_argument("--input", required=True)
    sample_skims.add_argument("--fraction", type=float, default=0.05)
    sample_skims.add_argument("--seed", type=int, default=42)
    sample_skims.add_argument("--compact-workers", type=int, default=4)
    sample_skims.add_argument("--population-sample", type=float, default=1.0)
    sample_skims.add_argument(
        "--output",
        default=str(EXAMPLE_DIR / "upstream" / "skimsEmissionsTotals_sample.csv.gz"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "events":
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        events_stats = sample_events_by_vehicle(
            input_path=args.input,
            output_path=output,
            fraction=args.fraction,
            seed=args.seed,
            vehicle_column=args.vehicle_column,
        )
        print(f"sampled events: {output}")
        print(
            "  vehicles kept: "
            f"{events_stats['selected_vehicles']} / {events_stats['unique_vehicles']}"
        )
        print(f"  rows kept: {events_stats['kept_rows']} / {events_stats['total_rows']}")
        return 0

    if args.command == "skims":
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        skims_stats = sample_skims_by_fraction(
            input_path=args.input,
            output_path=output,
            fraction=args.fraction,
            seed=args.seed,
            compact_workers=args.compact_workers,
            population_sample=args.population_sample,
        )
        print(f"sampled skims: {output}")
        print(f"  rows kept: {skims_stats['kept_rows']} / {skims_stats['total_rows']}")
        print(f"  expansion factor: {skims_stats['expansion_factor']}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
