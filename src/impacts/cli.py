import argparse
import os
import sys
from typing import List, Optional

from impacts.grid_nox_to_no2 import DEFAULT_STEP_ORDER, INPUT_DIR, OUTPUT_DIR, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IMPACTS preprocessing steps for GRID outputs."
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List available step names and exit.",
    )
    parser.add_argument(
        "--step",
        action="append",
        choices=DEFAULT_STEP_ORDER,
        help="Run a specific step (can be repeated).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps in the default order.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Path to the input data directory (default: <repo>/data/input).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Path to the output data directory (default: <repo>/data/output).",
    )
    parser.add_argument(
        "--bounding-box",
        default=None,
        help="Bounding box as minx,miny,maxx,maxy for IMPACTS bounding box.",
    )
    parser.add_argument(
        "--counties-path",
        default=None,
        help="Path to a counties shapefile/geojson for IMPACTS bounding box.",
    )
    return parser


def pick_steps(args: argparse.Namespace) -> List[str]:
    if args.list_steps:
        return []
    if args.all:
        return DEFAULT_STEP_ORDER
    if args.step:
        return args.step
    return []


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_steps:
        for step in DEFAULT_STEP_ORDER:
            print(step)
        return 0

    steps = pick_steps(args)
    if not steps:
        parser.error("Choose --all or at least one --step.")

    if args.bounding_box:
        os.environ["IMPACTS_BOUNDING_BOX"] = args.bounding_box
    if args.counties_path:
        os.environ["IMPACTS_COUNTIES_PATH"] = args.counties_path

    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR

    run_pipeline(steps, input_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())