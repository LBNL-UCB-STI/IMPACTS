import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

STEP_SCRIPTS = {
    "convert_cmaq_polygon": "0_convert_cmaq_polygon.py",
    "generate_xwalk": "1_generate_xwalk_gob_isrm.py",
    "cmaq_ratio_to_isrm": "2_cmaq_ratio_to_isrm.py",
    "nox_to_no2_isrm": "3_nox_to_no2_isrm.py",
}

DEFAULT_STEP_ORDER = list(STEP_SCRIPTS.keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run IMPACTS preprocessing steps for ISRM outputs."
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
        "--r-script-dir",
        default=None,
        help="Directory containing the R scripts (defaults to src/impacts).",
    )
    parser.add_argument(
        "--workdir",
        default=None,
        help="Working directory for running R scripts.",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory mounted into the container (default: /input).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory mounted into the container (default: /output).",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    r_script_dir = Path(
        args.r_script_dir
        or os.environ.get("IMPACTS_R_SCRIPT_DIR", repo_root / "src" / "impacts")
    ).resolve()
    workdir = Path(
        args.workdir
        or os.environ.get("IMPACTS_WORKDIR", r_script_dir)
    ).resolve()
    input_dir = Path(
        args.input_dir or os.environ.get("IMPACTS_INPUT_DIR", "/input")
    ).resolve()
    output_dir = Path(
        args.output_dir or os.environ.get("IMPACTS_OUTPUT_DIR", "/output")
    ).resolve()
    return r_script_dir, workdir, input_dir, output_dir


def pick_steps(args: argparse.Namespace) -> List[str]:
    if args.list_steps:
        return []
    if args.all:
        return DEFAULT_STEP_ORDER
    if args.step:
        return args.step
    return []


def run_steps(
    steps: List[str],
    r_script_dir: Path,
    workdir: Path,
    input_dir: Path,
    output_dir: Path,
    bounding_box: Optional[str],
    counties_path: Optional[str],
    dry_run: bool,
) -> int:
    env = os.environ.copy()
    env.update(
        {
            "IMPACTS_INPUT_DIR": str(input_dir),
            "IMPACTS_OUTPUT_DIR": str(output_dir),
            "IMPACTS_R_SCRIPT_DIR": str(r_script_dir),
        }
    )
    if bounding_box:
        env["IMPACTS_BOUNDING_BOX"] = bounding_box
    if counties_path:
        env["IMPACTS_COUNTIES_PATH"] = counties_path

    for step in steps:
        script_name = STEP_SCRIPTS[step]
        script_path = r_script_dir / script_name
        cmd = [sys.executable, str(script_path)]
        if dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True, cwd=str(workdir), env=env)
    return 0


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

    r_script_dir, workdir, input_dir, output_dir = resolve_paths(args)
    return run_steps(
        steps,
        r_script_dir,
        workdir,
        input_dir,
        output_dir,
        args.bounding_box,
        args.counties_path,
        args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
