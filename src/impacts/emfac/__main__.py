from __future__ import annotations

import argparse
import sys

from impacts.emfac.activities.main import main as run_activities
from impacts.emfac.fleet.main import main as run_fleet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.emfac",
        description="Run EMFAC activities or fleet workflows.",
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        choices=("activities", "fleet"),
        default="fleet",
        help="Workflow to run. Defaults to fleet.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the EMFAC settings YAML file.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workflow == "activities":
        run_activities(args.config_path)
        return
    run_fleet(args.config_path)


if __name__ == "__main__":
    main(sys.argv[1:])
