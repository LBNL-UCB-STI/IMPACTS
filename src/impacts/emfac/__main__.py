from __future__ import annotations

import argparse
import sys

from impacts.emfac.activities.main import main as run_activities
from impacts.emfac.activities.main import run_workflow as run_activities_workflow
from impacts.emfac.config import load_activities_workflow_from_data
from impacts.emfac.config import load_fleet_workflow
from impacts.emfac.fleet.main import main as run_fleet
from impacts.emfac.fleet.main import run_workflow as run_fleet_workflow
from impacts.emfac.fleet.main import _missing_activities_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.emfac",
        description="Run EMFAC activities, fleet, or the full EMFAC workflow.",
    )
    parser.add_argument(
        "workflow",
        nargs="?",
        choices=("activities", "fleet"),
        default=None,
        help="Optional workflow to run. Omit to run the full EMFAC workflow.",
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
    if args.workflow == "fleet":
        run_fleet(args.config_path)
        return
    fleet_workflow = load_fleet_workflow(args.config_path)
    missing = _missing_activities_outputs(fleet_workflow)
    if missing:
        print("Running EMFAC activities first because required activities outputs are missing:")
        for path in missing.values():
            print(f"  missing: {path}")
        activities_workflow = load_activities_workflow_from_data(
            dict(fleet_workflow["config"]["activities"]),
            source_label="<emfac.activities>",
        )
        run_activities_workflow(activities_workflow)
    run_fleet_workflow(fleet_workflow)


if __name__ == "__main__":
    main(sys.argv[1:])
