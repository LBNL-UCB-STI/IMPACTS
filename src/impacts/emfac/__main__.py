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
        "args",
        nargs="*",
        help="Optional workflow name and/or config path. Supported forms: "
        "'python -m impacts.emfac', "
        "'python -m impacts.emfac examples/emfac/settings.yaml', "
        "'python -m impacts.emfac activities examples/emfac/settings.yaml', "
        "'python -m impacts.emfac fleet examples/emfac/settings.yaml'.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the EMFAC settings YAML file.",
    )
    return parser


def _parse_workflow_and_config(args: argparse.Namespace) -> tuple[str | None, str | None]:
    workflow: str | None = None
    config_path = args.config_path
    positional = list(args.args or [])
    if positional and positional[0] in {"activities", "fleet"}:
        workflow = positional.pop(0)
    if positional:
        if config_path is not None:
            raise SystemExit("Specify the EMFAC config path either positionally or with --config, not both.")
        if len(positional) > 1:
            raise SystemExit("Too many positional arguments. Expected at most one config path.")
        config_path = positional[0]
    return workflow, config_path


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    workflow, config_path = _parse_workflow_and_config(args)
    if workflow == "activities":
        run_activities(config_path)
        return
    if workflow == "fleet":
        run_fleet(config_path)
        return
    fleet_workflow = load_fleet_workflow(config_path)
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
