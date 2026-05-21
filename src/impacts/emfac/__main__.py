from __future__ import annotations

import argparse
import sys
from pathlib import Path

from impacts.emfac.activities.main import run_workflow as run_activities_workflow
from impacts.emfac.fleet.main import run_workflow as run_fleet_workflow
from impacts.emfac.fleet.main import main as run_fleet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.emfac",
        description="Run EMFAC activities, fleet, or the full EMFAC workflow.",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Optional workflow name and/or config path. Supported forms: "
        "'python -m impacts.emfac --config settings.yaml', "
        "'python -m impacts.emfac activities --config settings.yaml', "
        "'python -m impacts.emfac fleet --config settings.yaml'.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the impacts settings YAML file.",
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
            raise SystemExit("Specify the config path either positionally or with --config, not both.")
        if len(positional) > 1:
            raise SystemExit("Too many positional arguments. Expected at most one config path.")
        config_path = positional[0]
    return workflow, config_path


def _run_activities(config_path: str) -> None:
    from impacts.config.settings_builder import load_settings_from_yaml
    from impacts.emfac.preparation import build_activities_workflow_from_settings
    settings = load_settings_from_yaml(config_path)
    workflow = build_activities_workflow_from_settings(settings, Path(config_path))
    run_activities_workflow(workflow)


def _ensure_activities(config_path: str) -> None:
    from impacts.config.settings_builder import load_settings_from_yaml
    from impacts.emfac.preparation import build_activities_workflow_from_settings
    settings = load_settings_from_yaml(config_path)
    workflow = build_activities_workflow_from_settings(settings, Path(config_path))
    sentinel = Path(str(workflow["paths"]["final_activity_emfacid_output_passenger"]))
    if not sentinel.exists():
        print(f"Running EMFAC activities first — outputs missing under {sentinel.parent}")
        run_activities_workflow(workflow)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    workflow, config_path = _parse_workflow_and_config(args)
    if workflow == "activities":
        _run_activities(config_path)
        return
    # fleet or full run: ensure activities outputs exist, then run fleet
    _ensure_activities(config_path)
    run_fleet(config_path)


if __name__ == "__main__":
    main(sys.argv[1:])
