from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
        "'python -m impacts.emfac fleet --activities-manifest activities_manifest.yaml'.",
    )
    parser.add_argument(
        "--config",
        dest="config_path",
        help="Path to the impacts settings YAML file.",
    )
    parser.add_argument(
        "--activities-manifest",
        dest="activities_manifest_path",
        help="Path to the activities_manifest.yaml file for fleet runs.",
    )
    return parser


def _parse_workflow_and_config(args: argparse.Namespace) -> tuple[str | None, str | None, str | None]:
    workflow: str | None = None
    config_path = args.config_path
    activities_manifest_path = args.activities_manifest_path
    positional = list(args.args or [])
    if positional and positional[0] in {"activities", "fleet"}:
        workflow = positional.pop(0)
    if positional:
        if config_path is not None or activities_manifest_path is not None:
            raise SystemExit("Specify the path either positionally or with a flag, not both.")
        if len(positional) > 1:
            raise SystemExit("Too many positional arguments. Expected at most one config path.")
        if workflow == "fleet":
            activities_manifest_path = positional[0]
        else:
            config_path = positional[0]
    return workflow, config_path, activities_manifest_path


def _ensure_activities(config_path: str) -> None:
    from impacts.config.settings_builder import load_settings_from_yaml
    from impacts.emfac.preparation import ensure_emfac_activities_outputs
    settings = load_settings_from_yaml(config_path)
    ensure_emfac_activities_outputs(settings, Path(config_path))


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    workflow, config_path, activities_manifest_path = _parse_workflow_and_config(args)
    if workflow == "activities":
        if not config_path:
            raise SystemExit("--config is required. Usage: python -m impacts.emfac activities --config <settings.yaml>")
        _ensure_activities(config_path)
        return
    if workflow == "fleet":
        if not activities_manifest_path:
            raise SystemExit(
                "--activities-manifest is required. Usage: python -m impacts.emfac fleet --activities-manifest <activities_manifest.yaml>"
            )
        run_fleet(activities_manifest_path=activities_manifest_path)
        return
    if not config_path:
        raise SystemExit("--config is required. Usage: python -m impacts.emfac --config <settings.yaml>")
    activities_manifest = _ensure_activities(config_path)
    run_fleet(activities_manifest_path=activities_manifest["activities_manifest_path"])


if __name__ == "__main__":
    main(sys.argv[1:])
