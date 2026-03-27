from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from impacts.postprocessor import _read_table
from impacts.preprocessor import preprocess_workflow
from impacts.runner import run_from_input_manifest

DEFAULT_WORKFLOW = EXAMPLE_DIR / "settings.yaml"
DEFAULT_WORKSPACE = EXAMPLE_DIR / "workspace"


def _print_artifact_preview(label: str, path: Path) -> None:
    if not path.exists():
        print(f"{label}: not produced")
        return

    table = _read_table(str(path))
    preview_cols = [
        col
        for col in [
            "linkId",
            "vehicleTypeId",
            "GRID",
            "cell_id",
            "zone_isrm",
            "zone_grid100",
            "tons_per_year_NOx_allocated",
            "tons_per_year_NOx_allocated_allocated",
            "tons_per_year_NOx_allocated_allocated_allocated",
            "tons_per_year_PM2_5_allocated",
            "tons_per_year_PM2_5_allocated_allocated",
            "tons_per_year_PM2_5_allocated_allocated_allocated",
        ]
        if col in table.columns
    ]

    print(f"{label}: {path}")
    print(f"{label} rows: {len(table)}")
    print(f"{label} columns: {', '.join(table.columns)}")
    if preview_cols:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(table[preview_cols].head(10).to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone PILATES impacts example through emissions allocation.")
    parser.add_argument("--config", default=str(DEFAULT_WORKFLOW))
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--keep-workspace", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )

    workflow = Path(args.config).resolve()
    workspace = Path(args.workspace).resolve()
    if workspace.exists() and not args.keep_workspace:
        shutil.rmtree(workspace)

    logging.getLogger(__name__).info("Example workflow: preprocess")
    preprocess_manifest = preprocess_workflow(
        runtime_config_path=workflow,
        staging_dir=workspace,
    )
    logging.getLogger(__name__).info("Example workflow: run through emissions allocation")
    run_manifest = run_from_input_manifest(
        input_manifest_path=preprocess_manifest["inputs_manifest_path"],
        output_dir=workspace,
        run_dispersion=False,
    )
    inmap_allocation_path = Path(run_manifest["raw_outputs"]["emissions_inmap_grid_allocated"])
    aermod_allocation_path = Path(run_manifest["raw_outputs"].get("emissions_aermod_grid_allocated") or "")
    print(f"inputs manifest: {preprocess_manifest['inputs_manifest_path']}")
    print(f"run manifest: {run_manifest['run_manifest_path']}")
    print(f"stopped after: {run_manifest['execution']['stopped_after']}")
    _print_artifact_preview("inmap_grid allocation", inmap_allocation_path)
    if run_manifest["raw_outputs"].get("emissions_aermod_grid_allocated"):
        _print_artifact_preview("aermod_grid allocation", aermod_allocation_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
