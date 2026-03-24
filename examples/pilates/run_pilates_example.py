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

DEFAULT_WORKFLOW = EXAMPLE_DIR / "workflow.yaml"
DEFAULT_WORKSPACE = EXAMPLE_DIR / "workspace"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone PILATES impacts example through emissions allocation.")
    parser.add_argument("--workflow-config", default=str(DEFAULT_WORKFLOW))
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

    workflow = Path(args.workflow_config).resolve()
    workspace = Path(args.workspace).resolve()
    if workspace.exists() and not args.keep_workspace:
        shutil.rmtree(workspace)

    logging.getLogger(__name__).info("Example workflow: preprocess")
    preprocess_manifest = preprocess_workflow(
        workflow_config_path=workflow,
        staging_dir=workspace,
    )
    logging.getLogger(__name__).info("Example workflow: run through emissions allocation")
    run_manifest = run_from_input_manifest(
        input_manifest_path=preprocess_manifest["inputs_manifest_path"],
        output_dir=workspace / "output",
        run_dispersion=False,
    )
    allocation_path = Path(run_manifest["raw_outputs"]["emissions_inmap_grid_allocated"])
    allocated = _read_table(str(allocation_path))
    preview_cols = [
        col
        for col in [
            "hour",
            "linkId",
            "vehicleTypeId",
            "process",
            "GRID",
            "cell_id",
            "proportion",
            "em_NOx_allocated",
            "em_PM2_5_allocated",
            "em_ROG_allocated",
        ]
        if col in allocated.columns
    ]
    print(f"inputs manifest: {preprocess_manifest['inputs_manifest_path']}")
    print(f"run manifest: {run_manifest['run_manifest_path']}")
    print(f"allocation artifact: {allocation_path}")
    print(f"stopped after: {run_manifest['execution']['stopped_after']}")
    print(f"rows: {len(allocated)}")
    print(f"columns: {', '.join(allocated.columns)}")
    if preview_cols:
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(allocated[preview_cols].head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
