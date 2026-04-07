"""Main entrypoint for the step-based EMFAC assembly workflow."""

from pathlib import Path
import sys

from impacts.emfac.config import load_default_workflow
from impacts.emfac.config import load_workflow
from impacts.emfac.step1_clean_input_tables import run_step1
from impacts.emfac.step2_append_nh3 import run_step2
from impacts.emfac.step3_append_black_carbon import run_step3
from impacts.emfac.step4_append_road_dust import run_step4
from impacts.emfac.step5_finalize_output import run_step5


def _configure_run(config_path: str | Path | None = None) -> dict[str, object]:
    if config_path is None:
        return load_default_workflow()
    return load_workflow(config_path)


def _print_run_banner(workflow: dict[str, object]) -> None:
    run = workflow["run"]
    print(f"\n{'='*50}")
    print(f"  EMFAC TABLE ASSEMBLY - {run['region_label']} {run['calendar_year']}")
    print(f"  Output: {workflow['paths']['final_output']}")
    print(f"  Fleet Output: {workflow['paths']['final_fleet_output']}")
    print(f"{'='*50}\n")


def run_all_steps(config_path: str | Path | None = None) -> None:
    workflow = _configure_run(config_path)
    _print_run_banner(workflow)
    workflow = run_step1(workflow)
    workflow = run_step2(workflow)
    workflow = run_step3(workflow)
    workflow = run_step4(workflow)
    run_step5(workflow)
    print("  DONE")


def main(config_path: str | Path | None = None) -> None:
    run_all_steps(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
