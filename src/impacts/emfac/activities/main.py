"""Main entrypoint for the step-based EMFAC assembly workflow."""

from pathlib import Path
import sys

from impacts.emfac.config import load_activities_workflow
from impacts.emfac.config import load_default_activities_workflow
from impacts.emfac.activities.step1_prepare_emissions_and_activities_tables import run_step1
from impacts.emfac.activities.step2_build_comprehensive_project_analysis import run_step2
from impacts.emfac.activities.step3_fill_project_analysis_rates import run_step3
from impacts.emfac.activities.step4_finalize_output import run_step4
from impacts.emfac.common import raise_runtime_error
from impacts.emfac.common import write_failure_trace
from impacts.emfac.common import write_trace


def _configure_run(config_path: str | Path | None = None) -> dict[str, object]:
    if config_path is None:
        return load_default_activities_workflow()
    return load_activities_workflow(config_path)


def _print_run_banner(workflow: dict[str, object]) -> None:
    run = workflow["run"]
    print(f"\n{'='*50}")
    print(f"  EMFAC TABLE ASSEMBLY - {run['region_label']} {run['calendar_year']}")
    print(f"  Passenger Rates Output: {workflow['paths']['final_output_passenger']}")
    print(f"  Passenger Inventory Final Activity Output: {workflow['paths']['final_activity_output_passenger']}")
    print(f"  Passenger Inventory Final Fleet Output: {workflow['paths']['final_fleet_output_passenger']}")
    print(f"  Freight Rates Output: {workflow['paths']['final_output_freight']}")
    print(f"  Freight Inventory Final Activity Output: {workflow['paths']['final_activity_output_freight']}")
    print(f"  Freight Inventory Final Fleet Output: {workflow['paths']['final_fleet_output_freight']}")
    print(f"{'='*50}\n")


def _run_step(workflow: dict[str, object], *, step_name: str, runner) -> dict[str, object]:
    write_trace(workflow, f"{step_name}_start", {"step": step_name, "status": "started"})
    try:
        updated_workflow = runner(workflow)
    except Exception as error:
        write_failure_trace(workflow, step=step_name, error=error, payload={"status": "failed"})
        raise_runtime_error(step_name, error)
    write_trace(updated_workflow, f"{step_name}_success", {"step": step_name, "status": "completed"})
    return updated_workflow


def run_workflow(workflow: dict[str, object]) -> dict[str, object]:
    _print_run_banner(workflow)
    write_trace(
        workflow,
        "workflow_start",
        {
            "status": "started",
            "run": workflow["run"],
            "paths": workflow["paths"],
        },
    )
    try:
        workflow = _run_step(workflow, step_name="step1_prepare_emissions_and_activities_tables", runner=run_step1)
        workflow = _run_step(workflow, step_name="step2_build_comprehensive_project_analysis", runner=run_step2)
        workflow = _run_step(workflow, step_name="step3_fill_project_analysis_rates", runner=run_step3)
        workflow = _run_step(workflow, step_name="step4_finalize_output", runner=run_step4)
    except Exception as error:
        if isinstance(error, RuntimeError):
            write_failure_trace(workflow, step="workflow", error=error, payload={"status": "failed"})
        raise
    write_trace(workflow, "workflow_success", {"status": "completed"})
    print("  DONE")
    return workflow


def run_all_steps(config_path: str | Path | None = None) -> None:
    try:
        workflow = _configure_run(config_path)
    except Exception as error:
        raise_runtime_error("config_load", error)
    run_workflow(workflow)


def main(config_path: str | Path | None = None) -> None:
    run_all_steps(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
