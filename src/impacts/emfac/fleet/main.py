"""Main entrypoint for the step-based fleet workflow."""

from copy import deepcopy
from pathlib import Path
import sys

from impacts.emfac.config import load_activities_workflow_from_data
from impacts.emfac.config import load_default_fleet_workflow
from impacts.emfac.config import load_fleet_workflow
from impacts.emfac.common import raise_runtime_error
from impacts.emfac.common import write_failure_trace
from impacts.emfac.common import write_trace
from impacts.emfac.activities.main import run_workflow as run_activities_workflow
from impacts.emfac.fleet.step1_build_vehicle_types import run_step1
from impacts.emfac.fleet.step2_map_emfac_bus_bike import run_step2
from impacts.emfac.fleet.step3_map_emfac_atlas import run_step3
from impacts.emfac.fleet.step4_map_emfac_frism import run_step4
from impacts.emfac.fleet.step5_map_emfac_rates import run_step5


def _configure_run(config_path: str | Path | None = None) -> dict[str, object]:
    if config_path is None:
        return load_default_fleet_workflow()
    return load_fleet_workflow(config_path)


def _print_run_banner(workflow: dict[str, object]) -> None:
    output_path = workflow["config"]["output"]
    print(f"{'='*50}")
    print(f"  EMISSIONS PROCESSING - {str(workflow['area']).upper()} REGION")
    print(f"  Scenario: {workflow['scenario']}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")


def _run_step(workflow: dict[str, object], *, step_name: str, runner) -> dict[str, object]:
    write_trace(workflow, f"{step_name}_start", {"step": step_name, "status": "started"})
    try:
        updated_workflow = runner(workflow)
    except Exception as error:
        write_failure_trace(workflow, step=step_name, error=error, payload={"status": "failed"})
        raise_runtime_error(step_name, error)
    write_trace(updated_workflow, f"{step_name}_success", {"step": step_name, "status": "completed"})
    return updated_workflow


def _bootstrap_emfac_outputs_if_needed(workflow: dict[str, object]) -> dict[str, object]:
    activities = workflow["config"]["activities"]
    required_outputs = {
        "passenger_rates_file": Path(str(activities["passenger_rates_file"])),
        "passenger_activity_file": Path(str(activities["passenger_activity_file"])),
        "passenger_fleet_file": Path(str(activities["passenger_fleet_file"])),
        "freight_rates_file": Path(str(activities["freight_rates_file"])),
        "freight_activity_file": Path(str(activities["freight_activity_file"])),
        "freight_fleet_file": Path(str(activities["freight_fleet_file"])),
    }
    missing = {label: path for label, path in required_outputs.items() if not path.exists()}
    if not missing:
        return workflow

    activities_config = deepcopy(activities)

    print("Bootstrapping EMFAC activities because required fleet inputs are missing:")
    for path in missing.values():
        print(f"  missing: {path}")
    activities_workflow = load_activities_workflow_from_data(
        activities_config,
        source_label="<emfac.activities>",
    )
    activities_workflow = run_activities_workflow(activities_workflow)

    activities["passenger_rates_file"] = str(Path(str(activities_workflow["paths"]["final_output_passenger"])).resolve())
    activities["passenger_activity_file"] = str(Path(str(activities_workflow["paths"]["final_activity_output_passenger"])).resolve())
    activities["passenger_fleet_file"] = str(Path(str(activities_workflow["paths"]["final_fleet_output_passenger"])).resolve())
    activities["freight_rates_file"] = str(Path(str(activities_workflow["paths"]["final_output_freight"])).resolve())
    activities["freight_activity_file"] = str(Path(str(activities_workflow["paths"]["final_activity_output_freight"])).resolve())
    activities["freight_fleet_file"] = str(Path(str(activities_workflow["paths"]["final_fleet_output_freight"])).resolve())
    return workflow


def run_all_steps(config_path: str | Path | None = None) -> None:
    """Run the active fleet workflow steps."""
    try:
        workflow = _configure_run(config_path)
    except Exception as error:
        raise_runtime_error("config_load", error)
    workflow = _bootstrap_emfac_outputs_if_needed(workflow)
    _print_run_banner(workflow)
    write_trace(
        workflow,
        "workflow_start",
        {
            "status": "started",
            "area": workflow["area"],
            "scenario": workflow["scenario"],
            "output": workflow["config"]["output"],
            "paths": workflow["paths"],
        },
    )
    try:
        workflow = _run_step(workflow, step_name="step1_build_vehicle_types", runner=run_step1)
        workflow = _run_step(workflow, step_name="step2_map_emfac_bus_bike", runner=run_step2)
        workflow = _run_step(workflow, step_name="step3_map_emfac_atlas", runner=run_step3)
        workflow = _run_step(workflow, step_name="step4_map_emfac_frism", runner=run_step4)
        workflow = _run_step(workflow, step_name="step5_map_emfac_rates", runner=run_step5)
    except Exception as error:
        if isinstance(error, RuntimeError):
            write_failure_trace(workflow, step="workflow", error=error, payload={"status": "failed"})
        raise
    write_trace(workflow, "workflow_success", {"status": "completed"})
    vehicles_output_file = workflow.get("mapped_passenger_vehicles_file", "")
    carriers_output_file = workflow.get("mapped_freight_carriers_file", "")
    passenger_vehicle_types_output = workflow.get("passenger_vehicle_types_with_rates_file", "")
    freight_vehicle_types_output = workflow.get("freight_vehicle_types_with_rates_file", "")
    if vehicles_output_file:
        print(f"  Passenger vehicles file: {vehicles_output_file}")
    if carriers_output_file:
        print(f"  Freight carriers file: {carriers_output_file}")
    if passenger_vehicle_types_output:
        print(f"  Passenger vehicle types file: {passenger_vehicle_types_output}")
    if freight_vehicle_types_output:
        print(f"  Freight vehicle types file: {freight_vehicle_types_output}")
    print("  DONE")


def main(config_path: str | Path | None = None) -> None:
    """CLI-friendly entrypoint for the fleet workflow."""
    run_all_steps(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
