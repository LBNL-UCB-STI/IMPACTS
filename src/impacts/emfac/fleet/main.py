"""Main entrypoint for the step-based fleet workflow."""

from pathlib import Path
import sys

from impacts.emfac.config import load_default_fleet_workflow
from impacts.emfac.config import load_fleet_workflow
from impacts.emfac.common import raise_runtime_error
from impacts.emfac.common import write_failure_trace
from impacts.emfac.common import write_trace
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


def _missing_activities_outputs(workflow: dict[str, object]) -> dict[str, Path]:
    activities = workflow["config"]["activities"]
    activities_output_root = Path(str(activities["outputs"])).expanduser().resolve()
    frism_year = workflow["config"]["frism"]["year"]
    scenario_name = str(workflow["scenario"])
    rates_store_root = activities_output_root / "emissions" / f"{frism_year}-{scenario_name}"
    rates_store_dataset = rates_store_root / "dataset"
    rates_store_duckdb = rates_store_root / "dataset.duckdb"
    required_outputs = {
        "passenger_rates_file": Path(str(activities["passenger_rates_file"])),
        "passenger_activity_file": Path(str(activities["passenger_activity_file"])),
        "passenger_fleet_file": Path(str(activities["passenger_fleet_file"])),
        "freight_rates_file": Path(str(activities["freight_rates_file"])),
        "freight_activity_file": Path(str(activities["freight_activity_file"])),
        "freight_fleet_file": Path(str(activities["freight_fleet_file"])),
        "rates_store_dataset": rates_store_dataset,
        "rates_store_duckdb": rates_store_duckdb,
    }
    missing = {label: path for label, path in required_outputs.items() if not path.exists()}
    if not missing:
        partition_exists = any(rates_store_dataset.glob("emfacId=*/*.parquet"))
        if not partition_exists:
            missing["rates_store_partitions"] = rates_store_dataset
    return missing


def ensure_activities_outputs_exist(workflow: dict[str, object]) -> dict[str, object]:
    missing = _missing_activities_outputs(workflow)
    if not missing:
        return workflow
    missing_paths = "\n".join(f"  missing: {path}" for path in missing.values())
    raise FileNotFoundError(
        "Fleet workflow requires EMFAC activities outputs to exist before running.\n"
        f"{missing_paths}"
    )


def run_workflow(workflow: dict[str, object]) -> dict[str, object]:
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
    return workflow


def run_all_steps(config_path: str | Path | None = None) -> None:
    """Run the fleet-only workflow against existing activities outputs."""
    try:
        workflow = _configure_run(config_path)
    except Exception as error:
        raise_runtime_error("config_load", error)
    workflow = ensure_activities_outputs_exist(workflow)
    run_workflow(workflow)


def main(config_path: str | Path | None = None) -> None:
    """CLI-friendly entrypoint for the fleet workflow."""
    run_all_steps(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
