"""Main entrypoint for the step-based fleet workflow."""

from copy import deepcopy
from pathlib import Path
import sys

from impacts.emfac.activities.config import load_workflow_from_data as load_activities_workflow_from_data
from impacts.emfac.activities.main import run_workflow as run_activities_workflow
from impacts.emfac.fleet.config import load_default_workflow
from impacts.emfac.fleet.config import load_workflow
from impacts.emfac.fleet.step1_build_vehicle_types import run_step1
from impacts.emfac.fleet.step2_map_emfac_bus_bike import run_step2
from impacts.emfac.fleet.step3_map_emfac_atlas import run_step3
from impacts.emfac.fleet.step4_map_emfac_frism import run_step4
from impacts.emfac.fleet.step5_map_emfac_rates import run_step5


def _configure_run(config_path: str | Path | None = None) -> dict[str, object]:
    if config_path is None:
        return load_default_workflow()
    return load_workflow(config_path)


def _print_run_banner(workflow: dict[str, object]) -> None:
    output_path = workflow["config"]["output"]
    print(f"{'='*50}")
    print(f"  EMISSIONS PROCESSING - {str(workflow['area']).upper()} REGION")
    print(f"  Scenario: {workflow['scenario']}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")


def _extract_activities_config(emfac: dict[str, object], *, activities_output_root: Path) -> dict[str, object] | None:
    nested = emfac.get("activities")
    if isinstance(nested, dict):
        activities = deepcopy(nested)
    else:
        candidate_keys = ("region_label", "calendar_year", "model_year_groups", "inputs")
        if not all(key in emfac for key in candidate_keys):
            return None
        activities = {key: deepcopy(emfac[key]) for key in candidate_keys}
    activities["outputs"] = str(activities_output_root)
    return activities


def _bootstrap_emfac_outputs_if_needed(workflow: dict[str, object]) -> dict[str, object]:
    emfac = workflow["config"]["emfac"]
    required_outputs = {
        "rates_file": Path(str(emfac["rates_file"])),
        "activity_file": Path(str(emfac["activity_file"])),
        "fleet_file": Path(str(emfac["fleet_file"])),
    }
    missing = {label: path for label, path in required_outputs.items() if not path.exists()}
    if not missing:
        return workflow

    activities_output_root = Path(str(emfac.get("outputs", Path(str(workflow["config"]["output"])) / "emfac")))
    activities = _extract_activities_config(emfac, activities_output_root=activities_output_root)
    if activities is None:
        missing_paths = ", ".join(str(path) for path in missing.values())
        raise FileNotFoundError(
            "Fleet requires EMFAC rates/activity/fleet outputs before Step 5. "
            f"Missing: {missing_paths}. Configure fleet.emfac with region_label, calendar_year, "
            "model_year_groups, and inputs so it can bootstrap activities automatically."
        )

    print("Bootstrapping EMFAC activities because required fleet inputs are missing:")
    for path in missing.values():
        print(f"  missing: {path}")
    activities_workflow = load_activities_workflow_from_data(
        activities,
        source_label="<fleet.emfac>",
    )
    activities_workflow = run_activities_workflow(activities_workflow)

    emfac["rates_file"] = str(Path(str(activities_workflow["paths"]["final_output"])).resolve())
    emfac["activity_file"] = str(Path(str(activities_workflow["paths"]["final_activity_output"])).resolve())
    emfac["fleet_file"] = str(Path(str(activities_workflow["paths"]["final_fleet_output"])).resolve())
    return workflow


def run_all_steps(config_path: str | Path | None = None) -> None:
    """Run the active fleet workflow steps."""
    workflow = _configure_run(config_path)
    workflow = _bootstrap_emfac_outputs_if_needed(workflow)
    _print_run_banner(workflow)
    workflow = run_step1(workflow)
    workflow = run_step2(workflow)
    workflow = run_step3(workflow)
    workflow = run_step4(workflow)
    workflow = run_step5(workflow)
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
