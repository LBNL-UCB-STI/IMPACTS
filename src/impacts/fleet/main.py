"""Main entrypoint for the step-based fleet workflow."""

from pathlib import Path
import sys

from impacts.fleet.config import load_default_workflow
from impacts.fleet.config import load_workflow
from impacts.fleet.step1_build_vehicle_types import run_step1
from impacts.fleet.step2_map_emfac_bus_bike import run_step2
from impacts.fleet.step3_map_emfac_atlas import run_step3
from impacts.fleet.step4_map_emfac_frism import run_step4
from impacts.fleet.step5_map_emfac_rates import run_step5


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


def run_all_steps(config_path: str | Path | None = None) -> None:
    """Run the active fleet workflow steps."""
    workflow = _configure_run(config_path)
    _print_run_banner(workflow)
    workflow = run_step1(workflow)
    workflow = run_step2(workflow)
    workflow = run_step3(workflow)
    workflow = run_step4(workflow)
    workflow = run_step5(workflow)
    prepared_vehicle_types_file = workflow.get("built_vehicle_types_file", "")
    if prepared_vehicle_types_file:
        print(f"  Passenger car vehicle types file: {prepared_vehicle_types_file}")
    print("  DONE")


def main(config_path: str | Path | None = None) -> None:
    """CLI-friendly entrypoint for the fleet workflow."""
    run_all_steps(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
