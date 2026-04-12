"""Main entrypoint for the step-based fleet workflow."""

from pathlib import Path
import sys

from impacts.fleet.config import load_default_workflow
from impacts.fleet.config import load_workflow
from impacts.fleet.step1_build_vehicle_types import run_step1
from impacts.fleet.step2_prepare_passenger_fleet import run_step2


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
    """Run Step 1 of the fleet workflow."""
    workflow = _configure_run(config_path)
    _print_run_banner(workflow)
    workflow = run_step1(workflow)
    prepared_vehicle_types_file = workflow.get("built_vehicle_types_file", "")
    if prepared_vehicle_types_file:
        print(f"  Step 1 vehicle types file: {prepared_vehicle_types_file}")
    print("  DONE")


def main(config_path: str | Path | None = None) -> None:
    """CLI-friendly entrypoint for the fleet workflow."""
    run_all_steps(config_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
