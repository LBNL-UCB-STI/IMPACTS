from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from ..manifest.file_ops import resolve_path
from .common import is_remote_path
from .common import optional_local_path
from .common import stage_local_input
from .common import stage_optional_input


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    inputs,
    config_path: Path,
) -> dict[str, Any]:
    activity_totals_file = resolve_path(inputs.activity_totals_file, config_path)
    staged_activity_totals = None
    if activity_totals_file and not is_remote_path(activity_totals_file):
        staged_activity_totals = stage_local_input(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            key="activity_totals_file",
            source_path=activity_totals_file,
            relative_target=f"activity/{Path(activity_totals_file).name}",
            optional=True,
        )

    staged_isrm = stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm",
        source_path=resolve_path(inputs.isrm_zarr, config_path),
        relative_target="isrm",
    )
    staged_isrm_nox_to_no2_matrix_npz = stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm_nox_to_no2_matrix_npz",
        source_path=optional_local_path(resolve_path(inputs.isrm_nox_to_no2_matrix_npz, config_path)),
        relative_target=(
            f"dispersion/{Path(inputs.isrm_nox_to_no2_matrix_npz).name}" if inputs.isrm_nox_to_no2_matrix_npz else None
        ),
    )
    staged_persons = stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="persons",
        source_path=optional_local_path(resolve_path(inputs.persons_asim_out, config_path)),
        relative_target="population/persons.csv",
    )
    staged_households = stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="households",
        source_path=optional_local_path(resolve_path(inputs.households_asim_out, config_path)),
        relative_target="population/households.csv",
    )
    return {
        "staged_activity_totals": staged_activity_totals,
        "staged_isrm": staged_isrm,
        "staged_isrm_nox_to_no2_matrix_npz": staged_isrm_nox_to_no2_matrix_npz,
        "staged_persons": staged_persons,
        "staged_households": staged_households,
    }
