from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from ..manifest.file_ops import resolve_path
from .common import optional_local_path
from .common import required_local_path
from .common import stage_local_input
from .common import stage_optional_input


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    inputs,
    config_path: Path,
    beam_network_source: str,
    osm_source: str,
    inmap_grid_source: str,
) -> dict[str, Any]:
    activity_totals_file = resolve_path(inputs.activity_totals_file, config_path)
    staged_network = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="network",
        source_path=beam_network_source,
        relative_target=f"network/{Path(beam_network_source).name}",
    )
    staged_osm = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="osm_network",
        source_path=required_local_path(osm_source, "inputs.osm_links"),
        relative_target=f"osm/{Path(osm_source).name}",
    )
    staged_inmap_grid = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="inmap_grid",
        source_path=required_local_path(inmap_grid_source, "processing.grid.inmap_grid_path"),
        relative_target=f"inmap_grid/{Path(inmap_grid_source).name}",
    )
    staged_activity_totals = stage_optional_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="activity_totals_file",
        source_path=optional_local_path(activity_totals_file),
        relative_target=(
            f"activity/{Path(activity_totals_file).name}"
            if activity_totals_file
            else "activity/activity_totals.parquet"
        ),
    )

    staged_isrm = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm",
        source_path=required_local_path(resolve_path(inputs.isrm_zarr, config_path), "inputs.isrm_zarr"),
        relative_target="isrm",
    )
    staged_isrm_nox_to_no2_matrix_npz = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm_nox_to_no2_matrix_npz",
        source_path=required_local_path(
            resolve_path(inputs.isrm_nox_to_no2_matrix_npz, config_path),
            "inputs.isrm_nox_to_no2_matrix_npz",
        ),
        relative_target=f"dispersion/{Path(inputs.isrm_nox_to_no2_matrix_npz).name}",
    )
    return {
        "staged_network": staged_network,
        "staged_osm": staged_osm,
        "staged_inmap_grid": staged_inmap_grid,
        "staged_activity_totals": staged_activity_totals,
        "staged_isrm": staged_isrm,
        "staged_isrm_nox_to_no2_matrix_npz": staged_isrm_nox_to_no2_matrix_npz,
    }
