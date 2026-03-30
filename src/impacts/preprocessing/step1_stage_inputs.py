from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import optional_local_path
from ..common import required_local_path
from ..common import stage_local_input
from ..common import stage_optional_input
from ..manifest.file_ops import resolve_path

logger = logging.getLogger(__name__)


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    impacts,
    config_path: Path,
    beam_network_source: str,
    osm_source: str,
    inmap_grid_source: str,
) -> dict[str, Any]:
    log_step_banner("Preprocess Step 1", "Stage Inputs", logger=logger)
    emissions = impacts.emissions
    inmap = impacts.dispersions.inmap
    activity_totals_file = resolve_path(emissions.activity_totals_file, config_path)
    log_substep_banner("1.1", "stage BEAM network", logger=logger)
    staged_network = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="network",
        source_path=beam_network_source,
        relative_target=f"network/{Path(beam_network_source).name}",
    )
    log_substep_banner("1.2", "stage OSM network", logger=logger)
    staged_osm = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="osm_network",
        source_path=required_local_path(osm_source, "impacts.emissions.osm_network_folder"),
        relative_target=f"osm/{Path(osm_source).name}",
    )
    log_substep_banner("1.3", "stage InMAP grid", logger=logger)
    staged_inmap_grid = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="inmap_grid",
        source_path=required_local_path(inmap_grid_source, "impacts.dispersions.inmap.grid_path"),
        relative_target=f"inmap_grid/{Path(inmap_grid_source).name}",
    )
    log_substep_banner("1.4", "stage activity totals", logger=logger)
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

    log_substep_banner("1.5", "stage ISRM store", logger=logger)
    staged_isrm = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm",
        source_path=required_local_path(resolve_path(inmap.isrm_zarr, config_path), "impacts.dispersions.inmap.isrm_zarr"),
        relative_target="isrm",
    )
    log_substep_banner("1.6", "stage NOx to NO2 matrix", logger=logger)
    staged_isrm_nox_to_no2_matrix_npz = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="isrm_nox_to_no2_matrix_npz",
        source_path=required_local_path(
            resolve_path(inmap.isrm_nox_to_no2_matrix_npz, config_path),
            "impacts.dispersions.inmap.isrm_nox_to_no2_matrix_npz",
        ),
        relative_target=f"dispersion/{Path(inmap.isrm_nox_to_no2_matrix_npz).name}",
    )
    logger.info("Preprocess Step 1 complete")
    return {
        "staged_network": staged_network,
        "staged_osm": staged_osm,
        "staged_inmap_grid": staged_inmap_grid,
        "staged_activity_totals": staged_activity_totals,
        "staged_isrm": staged_isrm,
        "staged_isrm_nox_to_no2_matrix_npz": staged_isrm_nox_to_no2_matrix_npz,
    }
