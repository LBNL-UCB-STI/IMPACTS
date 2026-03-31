from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

from ..common import ensure_grid_cell_id
from ..common import generate_fishnet_from_bounds
from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_network_bounds
from ..common import read_vector
from ..common import stage_county_boundaries
from ..manifest.file_ops import file_entry

logger = logging.getLogger(__name__)


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    runtime_config,
    inmap,
    local_output_epsg: int,
) -> dict[str, str]:
    log_step_banner("Preprocess Step 2", "Prepare Grids", logger=logger)
    start_year = runtime_config.run.start_year
    if start_year is None:
        raise ValueError("run.start_year is required for county boundary staging")

    staged_inmap_grid = None
    resolved_inmap_grid_id = None
    if inmap.enabled:
        staged_inmap_grid = manifest_inputs["inmap_grid"]["staged_path"]
        log_substep_banner("2.1", "ensure InMAP grid id", logger=logger)
        staged_inmap_grid, resolved_inmap_grid_id = ensure_grid_cell_id(
            staged_inmap_grid,
            "srm_cell_id",
            source_col=inmap.grid_id,
        )
        manifest_inputs["inmap_grid"]["staged_path"] = staged_inmap_grid

    staged_aermod_grid = None
    resolved_aermod_grid_id = None
    if runtime_config.impacts.dispersions.aermod.enabled:
        log_substep_banner("2.2", "generate AERMOD grid", logger=logger)
        network_bounds = read_network_bounds(manifest_inputs["network"]["staged_path"])
        staged_inmap = read_vector(staged_inmap_grid) if staged_inmap_grid else None
        grid_size_meters = float(runtime_config.impacts.dispersions.aermod.grid_size_meters)
        staged_aermod_grid, resolved_aermod_grid_id = generate_fishnet_from_bounds(
            bounds=network_bounds,
            mask_gdf=staged_inmap,
            cell_size=grid_size_meters,
            target_path=str((input_root / "aermod_grid" / f"aermod_{grid_size_meters:g}m_fishnet.parquet").resolve()),
            target_epsg=int(local_output_epsg),
            cell_id_col="srv_cell_id",
        )
        manifest_inputs["aermod_grid"] = file_entry(
            kind="local",
            path=staged_aermod_grid,
            staged_path=staged_aermod_grid,
            optional=False,
        )
    geography = runtime_config.shared.geography
    log_substep_banner("2.3", "stage county boundaries", logger=logger)
    staged_county_boundaries = stage_county_boundaries(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        state_fips=geography.fips.state,
        county_fips_codes=list(geography.fips.counties),
        year=int(start_year),
        area_name=runtime_config.run.region,
        target_epsg=int(local_output_epsg),
    )
    logger.info("Preprocess Step 2 complete")
    return {
        "staged_inmap_grid": staged_inmap_grid,
        "staged_aermod_grid": staged_aermod_grid,
        "staged_county_boundaries": staged_county_boundaries,
        "resolved_inmap_grid_id": resolved_inmap_grid_id,
        "resolved_aermod_grid_id": resolved_aermod_grid_id,
    }
