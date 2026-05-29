from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

from ...common import assign_grid_cells_to_zones
from ...common import constrain_grid_to_network
from ...common import ensure_grid_cell_id
from ...common import generate_fishnet_from_bounds
from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_vector
from ...common import resolve_required_manifest_input
from ...common import stage_county_boundaries
from ...manifest.file_ops import file_entry

logger = logging.getLogger(__name__)


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    settings,
    inmap,
    local_output_epsg: int,
) -> dict[str, str]:
    log_step_banner("Preprocess Step 2", "Prepare Grids", logger=logger)
    start_year = settings.run.start_year
    if start_year is None:
        raise ValueError("run.start_year is required for county boundary staging")

    staged_inmap_grid = None
    resolved_inmap_grid_id = None
    if settings.impacts.pipeline.inmap:
        staged_inmap_grid = resolve_required_manifest_input(manifest_inputs, key="inmap_grid")
        log_substep_banner("2.1", "ensure InMAP grid id", logger=logger)
        staged_inmap_grid, resolved_inmap_grid_id = ensure_grid_cell_id(
            staged_inmap_grid,
            "inmap_cell_id",
            source_col=inmap.grid_id,
            output_path=str((input_root / "inmap_grid.parquet").resolve()),
        )
        log_substep_banner("2.2", "constrain InMAP grid to network", logger=logger)
        staged_inmap_grid = constrain_grid_to_network(
            grid_path=staged_inmap_grid,
            network_path=resolve_required_manifest_input(manifest_inputs, key="network"),
            grid_id_col="inmap_cell_id",
            target_epsg=int(local_output_epsg),
            output_path=str((input_root / "inmap_network_subset.parquet").resolve()),
        )
        manifest_inputs["inmap_grid"]["staged_path"] = staged_inmap_grid

    staged_aermod_grid = None
    staged_aermod_full_grid = None
    resolved_aermod_grid_id = None
    if settings.impacts.pipeline.aermod:
        log_substep_banner("2.3", "generate AERMOD grid", logger=logger)
        staged_inmap = read_vector(staged_inmap_grid) if staged_inmap_grid else None
        if staged_inmap is None or staged_inmap.empty:
            raise ValueError("Preprocess Step 2 requires a network-constrained InMAP grid before generating the AERMOD grid.")
        grid_size_meters = float(settings.impacts.dispersions.aermod.grid_size_meters)
        staged_aermod_full_grid, resolved_aermod_grid_id = generate_fishnet_from_bounds(
            bounds=tuple(float(v) for v in staged_inmap.total_bounds),
            mask_gdf=staged_inmap,
            cell_size=grid_size_meters,
            target_path=str((input_root / f"aermod_{grid_size_meters:g}m_fishnet.parquet").resolve()),
            target_epsg=int(local_output_epsg),
            cell_id_col="aermod_cell_id",
        )
        staged_aermod_full_grid = assign_grid_cells_to_zones(
            grid_path=staged_aermod_full_grid,
            zone_path=staged_inmap_grid,
            zone_id_col="inmap_cell_id",
            output_col="inmap_cell_id",
            target_epsg=int(local_output_epsg),
        )
        staged_aermod_grid = staged_aermod_full_grid
        manifest_inputs["aermod_grid"] = file_entry(
            kind="local",
            path=staged_aermod_full_grid,
            staged_path=staged_aermod_full_grid,
            optional=False,
        )
    geography = settings.shared.geography
    log_substep_banner("2.4", "stage county boundaries", logger=logger)
    staged_county_boundaries = stage_county_boundaries(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        state_fips=geography.fips.state,
        county_fips_codes=list(geography.fips.counties),
        year=int(start_year),
        area_name=settings.run.region,
        target_epsg=int(local_output_epsg),
    )
    logger.info("Preprocess Step 2 complete")
    return {
        "staged_inmap_grid": staged_inmap_grid,
        "staged_aermod_grid": staged_aermod_grid,
        "staged_aermod_full_grid": staged_aermod_full_grid,
        "staged_county_boundaries": staged_county_boundaries,
        "resolved_inmap_grid_id": resolved_inmap_grid_id,
        "resolved_aermod_grid_id": resolved_aermod_grid_id,
    }
