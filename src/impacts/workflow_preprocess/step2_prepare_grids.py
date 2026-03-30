from __future__ import annotations

from typing import Any
from typing import Dict

from ..manifest.file_ops import file_entry
from .common import ensure_grid_cell_id
from .common import generate_fishnet_from_bounds
from .common import read_network_bounds
from .common import read_vector
from .common import stage_county_boundaries


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    runtime_config,
    grid,
    local_output_epsg: int,
) -> dict[str, str]:
    start_year = runtime_config.run.start_year
    if start_year is None:
        raise ValueError("shared.start_year is required for county boundary staging")

    staged_inmap_grid = manifest_inputs["inmap_grid"]["staged_path"]
    staged_inmap_grid, resolved_inmap_grid_id = ensure_grid_cell_id(
        staged_inmap_grid,
        "srm_cell_id",
        source_col=grid.inmap_grid_id,
    )
    manifest_inputs["inmap_grid"]["staged_path"] = staged_inmap_grid

    network_bounds = read_network_bounds(manifest_inputs["network"]["staged_path"])
    staged_inmap = read_vector(staged_inmap_grid)
    staged_aermod_grid, resolved_aermod_grid_id = generate_fishnet_from_bounds(
        bounds=network_bounds,
        mask_gdf=staged_inmap,
        cell_size=100.0,
        target_path=str((input_root / "aermod_grid" / "aermod_100m_fishnet.parquet").resolve()),
        target_epsg=int(local_output_epsg),
        cell_id_col="srv_cell_id",
    )
    manifest_inputs["aermod_grid"] = file_entry(
        kind="local",
        path=staged_aermod_grid,
        staged_path=staged_aermod_grid,
        optional=False,
    )
    geography = runtime_config.run.geography
    staged_county_boundaries = stage_county_boundaries(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        state_fips=geography.fips.state,
        county_fips_codes=list(geography.fips.counties),
        year=int(start_year),
        area_name=runtime_config.run.region,
        target_epsg=int(local_output_epsg),
    )
    return {
        "staged_inmap_grid": staged_inmap_grid,
        "staged_aermod_grid": staged_aermod_grid,
        "staged_county_boundaries": staged_county_boundaries,
        "resolved_inmap_grid_id": resolved_inmap_grid_id,
        "resolved_aermod_grid_id": resolved_aermod_grid_id,
    }
