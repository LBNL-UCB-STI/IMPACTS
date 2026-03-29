from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from ..manifest.file_ops import file_entry
from .common import ensure_grid_cell_id
from .common import generate_fishnet_from_bounds
from .common import read_network_bounds
from .common import read_vector
from .common import required_local_path
from .common import stage_county_boundaries
from .common import stage_local_input


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    runtime_config,
    processing,
    grid,
    inmap_grid_source: str,
    aermod_grid_source: str | None,
    local_output_epsg: int,
) -> dict[str, str]:
    staged_inmap_grid = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="inmap_grid",
        source_path=required_local_path(inmap_grid_source, "processing.grid.inmap_grid_path"),
        relative_target=f"inmap_grid/{Path(inmap_grid_source).name}",
    )
    staged_inmap_grid, resolved_inmap_grid_id = ensure_grid_cell_id(
        staged_inmap_grid,
        "srm_cell_id",
        source_col=grid.inmap_grid_id,
    )
    manifest_inputs["inmap_grid"]["staged_path"] = staged_inmap_grid
    manifest_inputs["inmap_grid"]["path"] = inmap_grid_source

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
        path=aermod_grid_source or staged_aermod_grid,
        staged_path=staged_aermod_grid,
        optional=True,
    )
    geography = runtime_config.shared_context.geography
    staged_county_boundaries = stage_county_boundaries(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        state_fips=geography.fips.state,
        county_fips_codes=list(geography.fips.counties),
        year=int(runtime_config.shared_context.start_year or 2023),
        area_name=runtime_config.shared_context.region or processing.county_area_name,
        target_epsg=int(local_output_epsg),
    )
    return {
        "staged_inmap_grid": staged_inmap_grid,
        "staged_aermod_grid": staged_aermod_grid,
        "staged_county_boundaries": staged_county_boundaries,
        "resolved_inmap_grid_id": resolved_inmap_grid_id,
        "resolved_aermod_grid_id": resolved_aermod_grid_id,
    }
