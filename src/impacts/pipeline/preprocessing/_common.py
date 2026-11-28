from __future__ import annotations

import geopandas as gpd

from ...common import read_vector
from ...manifest.schema import PipelineConfig


def load_aermod_grid(pipeline: PipelineConfig) -> gpd.GeoDataFrame:
    grid = read_vector(str(pipeline.aermod_grid_path))
    if grid.crs is not None:
        grid = grid.to_crs(epsg=int(pipeline.output_epsg))

    grid_id = str(pipeline.aermod_grid_id)
    if grid_id not in grid.columns:
        raise ValueError(
            f"AERMOD grid missing expected id column '{grid_id}'. Available: {list(grid.columns)}"
        )
    return grid.rename(columns={grid_id: "aermod_cell_id"})
