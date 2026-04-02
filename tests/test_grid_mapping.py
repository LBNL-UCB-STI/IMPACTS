from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from impacts.common import assign_grid_cells_to_zones


def test_assign_grid_cells_to_zones_resolves_boundary_cell_by_overlap(tmp_path: Path):
    grid_path = tmp_path / "aermod_grid.parquet"
    zone_path = tmp_path / "inmap_grid.parquet"

    grid = gpd.GeoDataFrame(
        {"aermod_cell_id": [0]},
        geometry=[box(0.5, 0.0, 1.5, 1.0)],
        crs="EPSG:26910",
    )
    zones = gpd.GeoDataFrame(
        {"inmap_cell_id": [3330, 3349]},
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        crs="EPSG:26910",
    )
    grid.to_parquet(grid_path, index=False)
    zones.to_parquet(zone_path, index=False)

    output_path = assign_grid_cells_to_zones(
        grid_path=str(grid_path),
        zone_path=str(zone_path),
        zone_id_col="inmap_cell_id",
        output_col="inmap_cell_id",
        target_epsg=26910,
    )

    resolved = gpd.read_parquet(output_path)
    assert len(resolved) == 1
    assert int(resolved.loc[0, "inmap_cell_id"]) == 3330
