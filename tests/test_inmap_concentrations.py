from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.workflow.step2_compute_inmap_concentrations import _build_beam_inmap_concentrations_gdf


def test_build_beam_inmap_concentrations_gdf_preserves_inmap_cell_id(tmp_path: Path) -> None:
    grid = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101, 202],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "grid.parquet"
    grid.to_parquet(grid_path, index=False)

    concentrations = pd.DataFrame(
        {
            "inmap_cell_id": [101, 202],
            "SOA": [1.0, 2.0],
            "PrimaryPM25": [0.1, 0.2],
            "TotalPM25": [0.3, 0.4],
            "BC": [0.0, 0.0],
            "NO2": [5.0, 6.0],
        }
    )

    result = _build_beam_inmap_concentrations_gdf(
        concentrations=concentrations,
        inmap_grid_path=str(grid_path),
        grid_id_col="inmap_cell_id",
        source_id_col="inmap_cell_id",
    )

    assert "inmap_cell_id" in result.columns
    assert result["inmap_cell_id"].tolist() == [101, 202]
