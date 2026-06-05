from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _assemble_concentration_results
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _build_beam_inmap_concentrations_gdf
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _read_sparse_transfer_matrix_npz


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


def test_assemble_concentration_results_does_not_rescale_custom_no2_response() -> None:
    result = _assemble_concentration_results(
        receptor_cells=pd.Series([101, 202]).to_numpy(),
        factor=10.0,
        arrays={
            "PrimaryPM25": pd.Series([1.0, 2.0]).to_numpy(),
            "NO2": pd.Series([3.0, 4.0]).to_numpy(),
        },
        source_id_col="inmap_cell_id",
        already_scaled_outputs={"NO2"},
    )

    assert result["PrimaryPM25"].tolist() == [10.0, 20.0]
    assert result["NO2"].tolist() == [3.0, 4.0]


def test_assemble_concentration_results_scales_no2_when_requested() -> None:
    result = _assemble_concentration_results(
        receptor_cells=pd.Series([101, 202]).to_numpy(),
        factor=10.0,
        arrays={
            "PrimaryPM25": pd.Series([1.0, 2.0]).to_numpy(),
            "NO2": pd.Series([3.0, 4.0]).to_numpy(),
        },
        source_id_col="inmap_cell_id",
    )

    assert result["PrimaryPM25"].tolist() == [10.0, 20.0]
    assert result["NO2"].tolist() == [30.0, 40.0]


def test_sparse_no2_npz_loader_disables_tqdm_for_batch_logs(monkeypatch, tmp_path: Path) -> None:
    import numpy as np
    import impacts.pipeline.workflow.step2_compute_inmap_concentrations as step2_module

    calls = []

    class _FakeProgress:
        def update(self, value):
            pass

        def close(self):
            pass

    def _fake_tqdm(*args, **kwargs):
        calls.append(kwargs)
        return _FakeProgress()

    npz_path = tmp_path / "matrix.npz"
    np.savez(
        npz_path,
        source_ids=np.array([1], dtype=np.int64),
        receptor_ids=np.array([2], dtype=np.int64),
        values=np.array([3.0], dtype=np.float64),
        source_dim=np.array(4),
        receptor_dim=np.array(5),
    )
    monkeypatch.setattr(step2_module, "_running_with_real_terminal", lambda: False)
    monkeypatch.setattr(step2_module, "tqdm", _fake_tqdm)

    matrix = _read_sparse_transfer_matrix_npz(str(npz_path))

    assert matrix["source_dim"] == 4
    assert calls
    assert calls[0]["disable"] is True
