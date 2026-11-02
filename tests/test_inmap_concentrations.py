from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _assemble_concentration_results
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _build_beam_inmap_concentrations_gdf
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _read_sparse_transfer_matrix_npz
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _prepare_grid_emissions
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _study_area_receptor_cells
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _isrm_source_positions, _align_emissions_to_isrm_sources
from impacts.pipeline.workflow.step2_compute_inmap_concentrations import _matrix_response


def test_compact_isrm_retains_global_source_ids():
    sr = {'SOA': np.zeros((1, 2, 5)), 'source_ids': np.array([202, 101])}
    emis = pd.DataFrame({'cell': [101, 202, 303], 'PM25': [1., 2., 0.]})
    ids, values = _align_emissions_to_isrm_sources(emis, sr, 'cell')
    assert ids.tolist() == [101, 202]
    assert values.PM25.tolist() == [1., 2.]
    assert _isrm_source_positions(sr, ids).tolist() == [1, 0]
    emis.loc[2, 'PM25'] = 3.
    with pytest.raises(ValueError, match='missing nonzero-emission'):
        _align_emissions_to_isrm_sources(emis, sr, 'cell')


def test_dense_isrm_source_positions():
    sr = {'SOA': np.zeros((1, 4, 5))}
    assert _isrm_source_positions(sr, np.array([3, 1])).tolist() == [3, 1]
    with pytest.raises(ValueError, match='lacks requested'):
        _isrm_source_positions(sr, np.array([4]))


def test_compact_isrm_response_uses_global_ids_and_mask(tmp_path):
    import zarr
    sr = zarr.open_group(str(tmp_path/'compact.zarr'), mode='w')
    sr.create_dataset('SOA', shape=(1,2,5), data=np.zeros((1,2,5)))
    sr.create_dataset('source_ids', shape=(2,), data=np.array([202,101]))
    sr.create_dataset('PrimaryPM25', shape=(1,2,5), data=np.array([[[1.,2.,3.,4.,5.],[10.,20.,30.,40.,50.]]]))
    args = (sr,'PrimaryPM25',np.array([101,202]),np.array([2.,3.]),np.array([1,3]))
    np.testing.assert_allclose(_matrix_response(*args),[46.,92.])
    np.testing.assert_allclose(_matrix_response(*args,near_source_mask=np.array([[True,False],[False,True]])),[6.,80.])


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


def test_study_area_receptors_are_not_limited_to_emissions_cells() -> None:
    grid = gpd.GeoDataFrame(
        {"inmap_cell_id": [4, 1, 3]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:26910",
    )

    assert _study_area_receptor_cells(grid, receptor_dim=5).tolist() == [1, 3, 4]


def test_prepare_grid_emissions_accepts_legacy_pm2_5_column(tmp_path: Path) -> None:
    emissions = pd.DataFrame(
        {
            "inmap_cell_id": [1, 1, 2],
            "tons_per_year_PM2_5_inmap_allocated": [0.25, 0.75, 2.0],
        }
    )

    prepared, available = _prepare_grid_emissions(
        emissions,
        source_id_col="inmap_cell_id",
        working_dir=tmp_path,
        pollutants=["PM25"],
    )

    by_id = prepared.set_index("inmap_cell_id")
    assert by_id.loc[1, "tons_per_year_PM25_inmap_allocated"] == 1.0
    assert by_id.loc[2, "tons_per_year_PM25_inmap_allocated"] == 2.0
    assert "PM25" in available


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


def test_sparse_no2_npz_loader_uses_progress_helper(monkeypatch, tmp_path: Path) -> None:
    import numpy as np
    import impacts.pipeline.workflow.step2_compute_inmap_concentrations as step2_module

    calls = []
    updates = []

    class _FakeProgress:
        disable = True

        def update(self, value):
            updates.append(value)

        def close(self):
            updates.append("closed")

    def _fake_make_progress(*args, **kwargs):
        calls.append((args, kwargs))
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
    monkeypatch.setattr(step2_module, "make_progress", _fake_make_progress)

    matrix = _read_sparse_transfer_matrix_npz(str(npz_path))

    assert matrix["source_dim"] == 4
    assert calls
    assert calls[0][0][0] == "Loading NOx->NO2 npz"
    assert calls[0][1]["total"] == 5
    assert updates == [1, 1, 1, 1, 1, "closed"]
