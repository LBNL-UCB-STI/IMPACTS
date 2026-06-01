from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from impacts.pipeline.workflow import step3_compute_aermod_concentrations as aermod_step
from impacts.pipeline.workflow.step3_compute_aermod_concentrations import _attach_concentrations


def test_attach_concentrations_preserves_numeric_pollutants_and_boolean_support_masks() -> None:
    target_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11, 22],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    concentrations_df = pd.DataFrame(
        {
            "aermod_cell_id": [11],
            "PrimaryPM25": [0.25],
            "BC": [0.5],
            "NO2": [3.0],
            "has_aermod_primarypm25": [True],
            "has_aermod_bc": [True],
            "has_aermod_no2": [True],
        }
    )

    result = _attach_concentrations(
        target_grid=target_grid,
        concentrations_df=concentrations_df,
        target_id_col="aermod_cell_id",
    ).drop(columns="geometry")

    by_id = result.set_index("aermod_cell_id")
    assert by_id.loc[11, "PrimaryPM25"] == 0.25
    assert by_id.loc[11, "BC"] == 0.5
    assert by_id.loc[11, "NO2"] == 3.0
    assert bool(by_id.loc[11, "has_aermod_primarypm25"]) is True
    assert bool(by_id.loc[11, "has_aermod_bc"]) is True
    assert bool(by_id.loc[11, "has_aermod_no2"]) is True
    assert by_id.loc[22, "PrimaryPM25"] == 0.0
    assert by_id.loc[22, "BC"] == 0.0
    assert by_id.loc[22, "NO2"] == 0.0
    assert bool(by_id.loc[22, "has_aermod_primarypm25"]) is False
    assert bool(by_id.loc[22, "has_aermod_bc"]) is False
    assert bool(by_id.loc[22, "has_aermod_no2"]) is False


def test_apply_kernels_ignores_fft_support_roundoff(monkeypatch) -> None:
    def fake_fftconvolve(_grid, kernel, mode):
        assert mode == "full"
        if np.isclose(float(np.max(kernel)), 2.0):
            return np.array([[1e-9], [20.0], [1e-9]], dtype=float)
        return np.array([[1e-9], [1.0], [1e-9]], dtype=float)

    monkeypatch.setattr(aermod_step, "fftconvolve", fake_fftconvolve)

    source_df = pd.DataFrame(
        {
            "pattern_key": ["pattern-a"],
            "source_ix": [1],
            "source_iy": [0],
            "tons_per_year_PM2_5_aermod_allocated": [10.0],
        }
    )
    target_index = pd.DataFrame(
        {
            "aermod_cell_id": [101, 102, 103],
            "target_ix": [0, 1, 2],
            "target_iy": [0, 0, 0],
        }
    )
    kernel_library = {
        "pattern-a": {
            "dix": np.array([0], dtype=np.int32),
            "diy": np.array([0], dtype=np.int32),
            "response_per_ton": np.array([2.0], dtype=float),
        }
    }

    result = aermod_step._apply_kernels(
        source_df=source_df,
        target_index=target_index,
        target_id_col="aermod_cell_id",
        emissions_cols=["tons_per_year_PM2_5_aermod_allocated"],
        kernel_library=kernel_library,
    )

    assert result["PrimaryPM25"].tolist() == [0.0, 20.0, 0.0]
    assert result["TotalPM25"].tolist() == [0.0, 20.0, 0.0]
    assert result["has_aermod_primarypm25"].tolist() == [False, True, False]
