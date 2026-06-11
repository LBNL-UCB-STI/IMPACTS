from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
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


def test_attach_concentrations_rejects_duplicate_target_ids() -> None:
    target_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    concentrations_df = pd.DataFrame(
        {
            "aermod_cell_id": [11, 11],
            "PrimaryPM25": [0.25, 9.9],
        }
    )

    with pytest.raises(ValueError, match="one row per aermod_cell_id"):
        _attach_concentrations(
            target_grid=target_grid,
            concentrations_df=concentrations_df,
            target_id_col="aermod_cell_id",
        )


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
            "tons_per_year_PM25_aermod_allocated": [10.0],
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
        emissions_cols=["tons_per_year_PM25_aermod_allocated"],
        kernel_library=kernel_library,
    )

    assert result["PrimaryPM25"].tolist() == [0.0, 20.0, 0.0]
    assert result["TotalPM25"].tolist() == [0.0, 20.0, 0.0]
    assert result["has_aermod_primarypm25"].tolist() == [False, True, False]


def test_prepare_source_emissions_preserves_multiple_source_classes_per_cell(tmp_path: Path) -> None:
    emissions_gdf = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [101, 101],
            "source_temporal_class": ["CITYSTREET", "FREEWAY"],
            "source_release_height": [1.0, 3.5],
            "source_urban_class": [1000, 1000],
            "tons_per_year_PM25_aermod_allocated": [0.25, 0.75],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    result = aermod_step._prepare_source_emissions(
        emissions_gdf=emissions_gdf,
        source_id_col="aermod_cell_id",
        emissions_cols=["tons_per_year_PM25_aermod_allocated"],
        grid_size_meters=1.0,
        origin_x=0.5,
        origin_y=0.5,
        outputs_dir=tmp_path,
    ).sort_values("source_release_height").reset_index(drop=True)

    assert result["aermod_cell_id"].tolist() == [101, 101]
    assert result["source_temporal_class"].tolist() == ["CITYSTREET", "FREEWAY"]
    assert result["source_release_height"].tolist() == [1.0, 3.5]
    assert result["tons_per_year_PM25_aermod_allocated"].tolist() == [0.25, 0.75]


def test_prepare_source_emissions_rejects_missing_source_class_columns(tmp_path: Path) -> None:
    emissions_gdf = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [101],
            "tons_per_year_PM25_aermod_allocated": [0.25],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    with pytest.raises(ValueError, match="missing required source class columns"):
        aermod_step._prepare_source_emissions(
            emissions_gdf=emissions_gdf,
            source_id_col="aermod_cell_id",
            emissions_cols=["tons_per_year_PM25_aermod_allocated"],
            grid_size_meters=1.0,
            origin_x=0.5,
            origin_y=0.5,
            outputs_dir=tmp_path,
        )


def test_assign_source_pattern_keys_raises_when_no_fallback_available() -> None:
    source_df = pd.DataFrame(
        {
            "source_xm": [0.0],
            "source_ym": [0.0],
            "source_temporal_class": ["FREEWAY"],
            "source_release_height": [3.5],
            "source_urban_class": [1000],
        }
    )
    site_reference = pd.DataFrame({"DataSet_ID": ["site-a"], "site_xm": [0.0], "site_ym": [0.0]})

    with pytest.raises(ValueError, match="no available fallback"):
        aermod_step._assign_source_pattern_keys(
            source_df=source_df,
            site_reference=site_reference,
            available_pattern_keys={"site-a__1000__CITYSTREET__1"},
        )


def test_run_keeps_aermod_source_attrs_loaded_from_file(monkeypatch, tmp_path: Path) -> None:
    emissions_path = tmp_path / "emissions.parquet"
    gpd.GeoDataFrame(
        {
            "aermod_cell_id": [101],
            "source_temporal_class": ["FREEWAY"],
            "source_release_height": [3.5],
            "source_urban_class": [1000],
            "tons_per_year_PM25_aermod_allocated": [1.0],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    ).to_parquet(emissions_path, index=False)
    target_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [101],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    def fake_prepare_source_emissions(*, emissions_gdf, **_kwargs):
        assert "source_temporal_class" in emissions_gdf.columns
        assert "source_release_height" in emissions_gdf.columns
        assert "source_urban_class" in emissions_gdf.columns
        raise RuntimeError("stop after attribute check")

    monkeypatch.setattr(aermod_step, "_prepare_source_emissions", fake_prepare_source_emissions)
    pipeline = type(
        "Pipeline",
        (),
        {
            "aermod_enabled": True,
            "asrv_patterns_file": "patterns.parquet",
            "aermod_grid_path": "grid.parquet",
            "aermod_grid_id": "aermod_cell_id",
            "output_epsg": 26910,
            "grid_size_meters": 1.0,
            "pollutants": ["PM25"],
        },
    )()

    try:
        aermod_step.run(
            pipeline=pipeline,
            raw_dir=tmp_path,
            emissions_input_path=str(emissions_path),
            target_grid_gdf=target_grid,
        )
    except RuntimeError as exc:
        assert str(exc) == "stop after attribute check"
    else:
        raise AssertionError("expected fake source preparation to stop the run")
