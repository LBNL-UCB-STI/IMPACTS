from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon

from impacts.pipeline.workflow.power_generation import build_beam_hourly_charging_demand
from impacts.pipeline.workflow.step4_prepare_exposure import _build_full_exposure_grid
from impacts.pipeline.workflow.step4_prepare_exposure import _prepare_power_generation_inputs
from test_prepare_exposure import _pipeline


def test_power_layer_zero_resolves_compact_source_ids(monkeypatch):
    from impacts.pipeline.workflow import power_generation as power
    # Stored row order differs from global source-cell order.
    sr = {'source_ids': np.array([5, 2])}
    for matrix, _ in power.POWER_COMPONENTS.values():
        sr[matrix] = np.array([[[10., 20.], [30., 40.]]])
    monkeypatch.setattr(power, '_load_isrm_store', lambda _: sr)
    monkeypatch.setattr(power, '_map_plants_to_source_cells', lambda plants, _: plants.assign(isrm_source_cell=[2, 5]))
    plants = pd.DataFrame({'LAT': [0., 0.], 'LON': [0., 0.],
                           **{c: [power.KG_PER_SHORT_TON, 2 * power.KG_PER_SHORT_TON] for c in power.POWER_COMPONENTS}})
    result = power.compute_power_generation_isrm(plants, isrm_url='test', receptor_cells=np.array([0, 1]), source_layer_index=0)
    np.testing.assert_allclose(result.power_PrimaryPM25, np.array([50., 80.]) * power.tons_per_year_to_ug_per_s)
    with pytest.raises(ValueError, match='unavailable'):
        power.compute_power_generation_isrm(plants, isrm_url='test', receptor_cells=np.array([0]), source_layer_index=1)


def test_power_matrix_refuses_to_omit_emitting_sources(monkeypatch):
    from impacts.pipeline.workflow import power_generation as power
    sr = {'source_ids': np.array([5])}
    monkeypatch.setattr(power, '_load_isrm_store', lambda _: sr)
    monkeypatch.setattr(power, '_map_plants_to_source_cells', lambda plants, _: plants.assign(isrm_source_cell=[2]))
    plants = pd.DataFrame({'LAT': [0.], 'LON': [0.], **{c: [1.] for c in power.POWER_COMPONENTS}})
    with pytest.raises(ValueError, match='missing 1 emitting source cells'):
        power.compute_power_generation_isrm(plants, isrm_url='test', receptor_cells=np.array([0]), source_layer_index=0)




def test_beam_events_build_electric_charging_profile_for_all_vehicle_types(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "0.events.csv.gz"
    pd.DataFrame(
        {
            "type": ["PathTraversal"] * 6 + ["ModeChoice"],
            "vehicle": ["car-1", "car-1", "truck-1", "truck-1", "bus-1", "body-1", None],
            "vehicleType": ["Car", "Car", "HDDT", "HDDT", "BUS-DEFAULT", "BODY-TYPE-DEFAULT", None],
            "mode": ["car", "car", "car", "car", "bus", "walk", "car"],
            "primaryFuelType": ["Gasoline", "Gasoline", "Diesel", "Diesel", "Diesel", "Food", None],
            "departureTime": [0, 4000, 0, 5000, 0, 0, None],
            "arrivalTime": [1000, 5000, 1000, 6000, 1000, 1000, None],
            "length": [10_000, 10_000, 20_000, 20_000, 5_000, 1_000, None],
        }
    ).to_csv(events_path, index=False, compression="gzip")

    demand, summary, sessions = build_beam_hourly_charging_demand(
        events_path,
        force_all_vehicle_types_electric=True,
        one_charge_share=1.0,
        two_charge_share=0.0,
        home_charging_share=1.0,
    )

    assert len(demand) == 8760
    assert summary["eligible_vehicles"] == 3
    assert summary["eligible_vehicle_types"] == {"BUS-DEFAULT": 1, "Car": 1, "HDDT": 1}
    assert summary["eligible_modes"] == ["bus", "car"]
    assert summary["force_all_vehicle_types_electric"] is True
    assert {
        "charging_demand_BUS_DEFAULT_MWh",
        "charging_demand_Car_MWh",
        "charging_demand_HDDT_MWh",
    }.issubset(demand)
    assert np.isclose(
        demand["charging_demand_MWh"].sum(),
        sessions["energy_needed_kwh"].sum() / 1000.0 * 365,
    )












def test_power_isrm_is_added_after_traffic_inmap_and_aermod(tmp_path: Path) -> None:
    full_grid = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "inmap_cell_id": [101],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        crs="EPSG:26910",
    )
    grid_path = tmp_path / "grid.parquet"
    full_grid.to_parquet(grid_path)
    traffic = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [101],
            "inmap_PrimaryPM25": [2.0],
            "inmap_SecondaryPM25": [3.0],
            "inmap_BC": [0.0],
            "inmap_NO2": [0.0],
            "geometry": full_grid.geometry,
        },
        crs=full_grid.crs,
    )
    aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [11],
            "aermod_PrimaryPM25": [4.0],
            "aermod_SecondaryPM25": [0.0],
            "has_aermod_primarypm25": [True],
            "geometry": full_grid.geometry,
        },
        crs=full_grid.crs,
    )
    power = _prepare_power_generation_inputs(
        gpd.GeoDataFrame(
            {
                "inmap_cell_id": [101],
                "power_PrimaryPM25": [5.0],
                "power_SecondaryPM25": [6.0],
                "power_TotalPM25": [11.0],
            }
        )
    )
    result = _build_full_exposure_grid(
        pipeline=_pipeline(tmp_path, grid_path),
        prepared_inmap=traffic,
        prepared_aermod=aermod,
        prepared_power=power,
        working_dir=tmp_path,
    )
    # Traffic imputation chooses AERMOD primary (4), then power (5) is added.
    assert result.loc[0, "PrimaryPM25"] == 9.0
    assert result.loc[0, "SecondaryPM25"] == 9.0
    assert result.loc[0, "TotalPM25"] == 18.0
    assert result.loc[0, "power_TotalPM25"] == 11.0
