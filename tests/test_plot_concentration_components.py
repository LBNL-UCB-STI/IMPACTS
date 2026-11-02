import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box

from impacts.pipeline.postprocess.step4_plot_concentrations import (
    _add_pm25_component_columns,
    _add_local_nonlocal_component_columns,
    _clip_pm25_components_to_sfbay,
)


@pytest.mark.parametrize('pollutant', ['BC', 'NO2'])
def test_local_nonlocal_components_use_nonlocal_not_full_inmap(pollutant):
    source = pd.DataFrame({f'aermod_{pollutant}':[1.,2.],f'inmap_{pollutant}':[3.,4.],
                           f'inmap_only_{pollutant}':[10.,20.],f'power_{pollutant}':[100.,200.]})
    original = source.copy(deep=True)
    result = _add_local_nonlocal_component_columns(source,pollutant)
    assert result[f'{pollutant}_Local_AERMOD'].tolist()==[1.,2.]
    assert result[f'{pollutant}_Nonlocal_InMAP'].tolist()==[3.,4.]
    assert result[f'{pollutant}_TwoComponent_Total'].tolist()==[4.,6.]
    pd.testing.assert_frame_equal(source,original)


def test_local_nonlocal_components_reject_missing_or_invalid_data():
    with pytest.raises(ValueError,match='require concentration columns'):
        _add_local_nonlocal_component_columns(pd.DataFrame({'aermod_BC':[1.]}),'BC')
    with pytest.raises(ValueError,match='Invalid'):
        _add_local_nonlocal_component_columns(pd.DataFrame({'aermod_BC':[float('nan')],'inmap_BC':[1.]}),'BC')
    with pytest.raises(ValueError,match='Unsupported'):
        _add_local_nonlocal_component_columns(pd.DataFrame(),'PM25')


def test_pm25_component_columns_sum_local_nonlocal_and_secondary():
    source = pd.DataFrame(
        {
            "aermod_PrimaryPM25_scaled": [1.0, 2.0],
            "inmap_PrimaryPM25": [3.0, 4.0],
            "inmap_SecondaryPM25": [5.0, 6.0],
        }
    )

    result = _add_pm25_component_columns(source)

    assert result["PM25_Local_AERMOD"].tolist() == [1.0, 2.0]
    assert result["PM25_Nonlocal_InMAP"].tolist() == [3.0, 4.0]
    assert result["PM25_Secondary_InMAP"].tolist() == [5.0, 6.0]
    assert result["PM25_ThreeComponent_Total"].tolist() == [9.0, 12.0]


def test_pm25_component_columns_require_all_three_inputs():
    with pytest.raises(ValueError, match="inmap_SecondaryPM25"):
        _add_pm25_component_columns(
            pd.DataFrame(
                {
                    "aermod_PrimaryPM25_scaled": [1.0],
                    "inmap_PrimaryPM25": [2.0],
                }
            )
        )


def test_component_maps_are_clipped_to_nine_county_sfbay(tmp_path):
    concentrations = gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:26910",
    )
    network = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1).boundary, box(10, 10, 11, 11).boundary],
        crs="EPSG:26910",
    )
    region = gpd.GeoDataFrame(
        {"County": ["Alameda", "Los Angeles"]},
        geometry=[box(-1, -1, 2, 2), box(9, 9, 12, 12)],
        crs="EPSG:26910",
    )
    region_path = tmp_path / "regions.parquet"
    region.to_parquet(region_path)

    clipped_concentrations, clipped_network = _clip_pm25_components_to_sfbay(
        concentrations, network, str(region_path)
    )

    assert clipped_concentrations["value"].tolist() == [1.0]
    assert len(clipped_network) == 1


def test_component_maps_accept_tiger_name_column(tmp_path):
    concentrations = gpd.GeoDataFrame(
        {"value": [1.0, 2.0]},
        geometry=[box(0, 0, 1, 1), box(10, 10, 11, 11)],
        crs="EPSG:26910",
    )
    network = gpd.GeoDataFrame(
        geometry=[box(0, 0, 1, 1).boundary, box(10, 10, 11, 11).boundary],
        crs="EPSG:26910",
    )
    region = gpd.GeoDataFrame(
        {"NAME": ["Alameda", "Los Angeles"]},
        geometry=[box(-1, -1, 2, 2), box(9, 9, 12, 12)],
        crs="EPSG:26910",
    )
    region_path = tmp_path / "regions.geojson"
    region.to_file(region_path, driver="GeoJSON")

    clipped_concentrations, clipped_network = _clip_pm25_components_to_sfbay(
        concentrations, network, str(region_path)
    )

    assert clipped_concentrations["value"].tolist() == [1.0]
    assert len(clipped_network) == 1
