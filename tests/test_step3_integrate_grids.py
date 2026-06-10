from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Polygon

from impacts.pipeline.preprocessing import step3_integrate_grids
from impacts.pipeline.preprocessing.step3_integrate_grids import _build_synthetic_beam_links
from impacts.pipeline.preprocessing.step3_integrate_grids import _ensure_county_mass_conservation


def _county_frame() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "COUNTYFP": ["001", "013"],
            "NAME": ["Alpha", "Bravo"],
            "geometry": [
                Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
                Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )


def test_log_safe_osm_chordify_progress_forces_fixed_width_for_logs(monkeypatch) -> None:
    import osm_chordify.osm.intersect as osm_intersect

    calls = []

    def _fake_tqdm(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setenv("IMPACTS_OSM_CHORDIFY_TQDM_NCOLS", "77")
    monkeypatch.setattr(osm_intersect, "tqdm", _fake_tqdm)

    with step3_integrate_grids._log_safe_osm_chordify_progress():
        osm_intersect.tqdm(total=1, dynamic_ncols=True)

    assert calls[0]["dynamic_ncols"] is False
    assert calls[0]["ncols"] == 77
    assert osm_intersect.tqdm is _fake_tqdm


def test_ensure_county_mass_conservation_fills_zero_match_links() -> None:
    source = gpd.GeoDataFrame(
        {
            "linkId": [1],
            "geometry": [LineString([(1, 1), (2, 2)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    county = gpd.GeoDataFrame(
        {
            "linkId": [1],
            "county_COUNTYFP": [pd.NA],
            "county_proportion": [0.0],
            "county_link_length_m": [0.0],
            "geometry": [LineString([(1, 1), (2, 2)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    result = _ensure_county_mass_conservation(
        source_links=source,
        county_intersection=county,
        county_gdf=_county_frame(),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row["county_COUNTYFP"] == "001"
    assert row["county_proportion"] == 1.0
    assert row["county_link_length_m"] > 0.0


def test_ensure_county_mass_conservation_adds_partial_remainder_row() -> None:
    source = gpd.GeoDataFrame(
        {
            "linkId": [2],
            "geometry": [LineString([(1, 1), (5, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    full_length = float(source.geometry.length.iloc[0])
    county = gpd.GeoDataFrame(
        {
            "linkId": [2],
            "county_COUNTYFP": ["001"],
            "county_proportion": [0.4],
            "county_link_length_m": [full_length * 0.4],
            "geometry": [LineString([(1, 1), (5, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )

    result = _ensure_county_mass_conservation(
        source_links=source,
        county_intersection=county,
        county_gdf=_county_frame(),
    )

    assert len(result) == 2
    assert result["county_COUNTYFP"].tolist() == ["001", "001"]
    assert result["county_proportion"].sum() == 1.0
    remainder = result.loc[result["county_proportion"] < 1.0].sort_values(
        "county_proportion"
    ).iloc[0]
    assert remainder["county_proportion"] == 0.4
    synthetic = result.loc[result["county_proportion"] > 0.4].iloc[0]
    assert synthetic["county_proportion"] == 0.6


def test_build_synthetic_beam_links_appends_null_origin_car_links(tmp_path) -> None:
    mapped = gpd.GeoDataFrame(
        {
            "linkId": [10],
            "linkLength": [5.0],
            "linkModes": ["car"],
            "attributeOrigId": [100.0],
            "attributeOrigType": ["motorway"],
            "fromLocationX": [0.0],
            "fromLocationY": [0.0],
            "toLocationX": [5.0],
            "toLocationY": [0.0],
            "osm_id": [100],
            "name": ["a"],
            "highway": ["motorway"],
            "waterway": [pd.NA],
            "aerialway": [pd.NA],
            "barrier": [pd.NA],
            "man_made": [pd.NA],
            "railway": [pd.NA],
            "z_order": [0],
            "other_tags": [pd.NA],
            "geometry": [LineString([(0, 0), (5, 0)])],
            "edge_id": ["10"],
            "edge_length": [5.0],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    network = pd.DataFrame(
        {
            "linkId": [10, 20, 21],
            "linkLength": [5.0, 3.0, 4.0],
            "linkModes": ["car", "car;walk;bike", "walk"],
            "attributeOrigId": [100.0, pd.NA, pd.NA],
            "attributeOrigType": ["motorway", pd.NA, pd.NA],
            "fromLocationX": [0.0, 1.0, 2.0],
            "fromLocationY": [0.0, 1.0, 2.0],
            "toLocationX": [5.0, 4.0, 5.0],
            "toLocationY": [0.0, 1.0, 2.0],
        }
    )
    network_path = tmp_path / "network.csv.gz"
    network.to_csv(network_path, index=False, compression="gzip")

    result = _build_synthetic_beam_links(
        network_path=str(network_path),
        mapped_network=mapped,
        output_epsg=26910,
    )

    assert sorted(result["linkId"].tolist()) == [10, 20]
    synthetic = result.loc[result["linkId"] == 20].iloc[0]
    assert synthetic["edge_id"] == "beam_synthetic_20"
    assert synthetic["attributeOrigId"] is pd.NA or pd.isna(synthetic["attributeOrigId"])
    assert list(synthetic.geometry.coords) == [(1.0, 1.0), (4.0, 1.0)]
