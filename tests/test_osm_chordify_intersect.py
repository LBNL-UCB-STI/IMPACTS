from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString
from shapely.geometry import box

from osm_chordify.osm.intersect import intersect_road_network_with_zones


def test_chained_intersections_do_not_duplicate_zone_prefixes():
    network = gpd.GeoDataFrame(
        {
            "linkId": [1],
            "linkLength": [2.0],
        },
        geometry=[LineString([(0.25, 0.5), (1.75, 0.5)])],
        crs="EPSG:26910",
    )
    inmap = gpd.GeoDataFrame(
        {
            "inmap_cell_id": [11, 12],
            "isrm": ["a", "b"],
        },
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        crs="EPSG:26910",
    )
    aermod = gpd.GeoDataFrame(
        {
            "aermod_cell_id": [101, 102],
            "inmap_cell_id": [11, 12],
        },
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        crs="EPSG:26910",
    )

    with_inmap = intersect_road_network_with_zones(
        network,
        26910,
        inmap,
        output_epsg=26910,
        zone_label="inmap",
    )
    assert "inmap_cell_id" in with_inmap.columns
    assert "inmap_inmap_cell_id" not in with_inmap.columns
    assert "inmap_isrm" in with_inmap.columns

    chained = intersect_road_network_with_zones(
        with_inmap,
        26910,
        aermod,
        output_epsg=26910,
        zone_label="aermod",
    )
    columns = set(chained.columns)
    assert "aermod_cell_id" in columns
    assert "aermod_aermod_cell_id" not in columns
    assert "edge_inmap_cell_id" in columns
    assert "edge_inmap_inmap_cell_id" not in columns
    assert "aermod_inmap_cell_id" in columns
