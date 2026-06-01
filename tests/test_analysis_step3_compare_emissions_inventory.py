from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon

from impacts.analysis.step3_compare_emissions_inventory import run


def test_analysis_step3_writes_comparison_table_and_plots(tmp_path: Path) -> None:
    modeled = pd.DataFrame(
        {
            "county_COUNTYFP": ["001", "001", "013"],
            "vehicleTypeId": ["vt-1", "vt-2", "vt-3"],
            "process": ["RUNEX", "PMBW", "RUNEX"],
            "tons_per_year_PM2_5_county_allocated": [1.0, 2.0, 4.0],
            "tons_per_year_NOx_county_allocated": [3.0, 4.0, 5.0],
            "tons_per_year_BC_county_allocated": [0.5, 0.25, 0.75],
        }
    )
    modeled_path = tmp_path / "beam_emissions_by_county_process.parquet"
    modeled.to_parquet(modeled_path, index=False)

    inventory = pd.DataFrame(
        {
            "county": ["Alameda", "Contra Costa"],
            "pm25_runex_short_tons_per_year": [2.0, 4.0],
            "pm25_idlex_short_tons_per_year": [0.25, 0.75],
            "pm25_strex_short_tons_per_year": [0.1, 0.2],
            "pm25_pmbw_short_tons_per_year": [1.0, 0.5],
            "pm25_pmtw_short_tons_per_year": [0.4, 0.3],
            "pm25_diurn_short_tons_per_year": [99.0, 99.0],
            "pm25_prdust_short_tons_per_year": [5.0, 6.0],
            "nox_runex_short_tons_per_year": [7.0, 6.0],
            "bc_runex_short_tons_per_year": [0.75, 0.5],
            "bch_runex_short_tons_per_year": [99.0, 99.0],
        }
    )
    inventory_path = tmp_path / "sf-emfac-2018-emissions-inventory-with-activity.parquet"
    inventory.to_parquet(inventory_path, index=False)

    county_boundaries = gpd.GeoDataFrame(
        {
            "COUNTYFP": ["001", "013"],
            "NAME": ["Alameda", "Contra Costa"],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    county_boundaries_path = tmp_path / "county_boundaries.gpkg"
    county_boundaries.to_file(county_boundaries_path, driver="GPKG")

    output_dir = tmp_path / "analysis"
    outputs = run(
        modeled_emissions_path=str(modeled_path),
        inventory_path=str(inventory_path),
        county_boundaries_path=str(county_boundaries_path),
        output_dir=output_dir,
        county_order=["Alameda", "Contra Costa"],
        target_name="mobile_on_road_baaqmd",
        inventory_label="BAAQMD Mobile On-Road",
        pollutant_targets={
            "PM2.5": {
                "columns": (),
                "prefixes": ("pm25_",),
                "exclude_columns": ("pm25_prdust_short_tons_per_year",),
                "exclude_prefixes": (),
            },
            "NOx": {"columns": (), "prefixes": ("nox_",), "exclude_columns": (), "exclude_prefixes": ()},
            "BC": {
                "columns": ("bc_runex_short_tons_per_year",),
                "prefixes": (),
                "exclude_columns": (),
                "exclude_prefixes": (),
            },
        },
    )

    comparison = pd.read_parquet(outputs["comparison_parquet"])
    by_key = comparison.set_index(["county", "pollutant"])
    assert by_key.loc[("Alameda", "PM2.5"), "simulation_tons"] == 3.0
    assert by_key.loc[("Contra Costa", "PM2.5"), "simulation_tons"] == 4.0
    assert by_key.loc[("Alameda", "PM2.5"), "emfac_tons"] == 102.75
    assert by_key.loc[("Contra Costa", "PM2.5"), "emfac_tons"] == 104.75
    assert by_key.loc[("Alameda", "NOx"), "emfac_tons"] == 7.0
    assert by_key.loc[("Contra Costa", "BC"), "emfac_tons"] == 0.5

    assert Path(outputs["PM2.5_plot"]).exists()
    assert Path(outputs["NOx_plot"]).exists()
    assert Path(outputs["BC_plot"]).exists()


def test_analysis_step3_can_compare_road_dust_pm25_only(tmp_path: Path) -> None:
    modeled = pd.DataFrame(
        {
            "county_COUNTYFP": ["001"],
            "vehicleTypeId": ["vt-1"],
            "process": ["PRDUST"],
            "tons_per_year_PM2_5_county_allocated": [1.0],
        }
    )
    modeled_path = tmp_path / "beam_emissions_by_county_process.parquet"
    modeled.to_parquet(modeled_path, index=False)

    inventory = pd.DataFrame(
        {
            "county": ["Alameda"],
            "pm25_runex_short_tons_per_year": [2.0],
            "pm25_prdust_short_tons_per_year": [5.0],
        }
    )
    inventory_path = tmp_path / "inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)

    county_boundaries = gpd.GeoDataFrame(
        {
            "COUNTYFP": ["001"],
            "NAME": ["Alameda"],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        },
        geometry="geometry",
        crs="EPSG:26910",
    )
    county_boundaries_path = tmp_path / "county_boundaries.gpkg"
    county_boundaries.to_file(county_boundaries_path, driver="GPKG")

    outputs = run(
        modeled_emissions_path=str(modeled_path),
        inventory_path=str(inventory_path),
        county_boundaries_path=str(county_boundaries_path),
        output_dir=tmp_path / "analysis",
        county_order=["Alameda"],
        target_name="road_dust_baaqmd",
        inventory_label="BAAQMD Road Dust",
        pollutant_targets={
            "PM2.5": {
                "columns": ("pm25_prdust_short_tons_per_year",),
                "prefixes": (),
                "exclude_columns": (),
                "exclude_prefixes": (),
            }
        },
    )

    comparison = pd.read_parquet(outputs["comparison_parquet"])
    row = comparison.set_index(["county", "pollutant"]).loc[("Alameda", "PM2.5")]
    assert row["emfac_tons"] == 5.0
