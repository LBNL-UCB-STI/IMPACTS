from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.analysis.step1_compare_annual_targets import run


def test_analysis_step1_compares_modeled_totals_to_annual_targets(tmp_path: Path) -> None:
    modeled = pd.DataFrame(
        {
            "countyfp": ["001", "001", "001", "001"],
            "vehicleTypeId": ["pax-car", "ft-light", "ft-heavy", "ft-heavy"],
            "process": ["RUNEX", "RUNEX", "RUNEX", "PRDUST"],
            "tons_per_year_PM2_5_county_allocated": [10.0, 20.0, 30.0, 40.0],
            "tons_per_year_NOx_county_allocated": [100.0, 200.0, 300.0, 0.0],
        }
    )
    modeled_path = tmp_path / "beam_emissions_by_county_process.parquet"
    modeled.to_parquet(modeled_path, index=False)

    passenger_vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["pax-car"],
            "vehicleCategory": ["Car"],
            "vehicleClass": ["Car"],
            "vehicleUse": [""],
            "emfacVehicleCategory": ["LDA"],
        }
    )
    freight_vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["ft-light", "ft-heavy"],
            "vehicleCategory": ["Class12aVocational", "Class78Tractor"],
            "vehicleClass": ["Class 1&2A Vocational", "Class 7&8 Tractor"],
            "vehicleUse": ["Freight", "Freight"],
            "emfacVehicleCategory": ["LDT1", "T7 Tractor"],
        }
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)
    mapping = pd.DataFrame(
        {
            "emfac_vehicle_category": ["LDA", "LDT1", "T7 Tractor"],
            "generic_vehicle_category": ["passenger_cars", "light_duty_trucks", "heavy_duty_trucks"],
            "operation_days_per_year": [347, 347, 312],
        }
    )
    mapping_path = tmp_path / "emfac_vehicle_category_attributes.csv"
    mapping.to_csv(mapping_path, index=False)

    outputs = run(
        modeled_emissions_path=str(modeled_path),
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        vehicle_category_metadata_file=str(mapping_path),
        output_dir=tmp_path / "analysis",
        sector_targets=[
            {"source": "mobile_onroad", "sector": "passenger_cars", "annual_pm25_short_tons": 11.0, "annual_nox_short_tons": 101.0},
            {"source": "mobile_onroad", "sector": "light_duty_trucks", "annual_pm25_short_tons": 21.0, "annual_nox_short_tons": 201.0},
            {"source": "mobile_onroad", "sector": "heavy_duty_trucks", "annual_pm25_short_tons": 31.0, "annual_nox_short_tons": 301.0},
            {"source": "road_dust", "sector": "all", "annual_pm25_short_tons": 41.0, "annual_nox_short_tons": None},
        ],
    )

    comparison = pd.read_parquet(outputs["comparison_parquet"]).set_index(["source", "sector", "pollutant"])
    assert comparison.loc[("mobile_onroad", "passenger_cars", "PM2.5"), "simulation_tons"] == 10.0
    assert comparison.loc[("mobile_onroad", "light_duty_trucks", "PM2.5"), "simulation_tons"] == 20.0
    assert comparison.loc[("mobile_onroad", "heavy_duty_trucks", "PM2.5"), "simulation_tons"] == 30.0
    assert comparison.loc[("road_dust", "all", "PM2.5"), "simulation_tons"] == 40.0
    assert comparison.loc[("mobile_onroad", "heavy_duty_trucks", "NOx"), "simulation_tons"] == 300.0
    assert comparison.loc[("mobile_onroad", "passenger_cars", "NOx"), "target_tons"] == 101.0
