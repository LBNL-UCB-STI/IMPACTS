from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.analysis.step1_compare_fleet import run


def test_analysis_step1_compares_beam_fleet_to_emfac_by_emfac_id(tmp_path: Path) -> None:
    skims = pd.DataFrame(
        {
            "linkId": [1, 1, 2, 3, 3],
            "vehicleTypeId": ["pax-a", "pax-a", "pax-b", "ft-a", "ft-a"],
            "totVMT": [10.0, 10.0, 20.0, 30.0, 30.0],
        }
    )
    skims_path = tmp_path / "prepared_skims.parquet"
    skims.to_parquet(skims_path, index=False)

    passenger_vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["pax-a", "pax-b"],
            "emfacId": ["pre2004LDAGas", "post2014LDAGas"],
            "sampleProbabilityWithinCategory": [0.4, 0.6],
        }
    )
    freight_vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["ft-a"],
            "emfacId": ["2003to2006T7TractorDsl"],
            "sampleProbabilityWithinCategory": [1.0],
        }
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)

    passenger_vehicles = pd.DataFrame(
        {"vehicleTypeId": ["pax-a", "pax-a", "pax-b", "pax-b", "pax-b"]}
    )
    freight_carriers = pd.DataFrame({"vehicleTypeId": ["ft-a"]})
    passenger_vehicles_path = tmp_path / "vehicles--EM.parquet"
    freight_carriers_path = tmp_path / "carriers--EM.parquet"
    passenger_vehicles.to_parquet(passenger_vehicles_path, index=False)
    freight_carriers.to_parquet(freight_carriers_path, index=False)

    passenger_activity = pd.DataFrame(
        {
            "county": ["A", "A", "B", "B"],
            "emfacId": ["pre2004LDAGas", "pre2004LDAGas", "post2014LDAGas", "post2014LDAGas"],
            "process": ["RUNEX", "PMBW", "RUNEX", "PMBW"],
            "total_vmt_vehicle_miles_per_year": [100.0, 100.0, 300.0, 300.0],
            "population_vehicles": [20.0, 20.0, 80.0, 80.0],
        }
    )
    freight_activity = pd.DataFrame(
        {
            "county": ["A", "A"],
            "emfacId": ["2003to2006T7TractorDsl", "2003to2006T7TractorDsl"],
            "process": ["RUNEX", "PMBW"],
            "total_vmt_vehicle_miles_per_year": [400.0, 400.0],
            "population_vehicles": [50.0, 50.0],
        }
    )
    passenger_activity_path = tmp_path / "passenger_activity.parquet"
    freight_activity_path = tmp_path / "freight_activity.parquet"
    passenger_activity.to_parquet(passenger_activity_path, index=False)
    freight_activity.to_parquet(freight_activity_path, index=False)

    outputs = run(
        skims_emissions_path=str(skims_path),
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        emfac_passenger_activity_path=str(passenger_activity_path),
        emfac_freight_activity_path=str(freight_activity_path),
        output_dir=tmp_path / "analysis",
        passenger_vehicles_path=str(passenger_vehicles_path),
        freight_carriers_path=str(freight_carriers_path),
    )

    comparison = pd.read_parquet(outputs["comparison_parquet"]).set_index(["assignment_group", "emfacId"])
    assert comparison.loc[("passenger", "pre2004LDAGas"), "beam_population_weight"] == 2
    assert comparison.loc[("passenger", "post2014LDAGas"), "beam_population_weight"] == 3
    assert comparison.loc[("passenger", "pre2004LDAGas"), "beam_vmt"] == 10.0
    assert comparison.loc[("passenger", "post2014LDAGas"), "beam_vmt"] == 20.0
    assert comparison.loc[("passenger", "pre2004LDAGas"), "emfac_population_share"] == 0.2
    assert comparison.loc[("passenger", "post2014LDAGas"), "emfac_population_share"] == 0.8
    assert comparison.loc[("freight", "2003to2006T7TractorDsl"), "beam_vmt_share"] == 1.0


def test_analysis_step1_uses_actual_passenger_vehicle_assignments_when_available(tmp_path: Path) -> None:
    skims = pd.DataFrame(
        {
            "linkId": [1, 2],
            "vehicleTypeId": ["pax-a", "pax-b"],
            "totVMT": [10.0, 20.0],
        }
    )
    skims_path = tmp_path / "prepared_skims.parquet"
    skims.to_parquet(skims_path, index=False)

    passenger_vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["pax-a", "pax-b"],
            "emfacId": ["pre2004LDAGas", "post2014LDAGas"],
            "sampleProbabilityWithinCategory": [0.99, 0.01],
        }
    )
    freight_vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["ft-a"],
            "emfacId": ["2003to2006T7TractorDsl"],
            "sampleProbabilityWithinCategory": [1.0],
        }
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)

    passenger_vehicles = pd.DataFrame({"vehicleTypeId": ["pax-b", "pax-b", "pax-b", "pax-a"]})
    passenger_vehicles_path = tmp_path / "vehicles--EM.parquet"
    passenger_vehicles.to_parquet(passenger_vehicles_path, index=False)

    passenger_activity = pd.DataFrame(
        {
            "county": ["A", "B"],
            "emfacId": ["pre2004LDAGas", "post2014LDAGas"],
            "process": ["RUNEX", "RUNEX"],
            "total_vmt_vehicle_miles_per_year": [100.0, 300.0],
            "population_vehicles": [20.0, 80.0],
        }
    )
    freight_activity = pd.DataFrame(
        {
            "county": ["A"],
            "emfacId": ["2003to2006T7TractorDsl"],
            "process": ["RUNEX"],
            "total_vmt_vehicle_miles_per_year": [400.0],
            "population_vehicles": [50.0],
        }
    )
    passenger_activity_path = tmp_path / "passenger_activity.parquet"
    freight_activity_path = tmp_path / "freight_activity.parquet"
    passenger_activity.to_parquet(passenger_activity_path, index=False)
    freight_activity.to_parquet(freight_activity_path, index=False)

    outputs = run(
        skims_emissions_path=str(skims_path),
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        emfac_passenger_activity_path=str(passenger_activity_path),
        emfac_freight_activity_path=str(freight_activity_path),
        output_dir=tmp_path / "analysis",
        passenger_vehicles_path=str(passenger_vehicles_path),
    )

    comparison = pd.read_parquet(outputs["comparison_parquet"]).set_index(["assignment_group", "emfacId"])
    assert comparison.loc[("passenger", "pre2004LDAGas"), "beam_population_weight"] == 1
    assert comparison.loc[("passenger", "post2014LDAGas"), "beam_population_weight"] == 3
