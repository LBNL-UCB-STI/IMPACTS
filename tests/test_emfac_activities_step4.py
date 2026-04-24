from __future__ import annotations

import pandas as pd

from impacts.emfac.activities.step4_finalize_output import _build_activity_by_emfac_id_table


def test_build_activity_by_emfac_id_table_aggregates_to_county_emfac_id_and_process() -> None:
    activity_weights = pd.DataFrame(
        {
            "county": ["001", "001", "001", "001"],
            "vehicleCategory": ["LDA", "LDA", "LDA", "LDA"],
            "fuel": ["Gas", "Gas", "Gas", "Gas"],
            "modelYear": [1999, 1999, 2018, 2018],
            "total_vmt_vehicle_miles_per_year": [10.0, 10.0, 20.0, 20.0],
            "cvmt_vehicle_miles_per_year": [0.0, 0.0, 0.0, 0.0],
            "evmt_vehicle_miles_per_year": [0.0, 0.0, 0.0, 0.0],
            "population_vehicles": [2.0, 2.0, 4.0, 4.0],
            "trips_per_year": [5.0, 5.0, 7.0, 7.0],
            "pto_total_vmt_vehicle_miles_per_year": [0.0, 0.0, 0.0, 0.0],
        }
    )
    surface = pd.DataFrame(
        {
            "county": ["001", "001"],
            "vehicleCategory": ["LDA", "LDA"],
            "fuel": ["Gas", "Gas"],
            "modelYear": [1999, 2018],
            "process": ["RUNEX", "RUNEX"],
        }
    )
    model_year_groups = {
        "light_duty": [
            "<=2003",
            ">=2015",
        ]
    }

    result = _build_activity_by_emfac_id_table(
        activity_weights=activity_weights,
        surface=surface,
        model_year_groups=model_year_groups,
    ).sort_values("emfacId").reset_index(drop=True)

    assert result["emfacId"].tolist() == ["post2014LDAGas", "pre2004LDAGas"]
    assert result["process"].tolist() == ["RUNEX", "RUNEX"]
    assert result["total_vmt_vehicle_miles_per_year"].tolist() == [40.0, 20.0]
    assert result["population_vehicles"].tolist() == [8.0, 4.0]
