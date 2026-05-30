from __future__ import annotations

import pandas as pd

from impacts.pipeline.emfac.activities.step2_build_comprehensive_project_analysis import build_comprehensive_project_analysis


def test_build_comprehensive_project_analysis_uses_configured_pto_vehicle_categories() -> None:
    project_analysis = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "vehicleCategory": "Configured PTO Category",
                "fuel": "Dsl",
                "modelYear": "2013to2015",
                "process": "RUNEX",
                "speedMph_timeMin": "5",
            },
            {
                "county": "Alameda",
                "vehicleCategory": "Configured PTO Category",
                "fuel": "Dsl",
                "modelYear": "2013to2015",
                "process": "PTOEX",
                "speedMph_timeMin": "5",
            },
        ]
    )
    project_analysis_prdust = pd.DataFrame(
        columns=["county", "vehicleCategory", "fuel", "modelYear", "process", "roadCategory", "speedMph_timeMin"]
    )
    emissions_inventory = pd.DataFrame(
        [
            {
                "county": "Alameda",
                "vehicleCategory": "Configured PTO Category",
                "fuel": "Dsl",
                "modelYear": "2013to2015",
            }
        ]
    )
    emfac_category_fuel_mapping = pd.DataFrame(
        [
            {
                "emfac_vehicle_category": "Configured PTO Category",
                "emfac_fuel": "Dsl",
            }
        ]
    )

    result = build_comprehensive_project_analysis(
        project_analysis,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        emfac_category_fuel_mapping=emfac_category_fuel_mapping,
        pto_vehicle_categories=["Configured PTO Category"],
    )

    pto_rows = result.loc[result["process"].astype(str) == "PTOEX"]
    assert not pto_rows.empty
    assert set(pto_rows["vehicleCategory"].astype(str)) == {"Configured PTO Category"}
