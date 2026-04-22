from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.pipeline.workflow.prepare_emissions_from_skims import _filter_prepared_skims_by_assignment


def test_filter_prepared_skims_by_assignment_uses_vehicle_types_contract(tmp_path: Path) -> None:
    passenger_vehicle_types = pd.DataFrame(
        [
            {"vehicleTypeId": "pax-car", "vehicleCategory": "Car"},
            {"vehicleTypeId": "pax-bus", "vehicleCategory": "Bus"},
        ]
    )
    freight_vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "ft-md",
                "vehicleCategory": "Class456Vocational",
                "vehicleClass": "truck",
                "vehicleUse": "freight",
            },
        ]
    )
    passenger_vehicle_types_path = tmp_path / "vehicleTypes--atlas.csv"
    freight_vehicle_types_path = tmp_path / "vehicleTypes--frism.csv"
    passenger_vehicle_types.to_csv(passenger_vehicle_types_path, index=False)
    freight_vehicle_types.to_csv(freight_vehicle_types_path, index=False)

    prepared = pd.DataFrame(
        [
            {"linkId": 1, "vehicleTypeId": "pax-car", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "pax-bus", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
            {"linkId": 1, "vehicleTypeId": "ft-md", "process": "RUNEX", "NOx": 1.0, "observations": 2.0},
        ]
    )

    passenger_only = _filter_prepared_skims_by_assignment(
        prepared,
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        include_passenger=True,
        include_freight=False,
    )
    assert passenger_only["vehicleTypeId"].tolist() == ["pax-car", "pax-bus"]

    freight_only = _filter_prepared_skims_by_assignment(
        prepared,
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        include_passenger=False,
        include_freight=True,
    )
    assert freight_only["vehicleTypeId"].tolist() == ["ft-md"]
