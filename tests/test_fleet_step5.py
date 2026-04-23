from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.emfac.fleet.step5_map_emfac_rates import _build_special_category_rate_aliases
from impacts.emfac.fleet.step5_map_emfac_rates import _attach_idle_time_fraction
from impacts.emfac.fleet.step5_map_emfac_rates import _override_special_category_rate_paths


def test_attach_idle_time_fraction_uses_emfac_vehicle_category_lookup() -> None:
    vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["pax-1", "ft-1", "bike-1"],
            "emfacVehicleCategory": ["LDA", "T7 Tractor Class 8", ""],
        }
    )

    result = _attach_idle_time_fraction(
        vehicle_types,
        idle_time_fraction_lookup={
            "LDA": 0.0,
            "T7 Tractor Class 8": 0.1290,
        },
    )

    assert result["idleTimeFraction"].iloc[0] == 0.0
    assert result["idleTimeFraction"].iloc[1] == 0.1290
    assert pd.isna(result["idleTimeFraction"].iloc[2])


def test_attach_idle_time_fraction_defaults_missing_categories_to_zero() -> None:
    vehicle_types = pd.DataFrame(
        {
            "vehicleTypeId": ["pax-1", "bike-1"],
            "emfacVehicleCategory": ["LDA", ""],
        }
    )

    result = _attach_idle_time_fraction(
        vehicle_types,
        idle_time_fraction_lookup={"UBUS": 0.3550},
    )

    assert result["idleTimeFraction"].iloc[0] == 0.0
    assert pd.isna(result["idleTimeFraction"].iloc[1])


def test_special_category_rate_aliases_average_model_years_by_vmt_share_and_override_bus_path(tmp_path: Path) -> None:
    output_root = tmp_path / "emfac_output"
    store_root = output_root / "emissions" / "2018-Baseline"
    fleet_path = output_root / "activities" / "sfbay-emfac-2018-inventory-final-passenger-fleet.parquet"
    fleet_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"vehicleCategory": "UBUS", "fuel": "Dsl", "modelYear": "2018", "vmtShare": 0.25},
            {"vehicleCategory": "UBUS", "fuel": "Dsl", "modelYear": "2019", "vmtShare": 0.75},
        ]
    ).to_parquet(fleet_path, index=False)

    model_path = tmp_path / "vehicle_type_assignment_model.yaml"
    model_path.write_text(
        "\n".join(
            [
                "models:",
                "  freight_bayesian_dag:",
                "    scoring:",
                "      likelihood_floor: 0.01",
                "      weights:",
                "        prior_vmt_share: 1.0",
                "        naics_sector: 1.0",
                "        port_location: 1.0",
                "    evidence:",
                "      vehicle_categories:",
                "        Class12aVocational: [LDA]",
                "      fuel_types:",
                "        diesel: [Dsl]",
                "      category_fuel_support:",
                "        LDA: [Dsl]",
                "      naics_sector: []",
                "      port_location: []",
                "  passenger_bayesian_dag:",
                "    scoring:",
                "      weights:",
                "        bodytype: 1.0",
                "        fuel: 1.0",
                "        emfac_vmt: 0.0",
                "    evidence:",
                "      vehicle_categories:",
                "        car: [LDA]",
                "      fuel_types:",
                "        biodiesel: [Dsl]",
                "        diesel: [Dsl]",
                "      beam_vehicle_categories:",
                "        MediumDutyPassenger: [UBUS]",
                "      category_fuel_support:",
                "        LDA: [Gas]",
                "        UBUS: [Dsl]",
            ]
        ),
        encoding="utf-8",
    )

    for emfac_id, pm25 in [("2018UBUSDsl", 10.0), ("2019UBUSDsl", 30.0)]:
        partition = store_root / "dataset" / f"emfacId={emfac_id}"
        partition.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "county": "001",
                    "process": "RUNEX",
                    "speedMph_timeMin": 30,
                    "pm25_gram": pm25,
                }
            ]
        ).to_parquet(partition / f"{emfac_id}.parquet", index=False)

    config = {
        "output": str(output_root),
        "activities": {"passenger_fleet_file": str(fleet_path)},
        "vehicle_type_assignment": {"model_file": str(model_path)},
    }
    shared_rates_store = {
        "store_root": str(store_root),
        "relative_paths": {
            "2018UBUSDsl": str((store_root / "dataset" / "emfacId=2018UBUSDsl" / "2018UBUSDsl.parquet").relative_to(output_root)),
            "2019UBUSDsl": str((store_root / "dataset" / "emfacId=2019UBUSDsl" / "2019UBUSDsl.parquet").relative_to(output_root)),
        },
    }

    aliases = _build_special_category_rate_aliases(
        config=config,
        shared_rates_store=shared_rates_store,
    )

    alias_path = output_root / aliases[("UBUS", "Dsl")]
    alias_rates = pd.read_parquet(alias_path)
    assert alias_rates["pm25_gram"].iloc[0] == 25.0

    vehicle_types = pd.DataFrame(
        [
            {
                "vehicleTypeId": "bus-1",
                "adopt_fuel": "biodiesel",
                "emfacVehicleCategory": "UBUS",
                "emissionsRatesFile": "old/path.parquet",
            }
        ]
    )
    remapped = _override_special_category_rate_paths(
        vehicle_types,
        config=config,
        special_rate_aliases=aliases,
    )
    assert remapped["emissionsRatesFile"].iloc[0] == aliases[("UBUS", "Dsl")]
