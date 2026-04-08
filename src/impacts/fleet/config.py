"""Local configuration helpers for the step-based fleet workflow."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"
EXAMPLE_CONFIG_PATH = Path("examples/sfbay_fleet/settings.yaml")
MAPPINGS_DIR = CONFIG_DIR / "mappings"


class BeamClasses:
    CLASS_2B3_VOCATIONAL = "Class2b3Vocational"
    CLASS_456_VOCATIONAL = "Class456Vocational"
    CLASS_78_VOCATIONAL = "Class78Vocational"
    CLASS_78_TRACTOR = "Class78Tractor"
    CLASS_CAR = "Car"
    CLASS_BIKE = "Bike"
    CLASS_MDP = "MediumDutyPassenger"

    @classmethod
    def get_freight_classes(cls) -> list[str]:
        return [
            cls.CLASS_2B3_VOCATIONAL,
            cls.CLASS_456_VOCATIONAL,
            cls.CLASS_78_VOCATIONAL,
            cls.CLASS_78_TRACTOR,
        ]

    @classmethod
    def get_passenger_classes(cls) -> list[str]:
        return [cls.CLASS_CAR, cls.CLASS_BIKE, cls.CLASS_MDP]


emissions_config = {
    "pollutants": {
        "CH4": "rate_ch4_gram_float",
        "CO": "rate_co_gram_float",
        "CO2": "rate_co2_gram_float",
        "HC": "rate_hc_gram_float",
        "NH3": "rate_nh3_gram_float",
        "NOx": "rate_nox_gram_float",
        "PM": "rate_pm_gram_float",
        "PM10": "rate_pm10_gram_float",
        "PM2_5": "rate_pm2_5_gram_float",
        "ROG": "rate_rog_gram_float",
        "SOx": "rate_sox_gram_float",
        "TOG": "rate_tog_gram_float",
        "BC": "rate_bc_gram_float",
        "BCm": "rate_bcm_gram_float",
        "BCh": "rate_bch_gram_float",
    },
    "processes": [
        "RUNEX",
        "IDLEX",
        "STREX",
        "DIURN",
        "HOTSOAK",
        "RUNLOSS",
        "PMTW",
        "PMBW",
        "PRDUST",
    ],
}

vehicle_types_config = {
    "columns": [
        "vehicleTypeId",
        "seatingCapacity",
        "standingRoomCapacity",
        "lengthInMeter",
        "primaryFuelType",
        "primaryFuelConsumptionInJoulePerMeter",
        "primaryFuelCapacityInJoule",
        "primaryVehicleEnergyFile",
        "secondaryFuelType",
        "secondaryFuelConsumptionInJoulePerMeter",
        "secondaryVehicleEnergyFile",
        "secondaryFuelCapacityInJoule",
        "automationLevel",
        "maxVelocity",
        "passengerCarUnit",
        "rechargeLevel2RateLimitInWatts",
        "rechargeLevel3RateLimitInWatts",
        "vehicleCategory",
        "sampleProbabilityWithinCategory",
        "sampleProbabilityString",
    ]
}


def get_fuel_key(row):
    fuel = row["primaryFuelType"].lower()
    if fuel == "electricity":
        suffix = "only" if pd.isna(row["secondaryFuelType"]) else "hybrid"
        return f"{fuel}-{suffix}"
    return fuel


def _load_yaml(path: Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    return (data or {}).get("fleet", data or {})


def _expand_path(value: str | None) -> str | None:
    if value in (None, ""):
        return value
    return str(Path(value).expanduser())


def _expand_paths(config: dict) -> dict:
    config = deepcopy(config)
    config["work_dir"] = _expand_path(config.get("work_dir"))
    config["outputs"] = _expand_path(config.get("outputs"))
    emfac = config.get("emfac", {})
    if isinstance(emfac, dict):
        for key in ("rates_file", "fleet_file", "activity_file"):
            emfac[key] = _expand_path(emfac.get(key))
        config["emfac"] = emfac
    return config


def _load_mapping_table(path: Path) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            grouped.setdefault(row["mapping_group"], {})[row["source_value"]] = row["target_value"]
    return grouped


def _load_alternatives(path: Path) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[tuple[int, str]]]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            domain = row["domain"]
            source = row["source_value"]
            grouped.setdefault(domain, {}).setdefault(source, []).append((int(row["priority"]), row["alternative_value"]))
    return {
        domain: {
            source: [value for _, value in sorted(values, key=lambda item: item[0])]
            for source, values in per_domain.items()
        }
        for domain, per_domain in grouped.items()
    }


def load_mapping_config() -> dict:
    fuel_mappings = _load_mapping_table(MAPPINGS_DIR / "fuel_mappings.csv")
    class_mappings = _load_mapping_table(MAPPINGS_DIR / "class_mappings.csv")
    alternatives = _load_alternatives(MAPPINGS_DIR / "alternatives.csv")
    return {
        "fleet": {
            "ignore_beam_passenger_distribution": False,
            "ignore_beam_freight_distribution": False,
            "model_year_bins": [1993, 2006, 2018],
        },
        "atlas": {
            "enable_atlas_emfac_crosswalk": True,
            "emfac": "atlas/atlas_emfac_xwalk.csv",
            "alternatives": alternatives.get("atlas", {}),
        },
        "fuel": {
            "beam": fuel_mappings["beam"],
            "emfac-ft": fuel_mappings["emfac-ft"],
            "emfac-pax": fuel_mappings["emfac-pax"],
            "emfac-bus": fuel_mappings["emfac-bus"],
            "alternatives": alternatives.get("fuel", {}),
        },
        "class": {
            "emfac": {},
            "emfac-ft": class_mappings["emfac-ft"],
            "emfac-pax": class_mappings["emfac-pax"],
            "emfac-bus": class_mappings["emfac-bus"],
            "alternatives": alternatives.get("class", {}),
        },
    }


def _required_value(raw: dict, path: tuple[str, ...]):
    current = raw
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_workflow_settings(raw: dict, source_path: Path) -> None:
    required_paths = [
        ("region",),
        ("scenario",),
        ("work_dir",),
        ("outputs",),
        ("emfac",),
        ("emfac", "rates_file"),
        ("emfac", "fleet_file"),
        ("emfac", "activity_file"),
        ("beam",),
        ("beam", "freight_directory"),
        ("beam", "ft_vehicle_types_file"),
        ("beam", "pax_vehicles_file"),
        ("beam", "pax_vehicle_types_file"),
    ]
    missing = []
    for path in required_paths:
        value = _required_value(raw, path)
        if value is None or value == "":
            missing.append(".".join(path))
    if missing:
        raise ValueError(
            f"Fleet config at {source_path} is missing required keys: {', '.join(missing)}. "
            f"Use {EXAMPLE_CONFIG_PATH} as a starting point."
        )


def load_workflow(config_path: str | Path | None = None) -> dict:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml(source_path)
    _validate_workflow_settings(raw, source_path)
    raw = _expand_paths(raw)
    outputs = raw["outputs"]
    config = {
        "override_rates": True,
        "override_fleet": True,
        "run": {
            "output_dir": outputs,
            "emissions_dir": outputs,
        },
        "emfac": raw["emfac"],
        "beam": raw["beam"],
        "mapping": load_mapping_config(),
    }
    return {
        "area": raw["region"],
        "run_batch": Path(outputs).name,
        "scenario": raw["scenario"],
        "work_dir": raw["work_dir"],
        "config": config,
    }


def load_default_workflow() -> dict:
    return load_workflow(DEFAULT_CONFIG_PATH)
