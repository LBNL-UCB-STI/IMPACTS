"""Local configuration helpers for the step-based fleet workflow."""

from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parents[2]
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


def resolve_workflow_path(path_like: str | None) -> str:
    if path_like in (None, ""):
        raise ValueError("Expected a configured path value, got an empty value")
    source = Path(str(path_like)).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return str(source.resolve())


def read_table(path_like: str, *, dtype: str | None = "str") -> pd.DataFrame:
    resolved = Path(resolve_workflow_path(path_like))
    if resolved.suffix.lower() == ".parquet":
        frame = pd.read_parquet(resolved)
        if dtype == "str":
            return frame.fillna("").astype(str)
        return frame
    return pd.read_csv(resolved, dtype=dtype).fillna("")


def _normalize_configured_path(
    path_like: str | None,
    *,
    path_label: str,
    expect_directory: bool = False,
    must_exist: bool = True,
) -> str | None:
    if path_like in (None, ""):
        return None
    resolved = Path(resolve_workflow_path(path_like))
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Configured fleet path '{path_label}' does not exist: {resolved}")
    if expect_directory and must_exist and not resolved.is_dir():
        raise NotADirectoryError(f"Configured fleet path '{path_label}' is not a directory: {resolved}")
    if not expect_directory and must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Configured fleet path '{path_label}' is not a file: {resolved}")
    return str(resolved)


def _ingest_configured_sources(config: dict) -> dict:
    config = deepcopy(config)
    config["output"] = _normalize_configured_path(
        config.get("output"),
        path_label="output",
        must_exist=False,
    )
    emfac = config.get("emfac", {})
    if isinstance(emfac, dict):
        emfac["rates_file"] = _normalize_configured_path(emfac.get("rates_file"), path_label="emfac.rates_file")
        emfac["activity_file"] = _normalize_configured_path(emfac.get("activity_file"), path_label="emfac.activity_file")
        emfac["atlas_emfac_xwalk"] = _normalize_configured_path(
            emfac.get("atlas_emfac_xwalk"),
            path_label="emfac.atlas_emfac_xwalk",
        )
        config["emfac"] = emfac
    beam = config.get("beam", {})
    if isinstance(beam, dict):
        beam["freight_directory"] = _normalize_configured_path(
            beam.get("freight_directory"),
            path_label="beam.freight_directory",
            expect_directory=True,
        )
        beam["ft_vehicle_types_file"] = _normalize_configured_path(
            beam.get("ft_vehicle_types_file"),
            path_label="beam.ft_vehicle_types_file",
        )
        beam["pax_vehicles_file"] = _normalize_configured_path(
            beam.get("pax_vehicles_file"),
            path_label="beam.pax_vehicles_file",
            must_exist=False,
        )
        beam["pax_vehicle_types_file"] = _normalize_configured_path(
            beam.get("pax_vehicle_types_file"),
            path_label="beam.pax_vehicle_types_file",
            must_exist=False,
        )
        config["beam"] = beam
    atlas = config.get("atlas", {})
    if isinstance(atlas, dict):
        atlas["vehicles_file"] = _normalize_configured_path(
            atlas.get("vehicles_file"),
            path_label="atlas.vehicles_file",
        )
        atlas["vehicles_types_file"] = _normalize_configured_path(
            atlas.get("vehicles_types_file"),
            path_label="atlas.vehicles_types_file",
        )
        atlas["curb_weight_mapping_file"] = _normalize_configured_path(
            atlas.get("curb_weight_mapping_file"),
            path_label="atlas.curb_weight_mapping_file",
        )
        config["atlas"] = atlas
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
        ("output",),
        ("emfac",),
        ("emfac", "rates_file"),
        ("emfac", "activity_file"),
        ("emfac", "atlas_emfac_xwalk"),
        ("atlas",),
        ("atlas", "vehicles_file"),
        ("atlas", "vehicles_types_file"),
        ("atlas", "curb_weight_mapping_file"),
        ("beam",),
        ("beam", "freight_directory"),
        ("beam", "ft_vehicle_types_file"),
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
    raw = _ingest_configured_sources(raw)
    output_root = raw["output"]
    config = {
        "output": output_root,
        "emfac": raw["emfac"],
        "atlas": raw["atlas"],
        "beam": raw["beam"],
        "mapping": load_mapping_config(),
    }
    atlas_xwalk = raw.get("emfac", {}).get("atlas_emfac_xwalk")
    if atlas_xwalk:
        config["mapping"]["atlas"]["emfac"] = atlas_xwalk
    return {
        "area": raw["region"],
        "run_batch": Path(output_root).name,
        "scenario": raw["scenario"],
        "config": config,
    }


def load_default_workflow() -> dict:
    return load_workflow(DEFAULT_CONFIG_PATH)
