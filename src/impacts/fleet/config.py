"""Local configuration helpers for the step-based fleet workflow."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parents[2]
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


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


def resolve_workflow_path(path_like: str | None) -> str:
    if path_like in (None, ""):
        raise ValueError("Expected a configured path value, got an empty value")
    source = Path(str(path_like)).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return str(source.resolve())


def read_table(
    path_like: str,
    *,
    dtype: str | None = "str",
    columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    resolved = Path(resolve_workflow_path(path_like))
    if resolved.suffix.lower() == ".parquet":
        frame = pd.read_parquet(resolved, columns=list(columns) if columns is not None else None)
        if dtype == "str":
            return frame.fillna("").astype(str)
        return frame
    read_kwargs = {"dtype": dtype}
    if columns is not None:
        read_kwargs["usecols"] = list(columns)
    return pd.read_csv(resolved, **read_kwargs).fillna("")


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
    mapping = config.get("mapping", {})
    if isinstance(mapping, dict):
        mapping["emfac_atlas_map"] = _normalize_configured_path(
            mapping.get("emfac_atlas_map"),
            path_label="mapping.emfac_atlas_map",
        )
        mapping["emfac_beam_class_map"] = _normalize_configured_path(
            mapping.get("emfac_beam_class_map"),
            path_label="mapping.emfac_beam_class_map",
        )
        mapping["emfac_beam_fuel_map"] = _normalize_configured_path(
            mapping.get("emfac_beam_fuel_map"),
            path_label="mapping.emfac_beam_fuel_map",
        )
        mapping["emfac_beam_fuel_alternatives_map"] = _normalize_configured_path(
            mapping.get("emfac_beam_fuel_alternatives_map"),
            path_label="mapping.emfac_beam_fuel_alternatives_map",
        )
        mapping["naics_emfac_sector_map"] = _normalize_configured_path(
            mapping.get("naics_emfac_sector_map"),
            path_label="mapping.naics_emfac_sector_map",
        )
        mapping["naics_emfac_priority_map"] = _normalize_configured_path(
            mapping.get("naics_emfac_priority_map"),
            path_label="mapping.naics_emfac_priority_map",
        )
        mapping["payloadtype_emfac_map"] = _normalize_configured_path(
            mapping.get("payloadtype_emfac_map"),
            path_label="mapping.payloadtype_emfac_map",
        )
        mapping["beam_freight_class_alternatives_map"] = _normalize_configured_path(
            mapping.get("beam_freight_class_alternatives_map"),
            path_label="mapping.beam_freight_class_alternatives_map",
        )
        mapping["beam_passenger_bodytype_alternatives_map"] = _normalize_configured_path(
            mapping.get("beam_passenger_bodytype_alternatives_map"),
            path_label="mapping.beam_passenger_bodytype_alternatives_map",
        )
        mapping["fastsim_bodytype_xwalk_file"] = _normalize_configured_path(
            mapping.get("fastsim_bodytype_xwalk_file"),
            path_label="mapping.fastsim_bodytype_xwalk_file",
        )
        mapping["fastsim_atlas_fuel_mapping_file"] = _normalize_configured_path(
            mapping.get("fastsim_atlas_fuel_mapping_file"),
            path_label="mapping.fastsim_atlas_fuel_mapping_file",
        )
        config["mapping"] = mapping
    emfac = config.get("emfac", {})
    if isinstance(emfac, dict):
        emfac["rates_file"] = _normalize_configured_path(emfac.get("rates_file"), path_label="emfac.rates_file")
        emfac["activity_file"] = _normalize_configured_path(emfac.get("activity_file"), path_label="emfac.activity_file")
        emfac["fleet_file"] = _normalize_configured_path(emfac.get("fleet_file"), path_label="emfac.fleet_file")
        config["emfac"] = emfac
    frism = config.get("frism", {})
    if isinstance(frism, dict):
        frism["carriers_file"] = _normalize_configured_path(
            frism.get("carriers_file"),
            path_label="frism.carriers_file",
        )
        frism["payloads_file"] = _normalize_configured_path(
            frism.get("payloads_file"),
            path_label="frism.payloads_file",
        )
        frism["tours_file"] = _normalize_configured_path(
            frism.get("tours_file"),
            path_label="frism.tours_file",
        )
        config["frism"] = frism
    fastsim = config.get("fastsim", {})
    if isinstance(fastsim, dict):
        passenger = fastsim.get("passenger", {})
        if isinstance(passenger, dict):
            passenger["vehicle_types_file"] = _normalize_configured_path(
                passenger.get("vehicle_types_file"),
                path_label="fastsim.passenger.vehicle_types_file",
            )
            passenger["fastsim_data_folder"] = _normalize_configured_path(
                passenger.get("fastsim_data_folder"),
                path_label="fastsim.passenger.fastsim_data_folder",
                expect_directory=True,
                must_exist=False,
            )
            fastsim["passenger"] = passenger
        freight = fastsim.get("freight", {})
        if isinstance(freight, dict):
            freight["vehicle_types_file"] = _normalize_configured_path(
                freight.get("vehicle_types_file"),
                path_label="fastsim.freight.vehicle_types_file",
            )
            freight["fastsim_data_folder"] = _normalize_configured_path(
                freight.get("fastsim_data_folder"),
                path_label="fastsim.freight.fastsim_data_folder",
                expect_directory=True,
                must_exist=False,
            )
            fastsim["freight"] = freight
        config["fastsim"] = fastsim
    atlas = config.get("atlas", {})
    if isinstance(atlas, dict):
        atlas["vehicles_file"] = _normalize_configured_path(
            atlas.get("vehicles_file"),
            path_label="atlas.vehicles_file",
        )
        atlas["households_file"] = _normalize_configured_path(
            atlas.get("households_file"),
            path_label="atlas.households_file",
        )
        atlas["persons_file"] = _normalize_configured_path(
            atlas.get("persons_file"),
            path_label="atlas.persons_file",
        )
        if atlas.get("income_bins") is not None:
            atlas["income_bins"] = list(atlas["income_bins"])
        config["atlas"] = atlas
    return config


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
        ("seed",),
        ("output",),
        ("mapping",),
        ("mapping", "emfac_atlas_map"),
        ("mapping", "emfac_beam_class_map"),
        ("mapping", "emfac_beam_fuel_map"),
        ("mapping", "emfac_beam_fuel_alternatives_map"),
        ("mapping", "naics_emfac_sector_map"),
        ("mapping", "naics_emfac_priority_map"),
        ("mapping", "payloadtype_emfac_map"),
        ("mapping", "beam_freight_class_alternatives_map"),
        ("mapping", "beam_passenger_bodytype_alternatives_map"),
        ("mapping", "fastsim_bodytype_xwalk_file"),
        ("mapping", "fastsim_atlas_fuel_mapping_file"),
        ("emfac",),
        ("emfac", "rates_file"),
        ("emfac", "activity_file"),
        ("emfac", "fleet_file"),
        ("atlas",),
        ("atlas", "vehicles_file"),
        ("atlas", "households_file"),
        ("atlas", "persons_file"),
        ("frism",),
        ("frism", "carriers_file"),
        ("frism", "payloads_file"),
        ("frism", "tours_file"),
        ("fastsim",),
        ("fastsim", "passenger"),
        ("fastsim", "passenger", "vehicle_types_file"),
        ("fastsim", "freight"),
        ("fastsim", "freight", "vehicle_types_file"),
    ]
    missing = []
    for path in required_paths:
        value = _required_value(raw, path)
        if value is None or value == "":
            missing.append(".".join(path))
    if missing:
        raise ValueError(
            f"Fleet config at {source_path} is missing required keys: {', '.join(missing)}."
        )


def load_workflow(config_path: str | Path | None = None) -> dict:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml(source_path)
    _validate_workflow_settings(raw, source_path)
    raw = _ingest_configured_sources(raw)
    output_root = raw["output"]
    config = {
        "seed": raw["seed"],
        "output": output_root,
        "mapping": raw["mapping"],
        "emfac": raw["emfac"],
        "atlas": raw["atlas"],
        "frism": raw["frism"],
        "fastsim": raw["fastsim"],
    }
    return {
        "area": raw["region"],
        "scenario": raw["scenario"],
        "config": config,
    }


def load_default_workflow() -> dict:
    return load_workflow(DEFAULT_CONFIG_PATH)
