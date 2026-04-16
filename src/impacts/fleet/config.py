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


def _normalize_model_spec_path(path_like: str | None, *, path_label: str) -> str | None:
    resolved = _normalize_configured_path(
        path_like,
        path_label=path_label,
        expect_directory=False,
        must_exist=True,
    )
    if resolved is None:
        return None
    model_spec_path = Path(resolved)
    model_spec = _load_model_spec(str(model_spec_path))
    model_section = model_spec.get("model")
    if not isinstance(model_section, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain a top-level 'model' mapping."
        )
    evidence = model_section.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain model.evidence."
        )
    naics_evidence = evidence.get("naics_sector")
    if not isinstance(naics_evidence, list) or not naics_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no evidence.naics_sector entries in {model_spec_path}. "
            "It should contain the NAICS-sector-to-vehicle-category evidence mappings."
        )
    port_evidence = evidence.get("port_location")
    if not isinstance(port_evidence, list) or not port_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no evidence.port_location entries in {model_spec_path}. "
            "It should contain the zone-to-vehicle-category evidence mappings for port assignments."
        )
    return str(model_spec_path)


def _load_model_spec(model_spec_path: str) -> dict:
    spec_path = Path(model_spec_path)
    with spec_path.open() as f:
        return yaml.safe_load(f) or {}


def _ingest_configured_sources(config: dict) -> dict:
    config = deepcopy(config)
    flat_model_file = config.get("vehicle_type_assignment_model_settings")
    if flat_model_file not in (None, ""):
        vta = config.get("vehicle_type_assignment", {})
        if not isinstance(vta, dict):
            vta = {}
        vta["model_file"] = flat_model_file
        config["vehicle_type_assignment"] = vta
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
        mapping["emfac_fuel_alternatives_map"] = _normalize_configured_path(
            mapping.get("emfac_fuel_alternatives_map"),
            path_label="mapping.emfac_fuel_alternatives_map",
        )
        mapping["beam_class_alternatives_map"] = _normalize_configured_path(
            mapping.get("beam_class_alternatives_map"),
            path_label="mapping.beam_class_alternatives_map",
        )
        mapping["atlas_bodytype_alternatives_map"] = _normalize_configured_path(
            mapping.get("atlas_bodytype_alternatives_map"),
            path_label="mapping.atlas_bodytype_alternatives_map",
        )
        mapping["fastsim_atlas_bodytype_xwalk_file"] = _normalize_configured_path(
            mapping.get("fastsim_atlas_bodytype_xwalk_file"),
            path_label="mapping.fastsim_atlas_bodytype_xwalk_file",
        )
        mapping["fastsim_atlas_fuel_mapping_file"] = _normalize_configured_path(
            mapping.get("fastsim_atlas_fuel_mapping_file"),
            path_label="mapping.fastsim_atlas_fuel_mapping_file",
        )
        mapping["frism_atlas_map"] = _normalize_configured_path(
            mapping.get("frism_atlas_map"),
            path_label="mapping.frism_atlas_map",
        )
        mapping["fastsim_frism_bodytype_xwalk_file"] = _normalize_configured_path(
            mapping.get("fastsim_frism_bodytype_xwalk_file"),
            path_label="mapping.fastsim_frism_bodytype_xwalk_file",
        )
        mapping["fastsim_frism_fuel_mapping_file"] = _normalize_configured_path(
            mapping.get("fastsim_frism_fuel_mapping_file"),
            path_label="mapping.fastsim_frism_fuel_mapping_file",
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
        fastsim["passenger_vehicle_types_file"] = _normalize_configured_path(
            fastsim.get("passenger_vehicle_types_file"),
            path_label="fastsim.passenger_vehicle_types_file",
        )
        fastsim["freight_vehicle_types_file"] = _normalize_configured_path(
            fastsim.get("freight_vehicle_types_file"),
            path_label="fastsim.freight_vehicle_types_file",
        )
        fastsim["ldv_fastsim_data_folder"] = _normalize_configured_path(
            fastsim.get("ldv_fastsim_data_folder"),
            path_label="fastsim.ldv_fastsim_data_folder",
            expect_directory=True,
            must_exist=False,
        )
        fastsim["mhdv_fastsim_data_folder"] = _normalize_configured_path(
            fastsim.get("mhdv_fastsim_data_folder"),
            path_label="fastsim.mhdv_fastsim_data_folder",
            expect_directory=True,
            must_exist=False,
        )
        config["fastsim"] = fastsim
    vta = config.get("vehicle_type_assignment", {})
    if isinstance(vta, dict):
        raw_model = vta.get("model")
        if raw_model is not None:
            model = str(raw_model).strip().lower()
            if model != "dag":
                raise ValueError(
                    f"vehicle_type_assignment.model must be 'dag', got {raw_model}"
                )
            vta["model"] = model
        model_file = _normalize_model_spec_path(vta.get("model_file"), path_label="vehicle_type_assignment.model_file")
        if model_file is not None:
            vta["model_file"] = model_file
            model_spec = _load_model_spec(model_file)
            scoring = model_spec.get("model", {}).get("scoring", {})
            raw_floor = scoring.get("likelihood_floor", vta.get("likelihood_floor"))
            if raw_floor is not None:
                floor_value = float(raw_floor)
                if not (0.0 < floor_value < 1.0):
                    raise ValueError(
                        f"vehicle_type_assignment likelihood_floor must be between 0 and 1 exclusive, got {floor_value}"
                    )
                vta["likelihood_floor"] = floor_value

            weights = scoring.get("weights", {})
            raw_prior_vmt_share = weights.get("prior_vmt_share")
            if raw_prior_vmt_share is not None:
                prior_vmt_share = float(raw_prior_vmt_share)
                if prior_vmt_share < 0.0:
                    raise ValueError(
                        "vehicle_type_assignment prior_vmt_share weight must be non-negative, "
                        f"got {prior_vmt_share}"
                    )
                vta["prior_vmt_share"] = prior_vmt_share

            raw_naics_sector = weights.get("naics_sector")
            if raw_naics_sector is not None:
                naics_sector = float(raw_naics_sector)
                if naics_sector < 0.0:
                    raise ValueError(
                        "vehicle_type_assignment naics_sector weight must be non-negative, "
                        f"got {naics_sector}"
                    )
                vta["naics_sector"] = naics_sector

            raw_port_location = weights.get("port_location")
            if raw_port_location is not None:
                port_location = float(raw_port_location)
                if port_location < 0.0:
                    raise ValueError(
                        "vehicle_type_assignment port_location weight must be non-negative, "
                        f"got {port_location}"
                    )
                vta["port_location"] = port_location

        config["vehicle_type_assignment"] = vta

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
        ("mapping", "emfac_fuel_alternatives_map"),
        ("mapping", "beam_class_alternatives_map"),
        ("vehicle_type_assignment_model_settings",),
        ("mapping", "atlas_bodytype_alternatives_map"),
        ("mapping", "fastsim_atlas_bodytype_xwalk_file"),
        ("mapping", "fastsim_atlas_fuel_mapping_file"),
        ("mapping", "frism_atlas_map"),
        ("mapping", "fastsim_frism_bodytype_xwalk_file"),
        ("mapping", "fastsim_frism_fuel_mapping_file"),
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
        ("fastsim", "passenger_vehicle_types_file"),
        ("fastsim", "freight_vehicle_types_file"),
        ("fastsim", "ldv_fastsim_data_folder"),
        ("fastsim", "mhdv_fastsim_data_folder"),
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
        "vehicle_type_assignment": raw.get("vehicle_type_assignment", {}),
    }
    return {
        "area": raw["region"],
        "scenario": raw["scenario"],
        "config": config,
    }


def load_default_workflow() -> dict:
    return load_workflow(DEFAULT_CONFIG_PATH)
