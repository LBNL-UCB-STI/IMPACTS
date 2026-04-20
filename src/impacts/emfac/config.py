from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parents[2]
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _load_yaml_path(path: Path, *sections: str) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    current = data
    for section in sections:
        if not isinstance(current, dict):
            return {}
        current = current.get(section, {})
    return current if isinstance(current, dict) else {}


def _merge_dicts(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_emfac_root(path: Path) -> dict[str, object]:
    return _load_yaml_path(path, "emfac")


def _build_activities_config_from_root(emfac_root: dict[str, object]) -> dict[str, object]:
    activities = emfac_root.get("activities", {})
    if not isinstance(activities, dict):
        activities = {}
    defaults: dict[str, object] = {}
    root_region = emfac_root.get("region", {})
    if isinstance(root_region, dict) and "label" in root_region:
        defaults["region_label"] = deepcopy(root_region["label"])
    root_scenario = emfac_root.get("scenario", {})
    if isinstance(root_scenario, dict) and "year" in root_scenario:
        defaults["calendar_year"] = deepcopy(root_scenario["year"])
    if "output" in emfac_root:
        defaults["outputs"] = deepcopy(emfac_root["output"])
    mappings = emfac_root.get("mappings", {})
    if isinstance(mappings, dict):
        if "emfac_category_fuel_mapping_file" in mappings:
            defaults["emfac_category_fuel_mapping_file"] = deepcopy(mappings["emfac_category_fuel_mapping_file"])
        if "vehicle_operation_days_file" in mappings:
            defaults["vehicle_operation_days_file"] = deepcopy(mappings["vehicle_operation_days_file"])
    return _merge_dicts(defaults, activities)


def _build_fleet_config_from_root(emfac_root: dict[str, object]) -> dict[str, object]:
    fleet = emfac_root.get("fleet", {})
    if not isinstance(fleet, dict):
        fleet = {}
    defaults: dict[str, object] = {}
    root_region = emfac_root.get("region", {})
    root_scenario = emfac_root.get("scenario", {})
    if isinstance(root_region, dict) and "name" in root_region:
        defaults["region"] = deepcopy(root_region["name"])
    if isinstance(root_scenario, dict):
        if "year" in root_scenario:
            defaults["year"] = deepcopy(root_scenario["year"])
        if "name" in root_scenario:
            defaults["scenario"] = deepcopy(root_scenario["name"])
    for key in ("seed", "output"):
        if key in emfac_root:
            defaults[key] = deepcopy(emfac_root[key])
    mappings = emfac_root.get("mappings", {})
    if isinstance(mappings, dict):
        for key in (
            "atlas_emfac_xwalk_file",
            "emfac_category_fuel_mapping_file",
            "fastsim_category_fuel_mapping_file",
            "atlas_frism_xwalk_file",
            "vehicle_operation_days_file",
        ):
            if key in mappings:
                defaults[key] = deepcopy(mappings[key])
    merged = _merge_dicts(defaults, fleet)
    activities_defaults = _build_activities_config_from_root(emfac_root)
    nested_activities = merged.get("activities", {})
    if not isinstance(nested_activities, dict):
        nested_activities = {}
    merged["activities"] = _merge_dicts(activities_defaults, nested_activities)
    return merged


def resolve_workflow_path(path_like: str | None) -> str:
    if path_like in (None, ""):
        raise ValueError("Expected a configured path value, got an empty value")
    source = Path(str(path_like)).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return str(source.resolve())


def _expand_optional_path(value: str | None) -> str | None:
    if value in (None, ""):
        return value
    return resolve_workflow_path(value)


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


def _flatten_input_groups(inputs: dict) -> dict:
    flattened = deepcopy(inputs or {})
    for key, value in list(flattened.items()):
        if isinstance(value, list):
            merged: dict[str, object] = {}
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError(f"Expected '{key}' entries to be mappings, got: {item!r}")
                merged.update(item)
            flattened[key] = merged
    return flattened


def _unwrap_config_value(value: object) -> object:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def _mapping_from_entry(value: object) -> dict[str, object] | None:
    value = _unwrap_config_value(value)
    if isinstance(value, list):
        merged: dict[str, object] = {}
        for item in value:
            if not isinstance(item, dict):
                return None
            merged.update(item)
        value = merged
    return value if isinstance(value, dict) else None


def _folder_from_entry(value: object) -> str | None:
    mapping = _mapping_from_entry(value)
    if mapping is not None:
        folder = mapping.get("folder")
        return None if folder in (None, "") else str(folder)
    value = _unwrap_config_value(value)
    if value in (None, ""):
        return None
    return str(value)


def _value_from_entry(value: object, key: str) -> str | None:
    mapping = _mapping_from_entry(value)
    if mapping is not None:
        result = mapping.get(key)
        return None if result in (None, "") else str(result)
    return None


def _find_matching_file(path: str | None, patterns: tuple[str, ...], *, required: bool = True) -> str | None:
    if path in (None, ""):
        return None
    target = Path(resolve_workflow_path(path))
    if target.is_file():
        return str(target)
    if not target.exists():
        if required:
            raise FileNotFoundError(f"Configured path does not exist: {target}")
        return None
    matches = sorted(
        candidate
        for candidate in target.iterdir()
        if candidate.is_file() and any(pattern in candidate.name.lower() for pattern in patterns)
    )
    if matches:
        return str(matches[0])
    if required:
        raise FileNotFoundError(f"No files matching {patterns} found under: {target}")
    return None


def _normalize_pto_as_process(raw: dict) -> dict[str, object]:
    project_analysis = raw.get("project_analysis", {})
    if isinstance(project_analysis, list):
        project_analysis = _flatten_input_groups({"project_analysis": project_analysis}).get("project_analysis", {})
    if isinstance(project_analysis, str):
        project_analysis = {"main": project_analysis}
    project_analysis_main = _mapping_from_entry(project_analysis.get("main")) or {}
    config = deepcopy(project_analysis_main.get("pto_as_process") or {})
    enabled = bool(config.get("enabled", False))
    targets = [str(value).strip() for value in config.get("vehicle_category", [])]
    return {
        "enabled": enabled,
        "targets": targets,
    }


def _normalize_activities_inputs(raw: dict) -> dict:
    project_analysis = raw.get("project_analysis", {})
    emissions_inventory = raw.get("emissions_inventory", {})

    if isinstance(project_analysis, list) or isinstance(emissions_inventory, list):
        flattened = _flatten_input_groups(
            {
                "project_analysis": project_analysis,
                "emissions_inventory": emissions_inventory,
            }
        )
        project_analysis = flattened.get("project_analysis", project_analysis)
        emissions_inventory = flattened.get("emissions_inventory", emissions_inventory)

    if isinstance(project_analysis, str):
        project_analysis = {"main": project_analysis}
    if isinstance(emissions_inventory, str):
        emissions_inventory = {"main": emissions_inventory}

    project_analysis_root = _folder_from_entry(project_analysis.get("main"))
    black_carbon_root = _folder_from_entry(project_analysis.get("black_carbon"))
    black_carbon_pollutant = _value_from_entry(project_analysis.get("black_carbon"), "pollutant")
    road_dust_root = _folder_from_entry(project_analysis.get("paved_road_dust"))
    emissions_inventory_main = _folder_from_entry(emissions_inventory.get("main"))
    emissions_inventory_fallback = _folder_from_entry(emissions_inventory.get("fallback"))

    normalized = {
        "project_analysis_raw": project_analysis_root,
        "black_carbon_raw": _find_matching_file(black_carbon_root, ("bc",), required=False),
        "black_carbon_pollutant": black_carbon_pollutant,
        "emfac_category_fuel_mapping_file": _expand_optional_path(raw.get("emfac_category_fuel_mapping_file")),
        "vehicle_operation_days_file": _expand_optional_path(raw.get("vehicle_operation_days_file")),
        "statewide_inventory_raw": _find_matching_file(emissions_inventory_fallback, ("statewide",), required=False),
        "population_raw": _find_matching_file(emissions_inventory_main, ("population",), required=False),
        "trips_raw": _find_matching_file(emissions_inventory_main, ("trips",), required=False),
        "vmt_raw": _find_matching_file(emissions_inventory_main, ("vmt",), required=False),
        "emission_raw": _find_matching_file(emissions_inventory_main, ("emission",), required=True),
        "ghg_raw": _find_matching_file(emissions_inventory_main, ("ghg",), required=False),
        "rainy_days_file": _find_matching_file(road_dust_root, ("rainy_days",), required=False),
        "silt_loading_file": _find_matching_file(road_dust_root, ("silt_loading",), required=False),
    }
    path_keys = {
        "project_analysis_raw",
        "black_carbon_raw",
        "statewide_inventory_raw",
        "population_raw",
        "trips_raw",
        "vmt_raw",
        "emission_raw",
        "ghg_raw",
        "rainy_days_file",
        "silt_loading_file",
    }
    return {
        key: _expand_optional_path(value) if key in path_keys else value
        for key, value in normalized.items()
    }


def _expand_activities_paths(raw: dict) -> dict:
    raw = deepcopy(raw)
    raw["pto_as_process"] = _normalize_pto_as_process(raw)
    raw["inputs"] = _normalize_activities_inputs(raw)
    raw.update(raw["inputs"])
    outputs = raw["outputs"].format(calendar_year=raw["calendar_year"])
    raw["outputs"] = _expand_optional_path(outputs)
    return raw


def _required(raw: dict, path: tuple[str, ...]) -> object:
    current = raw
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_activities(raw: dict, source_path: Path) -> None:
    required = [
        ("region_label",),
        ("calendar_year",),
        ("outputs",),
        ("model_year_groups",),
        ("project_analysis_raw",),
        ("emfac_category_fuel_mapping_file",),
        ("vehicle_operation_days_file",),
        ("statewide_inventory_raw",),
        ("vmt_raw",),
        ("population_raw",),
        ("trips_raw",),
        ("emission_raw",),
        ("black_carbon_raw",),
        ("black_carbon_pollutant",),
        ("rainy_days_file",),
        ("silt_loading_file",),
    ]
    missing = [".".join(path) for path in required if _required(raw, path) in (None, "")]
    if missing:
        raise ValueError(f"EMFAC config at {source_path} is missing required keys: {', '.join(missing)}.")
    model_year_groups = raw.get("model_year_groups")
    if not isinstance(model_year_groups, dict):
        raise ValueError(
            f"EMFAC config at {source_path} must define model_year_groups as a mapping with light_duty and medium_heavy_duty."
        )
    required_group_keys = {"light_duty", "medium_heavy_duty"}
    missing_group_keys = sorted(required_group_keys - set(model_year_groups))
    if missing_group_keys:
        raise ValueError(
            f"EMFAC config at {source_path} is missing model_year_groups keys: {', '.join(missing_group_keys)}."
        )


def _build_activities_workflow(raw: dict[str, object], source_path: Path) -> dict[str, object]:
    raw = _expand_activities_paths(raw)
    _validate_activities(raw, source_path)

    year = int(raw["calendar_year"])
    region = str(raw["region_label"])
    outputs_root = Path(str(raw["outputs"])).expanduser()
    activities_output_root = outputs_root / "activities"
    tmp_root = outputs_root / "_tmp"
    trace_dir = tmp_root / "traces"
    region_slug = region.lower()
    base_name = f"{region_slug}-emfac-{year}"
    final_name = f"{base_name}-project-analysis-final"
    inventory_final_name = f"{base_name}-inventory-final"

    return {
        "run": {
            "region_label": region,
            "calendar_year": year,
            "outputs": str(outputs_root),
            "model_year_groups": {
                "light_duty": list(raw["model_year_groups"]["light_duty"]),
                "medium_heavy_duty": list(raw["model_year_groups"]["medium_heavy_duty"]),
            },
            "pto_as_process": raw["pto_as_process"],
        },
        "inputs": {
            key: raw[key]
            for key in (
                "project_analysis_raw",
                "emfac_category_fuel_mapping_file",
                "vehicle_operation_days_file",
                "black_carbon_raw",
                "black_carbon_pollutant",
                "statewide_inventory_raw",
                "population_raw",
                "trips_raw",
                "vmt_raw",
                "emission_raw",
                "ghg_raw",
                "rainy_days_file",
                "silt_loading_file",
            )
            if key in raw
        },
        "paths": {
            "outputs_root": str(outputs_root),
            "activities_output_root": str(activities_output_root),
            "tmp_root": str(tmp_root),
            "trace_dir": str(trace_dir),
            "project_analysis_source": str(tmp_root / f"{base_name}-project-analysis-source.parquet"),
            "project_analysis_passenger": str(tmp_root / f"{base_name}-project-analysis-passenger.parquet"),
            "project_analysis_freight": str(tmp_root / f"{base_name}-project-analysis-freight.parquet"),
            "project_analysis_bc": str(tmp_root / f"{base_name}-project-analysis-bc.parquet"),
            "project_analysis_prdust": str(tmp_root / f"{base_name}-project-analysis-prdust.parquet"),
            "project_analysis_nh3_rates": str(tmp_root / f"{base_name}-project-analysis-nh3-rates.parquet"),
            "emissions_inventory": str(tmp_root / f"{base_name}-inventory-intermediate-with-activity.parquet"),
            "statewide_inventory": str(tmp_root / f"statewide-emfac-{year}-emissions-inventory.parquet"),
            "matching_activity_output_passenger": str(tmp_root / f"{base_name}-inventory-matching-passenger-activity.parquet"),
            "matching_activity_output_freight": str(tmp_root / f"{base_name}-inventory-matching-freight-activity.parquet"),
            "final_output_passenger": str(activities_output_root / f"{final_name}-passenger-rates.parquet"),
            "final_activity_output_passenger": str(activities_output_root / f"{inventory_final_name}-passenger-activity.parquet"),
            "final_fleet_output_passenger": str(activities_output_root / f"{inventory_final_name}-passenger-fleet.parquet"),
            "final_output_freight": str(activities_output_root / f"{final_name}-freight-rates.parquet"),
            "final_activity_output_freight": str(activities_output_root / f"{inventory_final_name}-freight-activity.parquet"),
            "final_fleet_output_freight": str(activities_output_root / f"{inventory_final_name}-freight-fleet.parquet"),
        },
    }


def load_activities_workflow(config_path: str | Path | None = None) -> dict[str, object]:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _build_activities_config_from_root(_load_emfac_root(source_path))
    return _build_activities_workflow(raw, source_path)


def load_activities_workflow_from_data(raw: dict[str, object], *, source_label: str = "<in-memory>") -> dict[str, object]:
    return _build_activities_workflow(deepcopy(raw), Path(source_label))


def load_default_activities_workflow() -> dict[str, object]:
    return load_activities_workflow(DEFAULT_CONFIG_PATH)


def _load_model_spec(model_spec_path: str) -> dict:
    spec_path = Path(model_spec_path)
    with spec_path.open() as handle:
        return yaml.safe_load(handle) or {}


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


def _derive_emfac_output_paths(emfac: dict[str, object]) -> dict[str, object]:
    outputs = emfac.get("outputs")
    region_label = emfac.get("region_label")
    calendar_year = emfac.get("calendar_year")
    if outputs in (None, "") or region_label in (None, "") or calendar_year in (None, ""):
        return emfac
    region_slug = str(region_label).strip().lower()
    project_analysis_name = f"{region_slug}-emfac-{int(calendar_year)}-project-analysis-final"
    inventory_final_name = f"{region_slug}-emfac-{int(calendar_year)}-inventory-final"
    inventory_matching_name = f"{region_slug}-emfac-{int(calendar_year)}-inventory-matching"
    outputs_root = Path(str(outputs))
    activities_output_root = outputs_root / "activities"
    tmp_root = outputs_root / "_tmp"
    emfac["passenger_rates_file"] = str((activities_output_root / f"{project_analysis_name}-passenger-rates.parquet").resolve())
    emfac["passenger_activity_file"] = str((tmp_root / f"{inventory_matching_name}-passenger-activity.parquet").resolve())
    emfac["passenger_fleet_file"] = str((activities_output_root / f"{inventory_final_name}-passenger-fleet.parquet").resolve())
    emfac["freight_rates_file"] = str((activities_output_root / f"{project_analysis_name}-freight-rates.parquet").resolve())
    emfac["freight_activity_file"] = str((tmp_root / f"{inventory_matching_name}-freight-activity.parquet").resolve())
    emfac["freight_fleet_file"] = str((activities_output_root / f"{inventory_final_name}-freight-fleet.parquet").resolve())
    return emfac


def _ingest_fleet_sources(config: dict) -> dict:
    config = deepcopy(config)
    flat_model_file = config.get("vehicle_type_assignment_model_settings")
    if flat_model_file not in (None, ""):
        vta = config.get("vehicle_type_assignment", {})
        if not isinstance(vta, dict):
            vta = {}
        vta["model_file"] = flat_model_file
        config["vehicle_type_assignment"] = vta
    config["output"] = _normalize_configured_path(config.get("output"), path_label="output", must_exist=False)
    mappings = config.get("mappings", {})
    if isinstance(mappings, dict):
        for key in (
            "atlas_emfac_xwalk_file",
            "emfac_category_fuel_mapping_file",
            "fastsim_category_fuel_mapping_file",
            "atlas_frism_xwalk_file",
            "vehicle_operation_days_file",
        ):
            mappings[key] = _normalize_configured_path(mappings.get(key), path_label=f"mappings.{key}")
        config["mappings"] = mappings
    activities = config.get("activities", {})
    if isinstance(activities, dict):
        activities["outputs"] = _normalize_configured_path(
            activities.get("outputs"),
            path_label="activities.outputs",
            must_exist=False,
        )
        activities = _derive_emfac_output_paths(activities)
        config["activities"] = activities
    frism = config.get("frism", {})
    if isinstance(frism, dict):
        for key in ("carriers_file", "payloads_file", "tours_file"):
            frism[key] = _normalize_configured_path(frism.get(key), path_label=f"frism.{key}")
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
        model_file = _normalize_model_spec_path(vta.get("model_file"), path_label="vehicle_type_assignment.model_file")
        if model_file is not None:
            vta["model_file"] = model_file
            model_spec = _load_model_spec(model_file)
            scoring = model_spec.get("model", {}).get("scoring", {})
            if "likelihood_floor" not in scoring:
                raise ValueError("vehicle_type_assignment.model_file must define model.scoring.likelihood_floor")
            floor_value = float(scoring["likelihood_floor"])
            if not (0.0 < floor_value < 1.0):
                raise ValueError(
                    f"vehicle_type_assignment likelihood_floor must be between 0 and 1 exclusive, got {floor_value}"
                )
            vta["likelihood_floor"] = floor_value
            weights = scoring.get("weights", {})
            missing_weights = [key for key in ("prior_vmt_share", "naics_sector", "port_location") if key not in weights]
            if missing_weights:
                raise ValueError(
                    "vehicle_type_assignment.model_file must define model.scoring.weights for: "
                    + ", ".join(missing_weights)
                )
            for key in ("prior_vmt_share", "naics_sector", "port_location"):
                value = float(weights[key])
                if value < 0.0:
                    raise ValueError(f"vehicle_type_assignment {key} weight must be non-negative, got {value}")
                vta[key] = value
        config["vehicle_type_assignment"] = vta
    atlas = config.get("atlas", {})
    if isinstance(atlas, dict):
        for key in ("vehicles_file", "households_file", "persons_file"):
            atlas[key] = _normalize_configured_path(atlas.get(key), path_label=f"atlas.{key}")
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


def _validate_fleet(raw: dict, source_path: Path) -> None:
    required_paths = [
        ("region",),
        ("scenario",),
        ("seed",),
        ("output",),
        ("atlas_emfac_xwalk_file",),
        ("emfac_category_fuel_mapping_file",),
        ("vehicle_type_assignment_model_settings",),
        ("fastsim_category_fuel_mapping_file",),
        ("atlas_frism_xwalk_file",),
        ("vehicle_operation_days_file",),
        ("activities",),
        ("activities", "outputs"),
        ("activities", "region_label"),
        ("activities", "calendar_year"),
        ("activities", "model_year_groups"),
        ("activities", "project_analysis"),
        ("activities", "emissions_inventory"),
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
        raise ValueError(f"Fleet config at {source_path} is missing required keys: {', '.join(missing)}.")


def load_fleet_workflow(config_path: str | Path | None = None) -> dict:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _build_fleet_config_from_root(_load_emfac_root(source_path))
    _validate_fleet(raw, source_path)
    raw = _ingest_fleet_sources(raw)
    output_root = raw["output"]
    config = {
        "seed": raw["seed"],
        "output": output_root,
        "mappings": {
            key: raw[key]
            for key in (
                "atlas_emfac_xwalk_file",
                "emfac_category_fuel_mapping_file",
                "fastsim_category_fuel_mapping_file",
                "atlas_frism_xwalk_file",
                "vehicle_operation_days_file",
            )
        },
        "activities": raw["activities"],
        "atlas": raw["atlas"],
        "frism": raw["frism"],
        "fastsim": raw["fastsim"],
        "vehicle_type_assignment": raw.get("vehicle_type_assignment", {}),
    }
    return {
        "area": raw["region"],
        "scenario": raw["scenario"],
        "config": config,
        "paths": {
            "trace_dir": str(Path(str(output_root)).expanduser() / "_tmp" / "traces"),
        },
    }


def load_default_fleet_workflow() -> dict:
    return load_fleet_workflow(DEFAULT_CONFIG_PATH)
