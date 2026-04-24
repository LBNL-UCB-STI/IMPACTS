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
    if isinstance(root_scenario, dict) and "name" in root_scenario:
        defaults["scenario_name"] = deepcopy(root_scenario["name"])
    if "output" in emfac_root:
        defaults["outputs"] = deepcopy(emfac_root["output"])
    fleet = emfac_root.get("fleet", {})
    if isinstance(fleet, dict) and "vehicle_type_assignment_model_settings" in fleet:
        defaults["vehicle_type_assignment_model_settings"] = deepcopy(fleet["vehicle_type_assignment_model_settings"])
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
    activities_defaults = _build_activities_config_from_root(emfac_root)
    merged = _merge_dicts(defaults, fleet)
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


def _value_from_entry(value: object, key: str) -> object | None:
    mapping = _mapping_from_entry(value)
    if mapping is not None:
        result = mapping.get(key)
        return None if result in (None, "") else result
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
        emissions_inventory = {"inventory_folder": emissions_inventory}

    project_analysis_root = _folder_from_entry(project_analysis.get("main"))
    black_carbon_root = _folder_from_entry(project_analysis.get("black_carbon"))
    black_carbon_pollutant = _value_from_entry(project_analysis.get("black_carbon"), "pollutant")
    road_dust_root = _folder_from_entry(project_analysis.get("paved_road_dust"))
    road_category_map = _normalize_string_mapping(
        _value_from_entry(project_analysis.get("paved_road_dust"), "road_category_map"),
        lower_keys=True,
    )
    emissions_inventory_main = _folder_from_entry(emissions_inventory.get("inventory_folder"))
    emissions_inventory_fallback = _folder_from_entry(emissions_inventory.get("fallback_folder"))
    vehicle_category_attributes_file = _unwrap_config_value(emissions_inventory.get("vehicle_category_attributes_file"))

    normalized = {
        "project_analysis_raw": project_analysis_root,
        "black_carbon_raw": _find_matching_file(black_carbon_root, ("bc",), required=False),
        "black_carbon_pollutant": black_carbon_pollutant,
        "vehicle_type_assignment_model_settings": _normalize_model_spec_path(
            raw.get("vehicle_type_assignment_model_settings"),
            path_label="vehicle_type_assignment_model_settings",
        ),
        "vehicle_category_attributes_file": _expand_optional_path(
            None
            if vehicle_category_attributes_file in (None, "")
            else str(vehicle_category_attributes_file)
        ),
        "statewide_inventory_raw": _find_matching_file(emissions_inventory_fallback, ("statewide",), required=False),
        "population_raw": _find_matching_file(emissions_inventory_main, ("population",), required=False),
        "trips_raw": _find_matching_file(emissions_inventory_main, ("trips",), required=False),
        "vmt_raw": _find_matching_file(emissions_inventory_main, ("vmt",), required=False),
        "emission_raw": _find_matching_file(emissions_inventory_main, ("emission",), required=True),
        "ghg_raw": _find_matching_file(emissions_inventory_main, ("ghg",), required=False),
        "rainy_days_file": _find_matching_file(road_dust_root, ("rainy_days",), required=False),
        "silt_loading_file": _find_matching_file(road_dust_root, ("silt_loading",), required=False),
        "road_category_map": road_category_map,
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


def _normalize_string_mapping(mapping: object, *, lower_keys: bool = False, lower_values: bool = False) -> dict[str, str]:
    if mapping in (None, ""):
        return {}
    if not isinstance(mapping, dict):
        raise ValueError("Expected a mapping")
    normalized: dict[str, str] = {}
    for source, target in mapping.items():
        source_token = str(source).strip()
        target_token = str(target).strip()
        if not source_token or not target_token:
            continue
        if lower_keys:
            source_token = source_token.lower()
        if lower_values:
            target_token = target_token.lower()
        normalized[source_token] = target_token
    return normalized


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
        ("vehicle_type_assignment_model_settings",),
        ("vehicle_category_attributes_file",),
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
    emissions_inventory = raw.get("emissions_inventory", {})
    if isinstance(emissions_inventory, list):
        emissions_inventory = _flatten_input_groups({"emissions_inventory": emissions_inventory}).get(
            "emissions_inventory",
            {},
        )
    if not isinstance(emissions_inventory, dict):
        emissions_inventory = {}
    raw_fuel_map = emissions_inventory.get("fuel_map", {})
    normalized_fuel_map: dict[str, str] = {}
    if isinstance(raw_fuel_map, dict):
        for normalized_fuel, raw_fuels in raw_fuel_map.items():
            normalized_token = str(normalized_fuel).strip()
            if not normalized_token:
                continue
            candidates = raw_fuels if isinstance(raw_fuels, (list, tuple, set)) else [raw_fuels]
            for raw_fuel in candidates:
                raw_token = str(raw_fuel).strip()
                if not raw_token:
                    continue
                normalized_fuel_map[raw_token] = normalized_token

    year = int(raw["calendar_year"])
    region = str(raw["region_label"])
    scenario_name = str(raw.get("scenario_name", "Baseline"))
    outputs_root = Path(str(raw["outputs"])).expanduser()
    activities_output_root = outputs_root / "activities"
    tmp_root = outputs_root / "_tmp"
    trace_dir = tmp_root / "traces"
    region_slug = region.lower()
    base_name = f"{region_slug}-emfac-{year}"
    final_name = f"{base_name}-project-analysis-final"
    inventory_final_name = f"{base_name}-inventory-final"
    emissions_store_name = f"{year}-{scenario_name}"

    return {
        "run": {
            "region_label": region,
            "calendar_year": year,
            "scenario_name": scenario_name,
            "outputs": str(outputs_root),
            "model_year_groups": {
                "light_duty": list(raw["model_year_groups"]["light_duty"]),
                "medium_heavy_duty": list(raw["model_year_groups"]["medium_heavy_duty"]),
            },
            "pto_as_process": raw["pto_as_process"],
            "mappings": {
                **deepcopy(raw.get("mappings", {})),
                "fuel_map": normalized_fuel_map,
                "road_category_map": deepcopy(raw.get("road_category_map", {})),
            },
        },
        "inputs": {
            key: raw[key]
            for key in (
                "project_analysis_raw",
                "vehicle_type_assignment_model_settings",
                "vehicle_category_attributes_file",
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
            "final_activity_emfacid_output_passenger": str(
                activities_output_root / f"{inventory_final_name}-passenger-activity-by-emfacid.parquet"
            ),
            "final_fleet_output_passenger": str(activities_output_root / f"{inventory_final_name}-passenger-fleet.parquet"),
            "final_output_freight": str(activities_output_root / f"{final_name}-freight-rates.parquet"),
            "final_activity_output_freight": str(activities_output_root / f"{inventory_final_name}-freight-activity.parquet"),
            "final_activity_emfacid_output_freight": str(
                activities_output_root / f"{inventory_final_name}-freight-activity-by-emfacid.parquet"
            ),
            "final_fleet_output_freight": str(activities_output_root / f"{inventory_final_name}-freight-fleet.parquet"),
            "emissions_store_root": str(outputs_root / "emissions" / emissions_store_name),
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


def _extract_fleet_assignment_root(model_spec: dict, *, model_spec_path: Path) -> dict[str, object]:
    root = model_spec.get("fleet_assignment")
    if not isinstance(root, dict):
        raise ValueError(
            f"Configured fleet path has an invalid model spec file at {model_spec_path}. "
            "It must contain top-level fleet_assignment."
        )
    return root


def _extract_named_model(model_spec: dict, *, model_name: str, model_spec_path: Path) -> dict[str, object]:
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=model_spec_path)
    models = fleet_assignment.get("models")
    if not isinstance(models, dict):
        raise ValueError(
            f"Configured fleet path has an invalid model spec file at {model_spec_path}. "
            "It must contain fleet_assignment.models."
        )
    model_section = models.get(model_name)
    if not isinstance(model_section, dict):
        raise ValueError(
            f"Configured fleet path has an invalid model spec file at {model_spec_path}. "
            f"It must contain fleet_assignment.models.{model_name}."
        )
    return model_section


def _normalized_string_list(values: object, *, lower: bool = False) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        normalized.append(token.lower() if lower else token)
    return normalized


def build_model_category_fuel_mapping(model_spec_path: str | Path) -> pd.DataFrame:
    spec_path = Path(model_spec_path)
    model_spec = _load_model_spec(str(spec_path))
    rows: list[dict[str, str]] = []
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=spec_path)
    mappings = fleet_assignment.get("mappings", {})
    if not isinstance(mappings, dict):
        raise ValueError(f"Vehicle type assignment model file at {spec_path} must define fleet_assignment.mappings.")

    passenger_allowed_fuels_by_emfac_category = {
        "MCY": {"Gas"},
        "UBUS": {"Dsl", "Elec", "Gas"},
    }

    passenger_mapping = mappings.get("passenger", {})
    if not isinstance(passenger_mapping, dict):
        raise ValueError(f"Vehicle type assignment model file at {spec_path} must define mappings.passenger.")
    passenger_vehicle_categories = passenger_mapping.get("vehicle_categories", {})
    passenger_fuel_types = passenger_mapping.get("fuel_types", {})

    for beam_category, emfac_categories in passenger_vehicle_categories.items():
        beam_category_token = str(beam_category).strip()
        if not beam_category_token:
            continue
        for emfac_category in _normalized_string_list(emfac_categories):
            allowed_emfac_fuels = passenger_allowed_fuels_by_emfac_category.get(emfac_category)
            for adopt_fuel, emfac_fuels in passenger_fuel_types.items():
                adopt_fuel_token = str(adopt_fuel).strip().lower()
                if not adopt_fuel_token:
                    continue
                for emfac_fuel in _normalized_string_list(emfac_fuels):
                    if allowed_emfac_fuels is not None and emfac_fuel not in allowed_emfac_fuels:
                        continue
                    rows.append(
                        {
                            "group": "passenger",
                            "emfac_vehicle_category": emfac_category,
                            "emfac_fuel": emfac_fuel,
                            "beam_category": beam_category_token,
                            "adopt_fuel": adopt_fuel_token,
                        }
                    )

    freight_mapping = mappings.get("freight", {})
    if not isinstance(freight_mapping, dict):
        raise ValueError(f"Vehicle type assignment model file at {spec_path} must define mappings.freight.")
    freight_vehicle_categories = freight_mapping.get("vehicle_categories", {})
    freight_fuel_types = freight_mapping.get("fuel_types", {})
    for beam_category, emfac_categories in freight_vehicle_categories.items():
        beam_category_token = str(beam_category).strip()
        if not beam_category_token:
            continue
        for emfac_category in _normalized_string_list(emfac_categories):
            for adopt_fuel, emfac_fuels in freight_fuel_types.items():
                adopt_fuel_token = str(adopt_fuel).strip().lower()
                if not adopt_fuel_token:
                    continue
                for emfac_fuel in _normalized_string_list(emfac_fuels):
                    rows.append(
                        {
                            "group": "freight",
                            "emfac_vehicle_category": emfac_category,
                            "emfac_fuel": emfac_fuel,
                            "beam_category": beam_category_token,
                            "adopt_fuel": adopt_fuel_token,
                        }
                    )

    frame = pd.DataFrame(rows, columns=["group", "emfac_vehicle_category", "emfac_fuel", "beam_category", "adopt_fuel"])
    if frame.empty:
        raise ValueError(f"Vehicle type assignment model file at {spec_path} produced no EMFAC category/fuel mapping rows.")
    return frame.drop_duplicates().reset_index(drop=True)


def _normalize_fastsim_catalog_id(value: object) -> str:
    token = str(value or "").strip()
    if token.endswith("_lookup_table"):
        token = token[: -len("_lookup_table")]
    return token


def build_fuel_consumption_emfac_assignment_catalog(
    model_spec_path: str | Path,
    breakdown_path: str | Path,
) -> pd.DataFrame:
    spec_path = Path(model_spec_path)
    model_spec = _load_model_spec(str(spec_path))
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=spec_path)
    mappings = fleet_assignment.get("mappings", {})
    assignments = mappings.get("fuel_consumption", []) if isinstance(mappings, dict) else []
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(
            f"Vehicle type assignment model file at {spec_path} has no mappings.fuel_consumption rows."
        )

    breakdown = read_table(
        str(breakdown_path),
        dtype=None,
        columns=[
            "fastsim_id",
            "model_year",
            "fuel",
            "charge_behavior",
            "model_trim",
            "msrp_usd",
            "fastsim_relative_path",
        ],
    ).copy()
    breakdown["fastsim_id"] = breakdown["fastsim_id"].map(_normalize_fastsim_catalog_id)
    breakdown["model_year"] = pd.to_numeric(breakdown["model_year"], errors="coerce")
    breakdown["fuel"] = breakdown["fuel"].fillna("").astype(str).str.strip().str.lower()
    breakdown["charge_behavior"] = breakdown["charge_behavior"].fillna("").astype(str).str.strip().str.lower()
    breakdown["model_trim"] = breakdown["model_trim"].fillna("").astype(str).str.strip()
    breakdown["fastsim_relative_path"] = breakdown["fastsim_relative_path"].fillna("").astype(str).str.strip()
    breakdown["msrp_usd"] = pd.to_numeric(breakdown["msrp_usd"], errors="coerce")

    rows: list[dict[str, object]] = []
    for item in assignments:
        if not isinstance(item, dict):
            continue
        fastsim_id = _normalize_fastsim_catalog_id(item.get("fastsim_id"))
        emfac_vehicle_categories = _normalized_string_list(item.get("vehicle_categories"))
        emfac_fuels = _normalized_string_list(item.get("fuel_types"))
        if not fastsim_id or not emfac_vehicle_categories or not emfac_fuels:
            continue
        matched = breakdown[breakdown["fastsim_id"] == fastsim_id].copy()
        if matched.empty:
            raise ValueError(
                f"Fuel-consumption assignment row in {spec_path} could not be resolved in {breakdown_path}: "
                f"fastsim_id={fastsim_id}"
            )
        for _, matched_row in matched.iterrows():
            relative_path = str(matched_row.get("fastsim_relative_path", "")).strip()
            matched_charge_behavior = str(matched_row.get("charge_behavior", "") or "").strip().lower()
            matched_fuel = str(matched_row.get("fuel", "")).strip().lower()
            if not relative_path:
                continue
            for emfac_vehicle_category in emfac_vehicle_categories:
                for emfac_fuel in emfac_fuels:
                    rows.append(
                        {
                            "fastsim_id": fastsim_id,
                            "fastsim_relative_path": relative_path,
                            "emfac_vehicle_category": str(emfac_vehicle_category).strip(),
                            "emfac_fuel": str(emfac_fuel).strip(),
                            "model_year": matched_row.get("model_year"),
                            "fuel": matched_fuel,
                            "charge_behavior": matched_charge_behavior,
                            "model_trim": matched_row.get("model_trim"),
                            "msrp_usd": matched_row.get("msrp_usd"),
                        }
                    )
    frame = pd.DataFrame(
        rows,
        columns=[
            "fastsim_id",
            "fastsim_relative_path",
            "emfac_vehicle_category",
            "emfac_fuel",
            "model_year",
            "fuel",
            "charge_behavior",
            "model_trim",
            "msrp_usd",
        ],
    )
    if frame.empty:
        raise ValueError(
            f"Vehicle type assignment model file at {spec_path} produced no fuel-consumption assignment rows."
        )
    return frame.drop_duplicates().reset_index(drop=True)


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
    model_section = _extract_named_model(
        model_spec,
        model_name="freight_bayesian_dag",
        model_spec_path=model_spec_path,
    )
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=model_spec_path)
    mappings = fleet_assignment.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain fleet_assignment.mappings."
        )
    freight_mapping = mappings.get("freight")
    if not isinstance(freight_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain mappings.freight."
        )
    naics_evidence = freight_mapping.get("naics_sector")
    if not isinstance(naics_evidence, list) or not naics_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.freight.naics_sector entries in {model_spec_path}. "
            "It should contain the NAICS-sector-to-vehicle-category evidence mappings."
        )
    freight_vehicle_categories = freight_mapping.get("vehicle_categories")
    if not isinstance(freight_vehicle_categories, dict) or not freight_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.freight.vehicle_categories entries in {model_spec_path}. "
            "It should contain the FRISM-to-EMFAC vehicle-category evidence mappings."
        )
    invalid_freight_vehicle_categories = [
        key
        for key, value in freight_vehicle_categories.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_freight_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.freight.vehicle_categories entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_freight_vehicle_categories))
        )
    freight_fuel_types = freight_mapping.get("fuel_types")
    if not isinstance(freight_fuel_types, dict) or not freight_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.freight.fuel_types entries in {model_spec_path}. "
            "It should contain the FRISM-to-EMFAC fuel evidence mappings."
        )
    invalid_freight_fuel_types = [
        key
        for key, value in freight_fuel_types.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_freight_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.freight.fuel_types entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_freight_fuel_types))
        )
    port_evidence = freight_mapping.get("port_location")
    if not isinstance(port_evidence, list) or not port_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.freight.port_location entries in {model_spec_path}. "
            "It should contain the zone-to-vehicle-category evidence mappings for port assignments."
        )
    passenger_model = _extract_named_model(
        model_spec,
        model_name="passenger_bayesian_dag",
        model_spec_path=model_spec_path,
    )
    passenger_scoring = passenger_model.get("scoring")
    if not isinstance(passenger_scoring, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.passenger_bayesian_dag.scoring."
        )
    passenger_weights = passenger_scoring.get("weights")
    if not isinstance(passenger_weights, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.passenger_bayesian_dag.scoring.weights."
        )
    missing_passenger_weights = [
        key for key in ("fleet_vmt_prior", "income") if key not in passenger_weights
    ]
    if missing_passenger_weights:
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.passenger_bayesian_dag.scoring.weights is missing: "
            + ", ".join(sorted(missing_passenger_weights))
        )
    passenger_evidence = passenger_model.get("evidence")
    if not isinstance(passenger_evidence, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.passenger_bayesian_dag.evidence."
        )
    income_evidence = passenger_evidence.get("income")
    if not isinstance(income_evidence, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.passenger_bayesian_dag.evidence.income."
        )
    missing_income_evidence = [
        key for key in ("center_ratio", "sigma_ratio") if key not in income_evidence
    ]
    if missing_income_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.passenger_bayesian_dag.evidence.income is missing: "
            + ", ".join(sorted(missing_income_evidence))
        )
    mappings = fleet_assignment.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain fleet_assignment.mappings."
        )
    freight_mapping = mappings.get("freight")
    if not isinstance(freight_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain mappings.freight."
        )
    passenger_mapping = mappings.get("passenger")
    if not isinstance(passenger_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain mappings.passenger."
        )
    passenger_vehicle_categories = passenger_mapping.get("body_types")
    if not isinstance(passenger_vehicle_categories, dict) or not passenger_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.passenger.body_types entries in {model_spec_path}. "
            "It should contain the ATLAS-bodytype-to-EMFAC-category support mappings."
        )
    invalid_vehicle_categories = [
        key
        for key, value in passenger_vehicle_categories.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.body_types entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_vehicle_categories))
        )
    passenger_fuel_types = passenger_mapping.get("fuel_types")
    if not isinstance(passenger_fuel_types, dict) or not passenger_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.passenger.fuel_types entries in {model_spec_path}. "
            "It should contain the passenger BEAM-fuel-to-EMFAC fuel evidence mappings."
        )
    invalid_passenger_fuel_types = [
        key
        for key, value in passenger_fuel_types.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_passenger_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.fuel_types entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_passenger_fuel_types))
        )
    passenger_vehicle_categories = passenger_mapping.get("vehicle_categories")
    if not isinstance(passenger_vehicle_categories, dict) or not passenger_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.passenger.vehicle_categories entries in {model_spec_path}. "
            "It should contain the BEAM-category-to-EMFAC-category support mappings for passenger bus/bike mapping."
        )
    invalid_passenger_vehicle_categories = [
        key
        for key, value in passenger_vehicle_categories.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_passenger_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.vehicle_categories entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_passenger_vehicle_categories))
        )
    passenger_model_year_mapping = passenger_mapping.get("model_year", {})
    if passenger_model_year_mapping not in (None, "") and not isinstance(passenger_model_year_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.model_year in {model_spec_path}. "
            "It must be a mapping when provided."
        )
    assignment_rows = mappings.get("fuel_consumption")
    if not isinstance(assignment_rows, list) or not assignment_rows:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.fuel_consumption rows in {model_spec_path}. "
            "It should contain the fuel-consumption assignment rows."
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
    activities = config.get("activities", {})
    if isinstance(activities, dict):
        activities["outputs"] = _normalize_configured_path(
            activities.get("outputs"),
            path_label="activities.outputs",
            must_exist=False,
        )
        mappings = activities.get("mappings", {})
        if mappings in (None, ""):
            mappings = {}
        if not isinstance(mappings, dict):
            raise ValueError("activities.mappings must be a mapping")
        activities["mappings"] = mappings
        emissions_inventory = activities.get("emissions_inventory", {})
        if isinstance(emissions_inventory, dict):
            raw_fuel_map = emissions_inventory.get("fuel_map", {})
            if raw_fuel_map in (None, ""):
                raw_fuel_map = {}
            if not isinstance(raw_fuel_map, dict):
                raise ValueError(
                    "activities.emissions_inventory.fuel_map must be a mapping of "
                    "normalized fuel tokens to one or more raw EMFAC fuel labels"
                )
            normalized_fuel_map: dict[str, str] = {}
            for normalized_fuel, raw_fuels in raw_fuel_map.items():
                normalized_token = str(normalized_fuel).strip()
                if not normalized_token:
                    continue
                if isinstance(raw_fuels, (list, tuple, set)):
                    candidates = raw_fuels
                else:
                    candidates = [raw_fuels]
                for raw_fuel in candidates:
                    raw_token = str(raw_fuel).strip()
                    if not raw_token:
                        continue
                    normalized_fuel_map[raw_token] = normalized_token
            emissions_inventory["fuel_map"] = normalized_fuel_map
            activities["emissions_inventory"] = emissions_inventory
        activities = _derive_emfac_output_paths(activities)
        config["activities"] = activities
    frism = config.get("frism", {})
    if isinstance(frism, dict):
        for key in ("carriers_file", "payloads_file", "tours_file"):
            frism[key] = _normalize_configured_path(frism.get(key), path_label=f"frism.{key}")
        config["frism"] = frism
    beam = config.get("beam", {})
    if isinstance(beam, dict):
        beam["passenger_vehicle_types_file"] = _normalize_configured_path(
            beam.get("passenger_vehicle_types_file"),
            path_label="beam.passenger_vehicle_types_file",
        )
        beam["freight_vehicle_types_file"] = _normalize_configured_path(
            beam.get("freight_vehicle_types_file"),
            path_label="beam.freight_vehicle_types_file",
        )
        beam["fuel_consumption_catalog"] = _normalize_configured_path(
            beam.get("fuel_consumption_catalog"),
            path_label="beam.fuel_consumption_catalog",
        )
        config["beam"] = beam
    vta = config.get("vehicle_type_assignment", {})
    if isinstance(vta, dict):
        model_file = _normalize_model_spec_path(vta.get("model_file"), path_label="vehicle_type_assignment.model_file")
        if model_file is not None:
            vta["model_file"] = model_file
            model_spec = _load_model_spec(model_file)
            freight_model = _extract_named_model(
                model_spec,
                model_name="freight_bayesian_dag",
                model_spec_path=Path(model_file),
            )
            scoring = freight_model.get("scoring", {})
            if "likelihood_floor" not in scoring:
                raise ValueError(
                    "vehicle_type_assignment.model_file must define models.freight_bayesian_dag.scoring.likelihood_floor"
                )
            floor_value = float(scoring["likelihood_floor"])
            if not (0.0 < floor_value < 1.0):
                raise ValueError(
                    f"vehicle_type_assignment likelihood_floor must be between 0 and 1 exclusive, got {floor_value}"
                )
            vta["likelihood_floor"] = floor_value
            weights = scoring.get("weights", {})
            missing_weights = [
                key
                for key in ("fleet_vmt_prior", "naics_sector", "payload_mass", "port_location")
                if key not in weights
            ]
            if missing_weights:
                raise ValueError(
                    "vehicle_type_assignment.model_file must define models.freight_bayesian_dag.scoring.weights for: "
                    + ", ".join(missing_weights)
                )
            for key in ("fleet_vmt_prior", "naics_sector", "payload_mass", "port_location"):
                value = float(weights[key])
                if value < 0.0:
                    raise ValueError(f"vehicle_type_assignment {key} weight must be non-negative, got {value}")
                vta[key] = value
            vta["emfac_population_bias"] = float(scoring.get("emfac_population_bias", 1.0))
            vta["emfac_vmt_bias"] = float(scoring.get("emfac_vmt_bias", 0.0))
            fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=Path(model_file))
            mappings = fleet_assignment.get("mappings", {})
            freight_mapping = mappings.get("freight", {}) if isinstance(mappings, dict) else {}
            freight_vehicle_categories = freight_mapping.get("vehicle_categories", {})
            freight_fuel_types = freight_mapping.get("fuel_types", {})
            config["freight_bayesian_dag"] = {
                "likelihood_floor": floor_value,
                "fleet_vmt_prior": float(weights["fleet_vmt_prior"]),
                "naics_sector": float(weights["naics_sector"]),
                "payload_mass": float(weights["payload_mass"]),
                "port_location": float(weights["port_location"]),
                "emfac_population": float(scoring.get("emfac_population_bias", 1.0)),
                "emfac_vmt": float(scoring.get("emfac_vmt_bias", 0.0)),
            }
            config["freight_mapping"] = {
                "vehicle_categories": {
                    str(category).strip(): [
                        str(emfac_category).strip()
                        for emfac_category in emfac_categories
                        if str(emfac_category).strip()
                    ]
                    for category, emfac_categories in freight_vehicle_categories.items()
                    if str(category).strip()
                },
                "fuel_types": {
                    str(fuel).strip(): [
                        str(emfac_fuel).strip()
                        for emfac_fuel in emfac_fuels
                        if str(emfac_fuel).strip()
                    ]
                    for fuel, emfac_fuels in freight_fuel_types.items()
                    if str(fuel).strip()
                },
                "naics_sector": deepcopy(freight_mapping.get("naics_sector", [])),
                "port_location": deepcopy(freight_mapping.get("port_location", [])),
            }
            passenger_model = _extract_named_model(
                model_spec,
                model_name="passenger_bayesian_dag",
                model_spec_path=Path(model_file),
            )
            passenger_scoring = passenger_model.get("scoring", {})
            passenger_weights = passenger_scoring.get("weights", {})
            passenger_evidence = passenger_model.get("evidence", {})
            passenger_income_evidence = passenger_evidence.get("income", {}) if isinstance(passenger_evidence, dict) else {}
            passenger_mapping = mappings.get("passenger", {}) if isinstance(mappings, dict) else {}
            passenger_vehicle_categories = passenger_mapping.get("body_types", {})
            passenger_fuel_types = passenger_mapping.get("fuel_types", {})
            config["passenger_bayesian_dag"] = {
                "likelihood_floor": float(passenger_scoring.get("likelihood_floor", 1e-3)),
                "fleet_vmt_prior_weight": float(passenger_weights.get("fleet_vmt_prior", 1.0)),
                "income_weight": float(passenger_weights.get("income", 1.0)),
                "income_center_ratio": float(passenger_income_evidence.get("center_ratio", 0.30)),
                "income_sigma_ratio": float(passenger_income_evidence.get("sigma_ratio", 0.10)),
            }
            config["passenger_mapping"] = {
                "body_types": {
                    str(bodytype).strip().lower(): [
                        str(category).strip()
                        for category in categories
                        if str(category).strip()
                    ]
                    for bodytype, categories in passenger_vehicle_categories.items()
                    if str(bodytype).strip()
                },
                "fuel_types": {
                    str(fuel).strip(): [
                        str(emfac_fuel).strip()
                        for emfac_fuel in emfac_fuels
                        if str(emfac_fuel).strip()
                    ]
                    for fuel, emfac_fuels in passenger_fuel_types.items()
                    if str(fuel).strip()
                },
                "vehicle_categories": {
                    str(beam_category).strip(): [
                        str(category).strip()
                        for category in categories
                        if str(category).strip()
                    ]
                    for beam_category, categories in passenger_mapping.get("vehicle_categories", {}).items()
                    if str(beam_category).strip()
                },
            }
            config["fuel_consumption_mapping"] = [
                {
                    "fastsim_id": str(item.get("fastsim_id", "")).strip(),
                    "vehicle_categories": [
                        str(category).strip()
                        for category in item.get("vehicle_categories", [])
                        if str(category).strip()
                    ],
                    "fuel_types": [
                        str(fuel).strip()
                        for fuel in item.get("fuel_types", [])
                        if str(fuel).strip()
                    ],
                }
                for item in mappings.get("fuel_consumption", [])
                if isinstance(item, dict)
            ]
        config["vehicle_type_assignment"] = vta
    atlas = config.get("atlas", {})
    if isinstance(atlas, dict):
        for key in ("vehicles_file", "households_file", "persons_file"):
            atlas[key] = _normalize_configured_path(atlas.get(key), path_label=f"atlas.{key}")
        if atlas.get("income_bins") is not None:
            atlas["income_bins"] = list(atlas["income_bins"])
        fuel_map = atlas.get("fuel_map", {})
        if fuel_map in (None, ""):
            fuel_map = {}
        if not isinstance(fuel_map, dict):
            raise ValueError(
                "atlas.fuel_map must be a mapping of normalized BEAM fuel tokens "
                "to one or more source ATLAS fuel tokens"
            )
        normalized_fuel_map: dict[str, str] = {}
        for beam_fuel, raw_fuels in fuel_map.items():
            beam_fuel_token = str(beam_fuel).strip().lower()
            if not beam_fuel_token:
                continue
            if isinstance(raw_fuels, (list, tuple, set)):
                candidates = raw_fuels
            else:
                candidates = [raw_fuels]
            for raw_fuel in candidates:
                raw_fuel_token = str(raw_fuel).strip().lower()
                if not raw_fuel_token:
                    continue
                normalized_fuel_map[raw_fuel_token] = beam_fuel_token
        atlas["fuel_map"] = normalized_fuel_map
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
        ("vehicle_type_assignment_model_settings",),
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
        ("beam",),
        ("beam", "passenger_vehicle_types_file"),
        ("beam", "freight_vehicle_types_file"),
        ("beam", "fuel_consumption_catalog"),
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
        "activities": raw["activities"],
        "atlas": raw["atlas"],
        "frism": raw["frism"],
        "beam": raw["beam"],
        "rates": raw.get("rates", {}),
        "vehicle_type_assignment": raw.get("vehicle_type_assignment", {}),
        "freight_bayesian_dag": raw.get("freight_bayesian_dag", {}),
        "passenger_bayesian_dag": raw.get("passenger_bayesian_dag", {}),
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
