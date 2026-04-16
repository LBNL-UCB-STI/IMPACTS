from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


def _load_yaml(path: Path) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    return (data or {}).get("emfac", data or {})


def _expand_path(value: str | None) -> str | None:
    if value in (None, ""):
        return value
    return str(_resolve_path(Path(value).expanduser()))


def _resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    parent = path.parent
    if parent.exists():
        sibling = parent / path.name
        if sibling.exists():
            return sibling
    grandparent = parent.parent
    if grandparent.exists():
        candidate = grandparent / path.name
        if candidate.exists():
            return candidate
    return path


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
    target = _resolve_path(Path(path).expanduser())
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
    inputs = _flatten_input_groups(raw.get("inputs", {}))
    project_analysis = inputs.get("project_analysis", {})
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


def _normalize_inputs(raw: dict) -> dict:
    inputs = _flatten_input_groups(raw.get("inputs", {}))
    project_analysis = inputs.get("project_analysis", {})
    emissions_inventory = inputs.get("emissions_inventory", {})

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
        "statewide_inventory_raw": _find_matching_file(
            emissions_inventory_fallback,
            ("statewide",),
            required=False,
        ),
        "population_raw": _find_matching_file(emissions_inventory_main, ("population",), required=False),
        "trips_raw": _find_matching_file(emissions_inventory_main, ("trips",), required=False),
        "vmt_raw": _find_matching_file(emissions_inventory_main, ("vmt",), required=False),
        "emission_raw": _find_matching_file(emissions_inventory_main, ("emission",), required=True),
        "ghg_raw": _find_matching_file(emissions_inventory_main, ("ghg",), required=False),
        "rainy_days_file": _find_matching_file(road_dust_root, ("rainy_days",), required=False),
        "silt_loading_file": _find_matching_file(road_dust_root, ("silt_loading",), required=False),
    }
    return {key: _expand_path(value) for key, value in normalized.items()}


def _expand_paths(raw: dict) -> dict:
    raw = deepcopy(raw)
    raw["pto_as_process"] = _normalize_pto_as_process(raw)
    raw["inputs"] = _normalize_inputs(raw)
    outputs = raw["outputs"].format(calendar_year=raw["calendar_year"])
    raw["outputs"] = _expand_path(outputs)
    return raw


def _required(raw: dict, path: tuple[str, ...]) -> object:
    current = raw
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate(raw: dict, source_path: Path) -> None:
    required = [
        ("region_label",),
        ("calendar_year",),
        ("outputs",),
        ("model_year_groups",),
        ("inputs", "project_analysis_raw"),
        ("inputs", "statewide_inventory_raw"),
        ("inputs", "vmt_raw"),
        ("inputs", "population_raw"),
        ("inputs", "trips_raw"),
        ("inputs", "emission_raw"),
        ("inputs", "black_carbon_raw"),
        ("inputs", "black_carbon_pollutant"),
        ("inputs", "rainy_days_file"),
        ("inputs", "silt_loading_file"),
    ]
    missing = [".".join(path) for path in required if _required(raw, path) in (None, "")]
    if missing:
        raise ValueError(
            f"EMFAC config at {source_path} is missing required keys: {', '.join(missing)}."
        )
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


def _build_workflow(raw: dict[str, object], source_path: Path) -> dict[str, object]:
    raw = _expand_paths(raw)
    _validate(raw, source_path)

    year = int(raw["calendar_year"])
    region = str(raw["region_label"])
    outputs_root = Path(str(raw["outputs"])).expanduser()
    region_slug = region.lower()
    base_name = f"{region_slug}-emfac-{year}"
    final_name = f"{base_name}-project-analysis-final"

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
        "inputs": raw["inputs"],
        "paths": {
            "outputs_root": str(outputs_root),
            "trace_dir": str(outputs_root / "traces"),
            "project_analysis_source": str(outputs_root / f"{base_name}-project-analysis-source.parquet"),
            "project_analysis": str(outputs_root / f"{base_name}-project-analysis.parquet"),
            "project_analysis_bc": str(outputs_root / f"{base_name}-project-analysis-bc.parquet"),
            "project_analysis_prdust": str(outputs_root / f"{base_name}-project-analysis-prdust.parquet"),
            "project_analysis_nh3_rates": str(outputs_root / f"{base_name}-project-analysis-nh3-rates.parquet"),
            "emissions_inventory": str(
                outputs_root / f"{base_name}-emissions-inventory-with-activity.parquet"
            ),
            "statewide_inventory": str(
                outputs_root / f"statewide-emfac-{year}-emissions-inventory.parquet"
            ),
            "final_output": str(outputs_root / f"{final_name}-rates.parquet"),
            "final_activity_output": str(outputs_root / f"{final_name}-activity.parquet"),
            "final_fleet_output": str(outputs_root / f"{final_name}-fleet.parquet"),
        },
    }


def load_workflow(config_path: str | Path | None = None) -> dict[str, object]:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _load_yaml(source_path)
    return _build_workflow(raw, source_path)


def load_workflow_from_data(raw: dict[str, object], *, source_label: str = "<in-memory>") -> dict[str, object]:
    return _build_workflow(deepcopy(raw), Path(source_label))


def load_default_workflow() -> dict[str, object]:
    return load_workflow(DEFAULT_CONFIG_PATH)
