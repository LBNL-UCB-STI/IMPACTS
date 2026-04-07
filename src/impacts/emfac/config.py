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
    return str(Path(value).expanduser())


def _expand_paths(raw: dict) -> dict:
    raw = deepcopy(raw)
    for section in ("inputs",):
        for key, value in raw.get(section, {}).items():
            raw[section][key] = _expand_path(value)
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
        ("inputs", "emissions_inventory_raw"),
        ("inputs", "statewide_emissions_inventory_raw"),
        ("inputs", "population_raw"),
        ("inputs", "trips_raw"),
        ("inputs", "black_carbon_raw"),
        ("inputs", "rainy_days_file"),
        ("inputs", "silt_loading_file"),
    ]
    missing = [".".join(path) for path in required if _required(raw, path) in (None, "")]
    if missing:
        raise ValueError(
            f"EMFAC config at {source_path} is missing required keys: {', '.join(missing)}."
        )


def load_workflow(config_path: str | Path | None = None) -> dict[str, object]:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _expand_paths(_load_yaml(source_path))
    _validate(raw, source_path)

    year = int(raw["calendar_year"])
    region = str(raw["region_label"])
    outputs_root = Path(str(raw["outputs"])).expanduser()
    region_slug = region.lower()
    base_name = f"{region_slug}-emfac-{year}"
    final_name = f"{base_name}-project-analysis-with-nh3-bc-prdust"

    return {
        "run": {
            "region_label": region,
            "calendar_year": year,
            "outputs": str(outputs_root),
            "model_year_groups": list(raw["model_year_groups"]),
        },
        "inputs": raw["inputs"],
        "paths": {
            "outputs_root": str(outputs_root),
            "trace_dir": str(outputs_root / "traces"),
            "project_analysis_clean": str(outputs_root / f"{base_name}-project-analysis.parquet"),
            "emissions_inventory_and_activities": str(
                outputs_root / f"{base_name}-emissions-inventory-with-activity.parquet"
            ),
            "statewide_emissions_inventory": str(
                outputs_root / f"statewide-emfac-{year}-emissions-inventory.parquet"
            ),
            "project_analysis_with_nh3": str(outputs_root / f"{base_name}-project-analysis-with-nh3.parquet"),
            "project_analysis_with_nh3_bc": str(outputs_root / f"{base_name}-project-analysis-with-nh3-bc.parquet"),
            "project_analysis_with_nh3_bc_prdust": str(outputs_root / f"{final_name}.parquet"),
            "final_output": str(outputs_root / f"{final_name}-rates.parquet"),
            "final_fleet_output": str(outputs_root / f"{final_name}-fleet.parquet"),
        },
    }


def load_default_workflow() -> dict[str, object]:
    return load_workflow(DEFAULT_CONFIG_PATH)
