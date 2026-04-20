from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from impacts.analysis.step1_compare_emissions_inventory import run as run_step1
from impacts.common import normalize_county_fips
from impacts.common import resolve_required_manifest_input
from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.file_ops import load_structured_file
from impacts.manifest.file_ops import resolve_path
from impacts.manifest.schema import InputsManifest
from impacts.manifest.schema import RunManifest

logger = logging.getLogger(__name__)


def _humanize_target_name(name: str) -> str:
    return str(name).strip().replace("_", " ").replace("-", " ").title()


def _resolve_output_root(settings_path: str | Path) -> Path:
    settings = load_settings_from_yaml(settings_path)
    resolved = resolve_path(settings.impacts.local_output_folder, settings_path)
    if not resolved:
        raise ValueError("Analysis requires impacts.local_output_folder in settings.")
    return Path(resolved).resolve()


def _resolve_run_manifest_path(settings_path: str | Path) -> Path:
    candidate = _resolve_output_root(settings_path) / "run_manifest.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires workflow run_manifest.yaml in the configured impacts.local_output_folder. "
            f"Expected {candidate}."
        )
    return candidate


def _resolve_modeled_emissions_path(settings_path: str | Path) -> Path:
    run_manifest = RunManifest.from_dict(load_structured_file(_resolve_run_manifest_path(settings_path))).to_dict()
    candidate_raw = run_manifest.get("outputs", {}).get("beam_emissions_for_inmap")
    if not candidate_raw:
        raise ValueError(
            "Analysis requires beam_emissions_for_inmap in run_manifest.outputs."
        )
    candidate = Path(candidate_raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires workflow emissions outputs. "
            f"Expected {candidate}."
        )
    return candidate


def _resolve_county_boundaries_path(settings_path: str | Path) -> Path:
    run_manifest = RunManifest.from_dict(load_structured_file(_resolve_run_manifest_path(settings_path))).to_dict()
    input_manifest_path = run_manifest.get("input_manifest_path")
    if not input_manifest_path:
        raise ValueError("Analysis requires input_manifest_path in run_manifest.")
    input_manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    inputs = input_manifest.get("inputs", {}) or {}
    candidate = Path(resolve_required_manifest_input(inputs, key="county_boundaries")).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires staged county boundaries from preprocess. "
            f"Expected {candidate}."
        )
    return candidate


def _resolve_inventory_path(settings_path: str | Path) -> Path:
    settings = load_settings_from_yaml(settings_path)
    raw = settings.impacts.emissions.inventory_file
    if not raw:
        raise ValueError(
            "Analysis requires impacts.emissions.inventory_file in settings."
        )
    candidate = Path(resolve_path(raw, settings_path) or raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis inventory file was configured but not found. "
            f"Resolved path: {candidate}"
        )
    return candidate


def _resolve_inventory_target_path(settings_path: str | Path, raw: str) -> Path:
    candidate = Path(resolve_path(raw, settings_path) or raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis inventory target file was configured but not found. "
            f"Resolved path: {candidate}"
        )
    return candidate


def _resolve_settings_path_from_run_manifest(run_manifest_path: str | Path) -> Path:
    run_manifest = RunManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    input_manifest_path = run_manifest.get("input_manifest_path")
    if not input_manifest_path:
        raise ValueError("Analysis requires input_manifest_path in run_manifest.")
    input_manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    settings_source = input_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Analysis requires settings_source in input manifest.")
    return Path(settings_source).resolve()


def run_from_settings(
    *,
    settings_path: str | Path,
) -> Dict[str, str]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    settings = load_settings_from_yaml(settings_path)
    output_dir = _resolve_output_root(settings_path) / "analysis"
    county_boundaries_path = _resolve_county_boundaries_path(settings_path)
    modeled_emissions_path = _resolve_modeled_emissions_path(settings_path)
    if not settings.impacts.analysis.targets:
        raise ValueError("Analysis requires impacts.analysis.targets in settings.")
    county_order = []
    if settings.shared.geography.fips.counties:
        # order counties based on the staged county boundaries naming after FIPS filtering in step 1
        import geopandas as gpd

        county_gdf = gpd.read_file(county_boundaries_path)
        county_gdf["COUNTYFP"] = normalize_county_fips(county_gdf["COUNTYFP"])
        wanted = set(normalize_county_fips(pd.Series(list(settings.shared.geography.fips.counties))).dropna().tolist())
        county_order = (
            county_gdf.loc[county_gdf["COUNTYFP"].isin(wanted), ["COUNTYFP", "NAME"]]
            .drop_duplicates()
            .sort_values("COUNTYFP")["NAME"]
            .astype(str)
            .tolist()
        )
    outputs: Dict[str, str] = {}
    inventory_path = _resolve_inventory_target_path(settings_path, settings.impacts.analysis.inventory_file)
    for target in settings.impacts.analysis.targets:
        target_outputs = run_step1(
            modeled_emissions_path=str(modeled_emissions_path),
            inventory_path=str(inventory_path),
            county_boundaries_path=str(county_boundaries_path),
            output_dir=output_dir,
            county_order=county_order,
            target_name=target.name,
            inventory_label=f"{settings.impacts.analysis.inventory_label} {_humanize_target_name(target.name)}".strip(),
            pollutant_targets={
                pollutant: {
                    "columns": tuple(selector.columns),
                    "prefixes": tuple(selector.prefixes),
                    "exclude_columns": tuple(selector.exclude_columns),
                    "exclude_prefixes": tuple(selector.exclude_prefixes),
                }
                for pollutant, selector in target.pollutants.items()
            },
        )
        for key, value in target_outputs.items():
            outputs[f"{target.name}_{key}"] = value
    return outputs


def run_from_run_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, str]:
    settings_path = _resolve_settings_path_from_run_manifest(run_manifest_path)
    return run_from_settings(settings_path=settings_path)
