from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd

from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import InputsManifest
from .manifest.schema import PipelineConfig
from .manifest.schema import RunManifest
from .common import resolve_required_manifest_input

logger = logging.getLogger(__name__)

def _normalized_stage_label(label: str) -> str:
    text = str(label).strip()
    upper = text.upper()
    if upper.startswith("PREPROCESS STEP"):
        return upper
    if upper.startswith("STEP"):
        return f"WORKFLOW {upper}"
    return upper


def _log_step_banner(label: str, name: str) -> None:
    banner = f"========== ENTERING {_normalized_stage_label(label)}: {name.upper()} =========="
    sys.stdout.write("\n")
    sys.stdout.flush()
    logger.info("%s", banner)


def _resolve_runtime_output_root(
    *,
    input_manifest: Dict[str, Any],
) -> Path:
    settings_source = input_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Input manifest is missing settings_source; cannot resolve impacts.local_output_folder.")
    settings = load_settings_from_yaml(settings_source)
    resolved = resolve_path(settings.impacts.local_output_folder, settings_source)
    if not resolved:
        raise ValueError("Could not resolve impacts.local_output_folder from settings.")
    return Path(resolved).resolve()


def _humanize_target_name(name: str) -> str:
    return str(name).strip().replace("_", " ").replace("-", " ").title()


def _resolve_run_manifest_settings_path(run_manifest_path: str | Path) -> Path:
    run_manifest = RunManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    input_manifest_path = run_manifest.get("input_manifest_path")
    if not input_manifest_path:
        raise ValueError("Analysis requires input_manifest_path in run_manifest.")
    input_manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    settings_source = input_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Analysis requires settings_source in input manifest.")
    return Path(settings_source).resolve()


def _resolve_analysis_run_manifest_path(settings_path: str | Path) -> Path:
    candidate = Path(resolve_path(load_settings_from_yaml(settings_path).impacts.local_output_folder, settings_path)).resolve() / "run_manifest.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires workflow run_manifest.yaml in the configured impacts.local_output_folder. "
            f"Expected {candidate}."
        )
    return candidate


def _load_analysis_run_manifest(settings_path: str | Path) -> tuple[Path, dict[str, Any]]:
    run_manifest_path = _resolve_analysis_run_manifest_path(settings_path)
    run_manifest = RunManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    return run_manifest_path, run_manifest


def _load_analysis_context(settings_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest_path, run_manifest = _load_analysis_run_manifest(settings_path)
    input_manifest_path = run_manifest.get("input_manifest_path")
    if not input_manifest_path:
        raise ValueError("Analysis requires input_manifest_path in run_manifest.")
    input_manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    inputs = input_manifest.get("inputs", {}) or {}
    return run_manifest_path, run_manifest, input_manifest, inputs


def _resolve_analysis_modeled_emissions_path(settings_path: str | Path) -> Path:
    _, run_manifest = _load_analysis_run_manifest(settings_path)
    candidate_raw = run_manifest.get("outputs", {}).get("beam_emissions_by_county_process")
    if not candidate_raw:
        raise ValueError("Analysis requires beam_emissions_by_county_process in run_manifest.outputs.")
    candidate = Path(candidate_raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires county-intersected workflow emissions outputs. "
            f"Expected {candidate}."
        )
    return candidate


def _resolve_analysis_county_boundaries_path(settings_path: str | Path) -> Path:
    _, _, _, inputs = _load_analysis_context(settings_path)
    candidate = Path(resolve_required_manifest_input(inputs, key="county_boundaries")).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires staged county boundaries from preprocess. "
            f"Expected {candidate}."
        )
    return candidate


def _resolve_analysis_vehicle_types_paths(settings_path: str | Path) -> tuple[Path, Path]:
    _, _, _, inputs = _load_analysis_context(settings_path)
    passenger_candidate = Path(resolve_required_manifest_input(inputs, key="passenger_vehicle_types_input")).resolve()
    freight_candidate = Path(resolve_required_manifest_input(inputs, key="freight_vehicle_types_input")).resolve()
    if not passenger_candidate.exists():
        raise FileNotFoundError(
            "Analysis requires staged passenger vehicle types from preprocess. "
            f"Expected {passenger_candidate}."
        )
    if not freight_candidate.exists():
        raise FileNotFoundError(
            "Analysis requires staged freight vehicle types from preprocess. "
            f"Expected {freight_candidate}."
        )
    return passenger_candidate, freight_candidate


def _resolve_analysis_inventory_target_path(settings_path: str | Path, raw: str) -> Path:
    candidate = Path(resolve_path(raw, settings_path) or raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis inventory target file was configured but not found. "
            f"Resolved path: {candidate}"
        )
    return candidate


def run_analysis_from_settings(
    *,
    settings_path: str | Path,
) -> Dict[str, str]:
    from .analysis.step1_compare_annual_targets import run as run_step1
    from .analysis.step2_compare_emissions_inventory import run as run_step2
    from .common import normalize_county_fips

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    settings = load_settings_from_yaml(settings_path)
    output_dir = Path(resolve_path(settings.impacts.local_output_folder, settings_path)).resolve() / "analysis"
    modeled_emissions_path = _resolve_analysis_modeled_emissions_path(settings_path)
    outputs: Dict[str, str] = {}
    if settings.impacts.analysis.sector_targets:
        passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_analysis_vehicle_types_paths(settings_path)
        target_outputs = run_step1(
            modeled_emissions_path=str(modeled_emissions_path),
            passenger_vehicle_types_path=str(passenger_vehicle_types_path),
            freight_vehicle_types_path=str(freight_vehicle_types_path),
            output_dir=output_dir / "annual_targets",
            sector_targets=[
                {
                    "source": target.source,
                    "sector": target.sector,
                    "annual_pm25_short_tons": target.annual_pm25_short_tons,
                    "annual_nox_short_tons": target.annual_nox_short_tons,
                }
                for target in settings.impacts.analysis.sector_targets
            ],
        )
        for key, value in target_outputs.items():
            outputs[f"annual_targets_{key}"] = value
    if not settings.impacts.analysis.targets:
        return outputs
    county_boundaries_path = _resolve_analysis_county_boundaries_path(settings_path)
    county_order: list[str] = []
    if settings.shared.geography.fips.counties:
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
    inventory_path = _resolve_analysis_inventory_target_path(settings_path, settings.impacts.analysis.inventory_file)
    for target in settings.impacts.analysis.targets:
        target_outputs = run_step2(
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


def run_analysis_from_run_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, str]:
    return run_analysis_from_settings(settings_path=_resolve_run_manifest_settings_path(run_manifest_path))


def run_from_input_manifest(
    input_manifest_path: str | Path,
    output_dir: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    manifest = InputsManifest.from_dict(load_structured_file(input_manifest_path)).to_dict()
    pipeline = PipelineConfig.from_dict(manifest.get("pipeline", {}) or {})
    population_inputs = manifest.get("population_inputs", {}) or {}
    manifest_inputs = manifest.get("inputs", {}) or {}
    input_root = Path(manifest.get("input_dir", "")).resolve()

    output_root = _resolve_runtime_output_root(
        input_manifest=manifest,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded input manifest: %s", Path(input_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .pipeline.preprocessing.step3_integrate_grids import run as run_grid_intersection
    from .pipeline.workflow.step1_process_emissions import run as run_emissions_processing
    from .common import prepared_table_target

    _log_step_banner("PREPROCESS STEP 3", "network mapping and grid intersection")
    grid_intersection_paths, intersection_dfs = run_grid_intersection(
        pipeline,
        output_root,
        input_root,
        manifest_inputs=manifest_inputs,
    )

    _log_step_banner("STEP 1", "emissions processing")
    logger.info("Using Step 1 implementation: emissions_processing")
    emissions_outputs = run_emissions_processing(
        pipeline,
        output_root,
        input_root,
        grid_intersection_paths,
        intersection_dfs=intersection_dfs,
        manifest_inputs=manifest_inputs,
    )
    prepared_skims_candidate = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    prepared_skims_path = str(prepared_skims_candidate) if prepared_skims_candidate.exists() else None

    concentration_path: Optional[Path] = None
    aermod_concentration_path: Optional[Path] = None
    exposure_grid_path: Optional[Path] = None
    population_distribution_path: Optional[Path] = None
    population_counts_path: Optional[Path] = None
    if run_dispersion:
        from .pipeline.workflow.step2_compute_inmap_concentrations import run as run_inmap_dispersion
        from .pipeline.workflow.step3_compute_aermod_concentrations import run as run_aermod_dispersion
        from .pipeline.workflow.step4_prepare_exposure import run as run_prepare_exposure
        if pipeline.inmap_enabled and emissions_outputs.get("beam_emissions_for_inmap"):
            _log_step_banner("STEP 2", "inmap concentrations")
            logger.info("Using Step 2 implementation: inmap_concentrations_and_export")
            _, _, concentration_path = run_inmap_dispersion(
                pipeline=pipeline,
                raw_dir=output_root,
                emissions_input_path=emissions_outputs["beam_emissions_for_inmap"],
                inmap_study_area_grid_path=emissions_outputs.get("beam_inmap_study_area_grid"),
            )
            logger.info("InMAP concentrations complete: wrote %s", concentration_path)
        else:
            logger.info(
                "InMAP concentrations skipped: inmap_enabled=%s beam_emissions_for_inmap=%s",
                pipeline.inmap_enabled,
                emissions_outputs.get("beam_emissions_for_inmap"),
            )
        if pipeline.aermod_enabled and pipeline.asrv_patterns_file and emissions_outputs.get("beam_emissions_for_aermod"):
            _log_step_banner("STEP 3", "aermod concentrations")
            logger.info("Using Step 3 implementation: aermod_concentrations_and_export")
            _, _, aermod_concentration_path = run_aermod_dispersion(
                pipeline=pipeline,
                raw_dir=output_root,
                emissions_input_path=emissions_outputs["beam_emissions_for_aermod"],
            )
            logger.info("AERMOD concentrations complete: wrote %s", aermod_concentration_path)
        else:
            logger.info(
                "AERMOD concentrations skipped: aermod_enabled=%s asrv_patterns_file=%s beam_emissions_for_aermod=%s",
                pipeline.aermod_enabled,
                pipeline.asrv_patterns_file,
                emissions_outputs.get("beam_emissions_for_aermod"),
            )
        if concentration_path is not None:
            _log_step_banner("STEP 4", "prepare exposure")
            logger.info("Using Step 4 implementation: prepare_exposure")
            _, exposure_grid_path, population_distribution_path, population_counts_path = run_prepare_exposure(
                pipeline=pipeline,
                raw_dir=output_root,
                inmap_concentrations_path=str(concentration_path),
                aermod_concentrations_path=str(aermod_concentration_path) if aermod_concentration_path else None,
                population_inputs=population_inputs,
            )
            logger.info("Concentration distribution complete: wrote %s", exposure_grid_path)
            if population_distribution_path is not None:
                logger.info("Population distribution complete: wrote %s", population_distribution_path)
            if population_counts_path is not None:
                logger.info("Population counts complete: wrote %s", population_counts_path)
        else:
            logger.info("Exposure preparation skipped: beam_inmap_concentrations was not produced")
    else:
        logger.info("Dispersion skipped")

    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "input_manifest_path": str(Path(input_manifest_path).resolve()),
        "output_dir": str(output_root),
        "outputs_dir": str(output_root),
        "command": " ".join(sys.argv),
        "image": "not_recorded",
        "outputs": {
            "skims_emissions": prepared_skims_path,
            "county_intersection": grid_intersection_paths.get("county"),
            "inmap_intersection": grid_intersection_paths.get("inmap"),
            "aermod_intersection": grid_intersection_paths.get("aermod"),
            "aermod_full_grid": pipeline.aermod_full_grid_path,
            **emissions_outputs,
            "beam_inmap_concentrations": str(concentration_path) if concentration_path else None,
            "beam_inmap_concentrations_gpkg": (
                str(concentration_path.with_suffix(".gpkg")) if concentration_path else None
            ),
            "beam_aermod_concentrations": str(aermod_concentration_path) if aermod_concentration_path else None,
            "beam_aermod_concentrations_gpkg": (
                str(aermod_concentration_path.with_suffix(".gpkg")) if aermod_concentration_path else None
            ),
            "beam_concentration_distribution": str(exposure_grid_path) if exposure_grid_path else None,
            "beam_concentration_distribution_gpkg": (
                str(exposure_grid_path.with_suffix(".gpkg")) if exposure_grid_path else None
            ),
            "beam_population_distribution": str(population_distribution_path) if population_distribution_path else None,
            "beam_population_counts": (
                str(population_counts_path) if population_counts_path else None
            ),
            "beam_population_counts_gpkg": (
                str(population_counts_path.with_suffix(".gpkg"))
                if population_counts_path
                else None
            ),
        },
        "pipeline": pipeline.to_dict(),
        "population_inputs": population_inputs,
        "deterministic_contract": {
            "uses_only_manifest_paths": True,
            "uses_baked_work_data": False,
        },
        "execution": {
            "dispersion_completed": run_dispersion,
            "stopped_after": (
                "step4_prepare_exposure"
                if exposure_grid_path is not None or population_distribution_path is not None
                else (
                    "step3_compute_aermod_concentrations"
                    if aermod_concentration_path is not None
                    else (
                        "step2_compute_inmap_concentrations"
                        if concentration_path is not None
                        else "step1_process_emissions"
                    )
                )
            ),
        },
    }
    output_manifest = Path(run_manifest_path) if run_manifest_path else output_root / "run_manifest.yaml"
    run_manifest["run_manifest_path"] = str(output_manifest)
    typed_manifest = RunManifest.from_dict(run_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Run manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def run_from_settings(
    settings_path: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
) -> Dict[str, Any]:
    from impacts.preprocessor import preprocess_workflow

    preprocess_manifest = preprocess_workflow(
        settings_path=settings_path,
    )
    return run_from_input_manifest(
        input_manifest_path=preprocess_manifest["inputs_manifest_path"],
        output_dir=preprocess_manifest["input_dir"],
        run_manifest_path=run_manifest_path,
        run_dispersion=run_dispersion,
    )
