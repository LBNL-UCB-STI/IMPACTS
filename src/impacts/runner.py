from __future__ import annotations

import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional

import pandas as pd

from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import PreprocessManifest
from .manifest.schema import PipelineConfig
from .manifest.schema import PipelineManifest
from .common import resolve_required_manifest_input

logger = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[2]

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


def _record_stage_timing(stage_timings: dict[str, float], key: str, started_at: float) -> None:
    stage_timings[key] = round(time.perf_counter() - started_at, 2)


def _log_stage_timing_summary(stage_timings: dict[str, float]) -> None:
    if not stage_timings:
        return
    summary = ", ".join(f"{stage}={seconds:.2f}s" for stage, seconds in stage_timings.items())
    logger.info("Stage timing summary: %s", summary)


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


def _resolve_existing_run_manifest_path(
    *,
    output_root: Path,
    run_manifest_path: str | Path | None,
) -> Path | None:
    if run_manifest_path:
        candidate = Path(run_manifest_path).resolve()
        return candidate if candidate.exists() else None
    candidate = output_root / "pipeline_manifest.yaml"
    return candidate if candidate.exists() else None


def _load_existing_run_manifest(
    *,
    output_root: Path,
    run_manifest_path: str | Path | None,
) -> Dict[str, Any]:
    candidate = _resolve_existing_run_manifest_path(output_root=output_root, run_manifest_path=run_manifest_path)
    if candidate is None:
        return {}
    return PipelineManifest.from_dict(load_structured_file(candidate)).to_dict()


def _load_run_manifest_context(
    run_manifest_path: str | Path,
) -> tuple[dict[str, Any], Path, str]:
    manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    preprocess_manifest_path = manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Pipeline manifest is missing preprocess_manifest_path.")
    output_dir = manifest.get("output_dir")
    if not output_dir:
        raise ValueError("Pipeline manifest is missing output_dir.")
    return manifest, Path(str(output_dir)).resolve(), str(preprocess_manifest_path)


def _resolve_staged_intersection_paths(
    *,
    manifest_inputs: Dict[str, Any],
    pipeline: PipelineConfig,
) -> Dict[str, Optional[str]]:
    paths = {
        "county": resolve_required_manifest_input(manifest_inputs, key="county_intersection"),
        "inmap": (
            resolve_required_manifest_input(manifest_inputs, key="inmap_intersection")
            if pipeline.inmap_enabled
            else None
        ),
        "aermod": (
            resolve_required_manifest_input(manifest_inputs, key="aermod_intersection")
            if pipeline.aermod_enabled
            else None
        ),
    }
    for zone_label, path in paths.items():
        if not path:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Run requires staged {zone_label} intersection output from preprocess, but it was not found: {path}"
            )
    return paths


def _humanize_target_name(name: str) -> str:
    return str(name).strip().replace("_", " ").replace("-", " ").title()


def _resolve_run_manifest_settings_path(run_manifest_path: str | Path) -> Path:
    run_manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    preprocess_manifest_path = run_manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Analysis requires preprocess_manifest_path in pipeline manifest.")
    preprocess_manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    settings_source = preprocess_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Analysis requires settings_source in preprocess manifest.")
    return Path(settings_source).resolve()


def _resolve_analysis_run_manifest_path(settings_path: str | Path) -> Path:
    candidate = Path(resolve_path(load_settings_from_yaml(settings_path).impacts.local_output_folder, settings_path)).resolve() / "pipeline_manifest.yaml"
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires workflow pipeline_manifest.yaml in the configured impacts.local_output_folder. "
            f"Expected {candidate}."
        )
    return candidate


def _load_analysis_run_manifest(settings_path: str | Path) -> tuple[Path, dict[str, Any]]:
    run_manifest_path = _resolve_analysis_run_manifest_path(settings_path)
    run_manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    return run_manifest_path, run_manifest


def _load_analysis_context(settings_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest_path, run_manifest = _load_analysis_run_manifest(settings_path)
    preprocess_manifest_path = run_manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Analysis requires preprocess_manifest_path in pipeline manifest.")
    preprocess_manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    inputs = preprocess_manifest.get("inputs", {}) or {}
    return run_manifest_path, run_manifest, preprocess_manifest, inputs


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


def _resolve_analysis_skims_emissions_path(settings_path: str | Path) -> Path:
    _, run_manifest = _load_analysis_run_manifest(settings_path)
    candidate_raw = run_manifest.get("outputs", {}).get("skims_emissions")
    if not candidate_raw:
        raise ValueError("Analysis requires skims_emissions in run_manifest.outputs.")
    candidate = Path(candidate_raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            "Analysis requires prepared skims output from the workflow run manifest. "
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


def _resolve_optional_analysis_population_assignment_paths(
    settings_path: str | Path,
) -> tuple[Path | None, Path | None]:
    passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_analysis_vehicle_types_paths(settings_path)
    passenger_candidates = list(
        passenger_vehicle_types_path.parents[1].glob("urbansim/**/vehicles--*--EM.parquet")
    )
    passenger_vehicles_path = passenger_candidates[0].resolve() if passenger_candidates else None
    freight_candidates = list(
        freight_vehicle_types_path.parents[1].glob("**/carriers--*--EM.parquet")
    )
    freight_carriers_path = freight_candidates[0].resolve() if freight_candidates else None
    return passenger_vehicles_path, freight_carriers_path


def _resolve_analysis_inventory_target_path(settings_path: str | Path, raw: str) -> Path:
    raw_text = str(raw).strip()
    candidate = Path(resolve_path(raw_text, settings_path) or raw_text).resolve()
    if candidate.exists():
        return candidate
    fallback = (_REPO_ROOT / raw_text).resolve()
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Analysis inventory target file was configured but not found. "
        f"Tried {candidate} and {fallback}."
    )


def _resolve_analysis_inventory_emfacid_activity_path(
    settings_path: str | Path,
    *,
    manifest_key: str,
) -> Path:
    _, _, _, inputs = _load_analysis_context(settings_path)
    candidate = Path(resolve_required_manifest_input(inputs, key=manifest_key)).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Analysis Step 1 requires the EMFAC activity-by-emfacId file registered as "
            f"'{manifest_key}', but it was not found at {candidate}. "
            f"Re-run the EMFAC activities workflow: python -m impacts activities --config <config>"
        )
    return candidate


def _resolve_analysis_vehicle_category_metadata_path(settings_path: str | Path) -> Path:
    _, _, _, inputs = _load_analysis_context(settings_path)
    for key in ("vehicle_category_metadata_file_input",):
        try:
            resolved = Path(resolve_required_manifest_input(inputs, key=key)).resolve()
        except ValueError:
            continue
        if resolved.exists():
            return resolved
    settings = load_settings_from_yaml(settings_path)
    if not settings.impacts.emissions.vehicle_category_metadata_file:
        raise ValueError(
            "Analysis Step 2 requires impacts.emissions.vehicle_category_metadata_file when annual sector targets are configured."
        )
    return _resolve_analysis_inventory_target_path(
        settings_path,
        settings.impacts.emissions.vehicle_category_metadata_file,
    )


def run_analysis_from_settings(
    *,
    settings_path: str | Path,
) -> Dict[str, str]:
    from .analysis.step1_compare_fleet import run as run_step1
    from .analysis.step2_compare_annual_targets import run as run_step2
    from .analysis.step3_compare_emissions_inventory import run as run_step3
    from .common import normalize_county_fips

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    settings = load_settings_from_yaml(settings_path)
    output_dir = Path(resolve_path(settings.impacts.local_output_folder, settings_path)).resolve() / "postprocess" / "analysis"
    modeled_emissions_path = _resolve_analysis_modeled_emissions_path(settings_path)
    skims_emissions_path = _resolve_analysis_skims_emissions_path(settings_path)
    outputs: Dict[str, str] = {}
    passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_analysis_vehicle_types_paths(settings_path)
    passenger_vehicles_path, freight_carriers_path = _resolve_optional_analysis_population_assignment_paths(settings_path)
    passenger_activity_path = _resolve_analysis_inventory_emfacid_activity_path(
        settings_path,
        manifest_key="passenger_inventory_emfacid_file",
    )
    freight_activity_path = _resolve_analysis_inventory_emfacid_activity_path(
        settings_path,
        manifest_key="freight_inventory_emfacid_file",
    )
    fleet_outputs = run_step1(
        skims_emissions_path=str(skims_emissions_path),
        passenger_vehicle_types_path=str(passenger_vehicle_types_path),
        freight_vehicle_types_path=str(freight_vehicle_types_path),
        emfac_passenger_activity_path=str(passenger_activity_path),
        emfac_freight_activity_path=str(freight_activity_path),
        output_dir=output_dir / "fleet",
        passenger_vehicles_path=str(passenger_vehicles_path) if passenger_vehicles_path else None,
        freight_carriers_path=str(freight_carriers_path) if freight_carriers_path else None,
    )
    for key, value in fleet_outputs.items():
        outputs[f"fleet_{key}"] = value
    if settings.impacts.analysis.sector_targets:
        target_outputs = run_step2(
            modeled_emissions_path=str(modeled_emissions_path),
            passenger_vehicle_types_path=str(passenger_vehicle_types_path),
            freight_vehicle_types_path=str(freight_vehicle_types_path),
            vehicle_category_metadata_file=str(_resolve_analysis_vehicle_category_metadata_path(settings_path)),
            output_dir=output_dir / "annual_targets",
            sector_targets=[
                {
                    "source": target.source,
                    "sector": target.sector,
                    "annual_pm25_short_tons": target.annual_pm25_short_tons,
                    "annual_nox_short_tons": target.annual_nox_short_tons,
                    "annual_pm10_short_tons": target.annual_pm10_short_tons,
                    "annual_tog_short_tons": target.annual_tog_short_tons,
                    "annual_rog_short_tons": target.annual_rog_short_tons,
                    "annual_co_short_tons": target.annual_co_short_tons,
                    "annual_sox_short_tons": target.annual_sox_short_tons,
                }
                for target in settings.impacts.analysis.sector_targets
            ],
        )
        for key, value in target_outputs.items():
            outputs[f"annual_targets_{key}"] = value
    if not settings.impacts.analysis.inventory_targets:
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
    for target in settings.impacts.analysis.inventory_targets:
        target_outputs = run_step3(
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


def run_analysis_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, str]:
    return run_analysis_from_settings(settings_path=_resolve_run_manifest_settings_path(run_manifest_path))


def _run_stages_from_preprocess_manifest(
    preprocess_manifest_path: str | Path,
    run_manifest_path: str | Path | None = None,
    run_dispersion: bool = False,
    run_emissions: bool | None = None,
    run_inmap: bool | None = None,
    run_aermod: bool | None = None,
    run_exposure: bool | None = None,
) -> Dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=False,
    )
    manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    pipeline = PipelineConfig.from_dict(manifest.get("pipeline", {}) or {})
    population_inputs = manifest.get("population_inputs", {}) or {}
    manifest_inputs = manifest.get("inputs", {}) or {}
    input_root = Path(manifest.get("input_dir", "")).resolve()

    output_root = _resolve_runtime_output_root(
        input_manifest=manifest,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "emissions").mkdir(parents=True, exist_ok=True)
    (output_root / "concentrations").mkdir(parents=True, exist_ok=True)
    (output_root / "exposure").mkdir(parents=True, exist_ok=True)
    logger.info("Loaded preprocess manifest: %s", Path(preprocess_manifest_path).resolve())
    logger.info("Output directory: %s", output_root)

    from .common import prepared_table_target
    stage_timings: dict[str, float] = {}
    grid_intersection_paths = _resolve_staged_intersection_paths(
        manifest_inputs=manifest_inputs,
        pipeline=pipeline,
    )
    existing_run_manifest = _load_existing_run_manifest(
        output_root=output_root,
        run_manifest_path=run_manifest_path,
    )
    existing_outputs = dict(existing_run_manifest.get("outputs", {}) or {})

    execute_emissions = pipeline.emissions_enabled if run_emissions is None else bool(run_emissions)
    execute_inmap = pipeline.inmap_enabled if run_inmap is None else bool(run_inmap)
    execute_aermod = pipeline.aermod_enabled if run_aermod is None else bool(run_aermod)
    execute_exposure = pipeline.exposure_enabled if run_exposure is None else bool(run_exposure)
    if run_dispersion:
        execute_inmap = execute_inmap and pipeline.inmap_enabled
        execute_aermod = execute_aermod and pipeline.aermod_enabled
        execute_exposure = execute_exposure and pipeline.exposure_enabled
    elif run_inmap is None and run_aermod is None and run_exposure is None:
        execute_inmap = False
        execute_aermod = False
        execute_exposure = False

    emissions_outputs = {
        key: existing_outputs.get(key)
        for key in (
            "beam_emissions_by_county_process",
            "beam_emissions_for_inmap",
            "beam_inmap_study_area_grid",
            "beam_emissions_for_aermod",
        )
    }
    if execute_emissions:
        from .pipeline.workflow.step1_process_emissions import run as run_emissions_processing

        _log_step_banner("STEP 1", "emissions processing")
        logger.info("Using Step 1 implementation: emissions_processing")
        stage_started = time.perf_counter()
        emissions_outputs = run_emissions_processing(
            pipeline,
            output_root / "emissions",
            output_root / "emissions",
            grid_intersection_paths,
            manifest_inputs=manifest_inputs,
        )
        _record_stage_timing(stage_timings, "step1_process_emissions", stage_started)
    else:
        logger.info("Emissions processing skipped")
    prepared_skims_candidate = prepared_table_target(output_root / "emissions", "prepared_skims_for_grid_allocation")
    prepared_skims_path = str(prepared_skims_candidate) if prepared_skims_candidate.exists() else None

    concentration_path = (
        Path(existing_outputs["beam_inmap_concentrations"]).resolve()
        if existing_outputs.get("beam_inmap_concentrations")
        else None
    )
    aermod_concentration_path = (
        Path(existing_outputs["beam_aermod_concentrations"]).resolve()
        if existing_outputs.get("beam_aermod_concentrations")
        else None
    )
    exposure_grid_path = (
        Path(existing_outputs["beam_concentration_distribution"]).resolve()
        if existing_outputs.get("beam_concentration_distribution")
        else None
    )
    population_distribution_path = (
        Path(existing_outputs["beam_population_distribution"]).resolve()
        if existing_outputs.get("beam_population_distribution")
        else None
    )
    population_counts_path = (
        Path(existing_outputs["beam_population_counts"]).resolve()
        if existing_outputs.get("beam_population_counts")
        else None
    )
    if execute_inmap or execute_aermod or execute_exposure:
        from .pipeline.workflow.step2_compute_inmap_concentrations import run as run_inmap_dispersion
        from .pipeline.workflow.step3_compute_aermod_concentrations import run as run_aermod_dispersion
        from .pipeline.workflow.step4_prepare_exposure import run as run_prepare_exposure
        if execute_inmap and emissions_outputs.get("beam_emissions_for_inmap"):
            _log_step_banner("STEP 2", "inmap concentrations")
            logger.info("Using Step 2 implementation: inmap_concentrations_and_export")
            stage_started = time.perf_counter()
            _, _, concentration_path = run_inmap_dispersion(
                pipeline=pipeline,
                raw_dir=output_root / "concentrations",
                emissions_input_path=emissions_outputs["beam_emissions_for_inmap"],
                inmap_study_area_grid_path=emissions_outputs.get("beam_inmap_study_area_grid"),
            )
            _record_stage_timing(stage_timings, "step2_compute_inmap_concentrations", stage_started)
            logger.info("InMAP concentrations complete: wrote %s", concentration_path)
        else:
            logger.info(
                "InMAP concentrations skipped: inmap_enabled=%s beam_emissions_for_inmap=%s",
                execute_inmap,
                emissions_outputs.get("beam_emissions_for_inmap"),
            )
        if execute_aermod and pipeline.asrv_patterns_file and emissions_outputs.get("beam_emissions_for_aermod"):
            _log_step_banner("STEP 3", "aermod concentrations")
            logger.info("Using Step 3 implementation: aermod_concentrations_and_export")
            stage_started = time.perf_counter()
            _, _, aermod_concentration_path = run_aermod_dispersion(
                pipeline=pipeline,
                raw_dir=output_root / "concentrations",
                cache_dir=output_root / "_tmp",
                emissions_input_path=emissions_outputs["beam_emissions_for_aermod"],
            )
            _record_stage_timing(stage_timings, "step3_compute_aermod_concentrations", stage_started)
            logger.info("AERMOD concentrations complete: wrote %s", aermod_concentration_path)
        else:
            logger.info(
                "AERMOD concentrations skipped: aermod_enabled=%s asrv_patterns_file=%s beam_emissions_for_aermod=%s",
                execute_aermod,
                pipeline.asrv_patterns_file,
                emissions_outputs.get("beam_emissions_for_aermod"),
            )
        if execute_exposure and concentration_path is not None:
            _log_step_banner("STEP 4", "prepare exposure")
            logger.info("Using Step 4 implementation: prepare_exposure")
            stage_started = time.perf_counter()
            _, exposure_grid_path, population_distribution_path, population_counts_path = run_prepare_exposure(
                pipeline=pipeline,
                raw_dir=output_root / "exposure",
                inmap_concentrations_path=str(concentration_path),
                aermod_concentrations_path=str(aermod_concentration_path) if aermod_concentration_path else None,
                manifest_inputs=manifest_inputs,
            )
            _record_stage_timing(stage_timings, "step4_prepare_exposure", stage_started)
            logger.info("Concentration distribution complete: wrote %s", exposure_grid_path)
            if population_distribution_path is not None:
                logger.info("Population distribution complete: wrote %s", population_distribution_path)
            if population_counts_path is not None:
                logger.info("Population counts complete: wrote %s", population_counts_path)
        else:
            logger.info(
                "Exposure preparation skipped: exposure_enabled=%s beam_inmap_concentrations=%s",
                execute_exposure,
                concentration_path,
            )
    else:
        logger.info("Concentration and exposure stages skipped")

    _log_stage_timing_summary(stage_timings)
    run_manifest = {
        "contract_version": manifest.get("contract_version", "1"),
        "model": "impacts",
        "preprocess_manifest_path": str(Path(preprocess_manifest_path).resolve()),
        "output_dir": str(output_root),
        "command": " ".join(sys.argv),
        "image": "not_recorded",
        "outputs": {
            **existing_outputs,
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
            "dispersion_completed": bool(execute_inmap or execute_aermod),
            "stage_timings_seconds": stage_timings,
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
    shutil.rmtree(output_root / "_tmp", ignore_errors=True)
    output_manifest = Path(run_manifest_path) if run_manifest_path else output_root / "pipeline_manifest.yaml"
    run_manifest["pipeline_manifest_path"] = str(output_manifest)
    typed_manifest = PipelineManifest.from_dict(run_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Pipeline manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def run_emissions_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=True,
        run_inmap=False,
        run_aermod=False,
        run_exposure=False,
    )


def run_inmap_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=False,
        run_inmap=True,
        run_aermod=False,
        run_exposure=False,
    )


def run_aermod_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=False,
        run_inmap=False,
        run_aermod=True,
        run_exposure=False,
    )


def run_exposure_from_pipeline_manifest(
    *,
    run_manifest_path: str | Path,
) -> Dict[str, Any]:
    _, _, preprocess_manifest_path = _load_run_manifest_context(run_manifest_path)
    return _run_stages_from_preprocess_manifest(
        preprocess_manifest_path=preprocess_manifest_path,
        run_manifest_path=run_manifest_path,
        run_emissions=False,
        run_inmap=False,
        run_aermod=False,
        run_exposure=True,
    )
