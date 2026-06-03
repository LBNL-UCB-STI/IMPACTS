from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

import pandas as pd

from .common import log_step_banner
from .common import normalize_county_fips
from .common import resolve_required_manifest_input
from .config.settings_builder import load_settings_from_yaml
from .manifest.file_ops import load_structured_file
from .manifest.file_ops import resolve_path
from .manifest.file_ops import write_structured_file
from .manifest.schema import PostprocessManifest
from .manifest.schema import PipelineManifest
from .manifest.schema import PreprocessManifest

logger = logging.getLogger(__name__)


def _humanize_target_name(name: str) -> str:
    return str(name).strip().replace("_", " ").replace("-", " ").title()


def _resolve_settings_path(run_manifest_path: str | Path) -> Path:
    run_manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    preprocess_manifest_path = run_manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Postprocess requires preprocess_manifest_path in pipeline manifest.")
    preprocess_manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    settings_source = preprocess_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Postprocess requires settings_source in preprocess manifest.")
    return Path(settings_source).resolve()


def _resolve_pipeline_manifest_path(settings_path: str | Path) -> Path:
    candidate = (
        Path(resolve_path(load_settings_from_yaml(settings_path).impacts.local_output_folder, settings_path)).resolve()
        / "pipeline_manifest.yaml"
    )
    if not candidate.exists():
        raise FileNotFoundError(
            "Postprocess requires workflow pipeline_manifest.yaml in the configured impacts.local_output_folder. "
            f"Expected {candidate}."
        )
    return candidate


def _load_pipeline_manifest(settings_path: str | Path) -> tuple[Path, dict[str, Any]]:
    run_manifest_path = _resolve_pipeline_manifest_path(settings_path)
    run_manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    return run_manifest_path, run_manifest


def _load_context(settings_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest_path, run_manifest = _load_pipeline_manifest(settings_path)
    preprocess_manifest_path = run_manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Postprocess requires preprocess_manifest_path in pipeline manifest.")
    preprocess_manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    inputs = preprocess_manifest.get("inputs", {}) or {}
    return run_manifest_path, run_manifest, preprocess_manifest, inputs


def _resolve_modeled_emissions_path(settings_path: str | Path) -> Path:
    _, run_manifest = _load_pipeline_manifest(settings_path)
    candidate_raw = run_manifest.get("outputs", {}).get("beam_emissions_by_county_process")
    if not candidate_raw:
        raise ValueError("Postprocess requires beam_emissions_by_county_process in run_manifest.outputs.")
    candidate = Path(candidate_raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires county-intersected workflow emissions outputs. Expected {candidate}."
        )
    return candidate


def _resolve_skims_emissions_path(settings_path: str | Path) -> Path:
    _, run_manifest = _load_pipeline_manifest(settings_path)
    candidate_raw = run_manifest.get("outputs", {}).get("skims_emissions")
    if not candidate_raw:
        raise ValueError("Postprocess requires skims_emissions in run_manifest.outputs.")
    candidate = Path(candidate_raw).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires prepared skims output from the workflow run manifest. Expected {candidate}."
        )
    return candidate


def _resolve_county_boundaries_path(settings_path: str | Path) -> Path:
    _, _, _, inputs = _load_context(settings_path)
    candidate = Path(resolve_required_manifest_input(inputs, key="county_boundaries")).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires staged county boundaries from preprocess. Expected {candidate}."
        )
    return candidate


def _resolve_vehicle_types_paths(settings_path: str | Path) -> tuple[Path, Path]:
    _, _, _, inputs = _load_context(settings_path)
    passenger_candidate = Path(resolve_required_manifest_input(inputs, key="passenger_vehicle_types_input")).resolve()
    freight_candidate = Path(resolve_required_manifest_input(inputs, key="freight_vehicle_types_input")).resolve()
    if not passenger_candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires staged passenger vehicle types from preprocess. Expected {passenger_candidate}."
        )
    if not freight_candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires staged freight vehicle types from preprocess. Expected {freight_candidate}."
        )
    return passenger_candidate, freight_candidate


def _resolve_optional_population_assignment_paths(
    settings_path: str | Path,
) -> tuple[Path | None, Path | None]:
    passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_vehicle_types_paths(settings_path)
    passenger_candidates = list(
        passenger_vehicle_types_path.parents[1].glob("urbansim/**/vehicles--*--EM.parquet")
    )
    passenger_vehicles_path = passenger_candidates[0].resolve() if passenger_candidates else None
    freight_candidates = list(
        freight_vehicle_types_path.parents[1].glob("**/carriers--*--EM.parquet")
    )
    freight_carriers_path = freight_candidates[0].resolve() if freight_candidates else None
    return passenger_vehicles_path, freight_carriers_path


def _resolve_inventory_target_path(settings_path: str | Path, raw: str) -> Path:
    raw_text = str(raw).strip()
    candidate = Path(resolve_path(raw_text, settings_path) or raw_text).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Postprocess inventory target file was configured but not found. Expected {candidate}."
    )


def _resolve_inventory_emfacid_activity_path(
    settings_path: str | Path,
    *,
    manifest_key: str,
) -> Path:
    _, _, _, inputs = _load_context(settings_path)
    candidate = Path(resolve_required_manifest_input(inputs, key=manifest_key)).resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess step 1 requires the EMFAC activity-by-emfacId file registered as "
            f"'{manifest_key}', but it was not found at {candidate}. "
            f"Re-run the EMFAC activities workflow: python -m impacts activities --config <config>"
        )
    return candidate


def _resolve_vehicle_category_metadata_path(settings_path: str | Path) -> Path:
    _, _, _, inputs = _load_context(settings_path)
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
            "Postprocess step 2 requires impacts.emissions.vehicle_category_metadata_file when annual sector targets are configured."
        )
    return _resolve_inventory_target_path(
        settings_path,
        settings.impacts.emissions.vehicle_category_metadata_file,
    )


def _run_postprocess_steps(settings_path: str | Path) -> Dict[str, str]:
    from .pipeline.postprocess.step1_compare_fleet import run as run_step1
    from .pipeline.postprocess.step2_compare_annual_targets import run as run_step2
    from .pipeline.postprocess.step3_compare_emissions_inventory import run as run_step3

    settings = load_settings_from_yaml(settings_path)
    output_dir = (
        Path(resolve_path(settings.impacts.local_output_folder, settings_path)).resolve() / "postprocess"
    )
    modeled_emissions_path = _resolve_modeled_emissions_path(settings_path)
    skims_emissions_path = _resolve_skims_emissions_path(settings_path)
    outputs: Dict[str, str] = {}
    passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_vehicle_types_paths(settings_path)
    passenger_vehicles_path, freight_carriers_path = _resolve_optional_population_assignment_paths(settings_path)
    passenger_activity_path = _resolve_inventory_emfacid_activity_path(
        settings_path,
        manifest_key="passenger_inventory_emfacid_file",
    )
    freight_activity_path = _resolve_inventory_emfacid_activity_path(
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
            vehicle_category_metadata_file=str(_resolve_vehicle_category_metadata_path(settings_path)),
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
        logger.info("Postprocess steps complete: output_dir=%s outputs=%d", output_dir, len(outputs))
        return outputs
    county_boundaries_path = _resolve_county_boundaries_path(settings_path)
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
    inventory_path = _resolve_inventory_target_path(settings_path, settings.impacts.analysis.inventory_file)
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
    logger.info("Postprocess steps complete: output_dir=%s outputs=%d", output_dir, len(outputs))
    return outputs


def postprocess_from_pipeline_manifest(
    run_manifest_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    pipeline_manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    output_root = Path(str(pipeline_manifest.get("output_dir"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log_step_banner("Postprocess", "Run Postprocess Steps", logger=logger)
    settings_path = _resolve_settings_path(run_manifest_path)
    postprocess_outputs = _run_postprocess_steps(settings_path)

    postprocess_manifest = {
        "contract_version": pipeline_manifest.get("contract_version", "1"),
        "model": "impacts",
        "pipeline_manifest_path": str(Path(run_manifest_path).resolve()),
        "output_dir": str(output_root),
        "postprocess_outputs": postprocess_outputs,
        "validation": {
            "completed": True,
        },
        "notes": [
            "Postprocess runs maintained outputs after workflow completion.",
            "Pipeline completion is logged after postprocess steps and manifest writing.",
        ],
    }
    output_manifest = Path(manifest_path) if manifest_path else output_root / "postprocess_manifest.yaml"
    postprocess_manifest["postprocess_manifest_path"] = str(output_manifest)
    typed_manifest = PostprocessManifest.from_dict(postprocess_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Postprocess manifest written: %s", output_manifest)
    log_step_banner("Pipeline", "Impacts Complete", logger=logger)
    logger.info("Pipeline complete: postprocess_manifest=%s", output_manifest)
    return typed_manifest.to_dict()


def postprocess_from_settings(
    settings_path: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    from impacts.preprocessor import preprocess_workflow
    from impacts.runner import run_aermod_from_pipeline_manifest
    from impacts.runner import run_emissions_from_pipeline_manifest
    from impacts.runner import run_exposure_from_pipeline_manifest
    from impacts.runner import run_inmap_from_pipeline_manifest

    settings = load_settings_from_yaml(settings_path)
    preprocess_manifest = preprocess_workflow(
        settings_path=settings_path,
    )
    run_manifest_path = preprocess_manifest["pipeline_manifest_path"]
    if settings.impacts.pipeline.postsim.emissions:
        run_manifest = run_emissions_from_pipeline_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["pipeline_manifest_path"]
    if settings.impacts.pipeline.postsim.inmap:
        run_manifest = run_inmap_from_pipeline_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["pipeline_manifest_path"]
    if settings.impacts.pipeline.postsim.aermod:
        run_manifest = run_aermod_from_pipeline_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["pipeline_manifest_path"]
    if settings.impacts.pipeline.postsim.exposure:
        run_manifest = run_exposure_from_pipeline_manifest(
            run_manifest_path=run_manifest_path,
        )
        run_manifest_path = run_manifest["pipeline_manifest_path"]
    return postprocess_from_pipeline_manifest(
        run_manifest_path=run_manifest_path,
        manifest_path=manifest_path,
    )
