from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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

_OUTPUT_PATH_MARKERS = (
    "activities",
    "preprocess",
    "emissions",
    "concentrations",
    "exposure",
    "postprocess",
)


def _humanize_target_name(name: str) -> str:
    return str(name).strip().replace("_", " ").replace("-", " ").title()


def _localize_output_path(raw: str | Path, output_root: Path | None) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.exists() or output_root is None:
        return candidate.resolve()

    output_root = Path(output_root).resolve()
    direct_candidate = output_root / candidate.name
    if direct_candidate.exists():
        return direct_candidate.resolve()

    parts = candidate.parts
    for marker in _OUTPUT_PATH_MARKERS:
        if marker not in parts:
            continue
        marker_index = parts.index(marker)
        localized = output_root.joinpath(*parts[marker_index:])
        if localized.exists():
            return localized.resolve()

    return candidate.resolve()


def _normalize_input_roots(input_roots: tuple[str | Path, ...] | list[str | Path] | None) -> tuple[Path, ...]:
    return tuple(Path(root).expanduser().resolve() for root in (input_roots or ()))


def _localize_input_path(raw: str | Path, input_roots: tuple[Path, ...]) -> Path:
    candidate = Path(str(raw)).expanduser()
    if candidate.exists() or not input_roots:
        return candidate.resolve()

    parts = candidate.parts
    for input_root in input_roots:
        if not candidate.is_absolute():
            localized = input_root / candidate
            if localized.exists():
                return localized.resolve()

        for index, part in enumerate(parts):
            if part == input_root.name:
                localized = input_root.joinpath(*parts[index + 1:])
                if localized.exists():
                    return localized.resolve()

        for marker in ("vehicle-tech", "urbansim", "freight", "r5"):
            if marker not in parts:
                continue
            localized = input_root.joinpath(*parts[parts.index(marker):])
            if localized.exists():
                return localized.resolve()

    return candidate.resolve()


def _localize_path(
    raw: str | Path,
    *,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    candidate = _localize_output_path(raw, output_root)
    if candidate.exists():
        return candidate
    return _localize_input_path(raw, input_roots)


def _localized_pipeline_manifest(
    run_manifest_path: str | Path,
    *,
    output_root: Path | None = None,
) -> dict[str, Any]:
    run_manifest = PipelineManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    if output_root is None:
        return run_manifest

    output_root = Path(output_root).resolve()
    run_manifest["output_dir"] = str(output_root)
    if run_manifest.get("preprocess_manifest_path"):
        run_manifest["preprocess_manifest_path"] = str(
            _localize_output_path(run_manifest["preprocess_manifest_path"], output_root)
        )
    outputs = run_manifest.get("outputs", {}) or {}
    run_manifest["outputs"] = {
        key: str(_localize_output_path(value, output_root)) if value else value
        for key, value in outputs.items()
    }
    return run_manifest


def _resolve_settings_path(run_manifest_path: str | Path, *, output_root: Path | None = None) -> Path:
    run_manifest = _localized_pipeline_manifest(run_manifest_path, output_root=output_root)
    preprocess_manifest_path = run_manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Postprocess requires preprocess_manifest_path in pipeline manifest.")
    preprocess_manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    settings_source = preprocess_manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Postprocess requires settings_source in preprocess manifest.")
    if output_root is None:
        return Path(settings_source).resolve()

    settings_path = _localize_output_path(settings_source, output_root)
    if settings_path.exists():
        return settings_path

    inputs = preprocess_manifest.get("inputs", {}) or {}
    settings_entry = inputs.get("settings", {}) or {}
    for field in ("staged_path", "source_path", "path"):
        raw = settings_entry.get(field)
        if not raw:
            continue
        candidate = _localize_output_path(raw, output_root)
        if candidate.exists():
            return candidate

    local_settings_path = Path(output_root) / "settings.yaml"
    if local_settings_path.exists():
        return local_settings_path.resolve()
    return settings_path


def _resolve_pipeline_manifest_path(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
) -> Path:
    if run_manifest_path is not None:
        candidate = Path(run_manifest_path).resolve()
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Postprocess run manifest was not found: {candidate}")

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


def _load_pipeline_manifest(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_manifest_path = _resolve_pipeline_manifest_path(settings_path, run_manifest_path=run_manifest_path)
    run_manifest = _localized_pipeline_manifest(run_manifest_path, output_root=output_root)
    return run_manifest_path, run_manifest


def _coerce_optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def _resolve_postprocess_sample_config(
    settings,
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
) -> tuple[float, float, float | None]:
    population_settings = getattr(settings.impacts, "population", None)
    population_sample = float(getattr(population_settings, "population_sample", 1.0))
    transit_sample = float(getattr(population_settings, "transit_sample", 1.0))
    freight_sample = _coerce_optional_float(getattr(population_settings, "freight_sample", None))

    try:
        _, run_manifest = _load_pipeline_manifest(
            settings_path,
            run_manifest_path=run_manifest_path,
            output_root=output_root,
        )
    except FileNotFoundError:
        if run_manifest_path is not None:
            raise
        run_manifest = {}

    pipeline = run_manifest.get("pipeline", {}) or {}
    population_sample = float(pipeline.get("population_sample", population_sample))
    transit_sample = float(pipeline.get("transit_sample", transit_sample))
    freight_sample = _coerce_optional_float(pipeline.get("freight_sample", freight_sample))
    return population_sample, transit_sample, freight_sample


def _load_context(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    run_manifest_path, run_manifest = _load_pipeline_manifest(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    preprocess_manifest_path = run_manifest.get("preprocess_manifest_path")
    if not preprocess_manifest_path:
        raise ValueError("Postprocess requires preprocess_manifest_path in pipeline manifest.")
    preprocess_manifest = PreprocessManifest.from_dict(load_structured_file(preprocess_manifest_path)).to_dict()
    inputs = preprocess_manifest.get("inputs", {}) or {}
    return run_manifest_path, run_manifest, preprocess_manifest, inputs


def _load_context_for_resolution(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if run_manifest_path is None and output_root is None:
        return _load_context(settings_path)
    return _load_context(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )


def _resolve_input_path(
    inputs: dict[str, Any],
    *,
    key: str,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    return _localize_path(
        resolve_required_manifest_input(inputs, key=key),
        output_root=output_root,
        input_roots=input_roots,
    )


def _resolve_modeled_emissions_path(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
) -> Path:
    _, run_manifest = _load_pipeline_manifest(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    candidate_raw = run_manifest.get("outputs", {}).get("beam_emissions_by_county_process")
    if not candidate_raw:
        raise ValueError("Postprocess requires beam_emissions_by_county_process in run_manifest.outputs.")
    candidate = _localize_output_path(candidate_raw, output_root)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires county-intersected workflow emissions outputs. Expected {candidate}."
        )
    return candidate


def _resolve_skims_emissions_path(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
) -> Path:
    _, run_manifest = _load_pipeline_manifest(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    candidate_raw = run_manifest.get("outputs", {}).get("skims_emissions")
    if not candidate_raw:
        raise ValueError("Postprocess requires skims_emissions in run_manifest.outputs.")
    candidate = _localize_output_path(candidate_raw, output_root)
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires prepared skims output from the workflow run manifest. Expected {candidate}."
        )
    return candidate


def _resolve_county_boundaries_path(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    _, _, _, inputs = _load_context_for_resolution(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    candidate = _resolve_input_path(
        inputs,
        key="county_boundaries",
        output_root=output_root,
        input_roots=input_roots,
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess requires staged county boundaries from preprocess. Expected {candidate}."
        )
    return candidate


def _resolve_vehicle_types_paths(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> tuple[Path, Path]:
    _, _, _, inputs = _load_context_for_resolution(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    passenger_candidate = _resolve_input_path(
        inputs,
        key="passenger_vehicle_types_input",
        output_root=output_root,
        input_roots=input_roots,
    )
    freight_candidate = _resolve_input_path(
        inputs,
        key="freight_vehicle_types_input",
        output_root=output_root,
        input_roots=input_roots,
    )
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
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> tuple[Path | None, Path | None]:
    passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_vehicle_types_paths(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
        input_roots=input_roots,
    )
    passenger_candidates = list(
        passenger_vehicle_types_path.parents[1].glob("urbansim/**/vehicles--*--EM.parquet")
    )
    passenger_vehicles_path = passenger_candidates[0].resolve() if passenger_candidates else None
    freight_candidates = list(
        freight_vehicle_types_path.parents[1].glob("**/carriers--*--EM.parquet")
    )
    freight_carriers_path = freight_candidates[0].resolve() if freight_candidates else None
    return passenger_vehicles_path, freight_carriers_path


def _resolve_inventory_target_path(
    settings_path: str | Path,
    raw: str,
    *,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    raw_text = str(raw).strip()
    candidate = _localize_path(raw_text, output_root=output_root, input_roots=input_roots)
    if candidate.exists():
        return candidate
    candidate = _localize_path(resolve_path(raw_text, settings_path) or raw_text, output_root=output_root)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Postprocess inventory target file was configured but not found. Expected {candidate}."
    )


def _resolve_delta_baseline_concentration_path(
    settings_path: str | Path,
    raw: str,
    *,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    raw_text = str(raw).strip()
    candidate = _localize_path(raw_text, output_root=output_root, input_roots=input_roots)
    if candidate.exists():
        return candidate
    candidate = _localize_path(resolve_path(raw_text, settings_path) or raw_text, output_root=output_root)
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "Postprocess delta baseline concentration distribution was configured but not found. "
        f"Expected {candidate}."
    )


def _remove_stale_postprocess_delta_outputs(output_dir: Path) -> None:
    for subdir in ("delta_concentrations", "delta_exposure"):
        target_dir = output_dir / subdir
        if not target_dir.exists():
            continue
        for path in sorted(target_dir.glob("*")):
            if path.is_file() and path.suffix.lower() in {".png", ".parquet"}:
                path.unlink()
                logger.info("Removed stale postprocess delta output: %s", path)


def _resolve_inventory_emfacid_activity_path(
    settings_path: str | Path,
    *,
    manifest_key: str,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    _, _, _, inputs = _load_context_for_resolution(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    candidate = _resolve_input_path(
        inputs,
        key=manifest_key,
        output_root=output_root,
        input_roots=input_roots,
    )
    if not candidate.exists():
        raise FileNotFoundError(
            f"Postprocess step 1 requires the EMFAC activity-by-emfacId file registered as "
            f"'{manifest_key}', but it was not found at {candidate}. "
            f"Re-run the EMFAC activities workflow: python -m impacts activities --config <config>"
        )
    return candidate


def _resolve_vehicle_category_metadata_path(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
    input_roots: tuple[Path, ...] = (),
) -> Path:
    _, _, _, inputs = _load_context_for_resolution(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    try:
        resolved = _resolve_input_path(
            inputs,
            key="vehicle_category_metadata_file_input",
            output_root=output_root,
            input_roots=input_roots,
        )
        if resolved.exists():
            return resolved
    except ValueError:
        pass
    settings = load_settings_from_yaml(settings_path)
    if not settings.impacts.emissions.vehicle_category_metadata_file:
        raise ValueError(
            "Postprocess step 2 requires impacts.emissions.vehicle_category_metadata_file when annual sector targets are configured."
        )
    return _resolve_inventory_target_path(
        settings_path,
        settings.impacts.emissions.vehicle_category_metadata_file,
        output_root=output_root,
        input_roots=input_roots,
    )


def _run_postprocess_steps(
    settings_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    output_root: Path | None = None,
    allow_missing_source_inputs: bool = False,
    input_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> dict[str, str]:
    from .pipeline.postprocess.step1_compare_fleet import run as run_step1
    from .pipeline.postprocess.step2_compare_annual_targets import run as run_step2
    from .pipeline.postprocess.step3_compare_emissions_inventory import run as run_step3
    from .pipeline.postprocess.step4_plot_concentrations import run as run_step4
    from .pipeline.postprocess.step5_plot_exposure import run as run_step5
    from .pipeline.postprocess.step6_plot_delta_concentrations import run as run_step6
    from .pipeline.postprocess.step7_plot_delta_exposure import run as run_step7

    settings = load_settings_from_yaml(settings_path)
    if output_root is None:
        output_root = Path(resolve_path(settings.impacts.local_output_folder, settings_path)).resolve()
    else:
        output_root = Path(output_root).resolve()
    population_sample, transit_sample, freight_sample = _resolve_postprocess_sample_config(
        settings,
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    input_roots = _normalize_input_roots(input_roots)
    output_dir = output_root / "postprocess"
    modeled_emissions_path = _resolve_modeled_emissions_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    skims_emissions_path = _resolve_skims_emissions_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_root,
    )
    outputs: dict[str, str] = {}
    try:
        passenger_vehicle_types_path, freight_vehicle_types_path = _resolve_vehicle_types_paths(
            settings_path,
            run_manifest_path=run_manifest_path,
            output_root=output_root,
            input_roots=input_roots,
        )
        passenger_vehicles_path, freight_carriers_path = _resolve_optional_population_assignment_paths(
            settings_path,
            run_manifest_path=run_manifest_path,
            output_root=output_root,
            input_roots=input_roots,
        )
        passenger_activity_path = _resolve_inventory_emfacid_activity_path(
            settings_path,
            manifest_key="passenger_inventory_emfacid_file",
            run_manifest_path=run_manifest_path,
            output_root=output_root,
            input_roots=input_roots,
        )
        freight_activity_path = _resolve_inventory_emfacid_activity_path(
            settings_path,
            manifest_key="freight_inventory_emfacid_file",
            run_manifest_path=run_manifest_path,
            output_root=output_root,
            input_roots=input_roots,
        )
    except (FileNotFoundError, ValueError) as exc:
        if not allow_missing_source_inputs:
            raise
        logger.warning(
            "Skipping Steps 1-2: comparison source inputs are not available from this output directory: %s",
            exc,
        )
    else:
        fleet_outputs = run_step1(
            skims_emissions_path=str(skims_emissions_path),
            passenger_vehicle_types_path=str(passenger_vehicle_types_path),
            freight_vehicle_types_path=str(freight_vehicle_types_path),
            emfac_passenger_activity_path=str(passenger_activity_path),
            emfac_freight_activity_path=str(freight_activity_path),
            output_dir=output_dir / "fleet",
            passenger_vehicles_path=str(passenger_vehicles_path) if passenger_vehicles_path else None,
            freight_carriers_path=str(freight_carriers_path) if freight_carriers_path else None,
            population_sample=population_sample,
            transit_sample=transit_sample,
            freight_sample=freight_sample,
        )
        for key, value in fleet_outputs.items():
            outputs[f"fleet_{key}"] = value
        if settings.impacts.analysis.sector_targets:
            try:
                vehicle_category_metadata_path = _resolve_vehicle_category_metadata_path(
                    settings_path,
                    run_manifest_path=run_manifest_path,
                    output_root=output_root,
                    input_roots=input_roots,
                )
            except (FileNotFoundError, ValueError) as exc:
                if not allow_missing_source_inputs:
                    raise
                logger.warning(
                    "Skipping Step 2: annual-target source metadata is not available from this output directory: %s",
                    exc,
                )
            else:
                target_outputs = run_step2(
                    modeled_emissions_path=str(modeled_emissions_path),
                    passenger_vehicle_types_path=str(passenger_vehicle_types_path),
                    freight_vehicle_types_path=str(freight_vehicle_types_path),
                    vehicle_category_metadata_file=str(vehicle_category_metadata_path),
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
        else:
            logger.info("Skipping Step 2: no annual sector targets configured.")
    if settings.impacts.analysis.inventory_targets:
        county_boundaries_path = _resolve_county_boundaries_path(
            settings_path,
            run_manifest_path=run_manifest_path,
            output_root=output_root,
            input_roots=input_roots,
        )
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
        inventory_path = _resolve_inventory_target_path(
            settings_path,
            settings.impacts.analysis.inventory_file,
            output_root=output_root,
            input_roots=input_roots,
        )
        for target in settings.impacts.analysis.inventory_targets:
            target_outputs = run_step3(
                modeled_emissions_path=str(modeled_emissions_path),
                inventory_path=str(inventory_path),
                county_boundaries_path=str(county_boundaries_path),
                output_dir=output_dir / "emissions_inventory",
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
    else:
        logger.info("Skipping Step 3: no inventory targets configured.")

    conc_path = output_root / "exposure" / "beam_concentration_distribution.parquet"
    net_path = output_root / "preprocess" / "beam_osm_mapped.parquet"
    pop_path = output_root / "exposure" / "beam_population_counts.parquet"
    inmap_cells_path = output_root / "concentrations" / "beam_inmap_concentrations.parquet"
    if not inmap_cells_path.exists():
        inmap_cells_path = output_root / "preprocess" / "inmap_grid.parquet"
    delta_baseline_concentration_path = None
    delta_baseline_raw = getattr(settings.impacts.analysis, "delta_baseline_concentration_distribution_file", None)
    if delta_baseline_raw:
        delta_baseline_concentration_path = _resolve_delta_baseline_concentration_path(
            settings_path,
            delta_baseline_raw,
            output_root=output_root,
            input_roots=input_roots,
        )
    else:
        _remove_stale_postprocess_delta_outputs(output_dir)
    if conc_path.exists() and net_path.exists():
        map_outputs = run_step4(
            concentration_path=str(conc_path),
            network_path=str(net_path),
            output_dir=output_dir / "concentrations",
        )
        for key, value in map_outputs.items():
            outputs[f"concentration_{key}"] = value
    else:
        logger.info("Skipping Step 4: concentration or network output not found at %s", output_root)
    if pop_path.exists() and conc_path.exists() and net_path.exists():
        exp_outputs = run_step5(
            population_path=str(pop_path),
            concentration_path=str(conc_path),
            network_path=str(net_path),
            inmap_cells_path=str(inmap_cells_path) if inmap_cells_path.exists() else None,
            output_dir=output_dir / "exposure",
        )
        for key, value in exp_outputs.items():
            outputs[f"exposure_{key}"] = value
    else:
        logger.info("Skipping Step 5: exposure outputs not found at %s", output_root)
    delta_table_path = None
    if delta_baseline_concentration_path is None:
        logger.info("Skipping Steps 6-7: no delta baseline concentration distribution configured.")
    elif conc_path.exists() and net_path.exists():
        delta_outputs = run_step6(
            concentration_path=str(conc_path),
            delta_baseline_concentration_path=str(delta_baseline_concentration_path),
            network_path=str(net_path),
            output_dir=output_dir / "delta_concentrations",
        )
        delta_table_path = delta_outputs.get("delta_table")
        for key, value in delta_outputs.items():
            outputs[f"delta_concentration_{key}"] = value
    else:
        logger.info("Skipping Step 6: concentration or network output not found at %s", output_root)
    if delta_table_path and pop_path.exists() and net_path.exists() and inmap_cells_path.exists():
        delta_exposure_outputs = run_step7(
            population_path=str(pop_path),
            concentration_delta_path=str(delta_table_path),
            network_path=str(net_path),
            inmap_cells_path=str(inmap_cells_path),
            output_dir=output_dir / "delta_exposure",
        )
        for key, value in delta_exposure_outputs.items():
            outputs[f"delta_exposure_{key}"] = value
    elif delta_baseline_concentration_path is not None:
        logger.info("Skipping Step 7: delta table, population, network, or InMAP cell output not found at %s", output_root)

    logger.info("Postprocess steps complete: output_dir=%s outputs=%d", output_dir, len(outputs))
    return outputs


def postprocess_from_pipeline_manifest(
    run_manifest_path: str | Path,
    manifest_path: str | Path | None = None,
    output_root_override: str | Path | None = None,
    input_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> dict[str, Any]:
    output_root_arg = Path(output_root_override).resolve() if output_root_override is not None else None
    input_roots_arg = _normalize_input_roots(input_roots)
    pipeline_manifest = _localized_pipeline_manifest(run_manifest_path, output_root=output_root_arg)
    output_root = Path(str(pipeline_manifest.get("output_dir"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    log_step_banner("Postprocess", "Run Postprocess Steps", logger=logger)
    settings_path = _resolve_settings_path(run_manifest_path, output_root=output_root_arg)
    if output_root_arg is None:
        postprocess_outputs = _run_postprocess_steps(
            settings_path,
            run_manifest_path=run_manifest_path,
            input_roots=input_roots_arg,
        )
    else:
        postprocess_outputs = _run_postprocess_steps(
            settings_path,
            run_manifest_path=run_manifest_path,
            output_root=output_root,
            allow_missing_source_inputs=True,
            input_roots=input_roots_arg,
        )

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
    input_roots: tuple[str | Path, ...] | list[str | Path] | None = None,
) -> dict[str, Any]:
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
        input_roots=input_roots,
    )
