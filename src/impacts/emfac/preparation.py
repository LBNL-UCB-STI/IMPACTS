from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from ..common import log_step_banner
from ..common import log_substep_banner
from ..manifest.file_ops import write_structured_file
from ..manifest.schema import ActivitiesManifest

logger = logging.getLogger(__name__)

_ARCHIVE_NAME = "emissions_raw.tar.zst"



def _archive_path(beam_input_folder: Path, vehicle_folder: str) -> Path:
    return beam_input_folder / vehicle_folder / "emissions" / _ARCHIVE_NAME


def _outputs_exist(workflow: dict[str, Any]) -> bool:
    return Path(str(workflow["paths"]["final_activity_by_emfacid_output_passenger"])).exists()


def _activities_manifest_path(output_root: Path) -> Path:
    return output_root / "activities_manifest.yaml"


def _expected_output_path(cfg: dict[str, Any], start_year: int) -> Path:
    region_slug = str(cfg["region_label"]).lower()
    base_name = f"{region_slug}-emfac-{int(start_year)}-inventory-final"
    return Path(str(cfg["output_root"])) / "activities" / str(cfg["scenario"]) / "inventory" / f"{base_name}-passenger-activity-by-emfacid.parquet"


def _build_activities_manifest_payload(
    *,
    workflow: dict[str, Any],
    settings_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    paths = workflow["paths"]
    run = workflow["run"]
    return {
        "contract_version": "1",
        "model": "impacts",
        "settings_source": str(settings_path.resolve()),
        "output_dir": str(Path(str(paths["outputs_root"])).resolve()),
        "region_label": str(run["region_label"]),
        "calendar_year": int(run["calendar_year"]),
        "scenario": str(run["scenario"]),
        "vehicle_category_metadata_file": str(workflow["inputs"]["vehicle_category_metadata_file"]),
        "outputs": {
            "outputs_root": str(Path(str(paths["outputs_root"])).resolve()),
            "activities_output_root": str(Path(str(paths["activities_output_root"])).resolve()),
            "tmp_root": str(Path(str(paths["tmp_root"])).resolve()),
            "emissions_store_root": str(Path(str(paths["emissions_store_root"])).resolve()),
            "passenger_rates_file": str(Path(str(paths["final_output_passenger"])).resolve()),
            "passenger_activity_file": str(Path(str(paths["matching_activity_output_passenger"])).resolve()),
            "passenger_fleet_file": str(Path(str(paths["final_fleet_output_passenger"])).resolve()),
            "freight_rates_file": str(Path(str(paths["final_output_freight"])).resolve()),
            "freight_activity_file": str(Path(str(paths["matching_activity_output_freight"])).resolve()),
            "freight_fleet_file": str(Path(str(paths["final_fleet_output_freight"])).resolve()),
            "final_activity_by_emfacid_output_passenger": str(
                Path(str(paths["final_activity_by_emfacid_output_passenger"])).resolve()
            ),
            "final_activity_by_emfacid_output_freight": str(
                Path(str(paths["final_activity_by_emfacid_output_freight"])).resolve()
            ),
        },
        "notes": [
            "Activities manifest is the contract boundary between EMFAC activities and fleet.",
            "Fleet consumes this manifest instead of rediscovering activities outputs from settings.",
        ],
        "activities_manifest_path": str(manifest_path.resolve()),
    }


def _write_activities_manifest_from_cfg(
    *,
    cfg: dict[str, Any],
    settings_path: Path,
    calendar_year: int,
    manifest_path: Path,
) -> dict[str, Any]:
    region_slug = str(cfg["region_label"]).lower()
    outputs_root = Path(str(cfg["output_root"])).resolve()
    activities_output_root = outputs_root / "activities" / str(cfg["scenario"]) / "inventory"
    tmp_root = outputs_root / "activities" / "_tmp"
    project_analysis_name = f"{region_slug}-emfac-{int(calendar_year)}-project-analysis-final"
    inventory_final_name = f"{region_slug}-emfac-{int(calendar_year)}-inventory-final"
    inventory_matching_name = f"{region_slug}-emfac-{int(calendar_year)}-inventory-matching"
    typed_manifest = ActivitiesManifest.from_dict(
        {
            "contract_version": "1",
            "model": "impacts",
            "settings_source": str(settings_path.resolve()),
            "output_dir": str(outputs_root),
            "region_label": str(cfg["region_label"]),
            "calendar_year": int(calendar_year),
            "scenario": str(cfg["scenario"]),
            "vehicle_category_metadata_file": str(Path(str(cfg["vehicle_category_metadata_file"])).resolve()),
            "outputs": {
                "outputs_root": str(outputs_root),
                "activities_output_root": str(activities_output_root.resolve()),
                "tmp_root": str(tmp_root.resolve()),
                "emissions_store_root": str((outputs_root / "activities" / str(cfg["scenario"]) / "rates").resolve()),
                "passenger_rates_file": str((activities_output_root / f"{project_analysis_name}-passenger-rates.parquet").resolve()),
                "passenger_activity_file": str((tmp_root / f"{inventory_matching_name}-passenger-activity.parquet").resolve()),
                "passenger_fleet_file": str((activities_output_root / f"{inventory_final_name}-passenger-fleet.parquet").resolve()),
                "freight_rates_file": str((activities_output_root / f"{project_analysis_name}-freight-rates.parquet").resolve()),
                "freight_activity_file": str((tmp_root / f"{inventory_matching_name}-freight-activity.parquet").resolve()),
                "freight_fleet_file": str((activities_output_root / f"{inventory_final_name}-freight-fleet.parquet").resolve()),
                "final_activity_by_emfacid_output_passenger": str(
                    (activities_output_root / f"{inventory_final_name}-passenger-activity-by-emfacid.parquet").resolve()
                ),
                "final_activity_by_emfacid_output_freight": str(
                    (activities_output_root / f"{inventory_final_name}-freight-activity-by-emfacid.parquet").resolve()
                ),
            },
            "notes": [
                "Activities manifest is the contract boundary between EMFAC activities and fleet.",
                "Fleet consumes this manifest instead of rediscovering activities outputs from settings.",
            ],
            "activities_manifest_path": str(manifest_path.resolve()),
        }
    )
    write_structured_file(manifest_path, typed_manifest.to_dict())
    logger.info("Activities manifest written: %s", manifest_path)
    return typed_manifest.to_dict()


def _write_activities_manifest(
    *,
    workflow: dict[str, Any],
    settings_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    typed_manifest = ActivitiesManifest.from_dict(
        _build_activities_manifest_payload(
            workflow=workflow,
            settings_path=settings_path,
            manifest_path=manifest_path,
        )
    )
    write_structured_file(manifest_path, typed_manifest.to_dict())
    logger.info("Activities manifest written: %s", manifest_path)
    return typed_manifest.to_dict()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _entry_folder(value: Any) -> str | None:
    mapping = _mapping(value)
    if "folder" in mapping and mapping["folder"] not in (None, ""):
        return str(mapping["folder"])
    return None


def _entry_value(value: Any, key: str) -> Any:
    mapping = _mapping(value)
    result = mapping.get(key)
    return None if result in (None, "") else result


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s → %s", archive, destination)
    try:
        zstd = subprocess.run(
            ["zstd", "-d", str(archive), "--stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(["tar", "--warning=no-unknown-keyword", "-xf", "-", "-C", str(destination)], input=zstd.stdout, check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        subprocess.run(["tar", "--warning=no-unknown-keyword", "--zstd", "-xf", str(archive), "-C", str(destination)], check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        import tarfile
        with tarfile.open(str(archive), "r:zst") as tf:
            tf.extractall(str(destination))
        return
    except Exception:
        pass
    raise RuntimeError(
        f"Could not extract {archive}. Install zstd (module load zstd) or use Python 3.12+."
    )


def _resolve_activities_config(settings, config_path: Path) -> dict[str, Any]:
    from ..manifest.file_ops import resolve_path, resolve_required_path
    from ..config.path_registry import build_registry

    local_input_folder = Path(resolve_required_path(settings.impacts.local_input_folder, config_path, "impacts.local_input_folder")).resolve()
    local_output_folder = Path(resolve_path(settings.impacts.local_output_folder, config_path)).resolve()
    beam_input_folder = Path(resolve_required_path(settings.beam.local_input_folder, config_path, "beam.local_input_folder")).resolve()
    registry = build_registry(settings, config_path)

    cfg_activities = settings.impacts.activities if isinstance(settings.impacts.activities, dict) else {}
    project_analysis = _mapping(cfg_activities.get("project_analysis"))
    emissions_inventory = _mapping(cfg_activities.get("emissions_inventory"))
    region_name = settings.run.region
    emfac_root = local_input_folder

    configured_project_main = _mapping(project_analysis.get("main"))
    configured_black_carbon = _mapping(project_analysis.get("black_carbon"))
    configured_road_dust = _mapping(project_analysis.get("paved_road_dust"))

    def _locate_folder(name: str | None, default_name: str) -> Path:
        target = name or default_name
        found = registry.locate(target)
        if found:
            return found
        raw = Path(target).expanduser()
        return raw.resolve() if raw.is_absolute() else (emfac_root / raw).resolve()

    project_analysis_folder = _locate_folder(
        _entry_folder(project_analysis.get("main")),
        f"{region_name}-emfac-project-analysis",
    )
    black_carbon_folder = _locate_folder(
        _entry_folder(project_analysis.get("black_carbon")),
        f"{region_name}-emfac-moves-bc",
    )
    road_dust_folder = _locate_folder(
        _entry_folder(project_analysis.get("paved_road_dust")),
        "statewide-carb-road-dust",
    )
    inventory_folder = _locate_folder(
        str(emissions_inventory.get("inventory_folder") or "") or None,
        f"{region_name}-emfac-emissions-inventory",
    )
    fallback_folder = _locate_folder(
        str(emissions_inventory.get("fallback_folder") or "") or None,
        "statewide-emfac-emissions-inventory",
    )

    metadata_path = settings.impacts.emissions.vehicle_category_metadata_file
    if not metadata_path:
        raise ValueError("Missing required value: impacts.emissions.vehicle_category_metadata_file")
    vehicle_category_metadata_file = (
        registry.locate(str(metadata_path))
        or registry.locate(Path(str(metadata_path)).name)
    )
    if not vehicle_category_metadata_file:
        raise FileNotFoundError(
            f"Could not find vehicle_category_metadata_file '{metadata_path}'. "
            f"Searched beam data roots: {registry.roots}"
        )

    vehicle_folder = settings.impacts.population.vehicle_folder or "vehicle-tech"
    archive = _archive_path(beam_input_folder / region_name, vehicle_folder).resolve()

    return {
        "region_name": region_name,
        "region_label": cfg_activities.get("region_label", region_name.upper()),
        "scenario": settings.impacts.scenario,
        "seed": settings.impacts.seed,
        "output_root": local_output_folder,
        "archive": archive,
        "extract_root": inventory_folder.parent,
        "project_analysis_folder": project_analysis_folder,
        "black_carbon_folder": black_carbon_folder,
        "road_dust_folder": road_dust_folder,
        "inventory_folder": inventory_folder,
        "fallback_folder": fallback_folder,
        "vehicle_category_metadata_file": vehicle_category_metadata_file,
        "pto_as_process": configured_project_main.get("pto_as_process"),
        "black_carbon_pollutant": _entry_value(project_analysis.get("black_carbon"), "pollutant") or "BCh",
        "road_category_map": _entry_value(project_analysis.get("paved_road_dust"), "road_category_map") or {},
        "fuel_map": emissions_inventory.get("fuel_map") or {},
        "model_year_groups": cfg_activities.get("model_year_groups"),
    }


def _find_matching_file(path: Path, patterns: tuple[str, ...]) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    lowered = tuple(pattern.lower() for pattern in patterns)
    return any(
        child.is_file() and not child.name.startswith(".") and any(pattern in child.name.lower() for pattern in lowered)
        for child in path.iterdir()
    )


def _resolve_extracted_raw_folder(path: Path, *, extract_root: Path) -> Path:
    if path.exists():
        return path
    raw_candidate = extract_root / "emissions_raw" / path.name
    if raw_candidate.exists():
        return raw_candidate
    return path


def _normalize_raw_input_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(cfg)
    extract_root = Path(str(cfg["extract_root"])).resolve()
    for key in (
        "project_analysis_folder",
        "black_carbon_folder",
        "road_dust_folder",
        "inventory_folder",
        "fallback_folder",
    ):
        normalized[key] = str(
            _resolve_extracted_raw_folder(Path(str(cfg[key])).resolve(), extract_root=extract_root)
        )
    return normalized


def _missing_raw_inputs(cfg: dict[str, Any]) -> list[str]:
    cfg = _normalize_raw_input_paths(cfg)
    missing: list[str] = []
    if not Path(cfg["project_analysis_folder"]).exists():
        missing.append(str(cfg["project_analysis_folder"]))
    if not _find_matching_file(Path(cfg["black_carbon_folder"]), ("bc",)):
        missing.append(f"{cfg['black_carbon_folder']} (bc)")
    if not _find_matching_file(Path(cfg["road_dust_folder"]), ("rainy_days",)):
        missing.append(f"{cfg['road_dust_folder']} (rainy_days)")
    if not _find_matching_file(Path(cfg["road_dust_folder"]), ("silt_loading",)):
        missing.append(f"{cfg['road_dust_folder']} (silt_loading)")
    if not _find_matching_file(Path(cfg["fallback_folder"]), ("statewide",)):
        missing.append(f"{cfg['fallback_folder']} (statewide)")
    for label in ("population", "trips", "vmt", "emission"):
        if not _find_matching_file(Path(cfg["inventory_folder"]), (label,)):
            missing.append(f"{cfg['inventory_folder']} ({label})")
    return missing


def _ensure_raw_data(cfg: dict[str, Any]) -> None:
    cfg = _normalize_raw_input_paths(cfg)
    missing_before = _missing_raw_inputs(cfg)
    if not missing_before:
        logger.info("Raw activities inputs already present — skipping extraction.")
        return
    archive = Path(cfg["archive"])
    if not archive.exists():
        raise FileNotFoundError(
            f"EMFAC activities raw inputs are missing ({', '.join(missing_before)}) and the archive "
            f"was not found at {archive}. Place {_ARCHIVE_NAME} there or pre-extract the raw inputs."
        )
    _extract_archive(archive, Path(cfg["extract_root"]))
    cfg = _normalize_raw_input_paths(cfg)
    missing_after = _missing_raw_inputs(cfg)
    if missing_after:
        raise RuntimeError(
            f"Extraction of {archive} completed, but required EMFAC raw inputs are still missing: "
            f"{', '.join(missing_after)}. Check the archive structure and configured folder paths."
        )


def _build_workflow(settings, config_path: Path) -> dict[str, Any]:
    from ..config.settings import _build_activities_config_from_root
    from ..config.settings import _build_activities_workflow

    cfg = _normalize_raw_input_paths(_resolve_activities_config(settings, config_path))
    from ..config.path_registry import build_registry
    registry = build_registry(settings, config_path)
    fleet = dict(settings.impacts.fleet if isinstance(settings.impacts.fleet, dict) else {})
    assignment_model = fleet.get("assignment_model")
    if assignment_model:
        fleet["assignment_model"] = str(registry.locate_required(str(assignment_model), label="fleet.assignment_model"))
    activities_override: dict[str, Any] = {
        "project_analysis": {
            "main": {
                "folder": str(cfg["project_analysis_folder"]),
                **({"pto_as_process": cfg["pto_as_process"]} if cfg["pto_as_process"] else {}),
            },
            "black_carbon": {
                "folder": str(cfg["black_carbon_folder"]),
                "pollutant": str(cfg["black_carbon_pollutant"]),
            },
            "paved_road_dust": {
                "folder": str(cfg["road_dust_folder"]),
                **({"road_category_map": cfg["road_category_map"]} if cfg["road_category_map"] else {}),
            },
        },
        "emissions_inventory": {
            "inventory_folder": {"folder": str(cfg["inventory_folder"])},
            "fallback_folder": {"folder": str(cfg["fallback_folder"])},
            "vehicle_category_metadata_file": str(cfg["vehicle_category_metadata_file"]),
            **({"fuel_map": cfg["fuel_map"]} if cfg["fuel_map"] else {}),
        },
    }
    if cfg["model_year_groups"]:
        activities_override["model_year_groups"] = cfg["model_year_groups"]

    merged_emfac = {
        "region": {"name": cfg["region_name"], "label": cfg["region_label"]},
        "scenario": {"year": settings.run.start_year, "name": cfg["scenario"]},
        "seed": cfg["seed"],
        "output": str(cfg["output_root"]),
        "fleet": fleet,
        "activities": activities_override,
    }
    raw = _build_activities_config_from_root(merged_emfac)
    return _build_activities_workflow(raw, config_path)


def ensure_emfac_activities_outputs(settings, config_path: Path) -> dict[str, Any]:
    from .activities.main import run_workflow as run_activities_workflow

    cfg = _resolve_activities_config(settings, config_path)
    manifest_path = _activities_manifest_path(Path(str(cfg["output_root"])).resolve())

    log_step_banner("EMFAC Activities", "Ensure inventory outputs", logger=logger)

    log_substep_banner("1", "check existing outputs", logger=logger)
    expected_output = _expected_output_path(cfg, settings.run.start_year)
    if expected_output.exists():
        logger.info("Outputs already present at %s — skipping activities run.", expected_output)
        return _write_activities_manifest_from_cfg(
            cfg=cfg,
            settings_path=config_path,
            calendar_year=settings.run.start_year,
            manifest_path=manifest_path,
        )

    log_substep_banner("2", "ensure raw activity inputs", logger=logger)
    _ensure_raw_data(cfg)

    log_substep_banner("3", "build workflow config", logger=logger)
    workflow = _build_workflow(settings, config_path)

    log_substep_banner("4", "run activities workflow", logger=logger)
    run_activities_workflow(workflow)

    log_substep_banner("5", "validate outputs", logger=logger)
    if not _outputs_exist(workflow):
        output_dir = Path(str(workflow["paths"]["final_activity_by_emfacid_output_passenger"])).parent
        raise RuntimeError(
            f"EMFAC activities workflow completed but expected outputs not found in {output_dir}."
        )
    logger.info("EMFAC activities outputs validated.")
    return _write_activities_manifest(
        workflow=workflow,
        settings_path=config_path,
        manifest_path=manifest_path,
    )
