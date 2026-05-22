from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from ..common import log_step_banner
from ..common import log_substep_banner

logger = logging.getLogger(__name__)

_ARCHIVE_NAME = "emfac.tar.zst"


def _emfac_raw_root(local_input_folder: Path) -> Path:
    return local_input_folder / "emfac"


def _archive_path(beam_input_folder: Path, region: str) -> Path:
    return beam_input_folder / region / "vehicle-tech" / "emissions" / _ARCHIVE_NAME


def _outputs_exist(workflow: dict[str, Any]) -> bool:
    return Path(str(workflow["paths"]["final_activity_emfacid_output_passenger"])).exists()


def _expected_output_path(cfg: dict[str, Any], start_year: int) -> Path:
    region_slug = str(cfg["region_label"]).lower()
    base_name = f"{region_slug}-emfac-{int(start_year)}-inventory-final"
    return Path(str(cfg["output_root"])) / "activities" / f"{base_name}-passenger-activity-by-emfacid.parquet"


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
        subprocess.run(["tar", "-xf", "-", "-C", str(destination)], input=zstd.stdout, check=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    try:
        subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(destination)], check=True)
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
    from ..manifest.file_ops import resolve_path

    local_input_folder = Path(resolve_path(settings.impacts.local_input_folder, config_path)).resolve()
    beam_input_folder = Path(resolve_path(settings.beam.local_input_folder, config_path)).resolve()
    cfg_activities = settings.impacts.activities if isinstance(settings.impacts.activities, dict) else {}
    project_analysis = _mapping(cfg_activities.get("project_analysis"))
    emissions_inventory = _mapping(cfg_activities.get("emissions_inventory"))
    region_name = settings.run.region
    emfac_root = _emfac_raw_root(local_input_folder)

    configured_project_main = _mapping(project_analysis.get("main"))
    configured_black_carbon = _mapping(project_analysis.get("black_carbon"))
    configured_road_dust = _mapping(project_analysis.get("paved_road_dust"))

    project_analysis_folder = Path(
        resolve_path(_entry_folder(project_analysis.get("main")), config_path)
        if _entry_folder(project_analysis.get("main"))
        else str(emfac_root / f"{region_name}-emfac-project-analysis")
    ).resolve()
    black_carbon_folder = Path(
        resolve_path(_entry_folder(project_analysis.get("black_carbon")), config_path)
        if _entry_folder(project_analysis.get("black_carbon"))
        else str(emfac_root / f"{region_name}-emfac-moves-bc")
    ).resolve()
    road_dust_folder = Path(
        resolve_path(_entry_folder(project_analysis.get("paved_road_dust")), config_path)
        if _entry_folder(project_analysis.get("paved_road_dust"))
        else str(emfac_root / "statewide-carb-road-dust")
    ).resolve()
    inventory_folder = Path(
        resolve_path(str(emissions_inventory.get("inventory_folder")), config_path)
        if emissions_inventory.get("inventory_folder")
        else str(emfac_root / f"{region_name}-emfac-emissions-inventory")
    ).resolve()
    fallback_folder = Path(
        resolve_path(str(emissions_inventory.get("fallback_folder")), config_path)
        if emissions_inventory.get("fallback_folder")
        else str(emfac_root / "statewide-emfac-emissions-inventory")
    ).resolve()
    metadata_path = settings.impacts.emissions.vehicle_category_metadata_file
    if not metadata_path:
        raise ValueError("Missing required value: impacts.emissions.vehicle_category_metadata_file")
    vehicle_category_metadata_file = Path(resolve_path(str(metadata_path), config_path)).resolve()
    configured_archive = cfg_activities.get("raw_inputs_archive")
    archive = Path(
        resolve_path(str(configured_archive), config_path)
        if configured_archive
        else _archive_path(beam_input_folder, region_name)
    ).resolve()

    return {
        "region_name": region_name,
        "region_label": cfg_activities.get("region_label", region_name.upper()),
        "scenario": settings.impacts.scenario,
        "seed": settings.impacts.seed,
        "output_root": emfac_root,
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
        child.is_file() and any(pattern in child.name.lower() for pattern in lowered)
        for child in path.iterdir()
    )


def _missing_raw_inputs(cfg: dict[str, Any]) -> list[str]:
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
    missing_after = _missing_raw_inputs(cfg)
    if missing_after:
        raise RuntimeError(
            f"Extraction of {archive} completed, but required EMFAC raw inputs are still missing: "
            f"{', '.join(missing_after)}. Check the archive structure and configured folder paths."
        )


def _build_workflow(settings, config_path: Path) -> dict[str, Any]:
    from ..config.settings import _build_activities_config_from_root
    from ..config.settings import _build_activities_workflow

    cfg = _resolve_activities_config(settings, config_path)
    activities_override: dict[str, Any] = {
        "project_analysis": [
            {
                "main": [
                    {"folder": str(cfg["project_analysis_folder"])},
                    *([{"pto_as_process": cfg["pto_as_process"]}] if cfg["pto_as_process"] else []),
                ],
            },
            {
                "black_carbon": [
                    {"folder": str(cfg["black_carbon_folder"])},
                    {"pollutant": str(cfg["black_carbon_pollutant"])},
                ],
            },
            {
                "paved_road_dust": [
                    {"folder": str(cfg["road_dust_folder"])},
                    *([{"road_category_map": cfg["road_category_map"]}] if cfg["road_category_map"] else []),
                ],
            },
        ],
        "emissions_inventory": [
            {"inventory_folder": str(cfg["inventory_folder"])},
            {"fallback_folder": str(cfg["fallback_folder"])},
            {"vehicle_category_metadata_file": str(cfg["vehicle_category_metadata_file"])},
            *([{"fuel_map": cfg["fuel_map"]}] if cfg["fuel_map"] else []),
        ],
    }
    if cfg["model_year_groups"]:
        activities_override["model_year_groups"] = cfg["model_year_groups"]

    merged_emfac = {
        "region": {"name": cfg["region_name"], "label": cfg["region_label"]},
        "scenario": {"year": settings.run.start_year, "name": cfg["scenario"]},
        "seed": cfg["seed"],
        "output": str(cfg["output_root"]),
        "fleet": settings.impacts.fleet if isinstance(settings.impacts.fleet, dict) else {},
        "activities": activities_override,
    }
    raw = _build_activities_config_from_root(merged_emfac)
    return _build_activities_workflow(raw, config_path)


def ensure_emfac_activities_outputs(settings, config_path: Path) -> dict[str, Any]:
    from .activities.main import run_workflow as run_activities_workflow

    cfg = _resolve_activities_config(settings, config_path)

    log_step_banner("EMFAC Activities", "Ensure inventory outputs", logger=logger)

    log_substep_banner("1", "check existing outputs", logger=logger)
    expected_output = _expected_output_path(cfg, settings.run.start_year)
    if expected_output.exists():
        logger.info("Outputs already present at %s — skipping activities run.", expected_output)
        return {"paths": {"final_activity_emfacid_output_passenger": str(expected_output)}}

    log_substep_banner("2", "ensure raw activity inputs", logger=logger)
    _ensure_raw_data(cfg)

    log_substep_banner("3", "build workflow config", logger=logger)
    workflow = _build_workflow(settings, config_path)

    log_substep_banner("4", "run activities workflow", logger=logger)
    run_activities_workflow(workflow)

    log_substep_banner("5", "validate outputs", logger=logger)
    if not _outputs_exist(workflow):
        output_dir = Path(str(workflow["paths"]["final_activity_emfacid_output_passenger"])).parent
        raise RuntimeError(
            f"EMFAC activities workflow completed but expected outputs not found in {output_dir}."
        )
    logger.info("EMFAC activities outputs validated.")

    return workflow
