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


def _is_extracted(emfac_root: Path, region_name: str) -> bool:
    return (emfac_root / f"{region_name}-emfac-project-analysis").exists()


def _outputs_exist(workflow: dict[str, Any]) -> bool:
    return Path(str(workflow["paths"]["final_activity_emfacid_output_passenger"])).exists()


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


def _ensure_raw_data(emfac_root: Path, beam_input_folder: Path, region_name: str) -> None:
    if _is_extracted(emfac_root, region_name):
        logger.info("Raw data already present at %s — skipping extraction.", emfac_root)
        return
    archive = _archive_path(beam_input_folder, region_name)
    if not archive.exists():
        raise FileNotFoundError(
            f"EMFAC activities outputs are missing and the raw data archive was not found at {archive}. "
            f"Place {_ARCHIVE_NAME} there or pre-extract the data to {emfac_root}."
        )
    _extract_archive(archive, emfac_root)
    if not _is_extracted(emfac_root, region_name):
        raise RuntimeError(
            f"Extraction of {archive} completed but expected folder "
            f"{emfac_root / f'{region_name}-emfac-project-analysis'} was not found. "
            f"Check the archive structure."
        )


def _build_workflow(settings, config_path: Path) -> dict[str, Any]:
    from .config import _build_activities_config_from_root
    from .config import _build_activities_workflow
    from .config import _load_yaml_path
    from ..manifest.file_ops import resolve_path

    local_input_folder = Path(resolve_path(settings.impacts.local_input_folder, config_path)).resolve()
    beam_input_folder = Path(resolve_path(settings.beam.local_input_folder, config_path)).resolve()
    region_name = settings.run.region
    emfac_root = _emfac_raw_root(local_input_folder)
    emfac_root_config = _load_yaml_path(config_path, "emfac")

    activities_override: dict[str, Any] = {
        "project_analysis": [
            {
                "main": [
                    {"folder": str(emfac_root / f"{region_name}-emfac-project-analysis")},
                    *([{"pto_as_process": emfac_root_config.get("activities", {}).get("pto_as_process", {})}]
                      if emfac_root_config.get("activities", {}).get("pto_as_process") else []),
                ],
            },
            {
                "black_carbon": [
                    {"folder": str(emfac_root / f"{region_name}-emfac-moves-bc")},
                    {"pollutant": "BCh"},
                ],
            },
            {
                "paved_road_dust": [
                    {"folder": str(emfac_root / "statewide-carb-road-dust")},
                    *([{"road_category_map": emfac_root_config.get("activities", {}).get("road_category_map", {})}]
                      if emfac_root_config.get("activities", {}).get("road_category_map") else []),
                ],
            },
        ],
        "emissions_inventory": [
            {"inventory_folder": str(emfac_root / f"{region_name}-emfac-emissions-inventory")},
            {"fallback_folder": str(emfac_root / "statewide-emfac-emissions-inventory")},
            {"vehicle_category_attributes_file": str(
                beam_input_folder / region_name / "vehicle-tech" / "emissions" / "emfac_vehicle_category_attributes.csv"
            )},
            *([{"fuel_map": emfac_root_config.get("activities", {}).get("fuel_map", {})}]
              if emfac_root_config.get("activities", {}).get("fuel_map") else []),
        ],
    }
    if emfac_root_config.get("activities", {}).get("model_year_groups"):
        activities_override["model_year_groups"] = emfac_root_config["activities"]["model_year_groups"]

    output_root = local_input_folder / "emfac"
    merged_emfac = {**emfac_root_config, "output": str(output_root)}
    merged_emfac["activities"] = {**merged_emfac.get("activities", {}), **activities_override}

    raw = _build_activities_config_from_root(merged_emfac)
    return _build_activities_workflow(raw, config_path)


def ensure_emfac_activities_outputs(settings, config_path: Path) -> dict[str, Any]:
    from .activities.main import run_workflow as run_activities_workflow
    from ..manifest.file_ops import resolve_path

    local_input_folder = Path(resolve_path(settings.impacts.local_input_folder, config_path)).resolve()
    beam_input_folder = Path(resolve_path(settings.beam.local_input_folder, config_path)).resolve()
    region_name = settings.run.region
    emfac_root = _emfac_raw_root(local_input_folder)

    log_step_banner("EMFAC Activities", "Ensure inventory outputs", logger=logger)

    log_substep_banner("1", "build workflow config", logger=logger)
    workflow = _build_workflow(settings, config_path)

    log_substep_banner("2", "check existing outputs", logger=logger)
    if _outputs_exist(workflow):
        logger.info("Outputs already present — skipping activities run.")
        return workflow

    log_substep_banner("3", "extract raw data from beam archive", logger=logger)
    _ensure_raw_data(emfac_root, beam_input_folder, region_name)

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
