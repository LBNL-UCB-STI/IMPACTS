from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from typing import Dict

from ..common import log_step_banner
from ..common import log_substep_banner

logger = logging.getLogger(__name__)

_ARCHIVE_NAME = "emfac.tar.zst"


def _emfac_raw_root(local_input_folder: Path) -> Path:
    return local_input_folder / "emfac"


def _archive_path(beam_input_folder: Path, region: str) -> Path:
    return beam_input_folder / region / "vehicle-tech" / "emissions" / _ARCHIVE_NAME


def _is_extracted(emfac_root: Path, region_name: str) -> bool:
    sentinel = emfac_root / f"{region_name}-emfac-project-analysis"
    return sentinel.exists()


def _extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s → %s", archive, destination)
    # Try zstd pipeline (most compatible on HPC)
    try:
        zstd = subprocess.run(
            ["zstd", "-d", str(archive), "--stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        subprocess.run(
            ["tar", "-xf", "-", "-C", str(destination)],
            input=zstd.stdout,
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Fall back to tar --zstd (GNU tar 1.31+)
    try:
        subprocess.run(
            ["tar", "--zstd", "-xf", str(archive), "-C", str(destination)],
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # Fall back to Python tarfile with zstd filter (Python 3.12+)
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


def ensure_emfac_raw_data(
    local_input_folder: Path,
    beam_input_folder: Path,
    region_name: str,
) -> Path:
    emfac_root = _emfac_raw_root(local_input_folder)
    if _is_extracted(emfac_root, region_name):
        logger.info("EMFAC raw data already present at %s", emfac_root)
        return emfac_root
    archive = _archive_path(beam_input_folder, region_name)
    if not archive.exists():
        raise FileNotFoundError(
            f"EMFAC raw data not found at {emfac_root} and archive not found at {archive}. "
            f"Place {_ARCHIVE_NAME} at {archive} or pre-extract data to {emfac_root}."
        )
    log_substep_banner("prepare", f"extract EMFAC raw data from {archive.name}", logger=logger)
    _extract_archive(archive, emfac_root)
    if not _is_extracted(emfac_root, region_name):
        raise RuntimeError(
            f"Extraction of {archive} completed but expected folder "
            f"{emfac_root / f'{region_name}-emfac-project-analysis'} was not found. "
            f"Check the archive structure."
        )
    logger.info("EMFAC raw data extracted to %s", emfac_root)
    return emfac_root


def build_activities_workflow_from_settings(
    settings,
    config_path: Path,
) -> Dict[str, Any]:
    from .config import _build_activities_config_from_root
    from .config import _build_activities_workflow
    from ..manifest.file_ops import resolve_path

    local_input_folder = Path(
        resolve_path(settings.impacts.local_input_folder, config_path)
    ).resolve()
    beam_input_folder = Path(
        resolve_path(settings.beam.local_input_folder, config_path)
    ).resolve()
    region_name = settings.run.region

    log_step_banner("EMFAC Activities", "Prepare inputs", logger=logger)
    emfac_root = ensure_emfac_raw_data(local_input_folder, beam_input_folder, region_name)

    # Load the emfac: section from the settings YAML for region/scenario/model_year_groups etc.
    from .config import _load_yaml_path
    emfac_root_config = _load_yaml_path(config_path, "emfac")

    # Derive all input paths from emfac_root — known archive structure
    activities_override: Dict[str, Any] = {
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

    # Output under beam data vehicle-tech so activities/ lands alongside other beam artifacts
    output_root = beam_input_folder / region_name / "vehicle-tech"
    merged_emfac = {**emfac_root_config, "output": str(output_root)}
    merged_emfac.setdefault("activities", {})
    merged_emfac["activities"] = {**merged_emfac.get("activities", {}), **activities_override}

    raw = _build_activities_config_from_root(merged_emfac)
    return _build_activities_workflow(raw, config_path)
