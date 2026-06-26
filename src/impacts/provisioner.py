from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from .common import log_step_banner
from .common import log_substep_banner
from .config.settings import presim_activities_inventory_root
from .config.settings import presim_activities_manifest_path
from .config.settings import presim_activities_rates_root
from .config.settings import presim_activities_tmp_root
from .manifest.file_ops import write_structured_file
from .manifest.schema import ActivitiesManifest

logger = logging.getLogger(__name__)


def _outputs_exist(workflow: dict[str, Any]) -> bool:
    return Path(str(workflow["paths"]["final_activity_by_emfacid_output_passenger"])).exists()


def _presim_identity_kwargs(cfg: dict[str, Any]) -> dict[str, str | None]:
    return {
        "region": str(cfg["run_region"]),
        "output_run_name": cfg.get("output_run_name"),
        "run_scenario": str(cfg["run_scenario"]),
    }


def _activities_manifest_path(cfg: dict[str, Any]) -> Path:
    return presim_activities_manifest_path(
        Path(str(cfg["output_root"])).resolve(),
        **_presim_identity_kwargs(cfg),
    )


def _expected_output_path(cfg: dict[str, Any], start_year: int) -> Path:
    region_slug = str(cfg["region_label"]).lower()
    base_name = f"{region_slug}-emfac-{int(start_year)}-inventory-final"
    return (
        presim_activities_inventory_root(
            cfg["output_root"],
            str(cfg["scenario"]),
            **_presim_identity_kwargs(cfg),
        )
        / f"{base_name}-passenger-activity-by-emfacid.parquet"
    )


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
            "emissions_inventory": str(Path(str(paths["emissions_inventory"])).resolve()),
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
    presim_identity = _presim_identity_kwargs(cfg)
    activities_output_root = presim_activities_inventory_root(
        outputs_root,
        str(cfg["scenario"]),
        **presim_identity,
    )
    tmp_root = presim_activities_tmp_root(outputs_root, **presim_identity)
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
                "emissions_store_root": str(
                    presim_activities_rates_root(
                        outputs_root,
                        str(cfg["scenario"]),
                        **presim_identity,
                    ).resolve()
                ),
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
                "emissions_inventory": str(
                    (tmp_root / f"{region_slug}-emfac-{int(calendar_year)}-inventory-intermediate-with-activity.parquet").resolve()
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
    logger.info("EMFAC activities stage complete: activities_manifest=%s", manifest_path)
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
    logger.info("EMFAC activities stage complete: activities_manifest=%s", manifest_path)
    return typed_manifest.to_dict()


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _entry_folder(value: Any) -> str | None:
    mapping = _mapping(value)
    if "folder_in_archive" in mapping and mapping["folder_in_archive"] not in (None, ""):
        return str(mapping["folder_in_archive"])
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
    from .config.path_registry import build_registry, SettingsPathResolver

    _sr = SettingsPathResolver.from_settings(settings, config_path)
    local_input_folder = _sr.impacts_input
    local_input_folder.mkdir(parents=True, exist_ok=True)
    local_output_folder = _sr.impacts_output
    registry = build_registry(settings, config_path)

    cfg_activities = settings.impacts.activities if isinstance(settings.impacts.activities, dict) else {}
    project_analysis = _mapping(cfg_activities.get("project_analysis"))
    emissions_inventory = _mapping(cfg_activities.get("emissions_inventory"))
    region_name = settings.run.region

    configured_project_main = _mapping(project_analysis.get("main"))

    def _locate_folder(name: str | None, default_name: str) -> Path:
        target = name or default_name
        found = registry.locate(target)
        if found:
            return found
        raw = Path(target).expanduser()
        return raw.resolve() if raw.is_absolute() else (local_input_folder / raw).resolve()

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
        emissions_inventory.get("folder_in_archive") or None,
        f"{region_name}-emfac-emissions-inventory",
    )
    fallback_folder = _locate_folder(
        emissions_inventory.get("fallback_folder_in_archive") or None,
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

    archive_setting = str(cfg_activities.get("emissions_raw_archive") or "").strip()
    if not archive_setting:
        raise ValueError("Missing required setting: impacts.activities.emissions_raw_archive")
    archive = registry.locate_required(archive_setting, label="activities.emissions_raw_archive")

    return {
        "region_name": region_name,
        "region_label": cfg_activities.get("region_label", region_name.upper()),
        "scenario": settings.impacts.scenario,
        "run_region": settings.run.region,
        "run_scenario": settings.run.scenario,
        "output_run_name": getattr(settings.run, "output_run_name", None),
        "seed": settings.impacts.seed,
        "output_root": local_output_folder,
        "archive": archive,
        "extract_root": local_input_folder,
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
            f"was not found at {archive}. Check activities.emissions_raw_archive in your settings."
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
    from .config.path_registry import build_registry
    from .config.settings import _build_activities_config_from_root
    from .config.settings import _build_activities_workflow

    cfg = _normalize_raw_input_paths(_resolve_activities_config(settings, config_path))
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
        "presim": {
            "region": cfg["run_region"],
            "scenario": cfg["run_scenario"],
            "output_run_name": cfg["output_run_name"],
        },
        "fleet": fleet,
        "activities": activities_override,
    }
    raw = _build_activities_config_from_root(merged_emfac)
    return _build_activities_workflow(raw, config_path)


# ---------------------------------------------------------------------------
# Shared step runner
# ---------------------------------------------------------------------------

def _run_emfac_step(workflow: dict[str, Any], *, step_name: str, runner) -> dict[str, Any]:
    from impacts.pipeline.emfac._common import raise_runtime_error, write_failure_trace, write_trace
    write_trace(workflow, f"{step_name}_start", {"step": step_name, "status": "started"})
    try:
        updated_workflow = runner(workflow)
    except Exception as error:
        write_failure_trace(workflow, step=step_name, error=error, payload={"status": "failed"})
        raise_runtime_error(step_name, error)
    write_trace(updated_workflow, f"{step_name}_success", {"step": step_name, "status": "completed"})
    return updated_workflow


# ---------------------------------------------------------------------------
# Activities workflow
# ---------------------------------------------------------------------------

def _run_activities_steps(workflow: dict[str, Any]) -> dict[str, Any]:
    from impacts.pipeline.emfac._common import write_failure_trace, write_trace
    from impacts.pipeline.emfac.activities.step1_prepare_emissions_and_activities_tables import run_step1
    from impacts.pipeline.emfac.activities.step2_build_comprehensive_project_analysis import run_step2
    from impacts.pipeline.emfac.activities.step3_fill_project_analysis_rates import run_step3
    from impacts.pipeline.emfac.activities.step4_finalize_output import run_step4

    run = workflow["run"]
    paths = workflow["paths"]
    print(f"\n{'='*50}")
    print(f"  EMFAC TABLE ASSEMBLY - {run['region_label']} {run['calendar_year']}")
    print(f"  Passenger Rates Output: {paths['final_output_passenger']}")
    print(f"  Passenger Inventory Activity by Model Year Output: {paths['final_activity_by_model_year_output_passenger']}")
    print(f"  Passenger Inventory Activity by EMFAC ID Output: {paths['final_activity_by_emfacid_output_passenger']}")
    print(f"  Passenger Inventory Final Fleet Output: {paths['final_fleet_output_passenger']}")
    print(f"  Freight Rates Output: {paths['final_output_freight']}")
    print(f"  Freight Inventory Activity by Model Year Output: {paths['final_activity_by_model_year_output_freight']}")
    print(f"  Freight Inventory Activity by EMFAC ID Output: {paths['final_activity_by_emfacid_output_freight']}")
    print(f"  Freight Inventory Final Fleet Output: {paths['final_fleet_output_freight']}")
    print(f"{'='*50}\n")

    write_trace(workflow, "workflow_start", {"status": "started", "run": run, "paths": paths})
    try:
        workflow = _run_emfac_step(workflow, step_name="step1_prepare_emissions_and_activities_tables", runner=run_step1)
        workflow = _run_emfac_step(workflow, step_name="step2_build_comprehensive_project_analysis", runner=run_step2)
        workflow = _run_emfac_step(workflow, step_name="step3_fill_project_analysis_rates", runner=run_step3)
        workflow = _run_emfac_step(workflow, step_name="step4_finalize_output", runner=run_step4)
    except Exception as error:
        if isinstance(error, RuntimeError):
            write_failure_trace(workflow, step="workflow", error=error, payload={"status": "failed"})
        raise
    write_trace(workflow, "workflow_success", {"status": "completed"})
    print("  DONE")
    return workflow


# ---------------------------------------------------------------------------
# Fleet workflow
# ---------------------------------------------------------------------------

def _missing_fleet_activities_outputs(workflow: dict[str, Any]) -> dict[str, Path]:
    activities = workflow["config"]["activities"]
    rates_store_root = Path(str(activities["emissions_store_root"])).expanduser().resolve()
    rates_store_dataset = rates_store_root / "dataset"
    rates_store_duckdb = rates_store_root / "dataset.duckdb"
    required_outputs = {
        "passenger_rates_file": Path(str(activities["passenger_rates_file"])),
        "passenger_activity_file": Path(str(activities["passenger_activity_file"])),
        "passenger_fleet_file": Path(str(activities["passenger_fleet_file"])),
        "freight_rates_file": Path(str(activities["freight_rates_file"])),
        "freight_activity_file": Path(str(activities["freight_activity_file"])),
        "freight_fleet_file": Path(str(activities["freight_fleet_file"])),
        "rates_store_dataset": rates_store_dataset,
        "rates_store_duckdb": rates_store_duckdb,
    }
    missing = {label: path for label, path in required_outputs.items() if not path.exists()}
    if not missing:
        partition_exists = any(rates_store_dataset.glob("emfacId=*/*.parquet"))
        if not partition_exists:
            missing["rates_store_partitions"] = rates_store_dataset
    return missing


def _ensure_fleet_activities_exist(workflow: dict[str, Any]) -> dict[str, Any]:
    missing = _missing_fleet_activities_outputs(workflow)
    if not missing:
        return workflow
    missing_paths = "\n".join(f"  missing: {path}" for path in missing.values())
    raise FileNotFoundError(
        "Fleet workflow requires EMFAC activities outputs to exist before running.\n"
        f"{missing_paths}"
    )


def _run_fleet_steps(workflow: dict[str, Any]) -> dict[str, Any]:
    from impacts.pipeline.emfac._common import write_failure_trace, write_trace
    from impacts.pipeline.emfac.fleet.step1_build_vehicle_types import run_step1
    from impacts.pipeline.emfac.fleet.step2_map_emfac_bus_bike import run_step2
    from impacts.pipeline.emfac.fleet.step3_map_emfac_atlas import run_step3
    from impacts.pipeline.emfac.fleet.step4_map_emfac_frism import run_step4

    output_path = workflow["config"]["output"]
    print(f"{'='*50}")
    print(f"  EMISSIONS PROCESSING - {str(workflow['area']).upper()} REGION")
    print(f"  Scenario: {workflow['scenario']}")
    print(f"  Output: {output_path}")
    print(f"{'='*50}")

    write_trace(workflow, "workflow_start", {
        "status": "started",
        "area": workflow["area"],
        "scenario": workflow["scenario"],
        "output": output_path,
        "paths": workflow["paths"],
    })
    try:
        workflow = _run_emfac_step(workflow, step_name="step1_build_vehicle_types", runner=run_step1)
        workflow = _run_emfac_step(workflow, step_name="step2_map_emfac_bus_bike", runner=run_step2)
        workflow = _run_emfac_step(workflow, step_name="step3_map_emfac_atlas", runner=run_step3)
        workflow = _run_emfac_step(workflow, step_name="step4_map_emfac_frism", runner=run_step4)
    except Exception as error:
        if isinstance(error, RuntimeError):
            write_failure_trace(workflow, step="workflow", error=error, payload={"status": "failed"})
        raise
    write_trace(workflow, "workflow_success", {"status": "completed"})
    if workflow.get("mapped_passenger_vehicles_file"):
        print(f"  Passenger vehicles file: {workflow['mapped_passenger_vehicles_file']}")
    if workflow.get("mapped_freight_carriers_file"):
        print(f"  Freight carriers file: {workflow['mapped_freight_carriers_file']}")
    if workflow.get("mapped_passenger_vehicle_types_file"):
        print(f"  Passenger vehicle types file: {workflow['mapped_passenger_vehicle_types_file']}")
    if workflow.get("built_freight_vehicle_types_file"):
        print(f"  Freight vehicle types file: {workflow['built_freight_vehicle_types_file']}")
    print("  DONE")
    return workflow


def run_fleet(
    config_path: str | Path | None = None,
    *,
    activities_manifest_path: str | Path | None = None,
) -> None:
    """Run the fleet mapping workflow against existing activities outputs."""
    from impacts.config.settings import load_default_fleet_workflow
    from impacts.config.settings import load_fleet_workflow
    from impacts.config.settings import load_fleet_workflow_from_activities_manifest
    from impacts.pipeline.emfac._common import raise_runtime_error

    try:
        if activities_manifest_path is not None:
            workflow = load_fleet_workflow_from_activities_manifest(activities_manifest_path)
        elif config_path is not None:
            workflow = load_fleet_workflow(config_path)
        else:
            workflow = load_default_fleet_workflow()
    except Exception as error:
        raise_runtime_error("config_load", error)
    workflow = _ensure_fleet_activities_exist(workflow)
    _run_fleet_steps(workflow)


# ---------------------------------------------------------------------------
# Activities provisioning
# ---------------------------------------------------------------------------

def ensure_emfac_activities_outputs(settings, config_path: Path) -> dict[str, Any]:
    cfg = _resolve_activities_config(settings, config_path)
    manifest_path = _activities_manifest_path(cfg)

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

    log_substep_banner("4", "run activities steps", logger=logger)
    _run_activities_steps(workflow)

    log_substep_banner("5", "validate outputs", logger=logger)
    if not _outputs_exist(workflow):
        output_dir = Path(str(workflow["paths"]["final_activity_by_emfacid_output_passenger"])).parent
        raise RuntimeError(
            f"EMFAC activities provisioning completed but expected outputs not found in {output_dir}."
        )
    logger.info("EMFAC activities outputs validated.")
    return _write_activities_manifest(
        workflow=workflow,
        settings_path=config_path,
        manifest_path=manifest_path,
    )
