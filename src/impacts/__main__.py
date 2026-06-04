from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys

from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.file_ops import resolve_path

logger = logging.getLogger(__name__)


def _print_pipeline_banner() -> None:
    print()
    print("========== IMPACTS PIPELINE ==========")
    print(
        "IMPACTS stages inputs, prepares spatial joins, allocates corrected transportation emissions, "
        "computes air-quality concentrations, and publishes the final exposure artifact."
    )
    print()


def _resolve_pipeline_output_root(settings_path: str) -> Path:
    settings = load_settings_from_yaml(settings_path)
    return Path(
        resolve_path(settings.impacts.local_output_folder, settings_path) or settings.impacts.local_output_folder
    ).resolve()


def _resolve_pipeline_manifest_path(settings_path: str, manifest_name: str) -> Path:
    candidate = _resolve_pipeline_output_root(settings_path) / manifest_name
    if not candidate.exists():
        raise FileNotFoundError(
            f"Expected {manifest_name} in the configured impacts.local_output_folder, but it was not found: {candidate}"
        )
    return candidate


def _run_presim_from_settings(settings_path: str) -> None:
    from impacts.provisioner import run_fleet
    from impacts.provisioner import ensure_emfac_activities_outputs

    settings = load_settings_from_yaml(settings_path)
    activities_manifest_path: str | None = None
    if settings.impacts.pipeline.presim.activities or settings.impacts.pipeline.presim.fleet:
        activities_manifest = ensure_emfac_activities_outputs(settings, Path(settings_path))
        activities_manifest_path = activities_manifest["activities_manifest_path"]
    if settings.impacts.pipeline.presim.fleet:
        run_fleet(activities_manifest_path=activities_manifest_path)
    if activities_manifest_path:
        logger.info("Presim stage complete: activities_manifest=%s", activities_manifest_path)
    else:
        logger.info("Presim stage complete: no presim stages enabled")


def _run_postsim_from_settings(
    settings_path: str,
) -> None:
    from impacts.postprocessor import postprocess_from_pipeline_manifest
    from impacts.preprocessor import preprocess_workflow

    settings = load_settings_from_yaml(settings_path)
    pipeline_manifest_path: Path
    preprocess_manifest = preprocess_workflow(
        settings_path=settings_path,
    )
    pipeline_manifest_path = Path(preprocess_manifest["pipeline_manifest_path"]).resolve()
    if (
        settings.impacts.pipeline.postsim.emissions
        or settings.impacts.pipeline.postsim.inmap
        or settings.impacts.pipeline.postsim.aermod
        or settings.impacts.pipeline.postsim.exposure
    ):
        from impacts.runner import run_aermod_from_pipeline_manifest
        from impacts.runner import run_emissions_from_pipeline_manifest
        from impacts.runner import run_exposure_from_pipeline_manifest
        from impacts.runner import run_inmap_from_pipeline_manifest

        if settings.impacts.pipeline.postsim.emissions:
            run_manifest = run_emissions_from_pipeline_manifest(
                run_manifest_path=pipeline_manifest_path,
            )
            pipeline_manifest_path = Path(run_manifest["pipeline_manifest_path"]).resolve()
        if settings.impacts.pipeline.postsim.inmap:
            run_manifest = run_inmap_from_pipeline_manifest(
                run_manifest_path=pipeline_manifest_path,
            )
            pipeline_manifest_path = Path(run_manifest["pipeline_manifest_path"]).resolve()
        if settings.impacts.pipeline.postsim.aermod:
            run_manifest = run_aermod_from_pipeline_manifest(
                run_manifest_path=pipeline_manifest_path,
            )
            pipeline_manifest_path = Path(run_manifest["pipeline_manifest_path"]).resolve()
        if settings.impacts.pipeline.postsim.exposure:
            run_manifest = run_exposure_from_pipeline_manifest(
                run_manifest_path=pipeline_manifest_path,
            )
            pipeline_manifest_path = Path(run_manifest["pipeline_manifest_path"]).resolve()
    else:
        pipeline_manifest_path = _resolve_pipeline_manifest_path(settings_path, "pipeline_manifest.yaml")
    postprocess_from_pipeline_manifest(
        run_manifest_path=pipeline_manifest_path,
    )
    logger.info("Postsim stage complete: pipeline_manifest=%s", pipeline_manifest_path)


def _resolve_profile_output(*, output_root: Path, stem: str) -> Path:
    candidate = output_root / "profiling" / stem
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _resolve_profile_target(args: argparse.Namespace) -> tuple[Path, list[str], str] | None:
    if args.profile == "none":
        return None
    if args.command == "pipeline":
        output_root = _resolve_pipeline_output_root(args.config)
        forwarded_args = ["pipeline", "--config", args.config, "--profile", "none"]
        default_name = "pipeline"
        return output_root, forwarded_args, default_name


def _maybe_relaunch_with_profiler(args: argparse.Namespace) -> int | None:
    if getattr(args, "profile", "none") == "none":
        return None
    if os.environ.get("IMPACTS_PROFILE_ACTIVE") == "1":
        return None
    resolved = _resolve_profile_target(args)
    if resolved is None:
        return None
    output_root, forwarded_args, default_name = resolved
    default_stem = f"{default_name}.memray" if args.profile == "memray" else f"{default_name}.time.txt"
    profile_output = _resolve_profile_output(output_root=output_root, stem=default_stem)
    if args.profile == "memray":
        command = [
            sys.executable,
            "-m",
            "memray",
            "run",
            "-o",
            str(profile_output),
            "-m",
            "impacts",
            *forwarded_args,
        ]
    else:
        time_verbose_flag = "-l" if sys.platform == "darwin" else "-v"
        command = [
            "/usr/bin/time",
            time_verbose_flag,
            "-o",
            str(profile_output),
            sys.executable,
            "-m",
            "impacts",
            *forwarded_args,
        ]
    env = os.environ.copy()
    env["IMPACTS_PROFILE_ACTIVE"] = "1"
    completed = subprocess.run(command, env=env, check=False)
    return int(completed.returncode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts",
        description="Terminal-model contract for the impacts pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Stage explicit inputs and write preprocess_manifest.yaml")
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--manifest-path")

    presim = subparsers.add_parser("presim", help="Run the pre-simulation group from settings")
    presim.add_argument("--config", required=True)

    activities = subparsers.add_parser("activities", help="Run only the EMFAC activities stage from settings")
    activities.add_argument("--config", required=True)

    fleet = subparsers.add_parser("fleet", help="Run only the EMFAC fleet stage from an activities manifest")
    fleet.add_argument("--activities-manifest", required=True)

    emissions = subparsers.add_parser("emissions", help="Run only the emissions stage from a run manifest")
    emissions.add_argument("--run-manifest", required=True)

    inmap = subparsers.add_parser("inmap", help="Run only the InMAP concentration stage from a run manifest")
    inmap.add_argument("--run-manifest", required=True)

    aermod = subparsers.add_parser("aermod", help="Run only the AERMOD concentration stage from a run manifest")
    aermod.add_argument("--run-manifest", required=True)

    exposure = subparsers.add_parser("exposure", help="Run only the exposure preparation stage from a run manifest")
    exposure.add_argument("--run-manifest", required=True)

    postsim = subparsers.add_parser("postsim", help="Run the post-simulation group from settings")
    postsim.add_argument("--config", required=True)

    postprocess = subparsers.add_parser("postprocess", help="Publish the canonical impacts exposure table artifact")
    postprocess_group = postprocess.add_mutually_exclusive_group(required=True)
    postprocess_group.add_argument("--run-manifest", help="Path to an existing pipeline_manifest.yaml.")
    postprocess_group.add_argument("--config", help="Path to a settings YAML (re-runs the full pipeline first).")
    postprocess_group.add_argument(
        "--output-dir",
        help="Path to a completed pipeline output folder containing pipeline_manifest.yaml.",
    )
    postprocess.add_argument("--postprocess-manifest")

    pipeline = subparsers.add_parser("pipeline", help="Run the maintained stage sequence from settings, honoring impacts.pipeline flags")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--profile", choices=("none", "memray", "time"), default="none")
    derive_settings = subparsers.add_parser(
        "derive_settings_from_pilates",
        help="Generate an impacts settings file from main PILATES settings and the built-in impacts template.",
    )
    derive_settings.add_argument("--pilates-settings", required=True)
    derive_settings.add_argument("--output", required=True)

    sample_events = subparsers.add_parser(
        "sample_events",
        help="Create a smaller events sample by vehicle id.",
    )
    sample_events.add_argument("--input", required=True)
    sample_events.add_argument("--output", required=True)
    sample_events.add_argument("--fraction", type=float, required=True)
    sample_events.add_argument("--seed", type=int, default=42)
    sample_events.add_argument("--vehicle-column", default="vehicle")

    sample_skims = subparsers.add_parser(
        "sample_skims",
        help="Create a smaller skims sample by row fraction.",
    )
    sample_skims.add_argument("--input", required=True)
    sample_skims.add_argument("--output", required=True)
    sample_skims.add_argument("--fraction", type=float, required=True)
    sample_skims.add_argument("--seed", type=int, default=42)
    sample_skims.add_argument("--compact-workers", type=int, default=4)

    rdata_to_parquet = subparsers.add_parser(
        "rdata_to_parquet",
        help="Convert one RData file into one or more parquet files.",
    )
    rdata_to_parquet.add_argument("--input", required=True)
    rdata_to_parquet.add_argument("--output-dir", required=True)
    rdata_to_parquet.add_argument("--prefix")

    build_nox_to_no2 = subparsers.add_parser(
        "build_nox_to_no2",
        help="Build the workflow-ready full-domain NOx-to-NO2 matrix artifact.",
    )
    build_nox_to_no2.add_argument("--input-dir", required=True)
    build_nox_to_no2.add_argument("--output-dir", required=True)
    build_nox_to_no2.add_argument("--isrm-zarr")
    build_nox_to_no2.add_argument("--output-name")

    aggregate_emfac_activity = subparsers.add_parser(
        "aggregate_emfac_activity",
        help="Aggregate EMFAC county-year activity and write annual totals with explicit units.",
    )
    aggregate_emfac_activity.add_argument("--input", dest="inputs", action="append", required=True)
    aggregate_emfac_activity.add_argument("--output", required=True)
    aggregate_emfac_activity.add_argument("--county-col", default="countyfp")
    aggregate_emfac_activity.add_argument("--vehicle-category-col", default="vehicleCategory")
    aggregate_emfac_activity.add_argument("--year-col", default="year")
    aggregate_emfac_activity.add_argument("--vmt-col", default="totVMT")
    aggregate_emfac_activity.add_argument("--trips-col", default="totTrips")
    aggregate_emfac_activity.add_argument("--year", dest="default_year", type=int)
    aggregate_emfac_activity.add_argument(
        "--county-fips",
        dest="county_fips_filters",
        nargs="+",
        default=None,
        help="Filter to one or more county FIPS codes.",
    )
    aggregate_emfac_activity.add_argument(
        "--filter-year",
        dest="year_filters",
        nargs="+",
        type=int,
        default=None,
        help="Filter to one or more years.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pipeline" and args.profile == "memray":
        parser.error("--profile memray is not supported for 'pipeline'; use --profile time for full runs.")
    profiling_exit_code = _maybe_relaunch_with_profiler(args)
    if profiling_exit_code is not None:
        return profiling_exit_code

    if args.command == "preprocess":
        from impacts.preprocessor import preprocess_workflow

        preprocess_workflow(
            settings_path=args.config,
            manifest_path=args.manifest_path,
        )
        return 0

    if args.command == "presim":
        _run_presim_from_settings(args.config)
        return 0

    if args.command == "activities":
        from impacts.provisioner import ensure_emfac_activities_outputs

        settings = load_settings_from_yaml(args.config)
        ensure_emfac_activities_outputs(settings, Path(args.config))
        return 0

    if args.command == "fleet":
        from impacts.provisioner import run_fleet

        run_fleet(activities_manifest_path=args.activities_manifest)
        return 0

    if args.command == "emissions":
        from impacts.runner import run_emissions_from_pipeline_manifest

        run_emissions_from_pipeline_manifest(
            run_manifest_path=args.run_manifest,
        )
        return 0

    if args.command == "inmap":
        from impacts.runner import run_inmap_from_pipeline_manifest

        run_inmap_from_pipeline_manifest(
            run_manifest_path=args.run_manifest,
        )
        return 0

    if args.command == "aermod":
        from impacts.runner import run_aermod_from_pipeline_manifest

        run_aermod_from_pipeline_manifest(
            run_manifest_path=args.run_manifest,
        )
        return 0

    if args.command == "exposure":
        from impacts.runner import run_exposure_from_pipeline_manifest

        run_exposure_from_pipeline_manifest(
            run_manifest_path=args.run_manifest,
        )
        return 0

    if args.command == "postprocess":
        from impacts.postprocessor import postprocess_from_pipeline_manifest
        from impacts.postprocessor import postprocess_from_settings

        if args.run_manifest:
            postprocess_from_pipeline_manifest(
                run_manifest_path=args.run_manifest,
                manifest_path=args.postprocess_manifest,
            )
        elif args.config:
            postprocess_from_settings(
                settings_path=args.config,
                manifest_path=args.postprocess_manifest,
            )
        else:
            pipeline_manifest = Path(args.output_dir) / "pipeline_manifest.yaml"
            if not pipeline_manifest.exists():
                parser.error(f"pipeline_manifest.yaml not found in {args.output_dir}")
            postprocess_from_pipeline_manifest(
                run_manifest_path=pipeline_manifest,
                manifest_path=args.postprocess_manifest,
            )
        return 0

    if args.command == "postsim":
        _run_postsim_from_settings(args.config)
        return 0

    if args.command == "pipeline":
        _print_pipeline_banner()
        settings = load_settings_from_yaml(args.config)
        if settings.impacts.pipeline.presim.activities or settings.impacts.pipeline.presim.fleet:
            _run_presim_from_settings(args.config)
        _run_postsim_from_settings(args.config)
        logger.info("Pipeline command complete")
        return 0

    if args.command == "derive_settings_from_pilates":
        from impacts.config.settings_builder import derive_settings_from_pilates

        derive_settings_from_pilates(
            pilates_settings_path=args.pilates_settings,
            output_path=args.output,
        )
        return 0

    if args.command == "sample_events":
        from impacts.tools.beam.sample_beam_output import sample_events_by_vehicle

        sample_events_by_vehicle(
            input_path=args.input,
            output_path=args.output,
            fraction=args.fraction,
            seed=args.seed,
            vehicle_column=args.vehicle_column,
        )
        return 0

    if args.command == "sample_skims":
        from impacts.tools.beam.sample_beam_output import sample_skims_by_fraction

        sample_skims_by_fraction(
            input_path=args.input,
            output_path=args.output,
            fraction=args.fraction,
            seed=args.seed,
            compact_workers=args.compact_workers,
        )
        return 0

    if args.command == "rdata_to_parquet":
        from impacts.tools.inmap.rdata_conversion import rdata_to_parquet

        written = rdata_to_parquet(
            input_path=args.input,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
        for path in written:
            print(path)
        return 0

    if args.command == "build_nox_to_no2":
        from impacts.tools.inmap.build_complete_nox_to_no2_matrix import main as build_nox_to_no2_main

        build_nox_to_no2_main(
            [
                "--input-dir",
                args.input_dir,
                "--output-dir",
                args.output_dir,
                *(["--isrm-zarr", args.isrm_zarr] if args.isrm_zarr else []),
                *(["--output-name", args.output_name] if args.output_name else []),
            ]
        )
        return 0

    if args.command == "aggregate_emfac_activity":
        from impacts.tools.emfac.aggregate_emfac_activity import aggregate_emfac_activity

        aggregate_emfac_activity(
            input_paths=args.inputs,
            output_path=args.output,
            county_col=args.county_col,
            vehicle_category_col=args.vehicle_category_col,
            year_col=args.year_col,
            vmt_col=args.vmt_col,
            trips_col=args.trips_col,
            default_year=args.default_year,
            county_fips_filters=args.county_fips_filters,
            year_filters=args.year_filters,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    main()
