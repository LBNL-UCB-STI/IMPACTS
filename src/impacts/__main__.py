from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.file_ops import load_structured_file
from impacts.manifest.file_ops import resolve_path


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



def _resolve_run_output_root(*, input_manifest_path: str | None) -> Path:
    if not input_manifest_path:
        raise ValueError("run profiling requires input_manifest_path")
    manifest = load_structured_file(input_manifest_path)
    settings_source = manifest.get("settings_source")
    if not settings_source:
        raise ValueError("Input manifest is missing settings_source; cannot resolve impacts.local_output_folder.")
    return _resolve_pipeline_output_root(str(settings_source))


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
    if args.command == "run":
        output_root = _resolve_run_output_root(
            input_manifest_path=args.input_manifest,
        )
        forwarded_args = [
            "run",
            "--input-manifest",
            args.input_manifest,
            *(["--run-manifest", args.run_manifest] if args.run_manifest else []),
            "--profile",
            "none",
        ]
        default_name = "run"
        return output_root, forwarded_args, default_name
    return None


def _maybe_relaunch_with_profiler(args: argparse.Namespace) -> int | None:
    if args.profile == "none":
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

    preprocess = subparsers.add_parser("preprocess", help="Stage explicit inputs and write inputs_manifest.yaml")
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--manifest-path")

    run = subparsers.add_parser("run", help="Run the maintained impacts pipeline from staged inputs only")
    run.add_argument("--input-manifest", required=True)
    run.add_argument("--run-manifest")
    run.add_argument("--profile", choices=("none", "memray", "time"), default="none")
    run.add_argument(
        "--allow-heavy-profile",
        action="store_true",
        help="Explicitly opt in to memray profiling for full run executions.",
    )

    postprocess = subparsers.add_parser("postprocess", help="Publish the canonical impacts exposure table artifact")
    postprocess_group = postprocess.add_mutually_exclusive_group(required=True)
    postprocess_group.add_argument("--run-manifest")
    postprocess_group.add_argument("--config")
    postprocess.add_argument("--output-dir")
    postprocess.add_argument("--postprocess-manifest")

    analysis = subparsers.add_parser("analysis", help="Run maintained analysis outputs from workflow artifacts")
    analysis.add_argument("--config", required=True)

    emfac = subparsers.add_parser("emfac", help="Run EMFAC activities, fleet, or the full EMFAC workflow")
    emfac.add_argument(
        "workflow",
        nargs="?",
        choices=("activities", "fleet"),
        default=None,
        help="Optional workflow to run. Omit to run the full EMFAC workflow.",
    )
    emfac.add_argument("--config", required=True)

    pipeline = subparsers.add_parser("pipeline", help="Run preprocess, run, and postprocess end-to-end")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--profile", choices=("none", "memray", "time"), default="none")
    derive_settings = subparsers.add_parser(
        "derive_settings_from_pilates",
        help="Generate an impacts settings file from main PILATES settings and a thin impacts overlay.",
    )
    derive_settings.add_argument("--pilates-settings", required=True)
    derive_settings.add_argument("--model-config", required=True)
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "pipeline" and args.profile == "memray":
        parser.error("--profile memray is only supported on 'run'; full pipeline profiling is too expensive.")
    if args.command == "run" and args.profile == "memray" and not args.allow_heavy_profile:
        parser.error(
            "--profile memray for 'run' requires --allow-heavy-profile; use /usr/bin/time -l for full real runs."
        )
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

    if args.command == "run":
        from impacts.runner import run_from_input_manifest

        run_from_input_manifest(
            input_manifest_path=args.input_manifest,
            run_manifest_path=args.run_manifest,
        )
        return 0

    if args.command == "postprocess":
        from impacts.postprocessor import postprocess_from_run_manifest
        from impacts.postprocessor import postprocess_from_settings

        if args.run_manifest:
            if not args.output_dir:
                parser.error("--output-dir is required with --run-manifest")
            postprocess_from_run_manifest(
                run_manifest_path=args.run_manifest,
                output_dir=args.output_dir,
                manifest_path=args.postprocess_manifest,
            )
        else:
            postprocess_from_settings(
                settings_path=args.config,
                manifest_path=args.postprocess_manifest,
            )
        return 0

    if args.command == "pipeline":
        from impacts.postprocessor import postprocess_from_run_manifest
        from impacts.preprocessor import preprocess_workflow
        from impacts.runner import run_from_input_manifest

        _print_pipeline_banner()
        downstream_output_root = _resolve_pipeline_output_root(args.config)
        preprocess_manifest = preprocess_workflow(
            settings_path=args.config,
        )
        run_manifest = run_from_input_manifest(
            input_manifest_path=preprocess_manifest["inputs_manifest_path"],
            run_dispersion=True,
        )
        postprocess_from_run_manifest(
            run_manifest_path=run_manifest["run_manifest_path"],
            output_dir=downstream_output_root,
        )
        return 0

    if args.command == "analysis":
        from impacts.runner import run_analysis_from_settings

        run_analysis_from_settings(
            settings_path=args.config,
        )
        return 0

    if args.command == "emfac":
        from impacts.emfac.__main__ import main as run_emfac_main

        run_emfac_main(
            [
                *([args.workflow] if args.workflow else []),
                "--config",
                args.config,
            ]
        )
        return 0

    if args.command == "derive_settings_from_pilates":
        from impacts.pipeline.adapters.pilates import derive_settings_from_pilates

        derive_settings_from_pilates(
            pilates_settings_path=args.pilates_settings,
            impacts_model_config_path=args.model_config,
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
        from impacts.inmap.rdata_conversion import rdata_to_parquet

        written = rdata_to_parquet(
            input_path=args.input,
            output_dir=args.output_dir,
            prefix=args.prefix,
        )
        for path in written:
            print(path)
        return 0

    if args.command == "build_nox_to_no2":
        from impacts.inmap.build_complete_nox_to_no2_matrix import main as build_nox_to_no2_main

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
