from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from pathlib import Path
import pstats
import subprocess
import sys

from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.file_ops import load_structured_file
from impacts.manifest.file_ops import resolve_path

logger = logging.getLogger(__name__)
PROFILE_CHOICES = ("none", "time", "cpu", "memray", "all")


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


def _resolve_postsim_output_root(settings_path: str) -> Path:
    raw = os.environ.get("IMPACTS_POSTSIM_OUTPUT_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    settings = load_settings_from_yaml(settings_path)
    base_output_root = Path(
        resolve_path(settings.impacts.local_output_folder, settings_path) or settings.impacts.local_output_folder
    ).resolve()
    label = settings.run.output_run_name or settings.run.scenario
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = base_output_root / f"impacts-postsim--{settings.run.region}--{label}--{stamp}"
    suffix = 2
    while candidate.exists():
        candidate = base_output_root / f"impacts-postsim--{settings.run.region}--{label}--{stamp}_{suffix:02d}"
        suffix += 1
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
    postsim_output_root = _resolve_postsim_output_root(settings_path)
    postsim_output_root.mkdir(parents=True, exist_ok=True)
    logger.info("Postsim output directory: %s", postsim_output_root)
    pipeline_manifest_path: Path
    preprocess_manifest = preprocess_workflow(
        settings_path=settings_path,
        output_root_override=postsim_output_root,
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
    from impacts.config.path_registry import build_registry
    postprocess_from_pipeline_manifest(
        run_manifest_path=pipeline_manifest_path,
        output_root_override=postsim_output_root,
        input_roots=build_registry(settings, settings_path).roots,
    )
    logger.info("Postsim stage complete: pipeline_manifest=%s", pipeline_manifest_path)


def _resolve_profile_output(*, output_root: Path, stem: str) -> Path:
    candidate = output_root / "profiling" / stem
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def _run_output_root_from_manifest(manifest_path: str | Path) -> Path:
    manifest = load_structured_file(manifest_path)
    output_dir = manifest.get("output_dir")
    if not output_dir:
        raise ValueError(f"Manifest is missing output_dir: {manifest_path}")
    return Path(str(output_dir)).expanduser().resolve()


def _presim_output_root(settings_path: str | Path) -> Path:
    from impacts.config.settings import presim_run_root

    settings = load_settings_from_yaml(settings_path)
    base_output_root = _resolve_pipeline_output_root(str(settings_path))
    return presim_run_root(
        base_output_root,
        region=settings.run.region,
        output_run_name=settings.run.output_run_name,
        run_scenario=settings.run.scenario,
    ).resolve()


def _fleet_profile_output_root(activities_manifest_path: str | Path) -> Path:
    manifest = load_structured_file(activities_manifest_path)
    output_dir = manifest.get("output_dir")
    if output_dir:
        return Path(str(output_dir)).expanduser().resolve()
    outputs = manifest.get("outputs", {}) if isinstance(manifest.get("outputs"), dict) else {}
    outputs_root = outputs.get("outputs_root")
    if outputs_root:
        return Path(str(outputs_root)).expanduser().resolve()
    return Path(activities_manifest_path).expanduser().resolve().parent


def _postprocess_profile_output_root(args: argparse.Namespace) -> Path:
    if args.impact_output_dir:
        return Path(args.impact_output_dir).expanduser().resolve()
    if args.run_manifest:
        return _run_output_root_from_manifest(args.run_manifest)
    if args.config:
        return _resolve_pipeline_output_root(args.config)
    raise ValueError("Cannot resolve postprocess profile output root")


def _profile_forwarded_args(raw_argv: list[str]) -> list[str]:
    forwarded: list[str] = []
    skip_next = False
    for token in raw_argv:
        if skip_next:
            skip_next = False
            continue
        if token == "--profile":
            skip_next = True
            continue
        if token.startswith("--profile="):
            continue
        forwarded.append(token)
    return [*forwarded, "--profile", "none"]


def _resolve_profile_target(
    args: argparse.Namespace,
    raw_argv: list[str],
) -> tuple[Path, list[str], str, dict[str, str]]:
    forwarded_args = _profile_forwarded_args(raw_argv)
    extra_env: dict[str, str] = {}
    if args.command == "pipeline":
        output_root = _resolve_pipeline_output_root(args.config)
    elif args.command == "preprocess":
        output_root = _resolve_pipeline_output_root(args.config)
    elif args.command in {"presim", "activities"}:
        output_root = _presim_output_root(args.config)
    elif args.command == "postsim":
        output_root = _resolve_postsim_output_root(args.config)
        extra_env["IMPACTS_POSTSIM_OUTPUT_DIR"] = str(output_root)
    elif args.command == "fleet":
        output_root = _fleet_profile_output_root(args.activities_manifest)
    elif args.command in {"emissions", "inmap", "aermod", "exposure"}:
        output_root = _run_output_root_from_manifest(args.run_manifest)
    elif args.command == "postprocess":
        output_root = _postprocess_profile_output_root(args)
    else:
        raise ValueError(f"Profiling is not supported for command: {args.command}")
    return output_root, forwarded_args, str(args.command), extra_env


def _write_cpu_profile_report(profile_output: Path, report_output: Path) -> None:
    if not profile_output.exists():
        logger.warning("CPU profile output was not created: %s", profile_output)
        return
    with report_output.open("w", encoding="utf-8") as stream:
        stream.write("Top cumulative-time functions\n")
        stream.write("=============================\n\n")
        stats = pstats.Stats(str(profile_output), stream=stream)
        stats.strip_dirs().sort_stats("cumulative").print_stats(120)
        stream.write("\n\nTop internal-time functions\n")
        stream.write("===========================\n\n")
        stats.sort_stats("tottime").print_stats(120)


def _write_memray_summary(profile_output: Path, report_output: Path) -> None:
    if not profile_output.exists():
        logger.warning("Memray profile output was not created: %s", profile_output)
        return
    with report_output.open("w", encoding="utf-8") as stream:
        subprocess.run(
            [sys.executable, "-m", "memray", "summary", str(profile_output)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )


def _profile_artifacts(output_root: Path, profile_name: str) -> dict[str, Path]:
    return {
        "time": _resolve_profile_output(output_root=output_root, stem=f"{profile_name}.time.txt"),
        "cpu_pstats": _resolve_profile_output(output_root=output_root, stem=f"{profile_name}.cpu.pstats"),
        "cpu_report": _resolve_profile_output(output_root=output_root, stem=f"{profile_name}.cpu.txt"),
        "memray": _resolve_profile_output(output_root=output_root, stem=f"{profile_name}.memray.bin"),
        "memray_report": _resolve_profile_output(output_root=output_root, stem=f"{profile_name}.memray.txt"),
    }


def _time_prefix(output_path: Path) -> list[str]:
    time_verbose_flag = "-l" if sys.platform == "darwin" else "-v"
    return ["/usr/bin/time", time_verbose_flag, "-o", str(output_path)]


def _maybe_relaunch_with_profiler(args: argparse.Namespace, raw_argv: list[str]) -> int | None:
    if getattr(args, "profile", "none") == "none":
        return None
    if os.environ.get("IMPACTS_PROFILE_ACTIVE") == "1":
        return None
    resolved = _resolve_profile_target(args, raw_argv)
    output_root, forwarded_args, profile_name, extra_env = resolved
    artifacts = _profile_artifacts(output_root, profile_name)
    if args.profile == "memray":
        command = [
            sys.executable,
            "-m",
            "memray",
            "run",
            "-o",
            str(artifacts["memray"]),
            "-m",
            "impacts",
            *forwarded_args,
        ]
    elif args.profile == "cpu":
        command = [
            sys.executable,
            "-m",
            "impacts.profile_runner",
            "--pstats",
            str(artifacts["cpu_pstats"]),
            "--",
            *forwarded_args,
        ]
    elif args.profile == "all":
        command = [
            *_time_prefix(artifacts["time"]),
            sys.executable,
            "-m",
            "memray",
            "run",
            "-o",
            str(artifacts["memray"]),
            "-m",
            "impacts.profile_runner",
            "--pstats",
            str(artifacts["cpu_pstats"]),
            "--",
            *forwarded_args,
        ]
    else:
        command = [
            *_time_prefix(artifacts["time"]),
            sys.executable,
            "-m",
            "impacts",
            *forwarded_args,
        ]
    env = os.environ.copy()
    env["IMPACTS_PROFILE_ACTIVE"] = "1"
    env.update(extra_env)
    logger.info("Profiling enabled: mode=%s output_dir=%s", args.profile, output_root / "profiling")
    completed = subprocess.run(command, env=env, check=False)
    if args.profile in {"cpu", "all"}:
        _write_cpu_profile_report(artifacts["cpu_pstats"], artifacts["cpu_report"])
        logger.info("CPU profile written: %s", artifacts["cpu_report"])
    if args.profile in {"memray", "all"}:
        _write_memray_summary(artifacts["memray"], artifacts["memray_report"])
        logger.info("Memray profile written: %s", artifacts["memray"])
    return int(completed.returncode)


def _add_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile",
        choices=PROFILE_CHOICES,
        default="none",
        help=(
            "Optional profiler: none disables profiling; time records wall/resource usage; "
            "cpu writes cProfile pstats and text reports; memray records memory allocations; "
            "all records time, CPU, and memray outputs."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts",
        description="Terminal-model contract for the impacts pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Stage explicit inputs and write preprocess_manifest.yaml")
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--manifest-path")
    _add_profile_argument(preprocess)

    presim = subparsers.add_parser("presim", help="Run the pre-simulation group from settings")
    presim.add_argument("--config", required=True)
    _add_profile_argument(presim)

    activities = subparsers.add_parser("activities", help="Run only the EMFAC activities stage from settings")
    activities.add_argument("--config", required=True)
    _add_profile_argument(activities)

    fleet = subparsers.add_parser("fleet", help="Run only the EMFAC fleet stage from an activities manifest")
    fleet.add_argument("--activities-manifest", required=True)
    _add_profile_argument(fleet)

    emissions = subparsers.add_parser("emissions", help="Run only the emissions stage from a run manifest")
    emissions.add_argument("--run-manifest", required=True)
    _add_profile_argument(emissions)

    inmap = subparsers.add_parser("inmap", help="Run only the InMAP concentration stage from a run manifest")
    inmap.add_argument("--run-manifest", required=True)
    _add_profile_argument(inmap)

    aermod = subparsers.add_parser("aermod", help="Run only the AERMOD concentration stage from a run manifest")
    aermod.add_argument("--run-manifest", required=True)
    _add_profile_argument(aermod)

    exposure = subparsers.add_parser("exposure", help="Run only the exposure preparation stage from a run manifest")
    exposure.add_argument("--run-manifest", required=True)
    _add_profile_argument(exposure)

    postsim = subparsers.add_parser("postsim", help="Run the post-simulation group from settings")
    postsim.add_argument("--config", required=True)
    _add_profile_argument(postsim)

    postprocess = subparsers.add_parser("postprocess", help="Run postprocess comparisons and map plots")
    postprocess_group = postprocess.add_mutually_exclusive_group(required=True)
    postprocess_group.add_argument("--run-manifest", help="Path to an existing pipeline_manifest.yaml.")
    postprocess_group.add_argument("--config", help="Path to a settings YAML (re-runs the full pipeline first).")
    postprocess_group.add_argument(
        "--impact-output-dir",
        help="Path to a completed pipeline output folder containing pipeline_manifest.yaml.",
    )
    postprocess.add_argument("--postprocess-manifest")
    postprocess.add_argument(
        "--impact-input-dir",
        action="append",
        default=None,
        help=(
            "Additional local IMPACTS input directory used to resolve source paths recorded in manifests. "
            "For copied PILATES/BEAM outputs, pass the local production region folder, e.g. "
            "/path/to/beam-data/sfbay. Can be supplied more than once."
        ),
    )
    postprocess.add_argument(
        "--baseline-concentration-parquet",
        default=None,
        help=(
            "Path to a baseline beam_concentration_distribution.parquet for delta concentration "
            "and delta exposure analysis (steps 6-7). Overrides delta_baseline_concentration_distribution_file in settings."
        ),
    )
    _add_profile_argument(postprocess)

    pipeline = subparsers.add_parser("pipeline", help="Run the maintained stage sequence from settings, honoring impacts.pipeline flags")
    pipeline.add_argument("--config", required=True)
    _add_profile_argument(pipeline)
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
    build_nox_to_no2.add_argument("--output-file", required=True)
    build_nox_to_no2.add_argument("--cmaq-ratio-table")
    build_nox_to_no2.add_argument("--grid-polygon")
    build_nox_to_no2.add_argument("--geopoints-file")
    build_nox_to_no2.add_argument("--regional-matrix")
    build_nox_to_no2.add_argument("--isrm-zarr", required=True)

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
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    profiling_exit_code = _maybe_relaunch_with_profiler(args, raw_argv)
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

        baseline = args.baseline_concentration_parquet
        if args.run_manifest:
            postprocess_from_pipeline_manifest(
                run_manifest_path=args.run_manifest,
                manifest_path=args.postprocess_manifest,
                input_roots=args.impact_input_dir,
                baseline_concentration_override=baseline,
            )
        elif args.config:
            postprocess_from_settings(
                settings_path=args.config,
                manifest_path=args.postprocess_manifest,
                input_roots=args.impact_input_dir,
                baseline_concentration_override=baseline,
            )
        else:
            pipeline_manifest = Path(args.impact_output_dir) / "pipeline_manifest.yaml"
            if not pipeline_manifest.exists():
                parser.error(f"pipeline_manifest.yaml not found in {args.impact_output_dir}")
            postprocess_from_pipeline_manifest(
                run_manifest_path=pipeline_manifest,
                manifest_path=args.postprocess_manifest,
                output_root_override=args.impact_output_dir,
                input_roots=args.impact_input_dir,
                baseline_concentration_override=baseline,
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
        from impacts.tools.inmap.build_complete_nox_to_no2_matrix import build_complete_matrix

        output_path = build_complete_matrix(
            output_file=args.output_file,
            isrm_zarr=args.isrm_zarr,
            ratio_table_path=args.cmaq_ratio_table,
            regional_matrix_path=args.regional_matrix,
            grid_path=args.grid_polygon,
            geopoints_file=args.geopoints_file,
        )
        print(output_path)
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


if __name__ == "__main__":
    main()
