from __future__ import annotations

import argparse
from pathlib import Path

from impacts.config.settings_builder import load_settings_from_yaml
from impacts.manifest.file_ops import resolve_path


def _print_pipeline_banner() -> None:
    print()
    print("========== IMPACTS PIPELINE ==========")
    print(
        "IMPACTS stages inputs, prepares spatial joins, allocates corrected transportation emissions, "
        "computes air-quality concentrations, and publishes the final exposure artifact."
    )
    print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts",
        description="Terminal-model contract for the impacts pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Stage explicit inputs and write inputs_manifest.yaml")
    preprocess.add_argument("--config", required=True)
    preprocess.add_argument("--staging-dir", required=True)
    preprocess.add_argument("--manifest-path")

    run = subparsers.add_parser("run", help="Run the maintained impacts pipeline from staged inputs only")
    run_group = run.add_mutually_exclusive_group(required=True)
    run_group.add_argument("--input-manifest")
    run_group.add_argument("--config")
    run.add_argument("--output-dir")
    run.add_argument("--workspace")
    run.add_argument("--run-manifest")

    postprocess = subparsers.add_parser("postprocess", help="Publish the canonical impacts exposure table artifact")
    postprocess_group = postprocess.add_mutually_exclusive_group(required=True)
    postprocess_group.add_argument("--run-manifest")
    postprocess_group.add_argument("--config")
    postprocess.add_argument("--output-dir")
    postprocess.add_argument("--workspace")
    postprocess.add_argument("--postprocess-manifest")

    pipeline = subparsers.add_parser("pipeline", help="Run preprocess, run, and postprocess end-to-end")
    pipeline.add_argument("--config", required=True)
    pipeline.add_argument("--workspace", required=True)
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
        help="Aggregate EMFAC county-year totVMT and totTrips across one or more files.",
    )
    aggregate_emfac_activity.add_argument("--input", dest="inputs", action="append", required=True)
    aggregate_emfac_activity.add_argument("--output", required=True)
    aggregate_emfac_activity.add_argument("--county-col", default="countyfp")
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

    build_emfac_rates_store = subparsers.add_parser(
        "build_emfac_rates_store",
        help="Convert an EMFAC rates folder into partitioned Parquet and DuckDB outputs.",
    )
    build_emfac_rates_store.add_argument("--input-dir", required=True)
    build_emfac_rates_store.add_argument("--output-dir", required=True)
    build_emfac_rates_store.add_argument("--compression", default="zstd")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        from impacts.preprocessor import preprocess_workflow

        preprocess_workflow(
            settings_path=args.config,
            staging_dir=args.staging_dir,
            manifest_path=args.manifest_path,
        )
        return 0

    if args.command == "run":
        from impacts.runner import run_from_input_manifest
        from impacts.runner import run_from_settings

        if args.input_manifest:
            if not args.output_dir:
                parser.error("--output-dir is required with --input-manifest")
            run_from_input_manifest(
                input_manifest_path=args.input_manifest,
                output_dir=args.output_dir,
                run_manifest_path=args.run_manifest,
            )
        else:
            if not args.workspace:
                parser.error("--workspace is required with --config")
            run_from_settings(
                settings_path=args.config,
                workspace=args.workspace,
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
            if not args.workspace:
                parser.error("--workspace is required with --config")
            postprocess_from_settings(
                settings_path=args.config,
                workspace=args.workspace,
                manifest_path=args.postprocess_manifest,
            )
        return 0

    if args.command == "pipeline":
        from impacts.postprocessor import postprocess_from_run_manifest
        from impacts.preprocessor import preprocess_workflow
        from impacts.runner import run_from_input_manifest

        _print_pipeline_banner()
        workspace = Path(args.workspace).resolve()
        settings = load_settings_from_yaml(args.config)
        downstream_output_root = Path(
            resolve_path(settings.impacts.local_output_folder, args.config) or settings.impacts.local_output_folder
        ).resolve()
        preprocess_manifest = preprocess_workflow(
            settings_path=args.config,
            staging_dir=workspace,
        )
        run_manifest = run_from_input_manifest(
            input_manifest_path=preprocess_manifest["inputs_manifest_path"],
            output_dir=workspace,
            run_dispersion=True,
        )
        postprocess_from_run_manifest(
            run_manifest_path=run_manifest["run_manifest_path"],
            output_dir=downstream_output_root,
        )
        return 0

    if args.command == "derive_settings_from_pilates":
        from impacts.adapters.pilates import derive_settings_from_pilates

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
            year_col=args.year_col,
            vmt_col=args.vmt_col,
            trips_col=args.trips_col,
            default_year=args.default_year,
            county_fips_filters=args.county_fips_filters,
            year_filters=args.year_filters,
        )
        return 0

    if args.command == "build_emfac_rates_store":
        from impacts.tools.emfac.build_emfac_rates_store import build_emfac_rates_store

        result = build_emfac_rates_store(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            compression=args.compression,
        )
        print(f"input_file_count={result['input_file_count']}")
        print(f"parquet_file_count={result['parquet_file_count']}")
        print(f"output_dir={result['output_dir']}")
        print(f"parquet_root={result['parquet_root']}")
        print(f"duckdb_path={result['duckdb_path']}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":



    main()
