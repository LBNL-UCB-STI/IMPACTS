from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts",
        description="Terminal-model contract for the impacts pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Stage explicit inputs and write inputs_manifest.yaml")
    preprocess.add_argument("--workflow-config", required=True)
    preprocess.add_argument("--staging-dir", required=True)
    preprocess.add_argument("--manifest-path")
    preprocess_alias = subparsers.add_parser(
        "impacts_preprocess",
        help="Alias for preprocess to match the PILATES terminal-model contract naming.",
    )
    preprocess_alias.add_argument("--workflow-config", required=True)
    preprocess_alias.add_argument("--staging-dir", required=True)
    preprocess_alias.add_argument("--manifest-path")

    run = subparsers.add_parser("run", help="Run the maintained impacts pipeline from staged inputs only")
    run.add_argument("--input-manifest", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--run-manifest")
    run_alias = subparsers.add_parser(
        "impacts_run",
        help="Alias for run to match the PILATES terminal-model contract naming.",
    )
    run_alias.add_argument("--input-manifest", required=True)
    run_alias.add_argument("--output-dir", required=True)
    run_alias.add_argument("--run-manifest")

    postprocess = subparsers.add_parser("postprocess", help="Publish the canonical impacts exposure table artifact")
    postprocess.add_argument("--run-manifest", required=True)
    postprocess.add_argument("--output-dir", required=True)
    postprocess.add_argument("--postprocess-manifest")
    postprocess_alias = subparsers.add_parser(
        "impacts_postprocess",
        help="Alias for postprocess to match the PILATES terminal-model contract naming.",
    )
    postprocess_alias.add_argument("--run-manifest", required=True)
    postprocess_alias.add_argument("--output-dir", required=True)
    postprocess_alias.add_argument("--postprocess-manifest")

    pipeline = subparsers.add_parser("pipeline", help="Run preprocess, run, and postprocess end-to-end")
    pipeline.add_argument("--workflow-config", required=True)
    pipeline.add_argument("--workspace", required=True)

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
    sample_skims.add_argument("--population-sample", type=float, default=1.0)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"preprocess", "impacts_preprocess"}:
        from impacts.preprocessor import preprocess_workflow

        preprocess_workflow(
            workflow_config_path=args.workflow_config,
            staging_dir=args.staging_dir,
            manifest_path=args.manifest_path,
        )
        return 0

    if args.command in {"run", "impacts_run"}:
        from impacts.runner import run_from_input_manifest

        run_from_input_manifest(
            input_manifest_path=args.input_manifest,
            output_dir=args.output_dir,
            run_manifest_path=args.run_manifest,
        )
        return 0

    if args.command in {"postprocess", "impacts_postprocess"}:
        from impacts.postprocessor import postprocess_from_run_manifest

        postprocess_from_run_manifest(
            run_manifest_path=args.run_manifest,
            output_dir=args.output_dir,
            manifest_path=args.postprocess_manifest,
        )
        return 0

    if args.command == "pipeline":
        from impacts.postprocessor import postprocess_from_run_manifest
        from impacts.preprocessor import preprocess_workflow
        from impacts.runner import run_from_input_manifest

        workspace = Path(args.workspace).resolve()
        preprocess_manifest = preprocess_workflow(
            workflow_config_path=args.workflow_config,
            staging_dir=workspace,
        )
        run_manifest = run_from_input_manifest(
            input_manifest_path=preprocess_manifest["inputs_manifest_path"],
            output_dir=workspace / "output",
        )
        postprocess_from_run_manifest(
            run_manifest_path=run_manifest["run_manifest_path"],
            output_dir=workspace / "output",
        )
        return 0

    if args.command == "sample_events":
        from impacts.sampling import sample_events_by_vehicle

        sample_events_by_vehicle(
            input_path=args.input,
            output_path=args.output,
            fraction=args.fraction,
            seed=args.seed,
            vehicle_column=args.vehicle_column,
        )
        return 0

    if args.command == "sample_skims":
        from impacts.sampling import sample_skims_by_fraction

        sample_skims_by_fraction(
            input_path=args.input,
            output_path=args.output,
            fraction=args.fraction,
            seed=args.seed,
            compact_workers=args.compact_workers,
            population_sample=args.population_sample,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    main()
