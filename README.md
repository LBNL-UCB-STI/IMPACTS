# IMPACTS

`impacts` converts transportation emissions outputs into gridded air-quality concentration outputs and downstream exposure artifacts using InMAP/AERMOD-style workflows.

## Main commands

```bash
python -m impacts preprocess --config /path/settings.yaml --staging-dir /path/workspace
python -m impacts run --input-manifest /path/workspace/inputs_manifest.yaml --output-dir /path/workspace
python -m impacts postprocess --run-manifest /path/workspace/run_manifest.yaml --output-dir /path/impacts
python -m impacts pipeline --config /path/settings.yaml --workspace /path/workspace
```

## Outputs

- internal working files: `/path/workspace/staged` and `/path/workspace/outputs`
- run metadata: `/path/workspace/inputs_manifest.yaml` and `/path/workspace/run_manifest.yaml`
- published artifact: `/path/impacts/impacts_exposure_table.parquet`

## Notes

- The default ISRM location is `s3://inmap-model/isrm_v1.2.1.zarr/`.
- Legacy or exploratory code under `src/impacts/scratch/` is not part of the maintained execution path.
- A maintained PILATES overlay template lives at [pilates_settings.yaml](/Users/haitamlaarabi/Workspace/Models/inmap-aermod/impacts/src/impacts/adapters/pilates_settings.yaml).

## PILATES

PILATES-specific workflow details, example layout, settings shape, and example commands are in [docs/pilates.md](docs/pilates.md).
