# IMPACTS

`impacts` converts transportation emissions outputs into gridded air-quality concentration outputs and downstream exposure artifacts using InMAP/AERMOD-style workflows.

## Main commands

```bash
python -m impacts presim --config /path/settings.yaml
python -m impacts preprocess --config /path/settings.yaml
python -m impacts emissions --run-manifest /path/workspace/run_manifest.yaml
python -m impacts inmap --run-manifest /path/workspace/run_manifest.yaml
python -m impacts aermod --run-manifest /path/workspace/run_manifest.yaml
python -m impacts exposure --run-manifest /path/workspace/run_manifest.yaml
python -m impacts postprocess --run-manifest /path/workspace/run_manifest.yaml
python -m impacts pipeline --config /path/settings.yaml
```

## Outputs

- internal working files: `/path/workspace/staged` and `/path/workspace/outputs`
- run metadata: `/path/workspace/inputs_manifest.yaml` and `/path/workspace/run_manifest.yaml`
- published artifact: `/path/impacts/impacts_complete.txt`

## Notes

- The default ISRM location is `s3://inmap-model/isrm_v1.2.1.zarr/`.
- Exploratory code under `src/impacts/scratch/` is not part of the maintained execution path.
- The canonical settings template lives at [settings.yaml](src/impacts/config/settings.yaml).

## PILATES

PILATES-specific workflow details, example layout, settings shape, and example commands are in [docs/pilates.md](docs/pilates.md).
