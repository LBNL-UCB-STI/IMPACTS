# IMPACTS

`impacts` converts transportation emissions outputs into gridded air-quality concentration outputs and downstream exposure artifacts using InMAP/AERMOD-style workflows.

## Main commands

```bash
python -m impacts pipeline --config /path/settings.yaml
python -m impacts presim --config /path/settings.yaml
python -m impacts activities --config /path/settings.yaml
python -m impacts fleet --activities-manifest /path/activities_manifest.yaml
python -m impacts preprocess --config /path/settings.yaml
python -m impacts postsim --config /path/settings.yaml
python -m impacts analysis --config /path/settings.yaml
python -m impacts emissions --run-manifest /path/impacts_output/pipeline_manifest.yaml
python -m impacts inmap --run-manifest /path/impacts_output/pipeline_manifest.yaml
python -m impacts aermod --run-manifest /path/impacts_output/pipeline_manifest.yaml
python -m impacts exposure --run-manifest /path/impacts_output/pipeline_manifest.yaml
python -m impacts postprocess --run-manifest /path/impacts_output/pipeline_manifest.yaml
```

## Outputs

Workflow outputs are rooted at `impacts.local_output_folder`; managed inputs are rooted at `impacts.local_input_folder`. Relative paths resolve from the settings file directory first, then from the current working directory when an existing path is found.

The maintained manifests are:

- preprocess metadata: `preprocess_manifest.yaml`
- resumable pipeline state: `pipeline_manifest.yaml`
- postprocess metadata: `postprocess_manifest.yaml`

For `examples/pilates/settings.yaml`, `impacts.local_output_folder: impacts/impacts_output` resolves to `examples/pilates/impacts/impacts_output`.

## Notes

- The default ISRM location is `s3://inmap-model/isrm_v1.2.1.zarr/`.
- The canonical settings template lives at [settings.yaml](src/impacts/config/settings.yaml).
- HPC runs use [hpc/job_runner.sh](hpc/job_runner.sh), which writes Slurm logs under the configured `impacts.local_output_folder`.

## PILATES

PILATES-specific workflow details, example layout, settings shape, and example commands are in [docs/pilates.md](docs/pilates.md).
