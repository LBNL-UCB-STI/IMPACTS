# IMPACTS

Current release review: [Complete version review](docs/FULL_VERSION_REVIEW.md)
and [file-by-file change index](docs/CHANGE_INDEX.md).

For LDV charging, use [run_ldv_charging.py](scripts/run_ldv_charging.py):
the original collaborator algorithm and notebook parameters are preserved.
See [charging compatibility](docs/CRISTIAN_COMPATIBILITY.md) for inputs, checks,
and the distinction from archived designed Home80 sensitivity cases.

`impacts` converts transportation emissions outputs into gridded air-quality concentration outputs and downstream exposure artifacts using InMAP/AERMOD-style workflows.

## Main commands

```bash
python -m impacts pipeline --config /path/settings.yaml
python -m impacts presim --config /path/settings.yaml
python -m impacts activities --config /path/settings.yaml
python -m impacts fleet --activities-manifest /path/activities_manifest.yaml
python -m impacts preprocess --config /path/settings.yaml
python -m impacts postsim --config /path/settings.yaml
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
- HPC runs use [hpc/job_runner.sh](hpc/job_runner.sh), which detects the repository root from the script location and writes Slurm logs under the configured `impacts.local_output_folder`. Set `IMPACTS_DIR` only when intentionally running against a different checkout.

## PILATES

PILATES-specific workflow details, example layout, settings shape, and example commands are in [docs/pilates.md](docs/pilates.md).

## Reviewed research extensions (2026-11-01)

This checkout preserves the upstream CLI and adds events-to-EMFAC emissions,
near-source ISRM removal/AERMOD truncation, charging and power-sector workflows,
and human-exposure tables, figures and an interactive map.

Start with the [change and code map](docs/RELEASE_REVIEW.md),
[reproduction guide](docs/REPRODUCING.md), and
[validation results](docs/VALIDATION.md). Input data, ISRM/ASRV matrices,
collaborator notebooks, external power allocator, and generated results are
not bundled. The source archive is a review candidate; the upstream
`LICENSE.txt` contains `TBD` despite MIT package metadata. Resolve that
inconsistency before assigning a distribution license.

For the current hybrid method, explicitly set
`impacts.exposure.primary_pm25_integration_strategy: additive_inmap_aermod`.
The built-in YAML, example configurations and Python fallback now default to this strategy.
Traffic primary PM2.5, BC and NO2 use the 3 km near-source removal policy;
secondary PM2.5 and power-sector responses do not. Full `inmap_only_*` fields
preserve the baseline without removal. Concentration units are micrograms/m3.

The latest power workflow uses a new hourly charging curve, hourly
Cambium/eGRID allocation, annual plant emissions and an explicit ISRM layer.
Use [run_hourly_power.py](scripts/run_hourly_power.py) with externally supplied
assets. Recent reference cases used the explicitly authorized **Layer 0
approximation**, not elevated Layer 1; this limitation remains in their audit.

The native `impacts.power_generation` YAML route and command-line entry point
both call `pipeline/workflow/hourly_power.py`. The former WA/annual-energy
scaling method has been removed; old settings produce a migration error.
EVI profiles are comparison inputs, not event or power-demand replacements.
See [the hourly-only update](docs/HOURLY_POWER_UPDATE.md) for changes and validation.

## Charging code organization

See [charging modules and migration](docs/CHARGING_MODULES.md) for model selection,
entry points, and the old-to-new module mapping.

Historical electricity: see [the 2018 configuration, data years and validation limits](docs/HISTORICAL_POWER_2018.md).
