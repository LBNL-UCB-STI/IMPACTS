# IMPACTS: Intersection of Mobility Patterns and Air Concentration in Transportation Systems

A Python toolkit for working with InMAP/AERMOD Source-Receptor Matrices (ISRM) and converting BEAM emission outputs into pollutant concentrations.

## Overview

This package provides utilities to:
- Convert InMAP outputs to NOx-to-NO2 source-receptor relationships
- Map these relationships to ISRM grid cells
- Process BEAM emission data for air quality analysis

### Data Sources

The InMAP ISRM (Intervention Source-Receptor Matrix) is downloaded based on configuration, with the default location being:
- **S3 Bucket**: `s3://inmap-model/isrm_v1.2.1.zarr/`
- **Version**: v1.2.1

This ISRM matrix is then combined with the NOx-to-NO2 matrix calculated in-house to produce the final source-receptor relationships for air quality analysis.

## PILATES Terminal Contract

This repository now exposes a production-facing terminal-model contract for PILATES:

- `impacts.preprocessor`
- `impacts.runner`
- `impacts.postprocessor`

The maintained execution path is:

- `src/impacts/emissions/events_to_skims_emissions.py`
- `src/impacts/emissions/emissions_grid_mapping.py`
- `src/impacts/dispersion/isrm_dispersion.py`
- `src/impacts/network2grid/network_grid_clipping.py`

Legacy or exploratory code under `src/impacts/tmp/` is not part of the public execution contract.

### Contract shape

1. `preprocess`
   - resolves explicit upstream inputs from a runtime config YAML
   - stages deterministic inputs into a clean directory
   - writes `inputs_manifest.yaml`
2. `run`
   - consumes staged inputs only
   - executes the maintained impacts pipeline
   - writes raw outputs and `run_manifest.yaml`
3. `postprocess`
   - validates required raw outputs
   - publishes one canonical `impacts_exposure_table` artifact
   - writes `postprocess_manifest.yaml`

### CLI

```bash
python -m impacts preprocess --config /path/runtime.yaml --staging-dir /path/workspace
python -m impacts run --input-manifest /path/workspace/inputs_manifest.yaml --output-dir /path/workspace/output
python -m impacts postprocess --run-manifest /path/workspace/output/run_manifest.yaml --output-dir /path/workspace/output
```

End-to-end:

```bash
python -m impacts pipeline --config /path/runtime.yaml --workspace /path/workspace
```

### Canonical artifact

The postprocessor publishes:

- `output/canonical/impacts_exposure_table.parquet`
  - falls back to `output/canonical/impacts_exposure_table.csv.gz` if parquet support is unavailable

Each row is one cell and includes:

- `cell_id`
- `geometry_reference`
- exposure metrics from the raw concentration output
- `population_total`
- `households_total`
- `population_mix` as a structured JSON payload

Population integration is currently best-effort. Staged ActivitySim tables should carry a usable cell identifier. When they do not, the canonical artifact still exists but population mix fields remain empty placeholders.

### Runtime config inputs

The maintained runtime contract resolves the following inputs from `runtime.yaml`:

- `inputs.beam_network`
- `inputs.emissions_skims`
- `inputs.osm_pbf`
- optional `inputs.osm_links`
- optional `inputs.beam_mapdb`
- optional `inputs.activity_corrections`
- optional `inputs.isrm_zarr`
- optional ActivitySim population tables:
  - `inputs.households_asim_out`
  - `inputs.persons_asim_out`
- `processing.grid.inmap_grid_path`
- optional `processing.grid.aermod_grid_path`
- `processing.pollutants`
- `processing.annualization_days`

Standalone runs should provide these as explicit resolved paths in `runtime.yaml`.
They do not need to replicate a PILATES or BEAM output directory structure.

Integrated runs are different:

- `adapters/pilates.py` derives concrete runtime paths from PILATES settings
- this includes BEAM artifact discovery such as:
  - `inputs.osm_pbf` from `beam.local_input_folder` and `beam.router_directory`
  - `inputs.beam_network` from the discovered BEAM run root
  - `inputs.emissions_skims` from the latest discovered `ITERS/it.x/x.skimsEmissionsTotals.csv.gz`
- execution still consumes only the explicit derived `runtime.yaml`

### PILATES-facing config stub

A concrete example config block for PILATES lives in [pilates_model_config.yaml](/Users/haitamlaarabi/Workspace/Models/inmap-aermod/impacts/examples/pilates/pilates_model_config.yaml). It includes:

- a thin `impacts.runtime_overrides` section
- only `impacts`-specific settings that are not already derivable from PILATES shared or BEAM config

### Docker usage

Build:

```bash
docker build -t impacts .
```

Run with staged host inputs and outputs only:

```bash
docker run --rm \
  -v /path/workspace/input:/input \
  -v /path/workspace/output:/output \
  impacts run --input-manifest /input/inputs_manifest.yaml --output-dir /output
```

Then publish the canonical artifact:

```bash
docker run --rm \
  -v /path/workspace/output:/output \
  impacts postprocess --run-manifest /output/run_manifest.yaml --output-dir /output
```

The container no longer depends on baked-in `/work/data` for the public execution path.

## Examples

The repo includes a standalone PILATES example under [examples/pilates](/Users/haitamlaarabi/Workspace/Models/inmap-aermod/impacts/examples/pilates).

Run it from the repo root:

```bash
python examples/pilates/run_pilates_example.py
```

That example:

- stages explicit upstream inputs
- runs the maintained contract end to end
- writes manifests under `examples/pilates/workspace`
- publishes a canonical exposure table
- uses the configured runtime inputs directly


## Installation #WIP

```bash
# Clone the repository
git clone <repository-url>
cd impacts

# Install dependencies with Poetry
poetry install

# Or activate the environment
poetry shell
```

## Project Structure #WIP

```
isrm-wrapper/
├── src/isrm_wrapper/    # Main package code
├── data/
│   ├── raw/            # Input BEAM emissions and InMAP outputs
│   └── processed/      # Generated concentration outputs
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Requirements #WIP

- Python 3.8+
- Dependencies managed via Poetry (see `pyproject.toml`)

## Contributing #WIP

See `CONTRIBUTING.rst` for guidelines.

## License #WIP

See `LICENSE.txt` for details.
