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
   - resolves explicit upstream inputs from a workflow YAML
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
python -m impacts preprocess --workflow-config /path/workflow.yaml --staging-dir /path/workspace
python -m impacts run --input-manifest /path/workspace/inputs_manifest.yaml --output-dir /path/workspace/output
python -m impacts postprocess --run-manifest /path/workspace/output/run_manifest.yaml --output-dir /path/workspace/output
```

End-to-end:

```bash
python -m impacts pipeline --workflow-config /path/workflow.yaml --workspace /path/workspace
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

Population integration is currently best-effort. If staged ActivitySim tables do not yet contain a usable cell identifier, provide a `population_cell_mapping_path` in the workflow config and the postprocessor will use it. When neither is available, the canonical artifact still exists but population mix fields remain empty placeholders.

### Workflow config inputs

The orchestrated contract resolves the following maintained inputs:

- BEAM events via `emissions_events.events_input_path`
- or precomputed BEAM emissions skims via `emissions_grid_mapping.skims_input_path`
- link lengths via `emissions_events.link_length_path` or `osm_grid.beam_network_path`
- emissions rates via `emissions_rates.rates_dir`
- either:
  - `emissions_grid_mapping.mapping_input_path`, or
  - `osm_grid.osm_links_path`, `osm_grid.beam_network_path`, and `osm_grid.grid_cells_path`
- optional OSM/BEAM source placeholders via `osm_grid.osm_pbf_path` and `osm_grid.beam_mapdb_path`
- ISRM source via `dispersion_isrm.isrm_url`
- optional ActivitySim inputs via `activitysim_population.*`

### PILATES-facing config stub

A concrete example config block for PILATES lives in [pilates_model_config.yaml](/Users/haitamlaarabi/Workspace/Models/inmap-aermod/impacts/examples/pilates/pilates_model_config.yaml). It includes:

- local input and output dirs
- container input and output dirs
- entrypoint and command template
- canonical output filenames
- manifest filenames
- explicit placeholders for integration details that are not finalized yet

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
python examples/pilates/run_example.py
```

That example:

- stages explicit upstream inputs
- runs the maintained contract end to end
- writes manifests under `examples/pilates/workspace`
- publishes a canonical exposure table
- uses a fake in-memory ISRM store by default so it works without external data access


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
