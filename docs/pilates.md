# PILATES Integration

This document describes the `impacts` integration pattern for PILATES and the example scenario under `examples/pilates/`.

## Example layout

```text
examples/pilates/
  upstream/
  downstream/
  workspace/
    staged/
    outputs/
```

Meaning:

- `upstream/`: external handoff inputs
- `workspace/staged/`: copied and derived execution inputs
- `workspace/outputs/`: internal step outputs
- `downstream/`: published downstream-facing artifacts

## Settings model

The example uses one user-managed settings file:

- `examples/pilates/settings.yaml`

That file is PILATES-style source of truth and includes shared run context plus `impacts.runtime_overrides`.

During preprocess, `impacts` stages a resolved runtime contract under:

- `workspace/staged/config/runtime.yaml`

That staged runtime file is generated and should not be edited directly.

## Example settings shape

```yaml
run:
  region: sfbay
  start_year: 2017

impacts:
  runtime_overrides:
    inputs:
      beam_network: upstream/network.csv.gz
      emissions_skims: upstream/0.skimsEmissionsTotals_5pct_sample.csv.gz
      osm_pbf: upstream/sfbay-cbg5500-weakConn-network.osm.pbf
      activity_corrections: upstream/activity_corrections.csv
      isrm_zarr: s3://inmap-model/isrm_v1.2.1.zarr/
    processing:
      annualization_days: 330
      pollutants:
        - NH3
        - NOx
        - PM2_5
        - SOx
        - ROG
        - BCh
      grid:
        inmap_grid_path: upstream/isrm_polygon/isrm_polygon.shp
        inmap_grid_id: isrm
    outputs:
      output_dir: downstream
```

## Running the example

From the repo root:

```bash
python -m impacts pipeline --config examples/pilates/settings.yaml --workspace examples/pilates/workspace
```

Or with the helper script:

```bash
python examples/pilates/run_pilates_example.py
```

That produces:

- `examples/pilates/workspace/inputs_manifest.yaml`
- `examples/pilates/workspace/run_manifest.yaml`
- `examples/pilates/workspace/outputs/emissions_inmap_grid_allocated.parquet`
- optionally `examples/pilates/workspace/outputs/emissions_aermod_grid_allocated.parquet`
- `examples/pilates/downstream/impacts_exposure_table.parquet`

## Preparing smaller sample inputs

If your BEAM artifacts are too large, create smaller samples first.

Events are sampled by vehicle id, so each selected vehicle keeps its full trace. Skims are sampled by row fraction.

```bash
python examples/pilates/prepare_sample_data.py events \
  --input /path/to/0.events.csv.gz \
  --fraction 0.05
```

```bash
python examples/pilates/prepare_sample_data.py skims \
  --input /path/to/0.skimsEmissionsTotals.csv.gz \
  --fraction 0.05
```

Outputs land in:

- `examples/pilates/upstream/events_sample.csv.gz`
- `examples/pilates/upstream/skimsEmissionsTotals_sample.csv.gz`
