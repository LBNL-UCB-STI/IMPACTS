# PILATES-Like Example

This example simulates a PILATES integration locally.

This example now uses one user-managed settings file:

1. `settings.yaml`
   - PILATES-style source of truth for the example
   - includes shared run context plus `impacts.runtime_overrides`
   - is the only config you edit for the example

During preprocess, `impacts` stages a resolved runtime contract under `workspace/staged/config/runtime.yaml` for inspection and execution. That staged file is generated and should not be edited.

The current example settings shape is:

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

The builder maps that into the executable runtime contract staged under `workspace/staged/config/runtime.yaml`.

For the example:

1. external handoff inputs live under `upstream/`
2. internal working files live under `workspace/`
3. published downstream-facing artifacts live under `downstream/`
4. `settings.yaml` is the only checked-in settings file for the example
5. `run_pilates_example.py` calls `preprocess` and `run`
6. the example intentionally stops after emissions allocation

The example uses the configured `inputs.isrm_zarr` directly. Point that setting at a real local or remote Zarr store.

## Run

From the repo root:

```bash
python examples/pilates/run_pilates_example.py
```

That produces:

- `examples/pilates/workspace/inputs_manifest.yaml`
- `examples/pilates/workspace/run_manifest.yaml`
- `examples/pilates/workspace/outputs/emissions_inmap_grid_allocated.parquet`
  - or `emissions_inmap_grid_allocated.csv.gz` when parquet support is not installed
- optionally `examples/pilates/workspace/outputs/emissions_aermod_grid_allocated.parquet`
  - when `processing.grid.aermod_grid_path` is configured

## Build Sample Inputs From Real BEAM Outputs

If your real `events.csv.gz` and `skimsEmissionsTotals.csv.gz` files are too large, create smaller samples first.

Events are sampled by vehicle id, so each selected vehicle keeps its full trace. Skims are sampled by row fraction.

From the repo root:

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

That writes:

- `examples/pilates/upstream/events_sample.csv.gz`
- `examples/pilates/upstream/skimsEmissionsTotals_sample.csv.gz`

You can also run the same logic through the package CLI:

```bash
python -m impacts sample_events \
  --input /path/to/0.events.csv.gz \
  --output examples/pilates/upstream/events_sample.csv.gz \
  --fraction 0.05
```

```bash
python -m impacts sample_skims \
  --input /path/to/0.skimsEmissionsTotals.csv.gz \
  --output examples/pilates/upstream/skimsEmissionsTotals_sample.csv.gz \
  --fraction 0.05
```

## Relation to PILATES

This mirrors the expected black-box terminal-model contract:

- staged inputs only
- deterministic run command
- deterministic allocated-emissions artifact for pre-dispersion inspection
- no implicit filesystem assumptions

The example keeps all user-managed settings in `settings.yaml` and stages the generated runtime contract only inside `workspace/`.
