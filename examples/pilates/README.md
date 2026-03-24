# PILATES-Like Example

This example simulates a PILATES integration locally:

1. upstream model outputs live under `upstream/`
2. `workflow.yaml` points at those upstream files
3. `run_pilates_example.py` calls `preprocess` and `run`
4. the example intentionally stops after emissions are distributed across ISRM cells
5. outputs land in `workspace/`

The example uses the configured `dispersion_isrm.isrm_url` directly. Point that workflow setting at a real local or remote Zarr store.

## Run

From the repo root:

```bash
python examples/pilates/run_pilates_example.py
```

That produces:

- `examples/pilates/workspace/inputs_manifest.yaml`
- `examples/pilates/workspace/output/run_manifest.yaml`
- `examples/pilates/workspace/output/raw/emissions_grid_allocated.parquet`
  - or `emissions_grid_allocated.csv.gz` when parquet support is not installed

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

The example also includes a PILATES-facing config stub in `pilates_model_config.yaml` with placeholder fields for any integration details that are not yet finalized.
