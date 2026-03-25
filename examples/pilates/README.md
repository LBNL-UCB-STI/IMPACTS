# PILATES-Like Example

This example simulates a PILATES integration locally.

There are two config layers:

1. `pilates_model_config.yaml`
   - thin PILATES-facing `impacts` overlay
   - keeps only `impacts`-specific settings that are not already present in the main PILATES settings
   - uses a higher-level structure:
     - `runtime_overrides.emissions`
     - `runtime_overrides.dispersions`
     - `runtime_overrides.outputs`
2. `runtime.yaml`
   - generated executable runtime config for `impacts`
   - can be generated from the main PILATES settings plus `pilates_model_config.yaml`
   - should not be treated as the source of truth for shared PILATES settings

Important separation:

- standalone/example mode:
  - `runtime.yaml` contains explicit file paths
  - no BEAM output discovery is required
  - you do not need to replicate a full `beam_output/.../ITERS/...` tree unless you want to
- integrated/PILATES mode:
  - the PILATES adapter derives concrete paths before execution
  - BEAM artifact discovery happens there, not in preprocess/run/postprocess
  - the adapter translates the higher-level overlay into the native `runtime.yaml` contract consumed by execution

The intended no-redundancy path is:

```bash
python -m impacts derive_runtime_config_from_pilates \
  --pilates-settings /path/to/pilates_settings.yaml \
  --model-config examples/pilates/pilates_model_config.yaml \
  --output examples/pilates/runtime.yaml
```

That command derives shared geography settings such as county FIPS and local CRS from the main PILATES settings, then applies the `impacts.runtime_overrides` section from `pilates_model_config.yaml`.

In integrated mode it also derives BEAM-side inputs when PILATES provides the needed BEAM config, including:

- `inputs.osm_pbf`
- `inputs.beam_network`
- `inputs.emissions_skims`

The current overlay shape is:

```yaml
impacts:
  runtime_overrides:
    emissions:
      annualization_days: 330
      activity_correction_factors_file: null
      pollutants:
        - NH3
        - NOx
        - PM2_5
        - SOx
        - ROG
        - BCh
    dispersions:
      inmap:
        isrm_zarr_directory: isrm_zarr_directory
        isrm_zarr_s3bucket: s3://example/isrm.zarr
        grid_path: upstream/isrm_polygon/isrm_polygon.shp
        grid_epsg: 4326
        grid_id: zone_isrm
      aermod:
        grid_path: null
        grid_epsg: 4326
        grid_id: null
    outputs:
      output_dir: ./impacts/output
```

The adapter maps that into the executable native runtime config fields in `runtime.yaml`.

For the example:

1. upstream model outputs live under `upstream/`
2. `runtime.yaml` is a native `impacts` runtime config with fully resolved inputs and processing settings
3. `run_pilates_example.py` calls `preprocess` and `run`
4. the example intentionally stops after emissions allocation
5. outputs land in `workspace/`

The example uses the configured `inputs.isrm_zarr` directly. Point that runtime setting at a real local or remote Zarr store.

## Run

From the repo root:

```bash
python examples/pilates/run_pilates_example.py
```

That produces:

- `examples/pilates/workspace/inputs_manifest.yaml`
- `examples/pilates/workspace/output/run_manifest.yaml`
- `examples/pilates/workspace/output/raw/emissions_inmap_grid_allocated.parquet`
  - or `emissions_inmap_grid_allocated.csv.gz` when parquet support is not installed
- optionally `examples/pilates/workspace/output/raw/emissions_aermod_grid_allocated.parquet`
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

The example also includes a PILATES-facing config stub in `pilates_model_config.yaml` with an `impacts.runtime_overrides` section for paths and settings that are specific to the `impacts` model and should not be duplicated in the main PILATES settings.
