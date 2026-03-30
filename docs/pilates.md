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
- `src/impacts/adapters/pilates_settings.yaml` is the maintained starting template for the impacts overlay inside PILATES settings.

That file is a thin impacts overlay that lives directly under `impacts`.

During preprocess, `impacts` stages a resolved runtime contract under:

- `workspace/staged/config/runtime.yaml`

That staged runtime file is generated and should not be edited directly.

## Example settings shape

```yaml
run:
  region: sfbay
  scenario: base
  start_year: 2017

shared:
  geography:
    FIPS:
      state: "06"
      counties:
        - "001"
        - "013"
        - "041"
        - "055"
        - "075"
        - "081"
        - "085"
        - "095"
        - "097"
    local_crs: EPSG:26910

impacts:
  local_input_folder: upstream/
  local_output_folder: downstream/
  emissions:
    simulation_network_folder: upstream/
    osm_network_folder: upstream/
    emissions_rates_folder: upstream/
    activity_totals_file: upstream/Default_Statewide_2018_2025_2030_2040_2050_Annual_activity_totals.parquet
    annualization_days: 330
    pollutants:
      - NH3
      - NOx
      - PM2_5
      - SOx
      - ROG
      - BC
  dispersions:
    inmap:
      isrm_zarr: s3://inmap-model/isrm_v1.2.1.zarr/
      isrm_nox_to_no2_matrix_npz: '{local_input_folder}/nox_to_no2_full_isrm_matrix.npz'
      grid_path: '{local_input_folder}/isrm_polygon/isrm_polygon.shp'
      grid_id: isrm
      grid_epsg:
    aermod:
      grid_path:
      grid_epsg:
      grid_id:
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
  --input /path/to/0.events.parquet \
  --fraction 0.05
```

```bash
python examples/pilates/prepare_sample_data.py skims \
  --input /path/to/0.skimsEmissions.parquet \
  --fraction 0.05
```

Outputs land in:

- `examples/pilates/upstream/events_sample.parquet`
- `examples/pilates/upstream/skimsEmissions_sample.parquet`
