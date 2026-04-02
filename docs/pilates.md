# PILATES Integration

This document describes the `impacts` integration pattern for PILATES and the example scenario under `examples/pilates/`.

## Example layout

```text
examples/pilates/
  beam/
    beam_output/
  impacts/
  pilates/
    beam/
      production/
  workspace/
    staged/
    outputs/
```

Meaning:

- `beam/beam_output/`: BEAM run outputs consumed by `impacts`
- `pilates/beam/production/`: staged production inputs referenced by the example settings
- `impacts/`: downstream-facing published artifacts
- `workspace/staged/`: copied and derived execution inputs
- `workspace/outputs/`: internal step outputs

## Settings model

The example uses one user-managed settings file:

- `examples/pilates/settings.yaml`
- `src/impacts/adapters/pilates_settings.yaml` is the maintained starting template for the impacts overlay inside PILATES settings.

That file is a thin impacts overlay that lives directly under `impacts`.

During preprocess, `impacts` registers the user-provided settings file in the inputs manifest and writes generated intermediates under the workspace.

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

beam:
  local_input_folder: pilates/beam/production/
  local_output_folder: beam/beam_output/

impacts:
  local_input_folder: impacts/input/
  local_output_folder: impacts/impacts_output/
  emissions:
    osm_network_folder: r5/sfbay-cbg5500-weakConn-network
    emissions_rates_folder: vehicle-tech/emissions/2018-Baseline
    activity_totals_file: vehicle-tech/emissions/SFBay_2018_Annual_activity_totals.parquet
    annualization_days: 330
    population_sample: 0.1
    pollutants_map:
      NH3: NH3
      NOx: NOx
      PM2_5: PM2_5
      SOx: SOx
      ROG: ROG
      BC: BCh
  dispersions:
    inmap:
      enabled: true
      isrm_zarr: ~/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr
      isrm_nox_to_no2_matrix_npz: vehicle-tech/dispersions/inmap/nox_to_no2_full_isrm_matrix.npz
      isrm_nox_to_no2_matrix_factor: 1.0
      grid_path: vehicle-tech/dispersions/inmap/isrm_polygon_wgs84.gpkg
      grid_id: isrm
      grid_epsg: 4326
    aermod:
      enabled: true
      grid_size_meters: 100.0
      asrv_patterns_file: vehicle-tech/dispersions/aermod/aermod_patterns_wgs84.parquet
      asrv_patterns_epsg: 4326
  exposure:
    enabled: true
    population_folder: urbansim/2018
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

The full pipeline command produces:

- `examples/pilates/workspace/inputs_manifest.yaml`
- `examples/pilates/workspace/run_manifest.yaml`
- `examples/pilates/workspace/outputs/emissions_inmap_grid_allocated.parquet`
- optionally `examples/pilates/workspace/outputs/emissions_aermod_grid_allocated.parquet`
- `examples/pilates/impacts/impacts_output/impacts_exposure_table.parquet`

The helper script currently stops after emissions allocation and does not publish the final exposure artifact.

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

- `examples/pilates/beam/events_sample.parquet`
- `examples/pilates/beam/skimsEmissions_sample.parquet`
