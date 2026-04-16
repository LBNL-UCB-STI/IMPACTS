# PILATES Integration

This document describes the `impacts` integration pattern for PILATES and the example scenario under `examples/pipeline/pilates/`.

## Example layout

```text
examples/pipeline/pilates/
  beam/
    beam_output/
    production/
  impacts/
```

Meaning:

- `beam/beam_output/`: BEAM run outputs consumed by `impacts`
- `beam/production/`: production inputs referenced by the example settings
- `impacts/`: downstream-facing published artifacts

## Settings model

The example uses one user-managed settings file:

- `examples/pipeline/pilates/settings.yaml`
- `src/impacts/pipeline/settings.yaml` is the maintained starting template for the impacts overlay inside PILATES settings.

That file is a thin impacts overlay kept with the example tree.

During preprocess, `impacts` registers the user-provided settings file in the inputs manifest and writes managed artifacts under the configured `impacts.local_input_folder` and `impacts.local_output_folder`.

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
  local_input_folder: beam/production/
  local_output_folder: beam/beam_output/

impacts:
  local_input_folder: impacts/input/
  local_output_folder: impacts/impacts_output/
  emissions:
    osm_network_folder: r5/sfbay-cbg5500-weakConn-network
    emissions_rates_folder: vehicle-tech/emissions/2018-Baseline
    activity_totals_file: vehicle-tech/emissions/SFBay_2018_Annual_activity_totals.parquet
    annualization_days_or_file: src/impacts/emfac/activities/vehicle_operation_days_per_year.csv
    population_sample: 0.1
    pollutants: [NH3, NOx, PM25, SOx, ROG, BC]
  dispersions:
    inmap:
      enabled: true
      isrm_zarr: ~/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr
      isrm_nox_to_no2_matrix_npz: vehicle-tech/dispersions/inmap/nox_to_no2_full_isrm_matrix.npz
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
python -m impacts pipeline --config examples/pipeline/pilates/settings.yaml
```

The full pipeline command produces:

- `examples/pipeline/pilates/impacts/impacts_output/inputs_manifest.yaml`
- `examples/pipeline/pilates/impacts/impacts_output/run_manifest.yaml`
- `examples/pipeline/pilates/impacts/impacts_output/beam_emissions_for_inmap.parquet`
- optionally `examples/pipeline/pilates/impacts/impacts_output/beam_emissions_for_aermod.parquet`
- `examples/pipeline/pilates/impacts/impacts_output/beam_concentration_distribution.parquet`

## Preparing smaller sample inputs

If your BEAM artifacts are too large, create smaller samples first.

Events are sampled by vehicle id, so each selected vehicle keeps its full trace. Skims are sampled by row fraction.

```bash
python examples/pipeline/pilates/prepare_sample_data.py events \
  --input /path/to/0.events.parquet \
  --fraction 0.05
```

```bash
python examples/pipeline/pilates/prepare_sample_data.py skims \
  --input /path/to/0.skimsEmissions.parquet \
  --fraction 0.05
```

Outputs land in:

- `examples/pipeline/pilates/beam/events_sample.parquet`
- `examples/pipeline/pilates/beam/skimsEmissions_sample.parquet`
