# PILATES Integration

This document describes the `impacts` integration pattern for PILATES and the example scenario under `examples/pilates/`.

## Example layout

```text
examples/pilates/
  settings.yaml
  hpc-settings.yaml
  impacts/        # created at runtime, ignored by git
```

Meaning:

- `settings.yaml`: local example settings
- `hpc-settings.yaml`: HPC example settings
- `impacts`: runtime-managed IMPACTS inputs, outputs, manifests, and logs

BEAM inputs and outputs are user-managed runtime data and are ignored by git. The local
example points `beam.local_input_folder` at an external BEAM data root and writes
relative BEAM outputs under `beam/beam_output` when needed.

## Settings model

The example uses one user-managed settings file:

- `examples/pilates/settings.yaml`
- `src/impacts/config/settings.yaml` is the canonical native IMPACTS settings template.

During preprocess, `impacts` registers the user-provided settings file and source inputs in `preprocess_manifest.yaml`, then writes managed artifacts under the configured `impacts.local_input_folder` and `impacts.local_output_folder`.
Relative `impacts.*_folder` values are resolved against the settings file directory first. For `examples/pilates/settings.yaml`, `impacts/impacts_output` resolves to `examples/pilates/impacts/impacts_output`.

## Example settings shape

```yaml
run:
  region: sfbay
  scenario: base
  start_year: 2018

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
  local_input_folder: ~/Workspace/Models/beam/beam-data
  local_output_folder: beam/beam_output
  router_directory: r5/sfbay-cbg5500-weakConn-network

impacts:
  local_input_folder: impacts/impacts_inputs
  local_output_folder: impacts/impacts_output
  seed: 0
  scenario: 2018-Baseline
  pipeline:
    presim:
      activities: true
      fleet: true
    postsim:
      emissions: true
      inmap: true
      aermod: true
      exposure: true
  population:
    passenger_folder: urbansim/atlas-2019
    freight_folder: freight/20250730/2018-Baseline
    vehicle_folder: vehicle-tech
    atlas_year: 2019
    frism_year: 2018
    population_sample: 0.1
    transit_sample: 1.0
  emissions:
    include_non_osm_car_links: true
    include_passenger: true
    include_freight: true
    default_annualization_days:
      light_duty: 327.0
      medium_heavy_duty: 312.0
    pollutants: [NH3, NOx, PM25, SOx, ROG, BC]
  dispersions:
    inmap:
      isrm_zarr: ~/Workspace/Simulation/sfbay/inmap/isrm_v1.2.1.zarr
      isrm_nox_to_no2_ratios_file: vehicle-tech/dispersions/inmap/nox_to_no2_full_isrm_matrix.npz
      grid_path: vehicle-tech/dispersions/inmap/isrm_polygon_wgs84.gpkg
      grid_id: isrm
      grid_epsg: 4326
    aermod:
      grid_size_meters: 100.0
      asrv_patterns_file: vehicle-tech/dispersions/aermod/aermod_patterns_wgs84.parquet
      asrv_patterns_epsg: 4326
```

## Running the example

From the repo root:

```bash
python -m impacts pipeline --config examples/pilates/settings.yaml
```

The full pipeline command produces:

- `examples/pilates/impacts/impacts_output/log_*.log` for HPC runs launched through `hpc/job_runner.sh`
- `examples/pilates/impacts/impacts_output/preprocess_manifest.yaml`
- `examples/pilates/impacts/impacts_output/pipeline_manifest.yaml`
- `examples/pilates/impacts/impacts_output/postprocess_manifest.yaml`
- `examples/pilates/impacts/impacts_output/beam_emissions_for_inmap.parquet`
- optionally `examples/pilates/impacts/impacts_output/beam_emissions_for_aermod.parquet`
- `examples/pilates/impacts/impacts_output/beam_concentration_distribution.parquet`

HPC stage runs use the same settings file and stage names:

```bash
./hpc/job_runner.sh \
  -c examples/pilates/hpc-settings.yaml \
  -a my_account \
  -s postsim \
  -p lr7
```

Supported stages are `pipeline`, `preprocess`, `presim`, `activities`, `fleet`, `postsim`, `emissions`, `inmap`, `aermod`, `exposure`, `postprocess`, and `analysis`.

## Preparing smaller sample inputs

If your BEAM artifacts are too large, create smaller samples first.

Events are sampled by vehicle id, so each selected vehicle keeps its full trace. Skims are sampled by row fraction.

```bash
python -m impacts.tools.prepare_sample_data events \
  --input /path/to/0.events.parquet \
  --fraction 0.05
```

```bash
python -m impacts.tools.prepare_sample_data skims \
  --input /path/to/0.skimsEmissions.parquet \
  --fraction 0.05
```

Outputs land in:

- `examples/pilates/beam/events_sample.parquet`
- `examples/pilates/beam/skimsEmissions_sample.parquet`

The `examples/pilates/beam/**` tree is ignored by git because it contains user-managed BEAM runtime data.
