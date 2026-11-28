# PILATES Integration

This document describes the `impacts` integration pattern for PILATES and the example scenario under `examples/pilates/`.

## Container use

The image installs `derive_settings_from_pilates`, `presim`, and `postsim` on `PATH`. It has no command entrypoint, so PILATES can use each executable as its container command from any working directory:

Releases tagged `vMAJOR.MINOR.PATCH` publish `ghcr.io/lbnl-ucb-sti/impacts:<tag>` after the image smoke tests pass. The release workflow also checks an anonymous GHCR pull, which is required for HPC environments that do not authenticate to GHCR. No `latest` tag is published.

```bash
derive_settings_from_pilates --pilates-settings /app/workspace/bridge.yaml --output /app/workspace/settings.yaml
presim --config /app/workspace/settings.yaml
postsim --config /app/workspace/settings.yaml
```

## Run order and fleet handoff

PILATES owns the cross-model order. Its single IMPACTS-enabled choice runs `presim` before the first BEAM run, makes the generated fleet available to BEAM through its normal artifact handoff, then runs `postsim` after the final BEAM run. PILATES does not need separate switches for presim activities or fleet. IMPACTS does not launch BEAM.

`presim` ensures EMFAC activities first, writes `activities_manifest.yaml`, then runs fleet generation from that manifest. The manifest records the passenger and freight activity, fleet, and rate output paths. PILATES should register the actual generated files with Consist and hand the matching fleet files to BEAM. The manifest is IMPACTS' activities-to-fleet contract; `postsim --config` does not accept a separate fleet manifest. Postsim reads BEAM inputs through the settings and mounted input paths.

Mount the persistent PILATES run workspace at `/app/workspace`, the BEAM input tree at `/app/beam_input`, and one durable shared cache directory at `/app/impacts_cache`. The bridge settings should point IMPACTS outputs and run-specific inputs into `/app/workspace`, BEAM inputs into `/app/beam_input`, and set `impacts.shared_input_cache_dir` to `/app/impacts_cache`. PILATES should mount the same host cache for presim and postsim; by default its host location is `<run.output_directory>/.pilates-cache/impacts`, outside timestamped run folders. IMPACTS stores ISRM copies under `isrm/` and extracted EMFAC raw archives under `emfac_raw/<archive-key>/` within that shared directory.

For offline runs, set `impacts.analysis.basemap: none`. IMPACTS still writes the plots with concentration layers, road overlays, and scale bars, without requesting Carto tiles.

## OSM input selection

PILATES should pass the exact OSM PBF selected for the same BEAM run and iteration as an explicit IMPACTS input. Set `impacts.emissions.osm_network_file` to its path inside the mounted workspace and, when available, set `impacts.emissions.osm_network_artifact_id` to the Consist artifact ID. IMPACTS reads that path directly and records both values in `preprocess_manifest.yaml`; it does not query Consist to select an OSM artifact.

The same rule applies to the other upstream data IMPACTS consumes. PILATES points `beam.local_output_folder` at the current step's materialized BEAM output directory, which contains that step's `network.csv.gz`, emissions skims, and optional events file. IMPACTS resolves those files from that configured tree. When running standalone, point `beam.local_output_folder` at one intended BEAM run; IMPACTS errors if it finds multiple candidate networks or multiple `ITERS` trees instead of guessing which run to use.

Exposure residents are separate from BEAM's plan-bearing population. When exposure is enabled, PILATES passes the raw UrbanSim HDF5 it already selected for the run. IMPACTS reads the exact-year `persons`, `households`, and `blocks` tables and assigns residents to its grid. PILATES does not transform or export population tables. `impacts.population.passenger_folder` remains for fleet generation and is never an exposure fallback. The generated bridge points `impacts.exposure.population_h5` at the mounted HDF5.

The postsim manifest records the source HDF5 artifact and population-assignment summary. The summary reports supplied residents, residents assigned to the grid, residents missing home coordinates, and residents outside the grid. IMPACTS does not query Consist or use BEAM's plan-bearing population files as a substitute.

Its presim creates EMFAC activity and fleet tables from the configured EMFAC archive and associated static inputs. IMPACTS also reads configured passenger/freight vehicle type tables, dispersion grids, and ISRM data. Those paths remain explicit settings resolved against the declared BEAM input roots; they are not replaced with a Consist artifact from another run.

Standalone runs can set `impacts.emissions.osm_network_file` to a specific PBF. If omitted, IMPACTS uses the already-declared `beam.router_directory` and selects the first top-level `.pbf` by filename, matching PILATES's IMPACTS-enabled R5 selection. It logs and ignores `.vex` files; if there is no PBF, IMPACTS errors. The `beam.router_directory` setting is not copied into a hidden OSM setting.

```yaml
impacts:
  emissions:
    osm_network_file: /app/workspace/beam/output/sfbay/year-2018-iteration-0/.pilates-consist-outputs/beam_run/beam_r5_osm_file_2018_0.osm.pbf
    osm_network_artifact_id: <consist-artifact-id>
```

When InMAP is enabled, `impacts.dispersions.inmap.isrm_zarr` may point to a local Zarr store or a public `s3://`, `http://`, or `https://` Zarr store; IMPACTS copies either source into the shared cache. Remote URLs must expose a recursively listable Zarr store; a web page or a link to an archive is not a Zarr store URL. ISRM cache entries are keyed by source URL or local path. EMFAC raw cache entries are keyed by archive path, size, and modification time. To refresh data changed in place, remove the corresponding cache entry. The image does not contain ISRM or other scenario data. A local ISRM source mount is needed only when populating its cache for the first time. Omit the InMAP source mount if that stage is disabled or the configured ISRM store is public.

For example, with Docker:

```bash
docker run --rm \
  --mount type=bind,src=/path/to/run-workspace,dst=/app/workspace \
  --mount type=bind,src=/path/to/beam-input,dst=/app/beam_input,readonly \
  --mount type=bind,src=/path/to/shared/impacts-cache,dst=/app/impacts_cache \
  <image> derive_settings_from_pilates \
  --pilates-settings /app/workspace/bridge.yaml --output /app/workspace/settings.yaml
```

Use the same workspace, BEAM input, and shared-cache mounts with `presim --config /app/workspace/settings.yaml` and `postsim --config /app/workspace/settings.yaml`. For a local ISRM copy, add `--mount type=bind,src=/path/to/isrm_v1.2.1.zarr,dst=/app/inmap/isrm_v1.2.1.zarr,readonly` on the cache's first use and set `isrm_zarr` to that container path. The shared cache must be writable, visible to every run, and large enough for the ISRM and EMFAC raw inputs. For Apptainer/Singularity, bind the workspace, BEAM input, and shared cache as above, adding the local ISRM bind on its first cache population, and run the `docker://` image URI with the command after the image URI.

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
