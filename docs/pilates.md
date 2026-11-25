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

Mount the persistent PILATES run workspace at `/app/workspace`, the BEAM input tree at `/app/beam_input`, and one durable shared cache directory at `/app/impacts_cache`. The bridge settings should point IMPACTS outputs and run-specific inputs into `/app/workspace`, BEAM inputs into `/app/beam_input`, and set `impacts.shared_input_cache_dir` to `/app/impacts_cache`. PILATES should mount the same host cache for presim and postsim; by default its host location is `<run.output_directory>/.pilates-cache/impacts`, outside timestamped run folders. IMPACTS stores ISRM copies under `isrm/` and extracted EMFAC raw archives under `emfac_raw/<archive-key>/` within that shared directory.

Set `impacts.analysis.basemap` to `carto` (the default) to add Carto tiles, or `none` for offline plotting. With `none`, IMPACTS still writes the plots with concentration layers, road overlays, and scale bars. The former `skip_web_tiles` boolean remains accepted as a compatibility alias.

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

## Generated fleet handoff

After fleet generation, presim writes `fleet_manifest.yaml` beside
`activities_manifest.yaml`. Use the passenger vehicle types and assignments,
freight vehicle types and carriers, and referenced emissions rates from that run
when compiling the BEAM launch. For postsim, set `impacts.presim_fleet_manifest`
to the same manifest's container path. Postsim then uses its matching rates,
types, assignments, and EMFAC inventories and checks the EMFAC year and scenario.

If the original ATLAS vehicle table is stored under a different name from
`vehicles.parquet`, set `impacts.fleet.atlas.vehicles_file` to its path
relative to the BEAM input root, or to an absolute mounted path.

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
