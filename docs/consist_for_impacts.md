# Consist For Impacts

`impacts` records source inputs in its manifest; generated intermediates are written under the workspace. When running through PILATES, IMPACTS consumes the materialized BEAM files and raw UrbanSim HDF5 selected for that workflow step. It does not query Consist for a newer or latest upstream artifact. Standalone runs resolve files only under the configured BEAM output, R5, and population input paths.

## Mental model

Consist staging:

- find an upstream file
- log it as a Consist input artifact
- or, when Consist is unavailable or no active Consist run is present, store the original source path without copying
- store the tracked entry under a stable key in `manifest_inputs`
- resolve the artifact to a path only when a step needs to read it

## Manifest keys

The preprocessing manifest uses stable keys for the inputs and generated intermediates it tracks. Common input keys include:

- `settings`
- `network`
- `emissions_skims_input`
- `events_input`
- `osm_network`
- `rates_folder`
- `inmap_grid`
- `isrm_nox_to_no2_ratios_file`
- `asrv_patterns_file`
- `isrm`

## Upstream model input selection

PILATES materializes the selected files for the current step in the workspace and points the generated IMPACTS settings at those locations. Outside PILATES, IMPACTS uses the configured folders and the same deterministic filename selection rules for inputs that support local discovery.

- `network`: prefer a root-level `network.csv.gz` or `network.parquet`; otherwise resolve a unique matching file under `beam.local_output_folder`. Multiple recursive matches are an error
- `emissions_skims_input`: select the highest numbered iteration's emissions skims from the single `ITERS` tree under `beam.local_output_folder`; multiple `ITERS` trees are an error
- `events_input`: optionally select the highest numbered iteration's local BEAM events file from that same `ITERS` tree
- `osm_network`: read the explicit PILATES-supplied PBF and record its path and artifact ID; standalone uses the first top-level `.pbf` in `beam.router_directory` by filename, logging and ignoring `.vex` files
- `exposure_population_h5`: the raw UrbanSim HDF5 selected by PILATES; IMPACTS reads the exact-year persons, households, and blocks tables itself. `impacts.population.passenger_folder` remains for fleet generation and is not an exposure fallback

Other configured inputs—such as passenger and freight vehicle type tables, EMFAC rates and inventories, dispersion grids, and ISRM data—are resolved from their declared IMPACTS settings and recorded in the manifest. If vehicle type files are discovered from the configured vehicle folder, there must be exactly one match for each model. IMPACTS builds its EMFAC activity and fleet tables from the configured EMFAC source data. Exposure reads residents, households, and home coordinates from the raw year-scoped UrbanSim HDF5, including residents with no BEAM plans. IMPACTS does not query Consist for substitute BEAM or UrbanSim outputs.

## Current implementation

`preprocessing.step1_collect_inputs` resolves the configured source files and registers them through `register_managed_input(...)` or `register_local_input(...)`. A Consist-backed reference is recorded when a run is active and the dependency is installed; otherwise the manifest points directly to the local source. These registrations do not copy the source file.

Steps resolve manifest entries through `resolve_required_manifest_input(...)` immediately before file IO. Generated preprocessing outputs are written under the configured workspace and registered by path. Consist calls and entry serialization are centralized in `consist_artifacts.py`.

Presim creates EMFAC activities and fleet outputs before BEAM. PILATES owns registering and handing those files to BEAM. Postsim consumes the current BEAM outputs through the configured, mounted paths; IMPACTS does not query Consist to choose a newer run's artifacts.

The governing rule for `impacts` is:

“Use Consist to register the exact files selected upstream. Resolve their paths only where IMPACTS reads or writes them.”
