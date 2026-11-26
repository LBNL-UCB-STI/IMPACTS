# Consist For Impacts

`impacts` records source inputs in its manifest; generated intermediates are written under the workspace. When running through PILATES, IMPACTS consumes the materialized BEAM files and raw UrbanSim HDF5 selected for that workflow step. It does not query Consist for a newer or latest upstream artifact. Standalone runs resolve files only under the configured BEAM output, R5, and population input paths.

## Mental model

Consist staging:

- find an upstream file
- log it as a Consist input artifact
- or, when Consist is unavailable or no active Consist run is present, store the original source path without copying
- store the tracked entry under a stable key in `manifest_inputs`
- resolve the artifact to a path only when a step needs to read it

## Stable keys

The current manifest keys are already close to what Consist needs. These should become the primary artifact keys:

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

- `network`: resolve the selected `network.csv.gz` or `network.parquet` under `beam.local_output_folder`; duplicate candidates are an error
- `emissions_skims_input`: select the highest numbered iteration's emissions skims from the single `ITERS` tree under `beam.local_output_folder`; multiple `ITERS` trees are an error
- `events_input`: optionally select the highest numbered iteration's local BEAM events file from that same `ITERS` tree
- `osm_network`: read the explicit PILATES-supplied PBF and record its path and artifact ID; standalone uses the first top-level `.pbf` in `beam.router_directory` by filename, logging and ignoring `.vex` files
- `exposure_population_h5`: the raw UrbanSim HDF5 selected by PILATES; IMPACTS reads the exact-year persons, households, and blocks tables itself. `impacts.population.passenger_folder` remains for fleet generation and is not an exposure fallback

Other configured inputs—such as passenger and freight vehicle type tables, EMFAC rates and inventories, dispersion grids, and ISRM data—are resolved from their declared IMPACTS settings and recorded in the manifest. If vehicle type files are discovered from the configured vehicle folder, there must be exactly one match for each model. IMPACTS builds its EMFAC activity and fleet tables from the configured EMFAC source data. Exposure reads residents, households, and home coordinates from the raw year-scoped UrbanSim HDF5, including residents with no BEAM plans. IMPACTS does not query Consist for substitute BEAM or UrbanSim outputs.

## Recommended integration shape

1. Preprocess or workflow code resolves upstream files as it does today.
2. Source inputs are registered through `register_local_input(...)` or `register_managed_input(...)`, which do not physically copy the source file.
3. `manifest_inputs[key]` stores a Consist-backed entry when there is an active Consist run, otherwise it stores a no-copy local entry that still points at the original source path.
4. Workflow steps resolve the manifest entry to a concrete path right before `read_table(...)`, `read_vector(...)`, or other file IO.
5. Generated outputs can be logged with `log_output_reference(...)` and stored in the run manifest by stable key.

## Minimal API sketch

`impacts` should add a small local wrapper rather than importing Consist calls all over the codebase:

```python
def log_input(*, key: str, source_path: str, optional: bool = False) -> dict[str, Any]:
    ...

def log_output(*, key: str, path: str, optional: bool = False) -> dict[str, Any]:
    ...

def resolve_logged_path(entry: dict[str, Any]) -> str:
    ...
```

That wrapper should hide:

- Consist import details
- artifact metadata shape
- path normalization
- direct-path registration behavior for standalone runs without an active Consist run

## Where to start

Current state:

- source inputs are registered instead of copied
- workflow skims/event/network resolution can follow manifest entries instead of re-discovering copied inputs
- generated intermediates are still written under the workspace and referenced by path

The important rule for `impacts` is:

“Use Consist to register and find files by logical identity. Resolve to filesystem paths only at the edge where the code actually reads or writes the file.”
