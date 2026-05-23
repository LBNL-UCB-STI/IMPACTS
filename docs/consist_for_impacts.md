# Consist For Impacts

`impacts` now treats source-input staging as artifact registration, not file copying. Preprocess records source inputs in the manifest and uses Consist when available. Generated intermediates are still written under the workspace.

## Mental model

Consist staging:

- find an upstream file
- log it as a Consist input artifact
- or, if Consist is unavailable, store the original source path without copying
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

## PILATES-aligned lookup order

For upstream BEAM-origin files, `impacts` now tries PILATES-style artifact keys first and only falls back to local filesystem discovery when nothing is published yet.

- `network`: query latest `beam_network_final_*`, otherwise discover the BEAM network locally and register it
- `events_input`: query latest `events_parquet_*`, otherwise discover the latest local BEAM events file and register it
- `osm_network`: query canonical `beam_r5_osm_file`, otherwise resolve the local `.osm.pbf` under the configured BEAM network input folder and register it
- `population_inputs["persons"]`: query latest `beam_population_final_*`, otherwise discover local `persons.*` and register it
- `population_inputs["households"]`: query latest `beam_households_final_*`, otherwise discover local `households.*` and register it

Other inputs that do not already have a known PILATES handoff key are still registered in Consist under their maintained `impacts` manifest key.

## Recommended integration shape

1. Preprocess or workflow code resolves upstream files as it does today.
2. Source inputs are registered through `register_local_input(...)` or `register_managed_input(...)`, which do not physically copy the source file.
3. `manifest_inputs[key]` stores a Consist-backed entry when Consist works, otherwise it stores a no-copy local entry that still points at the original source path.
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
- direct-path registration behavior when Consist is unavailable

## Where to start

Current state:

- source inputs are registered instead of copied
- workflow skims/event/network resolution can follow manifest entries instead of re-discovering copied inputs
- generated intermediates are still written under the workspace and referenced by path

The important rule for `impacts` is:

“Use Consist to register and find files by logical identity. Resolve to filesystem paths only at the edge where the code actually reads or writes the file.”
