# IMPACTS clean release: change and review summary

The preferred LDV charging model now directly reuses Cristian's original notebook
functions and parameters. See [CRISTIAN_COMPATIBILITY.md](CRISTIAN_COMPATIBILITY.md).
Designed Home80/periodic/PHEV fuel-fallback cases below are historical sensitivity
alternatives, not the compatible production recommendation.

The comparison baseline is `LBNL-UCB-STI/IMPACTS` commit
`159a5598210905471a0412609d8414ccf97f355f`. The original working checkout had
the same HEAD; the additions were primarily uncommitted changes and research
scripts. The clean release was prepared separately, preserving the original.

## Main additions

| Capability | Main code locations |
|---|---|
| BEAM events to EMFAC emissions by vehicle, road, and process | `src/impacts/pipeline/workflow/prepare_emissions/event_emfac.py`; `from_events.py` |
| 3 km ISRM local-to-local removal and full InMAP-only references | `workflow/isrm_near_source.py`; `step2_compute_inmap_concentrations.py` |
| Circular 3 km AERMOD cutoff and compatible-class site matching | `workflow/step3_compute_aermod_concentrations.py` |
| Traffic InMAP + AERMOD + separate power increments | `workflow/step4_prepare_exposure.py` |
| Cristian charging-code adaptation and original-function parity tests | `workflow/charging/ldv_charging.py` |
| Home80, periodic SOC, and PHEV fuel fallback models | `workflow/charging/ldv_home_sensitivity.py` |
| Collaborator freight charging, hourly allocation, annual plant emissions, and ISRM | `scripts/analyze_collaborator_truck_charging.py`; `workflow/hourly_power.py`; `workflow/power_generation.py` |
| Optional passenger/freight/port/gateway products and future cases | `scripts/run_*products.py`; `run_future_fixed_factor_scenarios.py` |
| CES, school, and LTC exposure; original-style figures and web maps | `src/impacts/pipeline/postprocess/step8_human_exposure.py`; `step9_human_exposure_webmap.py` |

Here, `workflow/` means `src/impacts/pipeline/workflow/`. The underlying
EMFAC, ISRM, and AERMOD workflows already existed. Events adaptation,
near-source integration changes, charging/power modules, and expanded products
are local additions or modifications.

## Corrections made during cleanup

1. Corrected explicit `inmap_only` integration that used local-removed primary
   concentrations and still added AERMOD to BC/NO2; added a dedicated regression.
2. Restored the finite InMAP fallback in the retained AERMOD budget-normalization
   branch when the response budget is zero, preventing NaNs.
3. Initialized power-marker paths and delta-baseline variables in conditional
   postprocessing branches.
4. Replaced machine-specific paths with configuration/external-root arguments;
   packaged YAML and HTML assets; corrected cross-platform tests; made Memray optional.
5. Added isolated replay drivers that create fresh output directories and record
   the actual imported clean-code location.

## Required distinctions

- The current hybrid default is `additive_inmap_aermod` in Python and shipped
  YAML configurations. Explicit legacy settings remain supported for reproduction.
- Only traffic primary PM2.5, BC, and NO2 undergo near-source removal. Secondary
  PM2.5 and power responses do not. The full InMAP-only reference retains all responses.
- Power uses a single hourly algorithm shared by YAML and
  `scripts/run_hourly_power.py` through `workflow/hourly_power.py`. Obsolete
  configuration is rejected. See `HOURLY_POWER_UPDATE.md`.
- Real-data power replays retain the approved Layer 0 approximation and must not
  be described as elevated Layer 1 results.
- The reference LDV case uses existing EVs from Cristian's `3.events`; freight
  uses three collaborator charging files. Both curves already include 10x;
  combination must not apply another expansion.
- Previously authorized missing-source exclusions remain. Reproducibility does
  not establish complete matrix coverage or observational validity.
- A single baseline can use the native postsim pipeline without port/gateway
  classification; those products are optional attribution analyses.

## Delivery and validation

Start with [FULL_VERSION_REVIEW.md](FULL_VERSION_REVIEW.md) and the
[file-level index](CHANGE_INDEX.md). Detailed code mapping is also in
[RELEASE_REVIEW.md](RELEASE_REVIEW.md). Commands and external-input requirements
are in [REPRODUCING.md](REPRODUCING.md). Test counts, differences, and scope are
recorded in [VALIDATION.md](VALIDATION.md) and [HOURLY_POWER_UPDATE.md](HOURLY_POWER_UPDATE.md).
The latest full suite passed 402 tests. Complete file hashes are in `source_inventory.json`.

The source archive excludes large datasets, matrices, raw events, collaborator
notebooks, external BREATHE data/allocator, generated research results, and
virtual environments. Execution still requires those external inputs.
Upstream `LICENSE.txt` contains `TBD`, inconsistent with MIT package metadata;
resolve this before public distribution. Copyright or licensing statements
were not reassigned during this review.
