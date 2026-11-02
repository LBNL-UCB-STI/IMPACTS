# Installation and representative replays

This repository contains code, not the local research data warehouse. The
tested environment is recorded in `validation_environment.json`. Replays use
read-only reference inputs and a **new output directory**; never point a
replay output at an existing reference run.

## Install and test

Python 3.12 was used for this review. Create a separate environment for normal
use; the audit reused installed third-party dependencies but explicitly loaded
this checkout's `src` before the original editable installation.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[testing]"
.venv/Scripts/python -m pytest -o addopts='' -q
```

Collaborator notebook parity tests require both trusted notebooks in an
external directory. Without them those tests explicitly skip. To exercise them:

```powershell
$env:IMPACTS_CHARGING_NOTEBOOKS = 'D:/research-inputs/chargingcode'
.venv/Scripts/python -m pytest -o addopts='' -q
```

`load_notebook_functions` executes selected definitions. Supply only trusted
notebooks. The author-supplied originals are deliberately not redistributed.
All other synthetic module tests run without the research event/matrix files.

## Inputs needed

| Workflow | Required external inputs |
|---|---|
| Native example pipeline | Matching BEAM network, vehicle metadata, population/household assets, emission skims or events plus EMFAC factors, geographical grids/intersections, ISRM Zarr, ASRV patterns and NOx/NO2 response data specified by settings/manifests |
| Skim replay | Saved `preprocess_manifest.yaml` with still-available original unannualized skims, metadata/intersections and inventories; original annual outputs for comparison |
| Concentration replay | Saved preprocess/pipeline manifests, annual InMAP and AERMOD emissions, full receptor grids, compact/full matrix and ASRV source-response library |
| LDV Home80 replay | Saved `existing_ev_trips.parquet`, audited EV stops/batteries/home activity; reference Home80 BEV/PHEV vehicles/sessions/trips and hourly CSV |
| Freight replay | Three `charging_events_*.csv` collaborator files plus reference freight results |
| Hourly power | 8,760-row demand parquet, `beam_charging_summary.json`, BREATHE `scripts/allocate_power_plants_approx.py`, Cambium22 LowRECost p9 2040 hourly file, eGRID2023 and supplementary PM/NH3/VOC workbooks, ReEDS BA shapefile, source-mapped ISRM and Bay Area counties/grid |
| Human exposure | Population grid parquet, final concentration parquet, CES tracts, school/LTC geodatabase layers |

Supply the actual archived manifests; filenames alone do not prove that
networks, rates, sample fractions or matrix ordering match. Compact matrix
source IDs must refer to global ISRM IDs. Grid geometry and CRS are verified
in the concentration comparisons.

## Native traffic use

The upstream commands remain available (`python -m impacts --help`). Use a
copy of `examples/pilates/settings.yaml` with your paths. For the current
hybrid method set:

```yaml
impacts:
  exposure:
    primary_pm25_integration_strategy: additive_inmap_aermod
  power_generation:
    enabled: false
```

This is a settings fragment to merge into the full configuration, not a
complete input file. The inherited example defaults to the legacy imputation
strategy. No sample fraction is inferred from the name of an events file.
See `events_emfac.md` for the events fallback contract.

## Traffic regression commands

Run these from this repository root. Replace the paths with your local assets:

```powershell
python scripts/replay_skims_emissions.py --reference D:/reference/original-baseline --output D:/review/emissions
python scripts/replay_reference_case.py --reference D:/reference/new_3000m_circle --output D:/review/circle3000 --policy circle3000 --isrm D:/matrices/sfbay.zarr
python scripts/replay_reference_case.py --reference D:/reference/BF-06EP/Total --output D:/review/bf06-legacy --policy legacy4500 --isrm D:/matrices/sfbay.zarr
```

The first command regenerates emission preparation, annualization and grid
allocation. The next two recompute InMAP, AERMOD and integration from the
saved annual inputs, then compare receptor IDs, geometry and every numeric
field. They preserve the original partial-matrix source selections. There is
no extra 10x expansion on already annualized emissions.

Each stage runs in a separate process. `--stage inmap|aermod|merge|compare`
supports troubleshooting without rerunning successful stages. The audit used
the same cropped AERMOD convolution helper as the reference scenario; it
limits temporary arrays but retains the response and support calculation.
Concentration replays write Parquet (not duplicate large AERMOD GPKGs).

## Charging and power regression

```powershell
python scripts/replay_charging_power.py --outputs-root D:/reference/impacts_output --freight-input D:/inputs/ElectricTruck_event --breathe-root D:/external/PowerGeneration --isrm D:/matrices/sfbay.zarr --output D:/review/charging-power
```

This driver knows the archived case names listed in `RELEASE_REVIEW.md` and
compares seed-42 Home80 vehicle/session/trip results, three reconstructed
freight curves, hourly and annual dispatch, four Layer-0 power surfaces and
three LDV+freight sums. A new study should use the modules with its own explicit
inputs rather than pretending these dated reference case names are generic.

For new compatible LDV charging, prepare demand using
`scripts/run_ldv_charging.py`; see [CRISTIAN_COMPATIBILITY.md](CRISTIAN_COMPATIBILITY.md).
The Home80 replay above remains a historical sensitivity reproduction.

To run a new hourly power case after preparing demand:

```powershell
python scripts/run_hourly_power.py --demand-dir D:/inputs/charging-demand --baseline-dir D:/inputs/spatial-baseline --breathe-root D:/external/PowerGeneration --isrm-store D:/matrices/sfbay.zarr --source-layer-index 0 --output-dir D:/results/new-power
```

The layer argument is mandatory. Use Layer 1 only with verified elevated-layer
matrix coverage; Layer 0 is a ground-release approximation. The current
allocator remains fixed to the recorded Cambium 2040/eGRID input set; switching
scientific scenarios requires reviewing those settings, not renaming outputs.
The hourly parquet has `hour_of_year=0..8759` and `charging_demand_MWh` (energy
per one-hour bin, numerically equal to average MW). Fleet expansion and charger
efficiency must already be reflected. EVI data are not used in dispatch.

The native YAML pipeline uses the same implementation (enable the InMAP
workflow stage as well). Paths resolve relative to the settings file:

```yaml
impacts:
  power_generation:
    enabled: true
    hourly_demand_file: inputs/charging/sfbay_hourly_charging_demand.parquet
    charging_summary_file: inputs/charging/beam_charging_summary.json
    breathe_root: external/PowerGeneration
    county_boundaries_file: inputs/spatial/county_boundaries.gpkg
    source_layer_index: 0
    allow_incomplete_charging_scenario: false
```

This is a configuration fragment; retain the rest of the normal IMPACTS
settings, including the receptor grid and ISRM store. Choose a new output
directory for a new case. `plant_markers_file` is an optional plotting-only
override; otherwise maps use the plants selected by the current power run.
The former WA/annual-scaling configuration is no longer supported, including
`charging_events_file`, `precomputed_power_isrm_file`, and `plant_emissions_file`.
Prepare an hourly curve explicitly instead; no legacy calculation is selected
automatically. See `HOURLY_POWER_UPDATE.md` for the migration and validation.

## Exposure, figures and web comparison

```powershell
python scripts/replay_human_exposure.py --code-root D:/original/IMPACTS --population D:/inputs/population.parquet --concentration D:/review/circle3000/exposure/beam_concentration_distribution.parquet --ces D:/inputs/ces.parquet --facilities D:/inputs/facilities.gdb --output D:/review/exposure-reference
python scripts/replay_human_exposure.py --population D:/inputs/population.parquet --concentration D:/review/circle3000/exposure/beam_concentration_distribution.parquet --ces D:/inputs/ces.parquet --facilities D:/inputs/facilities.gdb --output D:/review/exposure-clean --compare D:/review/exposure-reference
```

These commands intentionally compare both code versions on the same freshly
recomputed concentration field. The first is a new reference execution, not
an attempt to overwrite an old published exposure figure. CSV/Parquet values,
PNG pixels and generated HTML are compared. School/LTC layer names match the
archived Bay Area assets in this driver.

## Archive and maintenance boundaries

`src/impacts` is the reusable library; `scripts/replay_*.py` are explicit audit
entry points; the remaining dated scripts are research history/diagnostics.
Many historical scripts still expect the documented dataset layout and case
names under the repository root. Portable external roots for those scripts:

- `IMPACTS_ISRM_PATH`: compact/full ISRM Zarr
- `IMPACTS_BREATHE_ROOT`: external PowerGeneration project
- `IMPACTS_BEAM_DATA_ROOT`: external BEAM metadata repository
- `IMPACTS_CHARGING_REFERENCE`: collaborator reference document

Do not run every historical sensitivity script as a production pipeline.
Their assumptions intentionally differ. Source-only upload excludes raw
vehicle IDs, collaborator data, spatial exposure outputs, `.venv`, caches and
matrices. Generated results and review logs remain in a separate local audit
directory. The ZIP contains no `.git` history; the local clean checkout retains
upstream history for review with `git diff`.
