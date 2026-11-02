# IMPACTS: complete review of the current version

Module paths have been consolidated; see [Charging module organization](CHARGING_MODULES.md).

Review date: November 2, 2026. This document describes the implemented functionality, calculation conventions, validation evidence, and items requiring review.

**Charging model update:** the preferred LDV pathway has been restored to the original Cristian notebook functions and settings. See [CRISTIAN_COMPATIBILITY.md](CRISTIAN_COMPATIBILITY.md). The designed Home80/periodic/electric-first cases described below are historical sensitivity cases and their reproduction evidence, not the recommended compatible model. The power module still consumes an explicitly supplied curve; saved concentrations were not replaced.

Comparison baseline: `LBNL-UCB-STI/IMPACTS`, commit `159a5598210905471a0412609d8414ccf97f355f`. This is the upstream reference recorded in the preceding review, not a claim that GitHub was queried again for its latest commit.

Review target: the separate `IMPACTS-clean-20261101` checkout, distributed in the November 2 source archive. The checkout name retains its creation date and does not indicate whether its contents have been updated. The original research checkout, inputs, and saved results were not overwritten.

**Conclusion: representative replays and module tests passed, and the code is ready for detailed review. Passing tests does not establish observational validation of every scientific assumption or execution of every scenario through every entry point.** Start with Sections 2, 6, 7, 10, and 12, then inspect the corresponding code.

## 1. Scope relative to the original project

The original project already provided EMFAC processing, emissions preparation, InMAP, AERMOD, concentration integration, and postprocessing. The extensions build on that foundation:

| Area | Main addition or modification | Effect on scientific calculations |
|---|---|---|
| Emissions inputs | Events-only activity reconstruction and EMFAC matching; emissions skims remain supported | Adds an input pathway |
| Vehicle and scenario products | Passenger, freight, port/non-port, and gateway attribution with separate outputs | Changes the source subset represented |
| Traffic ISRM | Global-ID alignment for compact matrices, 3 km area-overlap removal, full InMAP-only reference | Changes response selection |
| AERMOD | Circular 3 km cutoff, compatible-class fallback for missing site responses, updated cache identity | Changes response selection |
| Concentration integration | Direct addition of nonlocal InMAP and local AERMOD, followed by independent power PM2.5 | Changes the integration method |
| LDV charging | Notebook adaptation and parity tests; additional Home80, periodic SOC, and PHEV fuel fallback models | Includes both adaptation and substantive model changes |
| Freight charging | Collaborator sessions converted to hourly loads, separate scenarios, and 10x expansion | Adds electricity-demand inputs |
| Power generation | Shared hourly demand, technology/plant allocation, annual emissions, and ISRM workflow | Adds power-sector calculations |
| Exposure and presentation | CES, schools, LTC, tables, component maps, web maps, and differences | Adds analyses; display settings alone do not change concentrations |
| Software maintenance | Configuration, paths, packaging, tests, replay entry points, and documentation | Generally preserves scientific results; corrections are listed separately |

The [file-level change index](CHANGE_INDEX.md) links every added or modified file. File fingerprints are recorded in `source_inventory.json`.

## 2. Calculation chain and three essential boundaries

```text
Traffic: events + vehicle/network/rate inputs, or emissions skims
       -> classified emissions -> sample expansion and annualization -> ISRM / AERMOD
       -> integrated traffic concentrations

LDV EV trips -> charging-location/power selection -> SOC simulation -> hourly grid demand
Collaborator freight charging sessions -> hourly grid demand
       -> hourly power allocation -> annual plant emissions -> power ISRM PM2.5

Integrated traffic concentrations + power PM2.5
       -> final concentrations -> exposure / tables / maps / web
```

1. **Identify the inputs to each source pathway separately.** The validated LDV Home80 case uses existing EVs from Cristian's `3.events`; the three freight power cases use collaborator charging files; traffic replays use the original skims and audited BF-06EP emissions. These replays must not collectively be described as originating from the new `0.events`.
2. **Apply 10x expansion once.** The representative cases assume a 10% sample. The software cannot infer that sampling fraction from an arbitrary events file. Already expanded hourly demand or concentrations must not be multiplied by 10 again during combination.
3. **Adding electric-freight power emissions does not automatically replace diesel road emissions.** Adding collaborator loads estimates the additional electricity contribution. A net electrification assessment also requires consistent changes to on-road emissions for the same vehicles and scenario boundaries.

## 3. Events, EMFAC, and annual emissions

Main code:

- `src/impacts/pipeline/workflow/prepare_emissions/from_events.py`
- `src/impacts/pipeline/workflow/prepare_emissions/event_emfac.py`
- `src/impacts/pipeline/workflow/prepare_emissions/from_skims.py`
- `src/impacts/pipeline/workflow/prepare_emissions/annualization.py`
- `src/impacts/pipeline/emfac/_common.py`
- `src/impacts/pipeline/emfac/fleet/step3_map_emfac_atlas.py`

The events pathway reconstructs road/process activity and matches supplied vehicle mappings, network information, and EMFAC rates. A vehicle-type string alone is not sufficient to invent a factor; missing mappings or factors require explicit handling. The skims pathway continues to read existing emissions. Emissions skims must not be treated as activity and multiplied by emission factors a second time.

Annualization preserves vehicle-category operating days, sample fractions, and source attributes. The main annual emissions fields use short tons/year. Power emissions are first retained in kg/year and then converted to the units required for matrix calculations.

Current future-traffic cases use fixed emission factors. They compare activity/fleet scenarios and are not forecasts using future-year EMFAC factors. Road NO2 uses NOx emissions and a corresponding NO2 response/transfer artifact. The code distinguishes transfer artifacts that already include unit conversion from those that require it. No generic ppb-to-micrograms/m3 conversion has been added to correct arbitrary inputs automatically. Check the units and provenance before substituting a NOx-to-NO2 matrix.

Review vehicle-type mapping, fuel classes, emission processes, road allocation, sample fractions, operating days, inventory correction, and emissions associated with excluded sources.

## 4. Classification and scenario outputs

Main entry points: `scripts/run_ldv_freight_products.py`, `run_gateway_freight_products.py`, `run_future_fixed_factor_scenarios.py`, and `run_future_detailed_products.py`.

Classification tables are matched to events by BEAM vehicle ID and support Passenger, PortRelated, NonPortRelated, and gateway products. Port/gateway names come from the supplied table and should not be inferred from vehicle names or geographic proximity. The relationship between seaport/airport indicators and port-related status must follow that table.

The recorded detailed future products represent fixed-factor, 10x traffic emissions and concentrations, excluding buses and power generation. The availability of power functionality in the repository does not mean every products file contains power contributions.

Check whether the main groups sum to Total, whether gateway groups are mutually exclusive, and whether unmatched/unassigned groups remain. Do not add a subgroup to a total that already contains it.

### Running a baseline without port attribution

Port and gateway attribution are optional research analyses, not prerequisites for the native baseline pipeline. With existing BEAM outputs, the standard entry point is:

```bash
python -m impacts postsim --config baseline_settings.yaml
```

This runs configured emissions, InMAP, AERMOD, exposure, and postprocessing stages without a port-classification table. Ordinary vehicle metadata and emissions mappings are still required. Enable the hourly power module separately when charging-related power contributions are needed; its input must already contain the intended hourly charging load. Explicitly select the additive integration strategy and the correct sample fractions. This entry-point dependency was inspected; it is not a claim of an additional complete raw-events baseline replay.

## 5. Near-source treatment in ISRM and AERMOD

Main code: `isrm_near_source.py`, `step2_compute_inmap_concentrations.py`, and `step3_compute_aermod_concentrations.py`, under `src/impacts/pipeline/workflow/`.

### ISRM

Construct a circle of radius 3,000 m around the **source-grid polygon centroid**. For each source-receptor pair, zero the selected response when:

```text
area(receptor polygon intersected with source circle) / area(receptor polygon) > 0.5
```

This is not a receptor-centroid distance rule and does not zero the receptor's entire final concentration. A receptor can still receive nonlocal contributions from other sources.

Apply removal only to traffic primary PM2.5, BC, and NO2. Secondary PM2.5 and power-sector responses remain intact. Compact Zarr arrays are aligned using global source IDs; array positions must not be confused with original grid IDs. Receptors come from the complete study-area grid, not only source cells containing emissions.

### AERMOD

Retain template responses at source-to-receptor grid-center Euclidean distances of at most 3,000 m and zero responses beyond that distance. A diagonal point can therefore be removed even when each coordinate offset is less than 3 km. The extent and resolution of the pregenerated template still limit available responses.

When a matched site's template is unavailable, select an available nearby site only within the same release-height, temporal, and urban source class. Do not substitute incompatible classes. The cutoff policy is included in cache identity to prevent reuse across different policies.

**The two 3 km rules are not mathematically complementary.** ISRM uses an area threshold and AERMOD uses a distance cutoff. Boundary sensitivity remains a scientific question; equal radii alone do not prove a partition without overlap or gaps.

## 6. Integration equations, InMAP-only, and field definitions

Main code: `src/impacts/pipeline/workflow/step4_prepare_exposure.py`.

With `additive_inmap_aermod` explicitly selected:

```text
Traffic primary PM2.5 = nonlocal InMAP primary PM2.5 + local AERMOD primary PM2.5
Traffic secondary PM2.5 = full InMAP secondary PM2.5
Traffic total PM2.5 = traffic primary + traffic secondary
Final total PM2.5 = traffic total PM2.5 + power total PM2.5
BC / NO2 = nonlocal InMAP traffic component + local AERMOD traffic component
```

The additive branch does not multiply by an additional AERMOD indicator or renormalize AERMOD to an InMAP budget. Indicators and some scaling fields remain for diagnostics/compatibility; their presence does not mean the additive calculation uses budget scaling.

| Final-grid fields | Meaning in the current method |
|---|---|
| `inmap_PrimaryPM25`, `inmap_BC`, `inmap_NO2` | Traffic ISRM components used for integration, with selected near-source responses removed |
| `inmap_SecondaryPM25` | Full traffic secondary PM2.5 |
| `aermod_PrimaryPM25`, `aermod_BC`, `aermod_NO2` | Local AERMOD traffic components |
| `inmap_only_PrimaryPM25`, `inmap_only_TotalPM25`, `inmap_only_BC`, `inmap_only_NO2` | Full traffic InMAP-only baseline, without local-to-local removal |
| `power_PrimaryPM25`, `power_SecondaryPM25`, `power_TotalPM25` | Separate power contributions, without local-to-local removal |
| `PrimaryPM25`, `SecondaryPM25`, `TotalPM25` | Final results for the selected traffic strategy, including corresponding power components when supplied |
| `BC`, `NO2` | Integrated traffic results; the current power module does not supply power BC/NO2 concentrations |

Full baseline fields are named `InMAPOnly_*` at the InMAP stage and renamed `inmap_only_*` in the final grid. The explicit `inmap_only` strategy has been corrected to use the full baseline and omit AERMOD from BC/NO2 as well. If power inputs are also supplied, final PM fields still include power while the `inmap_only_*` diagnostics remain traffic-only.

**Configuration item for review:** the inherited example/default YAML still sets `primary_pm25_integration_strategy` to `impute_inmap_primary_in_aermod_domain`. The current research method requires an explicit `additive_inmap_aermod` setting, as used by the validated replay drivers. This documentation review does not change that default.

## 7. LDV charging: notebook compatibility and historical Home80 sensitivity

### Original-function adaptation

`charging/ldv_charging.py` loads trusted collaborator notebook functions, prepare parking stops, and provide performance/compatibility adaptations. Parity tests compare charging-stop and charger choices, SOC, and delivered energy. They establish agreement with the specified original functions, not that subsequent Home80 model changes leave the original algorithm unchanged.

### Historical replayed Home80 / Home80 model

Implementations: `charging/ldv_home_sensitivity.py`. Inspect the entry point and explicit arguments in `scripts/replay_charging_power.py`.

| Item | Explicit setting in the reference case |
|---|---|
| Vehicle inputs | Existing BEV/PHEV trips from Cristian's `3.events` and audited stops/batteries/activity; ICE vehicles are not all converted to EVs |
| Home80 | Approximately 80% of eligible terminal-home vehicles, stratified by powertrain, are home-preferred; this is not a fixed share of all vehicles or charging energy |
| Primary charging location | Final home stop for home-preferred vehicles; otherwise Work/Education, then another nonhome stop, then a home fallback |
| Minimum parking duration | Strictly `> 900 seconds` in the current implementation |
| AC power | 1.9 kW / 7.2 kW |
| AC type assignment | Deterministic pseudorandom assignment with 20% L1 / 80% L2 probabilities; a finite sample need not have exactly those shares |
| Traction energy | 0.202 battery-kWh/km for BEVs and the electric portion of PHEV travel |
| Charging efficiency | 90%, corresponding to approximately 0.2244 grid-kWh/km for electric travel; do not count this loss twice |
| SOC reserve | 2 kWh, not a fixed 2% |
| Initial SOC | Read from the existing stop audit, originally assigned by the collaborator framework; the current model then iterates representative days to periodic SOC convergence |
| Periodic calculation | At most 2,000 iterations; start/end daily energy tolerance of 1e-7 kWh |
| BEVs | Necessary top-ups at other stops; infeasible or nonconverged vehicles are excluded from the valid reference load and retained in diagnostics |
| PHEVs | Undo the intermediate 20% proxy and simulate full electric energy requirements chronologically; unmet electric travel is recorded as fuel-distance equivalents, not automatically converted into road fuel emissions |
| Reference PHEV policy | `primary_preference`, explicitly supplied by replay; different from the library default `home_only` |
| Reproducibility | Seed 42 and stable vehicle-ID hashing |

**Distinguish EVI settings from model settings.** Residential L1/L2 50:50 belongs to the EVI comparison curve. The reference LDV implementation still uses the 20:80 AC assignment above. Home80 does not force 80% of sessions or energy to occur at home. EVI is a scenario benchmark, not measured vehicle-level charging activity.

SOC figures show simulated reconstructions. Midnight SOC in the periodic representative day has already been affected by the preceding day's charging and must not be presented as the original beta-distribution assignment. A high full-charge fraction is related to charging toward capacity at selected stops and repeated-day operation; it requires separate observational assessment.

A plot may append `hour=24` with the `hour=0` value to close the line. Energy accounting must still use the original 24 hourly bins and must not include the display-only closing point as an extra hour.

## 8. Collaborator freight charging

Main code: `scripts/analyze_collaborator_truck_charging.py`, `run_baseline0_collaborator_pipeline.py`, and `replay_charging_power.py`.

The three independent cases are BF-06EP, BF-1EP, and HF-1EP. Session duration and exact hourly overlap are reconstructed from `time`, `energy_needed_kwh`, and `actual_charge_speed_kw`. Entire session energy is not assigned to the starting hour.

Current power references use BE records only; H2FC records are not mixed into BE grid charging. Curves use elapsed hours 0-23, expanded 10x and repeated for 365 representative days. Session tails after hour 24 are not automatically folded back by this first-day convention; disclose that choice when interpreting annualization. Negative SOC and other anomalies are diagnosed; negative SOC is not negative charging demand.

These are collaborator charging outputs. Matching scenario names do not establish a one-to-one correspondence with a particular traffic events file, and session counts are not unique electric-truck counts. The reference load does not establish zero unmet energy across the entire fleet.

## 9. Hourly power emissions and concentrations

Shared implementation: `src/impacts/pipeline/workflow/hourly_power.py`. Both the native YAML runner and `scripts/run_hourly_power.py` call it. `power_generation.py` retains direct matrix calculations, charging-bin allocation, and plotting utilities.

| Step | Implemented logic / review point |
|---|---|
| Input | `hour_of_year` must contain each hour 0-8759 exactly once; `charging_demand_MWh` must be finite and nonnegative, already representing the intended fleet's grid-side load |
| External assets | BREATHE allocator; currently fixed Cambium22 LowRECost p9 2040, eGRID2023, supplemental PM/NH3/VOC workbooks, and ReEDS spatial data |
| Hourly technology | Cambium `marg_gen_tech` for the corresponding hour |
| Generation requirement | `grid_MWh = enduse_MWh * (1 + distloss_rate_marg)`; grid distribution losses are distinct from charger efficiency |
| Plant allocation | External allocator distributes generation among eligible technologies/geographies in proportion to eGRID annual generation, with capacity fallback where required; not an economic dispatch optimization |
| Hourly emissions | Plant generation in MWh multiplied by an emission rate in kg/MWh |
| Annual check | Sum hourly plant emissions and compare each quantity with the allocator's annual table |
| Geographic selection | Only emitting plants physically inside the nine Bay Area counties enter concentration calculations; outside/unallocated generation is not reassigned into the Bay |
| ISRM | Convert plant kg/year through the short-ton convention for matrix calculations; require coverage of every selected source; no near-source removal |
| Source layer | Must be supplied explicitly; real-data reference replays use Layer 0, not elevated Layer 1 |

Power PM2.5 components are mapped as follows: direct PM2.5 to primary, VOC to SOA, NOx to nitrate, SOx to sulfate, and NH3 to ammonium. Total PM2.5 is primary plus these secondary components. Nitrate PM2.5 produced from NOx emissions is not a power NO2 concentration. The module does not construct power BC concentrations either.

The representative day is aligned by hour labels with Cambium `timestamp_local`. This check is not observational validation of DST or full calendar handling. Repeating a representative day for 365 days, combining data years, and omitting seasonal demand variation remain assumptions.

A regression test holds annual energy constant while moving demand between hours with different technologies and verifies that emissions change. The module recalculates from the hourly curve rather than using only annual energy.

## 10. Output files, exposure, figures, and web maps

| Location or product | Contents and interpretation |
|---|---|
| `concentrations/beam_inmap_concentrations.parquet` | ISRM-stage components for integration and full InMAP-only diagnostics; not every field is an unmodified InMAP-only result |
| `concentrations/beam_aermod_concentrations.parquet` | AERMOD-stage components, grid, and diagnostics |
| `exposure/beam_concentration_distribution.parquet` | Final integrated spatial concentrations and components; the directory name does not mean it contains only population averages |
| Scenario `products/...*_concentrations.parquet` | Exported group products, generally derived from the final integrated grid; consult the scenario README/manifest to determine power inclusion |
| `power_generation/plant_allocation/` | Hourly dispatch, plant generation/emissions, annual emissions, and QA |
| `power_generation/power_generation_isrm_concentrations.parquet` / `.gpkg` | Separate power ISRM PM2.5 components and geometry |
| `plant_emissions_used_for_concentration.csv` | Plants actually included in regional concentrations, not all candidate plants |
| `selected_plant_source_coverage.csv` | Selected plants and matrix-source coverage audit |
| `power_generation_summary.json` | Provenance, layer, scope, emissions, exclusions, and limitations |

Exposure calculations in `postprocess/step8_human_exposure.py` cover population-weighted concentrations, CES groups, schools, and LTC facilities. Web generation uses `step9_human_exposure_webmap.py` and `assets/human_exposure_webmap.html`. The existing presentation style is retained, with component/difference layers and facility markers that respond to map zoom.

Maps use a green-yellow-red scale, can be clipped to the Bay Area outline, and expose AERMOD, nonlocal InMAP, and TOTAL through the relevant entry points. Color limits may use a percentile cap. Values beyond that cap remain in the data; the legend maximum is not necessarily the data maximum. A 100 m rendering/output grid does not give every regional ISRM component independent physical information at 100 m resolution.

InMAP-only minus enhanced and cross-scenario current minus baseline are different comparisons. Read the layer definitions and align power inclusion, population scope, and valid CES samples before interpreting differences.

## 11. Engineering corrections and code organization

The clean review corrected explicit InMAP-only integration that used nonlocal components and still added AERMOD to BC/NO2. It also fixed uninitialized variables in conditional postprocessing paths and added a finite zero-budget fallback in another retained integration branch to prevent NaNs.

Configuration/manifest path handling, cross-platform test expectations, YAML/HTML packaging, Python requirements, and optional profiling dependencies were updated. Replay drivers record actual import locations to avoid accidentally executing the original editable installation. The source archive excludes raw data, matrices, results, external notebooks/allocator, and virtual environments.

**Readability assessment: the core retains the original `config -> preprocessor -> runner -> workflow -> postprocessor` organization, and power now follows that structure. Style is not fully uniform across all additions.** Some charging code remains compact; `scripts/` contains research drivers tied to dataset layouts and case names; helper defaults sometimes differ from explicit reference-case arguments. Prioritize shared `src/impacts` code and `replay_*` entry points. Do not treat every historical sensitivity script as the same production workflow.

See [REPRODUCING.md](REPRODUCING.md) for calls, inputs, and commands. The review did not redesign all research scripts into one charging-configuration system.

## 12. Validation evidence and untested scope

| Check | Completed evidence |
|---|---|
| Latest full test suite | November 2, 2026: 402 passed, zero failed, with external notebook references supplied; includes five restored-compatibility tests |
| Original-skims emissions regeneration | 20,603,480 InMAP-emissions rows; grouped comparison by source/class keys; largest absolute difference across outputs 5.68e-13 short tons/year |
| 3 km traffic replay | 2,559 ISRM receptors and 6,957,973 final 100 m cells; largest concentration difference 2.22e-16 micrograms/m3 |
| BF-06EP traffic compatibility replay | Its archived 4.5 km policy, same grid scale; largest difference 7.11e-15 micrograms/m3; not a 3 km result |
| LDV and three freight power cases | Replayed through the shared entry point on November 2; hourly allocation, annual emissions, 2,559 cells per case, and three combined surfaces matched; largest difference 1.08e-19 micrograms/m3 |
| Exposure postprocessing | 11 products: four Parquet, three CSV, three PNG, one HTML; numerical agreement, pixel-identical PNGs, byte-identical HTML |
| Packaging | Wheel build, isolated imports, CLI help, and YAML/HTML resources passed |
| Source checks | Python syntax, diff whitespace, size/path/common-secret-pattern checks, and original-source hash checks passed |

Charging/power reference values, already expanded 10x:

| Case | Daily grid load, MWh | Peak, MW | Peak hour |
|---|---:|---:|---:|
| LDV Home80/Home80 | 158.070889 | 15.704302 | 19 |
| Freight BF-06EP | 1,409.340412 | 105.121024 | 16 |
| Freight BF-1EP | 1,172.786960 | 92.533448 | 14 |
| Freight HF-1EP | 1,640.127618 | 134.110560 | 13 |

Large traffic/exposure validation was completed on November 1. Following the November 2 power refactor, the full tests and four power cases were rerun; unchanged large traffic calculations were not repeated. The subsequent complete review and English translation change documentation only, not model calculations.

The following remain outside completed validation: full raw-events traffic replays for BF-1EP/HF-1EP; rebuilding all network preprocessing; a real elevated Layer 1 replay; live EVI/DST verification; calibration against measured charging and air quality; installation into a fresh dependency environment; and comprehensive cross-browser interaction testing.

Traffic replays retain previously authorized missing-source exclusions: 114 original-skim sources and 103 BF-06EP sources. Numerical reproduction does not establish complete matrix coverage. Native processing still raises errors for missing matrix sources; explicit exclusions in individual research drivers are not a general default.

Evidence: `validation_results.json`, `hourly_power_validation.json`, `validation_environment.json`, and `VALIDATION.md`.

## 13. Reviewer checklist

- [ ] Explicitly select `additive_inmap_aermod` rather than relying on the inherited example default.
- [ ] Identify traffic, LDV, and freight inputs for every case, distinguishing `0.events`, Cristian's `3.events`, and collaborator CSVs.
- [ ] Apply 10x once per pathway; verify annual operating days and representative-day assumptions.
- [ ] Review the source-centroid/receptor-area ISRM rule and its boundary differences from the AERMOD circle.
- [ ] Accept and disclose excluded-source coverage, or obtain missing matrix responses and rerun.
- [ ] Label power results with Layer 0, Bay Area plant scope, approximate allocation, and fixed data years.
- [ ] Confirm Home80, L1/L2, grid/battery energy, and PHEV policy for the intended scenario; do not equate EVI settings with model settings.
- [ ] Disclose BEV exclusions, PHEV fuel fallback, freight first-day truncation, and unmet demand.
- [ ] Identify power contributions included in final PM2.5; do not imply power BC/NO2 coverage.
- [ ] Avoid double-counting groups; specify consistent PWC/CES/school/LTC denominators.
- [ ] Distinguish display limits from data ranges and verify difference direction and comparison scope.
- [ ] Before publication, resolve the upstream `LICENSE.txt` value of `TBD` versus MIT package metadata and confirm external code/data permissions; the review has not reassigned licensing.

## 14. Suggested reading order

Read this document and [CHANGE_INDEX.md](CHANGE_INDEX.md), then inspect:

1. `src/impacts/config/settings.yaml`, `settings.py`, and `runner.py`: entry points and parameters.
2. `prepare_emissions/event_emfac.py`, `from_events.py`, and `annualization.py`: emissions and expansion.
3. `isrm_near_source.py`, `step2_compute_inmap_concentrations.py`, and `step3_compute_aermod_concentrations.py`: responses and cutoffs.
4. `step4_prepare_exposure.py`: components, baselines, and final addition.
5. `charging/ldv_home_sensitivity.py` and `scripts/replay_charging_power.py`: current LDV arguments and SOC.
6. `scripts/analyze_collaborator_truck_charging.py`: freight sessions and hourly load.
7. `hourly_power.py` and `power_generation.py`: plant allocation, scope, and matrix multiplication.
8. `postprocess/step8_human_exposure.py` and `step9_human_exposure_webmap.py`: exposure and web output.
9. Corresponding `tests/`, machine-readable validation reports, and `source_inventory.json`: evidence and version fingerprints.

Workflow filenames without a full prefix are under `src/impacts/pipeline/workflow/`; postprocess files are under `src/impacts/pipeline/postprocess/`. This review documents the current implementation and outstanding decisions without presenting assumptions as completed observational validation.
