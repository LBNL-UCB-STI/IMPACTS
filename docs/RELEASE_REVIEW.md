# Changes relative to upstream IMPACTS

Current LDV recommendation: restored original notebook compatibility via
`scripts/run_ldv_charging.py`; see `CRISTIAN_COMPATIBILITY.md`. Designed
Home80/periodic cases below are archived sensitivities and prior replay evidence.

Review date: 2026-11-01. Upstream: https://github.com/LBNL-UCB-STI/IMPACTS,
commit `159a5598210905471a0412609d8414ccf97f355f` (main when checked).
The original local checkout had the same HEAD; differences were uncommitted
modifications and added research code, not a different upstream branch.

This clean checkout was created separately. The original data, code and results
were retained. Only source, tests, configuration templates and documentation
are included. See `source_inventory.json` for file-level additions/modifications
and SHA-256 fingerprints, and `VALIDATION.md` for measured results.

## What changed and where

Paths below are relative to the repository root. `W` means
`src/impacts/pipeline/workflow/`; `P` means `src/impacts/pipeline/postprocess/`.

| Area | Implementation | Change and scientific effect |
|---|---|---|
| Native emissions input | `W/prepare_emissions/from_events.py`, new `event_emfac.py`; `preprocessing/step1_collect_inputs.py` | Events-only fallback reconstructs per-link/process activity and uses supplied EMFAC rates; skims remain supported. Matching network, explicit vehicle mapping, county intersections and rates are required. Unmapped vehicle types fail rather than acquiring invented factors. |
| Annualization / class selection | `W/prepare_emissions/annualization.py`, `from_skims.py`; `config/settings.py`, `manifest/schema.py` | Preserves vehicle-category operation days, separate sample fractions and source attributes. Supports scenario/passenger/freight selections. Scaling is applied once; 10x is a scenario assumption, not an inherent event-file property. |
| EMFAC mapping | `pipeline/emfac/_common.py`, `fleet/step3_map_emfac_atlas.py` | Fuel/category mapping and exact model-year handling; see events documentation for supported mixtures and missing-factor behavior. EMFAC processing existed upstream; the events fallback and scenario adapters are additions. |
| ISRM response selection | `W/step2_compute_inmap_concentrations.py`, new `isrm_near_source.py`; `runner.py` | Uses global source IDs for compact stores and full study-area receptors. Removes near-source responses for traffic primary PM2.5/BC/NO2. Retains unmodified InMAP-only baseline columns. Missing matrix source coverage fails in native processing; historical scenario drivers explicitly filter authorized missing cells. |
| AERMOD | `W/step3_compute_aermod_concentrations.py` | Truncates response offsets beyond 3 km, includes policy in cache identity, and selects the nearest available ASRV site with the same source class when the original nearest-site pattern is missing. No fallback to an incompatible height/temporal/urban class. |
| Integration | `W/step4_prepare_exposure.py`, `runner.py` | Adds nonlocal traffic + local AERMOD directly in additive mode; secondary remains regional. Adds power only after traffic integration. Support indicators remain diagnostics; additive primary does not need indicator multiplication or renormalization. |
| Original charging adaptation | new `W/charging/ldv_charging.py` | Loads selected functions from trusted collaborator notebooks, prepares parking stops and optimizes assignment. Reference tests compare stop choices, charger choices, SOC and delivered energy to original functions. These optimizations are distinct from later model changes. |
| Designed LDV charging | new `W/charging/ldv_home_sensitivity.py` | Home-preference assignment and periodic SOC convergence. BEV feasibility checks, PHEV electric-first/fuel fallback. These are substantive model changes from the original single-day stochastic notebook, not merely compatibility fixes. |
| Other charging pathways | new `W/charging/beam_sessions.py`, `charging/truck_opportunity.py`, `charging_plots.py` | Native BEAM session reconstruction, class-specific truck opportunity-charging sensitivity and plotting. Keep these alternatives separate from the designed LDV and collaborator freight scenarios. |
| Power generation | new `W/hourly_power.py`, `W/power_generation.py`; `scripts/run_hourly_power.py` | Native YAML and CLI share hourly Cambium/eGRID allocation via a supplied external BREATHE allocator, annual emissions and direct ISRM. WA proxy and annual-scaling alternatives were removed on November 2. |
| Freight collaborator data | `scripts/analyze_collaborator_truck_charging.py`, `run_baseline0_collaborator_pipeline.py`, `combine_ldv_freight_power.py` | Exact overlap of session energy/power with hourly intervals; BE and H2FC separate; first 24 hours, 10x and repeated representative day explicitly recorded. Retains negative SOC/overlap diagnostics. Adds LDV and freight power components without a second expansion. |
| Source and gateway products | `scripts/run_ldv_freight_products.py`, `run_gateway_freight_products.py`, `run_future_fixed_factor_scenarios.py`, related `*products*` scripts | Passenger/port/non-port/gateway attribution using supplied vehicle-ID classification. Existing fixed-factor future cases are experiments, not future EMFAC rate forecasts. |
| Maps | `P/_common.py`, `step4_plot_concentrations.py`; `scripts/plot_gateway_products.py` | Green-yellow-red concentration maps, offline fallback, Bay Area masks, pollutant components, optional power-plant markers, shared scales. A display percentile cap is not a concentration maximum. |
| Human exposure / web | new `P/step8_human_exposure.py`, `step9_human_exposure_webmap.py`, `assets/human_exposure_webmap.html`; `postprocessor.py` | Population weighting, CES groups, schools/LTC, original-style figures and interactive 100-m raster rendering including InMAP-only minus enhanced difference. Plant/facility symbols scale with the map. HTML template contains empty data placeholders; data are populated only in output. |
| Configuration / bookkeeping | `config/*`, `manifest/file_ops.py`, `preprocessor.py`, `runner.py`, `common.py` | New optional settings/output paths, input detection and table compatibility. Preserves upstream command structure. |

## Scientific contract for the current hybrid method

1. ISRM: construct a 3,000 m circle around each source-cell centroid. A receptor
   response is zeroed when intersection area / receptor-cell area is **greater
   than 0.5**. This is an area-overlap rule, not simply centroid distance.
2. Apply that mask only to traffic primary PM2.5, BC and NO2 responses. Retain
   secondary PM2.5 and all power-generation responses.
3. AERMOD: retain offsets whose Euclidean source-to-receptor grid-center distance
   is at most 3,000 m. Diagonal points outside the circle are zeroed. The
   pre-generated template's resolution/extent still limits available responses.
4. Explicit additive mode:
   `traffic primary = nonlocal ISRM primary + AERMOD primary`;
   `traffic total PM2.5 = traffic primary + ISRM secondary`.
   BC and NO2 likewise sum their traffic components. Power PM2.5 is then added.
5. InMAP-only uses the **full** response, without near-source removal. The
   `inmap_only_*` fields remain traffic-only diagnostic baselines; if optional
   power inputs are supplied, final `PrimaryPM25/SecondaryPM25/TotalPM25` include
   power even when the traffic strategy is InMAP-only.
6. These distinct AERMOD-distance and ISRM-area rules do not mathematically
   guarantee a seamless partition at a 3 km boundary. Radius/overlap sensitivity
   remains scientific work, not something a software parity check validates.

The BF-06EP historical comparison explicitly uses 4.5 km ISRM removal and the
full available AERMOD template. It runs in a separate process so its policy
cannot leak into the 3 km case. Do not compare methods under different cutoffs
without identifying that change.

## Clean-release fixes from this review

- Fixed explicit `inmap_only` integration: use unmodified primary/BC/NO2 fields
  and omit AERMOD from final traffic concentration. Previously the branch still
  used local-removed primary and added AERMOD to BC/NO2. Added a regression with
  deliberately different full/nonlocal inputs plus an independent power term.
- Restored finite InMAP fallback for legacy AERMOD budget scaling when AERMOD
  has zero mass but the target budget is positive. Normal additive mode is
  unaffected.
- Initialized postprocess power-marker and delta-baseline variables before
  conditional map creation, preventing unbound variables in alternate paths.
- Removed machine-specific source paths from research drivers; external roots
  can be specified by environment variables or explicit replay arguments.
- Added non-overwriting replay entry points. Recorded import paths ensure the
  clean source is executed rather than the original editable installation.
- Corrected OS-specific test assertions (path separators and `time` flags).
  Notebook parity tests accept `IMPACTS_CHARGING_NOTEBOOKS`; absent external
  notebooks produce explicit skips, not false parity passes.
- Packaged YAML defaults and the HTML template; made Memray an optional
  non-Windows profiling extra. CPU profiling remains portable; `/usr/bin/time`
  and Memray modes still require a supported host/tool installation.
- Raised the advertised Python minimum to 3.11, matching the existing pinned
  Zarr 3.1.5 requirement. The real replays used Python 3.12.
- Excluded events, results, matrices, external datasets, notebooks, caches,
  virtual environments and local secrets from the source archive.

## Review findings and limits

| Finding | Disposition |
|---|---|
| Old power results could survive disabling power or changing inputs | Fixed: automatic runner reuse is disabled; exposure recomputes enabled power, invalidated combined outputs cannot be postprocessed. See `POWER_RESULT_ISOLATION.md`. |
| Inherited default selected legacy imputation | Fixed on November 2: Python fallback and shipped YAML use `additive_inmap_aermod`. Explicit legacy choices remain available; saved manifests retain their recorded strategy. See `HYBRID_DEFAULT_UPDATE.md`. |
| Historical scalar power route differed from hourly dispatch | Removed on November 2. Native YAML and `run_hourly_power.py` now use the same implementation; obsolete configuration is rejected. See `HOURLY_POWER_UPDATE.md`. |
| Compact matrix has missing traffic sources | Existing authorized exclusions retained in representative tests (114 original-skim sources; 103 BF-06EP sources). No claim of complete-source coverage. |
| Recent power runs use Layer 0 | Preserve documented temporary assumption. Elevated Layer 1 support is tested structurally but not empirically revalidated by a real Layer 1 run here. |
| Power allocation is approximate | Cambium marginal technology plus proportional eGRID allocation is not MCGRID optimization or observed hourly plant dispatch. Outside-Bay plants/unallocated generation are reported, not reassigned into the Bay. |
| Power chemical outputs | ISRM PM2.5 components are available. NOx emissions alone are not a power NO2 concentration; no invented power BC/NO2 fields. |
| LDV validation | EVI is a scenario benchmark, not measured charging validation. Periodic SOC, 80% preference and .202 kWh/km battery energy / .9 efficiency are modeling assumptions. |
| Freight validation | Collaborator successful charging records, first-day truncation and 10x are supplied-load scenarios. Negative SOC is flagged, not treated as negative charging energy. |
| Dataset reuse | Dispersion replays reuse audited annual emissions; the separate skim replay recomputes emissions/annualization. Preprocessing network intersections and population assets are read-only reused inputs. Full raw-event reruns of BF-1EP/HF-1EP were excluded by agreed scope. |
| Code execution / web | Notebook functions are executable trusted code, not sandboxed documents. Generated maps can contain sensitive spatial data; the upload template contains no actual data. Web libraries/tiles may require internet even though concentration payload is embedded. |
| Redistribution | Upstream `LICENSE.txt` is `TBD` while metadata says MIT. This review does not resolve ownership/licensing. Collaborator notebooks and external allocator/data are not bundled. |

The regression results establish reproducibility under the recorded inputs and
dependencies. They do not establish predictive accuracy of EMFAC, charging,
ISRM, AERMOD, or the power allocation assumptions.

Electricity-year follow-up: [2018 historical allocation, all four charging cases and pending ISRM coverage](HISTORICAL_POWER_2018.md).
