# Validation results — 2026-11-01

November 2 updates: the hourly-only power refactor passed 397 tests and four power replays. The subsequent original-charging compatibility restoration passed 402 tests and a real-data charging comparison; see [CRISTIAN_COMPATIBILITY.md](CRISTIAN_COMPATIBILITY.md). See [HOURLY_POWER_UPDATE.md](HOURLY_POWER_UPDATE.md) for the power refactor. The results below document the November 1 validation.

**The agreed representative replays passed.** The clean checkout reproduces
the saved additive traffic and charging/power results within floating-point
rounding. The explicit InMAP-only correction is separately tested; it is an
intentional change to a previously incorrect alternate branch.

Machine-readable details: [validation_results.json](validation_results.json).
Environment: [validation_environment.json](validation_environment.json).
Code/reference map: [RELEASE_REVIEW.md](RELEASE_REVIEW.md).

## Test suite

| Execution | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Original working tree, before release fixes | 383 | 6 | 0 |
| Clean checkout, external notebook references supplied | **392** | **0** | **0** |
| Clean checkout, without external notebooks | **379** | **0** | **13** |

The original failures were the locally changed example settings and OS-specific
test expectations for paths and `/usr/bin/time` flags. The clean suite adds
three regression tests for InMAP-only, zero-budget fallback and conditional
postprocessing. Thirteen reference-notebook tests skip explicitly when the
notebooks are absent. Six Zarr deprecation warnings originate in synthetic
test-fixture construction; no numerical test failed.

The suite exercises emissions, mapping, annualization, compact matrix IDs,
near-source masks, AERMOD kernels/support, integration, charging/SOC,
truck/native EV helpers, scenario/gateway products, plotting and web data.
This is the existing full suite plus the new fixes, not a claim of 100% code
coverage or real-data execution of every archived sensitivity script.

## Annual emissions regenerated from original skims

Original reference: `impacts-postsim--sfbay--base--20260901-165739`.
The run rebuilt preparation, annualization, inventory correction and spatial
allocation from the manifest's `0.skimsEmissions.parquet`. It did not merely
copy annual emissions. All six exported pollutant columns were compared after
grouping by scientifically relevant source/class keys (parallel joins need not
retain row order).

| Product | Compared groups | Maximum absolute difference (short tons/year) |
|---|---:|---:|
| InMAP source cell × vehicle type × process × road class × release height | 5,405,089 | 2.84 × 10⁻¹⁴ |
| AERMOD cell × release height × temporal class × urban class | 301,241 | 4.26 × 10⁻¹⁴ |
| County × vehicle type × process × road class × release height | 272,538 | 5.68 × 10⁻¹³ |

The full InMAP emissions export has 20,603,480 rows before the comparison
aggregation. Original input and output files were not replaced.

## Traffic concentration replays

Both cases recomputed 2,559 InMAP receptors and **6,957,973 final 100-m cells**,
including the AERMOD stage. Every numeric field, receptor ID/order and geometry
was checked against the saved counterpart; NaN diagnostic fields were compared
consistently. Concentration tolerance: `rtol=1e-9`, `atol=1e-10`.

| Reference case | Replayed policy | Largest absolute concentration difference (µg/m³) |
|---|---|---:|
| `original-skims-3km-test-20261015/new_3000m_circle` | 3 km ISRM area overlap; 3 km circular AERMOD cutoff | 2.22 × 10⁻¹⁶ |
| `events-sfbay-future-scenarios-20261011/BF-06EP/Total` | Historical 4.5 km ISRM removal; full AERMOD template | 7.11 × 10⁻¹⁵ |

Population-weighted final concentrations, each shown as
**saved reference → clean replay**, in µg/m³:

| Case | PM2.5 | BC | NO2 |
|---|---|---|---|
| Original skim / 3 km | 1.523551 → 1.523551 | 0.070514 → 0.070514 | 5.048387 → 5.048387 |
| BF-06EP / legacy 4.5 km | 0.633899 → 0.633899 | 0.026874 → 0.026874 | 2.666672 → 2.666672 |

These checks use all population rows joined to the concentration grid
(6,687,433 people), not the more restrictive valid-CES manuscript denominator.
They should not be substituted directly for an earlier CES-filtered table.
The full InMAP-only diagnostic columns also matched. No additional fleet
expansion or power generation was applied in these traffic comparisons.

## LDV and freight charging → power

LDV: seed-42 BEV Home80/PHEV Home80 periodic model, .202 battery kWh/km,
90% charging efficiency; exact same eligible/excluded vehicles as the reference.
Vehicle, trip and charging-session tables were compared, not only the curves.
Freight: all three raw collaborator CSVs were reconstructed through exact
hour/session overlap, BE records only, first 24 hours, 10x.

| Case | Daily charging (MWh, 10x) | Peak (MW) | Peak hour | Largest ISRM difference (µg/m³) |
|---|---:|---:|---:|---:|
| LDV Home80/Home80 | 158.070889 | 15.704302 | 19 | 5.42 × 10⁻²⁰ |
| Freight BF-06EP | 1,409.340412 | 105.121024 | 16 | 1.08 × 10⁻¹⁹ |
| Freight BF-1EP | 1,172.786960 | 92.533448 | 14 | 1.08 × 10⁻¹⁹ |
| Freight HF-1EP | 1,640.127618 | 134.110560 | 13 | 0 |

For every case, the audit reran the external allocator, IMPACTS hourly plant
emissions, annual totals, Bay Area plant selection and Layer-0 ISRM multiplication.
Hourly dispatch tables, annual plant emissions and all PM2.5 concentration
components matched. All three LDV+freight combined surfaces also matched.
Each power surface has 2,559 receptors. No local-to-local removal was applied
to power, and no extra scaling was applied to already 10x curves.

This is reproducibility of the recorded **Layer 0, Cambium 2040/eGRID proxy**
scenario. It is not a new validation of elevated-layer transport, physical
dispatch, charging behavior, or freight feasibility.

## Exposure, plots and web

Ran both the original and clean postprocessor on the same newly recomputed
3 km concentration surface. Compared 11 outputs:

- Four Parquet products: populated grid, CES tracts, schools, LTC.
- Three original-style CES CSV tables.
- Three PNG figures: **pixel-identical** between code versions.
- Interactive 100-m concentration/exposure HTML: **byte-identical**.

One generated 3×3 comparison figure was also visually inspected. Existing web
tests verify data/raster construction; no new cross-browser interaction audit
was performed. This postprocess comparison is a fresh reference run, not a
claim that a historically different concentration scenario has identical plots.

## Packaging and source checks

- Wheel built successfully; isolated imports and `python -m impacts --help`
  passed from an extracted wheel outside the source tree.
- Packaged default YAML and HTML asset were present; Python minimum is 3.11,
  and Memray is optional/non-Windows.
- Source inventory excludes raw data, matrices, notebooks, results, virtual
  environments, `.git`, and external allocator/data.
- Python syntax, `git diff --check`, file-size checks, local path/common-secret
  pattern checks passed. The original copied source-file hashes remained
  unchanged. Pattern scanning is not a proof that every possible secret or
  redistribution issue has been found.

## Deliberate scope limits

As agreed, BF-1EP and HF-1EP **full traffic pipelines** were not rerun; their
freight power cases were rerun. Network preprocessing/intersections and
population input products were reused. EMFAC events conversion and other
branches were covered by the module suite, not by a fresh full raw-events run
of every scenario. No real elevated Layer 1 replay, live EVI API request, new
field-data calibration, or clean-environment dependency installation was
performed. The third-party environment was held fixed to isolate code changes.

The source package remains a review candidate because the upstream license
file says `TBD`; no new distribution license has been assigned by this review.
