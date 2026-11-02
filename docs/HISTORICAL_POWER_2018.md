# Historical electricity for the 2018 baseline

The default power configuration now selects `historical_generation_share` and
`electricity_year: 2018`. Power remains opt-in. Changing a Cambium filename or
relabeling a 2040 result as 2018 is not supported. Cambium 2022 projection files
available for this project start in 2024; there is no 2018 projection file.

## Inputs and method

- EPA [eGRID2018 v2](https://www.epa.gov/sites/default/files/2020-03/egrid2018_data_v2.xlsx):
  plant locations, annual net generation, NOx, SO2 and greenhouse gases.
- EPA [supplemental workbooks](https://www.epa.gov/egrid/egrid-related-materials):
  the **2018 worksheets** for PM2.5, NH3 and VOC. The NH3/VOC filenames contain
  `2021`, but their workbooks also contain 2018 inventory estimates. The release
  name is not the inventory year. The estimates use EPA's draft NEI-based
  methodology, not necessarily measurements made at every plant in 2018.
- ReEDS `p9` polygon: allocation domain. This is a geographic modeling region,
  not a claim to reconstruct the historical CAISO balancing-authority footprint.
- eGRID2018 `GGL18`, Western region: grid gross loss fraction 0.0480444635753805.
- Existing hourly charger-meter demand, already fleet-expanded. No second
  fleet multiplier or second charger-efficiency correction is applied.

For each plant with positive annual net generation inside the selected region:

```
weight_i = generation_2018_i / sum(generation_2018_all_selected_plants)
added_generation_i,h = charging_meter_MWh_h / (1 - grid_loss) * weight_i
emissions_i,h = added_generation_i,h * emission_rate_i_kg_per_MWh
annual_emissions_i = sum_h(emissions_i,h)
```

The denominator includes wind, solar, hydro and nuclear plants. Plants with
missing or nonpositive net generation receive no allocation and are listed
separately. Missing emissions at explicitly noncombustion primary fuels are
treated as zero direct emissions, with that assumption recorded. Other missing
rates cause an error unless `missing_rate_policy: subregion` is explicitly set;
then a same-year eGRID subregion rate is used and labeled for each plant.
Existing valid plant-level factors, including zero values, are retained.

This is **historical average generation accounting**. Fixed annual shares apply
at every hour, including for solar generation. It does not model hourly plant
availability, imports, storage, marginal dispatch or transmission constraints.
Consequently, shifting the charging curve while preserving annual energy changes
hourly emissions but does not change annual emissions under this method.

Only emissions from plants located inside the nine Bay Area counties enter the
current concentration calculation. Outside-Bay allocations are not redistributed.
Annual pollutant masses are multiplied by the explicitly selected ISRM source
layer. No local-to-local removal is applied to power emissions. NOx contributes
to secondary nitrate PM2.5; this does not create a direct power NO2 or BC product.
Layer 0 remains an explicitly labeled ground-release approximation.

## Configuration

```yaml
power_generation:
  enabled: true
  allocation_method: historical_generation_share
  electricity_year: 2018
  balancing_area: p9
  grid_loss_region: Western
  missing_rate_policy: error  # Set subregion only to permit labeled same-year proxies.
  hourly_demand_file: /path/to/sfbay_hourly_charging_demand.parquet
  charging_summary_file: /path/to/beam_charging_summary.json
  breathe_root: /path/to/PowerGeneration
  county_boundaries_file: /path/to/county_boundaries.gpkg
  source_layer_index: 0
```

The historical data loader expects `input_Data/egrid2018_data_v2.xlsx`, the three
existing EPA supplemental workbooks and `ExternalData/ReEDS_BA/US_PCA/US_PCA.shp`
under `breathe_root`. Other inventory years require their matching workbook and
worksheets; missing data are never replaced by another year automatically.
Only 8760-hour demand files are currently supported, including representative
non-leap-year profiles. No daylight-saving shift is applied.

Use `scripts/run_hourly_power.py --help` for standalone execution. The same
implementation is called by the native YAML runner. Rebuild the preprocess
manifest after changing YAML; an existing manifest retains its saved settings.

The projection path remains available through explicit
`allocation_method: cambium_marginal_technology`, a supported electricity year,
`balancing_area`, `cambium_scenario`, and `egrid_year: 2023`. Its external allocator
currently uses eGRID2023 and 2021 supplemental factors. These years are recorded
and must not be described as a year-matched historical baseline. The selected
Cambium file is staged under the external allocator's fixed filename convention;
its original scenario path is recorded. Historical replay scripts explicitly
retain their original 2040 settings rather than inheriting the new default.

## Implementation and audit outputs

- `src/impacts/pipeline/workflow/historical_power.py`: inventory reading, rate
  provenance, generation shares and hourly/annual emission accounting.
- `src/impacts/pipeline/workflow/hourly_power.py`: method routing and ISRM.
- `src/impacts/config/settings.py` and `settings.yaml`: settings and validation.
- `scripts/run_hourly_power.py`: standalone CLI.
- `tests/test_historical_power.py`: energy conservation, zero-emission shares,
  missing rates, year isolation and native orchestration.

Outputs include the effective configuration, input SHA256 hashes, plant-level
factor sources, excluded generation records, hourly plant emissions, annual
emissions, source coverage and the selected ISRM layer. If an emitting source is
absent from the matrix, concentration execution fails and the summary records
the missing plants and source cells; absent coefficients are not treated as zero.

The research checkout's native runner has also been moved to this hourly entry
point. Its obsolete active example power settings were replaced by the disabled
modern template; historical debug settings were backed up outside the repository.
Saved historical outputs and their manifests were not rewritten.

## Real-input verification on 2026-11-02

The existing LDV Home80/Home80 curve was held fixed at 158.070889 MWh/day,
already expanded 10x. This is the Cristian `3.events` existing-EV curve, not a
new charging simulation from `0.events` and not a change to the charging model.

The 2018 inventory contains 533 positive-generation plants in p9. The run
allocated 57,695.874491 MWh/year of charger load to 60,607.740891 MWh/year of
generation after historical grid losses. Same-year regional proxies were used
explicitly for missing conventional factors; the NOx/SO2 proxy plants represent
0.25485% of reference generation. All selected plants have direct supplemental
2018 PM2.5, NH3 and VOC rate entries.

The Bay Area selection contains 87 emitting plants. The local compact matrix
lacks source cell **1534**, containing **Nove Power Plant (ORISPL 50831)**.
Both local compact matrices were checked; neither contains that source row.
Remote ISRM metadata access returned **HTTP 403 Forbidden**. The run preserved
complete hourly and annual emissions and stopped before producing a complete
concentration product. Nove's assigned generation is 0.367653 MWh/year and its
primary PM2.5 emission is 0.064466 kg/year. These small totals do not establish
that its local concentration contribution is zero.

Any later authorized partial concentration output must be separately labeled
and must retain the complete emissions and coverage audit. Comparisons with
the prior 2040 result change both electricity assumptions and allocation method;
they cannot isolate a year-only effect.

## Four-case emissions delivery

At the user's request, the common 2018 grid was applied to all four current
charging curves: LDV Home80, BF-06EP, BF-1EP and HF-1EP. The existing 10x curves
were retained without changing charging assumptions or adding another multiplier.
Three combined annual plant tables and hourly aggregate tables were also written
for LDV plus each freight case. Future freight scenarios therefore use a fixed
2018 grid sensitivity, not a year-matched future electricity forecast.

| Case | Charger demand (MWh/day) | Added generation (MWh/year) | Bay Area primary PM2.5 emissions (kg/year) |
|---|---:|---:|---:|
| LDV | 158.071 | 60,607.741 | 94.913 |
| BF-06EP | 1,409.340 | 540,371.089 | 846.229 |
| BF-1EP | 1,172.787 | 449,671.465 | 704.192 |
| HF-1EP | 1,640.128 | 628,859.813 | 984.804 |

Each case contains 4,669,080 plant-hour rows (533 plants x 8760 hours). Annual
and hourly energy/emission totals agree. Files contain complete emissions,
including Nove; concentration products are intentionally absent pending the
missing matrix row. The user explicitly chose to wait for full matrix coverage
rather than publish partial concentrations.

The standalone CLI supports `--emissions-only` for this situation. The native
pipeline still requires complete source coverage before passing power
concentrations to exposure. No partial concentration fallback was introduced.
The default missing-rate policy remains `error`; these audited runs explicitly
selected `subregion`. In addition to the 0.25485% NOx/SO2 reference-generation
share using proxies, CH4/N2O proxies cover 6.44058%, largely because missing
geothermal factors are not silently treated as zero.

Final validation: upload checkout **430 passed**. Research checkout **427 passed**
with the same six pre-existing terminal-contract failures (paths/profiling/example
contracts), not new power failures. All four real-input emissions cases passed
hourly/annual accounting, nonnegative/finite checks and factor-year checks.
No complete concentration result was validated, because source 1534 is absent.
