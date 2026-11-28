# ACS-adjusted exposure population: control specification

## Status and scope

IMPACTS uses baseline-year ACS tables to calibrate whole-household weights to county controls and assign the weighted residents to the existing grid. With a Census bundle configured, it also allocates the group quarters population to the grid. The exposure table reports raw UrbanSim population, ACS-adjusted household population, group quarters population, and the adjusted total side by side.

The configured baseline year is `run.start_year` (currently 2018). All ACS acquisition derives its release year from this setting. For a baseline year `Y`, the corresponding five-year ACS release is `Y-4` through `Y`; the one-year release is `Y` when the requested geography and table are available. Do not silently substitute a newer release.

For the current baseline, that means:

- **2018 ACS 1-year:** county-level regional controls where available.
- **2014–2018 ACS 5-year:** tract and block-group controls for spatial allocation and local calibration.
- **No ACS block-level controls:** any placement below tract or block group is an allocation method and must be described as such.

The Census Bureau lists county-level availability for 2018 ACS 1-year estimates and tract/block-group geography for 2018 ACS 5-year estimates. [2018 ACS 1-year availability](https://www.census.gov/data/developers/data-sets/acs-1year.2018.html) · [2018 ACS 5-year geography levels](https://api.census.gov/data/2018/acs/acs5/geography.html)

## Population components and controls

Keep three components identifiable in outputs:

1. **Raw UrbanSim household population:** the existing, full geocoded persons and households supplied by PILATES. Keep this as the unadjusted reference result.
2. **ACS-adjusted household population:** reweight or calibrate whole households, applying each household's weight to all its resident person records. Do not create household members from ACS totals or treat GQ residents as household members.
3. **Group-quarters population:** a separate non-household population component. Keep its count, source, and spatial allocation method separate from household counts and household population.

Relevant ACS table universes and controls:

- `B11001`: total households; universe is **households**.
- `B25009`: households by tenure and household size; universe is **occupied housing units**. For occupied units, this total equals the household count. Its 7-or-more-person category is open-ended, so converting it to persons needs an explicitly sourced average household size.
- `B11002`: population in households; universe is **population in households**. Use it to compare with the person total implied by the household rows and decide whether it can be a compatible hard control.
- `B01003`: total population; universe is **total population**. Use as a check that household plus GQ components reconcile.

GQ control:

- `B26001`: group-quarters population; universe is **population in group quarters**. ACS 5-year GQ estimates are available down to tract geography. A finer distribution must be sourced or allocated separately; do not imply ACS provides block-level GQ counts. [Census ACS GQ table guidance](https://www.census.gov/content/dam/Census/programs-surveys/acs/guidance/training-presentations/Intro%20to%20ACS-GQ%20Data.pdf)

Before setting hard constraints, compare each estimate and its margin of error at the selected geography. Household counts, household-size counts, and household-person totals are independently estimated and may not reconcile. If they conflict, the method must state which controls are binding, how uncertainty affects calibration, and what residuals remain. Do not claim exact ACS agreement across all measures unless the achieved totals demonstrate it.

## Verified 2018 county control totals

The reported figures were checked against the Census Bureau's downloadable 2018 ACS table-based summary files, summing the nine configured Bay Area counties (Alameda, Contra Costa, Marin, Napa, San Francisco, San Mateo, Santa Clara, Solano, and Sonoma). The summary file contains estimates and margins of error. [Census 2018 ACS Summary File](https://www.census.gov/programs-surveys/acs/data/summary-file.2018.html)

| Control | 2018 ACS 1-year county sum | 2014–2018 ACS 5-year county sum |
| --- | ---: | ---: |
| Total population (`B01003_001E`) | 7,753,023 | 7,675,798 |
| Households (`B11001_001E`; also `B25009_001E`) | 2,731,126 | 2,715,010 |
| Population in households (`B11002_001E`) | 7,614,828 | 7,536,457 |
| Group-quarters population (`B26001_001E`) | 138,195 | 139,341 |

The 2018 ACS 1-year household and GQ estimates sum exactly to its total population estimate: 7,614,828 + 138,195 = 7,753,023. The sum of the nine county household-count MOEs in quadrature is approximately 9,610 (1-year) and 3,769 (5-year); the corresponding population-in-households MOEs are approximately 6,036 and 2,599, and GQ MOEs approximately 6,036 and 2,599. The total-population table uses a Census sentinel for its county MOE, so no combined MOE is reported here. These are regional combinations of county estimates, not a separately published ACS regional estimate.

The 2018 ACS 5-year sums differ from the 1-year sums because they represent a different survey period. For 2018 exposure, use 2018 ACS 1-year county totals as the binding household and GQ controls. Use 2014–2018 ACS 5-year tract `B26001` estimates to distribute the GQ county control among tracts, documenting the rescaling. Household locations already come from the block-level UrbanSim records; household calibration preserves those locations and weights households within each county. Do not combine releases without recording the distinction.

The email thread reports 7,195,368 UrbanSim persons and an approximately 7.18 million household-person total inferred from B25009. This calculation is reproducible from the 2018 ACS 1-year table and matching 2018 California PUMS housing records:

- The tract-to-PUMA crosswalk identifies 55 PUMAs covering the nine target counties, with no selected PUMA crossing into a county outside the region.
- Applying PUMS household weights (`WGTP`) to California PUMS records with `NP >= 7` gives a regional mean of **8.028 persons per 7-or-more-person household** (47,780 weighted households).
- Applying that mean to the regional B25009 household-size counts gives **7,188,954 implied household residents**. This is a derived estimate, not a direct ACS population control; the sampling uncertainty of the PUMS mean has not been propagated into an MOE.
- The regional 7-or-more-person household estimates are 27,484 owner-occupied (sum-of-county MOE about 2,540) and 19,406 renter-occupied (MOE about 2,549).

The derived 7,188,954 is close to the reported UrbanSim 7,195,368 (difference 6,414), but is **425,874 below** the direct B11002 household-population estimate of 7,614,828. The difference is much larger than the county-table MOEs, although the derived estimate's PUMS uncertainty has not been propagated, so this is not a formal significance test. Do not force B11002 and the B25009-derived person estimate to match simultaneously; the inspected UrbanSim baseline has exactly 2,731,126 households, matching the same-region 2018 B11001/B25009 household total.

## Checked UrbanSim baseline file

The supplied `model_data_2019.h5` was inspected read-only. It contains separate `/2018` and `/2019` table groups, so a 2018 run must select `/2018` from `run.start_year`, not infer the data year from the filename or choose the latest group. The `/2018/households` table has **2,731,126 rows**, exactly matching the 2018 ACS 1-year household count. `/2018/persons` has **7,195,368 rows**; the household `persons` column also sums to 7,195,368. The mean is **2.6346 residents per household**. This confirms the real PILATES input pair and resolves the earlier need to compare the actual file.

For the household component, the evidence supports using county ACS 1-year **B11001 household counts** and **B11002 household-population totals** as binding totals. Entropy calibration by household size adjusts each whole household's weight while preserving its UrbanSim home location. This follows the email thread's short-term proposal to reweight to ACS person totals. Treat B25009 and the PUMS-derived household-person total as diagnostics, not simultaneous hard constraints: those estimates cannot be reconciled exactly with B11002. Keep B26001 group-quarters residents outside household weights.

The 2018 ACS 1-year GQ county total is 138,195, but the location source for assigning those residents below county/tract level remains a separate choice. Census 2020 P5 provides block-level GQ counts, but the Census Bureau documents COVID-related GQ enumeration risks, especially for college students. 2010 Census P42 provides an earlier block-level distribution that predates the pandemic and aligns with the 2010 geography used by the 2014–2018 ACS, but it is eight years older than the analysis baseline. These should be treated as spatial allocation weights, scaled to the selected 2018 ACS GQ controls, not as 2018 population totals. [2020 Census P5 table](https://data.census.gov/table/DECENNIALPL2020.P5) · [Census assessment of 2020 COVID-related quality risks](https://www2.census.gov/programs-surveys/decennial/2020/program-management/evaluate-docs/EAE-2020-quality-impact-covid19.pdf) · [2010 Census P42 summary file](https://www2.census.gov/programs-surveys/decennial/rdo/about/redistricting/gqsf.pdf)

## IMPACTS implementation boundaries

Keep input acquisition, household calibration, GQ allocation, and grid assignment separate:

1. **ACS acquisition/cache module:** fetch only the release derived from `run.start_year`; cache immutable source tables under a durable path keyed by product, year or period, table IDs, geography, and boundary vintage. Record retrieval URL, retrieval date, checksums, and ACS metadata. This is input management, not a pipeline transformation.
2. **Population calibration step:** use the persons, households, and blocks tables for the configured year in the UrbanSim HDF5 registered in preprocessing Step 1. Produce a distinct adjusted household population and calibration diagnostics.
3. **GQ preparation/allocation step:** create GQ population records or cell totals separately from household records. Record the ACS geography and any ancillary spatial source and allocation method.
4. **Cell assignment step:** assign raw household, adjusted household, and GQ components to the existing exposure grid. Do not hide ACS calibration or GQ placement inside the existing AERMOD cell-attribute builder.
5. **Exposure outputs:** retain the current raw-population output and add clearly named ACS-adjusted household, GQ, and combined population results. Each result must identify its population basis in the manifest.

This separation fits the current code: preprocessing Step 1 resolves explicit population inputs in `pipeline/preprocessing/step1_collect_inputs.py`; `pipeline/preprocessing/step4_assign_raw_population_to_grid.py` assigns raw UrbanSim residents to the exposure grid; and `pipeline/preprocessing/step5_build_aermod_cell_attributes.py` builds AERMOD cell attributes from the grid and assigned population counts. Exposure inputs are prepared in `pipeline/workflow/step4_prepare_exposure.py`, and exposure aggregation is in `pipeline/postprocess/step5_plot_exposure.py`. ACS calibration and group-quarters allocation should remain separate preparation steps rather than being folded into either existing step.

## Group quarters allocation in preprocessing Step 8

The household-weight step targets the selected 2018 ACS 1-year county controls and has fixture tests. It was also run against the downloaded ACS tables and the `/2018/households` table in `/Volumes/HG40/Workspace/BREATHE/urbansim/model_data_2019.h5`; county household and household-person controls matched. The HDF5 input's block IDs and `county_id` are 2010-vintage Census geography.

The BREATHE bundle contains the baseline ACS tables, 2010 P42 block counts,
and January 2018 LUCA group quarters address counts. Set
`impacts.exposure.census_bundle` to the archive path visible to IMPACTS.
Step 1 verifies file hashes and unpacks the bundle into the shared input
cache. The configured bundle supplies the ACS tables offline.

Step 8 scales the ACS five-year tract estimates to each county's one-year
total, then uses positive P42 block counts as shares within each tract.
For the 27 tracts with a positive target and zero P42 residents, LUCA
identifies candidate blocks. A single candidate receives the tract total.
With multiple candidates, the step uses verified occupancy if available
for all blocks, or uses LUCA address counts as modeled weights. A future
bundle can supply verified occupancy as a checksummed CSV with
`block_geoid,residents` columns. Set its path in the bundle manifest's
`group_quarters_occupancy` field and include its source and vintage in
the corresponding file entry. Bundle v2 has no verified occupancy CSV.
[P42 documentation](https://www2.census.gov/programs-surveys/decennial/rdo/about/redistricting/gqsf.pdf) · [LUCA documentation](https://www2.census.gov/geo/pdfs/partnerships/luca/D-2032.pdf)

The block allocation table records each tract target, allocation method,
block share, and assigned grid cell. Placement uses the UrbanSim block
coordinates. Missing coordinates and points outside the grid stay in
the block totals and are reported in the manifest. They contribute no
population to the grid exposure results.
