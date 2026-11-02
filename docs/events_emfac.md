# Events-to-EMFAC emissions fallback

When no `skimsEmissions` input exists, Step 1 now calls
`prepare_emissions/event_emfac.py` through `from_events.build_staged_skims_from_events`.
It writes **unannualized grams per link / vehicle type / process**, in the explicit
pollutant-column format understood by the existing skims consumer. The normal
annualization and grid-allocation stages then run exactly once. Existing skims
inputs retain their existing execution path.

## Required inputs

- BEAM events with `type`, `mode`, `vehicle`, `vehicleType`, `departureTime`,
  `arrivalTime`, `links`, `linkTravelTime`; optional `PayloadWeightInKg`.
- The **matching** network, with link lengths and OSM road categories.
- County intersections regenerated for that network. The initial adapter has
  explicit county-name support for the nine Bay Area counties.
- Vehicle tables explicitly mapping the event `vehicleType` to `emfacId`.
  Native IDs are not guessed from income strings or vehicle labels. PRDUST needs
  curb mass; IDLEX needs an explicit `idleTimeFraction`.
- Native EMFAC `rates/dataset/emfacId=.../*.parquet` partitions, or the `dataset`
  directory itself. County, road type, and speed/soak dimensions are retained.
- Confirmed fleet sample fractions, annualization days, and release heights for
  the downstream annualization stage. **Do not reuse 0.005 for a different run
  without confirming its sample design.** The events conversion applies no
  fleet expansion, no annualization, no electric conversion, and no power load.

Preprocess accepts a direct iteration folder such as `beam-fc07-v260419-it3`,
including its `3.events.parquet`, without requiring `3.skimsEmissions.parquet`.
Use a new settings/output location and do not carry old prepared-skims caches
into a new-events run. This adapter does not prove that supplied intersections
or metadata belong to the new network: that remains an input provenance check.

## Calculation conventions

Process units and native null-component conventions were checked against
[BEAM VehicleEmissions.scala](https://github.com/LBNL-UCB-STI/beam/blob/develop/src/main/scala/beam/agentsim/agents/vehicles/VehicleEmissions.scala)
on 2026-09-25. This is a reconstruction from road PathTraversal events, not a
claim of bitwise reproduction of BEAM's internal emissions simulation.

| Process | Activity multiplied by the native factor |
| --- | --- |
| RUNEX, PMBW, PMTW, PTOEX | Link miles |
| PRDUST | Link miles × ((curb mass + payload kg) / 907.18474)^1.02 |
| RUNLOSS | Link travel hours |
| DIURN | Observed inter-trip parking hours |
| STREX, HOTSOAK | One observed restart / parking transition |
| IDLEX | Travel hours × configured idleTimeFraction; working-idle approximation |

Link lengths come from the supplied network, not from distributing the total
trip distance in proportion to travel time. Link clock advances along the trip.
Rates are selected **before** aggregating activity across speeds or soak times.
Numeric activity bins use the nearest center (lower center breaks a tie);
interval bins prefer containment and otherwise the nearest center, following
BEAM's nearest-bin fallback. Rates are specific to county and road type.

For county-crossing links, county factors are averaged by in-domain link length.
The result is a full-link factor; domain/grid fractions are applied only later.
This is a link-level approximation, not separate emission factors for every
subsegment. Sample expansion and grams-to-short-tons conversion occur only in
the existing annualization module.

## Validation and explicit limitations

- Unknown vehicle IDs, absent partitions, missing pollutant columns, missing
  county/road rate rows, invalid rates, and conflicting duplicate factors fail
  loudly. Identical/complementary partition rows are coalesced, not summed.
- **Null pollutant cells within a matched native process row** follow BEAM's
  zero-component convention. This is distinct from an absent row/partition;
  it should not be interpreted as empirical proof of zero emissions.
- Only on-road modes are selected. Process applicability comes from the mapped
  EMFAC partition. Transit/non-road emissions are not invented.
- First-trip cold-start/parking duration, final overnight parking and hotelling
  cannot be inferred reliably from PathTraversal alone and are not invented.
  Consecutive trips with zero dwell generate no parking/restart emissions.
  Initial/final omissions can understate parking/start processes: this is **not
  yet a validated full-day fleet inventory** without boundary assumptions.
- Missing payload values imply zero payload. Freight dust requires review of
  payload coverage. Explicit working-idle parameters are required when IDLEX
  exists; parking is not automatically treated as engine-on idling.
- All supplied road trips, including hours beyond 24, are retained. Audit records
  min/max times. A representative-day interpretation must be confirmed before
  treating the output as an annual inventory. No timezone shift is applied.
- Link-list errors, missing network links, negative/nonfinite durations and
  overlapping vehicle trips fail instead of being silently dropped.
- Output is written to a `.partial` file and promoted only on success. An
  interrupted partial file cannot be reused as valid skims.

`events_emfac_audit.json` records source, mapped types, EMFAC partitions,
time span, activity counts, units, and limitations. The fixture tests exercise
the actual Step 1 fallback through annualization and validate known totals.

## Current fc07 inputs

The local 271 EMFAC partitions can be read successfully. However, the supplied
`3.events.parquet` uses passenger/freight IDs such as `cv-L1-17000-2019` and
`ft-hdt-Dsl`, whereas the existing mapped vehicle tables use `paxcar-*` and
`frt-*`. This fix intentionally reports the missing explicit mapping instead
of borrowing an arbitrary emission factor. New-run vehicle metadata and sample
fractions still need confirmation before a production concentration run.
