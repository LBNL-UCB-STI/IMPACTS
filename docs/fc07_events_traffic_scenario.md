# fc07 events: traffic-only concentration scenario

Entry point: `scripts/run_events_traffic_pipeline.py --output <independent-directory> --stage <stage>`.
Stages: setup, mapping, activity, emissions, allocate, inmap, aermod, exposure.

This scenario uses `chargingcode/beam-fc07-v260419-it3/3.events.parquet` and its
network. It does not use charging demand or power-generation emissions.

## Fleet and activity assumptions

- No extra sample expansion: population, freight and transit sample fractions
  are 1. Results represent the observed event fleet, not a verified full regional fleet.
- Passenger EMFAC composition is a proxy from the baseline vehicle population,
  conditioned on powertrain and the original vehicle metadata's income band.
  Vehicle-count shares approximate VMT shares within each group. The event ID's
  year token is not treated as an individual EMFAC model year.
- Freight retains all existing `sourceVehicleTypeId` category/year combinations.
  Driving uses normalized VMT priors; parking/start uses population priors.
- Native local 2018 EMFAC factors retain county, process, speed/soak dependence.
  Missing partitions may only be recovered from exact category/fuel/year matches
  in the original grouped factor tables, into the scenario's own inputs folder.
- The representative day is simulation seconds 0 through 86400. Crossing trips
  are clipped link by link. Existing category-specific operation days annualize
  activity. No separate hourly-to-annual fleet multiplier is applied.
- Starts/parking are inferred only between observed trips. Unobserved initial
  cold starts, overnight dwell, and hotelling are not invented. PHEV factors are
  baseline mixture factors, not event-specific engine-on/off estimates.
- BEAM PathTraversal weight is gross vehicle weight (curb plus payload). Road
  dust uses this weight once. Working idle uses mapped idleTimeFraction.
- Missing-route trips and zero-duration link activity are reported, not assigned
  invented locations/speeds. Overlapping trip pairs produce no inferred dwell.

## Dispersion contract

Reuse spatial intersections only after verifying new and baseline networks
have identical link IDs, geometry, lengths and road categories. Concentrations
and emissions must be regenerated; no previous scenario concentration is reused.

The traffic ISRM source layer is ground level (0). Compact stores must translate
global source grid IDs through `source_ids`. Missing nonzero-emission sources
raise an error; they must never be silently dropped or replaced with zero.

The maintained local-removal mask uses a 4500-m circle about each source-cell
centroid and suppresses receptor cells whose polygon is more than 50% covered.
The integrated branch removes this contribution for primary PM2.5, BC and NO2,
then adds AERMOD directly. Secondary PM2.5 is unchanged. InMAP-only fields retain
the full unsuppressed primary response for comparison.

## Audit products

`scenario.json`, `network_validation.json`, `inputs/vehicle_emfac_mixture.csv`,
`activity_audit.json`, and `emissions_audit.json` record the assumptions and
intermediate totals. A pipeline manifest and final concentration files exist
only when the corresponding stages have actually completed.
