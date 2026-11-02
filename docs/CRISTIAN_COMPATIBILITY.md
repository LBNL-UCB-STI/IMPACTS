# Restored Cristian charging compatibility

Module paths have been consolidated; see [Charging module organization](CHARGING_MODULES.md).

The preferred LDV charging entry point is now `scripts/run_ldv_charging.py`.
It executes the original functions from the supplied collaborator notebooks
and reads their literal parameters. The notebooks are external, read-only inputs
and are identified by SHA-256 in each output. No original notebook file is edited.

## Preserved model

For the supplied reference notebooks:

- Traction consumption is 0.17 kWh/km, multiplied by 0.1 for batteries below 30 kWh.
- Initial SOC uses the original mileage-dependent beta-distribution function.
- Original critical-charge preassignment, P1=0.45, P2=0.04, SH=0.8, charger
  weights, and L1/L2/DC powers of 1.9/7.2/50 kW are retained.
- Charging/SOC integration uses the original `fill_time_bins` function.
- Vehicle filtering uses post-charge SOC >=2 kWh in every modeled bin, as in
  the notebook. Before-charge physical feasibility is not silently substituted.
- No electric-first/fuel-switching model, periodic SOC iteration, extra charger
  efficiency, or EVI demand fitting is added.

The only sampling compatibility adjustment restores pandas 2/NumPy weighted
sampling semantics under pandas 3. It is scoped to a serial call and restores
the original method and NumPy RNG state afterward. This is not a thread-safe
service API. Notebook I/O and plotting cells are not executed.

## Inputs and execution

Prefer the original prepared hourly activity table, containing vehicle IDs,
PTIDs, batteries, travel distances, times, activity types, charger counts, and
net grid load. The code validates required fields rather than inventing missing
charger or grid inputs.

```bash
python scripts/run_ldv_charging.py --activity-file D:/inputs/expanded_activity.csv --notebooks D:/external/chargingcode --fleet-scale 10 --output D:/results/cristian-original
```

The notebook folder must contain `GenerateVehicleActivity.ipynb` and
`AssignChargingEvents.ipynb`. For an audited one-row-per-stop table, explicitly
add `--input-layout stops`. This expands parking into hourly intervals, retaining
distance only in the first interval. It does not infer batteries, home locations,
charger availability, or grid load. A table prepared using earlier proxy inputs
still carries those proxy assumptions; algorithm compatibility does not make
the input data observationally validated.

The native power workflow continues to accept an explicitly prepared hourly
load; it does not independently select or run a charging model. To export its
8,760-hour interface, add `--annualize-representative-day`. This explicitly repeats
the retained single-day curve 365 times without forcing SOC equilibrium.
Exclusions and SOC drift remain in `beam_charging_summary.json`; a failed
readiness check is not automatically overridden. Default execution generates
charging results only, without annualization, dispatch, or concentrations.

Outputs include `charging_profiles.parquet`, the full `charging_activity_audit.parquet`,
`vehicle_soc_audit.csv`, `charging_curve.csv`, `charging_curve.png`, a daily demand
Parquet, and `beam_charging_summary.json`. The plot includes a display-only hour 24
point; accounting uses 24 hourly bins. All outputs require a fresh directory.

## Restored real-data run and compatibility checks

The restored run reused the existing audited stop table derived from Cristian's
existing-EV events, with 10x expansion. It did not re-extract the raw events.
There were 1,543 modeled vehicles, 1,539 retained and four filtered out under
the original post-charge rule. The restored curve is 87.476848 MWh/day with
a 7.060442 MW peak at hour 19. PHEV and BEV labels in this entry point follow
the original battery-capacity proxy and are named accordingly.

Against the previously verified optimized compatibility adapter on the same
real input, stop choices and charger powers matched exactly, retained vehicles
matched, and the largest hourly energy difference was 1.14e-12 sample kWh.
Maximum per-stop energy and SOC differences were 1.42e-14 and 2.84e-14 kWh.
The supplied notebook files were not changed.

The retained sample gained 2,459.118353 kWh of SOC during the simulated day,
compared with 6,288.566454 kWh of traction consumption. This is a single-day
initialization effect, not a periodic equilibrium. It is reported without
changing the original algorithm. The restored output is consequently not
automatically marked ready for full-fleet annual power calculations.

Five new tests cover original/optimized parity with both powertrains, conservative
hour-bin expansion, removal of stale input energy proxies, restoration of sampling
state, and reading literal settings from a notebook copy. The full suite now
passes 402 tests with external reference notebooks supplied.

## Status of previous designed charging cases

Home80 periodic/electric-first modules and explicitly named sensitivity/replay
scripts remain archived research alternatives. They are not the recommended
compatible LDV model. `replay_charging_power.py` and
`combine_ldv_freight_power.py` reproduce their documented designed Home80 cases;
they must not be used to claim original-notebook results. Their saved curves,
power concentrations, and validation evidence remain unchanged and are labeled
as historical cases. Use the restored entry point for new compatible charging.
