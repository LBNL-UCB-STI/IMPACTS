# Charging module organization

The charging implementation is grouped under `src/impacts/pipeline/workflow/charging/`.
This is a structural refactor: no consumption rates, battery sizes, SOC rules,
charging preferences, fleet multipliers, or load-aggregation algorithms changed.

| Module | Responsibility | Entry points / consumers |
| --- | --- | --- |
| `ldv_charging.py` | Original notebook loading, existing-EV activity preparation, original LDV execution, and parity-tested optimized helpers | `scripts/run_ldv_charging.py` (preferred original-compatible entry); `scripts/run_existing_ev_charging_code.py` (historical input/optimized workflow) |
| `ldv_home_sensitivity.py` | Shared LDV schedules, periodic SOC, home-preference experiments, and PHEV sensitivities | Home80/Home90 comparison scripts and historical `scripts/replay_charging_power.py` |
| `beam_sessions.py` | Reconstruct load from charging sessions already recorded in BEAM events | `scripts/run_native_ev_baseline.py` |
| `truck_opportunity.py` | Truck-class energy and battery proxies with parking-based opportunity charging | `scripts/rebuild_truck_charging.py` |

## Original-compatible LDV workflow

`scripts/run_ldv_charging.py` calls `charging.ldv_charging.run_ldv_charging`.
Provide a prepared hourly activity table, or explicitly select an audited stop
table with `--input-layout stops`, and the external collaborator notebooks.
The entry does not automatically extract raw events or select the experimental
home-priority model. It generates charging results; annualization requires the
existing explicit option and retains readiness/SOC-drift checks.

`prepare_existing_ev_stops` adapts existing EV trips with battery metadata and
activity labels. `prepare_passenger_stops` is the older hypothetical-passenger
helper; its proxy assumptions are retained and are not substituted for actual
battery metadata. `assign_stops_indexed` and `fill_stops` remain for documented
compatibility tests and historical scripts, not a second default model.

## Migration from the previous layout

| Previous module in `workflow/` | New module in `workflow/charging/` |
| --- | --- |
| `cristian_charging.py`, `notebook_charging.py`, `existing_ev_charging.py` | `ldv_charging.py` |
| `home_priority_charging.py`, `phev_home_charging.py` | `ldv_home_sensitivity.py` |
| `native_ev_charging.py` | `beam_sessions.py` |
| `truck_charging.py` | `truck_opportunity.py` |

For example, import `run_ldv_charging` from
`impacts.pipeline.workflow.charging.ldv_charging`. Function arguments are unchanged. The preferred entry is now
`scripts/run_ldv_charging.py` and the function is `run_ldv_charging`; update
commands/imports that previously used the Cristian-named entry. Old module files are removed; external notebooks or
scripts importing those Python paths must update their imports using this table.
All in-repository consumers are updated. Historical saved outputs/manifests are
not rewritten, so their recorded source paths remain descriptions of those runs.

The home-sensitivity module is deliberately separate: its periodic SOC and
electric-first PHEV assumptions must not be presented as original Cristian code.
The collaborator truck charging-event analysis can supply load directly without
running `truck_opportunity.py`. Power-plant dispatch and concentration workflows
remain separate and accept the explicitly selected hourly load.

## Scope across the two worktrees

The GitHub worktree already had the restored Cristian entry. The research
worktree receives that same entry and its five existing tests in this change.
Charging implementations are synchronized; unrelated traffic, power, data,
configuration, and historical output differences are preserved. Tests named
after the former modules retain their filenames to preserve review history.
Previously exported ZIP files are snapshots and are not updated by this refactor.

## Refactor validation (2026-11-02)

The four implementations are byte-identical between the research and GitHub
worktrees. AST comparisons confirm all 22 functions/classes and module-level
constants are unchanged. Git diff whitespace checks and Python parsing pass.

| Worktree | Before | After |
| --- | --- | --- |
| GitHub upload checkout | 402 passed | 402 passed |
| Research checkout | 383 passed, 6 failed | 388 passed, the same 6 failed |

The five additional research tests are the existing restored-Cristian tests
copied from the upload checkout. The six pre-existing research failures are in
`test_terminal_contract.py` (settings/profile/entry-point contracts), outside
the charging refactor. No failure was introduced or suppressed.

An existing-EV prepared-stop dataset was replayed with the unchanged original
notebooks, seed and 10x multiplier. The hourly curve, vehicle SOC audit, retained
charging profiles and complete activity table match the pre-refactor restored
run exactly: 87.476848 MWh/day; 7.060442 MW peak at hour 19; 1,539 retained
vehicles. Original SOC drift/readiness findings remain; the replay did not
annualize, dispatch power plants, or recompute concentrations.

See [machine-readable validation](charging_refactor_validation.json). To rerun
the suite, set `IMPACTS_CHARGING_NOTEBOOKS` to the trusted original notebook
directory and run `python -m pytest tests -o addopts="" -q` from the worktree.

The temporary module name `ldv_cristian.py` has been replaced by
`ldv_charging.py`. Cristian attribution and model provenance remain documented.
