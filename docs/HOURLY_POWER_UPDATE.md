# Hourly-only power generation — November 2, 2026

The clean release now supports one power-generation algorithm. Both the native
`impacts.power_generation` YAML stage and `scripts/run_hourly_power.py` call
`src/impacts/pipeline/workflow/hourly_power.py`.

## Calculation

1. Read the complete 8,760-hour grid-side charging-demand curve. Fleet expansion
   and charging losses must already be included; this stage adds no fleet multiplier.
2. Use the external BREATHE allocator with Cambium marginal technology and
   distribution losses, then proportional eGRID generator allocation. The current
   input set remains Cambium22 LowRECost p9 2040/eGRID2023 plus supplementary rates.
3. Calculate hourly plant generation and pollutant emissions, sum annual emissions,
   and check that hourly and annual totals agree.
4. Select emitting plants physically inside the nine Bay Area counties, check
   matrix coverage, and apply their annual emissions directly to the explicitly
   selected ISRM source layer. No local-to-local removal applies to power emissions.
5. Write annual/hourly tables, concentration Parquet/GPKG, plant coverage, QA and
   the power-only PM2.5 map. Existing traffic/exposure integration consumes this
   power concentration output.

This remains approximate plant allocation, not optimized grid dispatch. Outside-Bay
plants and unallocated generation are not reassigned into the Bay. Reference
replays use the previously approved Layer 0 approximation; Layer 1 requires a
verified elevated-layer matrix. EVI remains a charging comparison only.

## Removed and changed code

| Location | Change |
|---|---|
| `workflow/power_generation.py` | Deleted WA/fleet reference scaling, annual-energy ratio orchestration, and saved-concentration scaling. Removed the emissions scaling parameter from direct ISRM multiplication. Charging utilities and direct matrix/plot functions remain. |
| `workflow/hourly_power.py` | Shared hourly workflow, input validation, dispatch, geographic selection and ISRM outputs. |
| `config/settings.py`, `config/settings.yaml` | Replaced old fields with explicit hourly-demand, summary, allocator, county and layer inputs. Old fields raise migration errors, including when disabled. |
| `preprocessor.py`, `runner.py` | Resolve new input paths and run the hourly implementation in the native pipeline. |
| `postprocessor.py` | Plant markers default to plants selected by the current power run; optional override is `plant_markers_file`. |
| `scripts/run_hourly_power.py`, `continue_beam_charging_pipeline.py` | CLI and existing continuation wrappers delegate to the shared implementation. |
| `scripts/replay_charging_power.py` | Exercises the shared native orchestration with real LDV and freight inputs. |
| `tests/test_hourly_power.py` | Reject obsolete settings/invalid curves, enforce explicit inputs, check orchestration and unmet-energy guard, and demonstrate hourly shape sensitivity. |

Paths beginning with `workflow/` are relative to `src/impacts/pipeline/`.
Migration examples are in [REPRODUCING.md](REPRODUCING.md). Use a new output
directory; removing the old method does not relabel or recalculate archived outputs.
The original research checkout and saved results were not modified.

## Validation

The complete test suite passed: **397 passed, zero failed**, with external notebook
references supplied. Six tests of deleted legacy power behavior were removed and
eleven hourly workflow tests were added to the November 1 suite of 392.

Four full charging/power replays passed against the saved reference results:

| Case | Charging MWh/day | Peak MW | Peak hour | Maximum concentration difference, µg/m³ |
|---|---:|---:|---:|---:|
| LDV Home80 | 158.070889 | 15.704302 | 19 | 5.42e-20 |
| Freight BF-06EP | 1,409.340412 | 105.121024 | 16 | 1.08e-19 |
| Freight BF-1EP | 1,172.786960 | 92.533448 | 14 | 1.08e-19 |
| Freight HF-1EP | 1,640.127618 | 134.110560 | 13 | 0 |

Hourly dispatch, annual plant emissions and all 2,559 concentration cells per case
matched within floating-point tolerance. The three LDV+freight concentration sums
also matched. Charging vehicle/session/trip outputs were checked as part of replay.
The synthetic timing test moves the same 10 MWh between different technology hours
and verifies that emissions change; an annual-only scalar cannot pass this test.

The unchanged large traffic/exposure cases were not rerun for this power-only
refactor; their November 1 evidence remains in [VALIDATION.md](VALIDATION.md).
Machine-readable current results: [hourly_power_validation.json](hourly_power_validation.json).
