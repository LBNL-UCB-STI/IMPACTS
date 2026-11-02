# Power result isolation in repeated runs (2026-11-02)

## Confirmed defect

The runner used to initialize its power concentration path from a previous
pipeline manifest without checking the current enabled flag. Exposure-only
runs could therefore consume an earlier power result after power was disabled,
or reuse a result from a different load/configuration. The new regression tests
reproduced this behavior before the fix (nine failures in ten initial cases).

## Current policy

Automatic reuse of previous power concentrations is disabled. The runner starts
with no power path and clears all old `power_generation_*` output references.
When power is enabled and either InMAP or exposure is requested, it calls the
current power implementation with the current preprocess configuration. An
exposure-only run now recomputes power too. A combined InMAP/exposure invocation
computes power once. Failure or a missing current result stops execution; there
is no fallback to a previous power concentration.

When a current or previous run involves power, old combined-concentration and
population/exposure references are also invalidated. Successfully rerunning
exposure publishes new paths. Skipping exposure leaves those references null.
The postprocessor rejects explicitly invalidated combined results instead of
finding and plotting an old file at the conventional output location.

The manifest records `execution.power_generation` with the enabled flag, status
(`disabled`, `not_run`, or `computed`), reuse policy and invalidated references.
The effective power settings remain under `pipeline.power_generation`.
Old physical files are retained. Null references do not mean those files were
deleted, and aborted runs do not publish a successful new pipeline manifest.

This deliberately avoids claiming that matching filenames or configuration
strings establish cache validity. No automatic cache reuse remains to validate.
Future caching would require a versioned fingerprint of the load, summary,
plant/Cambium inputs, ISRM, grid, model code and relevant settings. The current
tradeoff is additional runtime for repeated exposure calculations.

## Configuration and validation boundaries

Stage commands use the saved preprocess manifest. Editing a standalone settings
YAML does not retroactively change that manifest; preprocess the intended
configuration before running the new scenario.

Regression coverage includes enabled-to-disabled transitions, contradictory
old paths/flags, exposure-only execution, changed demand contents at the same
path, changed input paths, changed layer, skipped exposure, failure without
fallback, and prevention of stale postprocess plots. Power calculation is
stubbed in these runner tests: they verify orchestration and isolation, not
plant-dispatch physics. Historical concentration products were not recomputed.

The same isolation policy is applied to both worktrees. Their previously
existing power implementations are otherwise unchanged; this patch does not
migrate the research checkout's older backend to the upload checkout's hourly
backend. Scientific parameters and charging algorithms are unchanged.

Final suite: upload checkout 422 passed; research checkout 408 passed with
the same six pre-existing terminal-contract failures. All 12 new isolation
tests pass in both checkouts.
