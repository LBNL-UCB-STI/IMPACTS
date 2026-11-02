# Hybrid integration default correction (2026-11-02)

The inherited default selected `impute_inmap_primary_in_aermod_domain`, which
uses AERMOD primary PM2.5 where its indicator is true and InMAP elsewhere.
That is inconsistent with the current local/nonlocal hybrid method because it
omits the nonlocal InMAP primary contribution at AERMOD-supported receptors.

The Python fallback in `src/impacts/config/defaults.py`, built-in settings YAML,
and both shipped PILATES examples now select `additive_inmap_aermod`.
The intended primary traffic calculation is nonlocal InMAP + local AERMOD,
without an indicator gate or AERMOD budget scaling. Secondary InMAP is retained,
and supplied plant PM components are added after traffic integration. Full
InMAP-only diagnostic columns are unchanged. No emission factors, ISRM masks,
AERMOD cutoffs, charging parameters or plant allocation formulas changed here.

## Provenance and existing runs

The manifest already records the effective choice at
`pipeline.primary_pm25_integration_strategy`; no duplicate schema field was
necessary. The runner now logs it explicitly, and regression tests check the
serialized manifest for omitted, explicit additive and explicit legacy choices.

Changing a default does not rewrite saved YAML configurations, preprocess
manifests, final outputs, or historical results. A resumed run uses its stored
preprocess configuration: an explicit old strategy continues to select that
strategy. To produce a new hybrid result, preprocess with the corrected settings
in a fresh output directory, or deliberately migrate a copy of the run inputs.
For historical runs missing the strategy field, inspect the original settings
before replay; the fallback has changed. Existing representative replay drivers
already explicitly select additive integration, so this finding alone does not
show that those saved replay results used the wrong strategy.

## Validation scope

New tests cover the Python exposure fallback and shipped YAML strategy values;
an omitted manifest field; direct addition with either indicator value; plant
PM addition; preservation of the full traffic-only reference; and effective
strategy serialization through the runner. Existing tests continue to cover
explicit legacy and InMAP-only choices. Test and replay agreement demonstrate
implementation consistency, not independent scientific or observational model
validation. No historical concentration product was recomputed in this fix.

The research checkout has separate pre-existing settings/profile contract
failures. Its old HPC file also contains an unsupported activity `folder` key;
this is unrelated to the exposure strategy and was not migrated here. The new
strategy-value test checks the exposure subsection; complete configuration-loader
coverage remains in the existing terminal tests. The GitHub HPC configuration
loads with the corrected strategy.

Final checks: GitHub checkout 410 passed; research checkout 396 passed with
the same six pre-existing terminal-contract failures. No new failure remains.
