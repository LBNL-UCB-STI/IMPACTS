# Changelog

## 2026-05-14

### Changed

- Pinned `osm-chordify` to `0.2.4` in `setup.cfg` and `hpc/requirements-hpc.txt`.
- Added `CLAUDE.md` project-scope rules to keep all work inside `impacts/` unless explicitly approved by the user.

### Migrated

- Updated the `osm-chordify` integration to the `0.2.4` release schema.
- Replaced old prefixed edge-column assumptions with the flat `0.2.4` contract where edge attributes pass through unchanged.
- Updated zone metric handling to use `{label}_proportion` and `{label}_link_length_m`.
- Switched step 3 intersections to `prefilter_zones_to_bbox`.
- Updated settings mapping columns to use `linkId` and `proportion`.
- Updated downstream allocation code in `prepare_emissions_from_skims.py`, `prepare_emissions_from_events.py`, and `step1_process_emissions.py` to consume the maintained schema.

### Removed

- Removed support for the old `0.2.3`-era duplicated and stacked prefixes such as `edge_edge_*`, `inmap_inmap_*`, and `aermod_aermod_*`.
- Removed runtime dependence on legacy intersection columns such as `*_zone_edge_proportion`, `*_zone_link_length_m`, and `*_edge_link_length_m`.
- Removed the county post-processing repair path that translated `county_COUNTYFP` after intersection.
- Tightened zone-schema normalization so step 3 now requires the maintained canonical schema instead of tolerating partial or legacy-shaped frames.

### Tests

- Updated tests to assert only the `0.2.4` flat schema in allocation and county-conservation paths.
- Kept explicit regression checks that verify old duplicated-prefix bugs do not return.
- Verified the migration with:

```bash
pytest -q tests/test_prepare_emissions_from_skims.py tests/test_step1_process_emissions.py tests/test_step3_integrate_grids.py tests/test_osm_chordify_intersect.py
```

- Result: `21 passed`
