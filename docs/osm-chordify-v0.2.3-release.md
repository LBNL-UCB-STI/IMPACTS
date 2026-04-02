## v0.2.3

This release fixes chained intersection schema behavior in `osm_chordify.osm.intersect` and strengthens regression coverage around labeled grid workflows.

### Fixed
- Stopped doubling semantic zone prefixes during chained intersections.
- Preserved labeled zone columns such as `inmap_*` and `aermod_*` instead of producing names like `inmap_inmap_*` or `aermod_aermod_*`.
- Kept downstream chained edge columns stable in repeated intersections, avoiding names like `edge_inmap_inmap_*`.

### Improved
- Removed the warning-prone void-row append path used for bbox-prefiltered no-hit zones.
- Added focused regression coverage for chained `inmap -> aermod` intersections.
- Expanded intersect-module coverage across save, reprojection, cache-miss, collision, and error branches.

### Validation
- `tests/test_intersect.py`: 41 passed
- `osm_chordify.osm.intersect` coverage: 86%
