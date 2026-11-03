# Traffic-only method differences in the exposure web map

The previous PM2.5 map difference subtracted `TotalPM25`, which can include
power generation, from the traffic-only `inmap_only_TotalPM25`. CES tract and
facility layers repeated this scope mismatch. This was a display/comparison
defect; it did not change the underlying concentration or emission products.

The map now calculates:

```
hybrid_traffic = final_total - power_total
method_difference = inmap_only_traffic - hybrid_traffic
```

An existing explicit traffic column is preferred and checked against
`final_total - power_total` when both components are available. Power is never
subtracted twice. Grid differences are calculated before raster compositing.
CES tracts use the population-weighted traffic field; schools and LTC facilities
use their assigned grid's traffic concentration. Facility delta symbols and
tooltips also use the traffic pair, even when the selected background is the
final concentration layer. Differences are computed before display rounding.

The PM2.5 selector distinguishes three absolute layers:

- Traffic InMAP-only (`impm`).
- Traffic AERMOD-enhanced (`trafficpm`).
- Final traffic plus power (`aepm`).

The signed `dpm` layer is explicitly a traffic method difference. BC and NO2
continue to compare their traffic fields; no power BC or direct NO2 contribution
is manufactured. The original map layout, plant markers and zoom behavior are
retained. All PM2.5 absolute layers share a display color cap.

Legacy traffic-only files without component fields remain readable. If the raw
grid includes power but a tract/facility input has neither an explicit traffic
field nor a power field, web-map generation fails with a request to regenerate
the human-exposure products. It must not silently assume that a final field is
traffic-only. Inconsistent explicit component fields also fail validation.

`webmap_comparison.json` records the formula, layer scopes and input paths.
The same scope information is embedded in the web map. No existing concentration
columns are renamed or overwritten by the map generator.

## Changed files

- `src/impacts/pipeline/postprocess/step9_human_exposure_webmap.py`
- `src/impacts/pipeline/postprocess/assets/human_exposure_webmap.html`
- `tests/test_human_exposure_webmap.py`

The fixes are synchronized to the research and upload checkouts. Old generated
HTML files are not automatically rewritten; rerun Step 9, or the human-exposure
postprocessing that invokes it, to refresh them. Other historical tables and
static comparison figures were not regenerated in this patch.

## Verification on 2026-11-02

Regression cases cover nonzero power, no power, tract/school/LTC fields, explicit
traffic and subtraction fallback, inconsistent/missing components, and the
frontend's facility-delta field selection. The upload suite passed **441 tests**.
The research suite passed **438**, with its same six pre-existing terminal
contract failures; no new failures were introduced.

A separate verification map was generated from the existing
`events-fc07-baseline-native-ev10x-20260928` concentration/exposure products,
under `postprocess/human_exposure/web-method-comparison-fixed-20261102`.
It retains that historical scenario and is not a new 2018 electricity result.

All 6,957,973 input grid rows satisfy the corrected identity. The maximum removed
power bias in this particular grid is 0.0001405165161 micrograms/m3. Embedded
data were checked for 1,577 CES tracts, 1,713 mapped schools and 226 LTC facilities.
The generated payload includes all ten raster layers, including the new traffic
hybrid layer, and retains the final combined PM2.5 layer. A browser was unavailable,
so interactive visual inspection is not claimed. Source concentration and
exposure files were not changed. The separate 2018 concentration run remains
pending its missing ISRM source row; no partial 2018 result was generated.
