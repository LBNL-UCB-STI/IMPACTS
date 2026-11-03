import json
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import box, Point
from impacts.pipeline.postprocess import step9_human_exposure_webmap as webmap

from impacts.pipeline.postprocess.step9_human_exposure_webmap import (
    _absolute_maximum_composite,
    _maximum_composite,
    _replace_json_constant,
    _wrap_original_artifact,
)


def test_replace_json_constant_preserves_surrounding_template():
    template = '<script>const DATA = {"old":1};\nconst NEXT = true;</script>'
    result = _replace_json_constant(template, "DATA", {"new": [1, 2]})
    start = result.index("const DATA = ") + len("const DATA = ")
    value, length = json.JSONDecoder().raw_decode(result[start:])
    assert value == {"new": [1, 2]}
    assert result[start + length:].startswith(";\nconst NEXT")


def test_maximum_composite_uses_maximum_when_downsampling():
    values = np.array([1.0, 4.0, 2.0, 3.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    raster = _maximum_composite(values, rows, cols, (2, 2), max_dimension=1)
    assert raster.shape == (1, 1)
    assert raster[0, 0] == 4.0


def test_absolute_maximum_composite_preserves_sign_of_strongest_difference():
    values = np.array([-5.0, 4.0, 2.0, -3.0])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([0, 1, 0, 1])
    raster = _absolute_maximum_composite(values, rows, cols, (2, 2), max_dimension=1)
    assert raster.shape == (1, 1)
    assert raster[0, 0] == -5.0


def test_original_artifact_wrapper_restores_fixed_width_iframe():
    inner = '<!doctype html><html><body data-label="A & B"></body></html>'
    wrapped = _wrap_original_artifact(inner)

    class Parser(HTMLParser):
        srcdoc = None

        def handle_starttag(self, tag, attrs):
            if tag == "iframe":
                self.srcdoc = dict(attrs)["srcdoc"]

    parser = Parser()
    parser.feed(wrapped)
    assert parser.srcdoc == inner
    assert "max-width:736px" in wrapped
    assert 'sandbox="allow-scripts"' in wrapped


@pytest.mark.parametrize('power', [0., 5.])
def test_grid_method_difference_excludes_power_and_preserves_final(tmp_path, monkeypatch, power):
    grid = gpd.GeoDataFrame({
        'inmap_only_TotalPM25': [3., 3.], 'TotalPM25': [2.+power, 4.+power],
        'power_TotalPM25': [power, power], 'inmap_only_BC': [1., 1.], 'BC': [2., 2.],
        'inmap_only_NO2': [5., 5.], 'NO2': [6., 6.],
    }, geometry=[box(500000, 4100000, 500100, 4100100), box(500100, 4100000, 500200, 4100100)], crs=26910)
    path = tmp_path / 'grid.parquet'
    grid.to_parquet(path)
    monkeypatch.setattr(webmap, '_png_data_url', lambda values, cap: values.tolist())
    monkeypatch.setattr(webmap, '_signed_png_data_url', lambda values, cap: values.tolist())
    raw = webmap._raw_rasters(str(path))
    np.testing.assert_allclose(np.ravel(raw['rasters']['dpm']), [1., -1.])
    np.testing.assert_allclose(np.ravel(raw['rasters']['trafficpm']), [2., 4.])
    np.testing.assert_allclose(np.ravel(raw['rasters']['aepm']), [2.+power, 4.+power])
    assert raw['comparisonScope'] == 'traffic_only'
    assert raw['powerIncludedInFinal'] == bool(power)


@pytest.mark.parametrize('kind', ['tract', 'school', 'ltc'])
@pytest.mark.parametrize('explicit_traffic', [True, False])
def test_aggregate_and_facility_differences_use_traffic(tmp_path, kind, explicit_traffic):
    suffix = '_PW' if kind == 'tract' else ''
    data = {'CES_Group': ['0.00-0.25'], 'ModelPopulation': [100.],
            'SchoolName': ['school'], 'FacilityName': ['ltc'], 'Enrollment': [10.], 'TotalLicensedBeds': [20.]}
    for name, value in [('Final_TotalPM25', 7.), ('InMAP_TotalPM25', 3.), ('Traffic_TotalPM25', 2.),
                        ('Power_TotalPM25', 5.), ('Final_BC', 2.), ('InMAP_BC', 1.),
                        ('Final_NO2', 6.), ('InMAP_NO2', 5.)]:
        if name != 'Traffic_TotalPM25' or explicit_traffic:
            data[name+suffix] = [value]
    geometry = box(-122.5, 37.5, -122.4, 37.6) if kind == 'tract' else Point(-122.45, 37.55)
    frame = gpd.GeoDataFrame(data, geometry=[geometry], crs=4326)
    path = tmp_path / 'input.parquet'
    frame.to_parquet(path)
    if kind == 'tract':
        props = webmap._tract_data(str(path), require_component=True)['features'][0]['properties']
    else:
        props = webmap._facility_data(str(path), name_field='SchoolName' if kind == 'school' else 'FacilityName',
            weight_field='Enrollment' if kind == 'school' else 'TotalLicensedBeds', require_component=True)[0]
    assert props['dpm'] == 1.
    assert props['aepm'] == 7.
    assert props['trafficpm'] == 2.
    assert props['dbc'] == props['dno2'] == -1.


def test_missing_or_inconsistent_power_components_fail_instead_of_mixing_scopes():
    frame = pd.DataFrame({'final': [7.]})
    with pytest.raises(ValueError, match='Regenerate human-exposure'):
        webmap._traffic_pm25(frame, total='final', traffic='traffic', power='power', require_component=True)
    frame['traffic'], frame['power'] = 3., 5.
    with pytest.raises(ValueError, match='Inconsistent'):
        webmap._traffic_pm25(frame, total='final', traffic='traffic', power='power')


def test_legacy_traffic_only_products_remain_compatible():
    frame = pd.DataFrame({'final': [2., 4.]})
    np.testing.assert_allclose(webmap._traffic_pm25(frame, total='final', traffic='traffic', power='power'), [2., 4.])


def test_template_facility_delta_uses_traffic_even_on_final_layer():
    template = webmap._TEMPLATE_PATH.read_text(encoding='utf-8')
    assert 'aermod: "aepm"' not in template
    assert 'value="trafficpm"' in template
    assert 'Final PM2.5 (traffic + power)' in template
    assert 'excluded from every method-difference layer' in template
