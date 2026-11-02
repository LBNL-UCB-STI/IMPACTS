import json
from html.parser import HTMLParser

import numpy as np

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
