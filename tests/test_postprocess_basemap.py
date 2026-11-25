from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

from impacts.config.settings import Analysis
from impacts.pipeline.postprocess import _common


def test_skip_web_tiles_keeps_scale_bar(monkeypatch) -> None:
    add_basemap = Mock()
    contextily = SimpleNamespace(
        add_basemap=add_basemap,
        providers=SimpleNamespace(CartoDB=SimpleNamespace(Voyager=object())),
    )
    add_scalebar = Mock()
    monkeypatch.setitem(sys.modules, "contextily", contextily)
    monkeypatch.setattr(_common, "_add_scalebar", add_scalebar)

    ax = object()
    _common._add_basemap(ax, skip_web_tiles=True)

    add_basemap.assert_not_called()
    add_scalebar.assert_called_once_with(ax)


def test_analysis_parses_skip_web_tiles() -> None:
    assert Analysis.from_dict({"skip_web_tiles": True}).skip_web_tiles is True
