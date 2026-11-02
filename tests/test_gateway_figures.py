import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from plot_gateway_products import make_layout, load_values
from impacts.pipeline.postprocess._common import _grid_raster_layout, _rasterize_grid_values


def test_streamed_layout_matches_native_with_holes():
    grid = gpd.GeoDataFrame(geometry=[box(100, 300, 200, 400), box(400, 500, 500, 600), box(200, 300, 300, 400)])
    native = _grid_raster_layout(grid)
    streamed = make_layout(grid.bounds.to_numpy())
    assert native.shape == streamed.shape
    assert native.extent == streamed.extent
    np.testing.assert_array_equal(native.rows, streamed.rows)
    np.testing.assert_array_equal(native.cols, streamed.cols)
    np.testing.assert_array_equal(_rasterize_grid_values([1, 2, 3], native), _rasterize_grid_values([1, 2, 3], streamed))


def test_layout_rejects_wrong_resolution_and_duplicates():
    with pytest.raises(ValueError, match='100-m'):
        make_layout([[0, 0, 200, 100]])
    with pytest.raises(ValueError, match='Duplicate'):
        make_layout([[0, 0, 100, 100], [0, 0, 100, 100]])


def test_loading_preserves_values_and_checks_alignment(tmp_path):
    import pandas as pd
    d = pd.DataFrame(dict(aermod_cell_id=[10, 20, 30], TotalPM25=[6., 12., 18.],
        PrimaryPM25=[3., 6., 9.], SecondaryPM25=[3., 6., 9.], BC=[.1, .2, .3], NO2=[1., 2., 3.],
        aermod_PrimaryPM25_scaled=[1., 2., 3.], inmap_PrimaryPM25=[2., 4., 6.], inmap_SecondaryPM25=[3., 6., 9.]))
    path = tmp_path/'data.parquet'
    d.to_parquet(path, index=False)
    actual = load_values(path, np.array([True, False, True]), np.array([10, 20, 30]))
    np.testing.assert_array_equal(actual.NO2, [1., 3.])
    np.testing.assert_array_equal(actual.PM25_ThreeComponent_Total, [6., 18.])
    with pytest.raises(AssertionError):
        load_values(path, np.array([True]*3), np.array([30, 20, 10]))
