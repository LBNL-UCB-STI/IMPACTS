"""Default settings must retain nonlocal traffic when adding local AERMOD."""
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import yaml
from shapely.geometry import box

from impacts.config.settings import Exposure
from impacts.manifest.schema import PipelineConfig
from impacts.pipeline.workflow.step4_prepare_exposure import _build_full_exposure_grid
from test_prepare_exposure import _pipeline


@pytest.mark.parametrize('path', [None, 'src/impacts/config/settings.yaml',
                                 'examples/pilates/settings.yaml', 'examples/pilates/hpc-settings.yaml'])
def test_default_and_shipped_settings_select_additive(path):
    if path is None:
        exposure = Exposure.from_dict({})
    else:
        root = Path(__file__).resolve().parents[1]
        payload = yaml.safe_load((root / path).read_text(encoding="utf-8"))
        exposure = Exposure.from_dict(payload["impacts"].get("exposure", {}))
    assert exposure.primary_pm25_integration_strategy == 'additive_inmap_aermod'


@pytest.mark.parametrize('has_aermod', [True, False])
def test_omitted_strategy_adds_nonlocal_local_and_power_without_indicator_gate(tmp_path, has_aermod):
    grid = gpd.GeoDataFrame(dict(aermod_cell_id=[1], inmap_cell_id=[2]),
                            geometry=[box(0, 0, 100, 100)], crs=26910)
    path = tmp_path / 'grid.parquet'
    grid.to_parquet(path)
    payload = _pipeline(tmp_path, path).to_dict()
    payload.pop('primary_pm25_integration_strategy')
    pipeline = PipelineConfig.from_dict(payload)
    im = gpd.GeoDataFrame(dict(inmap_cell_id=[2], inmap_PrimaryPM25=[2.],
        inmap_SecondaryPM25=[5.], inmap_BC=[.2], inmap_NO2=[2.],
        inmap_only_PrimaryPM25=[20.], inmap_only_TotalPM25=[25.],
        inmap_only_BC=[2.], inmap_only_NO2=[20.]), geometry=grid.geometry, crs=grid.crs)
    ae = gpd.GeoDataFrame(dict(aermod_cell_id=[1], aermod_PrimaryPM25=[3.],
        aermod_BC=[.3], aermod_NO2=[3.], has_aermod_primarypm25=[has_aermod],
        has_aermod_bc=[has_aermod], has_aermod_no2=[has_aermod]),
        geometry=grid.geometry, crs=grid.crs)
    power = pd.DataFrame(dict(inmap_cell_id=[2], power_PrimaryPM25=[.4], power_SecondaryPM25=[.6]))
    result = _build_full_exposure_grid(pipeline=pipeline, prepared_inmap=im,
        prepared_aermod=ae, prepared_power=power, working_dir=tmp_path)
    # 2 nonlocal + 3 local + 0.4 plant primary; 5 traffic secondary + 0.6 plant secondary.
    np.testing.assert_allclose(result[['PrimaryPM25','SecondaryPM25','TotalPM25','BC','NO2']],
                               [[5.4, 5.6, 11., .5, 5.]])
    np.testing.assert_allclose(result.aermod_PrimaryPM25_scale_factor, [1.])
    np.testing.assert_allclose(result.inmap_only_TotalPM25, [25.])
