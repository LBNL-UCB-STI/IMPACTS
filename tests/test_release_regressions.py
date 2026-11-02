"""Scientific branch regressions discovered during release review."""
import geopandas as gpd
import numpy as np
from shapely.geometry import box

from test_prepare_exposure import _pipeline
from impacts.pipeline.workflow.step4_prepare_exposure import (
    _build_full_exposure_grid, _scale_aermod_primary_to_inmap_domain_primary,
)


def test_inmap_only_uses_unmodified_surface_even_when_aermod_and_power_present(tmp_path):
    geom = [box(0, 0, 100, 100)]
    grid = gpd.GeoDataFrame(dict(aermod_cell_id=[1], inmap_cell_id=[2]), geometry=geom, crs=26910)
    path = tmp_path / 'grid.parquet'
    grid.to_parquet(path)
    im = gpd.GeoDataFrame(dict(inmap_cell_id=[2], inmap_PrimaryPM25=[1.],
        inmap_SecondaryPM25=[3.], inmap_BC=[.1], inmap_NO2=[.2],
        inmap_only_PrimaryPM25=[5.], inmap_only_TotalPM25=[8.],
        inmap_only_BC=[.5], inmap_only_NO2=[.8]), geometry=geom, crs=26910)
    ae = gpd.GeoDataFrame(dict(aermod_cell_id=[1], aermod_PrimaryPM25=[10.],
        aermod_BC=[20.], aermod_NO2=[30.], has_aermod_primarypm25=[True]), geometry=geom, crs=26910)
    import pandas as pd
    power = pd.DataFrame(dict(inmap_cell_id=[2], power_PrimaryPM25=[.2], power_SecondaryPM25=[.3]))
    result = _build_full_exposure_grid(pipeline=_pipeline(tmp_path, path,
        primary_pm25_integration_strategy='inmap_only'), prepared_inmap=im,
        prepared_aermod=ae, prepared_power=power, working_dir=tmp_path)
    np.testing.assert_allclose(result[['PrimaryPM25','SecondaryPM25','TotalPM25','BC','NO2']], [[5.2,3.3,8.5,.5,.8]])


def test_legacy_scaling_falls_back_to_inmap_when_aermod_budget_zero():
    data = gpd.GeoDataFrame(dict(has_aermod_primarypm25=[True, False],
        inmap_PrimaryPM25=[2., 4.], aermod_PrimaryPM25=[0.,0.]),
        geometry=[box(0,0,1,1), box(1,0,2,1)], crs=26910)
    result = _scale_aermod_primary_to_inmap_domain_primary(data)
    np.testing.assert_allclose(result.aermod_PrimaryPM25_scaled, [2.,0.])


def test_human_exposure_receives_plants_without_road_map_or_valid_delta(monkeypatch, tmp_path):
    from pathlib import Path
    from types import SimpleNamespace
    from dataclasses import replace
    import impacts.postprocessor as post
    import impacts.pipeline.postprocess.step1_compare_fleet as fleet
    import impacts.pipeline.postprocess.step8_human_exposure as human
    from impacts.config.settings_builder import load_settings_from_yaml
    root = Path(__file__).resolve().parents[1]
    settings = load_settings_from_yaml(root / 'examples/pilates/settings.yaml')
    human_config = SimpleNamespace(enabled=True,
        ces_tracts_file='ces.parquet', sensitive_facilities_file='facilities.gpkg',
        schools_layer='schools', ltc_layer='ltc')
    settings = replace(settings, impacts=replace(settings.impacts,
        analysis=replace(settings.impacts.analysis, sector_targets=[], inventory_targets=[],
                         human_exposure=human_config),
        power_generation=replace(settings.impacts.power_generation, plant_markers_file='plants.csv')))
    monkeypatch.setattr(post, 'load_settings_from_yaml', lambda _: settings)
    for name in ['_resolve_modeled_emissions_path','_resolve_skims_emissions_path',
                 '_resolve_inventory_emfacid_activity_path','_resolve_vehicle_category_metadata_path',
                 '_resolve_county_boundaries_path']:
        monkeypatch.setattr(post, name, lambda *a, **kw: tmp_path / 'input.parquet')
    monkeypatch.setattr(post, '_resolve_vehicle_types_paths', lambda *a,**kw: (tmp_path/'p.csv', tmp_path/'f.csv'))
    monkeypatch.setattr(post, '_resolve_optional_population_assignment_paths', lambda *a,**kw: (None,None))
    monkeypatch.setattr(post, '_resolve_postprocess_sample_config', lambda *a,**kw: (.1,.1,.1))
    monkeypatch.setattr(fleet, 'run', lambda **kw: {})
    captured = {}
    def fake_human(**kw):
        captured.update(kw)
        return {'report': str(tmp_path/'report.json')}
    monkeypatch.setattr(human, 'run', fake_human)
    post._run_postprocess_steps(tmp_path/'settings.yaml', output_root=tmp_path/'result',
        baseline_concentration_override=tmp_path/'missing-baseline.parquet')
    assert captured['power_plants_path'] == str((tmp_path/'plants.csv').resolve())
