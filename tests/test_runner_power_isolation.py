"""A previous power-sector run must never contaminate a new runner invocation."""
from pathlib import Path
import importlib.util

import pytest
import yaml

from impacts import runner
from test_terminal_contract import _preprocess_manifest_payload


@pytest.fixture
def power_run(tmp_path, monkeypatch):
    from impacts.pipeline.workflow import power_generation, step4_prepare_exposure

    for name in ('county', 'inmap', 'aermod'):
        (tmp_path / f'beam_osm_{name}_intersection.parquet').write_text('grid')
    preprocess = _preprocess_manifest_payload(tmp_path)
    output = tmp_path / 'output'
    output.mkdir()
    old_power = output / 'old_power.parquet'
    old_power.write_text('previous scenario power')
    demand = tmp_path / 'demand.parquet'
    demand.write_text('first load')
    config = dict(enabled=True, hourly_demand_file=str(demand), source_layer_index=0)
    preprocess['pipeline']['power_generation'] = config.copy()
    preprocess_path = tmp_path / 'preprocess.yaml'
    manifest_path = output / 'pipeline_manifest.yaml'
    outputs = dict(beam_inmap_concentrations=str(output/'traffic.parquet'),
        power_generation_isrm_concentrations=str(old_power),
        power_generation_isrm_concentrations_gpkg=str(old_power.with_suffix('.gpkg')),
        power_generation_summary=str(output/'old_summary.json'),
        beam_concentration_distribution=str(output/'old_combined.parquet'),
        beam_concentration_distribution_gpkg=str(output/'old_combined.gpkg'),
        beam_population_distribution=str(output/'old_population.parquet'),
        beam_population_counts=str(output/'old_counts.parquet'),
        beam_population_counts_gpkg=str(output/'old_counts.gpkg'), skims_emissions=None)
    previous = dict(contract_version='1', model='impacts',
        preprocess_manifest_path=str(preprocess_path), output_dir=str(output),
        command='previous run', image='test', outputs=outputs,
        pipeline=preprocess['pipeline'].copy(), population_inputs={},
        deterministic_contract={}, execution={}, pipeline_manifest_path=str(manifest_path))
    manifest_path.write_text(yaml.safe_dump(previous))
    calls = {'power': [], 'exposure': []}

    def compute(**kwargs):
        calls['power'].append(dict(kwargs))
        current = kwargs['config']
        path = Path(kwargs['output_dir']) / 'power_generation_isrm_concentrations.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(Path(current['hourly_demand_file']).read_text())
        return path

    monkeypatch.setattr(power_generation, 'run_power_generation', compute, raising=False)
    if importlib.util.find_spec('impacts.pipeline.workflow.hourly_power'):
        from impacts.pipeline.workflow import hourly_power
        monkeypatch.setattr(hourly_power, 'run_hourly_power', compute)

    def exposure(**kwargs):
        calls['exposure'].append(kwargs)
        return None, output/'current_combined.parquet', None, None

    monkeypatch.setattr(step4_prepare_exposure, 'run', exposure)

    def run(*, enabled=True, execute_exposure=True, execute_inmap=False):
        preprocess['pipeline']['power_generation'] = dict(config, enabled=enabled)
        preprocess_path.write_text(yaml.safe_dump(preprocess))
        return runner._run_stages_from_preprocess_manifest(
            preprocess_path, run_manifest_path=manifest_path, output_root_override=output,
            run_emissions=False, run_inmap=execute_inmap, run_aermod=False,
            run_exposure=execute_exposure)

    return dict(run=run, calls=calls, config=config, demand=demand,
                manifest_path=manifest_path, previous=previous, old_power=old_power)


@pytest.mark.parametrize('old_enabled', [True, False])
def test_disabled_power_never_reaches_exposure_even_if_old_manifest_has_path(power_run, old_enabled):
    case = power_run
    case['previous']['pipeline']['power_generation'] = {'enabled': old_enabled}
    case['manifest_path'].write_text(yaml.safe_dump(case['previous']))
    result = case['run'](enabled=False)
    assert not case['calls']['power']
    assert case['calls']['exposure'][0]['power_generation_concentrations_path'] is None
    saved = yaml.safe_load(case['manifest_path'].read_text())
    for key in case['previous']['outputs']:
        if key.startswith('power_generation_'):
            assert result['outputs'][key] is None
            assert saved['outputs'][key] is None
    assert result['execution']['power_generation']['status'] == 'disabled'
    assert case['old_power'].read_text() == 'previous scenario power'


@pytest.mark.parametrize('enabled', [True, False])
def test_no_exposure_stage_does_not_republish_old_power_or_combined_results(power_run, enabled):
    case = power_run
    result = case['run'](enabled=enabled, execute_exposure=False)
    assert not case['calls']['power'] and not case['calls']['exposure']
    for key in ['power_generation_isrm_concentrations', 'power_generation_isrm_concentrations_gpkg',
                'beam_concentration_distribution', 'beam_concentration_distribution_gpkg',
                'beam_population_distribution', 'beam_population_counts', 'beam_population_counts_gpkg']:
        assert result['outputs'][key] is None
    assert result['outputs']['beam_inmap_concentrations'] == case['previous']['outputs']['beam_inmap_concentrations']


@pytest.mark.parametrize('change', ['none', 'contents', 'path', 'layer'])
def test_exposure_only_recomputes_power_with_current_inputs(power_run, tmp_path, change):
    case = power_run
    if change == 'contents':
        case['demand'].write_text('changed load at the same path')
    elif change == 'path':
        new = tmp_path/'other_load.parquet'
        new.write_text('other scenario')
        case['config']['hourly_demand_file'] = str(new)
    elif change == 'layer':
        case['config']['source_layer_index'] = 1
    result = case['run']()
    assert len(case['calls']['power']) == 1
    assert case['calls']['power'][0]['config'] == case['config']
    path = case['calls']['exposure'][0]['power_generation_concentrations_path']
    assert path != str(case['old_power'])
    assert Path(path).read_text() == Path(case['config']['hourly_demand_file']).read_text()
    assert result['outputs']['power_generation_isrm_concentrations'] == path
    assert result['execution']['power_generation']['status'] == 'computed'


def test_inmap_then_exposure_computes_power_once(power_run):
    case = power_run
    result = case['run'](execute_inmap=True)
    assert len(case['calls']['power']) == 1
    assert len(case['calls']['exposure']) == 1
    assert result['outputs']['power_generation_isrm_concentrations']


def test_failed_current_power_does_not_fall_back_to_old_concentration(power_run):
    case = power_run
    case['config']['hourly_demand_file'] = str(case['demand'].with_name('missing.parquet'))
    with pytest.raises(FileNotFoundError):
        case['run']()
    assert not case['calls']['exposure']


@pytest.mark.parametrize('enabled', [True, False])
def test_postprocess_rejects_invalidated_combined_file_even_when_it_still_exists(power_run, enabled):
    from impacts.postprocessor import postprocess_from_pipeline_manifest
    case = power_run
    stale = Path(case['previous']['outputs']['beam_concentration_distribution'])
    stale.write_text('old traffic plus power')
    case['run'](enabled=enabled, execute_exposure=False)
    with pytest.raises(ValueError, match='Run exposure before postprocessing'):
        postprocess_from_pipeline_manifest(case['manifest_path'])
    assert stale.read_text() == 'old traffic plus power'
