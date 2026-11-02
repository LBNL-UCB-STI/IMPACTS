from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from impacts.common import resolve_latest_events_local_path
from impacts.config.defaults import grams_per_short_ton, meters_per_mile
from impacts.pipeline.workflow.prepare_emissions.event_emfac import (
    EmfacRateStore, build_daily_emissions, load_vehicle_mapping,
)
from impacts.pipeline.workflow.prepare_emissions.from_skims import load_or_prepare_skims_df


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / 'rates' / 'dataset' / 'emfacId=test'
    root.mkdir(parents=True)
    rows = []
    for process, value, activity in [('RUNEX', 2., '30'), ('RUNEX', 8., '60'),
                                     ('RUNLOSS', 6., None), ('PMBW', 1., None),
                                     ('PMTW', .5, None), ('PRDUST', 3., None),
                                     ('STREX', 7., '60'), ('DIURN', 4., None),
                                     ('HOTSOAK', 5., None), ('IDLEX', 10., None)]:
        rows.append(dict(county='Alameda', process=process, roadCategory=None,
                         speedMph_timeMin=activity, nox_gram=value, pm25_gram=None))
    rates = root / 'test.parquet'
    pd.DataFrame(rows).to_parquet(rates, index=False)
    meta = dict(vehicleTypeId='car', emfacId='test', emfacVehicleCategory='LDA',
                curbWeightInKg=907.18474, idleTimeFraction=.1)
    passenger = tmp_path / 'passenger.csv'
    pd.DataFrame([meta]).to_csv(passenger, index=False)
    freight = tmp_path / 'freight.csv'
    pd.DataFrame([{**meta, 'vehicleTypeId': 'truck'}]).to_csv(freight, index=False)
    catalog = tmp_path / 'catalog.csv'
    pd.DataFrame([dict(emfac_vehicle_category='LDA', operation_days_per_year=10.,
                       tailpipe_height_meters=1.)]).to_csv(catalog, index=False)
    net = tmp_path / 'network.parquet'
    pd.DataFrame([dict(linkId=1, linkLength=meters_per_mile, attributeOrigType='primary'),
                  dict(linkId=2, linkLength=meters_per_mile*2, attributeOrigType='primary')]).to_parquet(net)
    zones = tmp_path / 'county.parquet'
    pd.DataFrame([dict(linkId=i, county_COUNTYFP='001', county_proportion=1.) for i in [1, 2]]).to_parquet(zones)
    events = tmp_path / '3.events.parquet'
    trips = pd.DataFrame([
        dict(type='PathTraversal', vehicle='a', vehicleType='car', departureTime=3600,
             arrivalTime=3900, links='1,2', linkTravelTime='120,180', mode='car', PayloadWeightInKg=0.),
        dict(type='PathTraversal', vehicle='a', vehicleType='car', departureTime=7500,
             arrivalTime=7560, links='1', linkTravelTime='60', mode='car', PayloadWeightInKg=0.),
    ])
    trips.to_parquet(events, index=False)
    return dict(root=tmp_path, events=events, net=net, zones=zones, rates=rates,
                folder=tmp_path/'rates', passenger=passenger, freight=freight, catalog=catalog)


def run_daily(x, name='daily.parquet'):
    output = x['root'] / name
    audit = build_daily_emissions(str(x['events']), str(x['net']), str(x['zones']),
                                 str(x['folder']), [str(x['passenger']), str(x['freight'])],
                                 ['NOx', 'PM25'], output)
    return pd.read_parquet(output), audit


def test_native_partition_daily_units_and_processes(inputs):
    frame, audit = run_daily(inputs)
    totals = frame.groupby('process').NOx.sum()
    # Link speeds: 30 mph, 40 mph, 60 mph. Calculate before aggregating.
    assert totals.RUNEX == pytest.approx(2*1 + 2*2 + 8*1)
    assert totals.RUNLOSS == pytest.approx(6 * 360/3600)
    assert totals.PMBW == pytest.approx(4.)
    assert totals.PMTW == pytest.approx(2.)
    assert totals.PRDUST == pytest.approx(12.)  # 1 short ton curb mass
    assert totals.IDLEX == pytest.approx(10 * .1 * 360/3600)
    assert totals.STREX == pytest.approx(7.)
    assert totals.DIURN == pytest.approx(4.)
    assert totals.HOTSOAK == pytest.approx(5.)
    assert frame.PM25.sum() == 0  # Matched native process row, null pollutant.
    assert audit['road_trips'] == 2
    assert audit['observed_parking_intervals'] == 1
    assert not audit['annualization_applied']
    assert not audit['fleet_expansion_applied']
    # Start occurs on next trip's first link at departure, not previous arrival.
    assert frame.loc[frame.process.eq('STREX'), ['hour', 'linkId']].values.tolist() == [[2, 1]]


def test_rate_duplicate_merge_and_conflict(inputs):
    frame = pd.read_parquet(inputs['rates'])
    pd.concat([frame, frame], ignore_index=True).to_parquet(inputs['rates'], index=False)
    daily, _ = run_daily(inputs)
    assert daily.groupby('process').NOx.sum().RUNEX == pytest.approx(14.)
    extra = frame.iloc[[0]].copy()
    extra['nox_gram'] = 999.
    pd.concat([frame, extra], ignore_index=True).to_parquet(inputs['rates'], index=False)
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        run_daily(inputs, 'bad.parquet')
    assert not (inputs['root']/'bad.parquet').exists()


def test_unknown_vehicle_fails_before_output(inputs):
    d = pd.read_parquet(inputs['events'])
    d['vehicleType'] = 'unmapped-new-type'
    d.to_parquet(inputs['events'])
    with pytest.raises(ValueError, match='mapping missing'):
        run_daily(inputs)
    assert not (inputs['root']/'daily.parquet').exists()


@pytest.mark.parametrize('fault', ['column', 'county', 'nan-rate', 'missing-partition'])
def test_missing_or_invalid_rate_is_not_zero(inputs, fault):
    d = pd.read_parquet(inputs['rates'])
    if fault == 'column':
        d = d.drop(columns='nox_gram')
    elif fault == 'county':
        d['county'] = 'Marin'
    elif fault == 'nan-rate':
        d.loc[0, 'nox_gram'] = np.inf
    elif fault == 'missing-partition':
        m = pd.read_csv(inputs['passenger'])
        m['emfacId'] = 'missing'
        m.to_csv(inputs['passenger'], index=False)
    d.to_parquet(inputs['rates'])
    with pytest.raises((ValueError, FileNotFoundError)):
        run_daily(inputs)
    assert not (inputs['root']/'daily.parquet').exists()


def test_payload_changes_only_dust(inputs):
    d = pd.read_parquet(inputs['events'])
    d['PayloadWeightInKg'] = 907.18474
    d.to_parquet(inputs['events'])
    daily, _ = run_daily(inputs)
    totals = daily.groupby('process').NOx.sum()
    assert totals.PRDUST == pytest.approx(12 * 2**1.02)
    assert totals.RUNEX == pytest.approx(14.)


def test_zero_dwell_does_not_create_starts_or_parking(inputs):
    d = pd.read_parquet(inputs['events'])
    d.loc[1, ['departureTime', 'arrivalTime']] = [3900, 3960]
    d.to_parquet(inputs['events'])
    daily, _ = run_daily(inputs)
    assert not set(daily.process) & {'DIURN', 'HOTSOAK', 'STREX'}


def test_direct_iteration_events_discovery(inputs):
    assert Path(resolve_latest_events_local_path(str(inputs['root']))) == inputs['events']


def test_main_step1_fallback_annualizes_once(inputs):
    def entry(p):
        return dict(kind='local', source_path=str(p), staged_path=str(p), exists=True)
    manifest = {k: entry(inputs[v]) for k, v in {
        'events_input': 'events', 'network': 'net', 'rates_folder': 'folder',
        'passenger_vehicle_types_input': 'passenger', 'freight_vehicle_types_input': 'freight',
    }.items()}
    # Optional missing skims must not shadow valid events.
    manifest['emissions_skims_input'] = {'exists': False}
    output = inputs['root']/'emissions'
    result = load_or_prepare_skims_df(
        input_root=output, intersection_path=str(inputs['zones']), beam_length_col='linkLength',
        prepared_skims_group_cols=['linkId', 'vehicleTypeId', 'process'],
        pollutants=['NOx', 'PM25'], vehicle_category_metadata_file=str(inputs['catalog']),
        annualization_days={'light_duty': 10., 'medium_heavy_duty': 10.},
        population_sample=.5, transit_sample=1., freight_sample=1.,
        include_passenger=True, include_freight=True, manifest_inputs=manifest,
        require_aermod_support=True,
    )
    assert result.loc[result.process.eq('RUNEX'), 'tons_per_year_NOx'].sum() == pytest.approx(14 * 2 * 10 / grams_per_short_ton)
    assert result.loc[result.process.eq('RUNEX'), 'totVMT'].sum() == pytest.approx(4 * 2 * 10)
    assert set(result.source_release_height) == {1.}
    assert (output/'events_emfac_audit.json').exists()


def test_county_partial_coverage_not_applied_twice(inputs):
    z = pd.read_parquet(inputs['zones'])
    z['county_proportion'] = .25
    z.to_parquet(inputs['zones'])
    daily, _ = run_daily(inputs)
    assert daily.groupby('process').NOx.sum().RUNEX == pytest.approx(14.)


def test_bin_selection_keeps_county_and_road_dimensions(inputs):
    mapping = load_vehicle_mapping([inputs['passenger']])
    store = EmfacRateStore(inputs['folder'], mapping, ['NOx'])
    assert store.factor('car', 'RUNEX', 'Alameda', 'primary', 60., 0.)[0] == 8.
    assert store.factor('car', 'RUNEX', 'Alameda', 'primary', 40., 0.)[0] == 2.
    with pytest.raises(ValueError, match='Missing EMFAC rate'):
        store.factor('car', 'RUNEX', 'Marin', 'primary', 40., 0.)


@pytest.mark.parametrize('fault', ['overlap', 'list-length', 'missing-link', 'negative-duration'])
def test_invalid_events_fail_without_publishing_skims(inputs, fault):
    d = pd.read_parquet(inputs['events'])
    if fault == 'overlap':
        d.loc[1, 'departureTime'] = 3800
    elif fault == 'list-length':
        d.loc[0, 'linkTravelTime'] = '120'
    elif fault == 'missing-link':
        d.loc[0, 'links'] = '999,2'
    else:
        d.loc[0, 'linkTravelTime'] = '-1,180'
    d.to_parquet(inputs['events'])
    with pytest.raises(ValueError):
        run_daily(inputs)
    assert not (inputs['root']/'daily.parquet').exists()


def test_preprocess_accepts_events_without_skims(inputs, monkeypatch):
    from impacts.config.settings_builder import load_settings_from_yaml
    from impacts.pipeline.preprocessing import step1_collect_inputs as collect

    class ReachedNextStage(Exception):
        pass

    settings_file = Path(__file__).resolve().parents[1]/'examples/pilates/settings.yaml'
    settings = load_settings_from_yaml(settings_file)
    monkeypatch.setattr(collect, 'resolve_required_path', lambda *a: str(inputs['root']))
    monkeypatch.setattr(collect, '_resolve_region_input_root', lambda **kw: inputs['root'])
    monkeypatch.setattr(collect, 'build_registry', lambda *a: None)
    monkeypatch.setattr(collect, 'find_latest_beam_network_reference', lambda **kw: None)
    monkeypatch.setattr(collect, 'find_latest_beam_events_reference', lambda **kw: None)
    monkeypatch.setattr(collect, 'resolve_beam_network_local_path', lambda *a: str(inputs['net']))

    def register(manifest, **kw):
        manifest[kw['key']] = dict(staged_path=kw['source_path'], exists=True)
        return kw['source_path']

    def next_stage(**kw):
        raise ReachedNextStage()

    monkeypatch.setattr(collect, '_register_manifest_input', register)
    monkeypatch.setattr(collect, 'find_beam_r5_osm_reference', next_stage)
    manifest = {}
    with pytest.raises(ReachedNextStage):
        collect.run(manifest_inputs=manifest, settings=settings, config_path=settings_file,
                    input_root=inputs['root']/'preprocess')
    assert 'emissions_skims_input' not in manifest
    assert manifest['events_input']['staged_path'] == str(inputs['events'])
