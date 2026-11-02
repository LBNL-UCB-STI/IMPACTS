import numpy as np
import pandas as pd
import pytest

from impacts.pipeline.workflow.charging.beam_sessions import reconstruct_sessions


def session(start, end, kwh, kind='ev-test', active=None):
    return dict(vehicle='v1', parkingZoneId='p1', vehicleType=kind,
                time=end, duration=end-start if active is None else active, fuel=kwh*3_600_000.)


def test_exact_hour_overlap_and_single_expansion():
    s=pd.DataFrame([session(1800, 5400, 10)])
    p=pd.DataFrame([dict(vehicle='v1', parkingZoneId='p1', time=1800)])
    curves,a=reconstruct_sessions(s,p,{'ev-test'},scale=10)
    np.testing.assert_allclose(curves['ev-test'][:2],[50,50])
    assert a.included_expanded_kwh.sum()==100


def test_midnight_clip_not_modulo_and_zero_energy():
    s=pd.DataFrame([session(84600,88200,10),session(90000,90000,0)])
    s.loc[1,'vehicle']='v2'
    p=pd.DataFrame([dict(vehicle='v1', parkingZoneId='p1', time=84600)])
    curves,a=reconstruct_sessions(s,p,{'ev-test'},scale=1)
    assert curves['ev-test'][23]==5
    assert curves['ev-test'][0]==0
    assert a.excluded_sample_kwh.sum()==5


def test_mean_power_over_connected_interval_not_active_duration():
    s=pd.DataFrame([session(0,7200,10,active=3600)])
    p=pd.DataFrame([dict(vehicle='v1', parkingZoneId='p1', time=0)])
    curves,a=reconstruct_sessions(s,p,{'ev-test'},scale=1)
    np.testing.assert_allclose(curves['ev-test'][:2],[5,5])
    assert a.mean_connected_kw.iloc[0]==5


def test_non_electric_excluded_and_missing_plugin_audited():
    s=pd.DataFrame([session(0,3600,10,kind='ft-hdt-Dsl')])
    p=pd.DataFrame(columns=['vehicle','parkingZoneId','time'])
    curves,a=reconstruct_sessions(s,p,{'ev-test'},scale=1)
    assert not curves
    assert a.excluded_sample_kwh.iloc[0]==10
    assert a.timing_method.iloc[0]=='end_minus_duration_fallback'


def test_invalid_energy_duration_rejected():
    s=pd.DataFrame([session(0,0,10)])
    p=pd.DataFrame(columns=['vehicle','parkingZoneId','time'])
    with pytest.raises(ValueError,match='zero active'):
        reconstruct_sessions(s,p,{'ev-test'})


def test_quarantine_duration_mismatch_preserves_audit():
    s=pd.DataFrame([session(0,39,1,active=786)])
    p=pd.DataFrame([dict(vehicle='v1',parkingZoneId='p1',time=0)])
    with pytest.raises(ValueError,match='exceeds'):
        reconstruct_sessions(s,p,{'ev-test'})
    curves,a=reconstruct_sessions(s,p,{'ev-test'},quarantine_invalid=True)
    assert not curves
    assert a.quarantined.iloc[0]
    assert a.excluded_sample_kwh.iloc[0]==1


def test_duplicate_disconnection_cannot_double_count():
    s=pd.DataFrame([session(0,3600,10),session(0,4000,10,active=3600)])
    p=pd.DataFrame([dict(vehicle='v1',parkingZoneId='p1',time=0)])
    curves,a=reconstruct_sessions(s,p,{'ev-test'},scale=1,quarantine_invalid=True)
    assert curves['ev-test'].sum()==10
    assert a.quarantined.sum()==1
    assert a.excluded_sample_kwh.sum()==10
