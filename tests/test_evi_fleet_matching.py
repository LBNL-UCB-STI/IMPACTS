"""Check matched-EVI denominators, structural DC omissions and power units."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('evi_matching',Path(__file__).resolve().parents[1]/'scripts/match_baseline_evi_profiles.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def payload():
    parts={}
    for charger in m.CHARGERS:
        parts[charger]={f'{kind}_{body}':[30000*share*bshare]*96
            for kind,share in m.TYPE_SHARES.items() for body,bshare in m.BODY_SHARES.items()
            if not (charger=='public_l3' and kind.startswith('phev'))}
    return {'results':{'weekday_load_profile':parts}}


def test_reweight_uses_source_subtype_counts_not_total_fleet():
    parts=m.unpack(payload())
    target=dict(phev20=0,phev50=9370,bev100=6020,bev250=50)
    weighted=m.reweight(parts,target)
    # Each synthetic available charger delivers 1 kW per vehicle.
    np.testing.assert_allclose(sum(weighted['home_l1'].values()),15440)
    np.testing.assert_allclose(sum(weighted['public_l3'].values()),6070)
    np.testing.assert_allclose(m.total(weighted,'phev'),9370*5)
    np.testing.assert_allclose(m.total(weighted,'bev'),6070*6)


def test_source_fleet_reconstruction():
    parts=m.unpack(payload())
    out=m.reweight(parts,{k:v*30000 for k,v in m.TYPE_SHARES.items()})
    np.testing.assert_allclose(m.total(out),m.total(parts))


def test_hourly_mean_not_sum_and_daily_energy():
    x=np.repeat(np.arange(24.),4)
    np.testing.assert_array_equal(m.hourly(x),np.arange(24.))
    stats=m.curve_metrics('test',x,1000)
    assert stats['daily_MWh']==pytest.approx(x.sum()/4/1000)
    assert stats['peak_MW']==pytest.approx(.023)


@pytest.mark.parametrize('bad',['missing','null','negative'])
def test_no_silent_missing_or_invalid_components(bad):
    data=payload()
    p=data['results']['weekday_load_profile']['home_l1']
    if bad=='missing':p.pop('phev50_sedan')
    elif bad=='null':p['phev50_sedan'][0]=None
    else:p['phev50_sedan'][0]=-1
    with pytest.raises(ValueError):m.unpack(data)
