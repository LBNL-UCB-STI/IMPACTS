import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_future_fixed_factor_scenarios import freight_donors, select_processes
from run_future_fixed_factor_scenarios import ADDITIONAL_APPROVED_ISRM_SOURCES


def test_additional_isrm_approval_is_scenario_scoped():
    assert ADDITIONAL_APPROVED_ISRM_SOURCES == {'HF-1EP': (1926, 1927, 2078)}
    assert ADDITIONAL_APPROVED_ISRM_SOURCES.get('BF-06EP', ()) == ()
    assert ADDITIONAL_APPROVED_ISRM_SOURCES.get('BF-1EP', ()) == ()


@pytest.fixture
def baseline():
    return pd.DataFrame(dict(sourceVehicleTypeId=['ft-ld1-Dsl','ft-ld1-Gas','ft-mdv-Dsl','cv-foo'],
        emfacId=['d','g','md','p'], fleetVmtPrior=[1.,3.,1.,1.], fleetPopulationPrior=[2.,2.,1.,1.],
        vmt_weight=[1.,1.,1.,1.], pop_weight=[1.,1.,1.,1.]))


def test_zero_tailpipe_pools_same_class_priors(baseline):
    for tech in ['BE','H2FC']:
        donors,policy=freight_donors('ft-ld1-'+tech,baseline)
        assert policy=='nonexhaust'
        assert set(donors.emfacId)=={'d','g'}
        np.testing.assert_allclose(donors.vmt_weight,[.25,.75])
        np.testing.assert_allclose(donors.pop_weight,[.5,.5])


def test_ice_and_passenger_factors_unchanged(baseline):
    for source in ['ft-ld1-Gas','cv-foo']:
        actual,policy=freight_donors(source,baseline)
        assert policy=='native'
        pd.testing.assert_frame_equal(actual,baseline.loc[baseline.sourceVehicleTypeId.eq(source)])


def test_phev_uses_declared_secondary_fuel(baseline):
    a,pa=freight_donors('ft-ld1-PHEV~engine_on',baseline)
    b,pb=freight_donors('ft-mdv-PHEV~engine_off',baseline)
    assert a.emfacId.tolist()==['g'] and pa=='native'
    assert b.emfacId.tolist()==['md'] and pb=='phev_off'
    with pytest.raises(ValueError):
        freight_donors('ft-ld1-PHEV',baseline)


def test_filters_keep_exact_nonexhaust_values():
    d=pd.DataFrame({'process':['RUNEX','PMBW','PMTW','PRDUST','DIURN','HOTSOAK','STREX','IDLEX'],
                    'pm25_gram':np.arange(8,dtype=float)})
    for policy,keep in [('nonexhaust',['PMBW','PMTW','PRDUST']),
                        ('phev_off',['PMBW','PMTW','PRDUST','DIURN'])]:
        actual=select_processes(d,policy)
        pd.testing.assert_frame_equal(actual,d.loc[d.process.isin(keep)])
    pd.testing.assert_frame_equal(select_processes(d,'native'),d)


def test_missing_nonexhaust_fails():
    with pytest.raises(ValueError):
        select_processes(pd.DataFrame({'process':['PMTW']}),'nonexhaust')
