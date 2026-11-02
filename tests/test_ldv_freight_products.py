"""Fleet-use classification precedes emission aggregation and dispersion."""
import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('run_ldv_freight_products',SCRIPTS/'run_ldv_freight_products.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_ldt_body_type_does_not_override_freight_use():
    frame=pd.DataFrame({'vehicleTypeId':['p','f','b'],
        'sourceVehicleTypeId':['cv-L1-17000-2019','ft-ld1-Dsl','BUS-AC'],
        'assignment_group':['passenger','freight','transit'],
        'emfacVehicleCategory':['LDT1','LDA','UBUS']})
    result=module.classify_mapping(frame).set_index('vehicleTypeId')
    assert result.loc['f','assignment_group']=='freight'
    assert result.loc['p','assignment_group']=='passenger'
    assert 'transit' not in module.GROUPS.values()


@pytest.mark.parametrize('group',['passenger',None,'unknown'])
def test_reject_misclassified_freight(group):
    frame=pd.DataFrame({'vehicleTypeId':['f'],'sourceVehicleTypeId':['ft-hdt-Dsl'],
                        'assignment_group':[group]})
    with pytest.raises(ValueError):module.classify_mapping(frame)


def test_reject_duplicate_cohort_keys():
    frame=pd.DataFrame({'vehicleTypeId':['same','same'],'sourceVehicleTypeId':['a','b'],
                        'assignment_group':['passenger','freight']})
    with pytest.raises(ValueError):module.classify_mapping(frame)
