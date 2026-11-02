import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_port_freight_products import validate_labels, rewritten, REFERENCE


def labels():
    return pd.DataFrame(dict(scenario=['2018-Baseline']*3, vehicleId=['ft-a','ft-b','ft-c'],
        beam_vehicle_id=['body-ft-a','body-ft-b','body-ft-c'],
        port_category=['seaport_related','airport_related','non_port_freight'],
        is_port_related=[True,True,False], is_non_port_freight=[False,False,True],
        has_seaport_stop=[True,False,False], has_airport_stop=[False,True,False]))


def test_mutually_exclusive_baseline_classification():
    d = validate_labels(labels())
    assert len(d) == 3
    assert d.is_port_related.sum() == d.has_seaport_stop.sum()+d.has_airport_stop.sum()


@pytest.mark.parametrize('kind', ['duplicate','body','overlap','category','nonport'])
def test_invalid_labels_fail(kind):
    d = labels()
    if kind == 'duplicate': d = pd.concat([d,d.iloc[:1]])
    if kind == 'body': d.loc[0,'beam_vehicle_id']='ft-a'
    if kind == 'overlap': d.loc[0,'has_airport_stop']=True
    if kind == 'category': d.loc[0,'port_category']='non_port_freight'
    if kind == 'nonport': d.loc[0,'is_non_port_freight']=True
    with pytest.raises(ValueError): validate_labels(d)


def test_only_scenario_output_paths_relocated(tmp_path):
    d = {'path': str(REFERENCE / 'concentrations/test.parquet'), 'shared': 'E:/shared/isrm.zarr'}
    r = rewritten(d, tmp_path)
    assert r['path'] == str(tmp_path / 'concentrations/test.parquet')
    assert r['shared'] == d['shared']
