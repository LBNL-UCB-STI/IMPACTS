from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_future_detailed_products as detailed


def test_catalog_uses_canonical_names_not_parallel_sorted_lists(tmp_path,monkeypatch):
    full=pd.DataFrame(dict(is_port_related=[True,True],gateway_ids=['a','b'],gateway_names=['Z port','A airport']))
    path=tmp_path/'labels.csv'; full.to_csv(path,index=False)
    monkeypatch.setattr(detailed.broad,'CLASSIFICATION',path)
    labels=pd.DataFrame(dict(is_port_related=[True],gateway_group=['a__and__b'],
        has_seaport_stop=[True],has_airport_stop=[True]))
    result=detailed.catalog(labels)[0]
    assert result['gateway_name']=='Z port + A airport'
    assert result['sample_vehicles']==1 and result['represented_vehicles']==10


def test_baseline_policy_preserves_full_diagonal_kernel(monkeypatch):
    from impacts.pipeline.workflow import step2_compute_inmap_concentrations as inmap
    from impacts.pipeline.workflow import step3_compute_aermod_concentrations as aermod
    with monkeypatch.context() as patch:
        patch.setattr(inmap,'build_near_source_mask',inmap.build_near_source_mask)
        patch.setattr(aermod,'_AERMOD_RESPONSE_RADIUS_METERS',aermod._AERMOD_RESPONSE_RADIUS_METERS)
        detailed.baseline_dispersion_policy()
        assert inmap.build_near_source_mask.keywords['radius_meters']==4500
        frame=pd.DataFrame(dict(Grid_X=[0.,3000.],Grid_Y=[0.,3000.],Distance=[0.,4242.],Concentration=[1.,.2]))
        kernel=aermod._build_kernel_for_pattern(frame,grid_size_meters=100.)
        np.testing.assert_allclose(kernel['response_per_ton'],[1.,.2])


def write_concentration(folder,value,ids=(1,2)):
    path=folder/'exposure'; path.mkdir(parents=True)
    data=pd.DataFrame({c:np.array([value,value]) for c in detailed.FIELDS})
    data['aermod_cell_id']=list(ids); data['inmap_cell_id']=[8,8]; data['geometry']=[b'a',b'b']
    data.to_parquet(path/'beam_concentration_distribution.parquet',index=False)


def test_sum_checks_every_component_and_geometry(tmp_path):
    targets=[tmp_path/'a',tmp_path/'b']; reference=tmp_path/'total'
    write_concentration(targets[0],1.); write_concentration(targets[1],2.); write_concentration(reference,3.)
    assert detailed.concentration_sum(targets,reference)['passed']
    bad=tmp_path/'bad';write_concentration(bad,3.1)
    with pytest.raises(AssertionError): detailed.concentration_sum(targets,bad)
    wrong_ids=tmp_path/'ids';write_concentration(wrong_ids,3.,ids=(2,1))
    with pytest.raises(ValueError,match='Receptor'): detailed.concentration_sum(targets,wrong_ids)
