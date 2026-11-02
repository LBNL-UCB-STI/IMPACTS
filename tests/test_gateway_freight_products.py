import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from gateway_aermod_crop import apply_cropped, original_apply
from run_gateway_freight_products import add_gateway_group, gateway_catalog


def test_combinations_not_duplicated_and_names_not_zipped():
    d=pd.DataFrame(dict(is_port_related=[True,True,True,True,False],
        gateway_ids=['a','b','b|a',None,None],gateway_names=['Alpha','Beta','Alpha|Beta',None,None],
        has_seaport_stop=[True]*4+[False],has_airport_stop=[False]*5))
    d=add_gateway_group(d)
    rows=gateway_catalog(d)
    assert d.gateway_group.tolist()==['a','b','a__and__b','Unknown_gateway','NonPort']
    assert sum(r['sample_vehicles'] for r in rows)==4
    assert next(r for r in rows if r['group']=='a__and__b')['gateway_name']=='Alpha + Beta'


@pytest.mark.parametrize('holes',[False,True])
@pytest.mark.parametrize('outside',[False,True])
def test_cropped_operator_matches_original(holes,outside):
    ix,iy=np.meshgrid(np.arange(-5,26),np.arange(10,38),indexing='ij')
    target=pd.DataFrame({'aermod_cell_id':np.arange(ix.size),'target_ix':ix.ravel(),'target_iy':iy.ravel()})
    if holes:target=target.loc[~((target.target_ix.between(0,5)) & target.target_iy.between(12,20))].copy()
    source=pd.DataFrame(dict(pattern_key=['a','a','b','b'],source_ix=[1,5,15,99 if outside else 23],source_iy=[17,22,25,34],
        tons_per_year_PM25=[3.,2.,0.,1.],tons_per_year_BC=[0.,1.,5.,0.],tons_per_year_NOx=[2.,0.,4.,1.]))
    kernels={g:dict(dix=np.array([-2,0,2,0],dtype=np.int32),diy=np.array([0,0,0,3],dtype=np.int32),
        response_per_ton=np.array([.1,.5,.2,.05])*(1 if g=='a' else 2)) for g in ['a','b']}
    kw=dict(source_df=source,target_index=target,target_id_col='aermod_cell_id',
        emissions_cols=['tons_per_year_PM25','tons_per_year_BC','tons_per_year_NOx'],kernel_library=kernels)
    a,b=original_apply(**kw),apply_cropped(**kw)
    for c in a:
        if a[c].dtype==bool or c=='aermod_cell_id':np.testing.assert_array_equal(a[c],b[c])
        else:np.testing.assert_allclose(a[c],b[c],atol=1e-12,rtol=1e-12)
