"""Independent traffic-product combination must preserve source contributions."""
import importlib.util
from pathlib import Path
import sys

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import Point

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import run_native_ev_baseline as baseline


def test_outer_join_sums_concentrations_not_identifiers(tmp_path):
    left=gpd.GeoDataFrame(dict(aermod_cell_id=[1,2],inmap_cell_id=[10,20],
        target_xm=[1.,2.],PrimaryPM25=[3.,4.],BC=[.1,.2],
        has_aermod_primarypm25=[True,False]),geometry=[Point(1,0),Point(2,0)],crs=26910)
    right=gpd.GeoDataFrame(dict(aermod_cell_id=[2,3],inmap_cell_id=[20,30],
        target_xm=[2.,3.],PrimaryPM25=[5.,6.],BC=[.3,.4],
        has_aermod_primarypm25=[True,True]),geometry=[Point(2,0),Point(3,0)],crs=26910)
    a,b,out=[tmp_path/n for n in ['a.parquet','b.parquet','combined.parquet']]
    left.to_parquet(a,index=False);right.to_parquet(b,index=False)
    baseline.combine_concentrations(a,b,out,'aermod_cell_id')
    result=gpd.read_parquet(out)
    np.testing.assert_array_equal(result.aermod_cell_id,[1,2,3])
    np.testing.assert_array_equal(result.inmap_cell_id,[10,20,30])
    np.testing.assert_allclose(result.PrimaryPM25,[3,9,6])
    np.testing.assert_allclose(result.BC,[.1,.5,.4])
    assert result.has_aermod_primarypm25.all()
    assert result.crs==left.crs
    assert pq.read_metadata(a).num_rows==2
