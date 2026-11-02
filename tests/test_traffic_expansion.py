"""Uniform traffic expansion must not alter geometry or integration factors."""
import importlib.util
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('scale_events_traffic', SCRIPTS / 'scale_events_traffic.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('name', ['TotalPM25', 'inmap_only_BC', 'InMAPOnly_NO2', 'aermod_PrimaryPM25_scaled'])
def test_concentrations_selected(name):
    assert module.scale_column(name, 'concentration')


@pytest.mark.parametrize('name', ['geometry', 'person_count', 'aermod_cell_id', 'has_aermod_bc', 'aermod_PrimaryPM25_scale_factor'])
def test_unscaled_fields(name):
    assert not module.scale_column(name, 'concentration')


def test_parquet_scaling_preserves_metadata_and_other_fields(tmp_path):
    original = pa.table({'TotalPM25': [1., 0., 2.], 'aermod_cell_id': [10, 20, 30],
                         'geometry': [b'one', b'two', b'three'],
                         'has_aermod_bc': [True, False, True],
                         'aermod_PrimaryPM25_scale_factor': [1., 1., 1.]})
    original = original.replace_schema_metadata({b'geo': b'preserved-test-metadata'})
    source, target = tmp_path / 'source.parquet', tmp_path / 'result.parquet'
    pq.write_table(original, source)
    audit = module.scale_parquet(source, target, 'concentration')
    result = pq.read_table(target)
    assert result['TotalPM25'].to_pylist() == [10., 0., 20.]
    for name in original.column_names[1:]:
        assert original[name].equals(result[name])
    assert result.schema.metadata[b'geo'] == original.schema.metadata[b'geo']
    assert audit['all_columns_checked']
    with pytest.raises(FileExistsError):
        module.scale_parquet(source, target, 'concentration')
