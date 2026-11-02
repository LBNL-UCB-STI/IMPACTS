import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    'original_tables', Path(__file__).parents[1] / 'scripts/export_original_style_human_exposure_tables.py'
)
tables = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tables)


def test_export_excludes_neighbor_counties_and_invalid_ces(tmp_path):
    data = pd.DataFrame({
        'TractID': range(6), 'County': ['Alameda'] * 4 + ['Yolo', 'Alameda'],
        'CIscoreP': [10, 30, 60, 90, 10, None],
        'CES_Group': ['0.00-0.25', '0.25-0.50', '0.50-0.75', '0.75-1.00', '0.00-0.25', 'invalid'],
        'CES_GroupID': [1, 2, 3, 4, 1, 0],
        'CESPopulation': [10] * 6, 'BEAM_POP': [10] * 6,
        'ModelGridCount': [1] * 6, 'ModelPopulation': [10] * 6,
    })
    for _, _, field in tables.METHODS:
        data[f'{field}_Exposure'] = 20.
        data[f'{field}_PW'] = 2.
    source = tmp_path / 'tracts.parquet'
    data.to_parquet(source)
    tract_path, summary_path, comparison_path = tables.export(source, tmp_path / 'tables')
    assert pd.read_csv(tract_path).TractID.tolist() == [0, 1, 2, 3]
    for path in [summary_path, comparison_path]:
        summary = pd.read_csv(path)
        assert summary.loc[summary.CES_Group.eq(tables.ALL_GROUP), 'ModelPopulation'].eq(40).all()
