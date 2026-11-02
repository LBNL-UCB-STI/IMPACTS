"""Audit the original-skims controlled dispersion test without omitting new sources."""
from research_paths import ISRM_PATH, BREATHE_ROOT, BEAM_DATA_ROOT, CHARGING_REFERENCE
import json
from pathlib import Path

import duckdb
import pandas as pd
import yaml
import zarr

import run_ldv_freight_products as previous

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'examples/pilates/impacts/impacts_output/impacts-postsim--sfbay--base--20260901-165739'
OUT = ROOT/'examples/pilates/impacts/impacts_output/original-skims-3km-test-20261015'


def main():
    OUT.mkdir(exist_ok=True)
    target = OUT/'input_coverage_audit.json'
    if target.exists():
        raise FileExistsError(target)
    manifest = yaml.safe_load((BASE/'preprocess_manifest.yaml').read_text())
    source = BASE/'emissions/beam_emissions_for_inmap.parquet'
    with duckdb.connect() as con:
        columns = [r[0] for r in con.execute('DESCRIBE SELECT * FROM read_parquet(?)', [str(source)]).fetchall()
                   if r[0].startswith('tons_per_year_')]
        data = con.execute('SELECT inmap_cell_id,'+','.join(f'sum({c}) AS {c}' for c in columns)+
                           ' FROM read_parquet(?) GROUP BY 1', [str(source)]).fetchdf().set_index('inmap_cell_id')
    matrix = zarr.open_group(ISRM_PATH, mode='r')
    missing = data.loc[~data.index.isin(matrix['source_ids'][:]) & data.abs().sum(axis=1).gt(0)]
    approved = pd.read_csv(previous.TOTAL/'inputs/missing_isrm_sources.csv').inmap_cell_id
    additional = missing.loc[~missing.index.isin(approved)]
    missing.to_csv(OUT/'missing_isrm_sources.csv')
    additional.to_csv(OUT/'additional_missing_isrm_sources.csv')
    report = dict(status='awaiting_additional_missing_source_authorization',
        original_skims=manifest['inputs']['emissions_skims_input']['source_path'],
        reference_manifest=str(BASE/'preprocess_manifest.yaml'),
        reference_annual_emissions=str(source),
        original_population_sample=manifest['pipeline']['population_sample'],
        original_transit_sample=manifest['pipeline']['transit_sample'],
        original_annualization_days=manifest['pipeline']['annualization_days'],
        annual_emissions_modified=False,concentration_generated=False,
        intended_aermod_radius_m=3000, intended_isrm_radius_m=3000,
        isrm_receptor_overlap_rule='strictly greater than 0.5',
        missing_source_count=len(missing), additional_missing_source_ids=list(map(int,additional.index)),
        missing_emission_fractions=(missing.sum()/data.sum()).to_dict(),
        additional_emission_fractions=(additional.sum()/data.sum()).to_dict(),
        remote_isrm_probe='HTTP 403 Forbidden for .zmetadata and PrimaryPM25/.zarray',
        note='The HF-1EP approval for 1926,1927,2078 is scenario-specific and does not authorize these additional source IDs.')
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
