"""Rebuild annual emissions from a reference manifest's unannualized skims."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare-only', action='store_true')
    args = p.parse_args()
    if args.output.exists() and not args.compare_only:
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, exist_ok=args.compare_only)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    manifest = yaml.safe_load((args.reference / 'preprocess_manifest.yaml').read_text(encoding='utf-8'))
    manifest['pipeline']['power_generation'] = {'enabled': False}
    path = args.output / 'preprocess_manifest.yaml'
    if not args.compare_only:
        path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding='utf-8')
        from impacts.runner import _run_stages_from_preprocess_manifest
        _run_stages_from_preprocess_manifest(path, run_manifest_path=args.output / 'pipeline_manifest.yaml',
            output_root_override=args.output, run_emissions=True, run_inmap=False, run_aermod=False, run_exposure=False)
    comparisons = []
    for stem, key in [('beam_emissions_for_inmap', 'inmap_cell_id'), ('beam_emissions_for_aermod', 'aermod_cell_id'),
                      ('beam_emissions_by_county_process', None)]:
        old, new = [base / 'emissions' / (stem + '.parquet') for base in (args.reference, args.output)]
        columns = [c for c in pq.read_schema(old).names if c.startswith('tons_per_year_')]
        if not columns:
            raise ValueError('No annual emissions columns: ' + stem)
        # Group before comparing: parallel joins need not preserve row order.
        keys = [key] if key else []
        keys += [c for c in ['county_COUNTYFP', 'vehicleTypeId', 'process', 'roadCategory',
                            'source_release_height', 'source_temporal_class', 'source_urban_class']
                 if c in pq.read_schema(old).names]
        import duckdb
        def grouped(file):
            select = ','.join('"' + c + '"' for c in keys)
            sums = ','.join(f'SUM("{c}") AS "{c}"' for c in columns)
            with duckdb.connect() as db:
                sql = 'SELECT ' + ((select + ',') if select else '') + sums + ' FROM read_parquet(?)'
                if keys:
                    sql += ' GROUP BY ALL ORDER BY ' + select
                return db.execute(sql, [str(file)]).df()
        a, b = grouped(old), grouped(new)
        pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-9, atol=1e-8)
        comparisons.append(dict(file=stem, groups=len(a), group_keys=keys,
            max_absolute_difference=float(np.abs(a[columns].to_numpy()-b[columns].to_numpy()).max())))
    (args.output / 'comparison.json').write_text(json.dumps(dict(status='passed', annualization_recomputed=True,
        input=manifest['inputs']['emissions_skims_input'], comparisons=comparisons), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
