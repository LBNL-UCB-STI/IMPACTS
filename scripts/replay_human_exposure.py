"""Run exposure, original-style figures/tables and HTML against explicit assets."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code-root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--population', required=True)
    p.add_argument('--concentration', required=True)
    p.add_argument('--ces', required=True)
    p.add_argument('--facilities', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--compare', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.code_root.resolve() / 'src'))
    sys.path.insert(0, str(args.code_root.resolve() / 'scripts'))
    import impacts
    import pandas as pd
    import geopandas as gpd
    from impacts.pipeline.postprocess.step8_human_exposure import run
    from export_original_style_human_exposure_tables import export
    outputs = run(population_path=args.population, concentration_path=args.concentration,
        ces_tracts_path=args.ces, sensitive_facilities_path=args.facilities,
        schools_layer='Schools_BayArea', ltc_layer='LTC_BayArea', output_dir=args.output)
    export(args.output / 'human_exposure_ces_tracts.parquet', args.output / 'tables')
    comparisons = []
    if args.compare:
        for file in args.output.rglob('*'):
            if not file.is_file() or file.suffix not in ('.parquet','.csv','.png','.html'):
                continue
            rel = file.relative_to(args.output)
            other = args.compare / rel
            if file.suffix == '.parquet':
                a, b = pd.read_parquet(file), pd.read_parquet(other)
                pd.testing.assert_frame_equal(a,b,check_dtype=False,rtol=1e-9,atol=1e-9)
            elif file.suffix == '.csv':
                pd.testing.assert_frame_equal(pd.read_csv(file),pd.read_csv(other),check_dtype=False,rtol=1e-9,atol=1e-9)
            elif file.suffix == '.png':
                import numpy as np
                from PIL import Image
                with Image.open(file) as a, Image.open(other) as b:
                    np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            else:
                assert file.read_bytes() == other.read_bytes(), rel
            comparisons.append(str(rel))
    (args.output / 'replay_audit.json').write_text(json.dumps(dict(status='passed',
        imported_code=str(impacts.__file__), generated=outputs, compared_files=comparisons), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
