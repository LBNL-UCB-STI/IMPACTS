"""Group saved charging load: passenger, LDT, MDT and HDT on one plot."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from impacts.pipeline.workflow.charging_plots import write_grouped_charging_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demand', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', action='store_true', help='Separate passenger and truck images.')
    args = parser.parse_args()
    before = hashlib.sha256(args.demand.read_bytes()).hexdigest()
    manifest = write_grouped_charging_curve(pd.read_parquet(args.demand), args.output_dir, split=args.split)
    assert hashlib.sha256(args.demand.read_bytes()).hexdigest() == before
    print(json.dumps({k: manifest[k] for k in ['images', 'daily_MWh', 'peak_MW']}, indent=2))


if __name__ == '__main__':
    main()
