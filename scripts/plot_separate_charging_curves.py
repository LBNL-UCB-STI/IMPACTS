"""Plot each saved vehicle-type charging series without rerunning the model."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from impacts.pipeline.workflow.charging_plots import write_separate_charging_curves
from impacts.pipeline.workflow.power_generation import _safe_profile_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demand', required=True, type=Path)
    parser.add_argument('--original-summary', type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    labels = {}
    if args.original_summary:
        summary = json.loads(args.original_summary.read_text(encoding='utf-8'))
        labels = {_safe_profile_token(t): t for t in summary['eligible_vehicle_types']}
    before = hashlib.sha256(args.demand.read_bytes()).hexdigest()
    manifest = write_separate_charging_curves(pd.read_parquet(args.demand), args.output_dir, labels=labels)
    assert hashlib.sha256(args.demand.read_bytes()).hexdigest() == before
    print(json.dumps({'independent_curves': len(manifest['curves']), 'overview': manifest['overview'],
                      'source_unchanged_sha256': before, 'output_dir': str(args.output_dir.resolve())}, indent=2))


if __name__ == '__main__':
    main()
