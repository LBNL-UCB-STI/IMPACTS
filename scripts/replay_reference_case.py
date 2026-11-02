"""Recompute a saved traffic case into a NEW directory with an explicit policy.

Input manifests and annual emissions are read-only; this driver deliberately
does not claim to reconstruct BEAM trips or regenerate annual emissions.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import pyarrow.parquet as pq
import yaml


def run_stage(args):
    import impacts
    from impacts.manifest.schema import PipelineConfig
    from impacts.pipeline.workflow import step2_compute_inmap_concentrations as im
    from impacts.pipeline.workflow import step3_compute_aermod_concentrations as ae
    from impacts.pipeline.workflow import step4_prepare_exposure as ex
    import geopandas as gpd

    assert Path(impacts.__file__).resolve().is_relative_to(ROOT)
    ref, out = args.reference.resolve(), args.output.resolve()
    if out == ref or ref.is_relative_to(out) or out.is_relative_to(ref):
        raise ValueError('Output must be separate from the reference directory.')
    manifest = yaml.safe_load((ref / 'pipeline_manifest.yaml').read_text(encoding='utf-8'))
    pipeline_data = yaml.safe_load((ref / 'preprocess_manifest.yaml').read_text(encoding='utf-8'))['pipeline']
    pipeline_data['power_generation'] = {'enabled': False}
    pipeline_data['primary_pm25_integration_strategy'] = 'additive_inmap_aermod'
    if args.isrm:
        pipeline_data['isrm_url'] = str(args.isrm.resolve())
    pipeline = PipelineConfig.from_dict(pipeline_data)
    if args.policy == 'legacy4500':
        im.build_near_source_mask = functools.partial(im.build_near_source_mask, radius_meters=4500.)
        ae._AERMOD_RESPONSE_RADIUS_METERS = 1e30
    for folder in ('concentrations', 'exposure', 'cache'):
        (out / folder).mkdir(parents=True, exist_ok=True)
    if args.stage == 'inmap':
        im.run(pipeline=pipeline, raw_dir=out / 'concentrations',
               emissions_input_path=manifest['outputs']['beam_emissions_for_inmap'],
               inmap_study_area_grid_path=pipeline.inmap_grid_path)
    elif args.stage == 'aermod':
        # Same memory-bounded convolution used to create the reference cases.
        from gateway_aermod_crop import apply_cropped
        ae._apply_kernels = apply_cropped
        def parquet_only(gdf, output_path):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            gdf.to_parquet(output_path, index=False)
        ae._write_outputs = parquet_only
        ae.run(pipeline=pipeline, raw_dir=out / 'concentrations', cache_dir=out / 'cache',
               emissions_input_path=manifest['outputs']['beam_emissions_for_aermod'])
    elif args.stage == 'merge':
        data = ex._build_full_exposure_grid(pipeline=pipeline,
            prepared_inmap=ex._prepare_inmap_exposure_inputs(gpd.read_parquet(out / 'concentrations/beam_inmap_concentrations.parquet')),
            prepared_aermod=ex._prepare_aermod_exposure_inputs(gpd.read_parquet(out / 'concentrations/beam_aermod_concentrations.parquet')),
            working_dir=out / 'exposure')
        data.to_parquet(out / 'exposure/beam_concentration_distribution.parquet', index=False)
    (out / 'replay_config.json').write_text(json.dumps(dict(
        imported_code=str(impacts.__file__), reference=str(ref), policy=args.policy,
        source_layer='traffic ground layer', power_generation=False,
        emissions_reused=True, additional_expansion=1,
        original_inputs=manifest['outputs'], pipeline=pipeline_data), indent=2), encoding='utf-8')


def compare(ref, output):
    comparisons = []
    for rel, key in [('concentrations/beam_inmap_concentrations.parquet', 'inmap_cell_id'),
                     ('concentrations/beam_aermod_concentrations.parquet', 'aermod_cell_id'),
                     ('exposure/beam_concentration_distribution.parquet', 'aermod_cell_id')]:
        old, new = [pq.ParquetFile(p / rel) for p in (ref, output)]
        assert old.metadata.num_rows == new.metadata.num_rows, rel
        assert set(old.schema_arrow.names) == set(new.schema_arrow.names), rel
        numeric = [f.name for f in new.schema_arrow if f.name != key and
                   (str(f.type).startswith(('float', 'double', 'int', 'uint', 'bool')))]
        differences = dict.fromkeys(numeric, 0.)
        n = 0
        for a, b in zip(old.iter_batches(batch_size=65536), new.iter_batches(batch_size=65536), strict=True):
            a, b = a.to_pandas(), b.to_pandas()
            np.testing.assert_array_equal(a[key], b[key], err_msg=rel)
            if 'geometry' in a:
                assert (a.geometry == b.geometry).all(), 'Geometry changed: ' + rel
            for col in numeric:
                x, y = a[col].to_numpy(), b[col].to_numpy()
                np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-10, equal_nan=True, err_msg=rel + ':' + col)
                finite = np.isfinite(x) & np.isfinite(y)
                if finite.any():
                    differences[col] = max(differences[col], float(np.max(np.abs(x[finite].astype(float) - y[finite].astype(float)))))
            n += len(a)
        comparisons.append(dict(file=rel, rows=n, max_absolute_difference=differences, geometry_and_ids_equal=True))
    (output / 'comparison.json').write_text(json.dumps(dict(status='passed', comparisons=comparisons), indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--isrm', type=Path)
    parser.add_argument('--policy', choices=['circle3000', 'legacy4500'], required=True)
    parser.add_argument('--stage', choices=['all', 'inmap', 'aermod', 'merge', 'compare'], default='all')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.stage == 'all':
        if args.output.exists():
            raise FileExistsError('Choose a fresh output directory: ' + str(args.output))
        (args.output / 'logs').mkdir(parents=True)
        for stage in ['inmap', 'aermod', 'merge', 'compare']:
            print('START', stage, flush=True)
            cmd = [sys.executable, '-X', 'utf8', __file__, '--reference', str(args.reference),
                   '--output', str(args.output), '--policy', args.policy, '--stage', stage]
            if args.isrm:
                cmd += ['--isrm', str(args.isrm)]
            with (args.output / 'logs' / (stage + '.log')).open('w', encoding='utf-8') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            print('DONE', stage, flush=True)
    elif args.stage == 'compare':
        compare(args.reference, args.output)
    else:
        run_stage(args)


if __name__ == '__main__':
    main()
