"""Exact support-bounded evaluation of the original AERMOD FFT operator.

Each source-pattern group is evaluated by the unmodified original function on
its source bounding box plus full kernel support. Two zero-emission receptor
anchors preserve rectangular grid bounds even when the receptor grid has holes.
No source, kernel element, real receptor, or concentration cutoff is discarded.
"""
import numpy as np
import pandas as pd

from impacts.pipeline.workflow.step3_compute_aermod_concentrations import _apply_kernels as original_apply


def apply_cropped(*, source_df, target_index, target_id_col, emissions_cols, kernel_library):
    result = original_apply(source_df=source_df.iloc[:0], target_index=target_index,
        target_id_col=target_id_col, emissions_cols=emissions_cols, kernel_library=kernel_library)
    columns = [c for c in result if c != target_id_col]
    arrays = {c: result[c].to_numpy(copy=True) for c in columns}
    tx = target_index.target_ix.to_numpy()
    ty = target_index.target_iy.to_numpy()
    xmin, xmax, ymin, ymax = tx.min(), tx.max(), ty.min(), ty.max()
    sentinel = int(target_index[target_id_col].max()) + 1
    for key, group in source_df.groupby('pattern_key', dropna=False):
        kernel = kernel_library.get(str(key))
        if kernel is None or kernel['response_per_ton'].size == 0:
            raise ValueError(f'No AERMOD ASRV kernel available for pattern {key}')
        valid = group.source_ix.between(xmin, xmax) & group.source_iy.between(ymin, ymax)
        group = group.loc[valid]
        if group.empty:
            continue
        dx, dy = np.abs(kernel['dix']).max(), np.abs(kernel['diy']).max()
        x0, x1 = max(xmin, group.source_ix.min()-dx), min(xmax, group.source_ix.max()+dx)
        y0, y1 = max(ymin, group.source_iy.min()-dy), min(ymax, group.source_iy.max()+dy)
        mask = (tx >= x0) & (tx <= x1) & (ty >= y0) & (ty <= y1)
        pos = np.flatnonzero(mask)
        roi = target_index.loc[mask, [target_id_col, 'target_ix', 'target_iy']]
        anchors = pd.DataFrame({target_id_col:[sentinel, sentinel+1], 'target_ix':[x0,x1], 'target_iy':[y0,y1]})
        roi = pd.concat([roi, anchors], ignore_index=True)
        partial = original_apply(source_df=group, target_index=roi, target_id_col=target_id_col,
            emissions_cols=emissions_cols, kernel_library={str(key):kernel})
        for c in columns:
            values = partial[c].to_numpy()[:len(pos)]
            if arrays[c].dtype == bool:
                arrays[c][pos] |= values
            else:
                arrays[c][pos] += values
    for c, values in arrays.items():
        result[c] = values
    return result
