"""Build a workflow-ready full-domain NOx-to-NO2 matrix.

This utility:
1. Runs the one-off NOx/NO2 preprocessing pipeline.
2. Reads the generated sparse/regional ``nox_to_no2_regional_grid_matrix.parquet``.
3. Expands it to the full ISRM source/receptor domain using ``isrm_v1.2.1.zarr``.

The result is a parquet matrix that can be passed directly as
``isrm_nox_to_no2_matrix`` into the maintained workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .isrm_nox_to_no2 import REGIONAL_NOX_TO_NO2_MATRIX_NAME
from .isrm_nox_to_no2 import run_pipeline


DEFAULT_LOCAL_ISRM = "isrm_v1.2.1.zarr"
DEFAULT_S3_ISRM = "s3://inmap-model/isrm_v1.2.1.zarr/"
DEFAULT_OUTPUT_NAME = "nox_to_no2_full_isrm_matrix.parquet"
DEFAULT_WRITE_CHUNK_ROWS = 256


def _load_isrm_store(isrm_url: str):
    import zarr

    if isrm_url.startswith("s3://"):
        try:
            import s3fs  # noqa: F401
        except ImportError as exc:
            raise ImportError("s3fs is required to read ISRM zarr from s3:// URLs") from exc
        return zarr.open(
            isrm_url,
            mode="r",
            storage_options={"anon": True, "client_kwargs": {"region_name": "us-east-2"}},
        )

    return zarr.open(isrm_url, mode="r")


def _resolve_isrm_url(isrm_zarr: str | None) -> str:
    if isrm_zarr:
        return isrm_zarr

    local_path = Path(DEFAULT_LOCAL_ISRM).resolve()
    if local_path.exists():
        return str(local_path)
    return DEFAULT_S3_ISRM


def _read_transfer_matrix(path: str | Path) -> pd.DataFrame:
    matrix = pd.read_parquet(path)
    matrix.index = pd.to_numeric(pd.Index(matrix.index), errors="coerce")
    matrix.columns = pd.to_numeric(pd.Index(matrix.columns), errors="coerce")
    matrix = matrix.loc[matrix.index.notna(), matrix.columns.notna()].copy()
    matrix.index = matrix.index.astype(int)
    matrix.columns = matrix.columns.astype(int)
    return matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _expand_to_full_isrm_domain(
    matrix: pd.DataFrame,
    *,
    source_dim: int,
    receptor_dim: int,
) -> pd.DataFrame:
    full_index = pd.Index(range(source_dim), dtype=int)
    full_columns = pd.Index(range(receptor_dim), dtype=int)
    return matrix.reindex(index=full_index, columns=full_columns, fill_value=0.0)


def _write_parquet_with_progress(
    matrix: pd.DataFrame,
    output_path: str | Path,
    *,
    chunk_rows: int = DEFAULT_WRITE_CHUNK_ROWS,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path = Path(output_path)
    writer = None
    progress = tqdm(
        total=matrix.shape[0],
        desc="Writing full ISRM matrix",
        unit="row",
        dynamic_ncols=True,
    )
    try:
        for start in range(0, matrix.shape[0], chunk_rows):
            stop = min(start + chunk_rows, matrix.shape[0])
            chunk = matrix.iloc[start:stop]
            table = pa.Table.from_pandas(chunk, preserve_index=True)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
            progress.update(stop - start)
    finally:
        if writer is not None:
            writer.close()
        progress.close()


def build_complete_matrix(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    isrm_zarr: str | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> Path:
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_pipeline(input_dir=str(input_root), output_dir=str(output_root))

    sparse_matrix_path = output_root / REGIONAL_NOX_TO_NO2_MATRIX_NAME
    if not sparse_matrix_path.exists():
        raise FileNotFoundError(f"Expected {sparse_matrix_path} after running NOx/NO2 preprocessing")

    isrm_url = _resolve_isrm_url(isrm_zarr)
    sr = _load_isrm_store(isrm_url)
    source_dim = int(sr["SOA"].shape[1])
    receptor_dim = int(sr["SOA"].shape[2])

    sparse_matrix = _read_transfer_matrix(sparse_matrix_path)
    full_matrix = _expand_to_full_isrm_domain(
        sparse_matrix,
        source_dim=source_dim,
        receptor_dim=receptor_dim,
    )

    output_path = output_root / output_name
    _write_parquet_with_progress(full_matrix, output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.utils.build_complete_nox_to_no2_matrix",
        description="Build a full-domain workflow-ready NOx-to-NO2 ISRM matrix.",
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing NOx/NO2 preprocessing inputs")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet outputs will be written")
    parser.add_argument(
        "--isrm-zarr",
        help=(
            "Local path or s3:// URL for the ISRM zarr store. "
            f"Defaults to ./{DEFAULT_LOCAL_ISRM} if present, otherwise {DEFAULT_S3_ISRM}"
        ),
    )
    parser.add_argument(
        "--output-name",
        default=DEFAULT_OUTPUT_NAME,
        help=f"Filename for the full-domain matrix. Default: {DEFAULT_OUTPUT_NAME}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_path = build_complete_matrix(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        isrm_zarr=args.isrm_zarr,
        output_name=args.output_name,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
