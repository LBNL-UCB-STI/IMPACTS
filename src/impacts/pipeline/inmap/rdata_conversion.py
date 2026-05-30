from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_rdata(path: str | Path) -> dict[str, Any]:
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("pyreadr is required to read .RData files") from exc
    return dict(pyreadr.read_r(str(path)))


def _sanitize_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(name).strip())
    return cleaned or "object"


def _coerce_to_dataframe(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.to_frame()
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.ndim == 0:
            return pd.DataFrame({"value": [arr.item()]})
        if arr.ndim == 1:
            return pd.DataFrame({"value": arr})
        if arr.ndim == 2:
            return pd.DataFrame(arr)
        flat = arr.reshape(arr.shape[0], -1)
        return pd.DataFrame(flat)
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return pd.DataFrame({"key": list(value.keys()), "value": list(value.values())})
    if isinstance(value, (list, tuple)):
        if not value:
            return pd.DataFrame()
        first = value[0]
        if isinstance(first, dict):
            return pd.DataFrame(list(value))
        return pd.DataFrame({"value": list(value)})
    return pd.DataFrame({"value": [value]})


def rdata_to_parquet(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
) -> list[Path]:
    source = Path(input_path).resolve()
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize_name(prefix or source.stem)
    objects = read_rdata(source)
    if not objects:
        raise ValueError(f"No objects found in {source}")

    written: list[Path] = []
    multiple = len(objects) > 1
    for name, value in objects.items():
        df = _coerce_to_dataframe(value)
        object_name = _sanitize_name(name)
        filename = f"{stem}__{object_name}.parquet" if multiple else f"{stem}.parquet"
        out_path = target_dir / filename
        df.to_parquet(out_path, index=False)
        written.append(out_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.inmap.rdata_conversion",
        description="Convert one RData file into one or more parquet files.",
    )
    parser.add_argument("--input", required=True, help="Path to the .RData file")
    parser.add_argument("--output-dir", required=True, help="Directory where parquet files will be written")
    parser.add_argument("--prefix", help="Optional filename prefix; defaults to the RData stem")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    written = rdata_to_parquet(
        input_path=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
    )
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
