from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional


def _parse_scalar(raw: str):
    value = raw.strip()
    if value in ("", "null", "Null", "NULL", "none", "None", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml_load(text: str):
    data: Dict[str, Any] = {}
    current_section: Optional[str] = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            continue
        key, raw_val = line.split(":", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if indent == 0:
            if raw_val == "":
                data[key] = {}
                current_section = key
            else:
                data[key] = _parse_scalar(raw_val)
                current_section = None
        else:
            if current_section is None:
                continue
            if not isinstance(data[current_section], dict):
                data[current_section] = {}
            data[current_section][key] = _parse_scalar(raw_val)
    return data


def load_structured_file(path: str | Path) -> Dict[str, Any]:
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text) or {}
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    loaded = _simple_yaml_load(text)
    if isinstance(loaded, dict):
        return loaded
    raise ValueError(f"Could not parse structured file: {manifest_path}")


def write_structured_file(path: str | Path, payload: Dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore

        text = yaml.safe_dump(payload, sort_keys=False)
    except Exception:
        text = json.dumps(payload, indent=2, sort_keys=False)
    output_path.write_text(text, encoding="utf-8")


def resolve_path(path: Optional[str], config_path: str | Path | None = None) -> Optional[str]:
    if path is None:
        return None
    raw = str(path).strip()
    if not raw:
        return raw
    if is_remote_path(raw):
        return raw
    resolved = Path(raw)
    if resolved.is_absolute():
        return str(resolved)
    if config_path is None:
        return str(resolved.resolve())
    return str((Path(config_path).resolve().parent / resolved).resolve())


def is_remote_path(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "http://", "https://"))


def sha256_path(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_dir():
        for child in sorted(p for p in target.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(target)).encode("utf-8"))
            digest.update(sha256_path(child).encode("utf-8"))
        return digest.hexdigest()

    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_path(source: str | Path, destination: str | Path) -> Path:
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif src.suffix.lower() == ".shp":
        base = src.with_suffix("")
        dest_base = dst.with_suffix("")
        for sibling in src.parent.glob(f"{base.name}.*"):
            target = dest_base.with_suffix(sibling.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sibling, target)
    else:
        shutil.copy2(src, dst)
    return dst


def file_entry(
    *,
    kind: str,
    path: str | Path,
    staged_path: str | Path,
    optional: bool = False,
) -> Dict[str, Any]:
    src = Path(path)
    staged = Path(staged_path)
    entry: Dict[str, Any] = {
        "kind": kind,
        "source_path": str(src),
        "staged_path": str(staged),
        "optional": optional,
        "exists": src.exists(),
    }
    if src.exists():
        entry["sha256"] = sha256_path(src)
    return entry


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import fastparquet  # noqa: F401

        return True
    except Exception:
        return False
