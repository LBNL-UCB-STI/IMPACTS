from __future__ import annotations

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
    records = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        records.append((indent, stripped))

    def parse_block(index: int, indent: int):
        if index >= len(records):
            return {}, index
        current_indent, stripped = records[index]
        if current_indent != indent:
            return {}, index
        if stripped.startswith("- "):
            return parse_list(index, indent)
        return parse_map(index, indent)

    def parse_map(index: int, indent: int):
        data: Dict[str, Any] = {}
        while index < len(records):
            current_indent, stripped = records[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                break
            if stripped.startswith("- "):
                break
            if ":" not in stripped:
                index += 1
                continue
            key, raw_val = stripped.split(":", 1)
            key = key.strip()
            raw_val = raw_val.strip()
            index += 1
            if raw_val != "":
                data[key] = _parse_scalar(raw_val)
                continue
            if index >= len(records):
                data[key] = None
                continue
            next_indent, next_stripped = records[index]
            if next_indent <= current_indent:
                data[key] = None
            elif next_stripped.startswith("- "):
                value, index = parse_list(index, next_indent)
                data[key] = value
            else:
                value, index = parse_map(index, next_indent)
                data[key] = value
        return data, index

    def parse_list(index: int, indent: int):
        items = []
        while index < len(records):
            current_indent, stripped = records[index]
            if current_indent < indent:
                break
            if current_indent != indent or not stripped.startswith("- "):
                break
            value = stripped[2:].strip()
            index += 1
            if value:
                items.append(_parse_scalar(value))
                continue
            if index >= len(records):
                items.append(None)
                continue
            next_indent, next_stripped = records[index]
            if next_indent <= current_indent:
                items.append(None)
            elif next_stripped.startswith("- "):
                value, index = parse_list(index, next_indent)
                items.append(value)
            else:
                value, index = parse_map(index, next_indent)
                items.append(value)
        return items, index

    parsed, _ = parse_block(0, records[0][0] if records else 0)
    return parsed


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
        return None
    if is_remote_path(raw):
        return raw
    resolved = Path(raw).expanduser()
    if resolved.is_absolute():
        return str(resolved)
    if config_path is None:
        return str(resolved.resolve())
    config_relative = (Path(config_path).resolve().parent / resolved).resolve()
    if config_relative.exists():
        return str(config_relative)
    cwd_candidate = (Path.cwd() / resolved).resolve()
    if cwd_candidate.exists():
        return str(cwd_candidate)
    return str(config_relative)


def resolve_required_path(path: str, config_path: str | Path | None, label: str) -> str:
    """Resolve a required root settings path; raise FileNotFoundError with searched locations if not found.

    Relative paths are tried against config-file parent first, then cwd.
    Absolute paths are validated for existence immediately.
    """
    raw = str(path).strip()
    if not raw:
        raise ValueError(f"Missing required config path: {label}")
    if is_remote_path(raw):
        return raw
    expanded = Path(raw).expanduser()
    if expanded.is_absolute():
        if not expanded.exists():
            raise FileNotFoundError(f"{label} not found: {expanded}")
        return str(expanded.resolve())
    tried: list[Path] = []
    if config_path is not None:
        candidate = (Path(config_path).resolve().parent / expanded).resolve()
        tried.append(candidate)
        if candidate.exists():
            return str(candidate)
    cwd_candidate = (Path.cwd() / expanded).resolve()
    tried.append(cwd_candidate)
    if cwd_candidate.exists():
        return str(cwd_candidate)
    raise FileNotFoundError(
        f"{label} not found: '{raw}'. Searched:\n"
        + "\n".join(f"  - {p}" for p in tried)
    )


def is_remote_path(path: str) -> bool:
    return path.startswith(("s3://", "gs://", "http://", "https://"))



def _copy_directory_with_progress(src: Path, dst: Path) -> None:
    from ..common import make_progress  # deferred to avoid circular import with common.py
    files = [path for path in src.rglob("*") if path.is_file()]
    progress = make_progress(f"Staging {src.name}", total=len(files), unit="file")
    try:
        for child in src.rglob("*"):
            relative = child.relative_to(src)
            target = dst / relative
            if child.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, target)
            progress.update(1)
    finally:
        progress.close()


def copy_path(source: str | Path, destination: str | Path) -> Path:
    src = Path(source)
    dst = Path(destination)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        _copy_directory_with_progress(src, dst)
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
    return entry
