"""Stage a versioned Census bundle into the persistent IMPACTS input cache."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory
from typing import Any

from .acs_data import ACS_TABLES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_census_bundle(
    archive: Path, cache_root: Path, *, year: int, region: str, state_fips: str
) -> dict[str, Any]:
    bundle_id = archive.name.removesuffix(".tar.gz")
    manifest_name = f"census/{bundle_id}.manifest.json"
    archive_sha256 = _sha256(archive)
    cache_dir = cache_root / "census" / "bundles" / f"{bundle_id}-{archive_sha256}"
    acs_tables = {
        period: {table: f"acs/{year}/{period}/{table.lower()}.dat" for table in tables}
        for period, tables in ACS_TABLES.items()
    }

    def validate_manifest(manifest: dict) -> dict[str, dict]:
        if (
            manifest["bundle_id"] != bundle_id
            or manifest["baseline_year"] != year
            or manifest["scenario_region"] != region
        ):
            raise ValueError(f"Census bundle {archive.name} does not match {region}, baseline year {year}")
        files = {}
        for entry in manifest["files"] + manifest.get("references", []):
            name = entry["path"]
            path = Path(name)
            if not path.parts or path.is_absolute() or ".." in path.parts or name in files:
                raise ValueError(f"Invalid file path in Census bundle: {name}")
            files[name] = entry
        return files

    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            manifest_member = tar.getmember(manifest_name)
        except KeyError as exc:
            raise ValueError(f"Census bundle is missing its source manifest: {manifest_name}") from exc
        if not manifest_member.isfile():
            raise ValueError(f"Census bundle manifest is not a regular file: {manifest_name}")
        with tar.extractfile(manifest_member) as stream:
            source_manifest_bytes = stream.read()
        try:
            manifest = json.loads(source_manifest_bytes)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON in Census bundle manifest: {manifest_name}") from exc
        entries = validate_manifest(manifest)

        if not cache_dir.is_dir():
            with TemporaryDirectory(dir=cache_dir.parent) as temp:
                stage = Path(temp) / "bundle"
                for name, entry in entries.items():
                    try:
                        member = tar.getmember(name)
                    except KeyError as exc:
                        raise ValueError(f"Census bundle is missing declared file: {name}") from exc
                    if not member.isfile() or member.size != entry["bytes"]:
                        raise ValueError(f"Invalid Census bundle member: {name}")
                    target = stage / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as source, target.open("wb") as dest:
                        shutil.copyfileobj(source, dest)
                    if _sha256(target) != entry["sha256"]:
                        raise ValueError(f"Census bundle checksum mismatch: {name}")
                manifest_path = stage / manifest_name
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_bytes(source_manifest_bytes)
                try:
                    stage.rename(cache_dir)
                except OSError:
                    if not cache_dir.is_dir():
                        raise
        else:
            manifest_path = cache_dir / manifest_name
            if not manifest_path.is_file() or manifest_path.read_bytes() != source_manifest_bytes:
                raise ValueError(f"Cached Census bundle manifest does not match its source archive: {manifest_path}")
            for name, entry in entries.items():
                path = cache_dir / name
                if not path.is_file() or path.stat().st_size != entry["bytes"] or _sha256(path) != entry["sha256"]:
                    raise ValueError(f"Cached Census bundle checksum mismatch or missing file: {path}")

    manifest_path = cache_dir / manifest_name

    def input_path(name: str) -> str:
        if name not in entries:
            raise ValueError(f"Census bundle {bundle_id} is missing a manifest entry for {name}")
        path = cache_dir / name
        if not path.is_file() or _sha256(path) != entries[name]["sha256"]:
            raise ValueError(f"Cached Census bundle checksum mismatch or missing file: {path}")
        return str(path)

    p42 = [name for name, entry in entries.items() if entry.get("table") == "P42"]
    if len(p42) != 1:
        raise ValueError(f"Census bundle {bundle_id} must declare one P42 archive")
    luca = f"census/luca/{year}/2020LUCA_ST{state_fips}_address_countlist.csv"
    if entries.get(luca, {}).get("field") != "CURRENT_COUNT_OF_GQs":
        raise ValueError(f"Census bundle {bundle_id} must declare LUCA block address counts for {state_fips}, {year}")
    occupancy = manifest.get("group_quarters_occupancy")
    return {
        "bundle_id": bundle_id,
        "archive_sha256": archive_sha256,
        "baseline_year": int(manifest["baseline_year"]),
        "scenario_region": str(manifest["scenario_region"]),
        "manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_vintages": sorted({str(entry["vintage"]) for entry in manifest["files"] if entry.get("vintage")}),
        "source_files": [
            {
                key: entry[key]
                for key in ("path", "dataset", "table", "vintage", "product", "geography", "field", "sha256")
                if key in entry
            }
            for entry in manifest["files"]
        ],
        "source_references": [
            {
                key: entry[key]
                for key in ("path", "title", "sha256")
                if key in entry
            }
            for entry in manifest.get("references", [])
        ],
        "acs_tables": {
            period: {table: input_path(name) for table, name in tables.items()}
            for period, tables in acs_tables.items()
        },
        "p42": input_path(p42[0]),
        "luca": input_path(luca),
        "occupancy": input_path(occupancy) if occupancy else None,
    }
