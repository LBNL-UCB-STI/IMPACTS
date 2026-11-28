from __future__ import annotations

import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from urllib.error import HTTPError
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import yaml

REFERENCE_FILENAME = "beam_concentration_distribution.parquet"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_reference(source: Path) -> Path:
    if source.is_file():
        return source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Reference scenario path does not exist: {source}")

    direct = [
        source / REFERENCE_FILENAME,
        source / "exposure" / REFERENCE_FILENAME,
        source / "postsim" / "exposure" / REFERENCE_FILENAME,
    ]
    matches = {path.resolve() for path in direct if path.is_file()}
    if not matches:
        matches = {path.resolve() for path in source.rglob(REFERENCE_FILENAME) if path.is_file()}
    if len(matches) != 1:
        candidates = "\n".join(f"  - {path}" for path in sorted(matches)) or "  (none)"
        raise ValueError(
            "Reference scenario directory must resolve to exactly one "
            f"{REFERENCE_FILENAME}; found {len(matches)} candidates:\n{candidates}. "
            "Select the exact postsim directory."
        )
    return next(iter(matches))


def _http_read(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def _http_source(source: str) -> tuple[bytes, str, str | None]:
    parsed = urlparse(source)
    if parsed.path.lower().endswith(".parquet"):
        return _http_read(source), source, None

    base = source.rstrip("/") + "/"
    manifest = None
    manifest_url = ""
    errors = []
    for candidate in (
        urljoin(base, "postsim_manifest.yaml"),
        urljoin(base, "postsim/postsim_manifest.yaml"),
    ):
        try:
            manifest = yaml.safe_load(_http_read(candidate)) or {}
            manifest_url = candidate
            break
        except HTTPError as exc:
            errors.append(f"{candidate} (HTTP {exc.code})")
    if manifest is None:
        raise FileNotFoundError(
            "Could not read a published postsim manifest. Tried: " + ", ".join(errors)
        )
    outputs = manifest.get("outputs", {}) if isinstance(manifest, dict) else {}
    entry = outputs.get("beam_concentration_distribution") if isinstance(outputs, dict) else None
    if entry is None:
        postprocess_outputs = manifest.get("postprocess_outputs", {}) if isinstance(manifest, dict) else {}
        entry = postprocess_outputs.get("beam_concentration_distribution") if isinstance(postprocess_outputs, dict) else None
    if isinstance(entry, dict):
        relative = entry.get("path") or entry.get("relative_path")
        manifest_sha = entry.get("sha256")
    else:
        relative = entry
        manifest_sha = None
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(
            f"Published manifest {manifest_url} has no relative concentration-distribution path."
        )
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Published manifest contains an unsafe relative artifact path: {relative!r}")
    artifact_url = urljoin(manifest_url, relative)
    artifact = urlparse(artifact_url)
    if (artifact.scheme, artifact.netloc) != (parsed.scheme, parsed.netloc):
        raise ValueError("Published reference artifact must be served by the same HTTP origin as its manifest.")
    return _http_read(artifact_url), artifact_url, manifest_sha


def _read_analysis_reference(run_root: Path) -> dict | None:
    path = run_root / "analysis" / "analysis_manifest.yaml"
    if not path.is_file():
        return None
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    value = payload.get("reference_scenario") if isinstance(payload, dict) else None
    return value if isinstance(value, dict) else None


def pin_reference_scenario(
    source: str,
    run_root: str | Path,
    *,
    reference_root: str | Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, str | bool]:
    """Pin one reference file per IMPACTS run; later iterations reuse its exact bytes."""
    root = Path(run_root).expanduser().resolve()
    reference_dir = Path(reference_root).expanduser().resolve() if reference_root else root / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    destination = reference_dir / REFERENCE_FILENAME
    lock_path = reference_dir / ".reference.lock"

    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        prior = _read_analysis_reference(root)
        if destination.is_file():
            actual_sha = _sha256(destination)
            if expected_sha256 and actual_sha.lower() != expected_sha256.lower():
                raise ValueError(
                    "Pinned reference scenario checksum differs from the configured checksum: "
                    f"expected {expected_sha256}, found {actual_sha}. The saved run reference was not changed."
                )
            return {
                "source": str((prior or {}).get("source") or source),
                "resolved_source": str((prior or {}).get("resolved_source") or "already pinned for this run"),
                "saved_path": os.path.relpath(destination, root),
                "sha256": actual_sha,
                "reused": True,
            }

        parsed = urlparse(source)
        manifest_sha = None
        if parsed.scheme in {"http", "https"}:
            content, resolved_source, manifest_sha = _http_source(source)
            with tempfile.NamedTemporaryFile(dir=reference_dir, prefix=".reference-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(content)
        elif parsed.scheme:
            raise ValueError(f"Unsupported reference scenario URL scheme: {parsed.scheme}")
        else:
            resolved = _local_reference(Path(source).expanduser())
            resolved_source = str(resolved)
            with tempfile.NamedTemporaryFile(dir=reference_dir, prefix=".reference-", delete=False) as stream:
                temporary = Path(stream.name)
            shutil.copyfile(resolved, temporary)

        try:
            actual_sha = _sha256(temporary)
            if expected_sha256 and manifest_sha and expected_sha256.lower() != str(manifest_sha).lower():
                raise ValueError(
                    "Configured and published reference checksums disagree: "
                    f"{expected_sha256} != {manifest_sha}."
                )
            required_sha = expected_sha256 or manifest_sha
            if required_sha and actual_sha.lower() != str(required_sha).lower():
                raise ValueError(
                    "Reference scenario checksum mismatch: "
                    f"expected {required_sha}, downloaded/copied {actual_sha}."
                )
            # link() is atomic and refuses to replace an already pinned reference.
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    return {
        "source": source,
        "resolved_source": resolved_source,
        "saved_path": os.path.relpath(destination, root),
        "sha256": actual_sha,
        "reused": False,
    }


def write_analysis_manifest(
    path: str | Path,
    *,
    outputs: dict,
    reference_scenario: dict | None,
    run_root: str | Path,
) -> dict:
    target = Path(path)
    root = Path(run_root).resolve()
    analysis_root = target.parent.resolve()
    relative_outputs = {}
    for key, value in outputs.items():
        if isinstance(value, str):
            candidate = Path(value)
            if candidate.is_absolute() and candidate.resolve().is_relative_to(root):
                value = os.path.relpath(candidate.resolve(), analysis_root)
        relative_outputs[key] = value
    if reference_scenario:
        reference_scenario = dict(reference_scenario)
        pinned_path = root / str(reference_scenario["saved_path"])
        reference_scenario["saved_path"] = os.path.relpath(pinned_path, analysis_root)
    payload = {
        "contract_version": "1",
        "model": "impacts",
        "analysis_outputs": relative_outputs,
        "reference_scenario": reference_scenario,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    os.replace(temporary, target)
    return payload


def validate_reference_grid(current_path: str | Path, reference_path: str | Path) -> None:
    """Require matching pollutant fields and an identical AERMOD-to-InMAP cell grid."""
    import duckdb

    required = {
        "aermod_cell_id",
        "inmap_cell_id",
        "TotalPM25",
        "PrimaryPM25",
        "SecondaryPM25",
        "BC",
        "NO2",
    }
    connection = duckdb.connect()
    try:
        current = Path(current_path).as_posix().replace("'", "''")
        reference = Path(reference_path).as_posix().replace("'", "''")
        current_columns = {
            str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{current}')").fetchall()
        }
        reference_columns = {
            str(row[0]) for row in connection.execute(f"DESCRIBE SELECT * FROM read_parquet('{reference}')").fetchall()
        }
        missing = sorted(required - current_columns), sorted(required - reference_columns)
        if missing[0] or missing[1]:
            raise ValueError(
                "Reference scenario pollutant/grid schema is incompatible; "
                f"current missing={missing[0]}, reference missing={missing[1]}"
            )
        mismatch = connection.execute(
            f"""
            SELECT COUNT(*) FROM (
                (SELECT aermod_cell_id, inmap_cell_id FROM read_parquet('{current}')
                 EXCEPT SELECT aermod_cell_id, inmap_cell_id FROM read_parquet('{reference}'))
                UNION ALL
                (SELECT aermod_cell_id, inmap_cell_id FROM read_parquet('{reference}')
                 EXCEPT SELECT aermod_cell_id, inmap_cell_id FROM read_parquet('{current}'))
            )
            """
        ).fetchone()[0]
        if mismatch:
            raise ValueError(
                "Reference scenario uses a different AERMOD/InMAP grid or cell mapping "
                f"({int(mismatch)} unmatched mappings)."
            )
    finally:
        connection.close()
