"""Resolve local ISRM paths and cache public remote Zarr stores persistently."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from urllib.parse import urlsplit

from .manifest.file_ops import is_remote_path


def localize_isrm_store(source: str, cache_root: Path) -> str:
    """Copy a local or remote ISRM store into a persistent cache and reuse it."""
    remote = is_remote_path(source)
    if remote:
        protocol = urlsplit(source).scheme
        identity = source
        basename = PurePosixPath(urlsplit(source).path.rstrip("/")).name or "isrm.zarr"
    else:
        local_source = Path(source).expanduser().resolve()
        identity = str(local_source)
        basename = local_source.name

    destination = cache_root / hashlib.sha256(identity.encode("utf-8")).hexdigest() / basename
    if destination.exists():
        return str(destination)

    if remote:
        import fsspec

        fs, remote_path = fsspec.core.url_to_fs(source, **({"anon": True} if protocol == "s3" else {}))
        remote_path = remote_path.rstrip("/")
        files = fs.find(remote_path, withdirs=False)
        if not files:
            raise FileNotFoundError(f"Remote ISRM Zarr store is empty or unavailable: {source}")
    elif not local_source.is_dir():
        raise FileNotFoundError(f"Local ISRM Zarr store is not a directory: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=".isrm-", dir=destination.parent))
    staged = temporary_root / basename
    try:
        if remote:
            staged.mkdir()
            prefix = fs._strip_protocol(remote_path).rstrip("/")
            for remote_file in files:
                normalized = fs._strip_protocol(remote_file)
                relative = normalized[len(prefix):].lstrip("/")
                if not relative:
                    relative = PurePosixPath(normalized).name
                relative_path = PurePosixPath(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(f"Unsafe path in remote ISRM Zarr listing: {remote_file}")
                local_file = staged.joinpath(*relative_path.parts)
                local_file.parent.mkdir(parents=True, exist_ok=True)
                fs.get_file(remote_file, str(local_file))
        else:
            shutil.copytree(local_source, staged)
        try:
            os.rename(staged, destination)
        except OSError:
            # Another run populated this same content-addressed cache entry.
            if not destination.exists():
                raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return str(destination)
