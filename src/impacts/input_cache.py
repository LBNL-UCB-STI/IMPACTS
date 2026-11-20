from __future__ import annotations

import hashlib
from pathlib import Path


def emfac_raw_cache_root(archive: Path, shared_cache_root: Path) -> Path:
    stat = archive.stat()
    identity = f"{archive.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return shared_cache_root / "emfac_raw" / key
