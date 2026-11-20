from pathlib import Path

from impacts.input_cache import emfac_raw_cache_root


def test_emfac_raw_cache_key_reuses_same_archive_and_changes_with_input(tmp_path: Path):
    archive = tmp_path / "emissions_raw.tar.zst"
    archive.write_bytes(b"archive-v1")
    cache_root = tmp_path / "shared-cache"

    first = emfac_raw_cache_root(archive, cache_root)
    assert emfac_raw_cache_root(archive, cache_root) == first

    archive.write_bytes(b"archive-v2-with-different-size")
    assert emfac_raw_cache_root(archive, cache_root) != first
