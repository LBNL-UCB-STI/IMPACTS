from pathlib import Path
import sys
from types import SimpleNamespace

from impacts.isrm_store import localize_isrm_store


def test_local_isrm_store_is_copied_once_and_reused(tmp_path: Path):
    store = tmp_path / "local.zarr"
    store.mkdir()
    (store / ".zgroup").write_text("zarr")

    first = Path(localize_isrm_store(str(store), tmp_path / "cache"))
    assert first != store
    assert (first / ".zgroup").read_text() == "zarr"
    (store / ".zgroup").unlink()
    store.rmdir()
    assert localize_isrm_store(str(store), tmp_path / "cache") == str(first)


def test_remote_isrm_store_is_cached_and_reused(monkeypatch, tmp_path: Path):
    class FakeFS:
        def __init__(self):
            self.downloads = 0

        def _strip_protocol(self, path):
            return path.removeprefix("s3://bucket/")

        def find(self, path, withdirs=False):
            assert path == "dataset.zarr"
            return ["dataset.zarr/.zgroup", "dataset.zarr/0/0"]

        def get_file(self, source, destination):
            self.downloads += 1
            Path(destination).write_text(source)

    fs = FakeFS()
    calls = []

    def url_to_fs(url, **options):
        calls.append((url, options))
        return fs, "dataset.zarr"

    monkeypatch.setitem(
        sys.modules,
        "fsspec",
        SimpleNamespace(core=SimpleNamespace(url_to_fs=url_to_fs)),
    )
    source = "s3://bucket/dataset.zarr"
    first = Path(localize_isrm_store(source, tmp_path / "cache"))
    assert (first / ".zgroup").read_text() == "dataset.zarr/.zgroup"
    assert (first / "0" / "0").read_text() == "dataset.zarr/0/0"
    second = Path(localize_isrm_store(source, tmp_path / "cache"))

    assert second == first
    assert fs.downloads == 2
    assert calls == [(source, {"anon": True})]
