from pathlib import Path

import pytest

from impacts.common import resolve_beam_network_local_path
from impacts.common import resolve_emissions_skims_local_path


def test_beam_network_resolution_rejects_duplicate_run_matches(tmp_path: Path) -> None:
    first = tmp_path / "run-a" / "network.csv.gz"
    second = tmp_path / "run-b" / "network.csv.gz"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="Multiple BEAM network files"):
        resolve_beam_network_local_path(str(tmp_path))


def test_beam_skims_resolution_rejects_multiple_iteration_trees(tmp_path: Path) -> None:
    (tmp_path / "run-a" / "ITERS").mkdir(parents=True)
    (tmp_path / "run-b" / "ITERS").mkdir(parents=True)

    with pytest.raises(ValueError, match="Multiple BEAM ITERS directories"):
        resolve_emissions_skims_local_path(str(tmp_path))
