from pathlib import Path

import pytest

from impacts.common import resolve_beam_network_local_path
from impacts.common import resolve_emissions_skims_local_path
from impacts.common import resolve_latest_events_local_path


def test_network_resolution_rejects_multiple_run_candidates(tmp_path: Path) -> None:
    first = tmp_path / "run-a" / "network.csv.gz"
    second = tmp_path / "run-b" / "network.csv.gz"
    first.parent.mkdir()
    second.parent.mkdir()
    first.touch()
    second.touch()

    with pytest.raises(ValueError, match="Multiple BEAM network"):
        resolve_beam_network_local_path(str(tmp_path))


def test_skims_resolution_uses_highest_iteration_in_single_tree(tmp_path: Path) -> None:
    iters = tmp_path / "ITERS"
    earlier = iters / "it.0" / "0.skimsEmissions.parquet"
    latest = iters / "it.2" / "2.skimsEmissions.parquet"
    earlier.parent.mkdir(parents=True)
    latest.parent.mkdir(parents=True)
    earlier.touch()
    latest.touch()

    assert resolve_emissions_skims_local_path(str(tmp_path)) == str(latest)


def test_skims_resolution_rejects_multiple_run_trees(tmp_path: Path) -> None:
    for run in ("run-a", "run-b"):
        skims = tmp_path / run / "ITERS" / "it.0" / "0.skimsEmissions.parquet"
        skims.parent.mkdir(parents=True)
        skims.touch()

    with pytest.raises(ValueError, match="Multiple BEAM ITERS"):
        resolve_emissions_skims_local_path(str(tmp_path))


def test_events_resolution_uses_same_single_iteration_tree(tmp_path: Path) -> None:
    events = tmp_path / "ITERS" / "it.1" / "1.events.parquet"
    events.parent.mkdir(parents=True)
    events.touch()

    assert resolve_latest_events_local_path(str(tmp_path)) == str(events)
