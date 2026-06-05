from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

import impacts.common as common_module
from impacts.common import assign_grid_cells_to_zones


def test_assign_grid_cells_to_zones_resolves_boundary_cell_by_overlap(tmp_path: Path):
    grid_path = tmp_path / "aermod_grid.parquet"
    zone_path = tmp_path / "inmap_grid.parquet"

    grid = gpd.GeoDataFrame(
        {"aermod_cell_id": [0]},
        geometry=[box(0.5, 0.0, 1.5, 1.0)],
        crs="EPSG:26910",
    )
    zones = gpd.GeoDataFrame(
        {"inmap_cell_id": [3330, 3349]},
        geometry=[box(0.0, 0.0, 1.0, 1.0), box(1.0, 0.0, 2.0, 1.0)],
        crs="EPSG:26910",
    )
    grid.to_parquet(grid_path, index=False)
    zones.to_parquet(zone_path, index=False)

    output_path = assign_grid_cells_to_zones(
        grid_path=str(grid_path),
        zone_path=str(zone_path),
        zone_id_col="inmap_cell_id",
        output_col="inmap_cell_id",
        target_epsg=26910,
    )

    resolved = gpd.read_parquet(output_path)
    assert len(resolved) == 1
    assert int(resolved.loc[0, "inmap_cell_id"]) == 3330


def test_generate_fishnet_progress_uses_stdout_tqdm_for_real_terminal(monkeypatch, tmp_path: Path):
    calls = []
    updates = []

    class _FakeProgress:
        def update(self, value):
            updates.append(value)

        def close(self):
            updates.append("closed")

    def _fake_tqdm(*args, **kwargs):
        calls.append(kwargs)
        return _FakeProgress()

    monkeypatch.setattr(common_module, "_running_with_real_terminal", lambda: True)
    monkeypatch.setattr(common_module, "tqdm", _fake_tqdm)
    monkeypatch.setattr(common_module, "write_vector", lambda *_args, **_kwargs: None)

    mask = gpd.GeoDataFrame(geometry=[box(0, 0, 20, 10)], crs="EPSG:26910")
    common_module.generate_fishnet_from_bounds(
        bounds=(0, 0, 20, 10),
        mask_gdf=mask,
        cell_size=10,
        target_path=str(tmp_path / "fishnet.parquet"),
        target_epsg=26910,
        cell_id_col="aermod_cell_id",
    )

    assert calls
    assert calls[0]["file"] is sys.stdout
    assert calls[0]["dynamic_ncols"] is True
    assert calls[0]["leave"] is True
    assert updates[-1] == "closed"


def test_generate_fishnet_progress_skips_tqdm_for_batch_logs(monkeypatch, tmp_path: Path):
    def _unexpected_tqdm(*_args, **_kwargs):
        raise AssertionError("batch-log fishnet progress should not create a tqdm bar")

    monkeypatch.setattr(common_module, "_running_with_real_terminal", lambda: False)
    monkeypatch.setattr(common_module, "tqdm", _unexpected_tqdm)
    monkeypatch.setattr(common_module, "write_vector", lambda *_args, **_kwargs: None)

    mask = gpd.GeoDataFrame(geometry=[box(0, 0, 20, 10)], crs="EPSG:26910")
    common_module.generate_fishnet_from_bounds(
        bounds=(0, 0, 20, 10),
        mask_gdf=mask,
        cell_size=10,
        target_path=str(tmp_path / "fishnet.parquet"),
        target_epsg=26910,
        cell_id_col="aermod_cell_id",
    )
