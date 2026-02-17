import os
import re
from multiprocessing import Pool
from typing import Iterable, Optional

import pandas as pd

from impacts import functions

try:
    import geopandas as gpd
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("geopandas is required to run this script") from exc

_GRID = None
_BOUNDING_BOX = None
_DATDIR = None


def _init_worker(grid_path: str, bbox_wkb: bytes, datdir: str) -> None:
    global _GRID, _BOUNDING_BOX, _DATDIR
    _GRID = gpd.read_file(grid_path)
    _BOUNDING_BOX = gpd.GeoSeries.from_wkb([bbox_wkb], crs="EPSG:4326")
    _DATDIR = datdir


def _map_one(grid_source_grid: str) -> Optional[pd.DataFrame]:
    return functions.map_to_grid(_DATDIR, grid_source_grid, _GRID, _BOUNDING_BOX)


def _extract_ids(files: Iterable[str]) -> list[str]:
    ids = []
    for name in files:
        match = re.match(r"^grid_(\d+)", name)
        if match:
            ids.append(match.group(1))
    return sorted(set(ids))


def main() -> None:
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    datdir = os.path.join(data_dir, "sfbay_grid_geopoints_inmap_1.9.6")

    grid_path = os.path.join(data_dir, "grid_polygon", "grid_polygon.shp")
    bounding_box = functions.get_bounding_box()

    idlist = _extract_ids(os.listdir(datdir))

    bbox_wkb = bounding_box.iloc[0].wkb
    results = []
    with Pool(processes=6, initializer=_init_worker, initargs=(grid_path, bbox_wkb, datdir)) as pool:
        for res in pool.map(_map_one, idlist):
            if res is not None:
                results.append(res)

    if not results:
        raise RuntimeError("No results produced from map_to_grid")

    res_df = pd.concat(results)
    out_path = os.path.join(data_dir, "SFB_NOX_NOX_GRID.RData")
    functions.write_rdata(out_path, {"res": res_df})


if __name__ == "__main__":
    main()
