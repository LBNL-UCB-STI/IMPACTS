import os

import pandas as pd

from impacts import functions

try:
    import geopandas as gpd
    from shapely.geometry import box
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("geopandas and shapely are required to run this script") from exc


def grid_polygons_from_centers(df: pd.DataFrame) -> gpd.GeoDataFrame:
    half_dx = functions.dx / 2
    half_dy = functions.dy / 2
    geometry = [
        box(x - half_dx, y - half_dy, x + half_dx, y + half_dy)
        for x, y in zip(df["x"], df["y"])
    ]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=functions.proj_baaqmd)
    return gdf


def main() -> None:
    mywd = "~/Dropbox/Research/SmartGrid_Behavioral/TransportationInitiative/ATLAS/BEAM_AQM/Rscripts"
    os.chdir(os.path.expanduser(mywd))

    rdata_path = os.path.join("..", "..", "RData", "cmaqtestNO2_ratio.RData")
    rdata = functions.read_rdata(rdata_path)
    if "pdat" not in rdata:
        raise KeyError("Expected 'pdat' in cmaqtestNO2_ratio.RData")
    pdat = rdata["pdat"]

    coords = functions.CR_XY(pdat["col"], pdat["row"])
    pdat = pdat.copy()
    pdat[["x", "y"]] = coords

    gdf = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]])
    gdf = gdf.drop(columns=["value"])

    out_path = os.path.join("..", "..", "RData", "baaqmd.shp")
    gdf.to_file(out_path)


if __name__ == "__main__":
    main()
