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
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

    isrm = gpd.read_file(os.path.join(data_dir, "isrm_polygon", "isrm_polygon.shp"))

    rdata_path = os.path.join(data_dir, "cmaqtestNO2_ratio.RData")
    rdata = functions.read_rdata(rdata_path)
    if "pdat" not in rdata:
        raise KeyError("Expected 'pdat' in cmaqtestNO2_ratio.RData")
    pdat = rdata["pdat"].copy()

    pdat["col"] = pdat["x"]
    pdat["row"] = pdat["y"]
    coords = functions.CR_XY(pdat["col"], pdat["row"])
    pdat[["x", "y"]] = coords

    mm = grid_polygons_from_centers(pdat[["x", "y", "value", "col", "row"]])
    mm = mm.drop(columns=["value"])

    shp_path = os.path.join(data_dir, "baaqmd.shp")
    mm.to_file(shp_path)

    mm = mm.to_crs(isrm.crs)
    kk = gpd.overlay(mm, isrm, how="intersection")
    kk["area"] = kk.geometry.area

    merged = kk.merge(pdat, on=["col", "row"], how="left")
    grouped = (
        merged.drop(columns="geometry")
        .groupby("isrm")
        .apply(lambda frame: (frame["value"] * frame["area"]).sum() / frame["area"].sum())
        .reset_index(name="value")
    )

    no2ratio = grouped.rename(columns={"value": "NO2_NOx_ratio"})
    functions.write_rdata(
        os.path.join(data_dir, "sfb.no2ratio_isrmGRID.RData"),
        {"no2ratio": no2ratio},
    )


if __name__ == "__main__":
    main()
