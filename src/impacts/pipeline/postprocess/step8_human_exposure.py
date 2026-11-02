"""Human-exposure products for grids, CES tracts, schools, and LTC facilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ._common import CMAP_GREEN_RED, _add_power_plants, _load_emitting_power_plants

logger = logging.getLogger(__name__)

METHOD_FIELDS = {
    "Final_TotalPM25": ("TotalPM25",),
    "Traffic_TotalPM25": ("Traffic_TotalPM25",),
    "InMAP_TotalPM25": ("inmap_only_TotalPM25",),
    "AERMOD_TotalPM25": ("aermod_PrimaryPM25_scaled", "aermod_SecondaryPM25"),
    "Power_TotalPM25": ("power_TotalPM25",),
    "Final_BC": ("BC",),
    "InMAP_BC": ("inmap_only_BC",),
    "AERMOD_BC": ("aermod_BC",),
    "Final_NO2": ("NO2",),
    "InMAP_NO2": ("inmap_only_NO2",),
    "AERMOD_NO2": ("aermod_NO2",),
}


def _extract_concentrations(path: str, cell_ids: set[int]) -> pd.DataFrame:
    parquet = pq.ParquetFile(path)
    available = set(parquet.schema_arrow.names)
    raw = {"aermod_cell_id", "TotalPM25", "BC", "NO2", "power_TotalPM25"}
    raw.update(field for fields in METHOD_FIELDS.values() for field in fields)
    columns = [column for column in raw if column in available and column != "Traffic_TotalPM25"]
    selected = []
    for batch in parquet.iter_batches(batch_size=500_000, columns=columns):
        frame = batch.to_pandas()
        keep = frame["aermod_cell_id"].isin(cell_ids)
        if keep.any():
            selected.append(frame.loc[keep].copy())
    if not selected:
        raise ValueError("No populated AERMOD cells matched the concentration distribution.")
    result = pd.concat(selected, ignore_index=True)
    def numeric_field(field: str) -> pd.Series:
        if field not in result:
            return pd.Series(0.0, index=result.index, dtype="float64")
        return pd.to_numeric(result[field], errors="coerce").fillna(0.0)

    power = numeric_field("power_TotalPM25")
    result["power_TotalPM25"] = power
    result["Traffic_TotalPM25"] = pd.to_numeric(result["TotalPM25"], errors="coerce").fillna(0.0) - power
    for output, inputs in METHOD_FIELDS.items():
        values = [numeric_field(field) for field in inputs]
        result[output] = sum(values)
    return result[["aermod_cell_id", *METHOD_FIELDS]].drop_duplicates("aermod_cell_id")


def _aggregate_tracts(grid: gpd.GeoDataFrame, tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if tracts.crs != grid.crs:
        tracts = tracts.to_crs(grid.crs)
    points = grid[["aermod_cell_id", "person_count", *METHOD_FIELDS, "geometry"]].copy()
    points.geometry = points.geometry.centroid
    tract_id = "TractID" if "TractID" in tracts else "Tract"
    assigned = gpd.sjoin(points, tracts[[tract_id, "geometry"]], how="inner", predicate="within")
    rows = []
    for key, group in assigned.groupby(tract_id, sort=False):
        population = float(group["person_count"].sum())
        row = {tract_id: key, "ModelGridCount": len(group), "ModelPopulation": population}
        for method in METHOD_FIELDS:
            burden = float((group[method] * group["person_count"]).sum())
            row[f"{method}_Exposure"] = burden
            row[f"{method}_PW"] = burden / population if population else np.nan
            row[f"{method}_Max"] = float(group[method].max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    return tracts.merge(summary, on=tract_id, how="left")


def _facility_products(facilities: gpd.GeoDataFrame, concentrations: pd.DataFrame, weight_fields: tuple[str, ...]):
    facilities = facilities.copy()
    cell_column = "AermodCellID" if "AermodCellID" in facilities else "aermod_cell_id"
    facilities[cell_column] = pd.to_numeric(facilities[cell_column], errors="coerce").astype("Int64")
    result = facilities.merge(concentrations, left_on=cell_column, right_on="aermod_cell_id", how="left", validate="many_to_one")
    result = gpd.GeoDataFrame(result, geometry="geometry", crs=facilities.crs)
    result["CES25_Valid"] = result.get("CES_Group", pd.Series(index=result.index, dtype="object")).isin(_CES_GROUPS).astype("int8")
    for weight_field in weight_fields:
        if weight_field not in result:
            continue
        weight = pd.to_numeric(result[weight_field], errors="coerce")
        for method in METHOD_FIELDS:
            result[f"{weight_field}_Exposure_{method}"] = weight * result[method]
    return result


def _plot_layer(gdf, column: str, title: str, label: str, path: Path, plants=None, *, log=False):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    values = pd.to_numeric(gdf[column], errors="coerce")
    positive = values[values > 0]
    if positive.empty:
        return None
    vmax = float(positive.quantile(0.99))
    norm = LogNorm(max(float(positive.quantile(0.01)), np.finfo(float).tiny), vmax) if log else Normalize(0, vmax)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300)
    ax.set_facecolor("#f4f2ed")
    plot = gdf.loc[values.gt(0), [column, "geometry"]].copy()
    if plot.geom_type.isin(["Point", "MultiPoint"]).all():
        plot.plot(ax=ax, column=column, cmap=CMAP_GREEN_RED, norm=norm, markersize=22,
                  edgecolor="white", linewidth=0.25, alpha=0.9, zorder=2)
    else:
        plot.plot(ax=ax, column=column, cmap=CMAP_GREEN_RED, norm=norm,
                  edgecolor="#777777", linewidth=0.08, rasterized=True, zorder=2)
    _add_power_plants(ax, plants)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=CMAP_GREEN_RED)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, shrink=0.78)
    cbar.set_label(label, fontsize=12)
    ax.set_title(title, fontsize=16, pad=14)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def _plot_ces_comparison(tracts, path: Path):
    import matplotlib.pyplot as plt
    if "CES_Group" not in tracts:
        return None
    order = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]
    rows = []
    for group, frame in tracts.groupby("CES_Group"):
        pop = pd.to_numeric(frame["ModelPopulation"], errors="coerce").fillna(0)
        for method in ("Final_TotalPM25_PW", "Traffic_TotalPM25_PW", "Power_TotalPM25_PW"):
            if method in frame and pop.sum() > 0:
                rows.append((group, method.replace("_PW", ""), float((frame[method].fillna(0) * pop).sum() / pop.sum())))
    data = pd.DataFrame(rows, columns=["CES_Group", "Method", "PWC"])
    if data.empty:
        return None
    pivot = data.pivot(index="CES_Group", columns="Method", values="PWC").reindex(order)
    fig, ax = plt.subplots(figsize=(11, 7), dpi=250)
    pivot.plot.bar(ax=ax, color=["#d73027", "#fdae61", "#1a9850"][:len(pivot.columns)])
    ax.set_xlabel("CalEnviroScreen percentile group")
    ax.set_ylabel("Population-weighted PM2.5 concentration (µg/m³)")
    ax.set_title("Human exposure by CalEnviroScreen quartile")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


_BV_PALETTE = np.array([
    ["#e8e8e8", "#ace4e4", "#5ac8c8"],
    ["#dfb0d6", "#a5add3", "#5698b9"],
    ["#be64ac", "#8c62aa", "#3b4994"],
])
_BAY_COUNTIES = ["Alameda", "Contra Costa", "Marin", "Napa", "San Francisco", "San Mateo", "Santa Clara", "Solano", "Sonoma"]
_CES_GROUPS = ["0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"]


def _classes(values):
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    cuts = np.quantile(numeric[np.isfinite(numeric)], [1 / 3, 2 / 3])
    return np.digitize(numeric, cuts, right=True), cuts


def _bivariate_legend(ax, pop_cuts, pm_cuts, scope):
    from matplotlib.patches import Rectangle
    ax.set_xlim(-0.35, 3.7); ax.set_ylim(-0.8, 5.45); ax.axis("off")
    ax.text(0, 5.15, "Bivariate legend", ha="left", va="top", fontsize=15)
    ax.text(0, 4.6, f"Tertiles among {scope}", ha="left", va="top", fontsize=10.5, color="#444")
    y0 = 1.2
    for y in range(3):
        for x in range(3):
            ax.add_patch(Rectangle((x, y0 + y), 1, 1, facecolor=_BV_PALETTE[y, x], edgecolor="white", linewidth=1.4))
    ax.annotate("", xy=(3.25, y0-.18), xytext=(0, y0-.18), arrowprops={"arrowstyle":"->","color":"#333"})
    ax.text(1.5, y0-.56, "Population", ha="center", va="top", fontsize=10.5)
    for i, label in enumerate(("Low", "Medium", "High")): ax.text(i+.5, y0-.15, label, ha="center", va="bottom", fontsize=8.8)
    ax.annotate("", xy=(-.15, y0+3.25), xytext=(-.15, y0), arrowprops={"arrowstyle":"->","color":"#333"})
    ax.text(-.32, y0+1.5, "AERMOD-enhanced\nPM₂.₅ concentration", ha="right", va="center", rotation=90, fontsize=10.5)
    ax.text(0, .18, "Class thresholds", fontsize=10.5)
    ax.text(0, -.01, f"Population: ≤{pop_cuts[0]:,.0f}, {pop_cuts[0]:,.0f}–{pop_cuts[1]:,.0f}, >{pop_cuts[1]:,.0f}\nPM₂.₅: ≤{pm_cuts[0]:.3f}, {pm_cuts[0]:.3f}–{pm_cuts[1]:.3f}, >{pm_cuts[1]:.3f} µg/m³", va="top", fontsize=9.2, color="#444")


def _plot_original_bivariate(data, counties, population_column, pm_column, path, *, tract=False):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    frame = data.loc[pd.to_numeric(data[population_column], errors="coerce").gt(0) & pd.to_numeric(data[pm_column], errors="coerce").notna()].copy()
    pc, pcut = _classes(frame[population_column]); mc, mcut = _classes(frame[pm_column])
    frame["_color"] = [_BV_PALETTE[y, x] for y, x in zip(mc, pc)]
    fig = plt.figure(figsize=(13.4, 11.7), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[4.8, 1.65]); ax=fig.add_subplot(gs[0,0]); leg=fig.add_subplot(gs[0,1])
    frame.plot(ax=ax, color=frame["_color"], edgecolor="white" if tract else "none", linewidth=.18 if tract else 0, rasterized=not tract)
    counties.boundary.plot(ax=ax, color="#424242", linewidth=.75 if tract else .55, zorder=3)
    for row in counties.itertuples(index=False):
        point=row.geometry.representative_point(); txt=ax.text(point.x, point.y, str(row.County), ha="center", va="center", fontsize=7.4, color="#2f2f2f", zorder=4)
        txt.set_path_effects([pe.withStroke(linewidth=2.4, foreground="white", alpha=.9)])
    ax.set_axis_off(); ax.set_aspect("equal"); ax.margins(.015)
    _bivariate_legend(leg, pcut, mcut, "CES tracts" if tract else "populated 100 m cells")
    fig.suptitle(("Bivariate Distribution of AERMOD-Enhanced PM₂.₅ PWC and Population\nCES Census Tracts in the San Francisco Bay Area" if tract else "Bivariate Distribution of AERMOD-Enhanced PM₂.₅ and Population\nPopulated 100 m Grid Cells in the San Francisco Bay Area"), fontsize=18)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return str(path)


def _weighted_fit(x, y, w):
    x=np.asarray(pd.to_numeric(x,errors="coerce"),float); y=np.asarray(pd.to_numeric(y,errors="coerce"),float); w=np.asarray(pd.to_numeric(w,errors="coerce"),float)
    ok=np.isfinite(x)&np.isfinite(y)&np.isfinite(w)&(w>0); x=x[ok]; y=y[ok]; w=w[ok]
    if len(x)<2 or np.ptp(x)==0: return None
    slope,intercept=np.polyfit(x,y,1,w=np.sqrt(w)); pred=intercept+slope*x; mean=np.average(y,weights=w)
    r2=1-np.sum(w*(y-pred)**2)/np.sum(w*(y-mean)**2) if np.sum(w*(y-mean)**2)>0 else np.nan
    return x,y,float(slope),float(intercept),float(r2)


def _plot_original_scatter(tracts, schools, ltc, path):
    import matplotlib.pyplot as plt
    groups=["0.00-0.25","0.25-0.50","0.50-0.75","0.75-1.00"]
    colors=dict(zip(groups,["#2878B5","#2A9D8F","#E07A2D","#8E63B8"])); labels=dict(zip(groups,["0–25%","25–50%","50–75%","75–100%"]));
    datasets=[("CES tracts",tracts,"ModelPopulation"),("Schools",schools,"Enrollment"),("LTC",ltc,"TotalLicensedBeds")]
    pollutants=[("PM2.5","Final_TotalPM25_PW","InMAP_TotalPM25_PW","Final_TotalPM25","InMAP_TotalPM25"),("BC","Final_BC_PW","InMAP_BC_PW","Final_BC","InMAP_BC"),("NO₂","Final_NO2_PW","InMAP_NO2_PW","Final_NO2","InMAP_NO2")]
    fig,axes=plt.subplots(3,3,figsize=(18.8,15.5),constrained_layout=True)
    for ri,(name,data,weight) in enumerate(datasets):
        for ci,(title,tx,ty,fx,fy) in enumerate(pollutants):
            ax=axes[ri,ci]; xcol=tx if ri==0 else fx; ycol=ty if ri==0 else fy; allfits=[]
            for group in groups:
                sub=data.loc[data["CES_Group"].eq(group) & pd.to_numeric(data[weight],errors="coerce").gt(0)]
                fit=_weighted_fit(sub[xcol],sub[ycol],sub[weight])
                if fit is None: continue
                x,y,slope,intercept,r2=fit; color=colors[group]; marker="o" if ri<2 else "^"
                ax.scatter(x,y,s=17 if ri<2 else 31,c=color,marker=marker,alpha=.34 if ri==0 else (.27 if ri==1 else .46),edgecolors="none",rasterized=True,label=f"CES {labels[group]} (n={len(x)}, R²={r2:.2f})")
                xx=np.linspace(x.min(),x.max(),120); ax.plot(xx,intercept+slope*xx,color=color,lw=2.3); allfits.append((x,y,np.asarray(pd.to_numeric(sub[weight],errors="coerce"),float)))
            overall=_weighted_fit(data[xcol],data[ycol],data[weight])
            if overall:
                x,y,slope,intercept,r2=overall; xx=np.linspace(x.min(),x.max(),140); ax.plot(xx,intercept+slope*xx,color="#111",lw=3,ls="-.",label=f"Overall weighted fit (R²={r2:.2f})")
            upper=1.045*float(max(pd.to_numeric(data[xcol],errors="coerce").max(),pd.to_numeric(data[ycol],errors="coerce").max())); ax.plot([0,upper],[0,upper],color="#666",lw=1.2,ls="--",label="1:1")
            ax.set(xlim=(0,upper),ylim=(0,upper),xlabel="AERMOD-enhanced pollutant (µg/m³)",ylabel="InMAP-only concentration (µg/m³)"); ax.set_aspect("equal"); ax.grid(True,color="#ddd",alpha=.75); ax.text(.98,.98,name,transform=ax.transAxes,ha="right",va="top")
            if ri==0: ax.set_title(title if title!="PM2.5" else "PM2.5: AERMOD-enhanced")
            ax.legend(loc="upper left",fontsize=7.2)
    fig.suptitle("Weighted scatter comparison\n25% CES-group fits weighted by model population, enrollment, or licensed beds",fontsize=16)
    fig.savefig(path,dpi=230,bbox_inches="tight",facecolor="white"); plt.close(fig); return str(path)


def run(*, population_path: str, concentration_path: str, ces_tracts_path: str,
        sensitive_facilities_path: str, schools_layer: str, ltc_layer: str,
        output_dir: Path, power_plants_path: str | None = None) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    population = gpd.read_parquet(population_path, columns=["aermod_cell_id", "person_count", "geometry"])
    population = population.loc[pd.to_numeric(population["person_count"], errors="coerce").fillna(0).gt(0)].copy()
    schools_source = gpd.read_file(sensitive_facilities_path, layer=schools_layer)
    ltc_source = gpd.read_file(sensitive_facilities_path, layer=ltc_layer)
    cell_ids = set(pd.to_numeric(population["aermod_cell_id"], errors="coerce").dropna().astype(int))
    for facilities in (schools_source, ltc_source):
        cell_column = "AermodCellID" if "AermodCellID" in facilities else "aermod_cell_id"
        cell_ids.update(pd.to_numeric(facilities[cell_column], errors="coerce").dropna().astype(int))
    concentrations = _extract_concentrations(concentration_path, cell_ids)
    grid = population.merge(concentrations, on="aermod_cell_id", how="inner", validate="one_to_one")
    grid = gpd.GeoDataFrame(grid, geometry="geometry", crs=population.crs)
    for method in METHOD_FIELDS:
        grid[f"{method}_Exposure"] = grid[method] * grid["person_count"]

    grid_path = output_dir / "human_exposure_grid.parquet"
    grid.to_parquet(grid_path, index=False)
    tracts = gpd.read_parquet(ces_tracts_path) if str(ces_tracts_path).lower().endswith(".parquet") else gpd.read_file(ces_tracts_path)
    tract_result = _aggregate_tracts(grid, tracts)
    # Preserve the field contract used by the original exposure webmap.
    tract_result["Hybrid_TotalPM25_PW"] = tract_result["Final_TotalPM25_PW"]
    tract_result["Hybrid_TotalPM25_Exposure"] = tract_result["Final_TotalPM25_Exposure"]
    tract_path = output_dir / "human_exposure_ces_tracts.parquet"
    tract_result.to_parquet(tract_path, index=False)
    tract_result.to_file(output_dir / "human_exposure_ces_tracts.gpkg", layer="human_exposure_ces_tracts", driver="GPKG")

    schools = _facility_products(schools_source, concentrations, ("Enrollment",))
    ltc = _facility_products(ltc_source, concentrations, ("TotalLicensedBeds", "TotalPatientDays"))
    schools_path = output_dir / "human_exposure_schools.parquet"
    ltc_path = output_dir / "human_exposure_ltc.parquet"
    schools.to_parquet(schools_path, index=False)
    ltc.to_parquet(ltc_path, index=False)
    facilities_gpkg = output_dir / "human_exposure_sensitive_facilities.gpkg"
    schools.to_file(facilities_gpkg, layer="schools", driver="GPKG")
    ltc.to_file(facilities_gpkg, layer="ltc", driver="GPKG")

    plants = _load_emitting_power_plants(power_plants_path, grid.crs)
    tract_plants = plants.to_crs(tract_result.crs) if plants is not None else None
    school_plants = plants.to_crs(schools.crs) if plants is not None else None
    ltc_plants = plants.to_crs(ltc.crs) if plants is not None else None
    valid_tracts = tract_result.loc[
        tract_result["ModelPopulation"].fillna(0).gt(0)
        & tract_result["County"].isin(_BAY_COUNTIES)
        & tract_result["CES_Group"].isin(_CES_GROUPS)
    ].copy()
    counties = valid_tracts[["County", "geometry"]].dissolve(by="County", as_index=False)
    bay_union = counties.geometry.union_all()
    grid_for_plot = grid.loc[grid.geometry.centroid.within(bay_union)].copy()
    schools_for_plot = schools.loc[schools["CES_Group"].isin(_CES_GROUPS)].copy()
    ltc_for_plot = ltc.loc[ltc["CES_Group"].isin(_CES_GROUPS)].copy()
    plot_specs = {
        "bivariate_100m": _plot_original_bivariate(grid_for_plot, counties, "person_count", "Final_TotalPM25", output_dir / "Bivariate_100m_Population_AERMOD_Enhanced_PM25.png"),
        "bivariate_ces_tract": _plot_original_bivariate(valid_tracts, counties, "ModelPopulation", "Final_TotalPM25_PW", output_dir / "Bivariate_CES_Tract_AERMOD_PM25_PWC_Population.png", tract=True),
        "weighted_scatter_3x3": _plot_original_scatter(valid_tracts, schools_for_plot, ltc_for_plot, output_dir / "Weighted_Scatter_3x3_Indicator_Corrected.png"),
    }

    report = {
        "population": float(grid["person_count"].sum()),
        "populated_grid_cells": int(len(grid)),
        "ces_tracts": int(len(tract_result)),
        "schools": int(len(schools)),
        "ltc_facilities": int(len(ltc)),
        "power_included_in_final_total_pm25": bool(concentrations["Power_TotalPM25"].abs().gt(0).any()),
        "power_local_local_removal_applied": False,
        "population_weighted_concentrations": {
            method: float((grid[method] * grid["person_count"]).sum() / grid["person_count"].sum())
            for method in METHOD_FIELDS
        },
    }
    report_path = output_dir / "human_exposure_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs = {"grid": str(grid_path), "ces_tracts": str(tract_path), "schools": str(schools_path),
               "ltc": str(ltc_path), "report": str(report_path)}
    outputs.update({key: value for key, value in plot_specs.items() if value})
    from .step9_human_exposure_webmap import run as run_webmap
    outputs["webmap"] = run_webmap(
        concentration_path=concentration_path,
        tracts_path=str(tract_path),
        schools_path=str(schools_path),
        ltc_path=str(ltc_path),
        output_dir=output_dir / "web",
        power_plants_path=power_plants_path,
    )
    return outputs
