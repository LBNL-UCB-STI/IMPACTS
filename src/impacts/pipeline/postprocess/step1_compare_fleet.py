from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "impacts-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_table

logger = logging.getLogger(__name__)

_VEHICLE_CLASS_GROUP_MAP: dict[str, str] = {
    "Class 2&B3 Vocational": "class12ab3",
    "Class 4-6 Vocational": "class456",
    "Class 7&8 Tractor": "class78",
    "Class 7&8 Vocational": "class78",
}

_CATEGORY_CLASS_GROUP_MAP: dict[str, str] = {
    "LDA": "class12ab3",
    "LDT1": "class12ab3",
    "LDT2": "class12ab3",
    "MDV": "class12ab3",
    "MCY": "class12ab3",
    "LHD1": "class456",
    "LHD2": "class456",
    "UBUS": "class78",
    "SBUS": "class78",
    "MH": "class78",
}

_CLASS_GROUP_ORDER = ["class12ab3", "class456", "class78"]
_CLASS_GROUP_LABELS = {"class12ab3": "Class 1-2b3", "class456": "Class 4-5-6", "class78": "Class 7-8"}

_FUEL_COLORS: dict[str, str] = {
    "Dsl": "#d62728",
    "Gas": "#1f77b4",
    "Elec": "#2ca02c",
    "Phe": "#ff7f0e",
}


def _normalize_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()



def _model_year_sort_key(year_group: str) -> int:
    m = re.search(r"\d{4}", str(year_group))
    return int(m.group()) if m else 9999


def _derive_class_group(row: pd.Series) -> Optional[str]:
    vehicle_class = _normalize_token(row.get("vehicleClass") or "")
    if vehicle_class:
        result = _VEHICLE_CLASS_GROUP_MAP.get(vehicle_class)
        if result:
            return result
    category = _normalize_token(row.get("emfacVehicleCategory") or "")
    if not category:
        return None
    if category in _CATEGORY_CLASS_GROUP_MAP:
        return _CATEGORY_CLASS_GROUP_MAP[category]
    m = re.search(r"Class\s*(\d+)", category)
    if m:
        return "class456" if int(m.group(1)) <= 6 else "class78"
    if category.startswith("T7") or category in ("T6TS", "T7IS"):
        return "class78"
    return None


def _load_vehicle_metadata_lookup(
    *,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    passenger = read_table(passenger_vehicle_types_path).copy()
    freight = read_table(freight_vehicle_types_path).copy()
    for df in (passenger, freight):
        if "emfacModelYearGroup" in df.columns and "model_year_group" not in df.columns:
            df.rename(columns={"emfacModelYearGroup": "model_year_group"}, inplace=True)
        if "emfacResolvedModelYear" in df.columns and "model_year_group" not in df.columns:
            df.rename(columns={"emfacResolvedModelYear": "model_year_group"}, inplace=True)
    combined = pd.concat([passenger, freight], ignore_index=True, sort=False)
    for col in ("emfacId", "emfacVehicleCategory", "emfacFuel", "model_year_group", "vehicleClass"):
        if col not in combined.columns:
            combined[col] = None
    combined["class_group"] = combined.apply(_derive_class_group, axis=1)
    return (
        combined[["emfacId", "class_group", "model_year_group", "emfacFuel"]]
        .rename(columns={"emfacFuel": "fuel"})
        .assign(
            emfacId=lambda df: df["emfacId"].map(_normalize_token),
            model_year_group=lambda df: df["model_year_group"].map(
                lambda v: _normalize_token(v) if pd.notna(v) else ""
            ),
            fuel=lambda df: df["fuel"].map(lambda v: _normalize_token(v) if pd.notna(v) else ""),
        )
        .loc[lambda df: df["emfacId"].ne("")]
        .drop_duplicates(subset=["emfacId"], keep="first")
        .reset_index(drop=True)
    )


def _load_beam_vehicle_lookup(
    *,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    passenger = read_table(passenger_vehicle_types_path).copy()
    freight = read_table(freight_vehicle_types_path).copy()
    passenger["assignment_group"] = "passenger"
    freight["assignment_group"] = "freight"
    vehicle_types = pd.concat([passenger, freight], ignore_index=True, sort=False)
    required = {"vehicleTypeId", "emfacId", "sampleProbabilityWithinCategory"}
    missing = sorted(required - set(vehicle_types.columns))
    if missing:
        raise ValueError(
            "Fleet comparison requires vehicle types inputs with vehicleTypeId, emfacId, and "
            f"sampleProbabilityWithinCategory. Missing: {missing}"
        )
    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_token)
    prepared["emfacId"] = prepared["emfacId"].map(_normalize_token)
    prepared["sampleProbabilityWithinCategory"] = pd.to_numeric(
        prepared["sampleProbabilityWithinCategory"],
        errors="coerce",
    ).fillna(0.0)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("") & prepared["emfacId"].ne("")].copy()
    return prepared[
        ["vehicleTypeId", "assignment_group", "emfacId", "sampleProbabilityWithinCategory"]
    ].drop_duplicates(subset=["vehicleTypeId"], keep="first")


def _build_beam_population_breakdown_from_assignments(
    *,
    assignments_path: str,
    vehicle_lookup: pd.DataFrame,
    assignment_group: str,
) -> pd.DataFrame:
    assignments = read_table(assignments_path).copy()
    required = {"vehicleTypeId"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(
            "Fleet comparison assignment inputs require vehicleTypeId. "
            f"Missing: {missing}"
        )
    prepared = assignments[["vehicleTypeId"]].copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_token)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()
    prepared = prepared.merge(
        vehicle_lookup.loc[vehicle_lookup["assignment_group"].eq(assignment_group), ["vehicleTypeId", "emfacId"]],
        how="inner",
        on="vehicleTypeId",
    )
    return (
        prepared.groupby(["emfacId"], dropna=False)
        .size()
        .reset_index(name="beam_population_weight")
        .assign(assignment_group=assignment_group)
        [["assignment_group", "emfacId", "beam_population_weight"]]
    )


def _build_beam_population_breakdown_with_actual_assignments(
    *,
    vehicle_lookup: pd.DataFrame,
    passenger_vehicles_path: Optional[str] = None,
    freight_carriers_path: Optional[str] = None,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    if passenger_vehicles_path:
        outputs.append(
            _build_beam_population_breakdown_from_assignments(
                assignments_path=passenger_vehicles_path,
                vehicle_lookup=vehicle_lookup,
                assignment_group="passenger",
            )
        )
    if freight_carriers_path:
        outputs.append(
            _build_beam_population_breakdown_from_assignments(
                assignments_path=freight_carriers_path,
                vehicle_lookup=vehicle_lookup,
                assignment_group="freight",
            )
        )
    if not outputs:
        raise ValueError(
            "Fleet comparison requires at least one BEAM assignment source "
            "(passenger_vehicles_path or freight_carriers_path)."
        )
    return pd.concat(outputs, ignore_index=True, sort=False)


def _build_beam_vmt_breakdown(
    *,
    skims_emissions_path: str,
    vehicle_lookup: pd.DataFrame,
) -> pd.DataFrame:
    skims = read_table(skims_emissions_path).copy()
    required = {"linkId", "vehicleTypeId", "totVMT"}
    missing = sorted(required - set(skims.columns))
    if missing:
        raise ValueError(
            "Fleet comparison requires prepared skims with linkId, vehicleTypeId, and totVMT. "
            f"Missing: {missing}"
        )
    prepared = skims[["linkId", "vehicleTypeId", "totVMT"]].copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_token)
    prepared["totVMT"] = pd.to_numeric(prepared["totVMT"], errors="coerce").fillna(0.0)
    prepared = (
        prepared.groupby(["linkId", "vehicleTypeId"], dropna=False)["totVMT"]
        .max()
        .reset_index()
    )
    prepared = prepared.merge(vehicle_lookup, how="inner", on="vehicleTypeId")
    return (
        prepared.groupby(["assignment_group", "emfacId"], dropna=False)["totVMT"]
        .sum()
        .reset_index(name="beam_vmt")
    )


def _aggregate_emfac_activity_by_emfac_id(path: str, *, assignment_group: str) -> pd.DataFrame:
    activity = read_table(path).copy()
    required = {"county", "emfacId", "process", "total_vmt_vehicle_miles_per_year", "population_vehicles"}
    missing = sorted(required - set(activity.columns))
    if missing:
        raise ValueError(
            "Fleet comparison requires EMFAC activity-by-emfacId files with county, emfacId, process, "
            f"total_vmt_vehicle_miles_per_year, and population_vehicles. Missing: {missing}"
        )
    prepared = activity[
        ["county", "emfacId", "total_vmt_vehicle_miles_per_year", "population_vehicles"]
    ].copy()
    prepared["county"] = prepared["county"].map(_normalize_token)
    prepared["emfacId"] = prepared["emfacId"].map(_normalize_token)
    prepared["total_vmt_vehicle_miles_per_year"] = pd.to_numeric(
        prepared["total_vmt_vehicle_miles_per_year"],
        errors="coerce",
    ).fillna(0.0)
    prepared["population_vehicles"] = pd.to_numeric(
        prepared["population_vehicles"],
        errors="coerce",
    ).fillna(0.0)
    prepared = (
        prepared.groupby(["county", "emfacId"], dropna=False)[
            ["total_vmt_vehicle_miles_per_year", "population_vehicles"]
        ]
        .max()
        .reset_index()
    )
    prepared["assignment_group"] = assignment_group
    return (
        prepared.groupby(["assignment_group", "emfacId"], dropna=False)[
            ["total_vmt_vehicle_miles_per_year", "population_vehicles"]
        ]
        .sum()
        .reset_index()
        .rename(
            columns={
                "total_vmt_vehicle_miles_per_year": "emfac_vmt",
                "population_vehicles": "emfac_population",
            }
        )
    )


def _build_comparison_table(
    *,
    beam_population_breakdown: pd.DataFrame,
    beam_vmt_breakdown: pd.DataFrame,
    emfac_passenger_activity_path: str,
    emfac_freight_activity_path: str,
) -> pd.DataFrame:
    beam = beam_population_breakdown.merge(
        beam_vmt_breakdown,
        how="outer",
        on=["assignment_group", "emfacId"],
    )
    emfac = pd.concat(
        [
            _aggregate_emfac_activity_by_emfac_id(emfac_passenger_activity_path, assignment_group="passenger"),
            _aggregate_emfac_activity_by_emfac_id(emfac_freight_activity_path, assignment_group="freight"),
        ],
        ignore_index=True,
    )
    comparison = beam.merge(emfac, how="outer", on=["assignment_group", "emfacId"])
    required_columns = {"beam_population_weight", "beam_vmt", "emfac_population", "emfac_vmt"}
    missing = sorted(required_columns - set(comparison.columns))
    if missing:
        raise ValueError(f"Fleet comparison is missing required columns after merge: {missing}")
    for column in ["beam_population_weight", "beam_vmt", "emfac_population", "emfac_vmt"]:
        comparison[column] = pd.to_numeric(comparison[column], errors="coerce").fillna(0.0)
    for assignment_group, group in comparison.groupby("assignment_group", dropna=False):
        mask = comparison["assignment_group"] == assignment_group
        beam_population_total = group["beam_population_weight"].sum()
        emfac_population_total = group["emfac_population"].sum()
        beam_vmt_total = group["beam_vmt"].sum()
        emfac_vmt_total = group["emfac_vmt"].sum()
        comparison.loc[mask, "beam_population_share"] = (
            comparison.loc[mask, "beam_population_weight"] / beam_population_total if beam_population_total > 0 else 0.0
        )
        comparison.loc[mask, "emfac_population_share"] = (
            comparison.loc[mask, "emfac_population"] / emfac_population_total if emfac_population_total > 0 else 0.0
        )
        comparison.loc[mask, "beam_vmt_share"] = (
            comparison.loc[mask, "beam_vmt"] / beam_vmt_total if beam_vmt_total > 0 else 0.0
        )
        comparison.loc[mask, "emfac_vmt_share"] = (
            comparison.loc[mask, "emfac_vmt"] / emfac_vmt_total if emfac_vmt_total > 0 else 0.0
        )
    comparison["population_share_difference"] = (
        comparison["beam_population_share"] - comparison["emfac_population_share"]
    )
    comparison["vmt_share_difference"] = comparison["beam_vmt_share"] - comparison["emfac_vmt_share"]
    return comparison.sort_values(["assignment_group", "emfacId"]).reset_index(drop=True)


def _write_tables(
    comparison: pd.DataFrame,
    *,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_parquet = output_dir / "step1_fleet_comparison.parquet"
    comparison_csv = output_dir / "step1_fleet_comparison.csv"
    comparison.to_parquet(comparison_parquet, index=False)
    comparison.to_csv(comparison_csv, index=False)
    return {
        "comparison_parquet": str(comparison_parquet),
        "comparison_csv": str(comparison_csv),
    }



def _draw_stacked_bars(
    ax: plt.Axes,
    *,
    beam_pivot: pd.DataFrame,
    emfac_pivot: pd.DataFrame,
    x_labels: list[str],
    segment_labels: list[str],
    segment_colors: list[str],
    x_axis_label: str,
    y_label: str,
    title: str,
) -> None:
    n_x = len(x_labels)
    x = list(range(n_x))
    width = 0.38
    x_beam = [pos - width / 2 for pos in x]
    x_emfac = [pos + width / 2 for pos in x]
    beam_bottoms = [0.0] * n_x
    emfac_bottoms = [0.0] * n_x
    for segment, color in zip(segment_labels, segment_colors):
        beam_vals = [
            float(beam_pivot.at[cat, segment])
            if cat in beam_pivot.index and segment in beam_pivot.columns
            else 0.0
            for cat in x_labels
        ]
        emfac_vals = [
            float(emfac_pivot.at[cat, segment])
            if cat in emfac_pivot.index and segment in emfac_pivot.columns
            else 0.0
            for cat in x_labels
        ]
        ax.bar(x_beam, beam_vals, bottom=beam_bottoms, width=width, color=color, label=segment)
        ax.bar(x_emfac, emfac_vals, bottom=emfac_bottoms, width=width, color=color, hatch="//", edgecolor="white")
        beam_bottoms = [b + v for b, v in zip(beam_bottoms, beam_vals)]
        emfac_bottoms = [b + v for b, v in zip(emfac_bottoms, emfac_vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    segment_handles = [Patch(facecolor=c, label=s) for s, c in zip(segment_labels, segment_colors)]
    style_handles = [
        Patch(facecolor="white", edgecolor="gray", label="BEAM"),
        Patch(facecolor="white", edgecolor="gray", hatch="//", label="EMFAC"),
    ]
    ax.legend(handles=segment_handles + style_handles, loc="upper right", fontsize=8)


def _build_class_model_year_panel_data(
    comparison: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    assignment_group: str,
) -> Optional[pd.DataFrame]:
    subset = (
        comparison.loc[comparison["assignment_group"].eq(assignment_group)]
        .merge(metadata[["emfacId", "class_group", "model_year_group"]], on="emfacId", how="left")
        .loc[lambda df: df["class_group"].notna() & df["model_year_group"].notna() & df["model_year_group"].ne("")]
    )
    if subset.empty:
        return None
    return (
        subset.groupby(["class_group", "model_year_group"], dropna=False)[["beam_vmt", "emfac_vmt"]]
        .sum().reset_index()
    )


def _build_model_year_fuel_panel_data(
    comparison: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    assignment_group: str,
) -> Optional[pd.DataFrame]:
    subset = (
        comparison.loc[comparison["assignment_group"].eq(assignment_group)]
        .merge(metadata[["emfacId", "model_year_group", "fuel"]], on="emfacId", how="left")
        .loc[lambda df: df["model_year_group"].notna() & df["model_year_group"].ne("") & df["fuel"].notna() & df["fuel"].ne("")]
    )
    if subset.empty:
        return None
    return (
        subset.groupby(["model_year_group", "fuel"], dropna=False)[["beam_vmt", "emfac_vmt"]]
        .sum().reset_index()
    )


def _plot_class_model_year_combined(
    comparison: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> Optional[str]:
    panel_data: dict[str, pd.DataFrame] = {}
    for assignment_group in ("passenger", "freight"):
        grouped = _build_class_model_year_panel_data(
            comparison, metadata=metadata, assignment_group=assignment_group,
        )
        if grouped is not None:
            panel_data[assignment_group] = grouped
    if not panel_data:
        return None
    all_years = sorted(
        {yr for g in panel_data.values() for yr in g["model_year_group"].unique()},
        key=_model_year_sort_key,
    )
    n_years = len(all_years)
    _cmap = plt.cm.tab10 if n_years <= 10 else plt.cm.tab20
    year_colors = {my: _cmap(i % _cmap.N) for i, my in enumerate(all_years)}
    groups_in_order = [g for g in ("passenger", "freight") if g in panel_data]
    n_panels = len(groups_in_order)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(7, 3 * 1.5), 6 * n_panels), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    for ax, assignment_group in zip(axes, groups_in_order):
        grouped = panel_data[assignment_group]
        class_groups = [c for c in _CLASS_GROUP_ORDER if c in grouped["class_group"].unique()]
        local_years = sorted(grouped["model_year_group"].unique().tolist(), key=_model_year_sort_key)
        display_labels = [_CLASS_GROUP_LABELS.get(c, c) for c in class_groups]
        beam_pivot = (
            grouped.pivot_table(index="class_group", columns="model_year_group", values="beam_vmt", aggfunc="sum")
            .reindex(index=class_groups, columns=local_years).fillna(0.0)
        )
        emfac_pivot = (
            grouped.pivot_table(index="class_group", columns="model_year_group", values="emfac_vmt", aggfunc="sum")
            .reindex(index=class_groups, columns=local_years).fillna(0.0)
        )
        beam_pivot.index = display_labels
        emfac_pivot.index = display_labels
        _draw_stacked_bars(
            ax,
            beam_pivot=beam_pivot,
            emfac_pivot=emfac_pivot,
            x_labels=display_labels,
            segment_labels=local_years,
            segment_colors=[year_colors[y] for y in local_years],
            x_axis_label="Vehicle class group",
            y_label="VMT (vehicle-miles)",
            title=f"{assignment_group.title()} — VMT by class group and model year",
        )
    output_path = output_dir / "step1_vmt_by_class_and_model_year.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _plot_model_year_fuel_combined(
    comparison: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> Optional[str]:
    panel_data: dict[str, pd.DataFrame] = {}
    for assignment_group in ("passenger", "freight"):
        grouped = _build_model_year_fuel_panel_data(
            comparison, metadata=metadata, assignment_group=assignment_group,
        )
        if grouped is not None:
            panel_data[assignment_group] = grouped
    if not panel_data:
        return None
    all_years = sorted(
        {yr for g in panel_data.values() for yr in g["model_year_group"].unique()},
        key=_model_year_sort_key,
    )
    groups_in_order = [g for g in ("passenger", "freight") if g in panel_data]
    n_panels = len(groups_in_order)
    fig, axes = plt.subplots(n_panels, 1, figsize=(max(7, len(all_years) * 1.2), 6 * n_panels), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    for ax, assignment_group in zip(axes, groups_in_order):
        grouped = panel_data[assignment_group]
        model_years = sorted(grouped["model_year_group"].unique().tolist(), key=_model_year_sort_key)
        fuels = [f for f in _FUEL_COLORS if f in grouped["fuel"].unique()]
        if not model_years or not fuels:
            continue
        beam_pivot = (
            grouped.pivot_table(index="model_year_group", columns="fuel", values="beam_vmt", aggfunc="sum")
            .reindex(index=model_years, columns=fuels).fillna(0.0)
        )
        emfac_pivot = (
            grouped.pivot_table(index="model_year_group", columns="fuel", values="emfac_vmt", aggfunc="sum")
            .reindex(index=model_years, columns=fuels).fillna(0.0)
        )
        _draw_stacked_bars(
            ax,
            beam_pivot=beam_pivot,
            emfac_pivot=emfac_pivot,
            x_labels=model_years,
            segment_labels=fuels,
            segment_colors=[_FUEL_COLORS[f] for f in fuels],
            x_axis_label="Model year group",
            y_label="VMT (vehicle-miles)",
            title=f"{assignment_group.title()} — VMT by model year and fuel",
        )
    output_path = output_dir / "step1_vmt_by_model_year_and_fuel.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def run(
    *,
    skims_emissions_path: str,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    emfac_passenger_activity_path: str,
    emfac_freight_activity_path: str,
    output_dir: Path,
    passenger_vehicles_path: Optional[str] = None,
    freight_carriers_path: Optional[str] = None,
) -> dict[str, str]:
    log_step_banner("Analysis Step 1", "Compare Fleet", logger=logger)
    log_substep_banner("1.1", "compare BEAM fleet against EMFAC by emfacId", logger=logger)
    vehicle_lookup = _load_beam_vehicle_lookup(
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )
    beam_population_breakdown = _build_beam_population_breakdown_with_actual_assignments(
        vehicle_lookup=vehicle_lookup,
        passenger_vehicles_path=passenger_vehicles_path,
        freight_carriers_path=freight_carriers_path,
    )
    beam_vmt_breakdown = _build_beam_vmt_breakdown(
        skims_emissions_path=skims_emissions_path,
        vehicle_lookup=vehicle_lookup,
    )
    comparison = _build_comparison_table(
        beam_population_breakdown=beam_population_breakdown,
        beam_vmt_breakdown=beam_vmt_breakdown,
        emfac_passenger_activity_path=emfac_passenger_activity_path,
        emfac_freight_activity_path=emfac_freight_activity_path,
    )
    outputs = _write_tables(comparison, output_dir=output_dir)
    metadata = _load_vehicle_metadata_lookup(
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )
    plot_path = _plot_class_model_year_combined(comparison, metadata=metadata, output_dir=output_dir)
    if plot_path:
        outputs["vmt_class_model_year_plot"] = plot_path
    plot_path = _plot_model_year_fuel_combined(comparison, metadata=metadata, output_dir=output_dir)
    if plot_path:
        outputs["vmt_model_year_fuel_plot"] = plot_path
    logger.info("Analysis Step 1 complete")
    return outputs
