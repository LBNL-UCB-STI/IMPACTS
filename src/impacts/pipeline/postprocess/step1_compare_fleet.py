from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Optional

import duckdb

from ._common import (
    CHART_LEGEND_FONTSIZE,
    CLASS_GROUP_LABELS,
    CLASS_GROUP_ORDER,
    PLOT_DPI,
    _advance_progress,
    _close_progress,
    _normalize_token,
    _set_progress_task,
    _step_progress,
    _style_chart_axes,
    load_postprocess_vehicle_metadata,
)  # configures matplotlib backend before pyplot

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd

from ...common import _duckdb_scan_expression
from ...common import configure_duckdb_connection
from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import read_table

logger = logging.getLogger(__name__)

_FUEL_COLORS: dict[str, str] = {
    "Dsl": "#d62728",
    "Gas": "#1f77b4",
    "Elec": "#2ca02c",
    "Phe": "#ff7f0e",
}


def _model_year_sort_key(year_group: str) -> int:
    m = re.search(r"\d{4}", str(year_group))
    return int(m.group()) if m else 9999


def _load_vehicle_metadata_lookup(
    *,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    population_sample: float = 1.0,
    transit_sample: float = 1.0,
    freight_sample: Optional[float] = None,
) -> pd.DataFrame:
    return (
        load_postprocess_vehicle_metadata(
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            population_sample=population_sample,
            transit_sample=transit_sample,
            freight_sample=freight_sample,
        )
        .loc[lambda df: df["emfacId"].ne("")]
        .groupby(["assignment_group", "emfacId", "class_group", "model_year_group", "fuel"], dropna=False)[
            ["fleet_vmt_prior", "fleet_population_prior"]
        ]
        .sum()
        .reset_index()
    )


def _load_beam_vehicle_lookup(
    *,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    population_sample: float,
    transit_sample: float,
    freight_sample: Optional[float],
) -> pd.DataFrame:
    metadata = load_postprocess_vehicle_metadata(
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        population_sample=population_sample,
        transit_sample=transit_sample,
        freight_sample=freight_sample,
    )
    return metadata.loc[metadata["emfacId"].ne("")][
        ["vehicleTypeId", "assignment_group", "emfacId", "sample_scale_factor"]
    ].drop_duplicates(subset=["vehicleTypeId"], keep="first")


def _build_beam_population_breakdown_from_assignments(
    *,
    assignments_path: str,
    vehicle_lookup: pd.DataFrame,
    assignment_group: str,
) -> pd.DataFrame:
    if Path(assignments_path).suffix.lower() == ".parquet":
        lookup = (
            vehicle_lookup.loc[
                vehicle_lookup["assignment_group"].eq(assignment_group),
                ["vehicleTypeId", "emfacId", "sample_scale_factor"],
            ]
            .assign(
                vehicleTypeId=lambda df: df["vehicleTypeId"].map(_normalize_token),
                emfacId=lambda df: df["emfacId"].map(_normalize_token),
                sample_scale_factor=lambda df: pd.to_numeric(
                    df["sample_scale_factor"],
                    errors="coerce",
                ).fillna(0.0),
            )
            .loc[lambda df: df["vehicleTypeId"].ne("") & df["emfacId"].ne("")]
            .drop_duplicates(subset=["vehicleTypeId"], keep="first")
            .reset_index(drop=True)
        )
        if lookup.empty:
            return pd.DataFrame(columns=["assignment_group", "emfacId", "beam_population_weight"])

        scan = _duckdb_scan_expression(assignments_path)
        con = duckdb.connect(database=":memory:")
        try:
            configure_duckdb_connection(
                con,
                working_dir=Path(assignments_path).parent,
                show_progress=False,
                profile="balanced",
            )
            columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
            if "vehicleTypeId" not in columns:
                raise ValueError("Fleet comparison assignment inputs require vehicleTypeId. Missing: ['vehicleTypeId']")
            con.register("vehicle_lookup", lookup)
            result = con.execute(
                f"""
                SELECT
                    ? AS assignment_group,
                    lookup.emfacId,
                    SUM(lookup.sample_scale_factor) AS beam_population_weight
                FROM {scan} AS assignments
                INNER JOIN vehicle_lookup AS lookup
                    ON TRIM(CAST(assignments.vehicleTypeId AS VARCHAR)) = lookup.vehicleTypeId
                WHERE TRIM(COALESCE(CAST(assignments.vehicleTypeId AS VARCHAR), '')) <> ''
                GROUP BY lookup.emfacId
                ORDER BY lookup.emfacId
                """,
                [assignment_group],
            ).fetchdf()
        finally:
            con.close()
        return result[["assignment_group", "emfacId", "beam_population_weight"]]

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
        vehicle_lookup.loc[
            vehicle_lookup["assignment_group"].eq(assignment_group),
            ["vehicleTypeId", "emfacId", "sample_scale_factor"],
        ],
        how="inner",
        on="vehicleTypeId",
    )
    prepared["sample_scale_factor"] = pd.to_numeric(
        prepared["sample_scale_factor"],
        errors="coerce",
    ).fillna(0.0)
    return (
        prepared.groupby(["emfacId"], dropna=False)["sample_scale_factor"]
        .sum()
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
    lookup = (
        vehicle_lookup[["vehicleTypeId", "assignment_group", "emfacId"]]
        .assign(
            vehicleTypeId=lambda df: df["vehicleTypeId"].map(_normalize_token),
            assignment_group=lambda df: df["assignment_group"].map(_normalize_token),
            emfacId=lambda df: df["emfacId"].map(_normalize_token),
        )
        .loc[lambda df: df["vehicleTypeId"].ne("") & df["emfacId"].ne("")]
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .reset_index(drop=True)
    )
    if lookup.empty:
        return pd.DataFrame(columns=["assignment_group", "emfacId", "beam_vmt"])

    scan = _duckdb_scan_expression(skims_emissions_path)
    con = duckdb.connect(database=":memory:")
    try:
        configure_duckdb_connection(
            con,
            working_dir=Path(skims_emissions_path).parent,
            show_progress=False,
            profile="balanced",
        )
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
        required = {"linkId", "vehicleTypeId", "totVMT"}
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                "Fleet comparison requires prepared skims with linkId, vehicleTypeId, and totVMT. "
                f"Missing: {missing}"
            )
        con.register("vehicle_lookup", lookup)
        result = con.execute(
            f"""
            WITH link_vehicle_vmt AS (
                SELECT
                    skims.linkId,
                    lookup.vehicleTypeId,
                    lookup.assignment_group,
                    lookup.emfacId,
                    MAX(COALESCE(TRY_CAST(skims.totVMT AS DOUBLE), 0.0)) AS link_vmt
                FROM {scan} AS skims
                INNER JOIN vehicle_lookup AS lookup
                    ON TRIM(CAST(skims.vehicleTypeId AS VARCHAR)) = lookup.vehicleTypeId
                WHERE TRIM(COALESCE(CAST(skims.vehicleTypeId AS VARCHAR), '')) <> ''
                GROUP BY
                    skims.linkId,
                    lookup.vehicleTypeId,
                    lookup.assignment_group,
                    lookup.emfacId
            )
            SELECT
                assignment_group,
                emfacId,
                SUM(link_vmt) AS beam_vmt
            FROM link_vehicle_vmt
            GROUP BY assignment_group, emfacId
            ORDER BY assignment_group, emfacId
            """
        ).fetchdf()
    finally:
        con.close()
    return result[["assignment_group", "emfacId", "beam_vmt"]]


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
        Patch(facecolor="white", edgecolor="gray", hatch="//", label="EMFAC fleet-prior target"),
    ]
    _style_chart_axes(ax, legend=False)
    ax.legend(
        handles=segment_handles + style_handles,
        loc="upper right",
        fontsize=CHART_LEGEND_FONTSIZE,
    )


def _build_class_model_year_panel_data(
    comparison: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    assignment_group: str,
) -> Optional[pd.DataFrame]:
    meta = metadata.loc[metadata["assignment_group"].eq(assignment_group)].copy()
    meta = meta.loc[
        meta["class_group"].notna()
        & meta["model_year_group"].notna()
        & meta["model_year_group"].ne("")
    ].copy()
    if meta.empty:
        return None

    prior_total = float(meta["fleet_vmt_prior"].sum())
    if prior_total <= 0:
        raise ValueError(
            f"Postprocess Step 1 requires positive fleetVmtPrior values for {assignment_group} vehicle types."
        )
    target_total = float(
        comparison.loc[comparison["assignment_group"].eq(assignment_group), "emfac_vmt"].sum()
    )
    meta["emfac_vmt"] = meta["fleet_vmt_prior"] / prior_total * target_total
    target = (
        meta.groupby(["class_group", "model_year_group"], dropna=False)["emfac_vmt"]
        .sum()
        .reset_index()
    )

    beam = (
        comparison.loc[comparison["assignment_group"].eq(assignment_group)]
        .merge(
            meta[["assignment_group", "emfacId", "class_group", "model_year_group"]],
            on=["assignment_group", "emfacId"],
            how="left",
        )
        .loc[lambda df: df["class_group"].notna() & df["model_year_group"].notna() & df["model_year_group"].ne("")]
    )
    beam = (
        beam.groupby(["class_group", "model_year_group"], dropna=False)["beam_vmt"]
        .sum()
        .reset_index()
    )
    return (
        beam.merge(target, on=["class_group", "model_year_group"], how="outer")
        .fillna({"beam_vmt": 0.0, "emfac_vmt": 0.0})
    )


def _build_model_year_fuel_panel_data(
    comparison: pd.DataFrame,
    *,
    metadata: pd.DataFrame,
    assignment_group: str,
) -> Optional[pd.DataFrame]:
    meta = metadata.loc[metadata["assignment_group"].eq(assignment_group)].copy()
    meta = meta.loc[
        meta["model_year_group"].notna()
        & meta["model_year_group"].ne("")
        & meta["fuel"].notna()
        & meta["fuel"].ne("")
    ].copy()
    if meta.empty:
        return None

    prior_total = float(meta["fleet_vmt_prior"].sum())
    if prior_total <= 0:
        raise ValueError(
            f"Postprocess Step 1 requires positive fleetVmtPrior values for {assignment_group} vehicle types."
        )
    target_total = float(
        comparison.loc[comparison["assignment_group"].eq(assignment_group), "emfac_vmt"].sum()
    )
    meta["emfac_vmt"] = meta["fleet_vmt_prior"] / prior_total * target_total
    target = (
        meta.groupby(["model_year_group", "fuel"], dropna=False)["emfac_vmt"]
        .sum()
        .reset_index()
    )

    beam = (
        comparison.loc[comparison["assignment_group"].eq(assignment_group)]
        .merge(
            meta[["assignment_group", "emfacId", "model_year_group", "fuel"]],
            on=["assignment_group", "emfacId"],
            how="left",
        )
        .loc[lambda df: df["model_year_group"].notna() & df["model_year_group"].ne("") & df["fuel"].notna() & df["fuel"].ne("")]
    )
    beam = (
        beam.groupby(["model_year_group", "fuel"], dropna=False)["beam_vmt"]
        .sum()
        .reset_index()
    )
    return (
        beam.merge(target, on=["model_year_group", "fuel"], how="outer")
        .fillna({"beam_vmt": 0.0, "emfac_vmt": 0.0})
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
        class_groups = [c for c in CLASS_GROUP_ORDER if c in grouped["class_group"].unique()]
        local_years = sorted(grouped["model_year_group"].unique().tolist(), key=_model_year_sort_key)
        display_labels = [CLASS_GROUP_LABELS.get(c, c) for c in class_groups]
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
    fig.savefig(output_path, dpi=PLOT_DPI)
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
    fig.savefig(output_path, dpi=PLOT_DPI)
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
    population_sample: float = 1.0,
    transit_sample: float = 1.0,
    freight_sample: Optional[float] = None,
) -> dict[str, str]:
    log_step_banner("Postprocess Step 1", "Compare Fleet", logger=logger)
    log_substep_banner("1.1", "compare BEAM fleet against EMFAC by emfacId", logger=logger)
    progress = _step_progress(7, "Postprocess Step 1")
    try:
        _set_progress_task(progress, "vehicle lookup", step_label="Postprocess Step 1")
        vehicle_lookup = _load_beam_vehicle_lookup(
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            population_sample=population_sample,
            transit_sample=transit_sample,
            freight_sample=freight_sample,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "BEAM assignments", step_label="Postprocess Step 1")
        beam_population_breakdown = _build_beam_population_breakdown_with_actual_assignments(
            vehicle_lookup=vehicle_lookup,
            passenger_vehicles_path=passenger_vehicles_path,
            freight_carriers_path=freight_carriers_path,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "BEAM skims VMT", step_label="Postprocess Step 1")
        beam_vmt_breakdown = _build_beam_vmt_breakdown(
            skims_emissions_path=skims_emissions_path,
            vehicle_lookup=vehicle_lookup,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "comparison table", step_label="Postprocess Step 1")
        comparison = _build_comparison_table(
            beam_population_breakdown=beam_population_breakdown,
            beam_vmt_breakdown=beam_vmt_breakdown,
            emfac_passenger_activity_path=emfac_passenger_activity_path,
            emfac_freight_activity_path=emfac_freight_activity_path,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "write tables", step_label="Postprocess Step 1")
        outputs = _write_tables(comparison, output_dir=output_dir)
        _advance_progress(progress)

        _set_progress_task(progress, "vehicle metadata", step_label="Postprocess Step 1")
        metadata = _load_vehicle_metadata_lookup(
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            population_sample=population_sample,
            transit_sample=transit_sample,
            freight_sample=freight_sample,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "plots", step_label="Postprocess Step 1")
        plot_path = _plot_class_model_year_combined(comparison, metadata=metadata, output_dir=output_dir)
        if plot_path:
            outputs["vmt_class_model_year_plot"] = plot_path
        plot_path = _plot_model_year_fuel_combined(comparison, metadata=metadata, output_dir=output_dir)
        if plot_path:
            outputs["vmt_model_year_fuel_plot"] = plot_path
        _advance_progress(progress)
    finally:
        _close_progress(progress)
    logger.info("Postprocess Step 1 complete")
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 1 from a pipeline output directory using manifest-resolved paths."""
    from impacts.postprocessor import (
        _load_pipeline_manifest,
        _resolve_inventory_emfacid_activity_path,
        _resolve_optional_population_assignment_paths,
        _resolve_skims_emissions_path,
        _resolve_vehicle_types_paths,
    )

    from ._common import settings_path_from_output_dir

    output_dir = Path(output_dir)
    run_manifest_path = output_dir / "pipeline_manifest.yaml"
    settings_path = settings_path_from_output_dir(output_dir)
    _, run_manifest = _load_pipeline_manifest(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    pipeline = run_manifest.get("pipeline", {}) or {}
    skims_path = _resolve_skims_emissions_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    passenger_vt, freight_vt = _resolve_vehicle_types_paths(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    passenger_pop, freight_pop = _resolve_optional_population_assignment_paths(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    passenger_activity = _resolve_inventory_emfacid_activity_path(
        settings_path,
        manifest_key="passenger_inventory_emfacid_file",
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    freight_activity = _resolve_inventory_emfacid_activity_path(
        settings_path,
        manifest_key="freight_inventory_emfacid_file",
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    return run(
        skims_emissions_path=str(skims_path),
        passenger_vehicle_types_path=str(passenger_vt),
        freight_vehicle_types_path=str(freight_vt),
        emfac_passenger_activity_path=str(passenger_activity),
        emfac_freight_activity_path=str(freight_activity),
        output_dir=output_dir / "postprocess" / "fleet",
        passenger_vehicles_path=str(passenger_pop) if passenger_pop else None,
        freight_carriers_path=str(freight_pop) if freight_pop else None,
        population_sample=float(pipeline.get("population_sample", 1.0)),
        transit_sample=float(pipeline.get("transit_sample", 1.0)),
        freight_sample=(
            float(pipeline["freight_sample"])
            if pipeline.get("freight_sample") is not None
            and str(pipeline.get("freight_sample")).strip() != ""
            else None
        ),
    )


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step1_compare_fleet",
        description="Run fleet comparison from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path,
                        help="Path to the main pipeline output folder.")
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir))
