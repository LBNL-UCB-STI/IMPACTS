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
import pandas as pd

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_table

logger = logging.getLogger(__name__)


def _normalize_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "target"


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


def _build_beam_population_breakdown(vehicle_lookup: pd.DataFrame) -> pd.DataFrame:
    return (
        vehicle_lookup.groupby(["assignment_group", "emfacId"], dropna=False)["sampleProbabilityWithinCategory"]
        .sum()
        .reset_index(name="beam_population_weight")
    )


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
    fallback = _build_beam_population_breakdown(vehicle_lookup)
    outputs: list[pd.DataFrame] = []
    used_groups: set[str] = set()
    if passenger_vehicles_path:
        outputs.append(
            _build_beam_population_breakdown_from_assignments(
                assignments_path=passenger_vehicles_path,
                vehicle_lookup=vehicle_lookup,
                assignment_group="passenger",
            )
        )
        used_groups.add("passenger")
    if freight_carriers_path:
        outputs.append(
            _build_beam_population_breakdown_from_assignments(
                assignments_path=freight_carriers_path,
                vehicle_lookup=vehicle_lookup,
                assignment_group="freight",
            )
        )
        used_groups.add("freight")
    fallback_groups = fallback.loc[~fallback["assignment_group"].isin(used_groups)].copy()
    if not fallback_groups.empty:
        outputs.append(fallback_groups)
    if not outputs:
        return fallback
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
    for column in ["beam_population_weight", "beam_vmt", "emfac_population", "emfac_vmt"]:
        comparison[column] = pd.to_numeric(comparison.get(column, 0.0), errors="coerce").fillna(0.0)
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


def _plot_metric(
    comparison: pd.DataFrame,
    *,
    assignment_group: str,
    metric: str,
    output_dir: Path,
) -> Optional[str]:
    subset = comparison.loc[comparison["assignment_group"].eq(assignment_group)].copy()
    if subset.empty:
        return None
    beam_column = f"beam_{metric}"
    emfac_column = f"emfac_{metric}"
    if beam_column not in subset.columns or emfac_column not in subset.columns:
        return None
    subset = subset.sort_values(beam_column, ascending=False).reset_index(drop=True)
    x = range(len(subset))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(subset) * 0.55), 5.5))
    ax.bar(
        [pos - width / 2 for pos in x],
        subset[beam_column].to_numpy(dtype=float),
        width=width,
        label="BEAM",
        color="#1f77b4",
    )
    ax.bar(
        [pos + width / 2 for pos in x],
        subset[emfac_column].to_numpy(dtype=float),
        width=width,
        label="EMFAC",
        color="#ff7f0e",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset["emfacId"].tolist(), rotation=70, ha="right")
    ax.set_ylabel("Total")
    ax.set_title(f"{assignment_group.title()} {metric.upper()} by EMFAC ID")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path = output_dir / f"step1_{_slugify(assignment_group)}_{_slugify(metric)}_by_emfacid.png"
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
    for assignment_group in ("passenger", "freight"):
        for metric in ("population", "vmt"):
            plot_path = _plot_metric(
                comparison,
                assignment_group=assignment_group,
                metric=metric,
                output_dir=output_dir,
            )
            if plot_path:
                outputs[f"{assignment_group}_{metric}_plot"] = plot_path
    logger.info("Analysis Step 1 complete")
    return outputs
