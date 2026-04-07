from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

MPH_TO_MPS = 0.44704
US_TON_TO_KG = 907.18474
RATE_COLUMN = "rateGram"
SPEED_COLUMN = "speedMps_timeMin"
VMT_WEIGHTED_PROCESSES = {"RUNEX", "PMBW", "PMTW"}
ACTIVITY_JOIN_COLUMNS = ["county", "vehicleCategory", "fuel", "modelYear"]
RATE_GROUP_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    "speedMps_timeMin_weightKg",
    "roadCategory",
    "pollutant",
]
FINAL_OUTPUT_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    "speedMps_timeMin_weightKg",
    "roadCategory",
    "pollutant",
    "rateGram",
]


def _model_year_group_label(group: dict[str, object]) -> str:
    min_year = group.get("min_year")
    max_year = group.get("max_year")
    if min_year is None:
        return "pre04"
    if max_year is None:
        return "post16"
    if int(min_year) == 2004 and int(max_year) == 2013:
        return "04to13"
    if int(min_year) == 2014 and int(max_year) == 2016:
        return "14to16"
    return f"{int(min_year)}to{int(max_year)}"


def _model_year_label(model_year: int, groups: list[dict[str, object]]) -> str:
    for group in groups:
        min_year = group.get("min_year")
        max_year = group.get("max_year")
        if (min_year is None or model_year >= int(min_year)) and (max_year is None or model_year <= int(max_year)):
            return _model_year_group_label(group)
    raise ValueError(f"Model year {model_year} is not covered by the configured model_year_groups")


def _format_numeric_series(values: pd.Series) -> pd.Series:
    return values.map(lambda value: f"{float(value):g}" if pd.notna(value) else pd.NA)


def _build_speed_time_weight(frame: pd.DataFrame) -> pd.Series:
    values = pd.Series(pd.NA, index=frame.index, dtype="object")
    prdust_mask = (frame["process"] == "PRDUST") & frame["vehicle_weight_tons"].notna()
    speed_mask = frame["process"].isin({"RUNEX", "PMBW"}) & frame[SPEED_COLUMN].notna()
    time_mask = (frame["process"] == "STREX") & frame[SPEED_COLUMN].notna()
    if prdust_mask.any():
        values.loc[prdust_mask] = "weightKg=" + _format_numeric_series(
            frame.loc[prdust_mask, "vehicle_weight_tons"] * US_TON_TO_KG
        )
    if speed_mask.any():
        values.loc[speed_mask] = "speedMps=" + _format_numeric_series(
            frame.loc[speed_mask, SPEED_COLUMN] * MPH_TO_MPS
        )
    if time_mask.any():
        values.loc[time_mask] = "timeMin=" + _format_numeric_series(frame.loc[time_mask, SPEED_COLUMN])
    return values


def _build_activity_weights(emissions_inventory_path: str) -> pd.DataFrame:
    weights = pd.read_parquet(
        Path(emissions_inventory_path).expanduser(),
        columns=["county", "vehicleCategory", "fuel", "modelYear", "total_vmt", "population", "trips"],
    )
    weights["county"] = weights["county"].astype(str)
    weights["vehicleCategory"] = weights["vehicleCategory"].astype(str)
    weights["fuel"] = weights["fuel"].astype(str)
    weights["modelYear"] = pd.to_numeric(weights["modelYear"], errors="raise").astype(int)
    weights["total_vmt"] = pd.to_numeric(weights["total_vmt"], errors="coerce")
    weights["population"] = pd.to_numeric(weights["population"], errors="coerce")
    weights["trips"] = pd.to_numeric(weights["trips"], errors="coerce")
    return (
        weights.groupby(ACTIVITY_JOIN_COLUMNS, dropna=False)[["total_vmt", "population", "trips"]]
        .sum(min_count=1)
        .reset_index()
    )

def _assign_model_year_groups(frame: pd.DataFrame, model_year_groups: list[dict[str, object]]) -> pd.DataFrame:
    labels = pd.Series(pd.NA, index=frame.index, dtype="object")
    model_year = frame["modelYear"].astype(int)
    for group in model_year_groups:
        min_year = group.get("min_year")
        max_year = group.get("max_year")
        mask = pd.Series(True, index=frame.index)
        if min_year is not None:
            mask &= model_year >= int(min_year)
        if max_year is not None:
            mask &= model_year <= int(max_year)
        labels.loc[mask] = _model_year_group_label(group)
    if labels.isna().any():
        missing_years = sorted(model_year.loc[labels.isna()].unique().tolist())
        raise ValueError(f"Model years {missing_years} are not covered by the configured model_year_groups")
    frame["modelYearGroup"] = labels
    return frame


def _build_study_area_wide_fleet(activity_weights: pd.DataFrame, model_year_groups: list[dict[str, object]]) -> pd.DataFrame:
    fleet = _assign_model_year_groups(
        activity_weights[["vehicleCategory", "fuel", "modelYear", "total_vmt"]],
        model_year_groups,
    )
    fleet = (
        fleet.groupby(["vehicleCategory", "fuel", "modelYearGroup"], dropna=False)["total_vmt"]
        .sum(min_count=1)
        .reset_index()
    )
    total_vmt = fleet["total_vmt"].sum(min_count=1)
    fleet["vmtShare"] = fleet["total_vmt"] / total_vmt if pd.notna(total_vmt) and total_vmt > 0 else pd.NA
    return fleet.rename(columns={"modelYearGroup": "modelYear"}).drop(columns=["total_vmt"])


def _load_finalize_inputs(workflow: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = Path(workflow["paths"]["project_analysis_with_nh3_bc_prdust"]).expanduser()
    frame = pd.read_parquet(source)
    activity_weights = _build_activity_weights(workflow["paths"]["emissions_inventory_and_activities"])
    return frame, activity_weights


def _build_finalize_outputs(
    frame: pd.DataFrame,
    activity_weights: pd.DataFrame,
    *,
    model_year_groups: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame["roadCategory"] = frame["road_category"]
    frame["modelYear"] = frame["modelYear"].astype(int)
    frame = frame.merge(
        activity_weights,
        on=ACTIVITY_JOIN_COLUMNS,
        how="left",
    )
    frame = _assign_model_year_groups(frame, model_year_groups)
    frame = frame.drop(columns=["modelYear"]).rename(columns={"modelYearGroup": "modelYear"})
    frame["speedMps_timeMin_weightKg"] = _build_speed_time_weight(frame)
    frame["rateGram"] = pd.to_numeric(frame[RATE_COLUMN], errors="coerce")
    frame["aggregation_weight"] = np.nan
    vmt_mask = frame["process"].isin(VMT_WEIGHTED_PROCESSES)
    prdust_mask = frame["process"] == "PRDUST"
    frame.loc[vmt_mask, "aggregation_weight"] = frame.loc[vmt_mask, "total_vmt"]
    frame.loc[prdust_mask, "aggregation_weight"] = frame.loc[prdust_mask, "population"]
    frame.loc[~vmt_mask & ~prdust_mask, "aggregation_weight"] = frame.loc[~vmt_mask & ~prdust_mask, "trips"]
    frame["valid_weight"] = frame["aggregation_weight"].notna() & (frame["aggregation_weight"] > 0) & frame["rateGram"].notna()
    frame["weighted_rate"] = frame["rateGram"] * frame["aggregation_weight"]

    weighted = (
        frame.loc[frame["valid_weight"]]
        .groupby(RATE_GROUP_COLUMNS, dropna=False)[["weighted_rate", "aggregation_weight"]]
        .sum()
        .reset_index()
    )
    weighted["rateGram"] = weighted["weighted_rate"] / weighted["aggregation_weight"]

    fallback = (
        frame.loc[~frame["valid_weight"]]
        .groupby(RATE_GROUP_COLUMNS, dropna=False)["rateGram"]
        .mean()
        .rename("rateGram")
        .reset_index()
    )
    grouped = pd.concat([weighted[RATE_GROUP_COLUMNS + ["rateGram"]], fallback], ignore_index=True)
    rates = grouped[FINAL_OUTPUT_COLUMNS].copy()
    fleet = _build_study_area_wide_fleet(activity_weights, model_year_groups)
    return rates, fleet


def _write_finalize_outputs(rates: pd.DataFrame, fleet: pd.DataFrame, workflow: dict[str, object]) -> None:
    target = Path(workflow["paths"]["final_output"]).expanduser()
    fleet_target = Path(workflow["paths"]["final_fleet_output"]).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    rates.to_parquet(target, index=False)
    fleet.to_parquet(fleet_target, index=False)


def run_step5(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 5. Finalize Compact Output")
    print("    5.1 Load assembled rates and activity weights")
    frame, activity_weights = _load_finalize_inputs(workflow)
    print("    5.2 Build final rate table and study-area fleet shares")
    rates, fleet = _build_finalize_outputs(
        frame,
        activity_weights,
        model_year_groups=workflow["run"]["model_year_groups"],
    )
    print("    5.3 Write final outputs")
    _write_finalize_outputs(rates, fleet, workflow)
    write_trace(
        workflow,
        "step5_finalize_output",
        {
            "input": frame_summary(frame, name="project_analysis_with_nh3_bc_prdust"),
            "activity_weights": frame_summary(activity_weights, name="activity_weights"),
            "rates": frame_summary(rates, name="final_rates"),
            "fleet": frame_summary(fleet, name="final_fleet"),
        },
    )
    return workflow
