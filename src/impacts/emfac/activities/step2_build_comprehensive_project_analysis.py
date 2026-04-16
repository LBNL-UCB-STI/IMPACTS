from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

ACTIVITY_COLUMN = "speedMph_timeMin"
SPEED_PROCESSES = ["RUNEX", "PMBW", "PTOEX"]
TIME_PROCESSES = ["STREX"]
OTHER_PROCESSES = ["DIURN", "HOTSOAK", "IDLEX", "PMTW", "RUNLOSS"]
PTO_VEHICLE_CATEGORIES = [
    "T7 Public Class 8",
    "T7 Utility Class 8",
    "T7 Single Concrete/Transit Mix Class 8",
    "T7 Single Dump Class 8",
    "T7 Single Other Class 8",
    "T7 SWCV Class 8",
]
POLLUTANT_COLUMNS = [
    "bc_gram",
    "ch4_gram",
    "co_gram",
    "co2_gram",
    "hc_gram",
    "nh3_gram",
    "nox_gram",
    "pm_gram",
    "pm10_gram",
    "pm25_gram",
    "rog_gram",
    "sox_gram",
    "tog_gram",
]
GROUP_KEYS = ["county", "vehicleCategory", "fuel", "modelYear", "process"]


def _format_numeric_series(values: pd.Series) -> pd.Series:
    return values.map(lambda value: f"{float(value):g}" if pd.notna(value) else pd.NA)


def _filter_to_supported_process_groups(
    rates: pd.DataFrame,
    *,
    project_analysis: pd.DataFrame,
    project_analysis_prdust: pd.DataFrame,
) -> pd.DataFrame:
    project_analysis_groups = project_analysis[GROUP_KEYS].drop_duplicates().assign(_supported=True)
    prdust_groups = project_analysis_prdust[GROUP_KEYS].drop_duplicates().assign(_supported=True)
    supported_groups = pd.concat([project_analysis_groups, prdust_groups], ignore_index=True).drop_duplicates()

    filtered = rates.merge(supported_groups, on=GROUP_KEYS, how="left")
    dropped = filtered.loc[filtered["_supported"].isna(), GROUP_KEYS].drop_duplicates().reset_index(drop=True)
    kept = filtered.loc[filtered["_supported"].notna()].drop(columns="_supported").reset_index(drop=True)
    dropped_group_count = int(len(dropped))
    if dropped_group_count > 0:
        print(
            f"    2.2 Drop unsupported county/vehicle/fuel/modelYear/process combinations not present in EMFAC source: "
            f"{dropped_group_count:,} group(s)"
        )
    return kept


def build_comprehensive_project_analysis(
    project_analysis: pd.DataFrame,
    *,
    project_analysis_prdust: pd.DataFrame,
    emissions_inventory: pd.DataFrame,
    pollutant_columns: list[str] | None = None,
) -> pd.DataFrame:
    base_rates = (
        project_analysis[["county", "vehicleCategory", "fuel", "modelYear"]]
        .drop_duplicates()
        .merge(
            emissions_inventory[["county", "vehicleCategory", "fuel", "modelYear"]].drop_duplicates(),
            on=["county", "vehicleCategory", "fuel", "modelYear"],
            how="inner",
        )
        .reset_index(drop=True)
    )

    speed_processes = pd.DataFrame({"process": ["RUNEX", "PMBW"]})
    pto_processes = pd.DataFrame({"process": ["PTOEX"]})
    time_processes = pd.DataFrame({"process": TIME_PROCESSES})
    other_processes = pd.DataFrame({"process": OTHER_PROCESSES})

    speeds = pd.DataFrame(
        {
            "speed_time": sorted(
                project_analysis.loc[
                    project_analysis["process"].isin(SPEED_PROCESSES),
                    ACTIVITY_COLUMN,
                ]
                .dropna()
                .drop_duplicates()
                .tolist()
            )
        }
    )

    times = pd.DataFrame(
        {
            "speed_time": sorted(
                project_analysis.loc[
                    project_analysis["process"].isin(TIME_PROCESSES),
                    ACTIVITY_COLUMN,
                ]
                .dropna()
                .drop_duplicates()
                .tolist()
            )
        }
    )

    speed_rates = base_rates.merge(speed_processes, how="cross").merge(speeds, how="cross")
    speed_rates["roadCategory"] = pd.NA

    pto_base_rates = base_rates.loc[
        base_rates["vehicleCategory"].astype(str).isin(PTO_VEHICLE_CATEGORIES)
    ].copy()
    pto_rates = pto_base_rates.merge(pto_processes, how="cross").merge(speeds, how="cross")
    pto_rates["roadCategory"] = pd.NA

    time_rates = base_rates.merge(time_processes, how="cross").merge(times, how="cross")
    time_rates["roadCategory"] = pd.NA

    prdust_rates = project_analysis_prdust[
        ["county", "vehicleCategory", "fuel", "modelYear", "process", "roadCategory", ACTIVITY_COLUMN]
    ].drop_duplicates().reset_index(drop=True)

    other_rates = base_rates.merge(other_processes, how="cross")
    other_rates["speed_time"] = pd.NA
    other_rates[ACTIVITY_COLUMN] = pd.NA
    other_rates["roadCategory"] = pd.NA

    rates = pd.concat(
        [speed_rates, pto_rates, time_rates, prdust_rates, other_rates],
        ignore_index=True,
        sort=False,
    )
    if "speed_time" in rates.columns:
        speed_time = rates.pop("speed_time")
        if ACTIVITY_COLUMN in rates.columns:
            rates[ACTIVITY_COLUMN] = speed_time.combine_first(rates[ACTIVITY_COLUMN])
        else:
            rates[ACTIVITY_COLUMN] = speed_time
    values = pd.Series(pd.NA, index=rates.index, dtype="object")
    process = rates["process"].astype(str)
    speed_values = pd.to_numeric(rates[ACTIVITY_COLUMN], errors="coerce")
    speed_mask = process.isin(SPEED_PROCESSES) & speed_values.notna()
    time_mask = (process == "STREX") & speed_values.notna()
    prdust_mask = process == "PRDUST"
    if speed_mask.any():
        values.loc[speed_mask] = _format_numeric_series(speed_values.loc[speed_mask])
    if time_mask.any():
        values.loc[time_mask] = _format_numeric_series(speed_values.loc[time_mask])
    if prdust_mask.any():
        values.loc[prdust_mask] = pd.NA
    rates[ACTIVITY_COLUMN] = values
    rates = _filter_to_supported_process_groups(
        rates,
        project_analysis=project_analysis,
        project_analysis_prdust=project_analysis_prdust,
    )
    for column in pollutant_columns or []:
        rates[column] = pd.NA
    return rates.reset_index(drop=True)


def _write_parquet(frame: pd.DataFrame, path: str) -> str:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return str(target)


def run_step2(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 2. Build Comprehensive Project Analysis")
    print("    2.1 Build comprehensive output surface")
    project_analysis = pd.read_parquet(Path(workflow["paths"]["project_analysis_source"]).expanduser().resolve())
    project_analysis_prdust = pd.read_parquet(Path(workflow["paths"]["project_analysis_prdust"]).expanduser().resolve())
    emissions_inventory = pd.read_parquet(Path(workflow["paths"]["emissions_inventory"]).expanduser().resolve())
    comprehensive = build_comprehensive_project_analysis(
        project_analysis,
        project_analysis_prdust=project_analysis_prdust,
        emissions_inventory=emissions_inventory,
        pollutant_columns=POLLUTANT_COLUMNS,
    )
    _write_parquet(comprehensive, workflow["paths"]["project_analysis"])
    write_trace(
        workflow,
        "step2_build_comprehensive_project_analysis",
        {
            "project_analysis": frame_summary(project_analysis, name="project_analysis_source"),
            "project_analysis_prdust": frame_summary(project_analysis_prdust, name="project_analysis_prdust"),
            "emissions_inventory": frame_summary(emissions_inventory, name="emissions_inventory"),
            "output": frame_summary(comprehensive, name="comprehensive_project_analysis"),
        },
    )
    return workflow
