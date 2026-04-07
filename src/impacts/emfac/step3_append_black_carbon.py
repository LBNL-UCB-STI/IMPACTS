from __future__ import annotations

from pathlib import Path

import pandas as pd

from impacts.emfac.common import assert_row_count
from impacts.emfac.common import frame_summary
from impacts.emfac.common import write_trace

RATE_COLUMN = "rateGram"
SPEED_COLUMN = "speedMps_timeMin"
PROJECT_ANALYSIS_COLUMNS = [
    "county",
    "vehicleCategory",
    "fuel",
    "modelYear",
    "process",
    SPEED_COLUMN,
    "pollutant",
    RATE_COLUMN,
]
BLACK_CARBON_POLLUTANTS = {"BC", "BCm", "BCh"}


def _read_project_analysis(path: str) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported project-analysis format for {target}. Expected .parquet")
    frame = pd.read_parquet(target)
    missing = [column for column in PROJECT_ANALYSIS_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Project-analysis parquet is missing required columns: {', '.join(missing)}")
    return frame[PROJECT_ANALYSIS_COLUMNS].copy()


def _normalize_black_carbon_rates(path: str, region_label: str | None) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if target.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported black-carbon format for {target}. Expected .csv")

    frame = pd.read_csv(target)
    missing = [
        column
        for column in ["sub_area", "vehicle_class", "fuel", "model_year", "process", "speed_time", "pollutant", "emission_rate", "calendar_year"]
        if column not in frame.columns
    ]
    if missing:
        raise ValueError(f"Black-carbon CSV is missing required columns: {', '.join(missing)}")

    if region_label:
        suffix = f"({region_label})"
        frame = frame.loc[frame["sub_area"].astype(str).str.endswith(suffix)].copy()
        if frame.empty:
            raise ValueError(f"No black-carbon rows matched region label {region_label!r} in {target}")

    frame["county"] = frame["sub_area"].astype(str).str.replace(r"\s*\([^)]*\)$", "", regex=True).str.strip()
    frame["vehicleCategory"] = frame["vehicle_class"].astype(str).str.strip()
    frame = frame.loc[frame["pollutant"].isin(BLACK_CARBON_POLLUTANTS)].copy()
    for column in ["model_year", "speed_time", "emission_rate"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["modelYear"] = frame["model_year"].astype(int)
    frame[SPEED_COLUMN] = frame["speed_time"]
    frame[RATE_COLUMN] = frame["emission_rate"]
    for column in ["county", "vehicleCategory", "fuel", "process", "pollutant"]:
        frame[column] = frame[column].astype(str).str.strip()
    return frame[PROJECT_ANALYSIS_COLUMNS].drop_duplicates().reset_index(drop=True)


def _load_black_carbon_inputs(
    *,
    black_carbon_rates_path: str,
    project_analysis_path: str,
    region_label: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    project_analysis = _read_project_analysis(project_analysis_path)
    black_carbon_rates = _normalize_black_carbon_rates(black_carbon_rates_path, region_label)
    return project_analysis, black_carbon_rates


def _append_black_carbon_rows(
    project_analysis: pd.DataFrame,
    black_carbon_rates: pd.DataFrame,
    *,
    drop_existing_black_carbon: bool,
) -> pd.DataFrame:
    result = project_analysis
    if drop_existing_black_carbon:
        result = result.loc[~result["pollutant"].isin(BLACK_CARBON_POLLUTANTS)].copy()
    return pd.concat([result, black_carbon_rates], ignore_index=True)


def _write_project_analysis(frame: pd.DataFrame, output_path: str) -> str:
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Unsupported output format for {target}. Expected .parquet")
    frame.to_parquet(target, index=False)
    return str(target)


def merge_black_carbon_into_project_analysis(
    *,
    black_carbon_rates_path: str,
    project_analysis_path: str,
    output_path: str,
    region_label: str | None = "SF",
    drop_existing_black_carbon: bool = True,
) -> str:
    project_analysis, black_carbon_rates = _load_black_carbon_inputs(
        black_carbon_rates_path=black_carbon_rates_path,
        project_analysis_path=project_analysis_path,
        region_label=region_label,
    )
    result = _append_black_carbon_rows(
        project_analysis,
        black_carbon_rates,
        drop_existing_black_carbon=drop_existing_black_carbon,
    )
    return _write_project_analysis(result, output_path)


def run_step3(workflow: dict[str, object]) -> dict[str, object]:
    print("  Step 3. Append Black Carbon")
    print("    3.1 Load project-analysis and black carbon inputs")
    project_analysis, black_carbon_rates = _load_black_carbon_inputs(
        black_carbon_rates_path=workflow["inputs"]["black_carbon_raw"],
        project_analysis_path=workflow["paths"]["project_analysis_with_nh3"],
        region_label=workflow["run"]["region_label"],
    )
    print("    3.2 Build and append BC rows")
    result = _append_black_carbon_rows(project_analysis, black_carbon_rates, drop_existing_black_carbon=True)
    non_bc_input_rows = int((~project_analysis["pollutant"].isin(BLACK_CARBON_POLLUTANTS)).sum())
    non_bc_output_rows = int((~result["pollutant"].isin(BLACK_CARBON_POLLUTANTS)).sum())
    assert_row_count(non_bc_input_rows, non_bc_output_rows, label="Black-carbon append non-BC preservation")
    assert_row_count(
        len(black_carbon_rates),
        int(result["pollutant"].isin(BLACK_CARBON_POLLUTANTS).sum()),
        label="Black-carbon row append",
    )
    print("    3.3 Write project-analysis with BC")
    _write_project_analysis(result, workflow["paths"]["project_analysis_with_nh3_bc"])
    write_trace(
        workflow,
        "step3_append_black_carbon",
        {
            "input": frame_summary(project_analysis, name="project_analysis_with_nh3"),
            "black_carbon_rows": frame_summary(black_carbon_rates, name="black_carbon_rows"),
            "result": frame_summary(result, name="project_analysis_with_nh3_bc"),
        },
    )
    return workflow
