"""Export IMPACTS human-exposure results in the original Direct Final CSV format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ALL_GROUP = "All Bay Area valid-CES"
BAY_COUNTIES = {
    "Alameda", "Contra Costa", "Marin", "Napa", "San Francisco",
    "San Mateo", "Santa Clara", "Solano", "Sonoma",
}

METHODS = (
    ("PM2.5", "Hybrid/AERMOD-enhanced", "Hybrid_TotalPM25"),
    ("PM2.5", "Traffic-related integrated", "Traffic_TotalPM25"),
    ("PM2.5", "InMAP-only", "InMAP_TotalPM25"),
    ("PM2.5", "AERMOD-only component", "AERMOD_TotalPM25"),
    ("PM2.5", "Power-generation component", "Power_TotalPM25"),
    ("BC", "Indicator-corrected final", "Final_BC"),
    ("BC", "InMAP-only", "InMAP_BC"),
    ("BC", "AERMOD-only", "AERMOD_BC"),
    ("NO2", "Indicator-corrected final", "Final_NO2"),
    ("NO2", "InMAP-only", "InMAP_NO2"),
    ("NO2", "AERMOD-only", "AERMOD_NO2"),
)

COMPARISONS = (
    ("PM2.5", "InMAP-only", "InMAP_TotalPM25", "Hybrid/AERMOD-enhanced", "Hybrid_TotalPM25"),
    ("BC", "InMAP-only", "InMAP_BC", "Indicator-corrected final", "Final_BC"),
    ("NO2", "InMAP-only", "InMAP_NO2", "Indicator-corrected final", "Final_NO2"),
)


def safe_percent(difference: float, reference: float) -> float:
    return difference / reference * 100.0 if np.isfinite(reference) and reference != 0 else np.nan


def subsets(data: pd.DataFrame):
    for group_id in (1, 2, 3, 4):
        group = data.loc[data["CES_GroupID"].eq(group_id)]
        yield str(group["CES_Group"].iloc[0]), group_id, group
    yield ALL_GROUP, 0, data


def make_summary(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_population = float(data["ModelPopulation"].sum())
    method_totals = {
        field: float(data[f"{field}_Exposure"].sum()) for _, _, field in METHODS
    }
    for group_label, group_id, group in subsets(data):
        population = float(group["ModelPopulation"].sum())
        population_share = population / total_population if total_population else np.nan
        for pollutant, method, field in METHODS:
            exposure = float(group[f"{field}_Exposure"].sum())
            rows.append(
                {
                    "CES_Group": group_label,
                    "CES_GroupID": group_id,
                    "Pollutant": pollutant,
                    "Method": method,
                    "MethodField": field,
                    "TractCount": int(len(group)),
                    "ModelPopulation": population,
                    "CESPopulation": float(group["CESPopulation"].sum()),
                    "PopulationWeightedConcentration_ug_m3": exposure / population if population else np.nan,
                    "TotalExposure_person_ug_m3": exposure,
                    "ExposureShare": exposure / method_totals[field] if method_totals[field] else np.nan,
                    "PopulationShare": population_share,
                }
            )
    return pd.DataFrame(rows)


def make_comparison(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_label, group_id, group in subsets(data):
        population = float(group["ModelPopulation"].sum())
        for pollutant, ref_name, ref_field, cmp_name, cmp_field in COMPARISONS:
            ref_exposure = float(group[f"{ref_field}_Exposure"].sum())
            cmp_exposure = float(group[f"{cmp_field}_Exposure"].sum())
            ref_pw = ref_exposure / population if population else np.nan
            cmp_pw = cmp_exposure / population if population else np.nan
            pw_difference = cmp_pw - ref_pw
            exposure_difference = cmp_exposure - ref_exposure
            rows.append(
                {
                    "CES_Group": group_label,
                    "CES_GroupID": group_id,
                    "Pollutant": pollutant,
                    "TractCount": int(len(group)),
                    "ModelPopulation": population,
                    "ReferenceMethod": ref_name,
                    "ComparisonMethod": cmp_name,
                    "Reference_PWConcentration_ug_m3": ref_pw,
                    "Comparison_PWConcentration_ug_m3": cmp_pw,
                    "Difference_PWConcentration_ug_m3": pw_difference,
                    "PercentDifference_PWConcentration": safe_percent(pw_difference, ref_pw),
                    "Reference_TotalExposure_person_ug_m3": ref_exposure,
                    "Comparison_TotalExposure_person_ug_m3": cmp_exposure,
                    "Difference_TotalExposure_person_ug_m3": exposure_difference,
                    "PercentDifference_TotalExposure": safe_percent(exposure_difference, ref_exposure),
                }
            )
    return pd.DataFrame(rows)


def export(source: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    data = pd.read_parquet(source)
    # The buffered concentration domain can intersect neighboring counties.
    # Match the nine-county, valid-CES selection used by the original plots.
    data = data.loc[
        data["County"].isin(BAY_COUNTIES)
        & data["CES_GroupID"].isin([1, 2, 3, 4])
        & pd.to_numeric(data["ModelPopulation"], errors="coerce").gt(0)
    ].copy()
    if data.empty:
        raise ValueError("No populated, valid-CES tracts in the nine Bay Area counties.")
    aliases = {
        "Hybrid_TotalPM25_Exposure": "Final_TotalPM25_Exposure",
        "Hybrid_TotalPM25_PW": "Final_TotalPM25_PW",
    }
    for target, source_field in aliases.items():
        if target not in data and source_field in data:
            data[target] = data[source_field]

    base_columns = [
        "TractID", "County", "CIscoreP", "CES_Group", "CES_GroupID",
        "CESPopulation", "BEAM_POP", "ModelGridCount", "ModelPopulation",
    ]
    if "CESPopulation" not in data:
        data["CESPopulation"] = data["TotPop19"]

    metric_fields = [field for _, _, field in METHODS]
    required = set(base_columns) | {
        f"{field}_{suffix}" for field in metric_fields for suffix in ("Exposure", "PW")
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise KeyError(f"Missing fields required by original table export: {missing}")

    original_metric_order = [
        "Hybrid_TotalPM25", "InMAP_TotalPM25", "AERMOD_TotalPM25",
        "Final_BC", "InMAP_BC", "AERMOD_BC",
        "Final_NO2", "InMAP_NO2", "AERMOD_NO2",
    ]
    added_metric_order = ["Traffic_TotalPM25", "Power_TotalPM25"]
    tract_columns = base_columns + [
        f"{field}_Exposure" for field in original_metric_order
    ] + [
        f"{field}_PW" for field in original_metric_order
    ] + [
        f"{field}_{suffix}" for field in added_metric_order for suffix in ("Exposure", "PW")
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    tract_path = output_dir / "CES25_Direct_Final_tracts.csv"
    summary_path = output_dir / "CES25_Direct_Final_Summary.csv"
    comparison_path = output_dir / "CES25_Direct_Final_Comparison.csv"
    data.loc[:, tract_columns].to_csv(tract_path, index=False)
    make_summary(data).to_csv(summary_path, index=False)
    make_comparison(data).to_csv(comparison_path, index=False)
    return tract_path, summary_path, comparison_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    for path in export(args.input, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
