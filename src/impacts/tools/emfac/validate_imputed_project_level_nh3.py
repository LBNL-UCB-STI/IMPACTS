from __future__ import annotations
"""Validate NH3 rows added to cleaned project-analysis parquet."""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

FUEL_MAP_OLD = {
    "Dsl": "diesel",
    "Gas": "gasoline",
    "Elec": "electricity",
    "NG": "naturalgas",
    "Phe": "pluginhybrid",
}

GROUP_KEYS = ["sub_area", "vehicle_class", "fuel"]


def _read_parquet(path: str | Path) -> pd.DataFrame:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Input path does not exist: {target}")
    if target.suffix.lower() != ".parquet":
        raise ValueError(f"Expected parquet input: {target}")
    return pd.read_parquet(target)


def _normalize_old_nh3(old_df: pd.DataFrame) -> pd.DataFrame:
    old_nh3 = old_df[(old_df["pollutant"] == "NH3") & (old_df["process"] == "RUNEX")].copy()
    old_nh3["sub_area"] = old_nh3["sub_area"].astype(str).str.replace(r"\s*\([^)]*\)\s*$", "", regex=True)
    old_nh3["fuel"] = old_nh3["fuel"].map(FUEL_MAP_OLD)
    return old_nh3


def _nh3_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["pollutant"] == "NH3") & (df["process"] == "RUNEX")].copy()


def _runex_keys(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df[df["process"] == "RUNEX"][
            ["calendar_year", "sub_area", "vehicle_class", "fuel", "model_year", "speed_time"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def _print_series(title: str, series: pd.Series, limit: int = 15) -> None:
    print(title)
    if series.empty:
        print("  none")
        return
    for key, value in series.head(limit).items():
        print(f"  {key}: {value}")


def validate_nh3(
    *,
    project_level_path: str,
    project_level_with_nh3_path: str,
    old_imputed_rates_path: str,
) -> None:
    base_df = _read_parquet(project_level_path)
    new_df = _read_parquet(project_level_with_nh3_path)
    old_df = _read_parquet(old_imputed_rates_path)

    base_nh3 = _nh3_rows(base_df)
    new_nh3 = _nh3_rows(new_df)
    old_nh3 = _normalize_old_nh3(old_df)

    added_nh3_rows = len(new_nh3) - len(base_nh3)
    print(f"nh3_rows_before={len(base_nh3)}")
    print(f"nh3_rows_after={len(new_nh3)}")
    print(f"nh3_rows_added={added_nh3_rows}")

    runex_keys = _runex_keys(new_df)
    nh3_keys = (
        new_nh3[["calendar_year", "sub_area", "vehicle_class", "fuel", "model_year", "speed_time"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    missing_nh3 = runex_keys.merge(
        nh3_keys,
        on=["calendar_year", "sub_area", "vehicle_class", "fuel", "model_year", "speed_time"],
        how="left",
        indicator=True,
    )
    missing_nh3 = missing_nh3[missing_nh3["_merge"] == "left_only"].drop(columns="_merge")
    print(f"remaining_missing_nh3_runex_rows={len(missing_nh3)}")

    _print_series(
        "remaining_missing_by_sub_area",
        missing_nh3["sub_area"].value_counts(),
    )
    _print_series(
        "remaining_missing_by_vehicle_class",
        missing_nh3["vehicle_class"].value_counts(),
    )
    _print_series(
        "remaining_missing_by_fuel",
        missing_nh3["fuel"].value_counts(),
    )

    new_compare = (
        new_nh3.groupby(["calendar_year", "sub_area", "vehicle_class", "fuel", "speed_time"], as_index=False)
        .agg(new_rate=("emission_rate", "mean"))
    )
    old_compare = (
        old_nh3.groupby(["calendar_year", "sub_area", "vehicle_class", "fuel", "speed_time"], as_index=False)
        .agg(old_rate=("emission_rate", "mean"))
    )
    comparison = old_compare.merge(
        new_compare,
        on=["calendar_year", "sub_area", "vehicle_class", "fuel", "speed_time"],
        how="inner",
    )
    comparison["abs_diff"] = (comparison["new_rate"] - comparison["old_rate"]).abs()
    comparison["pct_diff"] = comparison["abs_diff"] / comparison["old_rate"].replace(0, pd.NA) * 100

    print(f"overlap_rows_with_old={len(comparison)}")
    if not comparison.empty:
        print(f"overall_rate_correlation={comparison[['old_rate', 'new_rate']].corr().iloc[0, 1]:.6f}")
        print(f"overall_median_pct_diff={comparison['pct_diff'].dropna().median():.6f}")
        print(f"overall_mean_pct_diff={comparison['pct_diff'].dropna().mean():.6f}")

    by_group = (
        comparison.groupby(GROUP_KEYS, as_index=False)
        .agg(
            overlap_rows=("pct_diff", "size"),
            median_pct_diff=("pct_diff", "median"),
            mean_pct_diff=("pct_diff", "mean"),
        )
        .sort_values(["median_pct_diff", "mean_pct_diff"], ascending=False)
    )

    print("compare_new_vs_old_by_sub_area_vehicle_class_fuel")
    if by_group.empty:
        print("  none")
    else:
        for row in by_group.head(25).to_dict(orient="records"):
            print(
                "  "
                f"{row['sub_area']} | {row['vehicle_class']} | {row['fuel']} | "
                f"overlap_rows={row['overlap_rows']} | "
                f"median_pct_diff={row['median_pct_diff']:.6f} | "
                f"mean_pct_diff={row['mean_pct_diff']:.6f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m impacts.tools.emfac.validate_imputed_project_level_nh3",
        description="Print console-only validation for NH3 rows added to cleaned project-analysis parquet.",
    )
    parser.add_argument("--project-level", required=True, help="Base cleaned project-analysis parquet without NH3.")
    parser.add_argument("--project-level-with-nh3", required=True, help="Cleaned project-analysis parquet with NH3.")
    parser.add_argument("--old-imputed-rates", required=True, help="Old aggregated NH3 imputed parquet for comparison.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_nh3(
        project_level_path=args.project_level,
        project_level_with_nh3_path=args.project_level_with_nh3,
        old_imputed_rates_path=args.old_imputed_rates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
