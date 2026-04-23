from __future__ import annotations

import pandas as pd

LIGHT_DUTY_VEHICLE_CATEGORIES = {"LDA", "LDT1", "LDT2"}


def model_year_group_label(group: dict[str, object]) -> str:
    min_year = group.get("min_year")
    max_year = group.get("max_year")
    if min_year is None:
        return f"pre{int(max_year) + 1:02d}"
    if max_year is None:
        return f"post{int(min_year) - 1:02d}"
    return f"{int(min_year)}to{int(max_year)}"


def vehicle_group(vehicle_category: object) -> str:
    vehicle_category = str(vehicle_category).strip()
    if vehicle_category in LIGHT_DUTY_VEHICLE_CATEGORIES:
        return "light_duty"
    return "medium_heavy_duty"


def assign_model_year_groups(
    frame: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object]]],
    *,
    year_column: str = "modelYear",
    category_column: str = "vehicleCategory",
    output_column: str | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    target_column = output_column or year_column
    labels = pd.Series(pd.NA, index=result.index, dtype="object")
    model_year = pd.to_numeric(result[year_column], errors="raise").astype(int)
    vehicle_groups = result[category_column].map(vehicle_group)
    for current_group, groups in model_year_groups.items():
        group_mask = vehicle_groups == current_group
        if not group_mask.any():
            continue
        for group in groups:
            min_year = group.get("min_year")
            max_year = group.get("max_year")
            mask = group_mask.copy()
            if min_year is not None:
                mask &= model_year >= int(min_year)
            if max_year is not None:
                mask &= model_year <= int(max_year)
            labels.loc[mask] = model_year_group_label(group)
    if labels.isna().any():
        missing_rows = result.loc[labels.isna(), [category_column, year_column]].drop_duplicates()
        raise ValueError(
            "Some vehicleCategory/modelYear rows are not covered by the configured model_year_groups: "
            f"{missing_rows.to_dict(orient='records')[:20]}"
        )
    result[target_column] = labels
    return result
