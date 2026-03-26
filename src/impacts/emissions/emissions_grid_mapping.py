#!/usr/bin/env python
"""Map skimsEmissions (by BEAM link) to BEAM+OSM+GRID intersections."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLS = ["hour", "linkId", "vehicleTypeId", "process"]
OBS_COL = "observations"
PROPORTION_COL = "zone_edge_proportion"

DEFAULT_SKIMS_COLUMNS = {
    "hour": "hour",
    "link_id": "linkId",
    "vehicle_type": "vehicleTypeId",
    "process": "process",
    "emissions": "emissions",
    "observations": "observations",
    "iterations": "iterations",
    "travel_time": "travelTimeInSecond",
    "parking_duration": "parkingDurationInSecond",
}

DEFAULT_MAPPING_COLUMNS = {
    "link_id": "edge_linkId",
    "grid_id": "GRID",
    "proportion": "zone_edge_proportion",
}
DEFAULT_COUNTY_JOIN_COLUMNS = ["zone_GEOID", "zone_NAME", "zone_COUNTYFP"]
DEFAULT_UPSTREAM_GRID_JOIN_COLUMNS = ["zone_isrm", "GRID", "cell_id"]
DEFAULT_COUNTY_CORRECTION_COLUMNS = {
    "county_fips": "countyfp",
    "vmt_factor": "vmt_factor",
    "trips_factor": "trips_factor",
}
DEFAULT_PREPARED_GROUP_COLS = ["linkId", "vehicleTypeId", "process"]
DEFAULT_PREPARED_POLLUTANTS = ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BCh"]
DEFAULT_ANNUALIZATION_DAYS = 330.0
DEFAULT_BEAM_SCALE_KEYS = [
    "agentSampleSizeAsFractionOfPopulation",
    "simulationNameSampleSize",
    "sampleSizeAsFractionOfPopulation",
]


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _resolve_column_config(config: Optional[Dict[str, str]], defaults: Dict[str, str]) -> Dict[str, str]:
    resolved = defaults.copy()
    if config:
        resolved.update({k: v for k, v in config.items() if v})
    return resolved


def _normalize_county_fips(series: pd.Series) -> pd.Series:
    return series.astype(str).str.extract(r"(\d+)")[0].fillna("").str.zfill(3)


def _normalize_skims_columns(df: pd.DataFrame, skims_columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    columns = _resolve_column_config(skims_columns, DEFAULT_SKIMS_COLUMNS)
    return df.rename(
        columns={
            columns["travel_time"]: "travelTimeInSecond",
            columns["parking_duration"]: "parkingDurationInSecond",
            columns["hour"]: "hour",
            columns["link_id"]: "linkId",
            columns["vehicle_type"]: "vehicleTypeId",
            columns["observations"]: "observations",
            columns["iterations"]: "iterations",
            "emissionsProcess": "process",
            "pollutants": "emissions",
        }
    )


def read_skims_emissions(path: str, skims_columns: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Read skims emissions from parquet, csv.gz, or csv."""
    p = path.lower()
    if p.endswith(".parquet"):
        return _normalize_skims_columns(pd.read_parquet(path), skims_columns=skims_columns)
    if p.endswith(".csv.gz"):
        return _normalize_skims_columns(pd.read_csv(path, compression="gzip"), skims_columns=skims_columns)
    if p.endswith(".csv"):
        return _normalize_skims_columns(pd.read_csv(path), skims_columns=skims_columns)
    raise ValueError(f"Unsupported skims format: {path}. Use .csv, .csv.gz, or .parquet")


def read_mapping(path: str) -> pd.DataFrame:
    """Read BEAM+OSM+GRID mapping."""
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv") or p.endswith(".csv.gz"):
        compression = "gzip" if p.endswith(".csv.gz") else None
        return pd.read_csv(path, compression=compression)
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        return pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    except Exception as exc:
        raise ValueError(f"Unsupported mapping format: {path}") from exc


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}. Use .parquet, .csv.gz, or .csv")


def parse_emissions_string(emissions: str) -> Dict[str, float]:
    if emissions is None or (isinstance(emissions, float) and pd.isna(emissions)):
        return {}
    txt = str(emissions).strip()
    if not txt:
        return {}

    out: Dict[str, float] = {}
    for part in txt.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            continue
    return out


def expand_emissions_columns(df: pd.DataFrame, emissions_col: str = "emissions") -> pd.DataFrame:
    parsed = df[emissions_col].apply(parse_emissions_string)
    pollutants = sorted({k for d in parsed for k in d.keys()})
    for pol in pollutants:
        df[f"em_{pol}"] = parsed.apply(lambda d, p=pol: float(d.get(p, 0.0)))
    return df


def _totals_pollutant_columns(
    df: pd.DataFrame,
    required_pollutants: List[str],
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for pollutant in required_pollutants:
        for candidate in (
            pollutant,
            f"em_{pollutant}",
            f"tons_per_year_{pollutant}",
        ):
            if candidate in df.columns:
                resolved[pollutant] = candidate
                break
    return resolved


def prepare_skims_for_grid_allocation(
    skims_path: str,
    output_path: str,
    *,
    skims_columns: Optional[Dict[str, str]] = None,
    group_cols: Optional[List[str]] = None,
    required_pollutants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Prepare staged skims into a deterministic pre-allocation table.

    For compact skimsEmissions input, this expands the emissions string,
    multiplies pollutant values by observations, and aggregates by the
    configured grouping. For skimsEmissionsTotals input, this preserves
    totals and only downsamples to the configured pollutant subset.

    The resulting table is the grouped intermediate (C1): pollutant totals
    by linkId, vehicleTypeId, and process. Annualization to tons/year
    happens in a separate step.
    """
    skims_column_config = _resolve_column_config(skims_columns, DEFAULT_SKIMS_COLUMNS)
    df = read_skims_emissions(skims_path, skims_columns=skims_column_config)
    prepared_group_cols = group_cols or DEFAULT_PREPARED_GROUP_COLS
    missing_group_cols = [col for col in prepared_group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Prepared skims missing required grouping columns: {missing_group_cols}")

    pollutants = required_pollutants or DEFAULT_PREPARED_POLLUTANTS
    if skims_column_config["emissions"] in df.columns:
        df = expand_emissions_columns(df, emissions_col=skims_column_config["emissions"])
        observations_col = "observations"
        if observations_col not in df.columns:
            raise ValueError("Prepared skims require an observations column.")

        source_pollutant_cols = [f"em_{pollutant}" for pollutant in pollutants]
        for col in source_pollutant_cols:
            if col not in df.columns:
                df[col] = 0.0

        prepared = df[prepared_group_cols + [observations_col] + source_pollutant_cols].copy()
        prepared[observations_col] = pd.to_numeric(prepared[observations_col], errors="coerce").fillna(0.0)

        rename_map = {f"em_{pollutant}": pollutant for pollutant in pollutants}
        prepared = prepared.rename(columns=rename_map)
        pollutant_cols = list(rename_map.values())

        for col in pollutant_cols:
            prepared[col] = (
                pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)
                * prepared[observations_col]
            )

        aggregated = (
            prepared.groupby(prepared_group_cols, dropna=False)[pollutant_cols]
            .sum()
            .reset_index()
        )
    else:
        totals_cols = _totals_pollutant_columns(df, pollutants)
        prepared = df[prepared_group_cols + list(totals_cols.values())].copy()
        rename_map = {source: pollutant for pollutant, source in totals_cols.items()}
        prepared = prepared.rename(columns=rename_map)
        pollutant_cols = list(rename_map.values())
        for pollutant in pollutants:
            if pollutant not in prepared.columns:
                prepared[pollutant] = 0.0
        pollutant_cols = list(pollutants)
        for col in pollutant_cols:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce").fillna(0.0)
        aggregated = (
            prepared.groupby(prepared_group_cols, dropna=False)[pollutant_cols]
            .sum()
            .reset_index()
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        aggregated.to_parquet(out, index=False)
    elif out.name.lower().endswith(".csv.gz"):
        aggregated.to_csv(out, index=False, compression="gzip")
    else:
        raise ValueError("Prepared skims output must be .parquet or .csv.gz")
    return aggregated


def annualize_prepared_skims_for_grid_allocation(
    prepared_skims_path: str,
    output_path: str,
    *,
    group_cols: Optional[List[str]] = None,
    required_pollutants: Optional[List[str]] = None,
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS,
) -> pd.DataFrame:
    """Convert grouped pollutant totals (C1) into annualized tons/year (C2)."""
    if annualization_days <= 0:
        raise ValueError(f"Annualization days must be positive, got {annualization_days}")

    prepared = _read_table(prepared_skims_path)
    prepared_group_cols = group_cols or DEFAULT_PREPARED_GROUP_COLS
    pollutants = required_pollutants or DEFAULT_PREPARED_POLLUTANTS
    missing_group_cols = [col for col in prepared_group_cols if col not in prepared.columns]
    if missing_group_cols:
        raise ValueError(f"Annualized skims missing required grouping columns: {missing_group_cols}")

    out = prepared[prepared_group_cols].copy()
    for pollutant in pollutants:
        source_col = _first_existing(
            prepared,
            [
                pollutant,
                f"em_{pollutant}",
                f"tons_per_year_{pollutant}",
            ],
        )
        values = (
            pd.to_numeric(prepared[source_col], errors="coerce").fillna(0.0)
            if source_col is not None
            else pd.Series(np.zeros(len(prepared), dtype=float), index=prepared.index)
        )
        out[f"tons_per_year_{pollutant}"] = values * annualization_days / 1_000_000.0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        out.to_parquet(output, index=False)
    elif output.name.lower().endswith(".csv.gz"):
        out.to_csv(output, index=False, compression="gzip")
    else:
        raise ValueError("Annualized skims output must be .parquet or .csv.gz")
    return out


def _weighted_avg(g: pd.DataFrame, value_col: str, weight_col: str) -> float:
    w = g[weight_col].fillna(0.0).astype(float)
    v = g[value_col].fillna(0.0).astype(float)
    denom = w.sum()
    if denom <= 0:
        return float(v.mean()) if len(v) else 0.0
    return float((v * w).sum() / denom)


def aggregate_skims_weighted(
    skims_df: pd.DataFrame,
    group_cols: Optional[List[str]] = None,
    obs_col: str = OBS_COL,
) -> pd.DataFrame:
    """Observations-weighted aggregation of pollutants and time metrics."""
    df = skims_df.copy()
    if obs_col not in df.columns:
        raise ValueError(f"Missing required column: {obs_col}")

    group_cols = group_cols or [c for c in DEFAULT_GROUP_COLS if c in df.columns]
    if "linkId" in df.columns and "linkId" not in group_cols:
        group_cols = ["linkId"] + group_cols
    if not group_cols:
        raise ValueError("No group columns found. linkId is required in skims.")

    pollutant_cols = [c for c in df.columns if c.startswith("em_")]
    avg_cols = [c for c in ["travelTimeInSecond", "parkingDurationInSecond"] if c in df.columns]
    weighted_cols = pollutant_cols + avg_cols

    grouped = df.groupby(group_cols, dropna=False)
    out = grouped[obs_col].sum().reset_index().rename(columns={obs_col: "observations_sum"})
    for col in weighted_cols:
        out[col] = grouped.apply(
            lambda g, c=col: _weighted_avg(g, c, obs_col),
            include_groups=False,
        ).values
    if "iterations" in df.columns:
        out["iterations_max"] = grouped["iterations"].max().values
    return out


def _resolve_mapping_link_col(mapping_df: pd.DataFrame, mapping_columns: Optional[Dict[str, str]] = None) -> str:
    columns = _resolve_column_config(mapping_columns, DEFAULT_MAPPING_COLUMNS)
    candidates = [columns["link_id"], "edge_linkId", "linkId", "edge_link_id"]
    col = _first_existing(mapping_df, candidates)
    if col is None:
        raise ValueError("Mapping link column not found (expected edge_linkId/linkId/edge_link_id).")
    return col


def _resolve_mapping_grid_col(mapping_df: pd.DataFrame, mapping_columns: Optional[Dict[str, str]] = None) -> Optional[str]:
    columns = _resolve_column_config(mapping_columns, DEFAULT_MAPPING_COLUMNS)
    return _first_existing(
        mapping_df,
        [
            columns["grid_id"],
            "GRID",
            "grid",
            "zone",
            "cell_id",
            "Location",
            "zone_isrm",
        ],
    )


def _resolve_join_columns(
    weighted_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    mapping_columns: Optional[Dict[str, str]] = None,
) -> tuple[list[str], list[str]]:
    map_link_col = _resolve_mapping_link_col(mapping_df, mapping_columns=mapping_columns)
    left_on = ["linkId"]
    right_on = [map_link_col]

    for county_col in DEFAULT_COUNTY_JOIN_COLUMNS:
        if county_col not in weighted_df.columns:
            continue
        mapping_col = _first_existing(mapping_df, [f"edge_{county_col}", county_col])
        if mapping_col is None:
            continue
        left_on.append(county_col)
        right_on.append(mapping_col)

    for grid_col in DEFAULT_UPSTREAM_GRID_JOIN_COLUMNS:
        if grid_col not in weighted_df.columns:
            continue
        mapping_col = _first_existing(mapping_df, [f"edge_{grid_col}", grid_col])
        if mapping_col is None:
            continue
        left_on.append(grid_col)
        right_on.append(mapping_col)

    return left_on, right_on


def distribute_to_intersection(
    weighted_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    proportion_col: str = PROPORTION_COL,
    mapping_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Allocate weighted skims values to intersection rows via `proportion`."""
    weighted_df = weighted_df.copy()
    mapping_df = mapping_df.copy()
    latest_intersection_cols = [
        proportion_col,
        "edge_link_length_m",
        "zone_link_length_m",
    ]
    stale_merge_cols = [
        col
        for col in weighted_df.columns
        if col.endswith("_map") or col.endswith("_skims")
    ]
    if stale_merge_cols:
        weighted_df = weighted_df.drop(columns=stale_merge_cols)
    weighted_stale_intersection_cols = [
        col for col in latest_intersection_cols if col in weighted_df.columns
    ]
    if weighted_stale_intersection_cols:
        weighted_df = weighted_df.drop(columns=weighted_stale_intersection_cols)
    stale_mapping_cols = [
        col
        for col in mapping_df.columns
        if col.endswith("_map") or col.endswith("_skims")
    ]
    if stale_mapping_cols:
        mapping_df = mapping_df.drop(columns=stale_mapping_cols)

    if "linkId" not in weighted_df.columns:
        raise ValueError("Weighted skims dataframe must include linkId.")
    if proportion_col not in mapping_df.columns:
        raise ValueError(f"Mapping missing required column: {proportion_col}")

    map_grid_col = _resolve_mapping_grid_col(mapping_df, mapping_columns=mapping_columns)
    mapping = mapping_df.copy()
    # Keep voided rows (null proportion) — land cells with no road intersections
    # will produce null allocated emissions after the right join, allowing them
    # to be rendered separately (e.g. greyed out) in downstream visualizations.
    mapping[proportion_col] = pd.to_numeric(mapping[proportion_col], errors="coerce")
    left_on, right_on = _resolve_join_columns(
        weighted_df,
        mapping,
        mapping_columns=mapping_columns,
    )

    merged = weighted_df.merge(
        mapping,
        how="right",
        left_on=left_on,
        right_on=right_on,
        suffixes=("_skims", "_map"),
    )
    if proportion_col not in merged.columns:
        raise ValueError(f"Merged mapping is missing required proportion column: {proportion_col}")

    observations_source_col = "observations_sum" if "observations_sum" in merged.columns else "observations" if "observations" in merged.columns else None
    if observations_source_col is not None:
        merged["observations_allocated"] = merged[observations_source_col] * merged[proportion_col]

    # Pollutants: weighted by observations first (in weighted_df),
    # then allocated by intersection proportion.
    pollutant_cols = [
        c for c in merged.columns if c.startswith("em_") or c.startswith("tons_per_year_")
    ]
    for col in pollutant_cols:
        merged[f"{col}_allocated"] = merged[col] * merged[proportion_col]
        if col.startswith("tons_per_year_"):
            continue
        if observations_source_col == "observations_sum":
            merged[f"{col}_obs_weighted_allocated"] = merged[col] * merged[proportion_col]
        elif observations_source_col == "observations":
            merged[f"{col}_obs_weighted_allocated"] = (
                merged[col] * merged["observations_sum"] * merged[proportion_col]
            )

    # Time metrics remain observations-weighted averages and are copied for each segment.
    for col in ["travelTimeInSecond", "parkingDurationInSecond"]:
        if col in merged.columns:
            merged[f"{col}_weighted_avg"] = merged[col]

    if map_grid_col and map_grid_col in merged.columns:
        grid_values = pd.to_numeric(merged[map_grid_col], errors="coerce")
        merged["cell_id"] = grid_values.astype("Int64")
        merged["GRID"] = merged["cell_id"]

    return merged


def aggregate_allocated_intersection_rows(
    allocated_df: pd.DataFrame,
    *,
    group_cols: List[str],
) -> pd.DataFrame:
    """Collapse repeated split rows after allocation by summing additive fields."""
    available_group_cols = [col for col in group_cols if col in allocated_df.columns]
    if not available_group_cols:
        return allocated_df.copy()

    additive_cols = [
        col
        for col in allocated_df.columns
        if col.endswith("_allocated")
        or col in {
            "zone_edge_proportion",
            "zone_link_length_m",
            "edge_length_in_cell_m",
            "proportional_length_m",
            "beam_length_in_cell",
        }
    ]
    if not additive_cols:
        return allocated_df[available_group_cols].drop_duplicates().reset_index(drop=True)

    aggregated = (
        allocated_df.groupby(available_group_cols, dropna=False)[additive_cols]
        .sum()
        .reset_index()
    )
    return aggregated


def apply_activity_corrections(
    allocated_df: pd.DataFrame,
    activity_corrections_path: str,
    *,
    correction_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Apply county correction factors to annualized allocated rows.

    Matches county rows on zone_COUNTYFP and adjusts annualized pollutant
    totals by process family:
    - VMT-like: RUNEX, PMBW, PMTW, PRDUST, RUNLOSS
    - trip-like: HOTSOAK, DIURN, STREX
    - neutral: IDLEX and any unknown process
    Rows without a match or factor default to neutral factor 1.0.
    """
    columns = _resolve_column_config(correction_columns, DEFAULT_COUNTY_CORRECTION_COLUMNS)
    corrections = _read_table(activity_corrections_path)
    county_source_col = _first_existing(corrections, [columns["county_fips"]])
    if county_source_col is None:
        raise ValueError(
            f"County correction file must include the configured county FIPS column: {columns['county_fips']}."
        )
    if "zone_COUNTYFP" not in allocated_df.columns:
        raise ValueError("Allocated county dataframe must include zone_COUNTYFP for county correction matching.")

    vmt_col = _first_existing(corrections, [columns["vmt_factor"]])
    trips_col = _first_existing(corrections, [columns["trips_factor"]])

    merged = allocated_df.copy()
    factors = corrections.copy()
    factors["_county_fips_norm"] = _normalize_county_fips(factors[county_source_col])
    if vmt_col is not None:
        factors["_corr_vmt"] = pd.to_numeric(factors[vmt_col], errors="coerce").fillna(1.0)
    else:
        factors["_corr_vmt"] = 1.0
    if trips_col is not None:
        factors["_corr_trips"] = pd.to_numeric(factors[trips_col], errors="coerce").fillna(1.0)
    else:
        factors["_corr_trips"] = 1.0

    merged["_county_fips_norm"] = _normalize_county_fips(merged["zone_COUNTYFP"])
    merged = merged.merge(
        factors[["_county_fips_norm", "_corr_vmt", "_corr_trips"]].drop_duplicates(),
        how="left",
        on="_county_fips_norm",
    )
    merged["_corr_vmt"] = merged["_corr_vmt"].fillna(1.0).replace(0.0, 1.0)
    merged["_corr_trips"] = merged["_corr_trips"].fillna(1.0).replace(0.0, 1.0)

    vmt_processes = {"RUNEX", "PMBW", "PMTW", "PRDUST", "RUNLOSS"}
    trip_processes = {"HOTSOAK", "DIURN", "STREX"}
    process_values = merged.get("process", pd.Series("", index=merged.index)).astype(str).str.upper()
    correction_factor = pd.Series(np.ones(len(merged), dtype=float), index=merged.index)
    correction_factor = correction_factor.where(~process_values.isin(vmt_processes), merged["_corr_vmt"])
    correction_factor = correction_factor.where(~process_values.isin(trip_processes), merged["_corr_trips"])

    for col in [c for c in merged.columns if c.startswith("tons_per_year_") and c.endswith("_allocated")]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0) / correction_factor

    return merged.drop(columns=["_county_fips_norm", "_corr_vmt", "_corr_trips"])


def plot_county_pm25_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    *,
    pm25_col: str = "tons_per_year_PM2_5_allocated",
    county_col: str = "zone_COUNTYFP",
    county_name_col: str = "zone_NAME",
    title: str = "PM2.5 Emissions by County: Before vs After Correction",
    output_path: Optional[str] = None,
) -> None:
    """Generate a grouped bar plot of total PM2.5 per county before and after activity corrections.

    Parameters
    ----------
    before_df:
        Allocated emissions DataFrame before applying corrections.
    after_df:
        Allocated emissions DataFrame after applying corrections.
    pm25_col:
        Column name for allocated PM2.5 in tons/year.
    county_col:
        Column name for county FIPS code.
    county_name_col:
        Column name for county display name (optional, falls back to FIPS).
    title:
        Plot title.
    output_path:
        If provided, save the figure to this path instead of displaying it.
    """
    import matplotlib.pyplot as plt

    def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
        if pm25_col not in df.columns:
            raise ValueError(f"Column '{pm25_col}' not found in DataFrame.")
        if county_col not in df.columns:
            raise ValueError(f"Column '{county_col}' not found in DataFrame.")
        label_col = county_name_col if county_name_col in df.columns else county_col
        agg = (
            df.groupby(county_col, dropna=False)
            .agg(
                pm25=(pm25_col, "sum"),
                label=(label_col, "first"),
            )
            .reset_index()
        )
        return agg.sort_values(county_col)

    before_agg = _aggregate(before_df)
    after_agg = _aggregate(after_df)

    all_counties = sorted(set(before_agg[county_col]).union(after_agg[county_col]))
    before_map = before_agg.set_index(county_col)
    after_map = after_agg.set_index(county_col)

    labels = [
        before_map.loc[c, "label"] if c in before_map.index else after_map.loc[c, "label"]
        for c in all_counties
    ]
    before_vals = [before_map.loc[c, "pm25"] if c in before_map.index else 0.0 for c in all_counties]
    after_vals = [after_map.loc[c, "pm25"] if c in after_map.index else 0.0 for c in all_counties]

    x = np.arange(len(all_counties))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_counties) * 1.2), 6))
    ax.bar(x - width / 2, before_vals, width, label="Before correction", color="steelblue", alpha=0.85)
    ax.bar(x + width / 2, after_vals, width, label="After correction", color="darkorange", alpha=0.85)

    ax.set_xlabel("County")
    ax.set_ylabel("Total PM2.5 (tons/year)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.2f}"))
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def map_skims_emissions_to_intersection(
    skims_path: str,
    mapping_path: str,
    output_path: str,
    group_cols: Optional[List[str]] = None,
    skims_columns: Optional[Dict[str, str]] = None,
    mapping_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """End-to-end mapping helper."""
    skims_column_config = _resolve_column_config(skims_columns, DEFAULT_SKIMS_COLUMNS)
    mapping_column_config = _resolve_column_config(mapping_columns, DEFAULT_MAPPING_COLUMNS)

    skims = read_skims_emissions(skims_path, skims_columns=skims_column_config)
    prepared_group_cols = group_cols or DEFAULT_PREPARED_GROUP_COLS
    is_prepared = "emissions" not in skims.columns and any(
        col.startswith("em_") or col.startswith("tons_per_year_") for col in skims.columns
    )
    if is_prepared:
        weighted = skims.copy()
    else:
        skims = expand_emissions_columns(skims, emissions_col=skims_column_config["emissions"])
        weighted = aggregate_skims_weighted(skims, group_cols=prepared_group_cols, obs_col=OBS_COL)

    mapping = read_mapping(mapping_path)
    allocated = distribute_to_intersection(
        weighted,
        mapping,
        proportion_col=mapping_column_config["proportion"],
        mapping_columns=mapping_column_config,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_name = out.name.lower()
    if out.suffix.lower() == ".parquet":
        allocated.to_parquet(out, index=False)
    elif out_name.endswith(".csv.gz"):
        allocated.to_csv(out, index=False, compression="gzip")
    else:
        raise ValueError("Output must be .csv.gz or .parquet")
    return allocated
