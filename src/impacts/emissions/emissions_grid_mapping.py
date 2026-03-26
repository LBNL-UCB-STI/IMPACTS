#!/usr/bin/env python
"""Map skimsEmissions (by BEAM link) to BEAM+OSM+GRID intersections."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from impacts.defaults import DEFAULT_ANNUALIZATION_DAYS
from impacts.defaults import DEFAULT_CHUNK_SIZE
from impacts.defaults import DEFAULT_COUNTY_CORRECTION_COLUMNS
from impacts.defaults import DEFAULT_POLLUTANTS as DEFAULT_PREPARED_POLLUTANTS
from impacts.defaults import DEFAULT_SKIMS_COLUMNS
from impacts.defaults import GRAMS_PER_TON



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


def read_skims_emissions(
    path: str,
    pollutants: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Read skims emissions from parquet, csv.gz, or csv.

    Parameters
    ----------
    pollutants:
        If provided, only load these pollutant columns plus the required
        dimension columns (linkId, vehicleTypeId, process).
    """
    dim_cols = ["linkId", "vehicleTypeId", "process"]
    cols = dim_cols + [c for c in (pollutants or []) if c not in dim_cols]
    p = path.lower()
    if p.endswith(".parquet"):
        import pyarrow.parquet as pq
        available = set(pq.read_schema(path).names)
        cols = [c for c in cols if c in available] or None
        return pd.read_parquet(path, columns=cols)
    if p.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip", usecols=cols if pollutants else None)
    if p.endswith(".csv"):
        return pd.read_csv(path, usecols=cols if pollutants else None)
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
    df = read_skims_emissions(skims_path)
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
    missing_group_cols = [col for col in prepared_group_cols if col not in df.columns]
    if missing_group_cols:
        raise ValueError(f"Prepared skims missing required grouping columns: {missing_group_cols}")

    pollutants = required_pollutants or DEFAULT_PREPARED_POLLUTANTS
    if "emissions" in df.columns:
        df = expand_emissions_columns(df, emissions_col="emissions")
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
    prepared_group_cols = group_cols or ["linkId", "vehicleTypeId", "process"]
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


def _resolve_mapping_link_col(mapping_df: pd.DataFrame, mapping_columns: Optional[Dict[str, str]] = None) -> str:
    col = (mapping_columns or {}).get("link_id", "edge_linkId")
    if col not in mapping_df.columns:
        raise ValueError(f"Mapping link column '{col}' not found.")
    return col




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


_COUNTY_FIPS_CANDIDATES = [
    "county_zone_COUNTYFP", "county_COUNTYFP", "zone_COUNTYFP", "COUNTYFP", "countyfp",
]

_VMT_PROCESSES = {"RUNEX", "PMBW", "PMTW", "PRDUST", "RUNLOSS"}
_TRIP_PROCESSES = {"HOTSOAK", "DIURN", "STREX"}


def apply_county_corrections(
    allocated_df: pd.DataFrame,
    corrections_path: str,
    *,
    correction_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Multiply all ``*_allocated`` columns by a per-county, per-process factor.

    The factor applied per row depends on the ``process`` column:

    - VMT-like (RUNEX, PMBW, PMTW, PRDUST, RUNLOSS) → ``vmt_factor``
    - Trip-like (HOTSOAK, DIURN, STREX) → ``trips_factor``
    - All others (IDLEX, unknown) → 1.0 (neutral, no correction)

    Rows with no county match keep factor 1.0.  The correction is a
    **multiplication** applied to every ``*_allocated`` column.

    Parameters
    ----------
    corrections_path:
        CSV/parquet with one row per county: ``countyfp``, ``vmt_factor``,
        ``trips_factor``.
    correction_columns:
        Override default column name mapping.
    """
    columns = _resolve_column_config(correction_columns, DEFAULT_COUNTY_CORRECTION_COLUMNS)

    corrections = _read_table(corrections_path)
    fips_source_col = _first_existing(corrections, [columns["county_fips"]])
    if fips_source_col is None:
        raise ValueError(
            f"County corrections file must include county FIPS column: {columns['county_fips']}."
        )

    county_col = _first_existing(allocated_df, _COUNTY_FIPS_CANDIDATES)
    if county_col is None:
        raise ValueError(
            f"Allocated DataFrame must include a county FIPS column. "
            f"Expected one of: {_COUNTY_FIPS_CANDIDATES}."
        )

    vmt_col = _first_existing(corrections, [columns["vmt_factor"]])
    trips_col = _first_existing(corrections, [columns["trips_factor"]])

    factors = corrections.copy()
    factors["_fips_norm"] = _normalize_county_fips(factors[fips_source_col])
    factors["_corr_vmt"] = (
        pd.to_numeric(factors[vmt_col], errors="coerce").fillna(1.0).replace(0.0, 1.0)
        if vmt_col else 1.0
    )
    factors["_corr_trips"] = (
        pd.to_numeric(factors[trips_col], errors="coerce").fillna(1.0).replace(0.0, 1.0)
        if trips_col else 1.0
    )
    factors = factors[["_fips_norm", "_corr_vmt", "_corr_trips"]].drop_duplicates("_fips_norm")

    result = allocated_df.copy()
    result["_fips_norm"] = _normalize_county_fips(result[county_col])
    result = result.merge(factors, how="left", on="_fips_norm")
    result["_corr_vmt"] = result["_corr_vmt"].fillna(1.0)
    result["_corr_trips"] = result["_corr_trips"].fillna(1.0)

    # Select the right factor per row based on process
    process_upper = result.get("process", pd.Series("", index=result.index)).astype(str).str.upper()
    factor_arr = np.ones(len(result), dtype=np.float32)
    factor_arr = np.where(process_upper.isin(_VMT_PROCESSES), result["_corr_vmt"].to_numpy(dtype=np.float32), factor_arr)
    factor_arr = np.where(process_upper.isin(_TRIP_PROCESSES), result["_corr_trips"].to_numpy(dtype=np.float32), factor_arr)

    allocated_cols = [c for c in result.columns if c.endswith("_allocated")]
    for col in allocated_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) * factor_arr

    return result.drop(columns=["_fips_norm", "_corr_vmt", "_corr_trips"])


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


_SKIMS_DIMENSION_COLS = {
    "linkId", "vehicleTypeId", "process", "hour",
    "observations", "iterations", "travelTimeInSecond", "parkingDurationInSecond",
}

def annualize_skims(
    skims_df: pd.DataFrame,
    pollutants: List[str],
    annualization_days: float,
) -> pd.DataFrame:
    """Convert raw per-day gram emissions to annualized tons/year.

    For each pollutant, finds the source column (raw name, ``em_{p}``, or
    already ``tons_per_year_{p}``), applies
    ``tons_per_year_{p} = value × annualization_days / 1_000_000``
    and returns a DataFrame with only dimension columns + ``tons_per_year_*``.
    """
    if annualization_days <= 0:
        raise ValueError(f"annualization_days must be positive, got {annualization_days}")

    dim_cols = [c for c in skims_df.columns if c in _SKIMS_DIMENSION_COLS]
    out = skims_df[dim_cols].copy()
    factor = annualization_days / GRAMS_PER_TON

    for pollutant in pollutants:
        source_col = _first_existing(
            skims_df,
            [pollutant, f"em_{pollutant}", f"tons_per_year_{pollutant}"],
        )
        if source_col is None:
            out[f"tons_per_year_{pollutant}"] = 0.0
            continue
        values = pd.to_numeric(skims_df[source_col], errors="coerce").fillna(0.0)
        # Already annualized — don't scale again
        if source_col.startswith("tons_per_year_"):
            out[f"tons_per_year_{pollutant}"] = values
        else:
            out[f"tons_per_year_{pollutant}"] = values * factor

    return out

_INTERSECTION_METRIC_SUFFIXES = (
    "_length_m", "_surface_m2", "_link_length_m", "_piece_length_m",
    "_edge_length_m", "_edge_surface_m2", "_zone_surface_m2",
)


def _detect_prop_items(columns: List[str], proportion_labels: Optional[List[str]]) -> List[tuple]:
    if proportion_labels is not None:
        items = []
        for label in proportion_labels:
            candidates = [c for c in columns if c.startswith(f"{label}_") and "proportion" in c]
            if not candidates:
                raise ValueError(
                    f"No proportion column for label '{label}'. "
                    f"Expected a column like '{label}_zone_edge_proportion'."
                )
            items.append((label, candidates[0]))
        return items
    items = []
    for col in columns:
        if "_zone_edge_proportion" in col:
            items.append((col.split("_zone_edge_proportion")[0], col))
        elif "_zone_piece_proportion" in col:
            items.append((col.split("_zone_piece_proportion")[0], col))
    if not items:
        raise ValueError(
            "No labeled proportion columns found in intersection. "
            "Expected columns like aermod_zone_edge_proportion or inmap_zone_piece_proportion."
        )
    return items


def _select_intersection_cols(all_cols: List[str]) -> List[str]:
    """Drop geometry and heavy metric columns not needed for allocation."""
    return [
        c for c in all_cols
        if c != "geometry"
        and not any(c.endswith(s) for s in _INTERSECTION_METRIC_SUFFIXES)
    ]


def _read_intersection_selective(path: str) -> pd.DataFrame:
    """Load intersection parquet, reading only allocation-relevant columns."""
    p = path.lower()
    if not p.endswith(".parquet"):
        return read_mapping(path)
    try:
        import pyarrow.parquet as pq
        all_cols = pq.read_schema(path).names
        cols = _select_intersection_cols(all_cols)
        return pd.read_parquet(path, columns=cols)
    except Exception:
        return read_mapping(path)


def _allocate_chunk(
    inter_chunk: pd.DataFrame,
    skims_slim: pd.DataFrame,
    map_link_col: str,
    emission_cols: List[str],
    prop_items: List[tuple],
) -> pd.DataFrame:
    merged = inter_chunk.merge(
        skims_slim, how="left", left_on=map_link_col, right_on="linkId", sort=False
    )
    emission_matrix = np.column_stack([
        merged[col].to_numpy(dtype=np.float32, na_value=0.0) for col in emission_cols
    ])
    merged.drop(columns=emission_cols, inplace=True)
    new_cols: dict = {}
    for label, prop_col in prop_items:
        prop_arr = merged[prop_col].to_numpy(dtype=np.float32, na_value=0.0)
        allocated = emission_matrix * prop_arr[:, np.newaxis]
        for i, col in enumerate(emission_cols):
            new_cols[f"{col}_{label}_allocated"] = allocated[:, i]
    return pd.concat([merged, pd.DataFrame(new_cols, index=merged.index)], axis=1)


def allocate_emissions_to_labeled_intersection(
    skims_path,
    intersection_path: str,
    output_path: str,
    *,
    proportion_labels: Optional[List[str]] = None,
    mapping_columns: Optional[Dict[str, str]] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Allocate skims emissions to a labeled grid intersection.

    Skims are expected to already contain pre-calculated emission totals per
    (linkId, vehicleTypeId, process) — no observations weighting or expansion
    is performed.  All numeric columns that are not dimension/ID columns are
    treated as emission columns.

    Each emission column is scaled independently by each labeled proportion
    column, producing ``{pollutant}_{label}_allocated`` columns
    (e.g. ``CH4_aermod_allocated``, ``CH4_inmap_allocated``).  Proportions
    are never multiplied together.

    Processes linkIds in batches of ``chunk_size`` to bound peak memory, and
    writes output incrementally via ``ParquetWriter``.

    Returns
    -------
    int
        Total number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if isinstance(skims_path, pd.DataFrame):
        skims = skims_path
    else:
        skims = read_skims_emissions(skims_path)

    # Read only allocation-relevant columns from intersection (schema read is free)
    intersection = _read_intersection_selective(intersection_path)

    prop_items = _detect_prop_items(list(intersection.columns), proportion_labels)

    emission_cols = [
        c for c in skims.columns
        if c not in _SKIMS_DIMENSION_COLS and pd.api.types.is_numeric_dtype(skims[c])
    ]

    # Slim + downcast skims before chunked merges
    skims_keep = list({"linkId", "vehicleTypeId", "process"} & set(skims.columns)) + emission_cols
    skims_slim = skims[skims_keep].copy()
    for col in emission_cols:
        skims_slim[col] = (
            pd.to_numeric(skims_slim[col], errors="coerce").fillna(0.0).astype(np.float32)
        )
    skims_slim["linkId"] = skims_slim["linkId"].astype("category")

    # Downcast proportion columns in intersection to float32
    for _, pc in prop_items:
        intersection[pc] = (
            pd.to_numeric(intersection[pc], errors="coerce").fillna(0.0).astype(np.float32)
        )
    map_link_col = _resolve_mapping_link_col(intersection, mapping_columns=mapping_columns)
    intersection[map_link_col] = intersection[map_link_col].astype("category")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Pre-index both DataFrames by linkId once — replaces repeated O(n) isin
    # scans inside the loop with O(1) dict lookups per chunk.
    inter_by_link = {lid: grp for lid, grp in intersection.groupby(map_link_col, sort=False)}
    skims_by_link = {lid: grp for lid, grp in skims_slim.groupby("linkId", sort=False)}
    unique_link_ids = list(skims_by_link.keys())
    writer = None
    total_rows = 0

    for start in range(0, len(unique_link_ids), chunk_size):
        batch_ids = unique_link_ids[start : start + chunk_size]
        inter_parts = [inter_by_link[lid] for lid in batch_ids if lid in inter_by_link]
        if not inter_parts:
            continue
        inter_chunk = pd.concat(inter_parts, ignore_index=True)
        skims_chunk = pd.concat(
            [skims_by_link[lid] for lid in batch_ids if lid in skims_by_link],
            ignore_index=True,
        )
        chunk_result = _allocate_chunk(
            inter_chunk, skims_chunk, map_link_col, emission_cols, prop_items
        )
        total_rows += len(chunk_result)
        if out.suffix.lower() == ".parquet":
            table = pa.Table.from_pandas(chunk_result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="snappy")
            writer.write_table(table)
        else:
            chunk_result.to_csv(
                out, mode="a", index=False,
                header=not out.exists() or total_rows == len(chunk_result),
                compression="gzip" if out.name.endswith(".csv.gz") else None,
            )

    if writer is not None:
        writer.close()

    return total_rows


