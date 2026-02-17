#!/usr/bin/env python
"""Map skimsEmissions (by BEAM link) to BEAM+OSM+GRID intersections."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


DEFAULT_GROUP_COLS = ["hour", "linkId", "vehicleTypeId", "process"]
OBS_COL = "observations"
PROPORTION_COL = "proportion"


def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _normalize_skims_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "emissionsProcess": "process",
            "pollutants": "emissions",
            "travelTime": "travelTimeInSecond",
            "parkingDuration": "parkingDurationInSecond",
        }
    )


def read_skims_emissions(path: str) -> pd.DataFrame:
    """Read skims emissions from parquet or csv.gz."""
    p = path.lower()
    if p.endswith(".parquet"):
        return _normalize_skims_columns(pd.read_parquet(path))
    if p.endswith(".csv.gz"):
        return _normalize_skims_columns(pd.read_csv(path, compression="gzip"))
    raise ValueError(f"Unsupported skims format: {path}. Use .csv.gz or .parquet")


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
        out[col] = grouped.apply(lambda g, c=col: _weighted_avg(g, c, obs_col)).values
    if "iterations" in df.columns:
        out["iterations_max"] = grouped["iterations"].max().values
    return out


def _resolve_mapping_link_col(mapping_df: pd.DataFrame) -> str:
    col = _first_existing(mapping_df, ["edge_linkId", "linkId", "edge_link_id"])
    if col is None:
        raise ValueError("Mapping link column not found (expected edge_linkId/linkId/edge_link_id).")
    return col


def distribute_to_intersection(
    weighted_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    proportion_col: str = PROPORTION_COL,
) -> pd.DataFrame:
    """Allocate weighted skims values to intersection rows via `proportion`."""
    if "linkId" not in weighted_df.columns:
        raise ValueError("Weighted skims dataframe must include linkId.")
    if proportion_col not in mapping_df.columns:
        raise ValueError(f"Mapping missing required column: {proportion_col}")

    map_link_col = _resolve_mapping_link_col(mapping_df)
    mapping = mapping_df.copy()
    mapping[proportion_col] = mapping[proportion_col].fillna(0.0).astype(float)

    merged = weighted_df.merge(
        mapping,
        how="inner",
        left_on="linkId",
        right_on=map_link_col,
        suffixes=("_skims", "_map"),
    )

    merged["observations_allocated"] = merged["observations_sum"] * merged[proportion_col]

    # Pollutants: weighted by observations first (in weighted_df),
    # then allocated by intersection proportion.
    pollutant_cols = [c for c in merged.columns if c.startswith("em_")]
    for col in pollutant_cols:
        merged[f"{col}_allocated"] = merged[col] * merged[proportion_col]
        merged[f"{col}_obs_weighted_allocated"] = (
            merged[col] * merged["observations_sum"] * merged[proportion_col]
        )

    # Time metrics remain observations-weighted averages and are copied for each segment.
    for col in ["travelTimeInSecond", "parkingDurationInSecond"]:
        if col in merged.columns:
            merged[f"{col}_weighted_avg"] = merged[col]

    return merged


def map_skims_emissions_to_intersection(
    skims_path: str,
    mapping_path: str,
    output_path: str,
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """End-to-end mapping helper."""
    skims = read_skims_emissions(skims_path)
    skims = expand_emissions_columns(skims, emissions_col="emissions")
    weighted = aggregate_skims_weighted(skims, group_cols=group_cols, obs_col=OBS_COL)

    mapping = read_mapping(mapping_path)
    allocated = distribute_to_intersection(weighted, mapping, proportion_col=PROPORTION_COL)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        allocated.to_parquet(out, index=False)
    elif out.suffix.lower() == ".csv.gz":
        allocated.to_csv(out, index=False, compression="gzip")
    else:
        raise ValueError("Output must be .csv.gz or .parquet")
    return allocated

