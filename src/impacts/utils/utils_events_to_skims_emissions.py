#!/usr/bin/env python
"""Build a skimsEmissions-like table from BEAM events files."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from impacts.defaults import DEFAULT_BEAM_NETWORK_COLUMNS
from impacts.defaults import DEFAULT_EVENTS_COLUMNS

REQUIRED_EVENT_COLS = [
    "type",
    "vehicle",
    "vehicleType",
    "departureTime",
    "links",
    "linkTravelTime",
    "length",
]

SKIMS_COLS = [
    "hour",
    "linkId",
    "vehicleTypeId",
    "process",
    "emissions",
    "travelTimeInSecond",
    "parkingDurationInSecond",
    "observations",
    "iterations",
]


def _parse_list_field(raw: object) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if str(x).strip()]


def _parse_float_list(raw: object) -> List[float]:
    out: List[float] = []
    for x in _parse_list_field(raw):
        try:
            out.append(float(x))
        except ValueError:
            out.append(0.0)
    return out


def _pollutant_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("em_")]



def _read_any_table(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if p.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported rates file: {path}")


def _normalize_rates_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "emission_rate": "rate",
            "emissions_rate": "rate",
            "vehicleType": "vehicleTypeId",
            "vehicle_type": "vehicleTypeId",
            "emissionsProcess": "process",
        }
    )


def read_rates_directory(
    rates_dir: str,
    default_rate_basis: str = "per_event",
) -> pd.DataFrame:
    """Read all rates tables from directory (csv/csv.gz/parquet)."""
    root = Path(rates_dir)
    if not root.exists():
        raise FileNotFoundError(f"Rates directory not found: {rates_dir}")

    files = sorted(
        [
            p
            for p in root.iterdir()
            if p.is_file()
            and (str(p).lower().endswith(".csv") or str(p).lower().endswith(".csv.gz") or str(p).lower().endswith(".parquet"))
        ]
    )
    if not files:
        return pd.DataFrame(columns=["vehicleTypeId", "process", "pollutant", "rate", "rate_basis"])

    frames: List[pd.DataFrame] = []
    for f in files:
        df = _normalize_rates_columns(_read_any_table(f))
        if "vehicleTypeId" not in df.columns:
            df["vehicleTypeId"] = f.stem.replace(".csv", "").replace("_rates", "")
        if "rate_basis" not in df.columns:
            df["rate_basis"] = default_rate_basis
        frames.append(df)

    rates = pd.concat(frames, ignore_index=True)
    required = {"process", "pollutant", "rate", "rate_basis"}
    missing = required.difference(rates.columns)
    if missing:
        raise ValueError(f"Rates files missing required columns: {sorted(missing)}")

    rates["process"] = rates["process"].astype(str)
    rates["pollutant"] = rates["pollutant"].astype(str)
    rates["rate"] = pd.to_numeric(rates["rate"], errors="coerce").fillna(0.0)
    rates["rate_basis"] = rates["rate_basis"].astype(str).str.lower()
    return rates


def _weighted_avg(group: pd.DataFrame, col: str, wcol: str = "observations") -> float:
    w = group[wcol].fillna(0.0).astype(float)
    v = group[col].fillna(0.0).astype(float)
    denom = w.sum()
    if denom <= 0:
        return float(v.mean()) if len(v) else 0.0
    return float((v * w).sum() / denom)


def _serialize_emissions(row: pd.Series, pollutant_cols: Iterable[str]) -> str:
    parts: List[str] = []
    for col in pollutant_cols:
        value = float(row.get(col, 0.0) or 0.0)
        if value > 0:
            parts.append(f"{col.removeprefix('em_')}:{value}")
    return ";".join(parts)


def read_events(events_path: str) -> pd.DataFrame:
    p = events_path.lower()
    compression = "gzip" if p.endswith(".gz") else None
    return pd.read_csv(events_path, usecols=DEFAULT_EVENTS_COLUMNS, compression=compression)


def _build_path_traversal_rows(events: pd.DataFrame, link_lengths_m: Optional[Dict[int, float]] = None) -> pd.DataFrame:
    pt = events[events["type"] == "PathTraversal"].copy()
    pt["links_list"] = pt["links"].apply(_parse_list_field)
    pt["times_list"] = pt["linkTravelTime"].apply(_parse_float_list)
    pt = pt[pt["links_list"].map(len) > 0].copy()
    pt = pt[pt["links_list"].map(len) == pt["times_list"].map(len)].copy()

    exploded_rows: List[Dict[str, object]] = []
    for _, row in pt.iterrows():
        links = [int(x) for x in row["links_list"]]
        times = list(row["times_list"])
        total_tt = sum(times)
        total_len_m = float(row.get("length", 0.0) or 0.0)
        for link_id, t_sec in zip(links, times):
            if link_lengths_m and link_id in link_lengths_m:
                length_m = float(link_lengths_m[link_id])
            else:
                length_m = (total_len_m * (t_sec / total_tt)) if total_tt > 0 else 0.0
            dep = float(row["departureTime"])
            exploded_rows.append(
                {
                    "hour": int(dep // 3600),
                    "linkId": int(link_id),
                    "vehicleTypeId": str(row["vehicleType"]),
                    "vehicle": str(row["vehicle"]),
                    "process": "RUNEX",
                    "travelTimeInSecond": float(t_sec),
                    "parkingDurationInSecond": 0.0,
                    "distanceMiles": float(length_m) / 1609.34,
                    "observations": 1,
                }
            )
            exploded_rows.append(
                {
                    "hour": int(dep // 3600),
                    "linkId": int(link_id),
                    "vehicleTypeId": str(row["vehicleType"]),
                    "vehicle": str(row["vehicle"]),
                    "process": "RUNLOSS",
                    "travelTimeInSecond": float(t_sec),
                    "parkingDurationInSecond": 0.0,
                    "distanceMiles": 0.0,
                    "observations": 1,
                }
            )
            exploded_rows.append(
                {
                    "hour": int(dep // 3600),
                    "linkId": int(link_id),
                    "vehicleTypeId": str(row["vehicleType"]),
                    "vehicle": str(row["vehicle"]),
                    "process": "PMBW",
                    "travelTimeInSecond": float(t_sec),
                    "parkingDurationInSecond": 0.0,
                    "distanceMiles": float(length_m) / 1609.34,
                    "observations": 1,
                }
            )
            exploded_rows.append(
                {
                    "hour": int(dep // 3600),
                    "linkId": int(link_id),
                    "vehicleTypeId": str(row["vehicleType"]),
                    "vehicle": str(row["vehicle"]),
                    "process": "PMTW",
                    "travelTimeInSecond": float(t_sec),
                    "parkingDurationInSecond": 0.0,
                    "distanceMiles": float(length_m) / 1609.34,
                    "observations": 1,
                }
            )

    return pd.DataFrame(exploded_rows)


def _build_parking_rows(events: pd.DataFrame) -> pd.DataFrame:
    pt = events[events["type"] == "PathTraversal"].copy()
    pt["links_list"] = pt["links"].apply(_parse_list_field)
    pt["times_list"] = pt["linkTravelTime"].apply(_parse_float_list)
    pt = pt[pt["links_list"].map(len) > 0].copy()
    pt = pt[pt["links_list"].map(len) == pt["times_list"].map(len)].copy()
    pt["tripDurationSec"] = pt["times_list"].apply(sum)
    pt["tripEndSec"] = pt["departureTime"].astype(float) + pt["tripDurationSec"].astype(float)
    pt["firstLink"] = pt["links_list"].apply(lambda xs: int(xs[0]))
    pt["lastLink"] = pt["links_list"].apply(lambda xs: int(xs[-1]))
    pt = pt.sort_values(["vehicle", "departureTime"]).copy()
    pt["nextDepartureSec"] = pt.groupby("vehicle")["departureTime"].shift(-1).astype(float)
    pt["parkingDurationInSecond"] = (
        pt["nextDepartureSec"].fillna(pt["tripEndSec"]) - pt["tripEndSec"]
    ).clip(lower=0)

    out_rows: List[Dict[str, object]] = []
    for _, row in pt.iterrows():
        hour = int(float(row["tripEndSec"]) // 3600)
        base = {
            "hour": hour,
            "vehicleTypeId": str(row["vehicleType"]),
            "vehicle": str(row["vehicle"]),
            "travelTimeInSecond": 0.0,
            "parkingDurationInSecond": float(row["parkingDurationInSecond"]),
            "distanceMiles": 0.0,
            "observations": 1,
        }
        out_rows.append({**base, "linkId": int(row["lastLink"]), "process": "DIURN"})
        out_rows.append({**base, "linkId": int(row["lastLink"]), "process": "HOTSOAK"})
        out_rows.append({**base, "linkId": int(row["firstLink"]), "process": "STREX"})
    return pd.DataFrame(out_rows)


def _apply_rates(obs_df: pd.DataFrame, rates_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = obs_df.copy()
    if rates_df is None or rates_df.empty:
        return df

    rates = rates_df.copy()
    required = {"process", "pollutant", "rate", "rate_basis"}
    missing = required.difference(rates.columns)
    if missing:
        raise ValueError(f"rates_df missing required columns: {sorted(missing)}")

    join_cols = ["process"]
    if "vehicleTypeId" in rates.columns and "vehicleTypeId" in df.columns:
        join_cols.append("vehicleTypeId")

    merged = df.merge(rates, on=join_cols, how="left")

    def activity_amount(row: pd.Series) -> float:
        basis = str(row.get("rate_basis", "")).lower()
        if basis == "per_mile":
            return float(row.get("distanceMiles", 0.0) or 0.0)
        if basis == "per_hour":
            if str(row.get("process")) in ("DIURN",):
                return float(row.get("parkingDurationInSecond", 0.0) or 0.0) / 3600.0
            return float(row.get("travelTimeInSecond", 0.0) or 0.0) / 3600.0
        if basis == "per_event":
            return 1.0
        return 0.0

    merged["activity"] = merged.apply(activity_amount, axis=1)
    merged["emission_value"] = (
        merged["rate"].fillna(0.0).astype(float) * merged["activity"].fillna(0.0).astype(float)
    )

    pivot = (
        merged.pivot_table(
            index=[
                "hour",
                "linkId",
                "vehicleTypeId",
                "process",
                "travelTimeInSecond",
                "parkingDurationInSecond",
                "observations",
            ],
            columns="pollutant",
            values="emission_value",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    pivot.columns = [f"em_{c}" if c not in pivot.columns[:7] and not str(c).startswith("em_") else c for c in pivot.columns]
    return pivot


def _aggregate_skims_like(df: pd.DataFrame, iterations: int = 1) -> pd.DataFrame:
    group_cols = ["hour", "linkId", "vehicleTypeId", "process"]
    pollutant_cols = _pollutant_cols(df)
    grouped = df.groupby(group_cols, dropna=False)

    out = grouped["observations"].sum().reset_index().rename(columns={"observations": "observations"})
    for col in ["travelTimeInSecond", "parkingDurationInSecond"] + pollutant_cols:
        if col in df.columns:
            out[col] = grouped.apply(
                lambda g, c=col: _weighted_avg(g, c, "observations"),
                include_groups=False,
            ).values

    out["emissions"] = out.apply(lambda r: _serialize_emissions(r, pollutant_cols), axis=1)
    out["iterations"] = int(iterations)
    for c in SKIMS_COLS:
        if c not in out.columns:
            if c == "emissions":
                out[c] = ""
            elif c == "iterations":
                out[c] = int(iterations)
            else:
                out[c] = 0
    return out[SKIMS_COLS].sort_values(group_cols).reset_index(drop=True)


def build_skims_emissions_from_events(
    events_path: str,
    network_path: Optional[str] = None,
    rates_df: Optional[pd.DataFrame] = None,
    iterations: int = 1,
) -> pd.DataFrame:
    """Return a skimsEmissions-like structure from BEAM events.

    Notes:
    - Emits standard processes used in BEAM skims: RUNEX, RUNLOSS, PMTW,
      PMBW, DIURN, HOTSOAK, STREX.
    - If `rates_df` is None, emissions strings are empty and only activity
      metrics/observations are returned.
    """
    events = read_events(events_path)
    link_lengths_m: Optional[Dict[int, float]] = None
    if network_path:
        p = network_path.lower()
        compression = "gzip" if p.endswith(".gz") else None
        net = pd.read_csv(
            network_path,
            compression=compression,
            usecols=DEFAULT_BEAM_NETWORK_COLUMNS,
        )
        link_lengths_m = dict(zip(net["linkId"].astype(int), net["linkLength"].astype(float)))

    move_rows = _build_path_traversal_rows(events, link_lengths_m=link_lengths_m)
    park_rows = _build_parking_rows(events)
    obs_rows = pd.concat([move_rows, park_rows], ignore_index=True)

    emissions_rows = _apply_rates(obs_rows, rates_df=rates_df)
    skim_like = _aggregate_skims_like(emissions_rows, iterations=iterations)
    return skim_like


def write_skims_emissions(df: pd.DataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = output_path.lower()
    if p.endswith(".parquet"):
        df.to_parquet(output_path, index=False)
    elif p.endswith(".csv.gz"):
        df.to_csv(output_path, index=False, compression="gzip")
    else:
        raise ValueError("output_path must end with .csv.gz or .parquet")
