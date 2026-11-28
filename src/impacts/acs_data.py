"""Read ACS tables supplied in the versioned Census input bundle."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


ACS_TABLES = {
    "1yr": ("B01003", "B11001", "B11002", "B26001"),
    "5yr": ("B26001",),
}


def read_acs_estimates(
    path: str | Path,
    *,
    table: str,
    geography: str,
    geoid_prefixes: tuple[str, ...],
) -> pd.DataFrame:
    """Read estimate and MOE rows for selected county or tract GEOIDs."""
    if geography not in {"county", "tract"}:
        raise ValueError("ACS geography must be county or tract")
    columns = pd.read_csv(path, sep="|", nrows=0, dtype=str).columns
    geoid_column = next((column for column in columns if column.lstrip("#") == "GEO_ID"), None)
    estimate_column = f"{table}_001E"
    moe_column = f"{table}_001M"
    missing = [column for column in (geoid_column, estimate_column, moe_column) if column not in columns]
    if missing:
        raise ValueError(f"ACS {table} summary file is missing required columns: {missing}")

    records = []
    for chunk in pd.read_csv(
        path,
        sep="|",
        dtype=str,
        usecols=[geoid_column, estimate_column, moe_column],
        chunksize=100_000,
    ):
        geoid = chunk[geoid_column].str.split("US", n=1).str[-1]
        if geography == "county":
            is_geography = geoid.str.len().eq(5)
        else:
            is_geography = geoid.str.len().eq(11)
        selected = chunk.loc[is_geography & geoid.str.startswith(geoid_prefixes)].copy()
        if selected.empty:
            continue
        selected["geoid"] = geoid.loc[selected.index]
        selected["estimate"] = pd.to_numeric(selected[estimate_column], errors="coerce")
        selected["moe"] = pd.to_numeric(selected[moe_column], errors="coerce")
        if selected[["estimate", "moe"]].isna().to_numpy().any():
            raise ValueError(f"ACS {table} contains a missing estimate or margin of error")
        records.append(selected[["geoid", "estimate", "moe"]])

    if not records:
        raise ValueError(f"ACS {table} summary file has no matching {geography} rows")
    result = pd.concat(records, ignore_index=True)
    if result["geoid"].duplicated().any():
        raise ValueError(f"ACS {table} summary file has duplicate GEOIDs")
    return result.sort_values("geoid", ignore_index=True)
