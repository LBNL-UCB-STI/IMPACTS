"""Acquire baseline-year ACS summary tables into IMPACTS' persistent input cache."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Dict

import pandas as pd
import requests


ACS_TABLES = {
    "1yr": ("B01003", "B11001", "B11002", "B26001"),
    "5yr": ("B26001",),
}


def _table_url(year: int, period: str, table: str) -> str:
    if period not in ACS_TABLES:
        raise ValueError(f"Unsupported ACS release period: {period}")
    if year < 2018:
        raise ValueError("ACS summary-file acquisition requires a baseline year of 2018 or later")
    root = f"https://www2.census.gov/programs-surveys/acs/summary_file/{year}"
    if year <= 2020:
        root += "/prototype"
    else:
        root += "/table-based-SF/data"
    filename = f"acsdt{period[0]}y{year}-{table.lower()}.dat"
    return f"{root}/{period.upper()}Data/{filename}"


def acquire_acs_tables(year: int, cache_root: Path) -> Dict[str, Dict[str, str]]:
    """Fetch the year-matched 1-year and 5-year ACS tables, reusing cached files."""
    year = int(year)
    if year < 2018:
        raise ValueError("ACS summary-file acquisition requires a baseline year of 2018 or later")

    result: Dict[str, Dict[str, str]] = {}
    manifest_path = cache_root / "acs" / str(year) / "manifest.json"
    previous_sources: dict[str, dict[str, str]] = {}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_sources = previous.get("sources", {})
        except (json.JSONDecodeError, OSError):
            pass
    manifest: dict[str, object] = {
        "baseline_year": year,
        "sources": {},
    }
    target_root = cache_root / "acs" / str(year)
    target_root.mkdir(parents=True, exist_ok=True)

    for period, tables in ACS_TABLES.items():
        result[period] = {}
        for table in tables:
            url = _table_url(year, period, table)
            destination = target_root / period / f"{table.lower()}.dat"
            retrieved_at = None
            if not destination.is_file() or destination.stat().st_size == 0:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    prefix=f".{table.lower()}-", dir=destination.parent, delete=False
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    digest = hashlib.sha256()
                    try:
                        with requests.get(url, stream=True, timeout=(20, 120)) as response:
                            response.raise_for_status()
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    temporary.write(chunk)
                                    digest.update(chunk)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                        temporary.seek(0)
                        header = temporary.read(4096).upper()
                        if b"GEO_ID" not in header or table.encode("ascii") not in header:
                            raise ValueError(f"Census response is not the expected ACS {table} table: {url}")
                        os.replace(temporary_path, destination)
                    except Exception:
                        temporary_path.unlink(missing_ok=True)
                        raise
                file_hash = digest.hexdigest()
                retrieved_at = datetime.now(timezone.utc).isoformat()
            else:
                digest = hashlib.sha256()
                with destination.open("rb") as cached_file:
                    header_bytes = cached_file.read(4096)
                    header = header_bytes.upper()
                    digest.update(header_bytes)
                    for chunk in iter(lambda: cached_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                file_hash = digest.hexdigest()
                if b"GEO_ID" not in header or table.encode("ascii") not in header:
                    raise ValueError(f"Cached file is not the expected ACS {table} table: {destination}")
                prior_source = previous_sources.get(f"{period}:{table}", {})
                retrieved_at = prior_source.get("retrieved_at")

            result[period][table] = str(destination)
            manifest["sources"][f"{period}:{table}"] = {
                "url": url,
                "path": str(destination),
                "sha256": file_hash,
                "retrieved_at": retrieved_at,
            }

    manifest["manifest_written_at"] = datetime.now(timezone.utc).isoformat()
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return result


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
