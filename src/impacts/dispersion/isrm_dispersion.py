#!/usr/bin/env python
"""Convert grid emissions to concentration using ISRM transfer matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class DispersionConfig:
    """Configuration for ISRM-based dispersion."""

    isrm_url: str = "s3://inmap-model/isrm_v1.2.1.zarr/"
    emissions_input_path: Optional[str] = None
    concentration_output_path: str = "src/impacts/tmp/grid_concentration.parquet"
    concentration_factor: float = 28766.639
    include_bc: bool = False
    include_health: bool = False


DEFAULT_DISPERSION_CONFIG = DispersionConfig()
DEFAULT_EMISSIONS_COLUMNS = {
    "grid_id": "GRID",
    "rog": "tons_per_year_ROG",
    "nox": "tons_per_year_NOx",
    "nh3": "tons_per_year_NH3",
    "sox": "tons_per_year_SOx",
    "pm25": "tons_per_year_PM2_5",
    "bcv1": "tons_per_year_BCV1",
    "bcv3": "tons_per_year_BCV3",
}

def _first_existing(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    raise ValueError(f"Unsupported table format: {path}. Use .csv, .csv.gz, or .parquet")


def _get_column_or_zero(df: pd.DataFrame, ordered_names: Iterable[str]) -> pd.Series:
    name = _first_existing(df, ordered_names)
    if name is None:
        return pd.Series(np.zeros(len(df), dtype=float), index=df.index)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)


def _resolve_emissions_columns(config: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    resolved = DEFAULT_EMISSIONS_COLUMNS.copy()
    if config:
        resolved.update({k: v for k, v in config.items() if v})
    return resolved


def prepare_grid_emissions(
    emissions_df: pd.DataFrame,
    emissions_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Normalize emissions to ISRM input species and aggregate by grid."""
    columns = _resolve_emissions_columns(emissions_columns)
    grid_col = _first_existing(
        emissions_df,
        [columns["grid_id"], "GRID", "grid", "zone", "cell_id", "Location"],
    )
    if grid_col is None:
        raise ValueError("No grid id column found. Expected GRID/grid/zone/cell_id/Location")

    df = emissions_df.copy()
    df["GRID"] = pd.to_numeric(df[grid_col], errors="coerce")
    df = df[df["GRID"].notna()].copy()
    df["GRID"] = df["GRID"].astype(int)

    # Prioritize already-converted tons/year columns, then allocated or raw emissions columns.
    df["tons_per_year_ROG"] = _get_column_or_zero(
        df,
        [
            columns["rog"],
            "tons_per_year_ROG",
            "tons_per_year_ROG_allocated",
            "em_ROG_allocated",
            "em_ROG",
            "em_VOC_allocated",
            "em_VOC",
        ],
    )
    df["tons_per_year_NOx"] = _get_column_or_zero(
        df,
        [columns["nox"], "tons_per_year_NOx", "tons_per_year_NOx_allocated", "em_NOx_allocated", "em_NOx", "em_NOX_allocated", "em_NOX"],
    )
    df["tons_per_year_NH3"] = _get_column_or_zero(
        df,
        [columns["nh3"], "tons_per_year_NH3", "tons_per_year_NH3_allocated", "em_NH3_allocated", "em_NH3"],
    )
    df["tons_per_year_SOx"] = _get_column_or_zero(
        df,
        [columns["sox"], "tons_per_year_SOx", "tons_per_year_SOx_allocated", "em_SOx_allocated", "em_SOx", "em_SOX_allocated", "em_SOX"],
    )
    df["tons_per_year_PM2_5"] = _get_column_or_zero(
        df,
        [
            columns["pm25"],
            "tons_per_year_PM2_5",
            "tons_per_year_PM2_5_allocated",
            "em_PM2_5_allocated",
            "em_PM2_5",
            "em_PM25_allocated",
            "em_PM25",
        ],
    )
    df["tons_per_year_BCV1"] = _get_column_or_zero(
        df,
        [columns["bcv1"], "tons_per_year_BCV1", "tons_per_year_BCV1_allocated", "em_BCV1_allocated", "em_BCV1"],
    )
    df["tons_per_year_BCV3"] = _get_column_or_zero(
        df,
        [columns["bcv3"], "tons_per_year_BCV3", "tons_per_year_BCV3_allocated", "em_BCV3_allocated", "em_BCV3"],
    )

    return (
        df.groupby("GRID", dropna=False)[
            [
                "tons_per_year_ROG",
                "tons_per_year_NOx",
                "tons_per_year_NH3",
                "tons_per_year_SOx",
                "tons_per_year_PM2_5",
                "tons_per_year_BCV1",
                "tons_per_year_BCV3",
            ]
        ]
        .sum()
        .reset_index()
    )


def load_isrm_store(isrm_url: str = "s3://inmap-model/isrm_v1.2.1.zarr/"):
    """Load ISRM zarr store from S3 or local path."""
    import zarr

    if isrm_url.startswith("s3://"):
        try:
            import s3fs  # noqa: F401 — required by zarr's fsspec/s3 backend
        except ImportError as exc:
            raise ImportError("s3fs is required to read ISRM zarr from s3:// URLs") from exc
        return zarr.open(
            isrm_url,
            mode="r",
            storage_options={"anon": True, "client_kwargs": {"region_name": "us-east-2"}},
        )

    return zarr.open(isrm_url, mode="r")


def _matrix_response(sr, species_key: str, source_cells: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    transfer = sr[species_key].oindex[[0], source_cells, :]
    # transfer shape: (1, n_sources, n_receptors)
    return transfer[0, :, :].T.dot(source_values)


def compute_isrm_concentrations(
    grid_emissions_df: pd.DataFrame,
    sr,
    factor: float = 28766.639,
    include_bc: bool = False,
    include_health: bool = False,
    emissions_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Compute concentration fields from grid emissions using ISRM."""
    emis = prepare_grid_emissions(grid_emissions_df, emissions_columns=emissions_columns)

    if emis.empty:
        raise ValueError("No emissions rows found after GRID normalization.")

    source_cells = np.sort(emis["GRID"].unique().astype(int))
    source_indexed = emis.set_index("GRID").reindex(source_cells).fillna(0.0)

    soa = _matrix_response(sr, "SOA", source_cells, source_indexed["tons_per_year_ROG"].to_numpy())
    pno3 = _matrix_response(sr, "pNO3", source_cells, source_indexed["tons_per_year_NOx"].to_numpy())
    pnh4 = _matrix_response(sr, "pNH4", source_cells, source_indexed["tons_per_year_NH3"].to_numpy())
    pso4 = _matrix_response(sr, "pSO4", source_cells, source_indexed["tons_per_year_SOx"].to_numpy())
    primary = _matrix_response(
        sr,
        "PrimaryPM25",
        source_cells,
        source_indexed["tons_per_year_PM2_5"].to_numpy(),
    )

    total = soa + pno3 + pnh4 + pso4 + primary

    bcv1 = np.zeros_like(total)
    bcv3 = np.zeros_like(total)
    if include_bc:
        bcv1 = _matrix_response(
            sr,
            "PrimaryPM25",
            source_cells,
            source_indexed["tons_per_year_BCV1"].to_numpy(),
        )
        bcv3 = _matrix_response(
            sr,
            "PrimaryPM25",
            source_cells,
            source_indexed["tons_per_year_BCV3"].to_numpy(),
        )
        total = total + bcv1 + bcv3

    n_cells = int(sr["TotalPop"].shape[0]) if "TotalPop" in sr else int(total.shape[0])
    results = pd.DataFrame(
        {
            "GRID": np.arange(n_cells, dtype=int),
            "SOA": factor * soa,
            "pNO3": factor * pno3,
            "pNH4": factor * pnh4,
            "pSO4": factor * pso4,
            "PrimaryPM25": factor * primary,
            "TotalPM25": factor * total,
        }
    )

    if include_bc:
        results["BCV1"] = factor * bcv1
        results["BCV3"] = factor * bcv3

    if include_health and "TotalPop" in sr and "MortalityRate" in sr:
        total_pop = np.asarray(sr["TotalPop"][:])
        mortality_rate = np.asarray(sr["MortalityRate"][:])
        results["deathsK"] = (
            (np.exp(np.log(1.06) / 10.0 * results["TotalPM25"]) - 1.0)
            * total_pop
            * 1.096163
            * mortality_rate
            / 100000.0
            * 0.960899254
        )
        results["deathsL"] = (
            (np.exp(np.log(1.14) / 10.0 * results["TotalPM25"]) - 1.0)
            * total_pop
            * 1.096163
            * mortality_rate
            / 100000.0
            * 0.960899254
        )

    return results


def run_dispersion_from_file(
    emissions_input_path: str,
    output_path: Optional[str] = None,
    isrm_url: str = "s3://inmap-model/isrm_v1.2.1.zarr/",
    factor: float = 28766.639,
    include_bc: bool = False,
    include_health: bool = False,
    emissions_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read grid emissions and write ISRM concentration outputs."""
    emissions_df = _read_table(emissions_input_path)
    sr = load_isrm_store(isrm_url)
    concentrations = compute_isrm_concentrations(
        grid_emissions_df=emissions_df,
        sr=sr,
        factor=factor,
        include_bc=include_bc,
        include_health=include_health,
        emissions_columns=emissions_columns,
    )

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix == ".parquet":
            concentrations.to_parquet(out, index=False)
        elif suffix == ".gz" and out.name.lower().endswith(".csv.gz"):
            concentrations.to_csv(out, index=False, compression="gzip")
        elif suffix == ".csv":
            concentrations.to_csv(out, index=False)
        else:
            raise ValueError("Output path must end with .parquet, .csv, or .csv.gz")

    return concentrations
