#!/usr/bin/env python
"""Convert grid emissions to concentration using ISRM transfer matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from impacts.defaults import DEFAULT_DISPERSION_EMISSIONS_COLUMNS as DEFAULT_EMISSIONS_COLUMNS


@dataclass
class DispersionConfig:
    """Configuration for ISRM-based dispersion."""

    isrm_url: Optional[str] = None
    emissions_input_path: Optional[str] = None
    concentration_output_path: str = "src/impacts/tmp/grid_concentration.parquet"
    concentration_factor: float = 28766.639
    include_bc: bool = False
    include_health: bool = False


DEFAULT_DISPERSION_CONFIG = DispersionConfig()

def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    raise ValueError(f"Unsupported table format: {path}. Use .csv, .csv.gz, or .parquet")


def prepare_grid_emissions(emissions_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize emissions to ISRM input species and aggregate by grid."""
    if "inmap_srm_cell_id" not in emissions_df.columns:
        raise ValueError("No grid id column found. Expected inmap_srm_cell_id")

    emission_cols = [c for c in DEFAULT_EMISSIONS_COLUMNS if c != "inmap_srm_cell_id"]
    source_column_map = {}
    for col in emission_cols:
        if col in emissions_df.columns:
            source_column_map[col] = col
            continue
        labeled = f"{col}_inmap_allocated"
        if labeled in emissions_df.columns:
            source_column_map[col] = labeled

    df = emissions_df.copy()
    df["GRID"] = pd.to_numeric(df["inmap_srm_cell_id"], errors="coerce")
    df = df[df["GRID"].notna()].copy()
    df["GRID"] = df["GRID"].astype(int)

    for col in emission_cols:
        source_col = source_column_map.get(col)
        if source_col is None:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0)

    return df.groupby("GRID", dropna=False)[emission_cols].sum().reset_index()


def load_isrm_store(isrm_url: str):
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
) -> pd.DataFrame:
    """Compute concentration fields from grid emissions using ISRM."""
    emis = prepare_grid_emissions(grid_emissions_df)

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

    bch = np.zeros_like(total)
    if include_bc:
        bch = _matrix_response(
            sr,
            "PrimaryPM25",
            source_cells,
            source_indexed["tons_per_year_BCh"].to_numpy(),
        )
        total = total + bch

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
        results["BCh"] = factor * bch

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
    isrm_url: Optional[str] = None,
    factor: float = 28766.639,
    include_bc: bool = False,
    include_health: bool = False,
) -> pd.DataFrame:
    """Read grid emissions and write ISRM concentration outputs."""
    if not isrm_url:
        raise ValueError(
            "isrm_url must be configured. "
            "Set dispersions.inmap.isrm_zarr (or isrm_zarr_directory / isrm_zarr_s3bucket) in runtime.yaml."
        )
    emissions_df = _read_table(emissions_input_path)
    sr = load_isrm_store(isrm_url)
    concentrations = compute_isrm_concentrations(
        grid_emissions_df=emissions_df,
        sr=sr,
        factor=factor,
        include_bc=include_bc,
        include_health=include_health,
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
