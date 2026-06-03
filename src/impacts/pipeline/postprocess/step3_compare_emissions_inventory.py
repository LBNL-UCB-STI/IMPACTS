from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "impacts-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import normalize_county_fips
from ...common import read_table

logger = logging.getLogger(__name__)

_MODELED_POLLUTANT_COLUMNS = {
    "PM2.5": "tons_per_year_PM2_5_county_allocated",
    "NOx": "tons_per_year_NOx_county_allocated",
    "BC": "tons_per_year_BC_county_allocated",
}


def _load_county_lookup(county_boundaries_path: str) -> pd.DataFrame:
    import geopandas as gpd

    county_gdf = gpd.read_file(county_boundaries_path)
    if "COUNTYFP" not in county_gdf.columns or "NAME" not in county_gdf.columns:
        raise ValueError(
            "County boundaries must include COUNTYFP and NAME columns for postprocess step 3."
        )
    lookup = county_gdf[["COUNTYFP", "NAME"]].drop_duplicates()
    lookup["COUNTYFP"] = normalize_county_fips(lookup["COUNTYFP"])
    lookup["NAME"] = lookup["NAME"].astype("string")
    return lookup


def _aggregate_modeled_emissions(
    modeled_emissions_path: str,
    *,
    county_lookup: pd.DataFrame,
) -> pd.DataFrame:
    modeled = read_table(modeled_emissions_path)
    required_columns = {"county_COUNTYFP", "vehicleTypeId", "process"}
    missing = sorted(required_columns - set(modeled.columns))
    if missing:
        raise ValueError(
            "County-intersected modeled emissions input must include county_COUNTYFP, vehicleTypeId, and process "
            f"for postprocess step 3. Missing: {missing}"
        )
    modeled["county_COUNTYFP"] = normalize_county_fips(modeled["county_COUNTYFP"])
    modeled = modeled.loc[modeled["county_COUNTYFP"].notna()].copy()
    available = {
        pollutant: column
        for pollutant, column in _MODELED_POLLUTANT_COLUMNS.items()
        if column in modeled.columns
    }
    if not available:
        raise ValueError(
            "Modeled emissions input does not include any supported pollutant columns for postprocess step 3."
        )
    grouped = (
        modeled.groupby("county_COUNTYFP", dropna=False)[list(available.values())]
        .sum(numeric_only=True)
        .reset_index()
    )
    grouped = grouped.merge(county_lookup, how="left", left_on="county_COUNTYFP", right_on="COUNTYFP")
    rows: list[pd.DataFrame] = []
    for pollutant, column in available.items():
        rows.append(
            pd.DataFrame(
                {
                    "countyfp": grouped["county_COUNTYFP"].astype("string"),
                    "county": grouped["NAME"].astype("string"),
                    "pollutant": pollutant,
                    "simulation_tons": pd.to_numeric(grouped[column], errors="coerce").fillna(0.0),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "target"


def _aggregate_inventory_emissions(
    inventory_path: str,
    *,
    pollutant_targets: dict[str, dict[str, tuple[str, ...]]],
) -> pd.DataFrame:
    inventory = read_table(inventory_path)
    if "county" not in inventory.columns:
        raise ValueError("Inventory input must include county for postprocess step 3.")
    rows: list[pd.DataFrame] = []
    for pollutant, selector in pollutant_targets.items():
        explicit_columns = tuple(selector.get("columns", ()))
        prefixes = tuple(selector.get("prefixes", ()))
        exclude_columns = set(selector.get("exclude_columns", ()))
        exclude_prefixes = tuple(selector.get("exclude_prefixes", ()))
        if explicit_columns:
            pollutant_cols = [
                column for column in explicit_columns
                if column in inventory.columns
            ]
        else:
            pollutant_cols = [
                column for column in inventory.columns
                if any(column.startswith(prefix) for prefix in prefixes)
                and column.endswith("_short_tons_per_year")
            ]
        pollutant_cols = [
            column for column in pollutant_cols
            if column not in exclude_columns
            and not any(column.startswith(prefix) for prefix in exclude_prefixes)
        ]
        if not pollutant_cols:
            continue
        frame = inventory[["county"] + pollutant_cols].copy()
        for column in pollutant_cols:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        frame["emfac_tons"] = frame[pollutant_cols].sum(axis=1)
        grouped = (
            frame.groupby("county", dropna=False)["emfac_tons"]
            .sum()
            .reset_index()
        )
        grouped["pollutant"] = pollutant
        rows.append(grouped[["county", "pollutant", "emfac_tons"]])
    if not rows:
        raise ValueError(
            "Inventory input does not include any configured pollutant columns for postprocess step 3."
        )
    return pd.concat(rows, ignore_index=True)


def _build_comparison_table(
    *,
    modeled_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    county_order: list[str],
) -> pd.DataFrame:
    comparison = modeled_df.merge(
        inventory_df,
        how="outer",
        on=["county", "pollutant"],
    )
    required_columns = {"countyfp", "simulation_tons", "emfac_tons"}
    missing = sorted(required_columns - set(comparison.columns))
    if missing:
        raise ValueError(f"County emissions comparison is missing required columns after merge: {missing}")
    comparison["countyfp"] = comparison["countyfp"].astype("string")
    comparison["simulation_tons"] = pd.to_numeric(comparison["simulation_tons"], errors="coerce").fillna(0.0)
    comparison["emfac_tons"] = pd.to_numeric(comparison["emfac_tons"], errors="coerce").fillna(0.0)
    comparison["county"] = comparison["county"].astype("string")
    if county_order:
        comparison["county"] = pd.Categorical(
            comparison["county"],
            categories=county_order,
            ordered=True,
        )
    comparison = comparison.sort_values(["pollutant", "county"]).reset_index(drop=True)
    return comparison


def _write_comparison_table(comparison: pd.DataFrame, *, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "step3_emissions_comparison_by_county_pollutant.parquet"
    csv_path = output_dir / "step3_emissions_comparison_by_county_pollutant.csv"
    comparison.to_parquet(parquet_path, index=False)
    comparison.to_csv(csv_path, index=False)
    return {
        "comparison_parquet": str(parquet_path),
        "comparison_csv": str(csv_path),
    }


def _plot_county_comparison(
    comparison: pd.DataFrame,
    *,
    pollutant: str,
    inventory_label: str,
    output_dir: Path,
) -> Optional[str]:
    subset = comparison.loc[comparison["pollutant"] == pollutant].copy()
    if subset.empty:
        return None
    subset["county"] = subset["county"].astype(str)
    subset = subset.sort_values("county").reset_index(drop=True)
    x = range(len(subset))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(subset) * 0.9), 5))
    ax.bar(
        [pos - width / 2 for pos in x],
        subset["simulation_tons"].to_numpy(dtype=float),
        width=width,
        label="Simulation",
        color="#1f77b4",
    )
    ax.bar(
        [pos + width / 2 for pos in x],
        subset["emfac_tons"].to_numpy(dtype=float),
        width=width,
        label=inventory_label,
        color="#ff7f0e",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset["county"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("Annual tons")
    ax.set_title(f"{pollutant} by County: Simulation vs {inventory_label}")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()

    pollutant_slug = {
        "PM2.5": "pm25",
        "NOx": "nox",
        "BC": "bc",
    }.get(pollutant, pollutant.lower().replace(".", ""))
    filename = f"step3_county_{pollutant_slug}_simulation_vs_emfac.png"
    output_path = output_dir / filename
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def run(
    *,
    modeled_emissions_path: str,
    inventory_path: str,
    county_boundaries_path: str,
    output_dir: Path,
    county_order: list[str],
    target_name: str,
    inventory_label: str,
    pollutant_targets: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, str]:
    log_step_banner("Postprocess Step 3", f"Compare Emissions Inventory ({inventory_label})", logger=logger)
    log_substep_banner("3.1", f"compare modeled emissions with {inventory_label} inventory", logger=logger)
    county_lookup = _load_county_lookup(county_boundaries_path)
    modeled_df = _aggregate_modeled_emissions(
        modeled_emissions_path,
        county_lookup=county_lookup,
    )
    inventory_df = _aggregate_inventory_emissions(
        inventory_path,
        pollutant_targets=pollutant_targets,
    )
    inventory_df["emfac_tons"] = pd.to_numeric(inventory_df["emfac_tons"], errors="coerce").fillna(0.0)
    comparison = _build_comparison_table(
        modeled_df=modeled_df,
        inventory_df=inventory_df,
        county_order=county_order,
    )
    target_output_dir = output_dir / _slugify(target_name)
    outputs = _write_comparison_table(comparison, output_dir=target_output_dir)
    for pollutant in ("PM2.5", "NOx", "BC"):
        plot_path = _plot_county_comparison(
            comparison,
            pollutant=pollutant,
            inventory_label=inventory_label,
            output_dir=target_output_dir,
        )
        if plot_path:
            outputs[f"{pollutant}_plot"] = plot_path
    logger.info("Postprocess Step 3 complete")
    return outputs
