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

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_table

logger = logging.getLogger(__name__)

_MODELED_POLLUTANT_COLUMNS = {
    "PM2.5": "tons_per_year_PM2_5_county_allocated",
    "NOx": "tons_per_year_NOx_county_allocated",
}

_LIGHT_DUTY_EMFAC = {"LDA", "LDT1", "LDT2"}
_MEDIUM_DUTY_EMFAC = {"MDV"}


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "target"


def _normalize_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _classify_sector(row: pd.Series) -> str:
    vehicle_category = _normalize_token(row.get("vehicleCategory")).lower()
    vehicle_class = _normalize_token(row.get("vehicleClass")).lower()
    emfac_category = _normalize_token(row.get("emfacVehicleCategory")).upper()
    vehicle_use = _normalize_token(row.get("vehicleUse")).lower()

    if vehicle_category == "car":
        return "passenger_cars"
    if emfac_category in _LIGHT_DUTY_EMFAC:
        return "light_duty_trucks"
    if emfac_category in _MEDIUM_DUTY_EMFAC:
        return "medium_duty_trucks"
    if emfac_category.startswith("T6") or emfac_category.startswith("T7"):
        return "heavy_duty_trucks"
    if "class 1&2" in vehicle_class or "class12" in vehicle_category or vehicle_category.startswith("class12"):
        return "light_duty_trucks"
    if "class 4-6" in vehicle_class or "class456" in vehicle_category or vehicle_category.startswith("mdv"):
        return "medium_duty_trucks"
    if "class 7&8" in vehicle_class or "class78" in vehicle_category or vehicle_category.startswith("hdt") or vehicle_category.startswith("hdv"):
        return "heavy_duty_trucks"
    if vehicle_use == "freight":
        return "other"
    return "other"


def _load_vehicle_type_sectors(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    passenger = read_table(passenger_vehicle_types_path).copy()
    freight = read_table(freight_vehicle_types_path).copy()
    vehicle_types = pd.concat([passenger, freight], ignore_index=True, sort=False)
    if "vehicleTypeId" not in vehicle_types.columns:
        raise ValueError("Vehicle types input must include vehicleTypeId for analysis Step 1.")
    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_token)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()
    prepared["sector"] = prepared.apply(_classify_sector, axis=1)
    return (
        prepared[["vehicleTypeId", "sector"]]
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .reset_index(drop=True)
    )


def _build_targets_table(sector_targets: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in sector_targets:
        source = str(target["source"])
        sector = str(target["sector"])
        if target.get("annual_pm25_short_tons") is not None:
            rows.append(
                {
                    "source": source,
                    "sector": sector,
                    "pollutant": "PM2.5",
                    "target_tons": float(target["annual_pm25_short_tons"]),
                }
            )
        if target.get("annual_nox_short_tons") is not None:
            rows.append(
                {
                    "source": source,
                    "sector": sector,
                    "pollutant": "NOx",
                    "target_tons": float(target["annual_nox_short_tons"]),
                }
            )
    if not rows:
        raise ValueError("Analysis Step 1 requires configured annual sector targets.")
    return pd.DataFrame(rows)


def _aggregate_modeled_to_targets(
    modeled_emissions_path: str,
    *,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
) -> pd.DataFrame:
    modeled = read_table(modeled_emissions_path)
    required_columns = {"vehicleTypeId", "process"}
    missing = sorted(required_columns - set(modeled.columns))
    if missing:
        raise ValueError(
            "County-intersected modeled emissions input must include vehicleTypeId and process "
            f"for analysis Step 1. Missing: {missing}"
        )

    sector_lookup = _load_vehicle_type_sectors(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
    )
    modeled = modeled.copy()
    modeled["vehicleTypeId"] = modeled["vehicleTypeId"].map(_normalize_token)
    modeled["process"] = modeled["process"].map(_normalize_token).str.upper()
    modeled = modeled.merge(sector_lookup, how="left", on="vehicleTypeId")
    missing_vehicle_types = modeled.loc[modeled["sector"].isna(), "vehicleTypeId"].drop_duplicates().tolist()
    if missing_vehicle_types:
        raise ValueError(
            "Could not resolve analysis sector for modeled vehicleTypeId values: "
            f"{missing_vehicle_types[:10]}"
        )

    rows: list[pd.DataFrame] = []
    for pollutant, column in _MODELED_POLLUTANT_COLUMNS.items():
        if column not in modeled.columns:
            continue
        frame = modeled[["process", "sector", column]].copy()
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if pollutant == "PM2.5":
            road_dust = (
                frame.loc[frame["process"].eq("PRDUST"), [column]]
                .sum(numeric_only=True)
                .iloc[0]
            )
            non_road_dust = (
                frame.loc[~frame["process"].eq("PRDUST")]
                .groupby("sector", dropna=False)[column]
                .sum()
                .reset_index()
            )
            if not non_road_dust.empty:
                non_road_dust["source"] = "mobile_onroad"
                non_road_dust["pollutant"] = pollutant
                non_road_dust = non_road_dust.rename(columns={column: "simulation_tons"})
                rows.append(non_road_dust[["source", "sector", "pollutant", "simulation_tons"]])
            rows.append(
                pd.DataFrame(
                    [{"source": "road_dust", "sector": "all", "pollutant": pollutant, "simulation_tons": float(road_dust)}]
                )
            )
        else:
            grouped = frame.groupby("sector", dropna=False)[column].sum().reset_index()
            if grouped.empty:
                continue
            grouped["source"] = "mobile_onroad"
            grouped["pollutant"] = pollutant
            grouped = grouped.rename(columns={column: "simulation_tons"})
            rows.append(grouped[["source", "sector", "pollutant", "simulation_tons"]])

    if not rows:
        raise ValueError("Modeled emissions input does not include supported pollutant columns for analysis Step 1.")
    return pd.concat(rows, ignore_index=True)


def _build_comparison_table(*, modeled_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    comparison = targets_df.merge(modeled_df, how="outer", on=["source", "sector", "pollutant"])
    comparison["target_tons"] = pd.to_numeric(comparison.get("target_tons", 0.0), errors="coerce").fillna(0.0)
    comparison["simulation_tons"] = pd.to_numeric(comparison.get("simulation_tons", 0.0), errors="coerce").fillna(0.0)
    comparison["difference_tons"] = comparison["simulation_tons"] - comparison["target_tons"]
    comparison["simulation_to_target_ratio"] = (
        comparison["simulation_tons"] / comparison["target_tons"].where(comparison["target_tons"].ne(0.0))
    )
    return comparison.sort_values(["source", "pollutant", "sector"]).reset_index(drop=True)


def _write_comparison_table(comparison: pd.DataFrame, *, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "step1_annual_targets_comparison.parquet"
    csv_path = output_dir / "step1_annual_targets_comparison.csv"
    comparison.to_parquet(parquet_path, index=False)
    comparison.to_csv(csv_path, index=False)
    return {
        "comparison_parquet": str(parquet_path),
        "comparison_csv": str(csv_path),
    }


def _plot_source_pollutant_comparison(
    comparison: pd.DataFrame,
    *,
    source: str,
    pollutant: str,
    output_dir: Path,
) -> Optional[str]:
    subset = comparison.loc[
        comparison["source"].eq(source) & comparison["pollutant"].eq(pollutant)
    ].copy()
    if subset.empty:
        return None
    subset = subset.sort_values("sector").reset_index(drop=True)
    x = range(len(subset))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(subset) * 1.2), 5))
    ax.bar(
        [pos - width / 2 for pos in x],
        subset["simulation_tons"].to_numpy(dtype=float),
        width=width,
        label="Simulation",
        color="#1f77b4",
    )
    ax.bar(
        [pos + width / 2 for pos in x],
        subset["target_tons"].to_numpy(dtype=float),
        width=width,
        label="Target",
        color="#ff7f0e",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(subset["sector"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Annual tons")
    ax.set_title(f"{source} {pollutant}: Simulation vs Target")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    output_path = output_dir / f"step1_{_slugify(source)}_{_slugify(pollutant)}_simulation_vs_target.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def run(
    *,
    modeled_emissions_path: str,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    output_dir: Path,
    sector_targets: list[dict[str, object]],
) -> dict[str, str]:
    log_step_banner("Analysis Step 1", "Compare Annual Targets", logger=logger)
    log_substep_banner("1.1", "compare modeled emissions with configured annual targets", logger=logger)
    targets_df = _build_targets_table(sector_targets)
    modeled_df = _aggregate_modeled_to_targets(
        modeled_emissions_path,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
    )
    comparison = _build_comparison_table(
        modeled_df=modeled_df,
        targets_df=targets_df,
    )
    outputs = _write_comparison_table(comparison, output_dir=output_dir)
    for (source, pollutant), _ in comparison.groupby(["source", "pollutant"], dropna=False):
        plot_path = _plot_source_pollutant_comparison(
            comparison,
            source=str(source),
            pollutant=str(pollutant),
            output_dir=output_dir,
        )
        if plot_path:
            outputs[f"{source}_{pollutant}_plot"] = plot_path
    logger.info("Analysis Step 1 complete")
    return outputs
