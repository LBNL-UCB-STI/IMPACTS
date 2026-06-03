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
from ...common import read_table

logger = logging.getLogger(__name__)

_MODELED_POLLUTANT_COLUMNS = {
    "PM2.5": "tons_per_year_PM2_5_county_allocated",
    "NOx": "tons_per_year_NOx_county_allocated",
    "PM10": "tons_per_year_PM10_county_allocated",
    "TOG": "tons_per_year_TOG_county_allocated",
    "ROG": "tons_per_year_ROG_county_allocated",
    "CO": "tons_per_year_CO_county_allocated",
    "SOx": "tons_per_year_SOx_county_allocated",
}

_SECTOR_TARGET_FIELDS = {
    "annual_pm25_short_tons": "PM2.5",
    "annual_nox_short_tons": "NOx",
    "annual_pm10_short_tons": "PM10",
    "annual_tog_short_tons": "TOG",
    "annual_rog_short_tons": "ROG",
    "annual_co_short_tons": "CO",
    "annual_sox_short_tons": "SOx",
}


def _slugify(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").lower()
    return token or "target"


def _normalize_token(value: object) -> str:
    return str("" if pd.isna(value) else value).strip()


def _load_vehicle_type_sectors(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    *,
    vehicle_category_metadata_file: str,
) -> pd.DataFrame:
    passenger = read_table(passenger_vehicle_types_path).copy()
    freight = read_table(freight_vehicle_types_path).copy()
    vehicle_types = pd.concat([passenger, freight], ignore_index=True, sort=False)
    required_columns = {"vehicleTypeId", "emfacVehicleCategory"}
    missing = sorted(required_columns - set(vehicle_types.columns))
    if missing:
        raise ValueError(
            "Vehicle types input must include vehicleTypeId and emfacVehicleCategory for postprocess step 2. "
            f"Missing: {missing}"
        )
    category_mapping = read_table(vehicle_category_metadata_file).copy()
    mapping_required = {"emfac_vehicle_category", "generic_vehicle_category"}
    mapping_missing = sorted(mapping_required - set(category_mapping.columns))
    if mapping_missing:
        raise ValueError(
            "Vehicle category metadata input must include emfac_vehicle_category and generic_vehicle_category "
            f"for postprocess step 2. Missing: {mapping_missing}"
        )
    category_mapping["emfac_vehicle_category"] = category_mapping["emfac_vehicle_category"].map(_normalize_token)
    category_mapping["generic_vehicle_category"] = category_mapping["generic_vehicle_category"].map(_normalize_token)
    category_mapping = category_mapping.loc[
        category_mapping["emfac_vehicle_category"].ne("") & category_mapping["generic_vehicle_category"].ne("")
    ].copy()
    category_mapping = (
        category_mapping[["emfac_vehicle_category", "generic_vehicle_category"]]
        .drop_duplicates(subset=["emfac_vehicle_category"], keep="first")
        .reset_index(drop=True)
    )
    prepared = vehicle_types.copy()
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].map(_normalize_token)
    prepared["emfacVehicleCategory"] = prepared["emfacVehicleCategory"].map(_normalize_token)
    prepared = prepared.loc[prepared["vehicleTypeId"].ne("")].copy()
    prepared = prepared.merge(
        category_mapping,
        how="left",
        left_on="emfacVehicleCategory",
        right_on="emfac_vehicle_category",
    )
    return (
        prepared[["vehicleTypeId", "generic_vehicle_category"]]
        .rename(columns={"generic_vehicle_category": "sector"})
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .reset_index(drop=True)
    )


def _build_targets_table(sector_targets: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in sector_targets:
        source = str(target["source"])
        sector = str(target["sector"])
        for field, pollutant_label in _SECTOR_TARGET_FIELDS.items():
            value = target.get(field)
            if value is not None:
                rows.append(
                    {
                        "source": source,
                        "sector": sector,
                        "pollutant": pollutant_label,
                        "target_tons": float(value),
                    }
                )
    if not rows:
        raise ValueError("Postprocess Step 2 requires configured annual sector targets.")
    return pd.DataFrame(rows)


def _aggregate_modeled_to_targets(
    modeled_emissions_path: str,
    *,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    vehicle_category_metadata_file: str,
) -> pd.DataFrame:
    modeled = read_table(modeled_emissions_path)
    required_columns = {"vehicleTypeId", "process"}
    missing = sorted(required_columns - set(modeled.columns))
    if missing:
        raise ValueError(
            "County-intersected modeled emissions input must include vehicleTypeId and process "
            f"for postprocess step 2. Missing: {missing}"
        )

    sector_lookup = _load_vehicle_type_sectors(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
        vehicle_category_metadata_file=vehicle_category_metadata_file,
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
        raise ValueError("Modeled emissions input does not include supported pollutant columns for postprocess step 2.")
    return pd.concat(rows, ignore_index=True)


def _build_comparison_table(*, modeled_df: pd.DataFrame, targets_df: pd.DataFrame) -> pd.DataFrame:
    comparison = targets_df.merge(modeled_df, how="outer", on=["source", "sector", "pollutant"])
    required_columns = {"target_tons", "simulation_tons"}
    missing = sorted(required_columns - set(comparison.columns))
    if missing:
        raise ValueError(f"Annual target comparison is missing required columns after merge: {missing}")
    comparison["target_tons"] = pd.to_numeric(comparison["target_tons"], errors="coerce").fillna(0.0)
    comparison["simulation_tons"] = pd.to_numeric(comparison["simulation_tons"], errors="coerce").fillna(0.0)
    comparison["difference_tons"] = comparison["simulation_tons"] - comparison["target_tons"]
    comparison["simulation_to_target_ratio"] = (
        comparison["simulation_tons"] / comparison["target_tons"].where(comparison["target_tons"].ne(0.0))
    )
    return comparison.sort_values(["source", "pollutant", "sector"]).reset_index(drop=True)


def _write_comparison_table(comparison: pd.DataFrame, *, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "step2_annual_targets_comparison.parquet"
    csv_path = output_dir / "step2_annual_targets_comparison.csv"
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
    output_path = output_dir / f"step2_{_slugify(source)}_{_slugify(pollutant)}_simulation_vs_target.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def _plot_combined_pollutant_comparison(
    comparison: pd.DataFrame,
    *,
    primary_source: str,
    secondary_source: str,
    pollutant: str,
    output_dir: Path,
) -> Optional[str]:
    primary = comparison.loc[
        comparison["source"].eq(primary_source) & comparison["pollutant"].eq(pollutant)
    ].copy().sort_values("sector").reset_index(drop=True)
    secondary = comparison.loc[
        comparison["source"].eq(secondary_source) & comparison["pollutant"].eq(pollutant)
    ].copy().sort_values("sector").reset_index(drop=True)
    if primary.empty or secondary.empty:
        return None

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(max(7, len(primary) * 1.2), 8),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    width = 0.38
    x = range(len(primary))
    ax_top.bar(
        [pos - width / 2 for pos in x],
        primary["simulation_tons"].to_numpy(dtype=float),
        width=width,
        label="Simulation",
        color="#1f77b4",
    )
    ax_top.bar(
        [pos + width / 2 for pos in x],
        primary["target_tons"].to_numpy(dtype=float),
        width=width,
        label="Target",
        color="#ff7f0e",
    )
    ax_top.set_xticks(list(x))
    ax_top.set_xticklabels(primary["sector"].tolist(), rotation=30, ha="right")
    ax_top.set_ylabel("Annual tons")
    ax_top.set_title(f"{primary_source} {pollutant}: Simulation vs Target")
    ax_top.legend()
    ax_top.grid(axis="y", alpha=0.2)

    height = 0.38
    y = range(len(secondary))
    ax_bot.barh(
        [pos + height / 2 for pos in y],
        secondary["simulation_tons"].to_numpy(dtype=float),
        height=height,
        label="Simulation",
        color="#1f77b4",
    )
    ax_bot.barh(
        [pos - height / 2 for pos in y],
        secondary["target_tons"].to_numpy(dtype=float),
        height=height,
        label="Target",
        color="#ff7f0e",
    )
    ax_bot.set_yticks(list(y))
    ax_bot.set_yticklabels(secondary["sector"].tolist())
    ax_bot.set_xlabel("Annual tons")
    ax_bot.set_title(f"{secondary_source} {pollutant}: Simulation vs Target")
    ax_bot.legend()
    ax_bot.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    output_path = output_dir / f"step2_{_slugify(primary_source)}_{_slugify(pollutant)}_simulation_vs_target.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def run(
    *,
    modeled_emissions_path: str,
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    vehicle_category_metadata_file: str,
    output_dir: Path,
    sector_targets: list[dict[str, object]],
) -> dict[str, str]:
    log_step_banner("Postprocess Step 2", "Compare Annual Targets", logger=logger)
    log_substep_banner("2.1", "compare modeled emissions with configured annual targets", logger=logger)
    targets_df = _build_targets_table(sector_targets)
    modeled_df = _aggregate_modeled_to_targets(
        modeled_emissions_path,
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        vehicle_category_metadata_file=vehicle_category_metadata_file,
    )
    comparison = _build_comparison_table(
        modeled_df=modeled_df,
        targets_df=targets_df,
    )
    outputs = _write_comparison_table(comparison, output_dir=output_dir)
    plotted: set[tuple[str, str]] = set()
    for pollutant in comparison["pollutant"].unique():
        sources = set(comparison.loc[comparison["pollutant"].eq(pollutant), "source"].tolist())
        if "mobile_onroad" in sources and "road_dust" in sources:
            plot_path = _plot_combined_pollutant_comparison(
                comparison,
                primary_source="mobile_onroad",
                secondary_source="road_dust",
                pollutant=str(pollutant),
                output_dir=output_dir,
            )
            if plot_path:
                outputs[f"mobile_onroad_{pollutant}_plot"] = plot_path
            plotted.update({("mobile_onroad", pollutant), ("road_dust", pollutant)})
        for source in sorted(sources):
            if (source, pollutant) in plotted:
                continue
            plot_path = _plot_source_pollutant_comparison(
                comparison,
                source=str(source),
                pollutant=str(pollutant),
                output_dir=output_dir,
            )
            if plot_path:
                outputs[f"{source}_{pollutant}_plot"] = plot_path
    logger.info("Postprocess Step 2 complete")
    return outputs
