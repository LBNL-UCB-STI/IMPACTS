from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import duckdb

from ._common import (
    PLOT_DPI,
    _advance_progress,
    _close_progress,
    _duckdb_identifier,
    _normalize_token,
    _set_progress_task,
    _slugify,
    _step_progress,
    _style_chart_axes,
    load_postprocess_vehicle_metadata,
)  # configures matplotlib backend before pyplot

import matplotlib.pyplot as plt
import pandas as pd

from ...common import _duckdb_scan_expression
from ...common import configure_duckdb_connection
from ...common import log_step_banner
from ...common import log_substep_banner

logger = logging.getLogger(__name__)

_MODELED_POLLUTANT_COLUMNS = {
    "PM2.5": "tons_per_year_PM25_county_allocated",
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


def _load_vehicle_type_sectors(
    passenger_vehicle_types_path: str,
    freight_vehicle_types_path: str,
    *,
    vehicle_category_metadata_file: str,
) -> pd.DataFrame:
    prepared = load_postprocess_vehicle_metadata(
        passenger_vehicle_types_path=passenger_vehicle_types_path,
        freight_vehicle_types_path=freight_vehicle_types_path,
        vehicle_category_metadata_file=vehicle_category_metadata_file,
    )
    return (
        prepared[["vehicleTypeId", "sector"]]
        .loc[lambda df: df["sector"].ne("")]
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
    sector_lookup = _load_vehicle_type_sectors(
        passenger_vehicle_types_path,
        freight_vehicle_types_path,
        vehicle_category_metadata_file=vehicle_category_metadata_file,
    )
    sector_lookup = (
        sector_lookup.assign(
            vehicleTypeId=lambda df: df["vehicleTypeId"].map(_normalize_token),
            sector=lambda df: df["sector"].map(_normalize_token),
        )
        .loc[lambda df: df["vehicleTypeId"].ne("") & df["sector"].ne("")]
        .drop_duplicates(subset=["vehicleTypeId"], keep="first")
        .reset_index(drop=True)
    )
    scan = _duckdb_scan_expression(modeled_emissions_path)
    con = duckdb.connect()
    try:
        configure_duckdb_connection(
            con,
            working_dir=Path(modeled_emissions_path).parent,
            show_progress=False,
            profile="balanced",
        )
        columns = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM {scan}").fetchall()}
        required_columns = {"vehicleTypeId", "process"}
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(
                "County-intersected modeled emissions input must include vehicleTypeId and process "
                f"for postprocess step 2. Missing: {missing}"
            )
        available = {
            pollutant: column
            for pollutant, column in _MODELED_POLLUTANT_COLUMNS.items()
            if column in columns
        }
        if not available:
            raise ValueError(
                "Modeled emissions input does not include supported pollutant columns for postprocess step 2."
            )

        con.register("sector_lookup", sector_lookup)
        missing_vehicle_types = con.execute(
            f"""
            WITH modeled_vehicle_types AS (
                SELECT DISTINCT TRIM(CAST(vehicleTypeId AS VARCHAR)) AS vehicleTypeId
                FROM {scan}
                WHERE TRIM(COALESCE(CAST(vehicleTypeId AS VARCHAR), '')) <> ''
            )
            SELECT modeled_vehicle_types.vehicleTypeId
            FROM modeled_vehicle_types
            LEFT JOIN sector_lookup
                ON modeled_vehicle_types.vehicleTypeId = sector_lookup.vehicleTypeId
            WHERE sector_lookup.vehicleTypeId IS NULL
            LIMIT 10
            """
        ).fetchdf()["vehicleTypeId"].tolist()
        if missing_vehicle_types:
            raise ValueError(
                "Could not resolve analysis sector for modeled vehicleTypeId values: "
                f"{missing_vehicle_types}"
            )

        aggregate_columns = ",\n                    ".join(
            "SUM(COALESCE(TRY_CAST(modeled."
            f"{_duckdb_identifier(column)} AS DOUBLE), 0.0)) AS {_duckdb_identifier(_slugify(pollutant))}"
            for pollutant, column in available.items()
        )
        grouped = con.execute(
            f"""
            SELECT
                sector_lookup.sector,
                UPPER(TRIM(COALESCE(CAST(modeled.process AS VARCHAR), ''))) AS process,
                {aggregate_columns}
            FROM {scan} AS modeled
            INNER JOIN sector_lookup
                ON TRIM(CAST(modeled.vehicleTypeId AS VARCHAR)) = sector_lookup.vehicleTypeId
            GROUP BY sector_lookup.sector, UPPER(TRIM(COALESCE(CAST(modeled.process AS VARCHAR), '')))
            """
        ).fetchdf()
    finally:
        con.close()

    rows: list[pd.DataFrame] = []
    for pollutant in available:
        column = _slugify(pollutant)
        frame = grouped[["process", "sector", column]].copy()
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
            pollutant_grouped = frame.groupby("sector", dropna=False)[column].sum().reset_index()
            if pollutant_grouped.empty:
                continue
            pollutant_grouped["source"] = "mobile_onroad"
            pollutant_grouped["pollutant"] = pollutant
            pollutant_grouped = pollutant_grouped.rename(columns={column: "simulation_tons"})
            rows.append(pollutant_grouped[["source", "sector", "pollutant", "simulation_tons"]])

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
    ax.grid(axis="y", alpha=0.2)
    _style_chart_axes(ax)
    fig.tight_layout()
    output_path = output_dir / f"step2_{_slugify(source)}_{_slugify(pollutant)}_simulation_vs_target.png"
    fig.savefig(output_path, dpi=PLOT_DPI)
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
    ax_top.grid(axis="y", alpha=0.2)
    _style_chart_axes(ax_top)

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
    ax_bot.grid(axis="x", alpha=0.2)
    _style_chart_axes(ax_bot)

    fig.tight_layout()
    output_path = output_dir / f"step2_{_slugify(primary_source)}_{_slugify(pollutant)}_simulation_vs_target.png"
    fig.savefig(output_path, dpi=PLOT_DPI)
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
    progress = _step_progress(5, "Postprocess Step 2")
    try:
        _set_progress_task(progress, "targets", step_label="Postprocess Step 2")
        targets_df = _build_targets_table(sector_targets)
        _advance_progress(progress)

        _set_progress_task(progress, "modeled emissions", step_label="Postprocess Step 2")
        modeled_df = _aggregate_modeled_to_targets(
            modeled_emissions_path,
            passenger_vehicle_types_path=passenger_vehicle_types_path,
            freight_vehicle_types_path=freight_vehicle_types_path,
            vehicle_category_metadata_file=vehicle_category_metadata_file,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "comparison table", step_label="Postprocess Step 2")
        comparison = _build_comparison_table(
            modeled_df=modeled_df,
            targets_df=targets_df,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "write tables", step_label="Postprocess Step 2")
        outputs = _write_comparison_table(comparison, output_dir=output_dir)
        _advance_progress(progress)

        _set_progress_task(progress, "plots", step_label="Postprocess Step 2")
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
        _advance_progress(progress)
    finally:
        _close_progress(progress)
    logger.info("Postprocess Step 2 complete")
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 2 from a pipeline output directory using manifest-resolved paths."""
    from impacts.postprocessor import (
        _resolve_modeled_emissions_path,
        _resolve_vehicle_category_metadata_path,
        _resolve_vehicle_types_paths,
    )

    from ._common import settings_path_from_output_dir
    from ...config.settings_builder import load_settings_from_yaml

    output_dir = Path(output_dir)
    run_manifest_path = output_dir / "pipeline_manifest.yaml"
    settings_path = settings_path_from_output_dir(output_dir)
    settings = load_settings_from_yaml(settings_path)
    if not settings.impacts.analysis.sector_targets:
        logger.info("No sector targets configured, skipping Step 2.")
        return {}
    modeled_path = _resolve_modeled_emissions_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    passenger_vt, freight_vt = _resolve_vehicle_types_paths(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    vehicle_category_metadata = _resolve_vehicle_category_metadata_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    return run(
        modeled_emissions_path=str(modeled_path),
        passenger_vehicle_types_path=str(passenger_vt),
        freight_vehicle_types_path=str(freight_vt),
        vehicle_category_metadata_file=str(vehicle_category_metadata),
        output_dir=output_dir / "postprocess" / "annual_targets",
        sector_targets=[
            {
                "source": target.source,
                "sector": target.sector,
                "annual_pm25_short_tons": target.annual_pm25_short_tons,
                "annual_nox_short_tons": target.annual_nox_short_tons,
                "annual_pm10_short_tons": target.annual_pm10_short_tons,
                "annual_tog_short_tons": target.annual_tog_short_tons,
                "annual_rog_short_tons": target.annual_rog_short_tons,
                "annual_co_short_tons": target.annual_co_short_tons,
                "annual_sox_short_tons": target.annual_sox_short_tons,
            }
            for target in settings.impacts.analysis.sector_targets
        ],
    )


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step2_compare_annual_targets",
        description="Run annual targets comparison from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path,
                        help="Path to the main pipeline output folder.")
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir))
