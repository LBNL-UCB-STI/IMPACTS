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
    _set_progress_task,
    _slugify,
    _step_progress,
    _style_chart_axes,
)  # configures matplotlib backend before pyplot

import matplotlib.pyplot as plt
import pandas as pd

from ...common import _duckdb_scan_expression
from ...common import configure_duckdb_connection
from ...common import log_step_banner
from ...common import log_substep_banner
from ...common import normalize_county_fips
from ...common import read_table

logger = logging.getLogger(__name__)

_MODELED_POLLUTANT_COLUMNS = {
    "PM2.5": "tons_per_year_PM25_county_allocated",
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
        required_columns = {"county_COUNTYFP", "vehicleTypeId", "process"}
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(
                "County-intersected modeled emissions input must include county_COUNTYFP, vehicleTypeId, and process "
                f"for postprocess step 3. Missing: {missing}"
            )
        available = {
            pollutant: column
            for pollutant, column in _MODELED_POLLUTANT_COLUMNS.items()
            if column in columns
        }
        if not available:
            raise ValueError(
                "Modeled emissions input does not include any supported pollutant columns for postprocess step 3."
            )

        county_expr = (
            "LPAD(NULLIF(regexp_extract("
            "COALESCE(CAST(county_COUNTYFP AS VARCHAR), ''), '(\\d+)', 1), ''), 3, '0')"
        )
        aggregate_columns = ",\n                ".join(
            "SUM(COALESCE(TRY_CAST("
            f"{_duckdb_identifier(column)} AS DOUBLE), 0.0)) AS {_duckdb_identifier(column)}"
            for column in available.values()
        )
        grouped = con.execute(
            f"""
            WITH prepared AS (
                SELECT
                    {county_expr} AS county_COUNTYFP,
                    {", ".join(_duckdb_identifier(column) for column in available.values())}
                FROM {scan}
            )
            SELECT
                county_COUNTYFP,
                {aggregate_columns}
            FROM prepared
            WHERE county_COUNTYFP IS NOT NULL
            GROUP BY county_COUNTYFP
            """
        ).fetchdf()
    finally:
        con.close()

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
        pollutant_cols = [
            column for column in selector.get("columns", ())
            if column in inventory.columns
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


def _write_comparison_table(
    comparison: pd.DataFrame, *, output_dir: Path, target_slug: str
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step3_{target_slug}_emissions_comparison_by_county_pollutant"
    parquet_path = output_dir / f"{stem}.parquet"
    csv_path = output_dir / f"{stem}.csv"
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
    target_slug: str,
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
    ax.grid(axis="y", alpha=0.2)
    _style_chart_axes(ax)
    fig.tight_layout()

    pollutant_slug = {
        "PM2.5": "pm25",
        "NOx": "nox",
        "BC": "bc",
    }.get(pollutant, pollutant.lower().replace(".", ""))
    output_path = output_dir / f"step3_{target_slug}_county_{pollutant_slug}_simulation_vs_emfac.png"
    fig.savefig(output_path, dpi=PLOT_DPI)
    plt.close(fig)
    return str(output_path)


def _plot_regional_totals_comparison(
    comparison: pd.DataFrame,
    *,
    inventory_label: str,
    output_dir: Path,
    target_slug: str,
) -> Optional[str]:
    totals = (
        comparison.groupby("pollutant", dropna=False)[["simulation_tons", "emfac_tons"]]
        .sum()
        .reset_index()
    )
    if totals.empty:
        return None
    x = range(len(totals))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(7, len(totals) * 1.2), 5))
    ax.bar(
        [pos - width / 2 for pos in x],
        totals["simulation_tons"].to_numpy(dtype=float),
        width=width,
        label="Simulation",
        color="#1f77b4",
    )
    ax.bar(
        [pos + width / 2 for pos in x],
        totals["emfac_tons"].to_numpy(dtype=float),
        width=width,
        label=inventory_label,
        color="#ff7f0e",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(totals["pollutant"].tolist(), rotation=30, ha="right")
    ax.set_ylabel("Annual tons (all counties)")
    ax.set_title(f"Regional Total: Simulation vs {inventory_label}")
    ax.grid(axis="y", alpha=0.2)
    _style_chart_axes(ax)
    fig.tight_layout()
    output_path = output_dir / f"step3_{target_slug}_regional_totals_simulation_vs_emfac.png"
    fig.savefig(output_path, dpi=PLOT_DPI)
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
    progress = _step_progress(7, "Postprocess Step 3")
    try:
        _set_progress_task(progress, "county lookup", step_label="Postprocess Step 3")
        county_lookup = _load_county_lookup(county_boundaries_path)
        _advance_progress(progress)

        _set_progress_task(progress, "modeled emissions", step_label="Postprocess Step 3")
        modeled_df = _aggregate_modeled_emissions(
            modeled_emissions_path,
            county_lookup=county_lookup,
        )
        _advance_progress(progress)

        _set_progress_task(progress, "inventory emissions", step_label="Postprocess Step 3")
        inventory_df = _aggregate_inventory_emissions(
            inventory_path,
            pollutant_targets=pollutant_targets,
        )
        inventory_df["emfac_tons"] = pd.to_numeric(inventory_df["emfac_tons"], errors="coerce").fillna(0.0)
        _advance_progress(progress)

        _set_progress_task(progress, "comparison table", step_label="Postprocess Step 3")
        comparison = _build_comparison_table(
            modeled_df=modeled_df,
            inventory_df=inventory_df,
            county_order=county_order,
        )
        target_slug = _slugify(target_name)
        _advance_progress(progress)

        _set_progress_task(progress, "write tables", step_label="Postprocess Step 3")
        outputs = _write_comparison_table(comparison, output_dir=output_dir, target_slug=target_slug)
        _advance_progress(progress)

        _set_progress_task(progress, "county plots", step_label="Postprocess Step 3")
        for pollutant in ("PM2.5", "NOx", "BC"):
            plot_path = _plot_county_comparison(
                comparison,
                pollutant=pollutant,
                inventory_label=inventory_label,
                output_dir=output_dir,
                target_slug=target_slug,
            )
            if plot_path:
                outputs[f"{pollutant}_plot"] = plot_path
        _advance_progress(progress)

        _set_progress_task(progress, "regional totals plot", step_label="Postprocess Step 3")
        plot_path = _plot_regional_totals_comparison(
            comparison,
            inventory_label=inventory_label,
            output_dir=output_dir,
            target_slug=target_slug,
        )
        if plot_path:
            outputs["regional_totals_plot"] = plot_path
        _advance_progress(progress)
    finally:
        _close_progress(progress)
    logger.info("Postprocess Step 3 complete")
    return outputs


def run_from_output_dir(output_dir: Path) -> dict[str, str]:
    """Run Step 3 from a pipeline output directory using manifest-resolved paths."""
    import pandas as pd

    from impacts.postprocessor import (
        _humanize_target_name,
        _resolve_county_boundaries_path,
        _resolve_emissions_inventory_path,
        _resolve_modeled_emissions_path,
    )
    from impacts.config.path_registry import build_registry

    from ._common import settings_path_from_output_dir
    from ...common import normalize_county_fips
    from ...config.settings_builder import load_settings_from_yaml

    output_dir = Path(output_dir)
    run_manifest_path = output_dir / "pipeline_manifest.yaml"
    settings_path = settings_path_from_output_dir(output_dir)
    settings = load_settings_from_yaml(settings_path)
    if not settings.impacts.analysis.inventory_targets:
        logger.info("No inventory targets configured, skipping Step 3.")
        return {}
    registry = build_registry(settings, settings_path)
    modeled_path = _resolve_modeled_emissions_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    county_boundaries_path = _resolve_county_boundaries_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
    )
    county_order: list[str] = []
    if settings.shared.geography.fips.counties:
        import geopandas as gpd

        county_gdf = gpd.read_file(county_boundaries_path)
        county_gdf["COUNTYFP"] = normalize_county_fips(county_gdf["COUNTYFP"])
        wanted = set(
            normalize_county_fips(pd.Series(list(settings.shared.geography.fips.counties)))
            .dropna()
            .tolist()
        )
        county_order = (
            county_gdf.loc[county_gdf["COUNTYFP"].isin(wanted), ["COUNTYFP", "NAME"]]
            .drop_duplicates()
            .sort_values("COUNTYFP")["NAME"]
            .astype(str)
            .tolist()
        )
    inventory_path = _resolve_emissions_inventory_path(
        settings_path,
        run_manifest_path=run_manifest_path,
        output_root=output_dir,
        registry=registry,
        settings=settings,
    )
    _activities = dict(getattr(settings.impacts, "activities", None) or {})
    _model_source = dict(_activities.get("emissions_inventory") or {}).get("model_source") or "EMFAC"
    outputs: dict[str, str] = {}
    for target in settings.impacts.analysis.inventory_targets:
        target_outputs = run(
            modeled_emissions_path=str(modeled_path),
            inventory_path=str(inventory_path),
            county_boundaries_path=str(county_boundaries_path),
            output_dir=output_dir / "postprocess" / "emissions_inventory",
            county_order=county_order,
            target_name=target.name,
            inventory_label=f"{_model_source} {_humanize_target_name(target.name)}".strip(),
            pollutant_targets={
                pollutant: {
                    "columns": tuple(selector.columns),
                }
                for pollutant, selector in target.pollutants.items()
            },
        )
        for key, value in target_outputs.items():
            outputs[f"{target.name}_{key}"] = value
    return outputs


if __name__ == "__main__":
    import argparse
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        prog="python -m impacts.pipeline.postprocess.step3_compare_emissions_inventory",
        description="Run emissions inventory comparison from an IMPACTS output directory.",
    )
    parser.add_argument("output_dir", type=Path,
                        help="Path to the main pipeline output folder.")
    args = parser.parse_args()

    run_from_output_dir(Path(args.output_dir))
