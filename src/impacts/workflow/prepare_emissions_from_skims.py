"""Shared helpers for skims preparation and grid allocation inputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import duckdb
import pandas as pd

from ..common import annualize_prepared_skims_for_grid_allocation
from ..common import normalize_county_fips
from ..common import prepare_skims_for_grid_allocation
from ..common import prepared_table_target
from ..common import read_table
from ..common import resolve_manifest_input_path
from ..common import register_managed_input
from ..manifest.file_ops import file_entry
from . import _step_label

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def _existing_output(path: Path) -> Optional[str]:
    return str(path) if path.exists() else None


def _load_intersection_subset(path: str, columns: List[str]) -> pd.DataFrame:
    target = Path(path)
    frame = read_table(target)
    return frame[columns].copy()


def _load_intersection_subset_or_df(
    *,
    path: str,
    columns: List[str],
    intersection_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if intersection_df is not None:
        return intersection_df[columns].copy()
    return _load_intersection_subset(path, columns)


# ---------------------------------------------------------------------------
# substep 1.0 — cache check
# ---------------------------------------------------------------------------

def _reuse_existing_outputs(raw_dir: Path) -> Optional[Dict[str, Optional[str]]]:
    beam_emissions_for_inmap = _existing_output(raw_dir / "beam_emissions_for_inmap.parquet")
    if not beam_emissions_for_inmap:
        return None

    outputs = {
        "beam_activity_totals": _existing_output(raw_dir / "beam_activity_totals.parquet"),
        "beam_activity_correction_factors": _existing_output(raw_dir / "beam_activity_correction_factors.parquet"),
        "beam_emissions_for_aermod": _existing_output(raw_dir / "beam_emissions_for_aermod.parquet"),
        "beam_emissions_for_inmap": beam_emissions_for_inmap,
    }
    logger.info(
        "%s reusing existing emissions outputs; skipping recomputation (inmap=%s, aermod=%s)",
        _step_label("1.0"),
        outputs["beam_emissions_for_inmap"],
        outputs["beam_emissions_for_aermod"],
    )
    return outputs


# ---------------------------------------------------------------------------
# substep 1.1 — group intersection by link × zone
# ---------------------------------------------------------------------------

def _build_combined_grouped_table(
    *,
    intersection_path: str,
    intersection_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    required_cols = {
        "linkId",
        "countyfp",
        "county_zone_edge_proportion",
        "county_edge_link_length_m",
        "county_zone_link_length_m",
        "aermod_cell_id",
        "inmap_cell_id",
        "aermod_zone_edge_proportion",
        "aermod_edge_link_length_m",
        "aermod_zone_link_length_m",
        "inmap_zone_edge_proportion",
        "inmap_edge_link_length_m",
        "inmap_zone_link_length_m",
    }
    intersection = _load_intersection_subset_or_df(
        path=intersection_path,
        columns=list(required_cols),
        intersection_df=intersection_df,
    )
    missing = [col for col in required_cols if col not in intersection.columns]
    if missing:
        raise ValueError(
            f"{_step_label('1.1')} requires canonical intersection columns. Missing: {missing}"
        )

    metric_cols = [
        "county_zone_edge_proportion",
        "county_edge_link_length_m",
        "county_zone_link_length_m",
        "aermod_zone_edge_proportion",
        "aermod_edge_link_length_m",
        "aermod_zone_link_length_m",
        "inmap_zone_edge_proportion",
        "inmap_edge_link_length_m",
        "inmap_zone_link_length_m",
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("intersection_df", intersection)
        metric_select = ",\n                ".join([f"SUM(COALESCE(i.{col}, 0.0)) AS {col}" for col in metric_cols])
        grouped = con.execute(
            f"""
            SELECT
                i.linkId,
                i.aermod_cell_id,
                i.inmap_cell_id
                , i.countyfp
                {"," if metric_select else ""} {metric_select}
            FROM intersection_df AS i
            WHERE i.aermod_cell_id IS NOT NULL OR i.inmap_cell_id IS NOT NULL
            GROUP BY
                i.linkId,
                i.aermod_cell_id,
                i.inmap_cell_id
                , i.countyfp
            """
        ).df()
    finally:
        con.close()

    if grouped.empty:
        return None

    logger.info("%s BEAM mapping across grids rows=%d", _step_label("1.1"), len(grouped))
    return grouped


# ---------------------------------------------------------------------------
# substep 1.2 — allocate skims emissions to zones
# ---------------------------------------------------------------------------

def _build_combined_allocated_table(
    *,
    grouped_df: pd.DataFrame,
    skims_df: pd.DataFrame,
) -> Optional[pd.DataFrame]:
    if grouped_df is None or grouped_df.empty:
        return None

    emission_cols = [
        c for c in skims_df.columns
        if c.startswith("tons_per_year_") and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    activity_cols = [
        c for c in ("totVMT", "totTrips")
        if c in skims_df.columns and pd.api.types.is_numeric_dtype(skims_df[c])
    ]
    county_prop = "county_zone_edge_proportion"
    aermod_prop = "aermod_zone_edge_proportion"
    inmap_prop = "inmap_zone_edge_proportion"

    metric_cols = [
        col for col in grouped_df.columns
        if (col.startswith("aermod_") or col.startswith("inmap_")) and any(tag in col for tag in ("_proportion", "_length_m", "_surface_m2"))
    ]

    con = duckdb.connect(database=":memory:")
    try:
        con.register("grouped_df", grouped_df)
        con.register("skims_df", skims_df[["linkId", "vehicleTypeId", "process"] + activity_cols + emission_cols].copy())
        metric_select = ",\n                ".join([f"g.{col}" for col in metric_cols])
        county_activity = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{county_prop} AS DOUBLE), 0.0) AS {col}_county_allocated"
            for col in activity_cols
        ])
        county_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{county_prop} AS DOUBLE), 0.0) AS {col}_county_allocated"
            for col in emission_cols
        ])
        aermod_activity = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{aermod_prop} AS DOUBLE), 0.0) AS {col}_aermod_allocated"
            for col in activity_cols
        ])
        inmap_activity = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{inmap_prop} AS DOUBLE), 0.0) AS {col}_inmap_allocated"
            for col in activity_cols
        ])
        aermod_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{aermod_prop} AS DOUBLE), 0.0) AS {col}_aermod_allocated"
            for col in emission_cols
        ])
        inmap_alloc = ",\n                ".join([
            f"COALESCE(CAST(s.{col} AS DOUBLE), 0.0) * COALESCE(CAST(g.{inmap_prop} AS DOUBLE), 0.0) AS {col}_inmap_allocated"
            for col in emission_cols
        ])
        extra_select = ",\n                ".join([
            part
            for part in [
                metric_select,
                county_activity,
                aermod_activity,
                inmap_activity,
                county_alloc,
                aermod_alloc,
                inmap_alloc,
            ]
            if part
        ])
        allocated = con.execute(
            f"""
            SELECT
                g.linkId,
                s.vehicleTypeId,
                s.process,
                g.aermod_cell_id,
                g.inmap_cell_id,
                g.countyfp
                {"," if extra_select else ""} {extra_select}
            FROM grouped_df AS g
            LEFT JOIN skims_df AS s
                ON g.linkId = s.linkId
            """
        ).df()
    finally:
        con.close()

    if allocated.empty:
        return None

    logger.info("%s BEAM emissions allocated across grids rows=%d", _step_label("1.2"), len(allocated))
    return allocated


# ---------------------------------------------------------------------------
# substep 1.3 — aggregate beam activity totals by county
# ---------------------------------------------------------------------------

def _build_beam_activity_totals(
    allocated_df: pd.DataFrame,
    *,
    county_col: str,
) -> pd.DataFrame:
    county_activity = allocated_df[[county_col, "totVMT_county_allocated", "totTrips_county_allocated"]].copy()
    county_activity["_fips_norm"] = normalize_county_fips(county_activity[county_col])
    county_activity["totVMT_county_allocated"] = pd.to_numeric(county_activity["totVMT_county_allocated"], errors="coerce").fillna(0.0)
    county_activity["totTrips_county_allocated"] = pd.to_numeric(county_activity["totTrips_county_allocated"], errors="coerce").fillna(0.0)
    grouped = county_activity.groupby("_fips_norm", dropna=False).agg(
        totVMT=("totVMT_county_allocated", "sum"),
        totTrips=("totTrips_county_allocated", "sum"),
    ).reset_index()
    grouped = grouped.rename(columns={"_fips_norm": "countyfp"})
    zero_null_mask = grouped["countyfp"].isna() & grouped["totVMT"].eq(0.0) & grouped["totTrips"].eq(0.0)
    if zero_null_mask.any():
        grouped = grouped.loc[~zero_null_mask].reset_index(drop=True)
    return grouped


def _build_source_activity_totals(skims_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if {"countyfp", "distanceMiles_county", "tripCount_county"}.issubset(skims_df.columns):
        activity = skims_df[["countyfp", "distanceMiles_county", "tripCount_county"]].copy()
        vmt_col = "distanceMiles_county"
        trip_col = "tripCount_county"
    elif {"countyfp", "totVMT", "totTrips"}.issubset(skims_df.columns):
        activity = skims_df[["countyfp", "totVMT", "totTrips"]].copy()
        vmt_col = "totVMT"
        trip_col = "totTrips"
    else:
        return None

    activity["countyfp"] = normalize_county_fips(activity["countyfp"])
    activity[vmt_col] = pd.to_numeric(activity[vmt_col], errors="coerce").fillna(0.0)
    activity[trip_col] = pd.to_numeric(activity[trip_col], errors="coerce").fillna(0.0)
    grouped = activity.groupby("countyfp", dropna=False).agg(
        totVMT=(vmt_col, "sum"),
        totTrips=(trip_col, "sum"),
    ).reset_index()
    zero_null_mask = grouped["countyfp"].isna() & grouped["totVMT"].eq(0.0) & grouped["totTrips"].eq(0.0)
    if zero_null_mask.any():
        grouped = grouped.loc[~zero_null_mask].reset_index(drop=True)
    return grouped


def _resolve_staged_network_path(input_root: Path, manifest_inputs: Optional[Dict[str, Any]] = None) -> str:
    if manifest_inputs and "network" in manifest_inputs:
        return resolve_manifest_input_path(manifest_inputs["network"], label="network")
    network_dir = input_root / "network"
    for name in ["network.parquet", "network.csv.gz", "network.csv"]:
        candidate = network_dir / name
        if candidate.exists():
            return str(candidate)
    matches = sorted(
        path for path in network_dir.rglob("*")
        if path.is_file() and path.name in {"network.parquet", "network.csv.gz", "network.csv"}
    )
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"Could not find staged network file under {network_dir}")


def resolve_prepared_skims_path(input_root: Path) -> Optional[str]:
    for root in [input_root / "skims", input_root]:
        for name in ["prepared_skims_for_grid_allocation.parquet", "prepared_skims_for_grid_allocation.csv.gz"]:
            candidate = root / name
            if candidate.exists():
                return str(candidate)
    return None


def _resolve_staged_skims_input_path(input_root: Path, manifest_inputs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if manifest_inputs:
        for key in ["emissions_skims_input", "skims_from_events"]:
            entry = manifest_inputs.get(key)
            if entry:
                return resolve_manifest_input_path(entry, label=key)
    preferred = [
        "skims_from_events.parquet",
        "skims_from_events.csv.gz",
        "0.skimsEmissions.parquet",
        "0.skimsEmissions.csv.gz",
        "0.skimsEmissionsTotals.csv.gz",
    ]
    for root in [input_root / "skims", input_root]:
        for name in preferred:
            candidate = root / name
            if candidate.exists():
                return str(candidate)

    patterns = [
        "*.skimsEmissions.parquet",
        "*.skimsEmissions.csv.gz",
        "*.skimsEmissionsTotals.csv.gz",
        "skims_from_events.parquet",
        "skims_from_events.csv.gz",
    ]
    for root in [input_root / "skims", input_root]:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(path for path in root.rglob(pattern) if path.is_file())
            if matches:
                return str(matches[0])
    return None


def prepare_staged_skims_for_processing(
    *,
    input_root: Path,
    skims_input_source: str,
    beam_length_col: str,
    prepared_skims_group_cols: list[str],
    pollutants: list[str],
    pollutants_map: Dict[str, str],
    annualization_days: float,
    population_sample: float,
    network_path: Optional[str] = None,
) -> pd.DataFrame:
    resolved_network_path = network_path or _resolve_staged_network_path(input_root)
    prepared_grouped_skims_path = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    prepare_skims_for_grid_allocation(
        skims_path=skims_input_source,
        output_path=str(prepared_grouped_skims_path),
        group_cols=list(prepared_skims_group_cols),
        required_pollutants=list(pollutants),
        pollutants_map=dict(pollutants_map),
    )
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        network_path=resolved_network_path,
        beam_length_col=beam_length_col,
        group_cols=list(prepared_skims_group_cols),
        required_pollutants=list(pollutants),
        annualization_days=float(annualization_days),
        population_sample=float(population_sample),
    )
    return read_table(prepared_skims_path)


def load_or_prepare_skims_df(
    *,
    input_root: Path,
    intersection_path: str,
    beam_length_col: str,
    prepared_skims_group_cols: list[str],
    pollutants: list[str],
    pollutants_map: Dict[str, str],
    annualization_days: float,
    population_sample: float,
    manifest_inputs: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    prepared_path = resolve_prepared_skims_path(input_root)
    if prepared_path:
        logger.info("Step 1: using prepared skims %s", prepared_path)
        return read_table(prepared_path)

    network_path = _resolve_staged_network_path(input_root, manifest_inputs)
    skims_input_source = _resolve_staged_skims_input_path(input_root, manifest_inputs)
    if skims_input_source:
        logger.info("Step 1: preparing skims input %s", skims_input_source)
        return prepare_staged_skims_for_processing(
            input_root=input_root,
            skims_input_source=skims_input_source,
            beam_length_col=beam_length_col,
            prepared_skims_group_cols=prepared_skims_group_cols,
            pollutants=pollutants,
            pollutants_map=pollutants_map,
            annualization_days=annualization_days,
            population_sample=population_sample,
            network_path=network_path,
        )

    from .prepare_emissions_from_events import build_staged_skims_from_events

    events_skims_path = build_staged_skims_from_events(
        input_root=input_root,
        network_path=network_path,
        intersection_path=intersection_path,
        manifest_inputs=manifest_inputs,
    )
    if not events_skims_path:
        raise FileNotFoundError(
            "Could not find skims or events input under the staged input tree "
            f"{input_root}, including {input_root / 'skims'} and {input_root / 'events'}."
        )

    return prepare_staged_skims_for_processing(
        input_root=input_root,
        skims_input_source=events_skims_path,
        beam_length_col=beam_length_col,
        prepared_skims_group_cols=prepared_skims_group_cols,
        pollutants=pollutants,
        pollutants_map=pollutants_map,
        annualization_days=annualization_days,
        population_sample=population_sample,
        network_path=network_path,
    )


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

def prepare_skims_inputs(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    processing,
    skims_input_source: Optional[str],
    network_path: str,
    intersection_path: Optional[str] = None,
) -> dict[str, Any]:
    event_inputs: Optional[Dict[str, Any]] = None
    if not skims_input_source:
        if not intersection_path:
            raise ValueError(
                "intersection_path is required when deriving skims from events."
            )
        logger.info("Step 1: no skims artifact found; building skims from local events")
        from .prepare_emissions_from_events import prepare_events_inputs

        event_inputs = prepare_events_inputs(
            manifest_inputs=manifest_inputs,
            input_root=input_root,
            network_path=network_path,
            intersection_path=intersection_path,
            beam_length_col=processing.beam_length_col,
            prepared_skims_group_cols=list(processing.prepared_skims_group_cols),
            pollutants=list(processing.pollutants),
            pollutants_map=dict(processing.pollutants_map),
            annualization_days=float(processing.annualization_days),
            population_sample=float(processing.population_sample),
        )
        if event_inputs is None:
            raise FileNotFoundError(
                "No skimsEmissions or events file found under the staged input tree."
            )
        skims_input_source = event_inputs["skims_path"]

    staged_skims_input = register_managed_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="emissions_skims_input",
        source_path=skims_input_source,
        relative_target=f"skims/{Path(skims_input_source).name}",
        optional=True,
        prefer_reference=True,
        metadata={
            "workflow_step": "step1_prepare_skims_inputs",
            "artifact_family": "emissions_skims_input",
        },
    )

    prepared_grouped_skims_path = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    canonical_pollutants = list(processing.pollutants)
    prepare_skims_for_grid_allocation(
        skims_path=staged_skims_input,
        output_path=str(prepared_grouped_skims_path),
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=canonical_pollutants,
        pollutants_map=dict(processing.pollutants_map),
    )
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        network_path=network_path,
        beam_length_col=processing.beam_length_col,
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=canonical_pollutants,
        annualization_days=float(processing.annualization_days),
        population_sample=float(processing.population_sample),
    )
    manifest_inputs["prepared_skims_grouped"] = file_entry(
        kind="local",
        path=str(prepared_grouped_skims_path),
        staged_path=str(prepared_grouped_skims_path),
        optional=True,
    )
    manifest_inputs["prepared_skims_input"] = file_entry(
        kind="local",
        path=str(prepared_skims_path),
        staged_path=str(prepared_skims_path),
        optional=True,
    )
    skims_df = event_inputs["skims_df"] if event_inputs is not None else read_table(prepared_skims_path)
    source_activity_df = (
        event_inputs["activity_df"]
        if event_inputs is not None
        else _build_source_activity_totals(skims_df)
    )
    return {
        "staged_skims_input": str(staged_skims_input),
        "prepared_grouped_skims_path": str(prepared_grouped_skims_path),
        "prepared_skims_path": str(prepared_skims_path),
        "skims_df": skims_df,
        "activity_path": None if event_inputs is None else event_inputs["activity_path"],
        "activity_df": source_activity_df,
    }
