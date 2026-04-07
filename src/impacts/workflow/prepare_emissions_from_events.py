"""Helpers for deriving skims inputs from BEAM events.

When no ``skimsEmissions`` file is available, these helpers build a
zone-expanded skims table directly from BEAM PathTraversal events.

Activity columns (per zone, already proportioned by zone edge fraction)
------------------------------------------------------------------------
distanceMiles_{zone}           : float — VMT allocated to zone
travelTimeInSecond_{zone}      : float — VHT allocated to zone
parkingDurationInSecond_{zone} : float — parking duration allocated to zone
tripCount_{zone}               : float — trip count allocated to zone

where zone ∈ {county, aermod, inmap}.

Emission columns (when rates_folder is provided)
-------------------------------------------------
tons_per_year_{pollutant}_{zone}_allocated : float — annual tons per zone

Dimension / enrichment columns
-------------------------------
hour             : int   — hour of day
linkId           : int   — BEAM link identifier
vehicleTypeId    : str   — vehicle type
process          : str   — RUNEX, RUNLOSS, PMBW, PMTW, PRDUST, DIURN, HOTSOAK, STREX, IDLEX
roadCategory     : str   — OSM highway type
countyfp         : str   — county FIPS code
aermod_cell_id : int — AERMOD grid cell id
inmap_cell_id  : int — InMAP/ISRM grid cell id
speedMph         : float — mean link speed (0 for parked processes)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd

from ..config.defaults import grams_per_short_ton
from ..config.defaults import meters_per_mile
from ..config.defaults import parked_processes
from ..manifest.file_ops import file_entry
from ..common import prepared_table_target
from ..common import read_table
from ..common import resolve_manifest_input_path
from ..common import normalize_county_fips

logger = logging.getLogger(__name__)

_EVENTS_USECOLS = [
    "type",
    "vehicle",
    "vehicleType",
    "departureTime",
    "links",
    "linkTravelTime",
    "length",
]


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _parse_list_field(raw: object) -> List[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    s = str(raw).strip()
    return [x.strip() for x in s.split(",") if x.strip()] if s else []


def _parse_float_list(raw: object) -> List[float]:
    out: List[float] = []
    for x in _parse_list_field(raw):
        try:
            out.append(float(x))
        except ValueError:
            out.append(0.0)
    return out


def _read_events(path: str) -> pd.DataFrame:
    p = path.lower()
    compression = "gzip" if p.endswith(".gz") else None
    if p.endswith(".parquet"):
        return pd.read_parquet(path, columns=_EVENTS_USECOLS)
    return pd.read_csv(path, usecols=_EVENTS_USECOLS, compression=compression)



def _build_running_rows(pt: pd.DataFrame) -> List[Dict]:
    rows: List[Dict] = []
    for _, row in pt.iterrows():
        links = [int(x) for x in row["links_list"]]
        times = list(row["times_list"])
        total_tt = sum(times)
        total_len_m = float(row.get("length", 0.0) or 0.0)
        dep = float(row["departureTime"])
        hour = int(dep // 3600)
        vtype = str(row["vehicleType"])

        trip_count_per_link = 1.0 / len(links)
        for link_id, t_sec in zip(links, times):
            dist_miles = (total_len_m * (t_sec / total_tt) if total_tt > 0 else 0.0) / meters_per_mile
            speed_mph = (dist_miles / (t_sec / 3600)) if t_sec > 0 else 0.0
            base = {
                "hour": hour,
                "linkId": link_id,
                "vehicleTypeId": vtype,
                "travelTimeInSecond": float(t_sec),
                "parkingDurationInSecond": 0.0,
                "speedMph": speed_mph,
                "tripCount": trip_count_per_link,
            }
            rows.append({**base, "process": "RUNEX",   "distanceMiles": dist_miles})
            rows.append({**base, "process": "RUNLOSS", "distanceMiles": 0.0})
            rows.append({**base, "process": "PMBW",    "distanceMiles": dist_miles})
            rows.append({**base, "process": "PMTW",    "distanceMiles": dist_miles})
            rows.append({**base, "process": "PRDUST",  "distanceMiles": dist_miles})
    return rows


def _build_parked_rows(pt: pd.DataFrame) -> List[Dict]:
    pt = pt.copy()
    pt["tripDurationSec"] = pt["times_list"].apply(sum)
    pt["tripEndSec"] = pt["departureTime"].astype(float) + pt["tripDurationSec"]
    pt["firstLink"] = pt["links_list"].apply(lambda xs: int(xs[0]))
    pt["lastLink"] = pt["links_list"].apply(lambda xs: int(xs[-1]))
    pt = pt.sort_values(["vehicle", "departureTime"])
    pt["nextDepartureSec"] = pt.groupby("vehicle")["departureTime"].shift(-1).astype(float)
    pt["parkingDurationInSecond"] = (
        pt["nextDepartureSec"].fillna(pt["tripEndSec"]) - pt["tripEndSec"]
    ).clip(lower=0)

    rows: List[Dict] = []
    for _, row in pt.iterrows():
        hour = int(float(row["tripEndSec"]) // 3600)
        base = {
            "hour": hour,
            "vehicleTypeId": str(row["vehicleType"]),
            "travelTimeInSecond": 0.0,
            "parkingDurationInSecond": float(row["parkingDurationInSecond"]),
            "distanceMiles": 0.0,
            "speedMph": 0.0,
            "tripCount": 1.0,
        }
        rows.append({**base, "linkId": int(row["lastLink"]),  "process": "DIURN"})
        rows.append({**base, "linkId": int(row["lastLink"]),  "process": "HOTSOAK"})
        rows.append({**base, "linkId": int(row["lastLink"]),  "process": "IDLEX"})
        rows.append({**base, "linkId": int(row["firstLink"]), "process": "STREX"})
    return rows


def _aggregate(obs_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["hour", "linkId", "vehicleTypeId", "process"]
    grouped = obs_df.groupby(group_cols, dropna=False)
    out = grouped.agg(
        distanceMiles=("distanceMiles", "sum"),
        travelTimeInSecond=("travelTimeInSecond", "sum"),
        parkingDurationInSecond=("parkingDurationInSecond", "sum"),
        speedMph=("speedMph", "mean"),
        tripCount=("tripCount", "sum"),
    ).reset_index()
    return out.sort_values(group_cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

_INTERSECTION_ZONE_COLS = [
    "linkId",
    "countyfp",
    "county_zone_edge_proportion",
    "aermod_cell_id",
    "aermod_zone_edge_proportion",
    "inmap_cell_id",
    "inmap_zone_edge_proportion",
]


_ZONES = ["county", "aermod", "inmap"]

_RATE_BASIS_TO_ACTIVITY = {
    "per_mile": "distanceMiles",
    "per_hour": None,   # resolved per-row: parking vs travel time
    "per_event": "tripCount",
    "per_trip": "tripCount",
}


def _parse_interval_bounds(raw: object) -> tuple[Optional[float], Optional[float], bool]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, False
    text = str(raw).strip()
    if not text:
        return None, None, False
    right_inclusive = text.endswith("]")
    inner = text.strip("[]() ")
    parts = [part.strip() for part in inner.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Unsupported interval format: {raw}")

    def _to_float(value: str) -> Optional[float]:
        lower = value.lower()
        if lower in {"", "inf", "+inf", "infinity", "+infinity"}:
            return None
        if lower in {"-inf", "-infinity"}:
            return None
        return float(value)

    lower = _to_float(parts[0])
    upper = _to_float(parts[1])
    return lower, upper, right_inclusive


def _interval_mask(values: pd.Series, raw_interval: object) -> pd.Series:
    lower, upper, right_inclusive = _parse_interval_bounds(raw_interval)
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna()
    if lower is not None:
        mask &= numeric.ge(lower)
    if upper is not None:
        mask &= numeric.le(upper) if right_inclusive else numeric.lt(upper)
    return mask


def _normalize_rate_dimensions(rates: pd.DataFrame) -> pd.DataFrame:
    result = rates.copy()
    if "countyfp" in result.columns:
        result["countyfp"] = normalize_county_fips(result["countyfp"])
    if "roadCategory" in result.columns:
        result["roadCategory"] = result["roadCategory"].astype("string")
    return result


def _rate_row_mask(result: pd.DataFrame, rate_row: pd.Series, *, is_parked: pd.Series) -> pd.Series:
    mask = result["process"] == str(rate_row["process"])
    if "vehicleTypeId" in rate_row.index and pd.notna(rate_row.get("vehicleTypeId")):
        mask &= result["vehicleTypeId"] == str(rate_row["vehicleTypeId"])
    if "countyfp" in rate_row.index and pd.notna(rate_row.get("countyfp")):
        row_county = str(rate_row["countyfp"]).zfill(3)
        counties = result.get("countyfp", pd.Series(pd.NA, index=result.index)).astype("string")
        counties = counties.str.extract(r"(\d+)")[0].astype("string")
        counties = counties.where(counties.isna(), counties.str.zfill(3))
        mask &= counties == row_county
    if "roadCategory" in rate_row.index and pd.notna(rate_row.get("roadCategory")):
        mask &= result.get("roadCategory", pd.Series(pd.NA, index=result.index)).astype("string") == str(rate_row["roadCategory"])
    if "speed_mph_float_bins" in rate_row.index and pd.notna(rate_row.get("speed_mph_float_bins")):
        mask &= _interval_mask(result.get("speedMph", pd.Series(np.nan, index=result.index)), rate_row["speed_mph_float_bins"])
    if "time_minutes_float_bins" in rate_row.index and pd.notna(rate_row.get("time_minutes_float_bins")):
        minutes = (
            result.get("parkingDurationInSecond", pd.Series(0.0, index=result.index)).fillna(0.0) / 60.0
        ).where(
            is_parked,
            result.get("travelTimeInSecond", pd.Series(0.0, index=result.index)).fillna(0.0) / 60.0,
        )
        mask &= _interval_mask(minutes, rate_row["time_minutes_float_bins"])
    return mask


def _calculate_emissions(
    skims: pd.DataFrame,
    *,
    rates_folder: str,
    pollutants_map: Dict[str, str],
    annualization_days: float,
    population_sample: float,
) -> pd.DataFrame:
    """Join emission rates onto zone-expanded skims and add tons_per_year_{pollutant}_{zone}_allocated columns.

    Args:
        skims: zone-expanded skims with distanceMiles_{zone}, travelTimeInSecond_{zone},
               parkingDurationInSecond_{zone}, and tripCount_{zone} columns.
        rates_folder: directory of emission rate CSV/parquet files.
        pollutants_map: canonical_pollutant → rate_file_pollutant_name mapping.
        annualization_days: representative days per year.
        population_sample: simulation sample fraction; emissions scaled by 1/population_sample.
    """
    from ..tools.beam.events_to_skims_emissions import read_rates_directory

    rates = _normalize_rate_dimensions(read_rates_directory(rates_folder))
    rate_to_canonical = {v: k for k, v in pollutants_map.items()}
    rates = rates[rates["pollutant"].isin(set(pollutants_map.values()))].copy()
    if rates.empty:
        logger.warning("Step 1: no matching rates found in %s for configured pollutants", rates_folder)
        return skims

    rates["canonical_pollutant"] = rates["pollutant"].map(rate_to_canonical)
    rates["rate"] = pd.to_numeric(rates["rate"], errors="coerce").fillna(0.0)
    rates["rate_basis"] = rates["rate_basis"].astype(str).str.lower()

    agg_cols = [
        c
        for c in [
            "vehicleTypeId",
            "process",
            "countyfp",
            "roadCategory",
            "speed_mph_float_bins",
            "time_minutes_float_bins",
            "canonical_pollutant",
            "rate_basis",
        ]
        if c in rates.columns
    ]
    rates_agg = rates.groupby(agg_cols, dropna=False)["rate"].mean().reset_index()

    scale = annualization_days / population_sample / grams_per_short_ton
    is_parked = skims["process"].isin(set(parked_processes))

    result = skims.copy()
    matched_by_pollutant = {
        canonical: pd.Series(False, index=result.index)
        for canonical in pollutants_map
    }
    for canonical in pollutants_map:
        for zone in _ZONES:
            result[f"tons_per_year_{canonical}_{zone}_allocated"] = 0.0

    for _, rate_row in rates_agg.iterrows():
        canonical = str(rate_row["canonical_pollutant"])
        rate = float(rate_row["rate"])
        basis = str(rate_row["rate_basis"]).lower()

        mask = _rate_row_mask(result, rate_row, is_parked=is_parked)
        if not mask.any():
            continue
        matched_by_pollutant[canonical] = matched_by_pollutant[canonical] | mask

        for zone in _ZONES:
            col = f"tons_per_year_{canonical}_{zone}_allocated"
            if basis == "per_mile":
                activity = result.loc[mask, f"distanceMiles_{zone}"].fillna(0.0)
            elif basis == "per_hour":
                park_vals = result.loc[mask, f"parkingDurationInSecond_{zone}"].fillna(0.0) / 3600.0
                tt_vals = result.loc[mask, f"travelTimeInSecond_{zone}"].fillna(0.0) / 3600.0
                activity = park_vals.where(is_parked.loc[mask], tt_vals)
            else:
                activity = result.loc[mask, f"tripCount_{zone}"].fillna(0.0)

            result.loc[mask, col] = result.loc[mask, col] + rate * activity * scale

    sample_cols = [col for col in ["vehicleTypeId", "process", "countyfp", "roadCategory", "speedMph"] if col in result.columns]
    for canonical, matched_mask in matched_by_pollutant.items():
        unmatched = result.loc[~matched_mask]
        if unmatched.empty:
            continue
        sample = unmatched[sample_cols].drop_duplicates().head(10).to_dict(orient="records") if sample_cols else []
        raise ValueError(
            f"Step 1: no equivalent emission rate found for canonical pollutant '{canonical}' "
            f"for {len(unmatched)} skim rows. sample={sample}"
        )

    logger.info(
        "Step 1: calculated emissions for %d canonical pollutants, %d rate entries",
        len(pollutants_map),
        len(rates_agg),
    )
    return result


def _enrich_with_network(skims: pd.DataFrame, network_path: str) -> pd.DataFrame:
    """Join roadCategory (attributeOrigType) from network onto skims by linkId."""
    network = read_table(network_path)
    if not {"linkId", "attributeOrigType"}.issubset(network.columns):
        raise ValueError("Network input must include linkId and attributeOrigType for events-derived skims.")
    network = network[["linkId", "attributeOrigType"]].copy()
    network["linkId"] = pd.to_numeric(network["linkId"], errors="coerce")
    network = network.rename(columns={"attributeOrigType": "roadCategory"})
    return skims.merge(network, how="left", on="linkId")


_ACTIVITY_COLS = ["distanceMiles", "travelTimeInSecond", "parkingDurationInSecond", "tripCount"]
_PROPORTION_PAIRS = [
    ("county_zone_edge_proportion", "county"),
    ("aermod_zone_edge_proportion", "aermod"),
    ("inmap_zone_edge_proportion", "inmap"),
]


def _enrich_with_intersection(skims: pd.DataFrame, intersection_path: str) -> pd.DataFrame:
    """Expand skims by (county, aermod_cell, inmap_cell) using the staged intersection table
    and apply zone proportions to all activity columns.

    distanceMiles, travelTimeInSecond, parkingDurationInSecond are multiplied by each
    zone's edge proportion. speedMph is preserved unchanged (it is a rate, not a total).
    The raw proportion columns are dropped after application.
    """
    available = read_table(intersection_path)[_INTERSECTION_ZONE_COLS].copy()
    available["linkId"] = pd.to_numeric(available["linkId"], errors="coerce")
    prop_cols = [p for p, _ in _PROPORTION_PAIRS]
    for col in prop_cols:
        available[col] = pd.to_numeric(available[col], errors="coerce").fillna(0.0)

    merged = skims.merge(available, how="left", on="linkId")

    for prop_col, zone_label in _PROPORTION_PAIRS:
        prop = merged[prop_col].fillna(0.0)
        for act_col in _ACTIVITY_COLS:
            merged[f"{act_col}_{zone_label}"] = merged[act_col] * prop
        merged = merged.drop(columns=[prop_col])

    merged = merged.drop(columns=_ACTIVITY_COLS)
    return merged


def build_skims_from_events(
    events_path: str,
    *,
    network_path: str,
    intersection_path: str,
    rates_folder: Optional[str] = None,
    pollutants_map: Optional[Dict[str, str]] = None,
    annualization_days: Optional[float] = None,
    population_sample: Optional[float] = None,
) -> pd.DataFrame:
    """Parse PathTraversal events and return a skims DataFrame.
    """
    logger.info("Step 1: reading events from %s", events_path)
    events = _read_events(events_path)
    pt = events[events["type"] == "PathTraversal"].copy()
    pt["links_list"] = pt["links"].apply(_parse_list_field)
    pt["times_list"] = pt["linkTravelTime"].apply(_parse_float_list)
    pt = pt[pt["links_list"].map(len) > 0]
    pt = pt[pt["links_list"].map(len) == pt["times_list"].map(len)]

    running = pd.DataFrame(_build_running_rows(pt))
    parked = pd.DataFrame(_build_parked_rows(pt))
    obs = pd.concat([running, parked], ignore_index=True)

    skims = _aggregate(obs)
    logger.info(
        "Step 1: built %d skims rows from %d PathTraversal events",
        len(skims), len(pt),
    )

    skims = _enrich_with_network(skims, network_path)
    logger.info("Step 1: enriched with roadCategory from network")

    before = len(skims)
    skims = _enrich_with_intersection(skims, intersection_path)
    logger.info(
        "Step 1: expanded %d → %d rows with county/aermod/inmap zones",
        before, len(skims),
    )

    if rates_folder and pollutants_map and annualization_days is not None and population_sample is not None:
        skims = _calculate_emissions(
            skims,
            rates_folder=rates_folder,
            pollutants_map=pollutants_map,
            annualization_days=annualization_days,
            population_sample=population_sample,
        )

    return skims


def build_activity_table(skims: pd.DataFrame) -> pd.DataFrame:
    """Collapse the zone-expanded skims to county activity totals."""
    group_cols = ["countyfp"]
    return (
        skims
        .groupby(group_cols, dropna=False)
        .agg(
            totVMT=("distanceMiles_county", "sum"),
            totTrips=("tripCount_county", "sum"),
        )
        .reset_index()
    )


def find_staged_events_path(input_root: Path, manifest_inputs: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if manifest_inputs and "events_input" in manifest_inputs:
        return resolve_manifest_input_path(manifest_inputs["events_input"], label="events_input")
    search_roots = [input_root / "events", input_root]
    patterns = ["*.events.parquet", "*.events.csv.gz", "*.events.csv"]
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(path for path in root.rglob(pattern) if path.is_file())
            if matches:
                return str(matches[0])
    return None


def _write_staged_skims(skims_df: pd.DataFrame, *, input_root: Path) -> Path:
    skims_dir = input_root / "skims"
    skims_dir.mkdir(parents=True, exist_ok=True)
    skims_path = skims_dir / "skims_from_events.parquet"
    skims_df.to_parquet(skims_path, index=False)
    return skims_path


def build_staged_skims_from_events(
    *,
    input_root: Path,
    network_path: str,
    intersection_path: str,
    manifest_inputs: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    events_path = find_staged_events_path(input_root, manifest_inputs)
    if not events_path:
        return None

    logger.info("Step 1: building skims from staged events %s", events_path)
    skims_df = build_skims_from_events(
        events_path,
        network_path=network_path,
        intersection_path=intersection_path,
    )
    skims_path = _write_staged_skims(skims_df, input_root=input_root)
    logger.info("Step 1: wrote skims derived from events %d rows → %s", len(skims_df), skims_path)
    return str(skims_path)


def prepare_events_inputs(
    *,
    manifest_inputs: Dict,
    input_root: Path,
    network_path: str,
    intersection_path: str,
    beam_length_col: str,
    prepared_skims_group_cols: List[str],
    pollutants: List[str],
    pollutants_map: Dict[str, str],
    annualization_days: float,
    population_sample: float,
) -> Optional[Dict[str, Any]]:
    """Build prepared skims/activity tables from registered BEAM events."""
    events_path = find_staged_events_path(input_root, manifest_inputs)
    if not events_path:
        logger.info("Step 1: no events file found under %s", input_root)
        return None
    logger.info("Step 1: building skims from events %s", events_path)
    skims_df = build_skims_from_events(
        events_path,
        network_path=network_path,
        intersection_path=intersection_path,
    )
    skims_path = _write_staged_skims(skims_df, input_root=input_root)
    logger.info("Step 1: wrote skims derived from events %d rows → %s", len(skims_df), skims_path)
    skims_path = str(skims_path)
    manifest_inputs["skims_from_events"] = file_entry(
        kind="local",
        path=events_path,
        staged_path=skims_path,
        optional=True,
    )

    activity_df = build_activity_table(skims_df)
    skims_dir = input_root / "skims"
    skims_dir.mkdir(parents=True, exist_ok=True)
    activity_path = skims_dir / "activity_from_events.parquet"
    activity_df.to_parquet(activity_path, index=False)
    manifest_inputs["activity_from_events"] = file_entry(
        kind="local",
        path=events_path,
        staged_path=str(activity_path),
        optional=True,
    )
    logger.info("Step 1: wrote activity table %d rows → %s", len(activity_df), activity_path)

    from .prepare_emissions_from_skims import prepare_staged_skims_for_processing

    skims_df = prepare_staged_skims_for_processing(
        input_root=input_root,
        skims_input_source=skims_path,
        beam_length_col=beam_length_col,
        prepared_skims_group_cols=list(prepared_skims_group_cols),
        pollutants=list(pollutants),
        pollutants_map=dict(pollutants_map),
        annualization_days=float(annualization_days),
        population_sample=float(population_sample),
        network_path=network_path,
    )
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")

    return {
        "events_path": events_path,
        "source_skims_path": skims_path,
        "skims_path": str(prepared_skims_path),
        "activity_path": str(activity_path),
        "skims_df": skims_df,
        "activity_df": activity_df,
    }
