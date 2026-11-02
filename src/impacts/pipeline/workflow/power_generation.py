"""Charging utilities and direct plant-emission ISRM multiplication.

Hourly power orchestration is implemented in hourly_power.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Polygon

from ...common import read_table
from ...config.defaults import tons_per_year_to_ug_per_s
from .step2_compute_inmap_concentrations import _load_isrm_store

logger = logging.getLogger(__name__)

KG_PER_SHORT_TON = 907.18474
BEAM_NON_VEHICLE_TYPES = ("BODY-TYPE-DEFAULT",)
BEAM_CHARGER_POWER_KW = {"L1": 1.9, "L2": 7.2, "_F": 50.0}

POWER_COMPONENTS = {
    "VOC_kg": ("SOA", "power_SOA"),
    "NOx_kg": ("pNO3", "power_pNO3"),
    "NH3_kg": ("pNH4", "power_pNH4"),
    "SOx_kg": ("pSO4", "power_pSO4"),
    "PM2_5_kg": ("PrimaryPM25", "power_PrimaryPM25"),
}


def allocate_charging_event(
    profile_kwh: np.ndarray,
    *,
    start_hour: float,
    duration_hours: float,
    energy_kwh: float,
) -> None:
    """Energy-conserving allocation of a constant-power session to hour bins."""
    if duration_hours <= 0 or energy_kwh < 0:
        raise ValueError("Charging duration must be positive and energy nonnegative.")
    mean_kw = energy_kwh / duration_hours
    end_hour = start_hour + duration_hours
    left = start_hour
    while left < end_hour - 1e-12:
        hour = int(np.floor(left))
        right = min(end_hour, hour + 1.0)
        profile_kwh[hour % len(profile_kwh)] += mean_kw * (right - left)
        left = right


def _read_beam_charging_events(path: str | Path) -> pd.DataFrame:
    """Read only the BEAM columns needed by the events-only charging adapter."""
    columns = [
        "type", "vehicle", "vehicleType", "mode", "primaryFuelType",
        "departureTime", "arrivalTime", "length",
    ]
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BEAM charging events file does not exist: {path}")
    if str(path).lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path, columns=columns)
    compression = "gzip" if str(path).lower().endswith(".gz") else None
    return pd.read_csv(path, usecols=columns, compression=compression, low_memory=False)


def _weighted_choice_without_replacement(
    rng: np.random.Generator,
    positions: np.ndarray,
    weights: np.ndarray,
    count: int,
) -> np.ndarray:
    if count <= 0 or positions.size == 0:
        return np.empty(0, dtype=np.int64)
    count = min(int(count), int(positions.size))
    clean = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, 0.0, None)
    probabilities = clean / clean.sum() if clean.sum() > 0 else None
    return np.asarray(rng.choice(positions, size=count, replace=False, p=probabilities), dtype=np.int64)


def _safe_profile_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_")
    return token or "UNKNOWN"


def write_charging_curve_outputs(demand: pd.DataFrame, output_dir: Path,
                                 scenario_label: str = 'all simulated vehicle types electric') -> Path:
    """Write human-readable hourly demand tables and a mean daily curve."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    demand.to_csv(output_dir / "sfbay_hourly_charging_demand.csv", index=False)
    daily = demand.copy()
    daily["hour_of_day"] = pd.to_numeric(daily["hour_of_year"], errors="raise") % 24
    value_columns = [
        column for column in daily.columns
        if column.startswith("charging_demand_") and column.endswith("_MWh")
    ]
    daily = daily.groupby("hour_of_day", as_index=False)[value_columns].mean()
    daily.to_csv(output_dir / "sfbay_mean_daily_charging_curve.csv", index=False)

    hourly_ranges = (
        demand.assign(hour_of_day=pd.to_numeric(demand["hour_of_year"], errors="raise") % 24)
        .groupby("hour_of_day")["charging_demand_MWh"]
        .agg(lambda values: float(values.max() - values.min()))
    )
    repeated_representative_day = bool(np.allclose(hourly_ranges.to_numpy(), 0.0))
    curve_label = "Representative-day" if repeated_representative_day else "Mean daily"
    y_label = "Charging demand (MW)" if repeated_representative_day else "Mean charging demand (MW)"

    fig, ax = plt.subplots(figsize=(10, 6))
    total_column = "charging_demand_MWh"
    component_columns = [column for column in value_columns if column != total_column]
    # Large BEAM runs can contain dozens of vehicle types. Keep every type in
    # the CSV/Parquet outputs, but make the static figure readable by plotting
    # the eight largest annual contributors and aggregating the remainder.
    ranked_components = sorted(
        component_columns,
        key=lambda column: float(demand[column].sum()),
        reverse=True,
    )
    plotted_components = ranked_components[:8]
    remainder_components = ranked_components[8:]
    for column in plotted_components:
        label = column.removeprefix("charging_demand_").removesuffix("_MWh").replace("_", " ")
        ax.plot(daily["hour_of_day"], daily[column], linewidth=1.5, alpha=0.8, label=label)
    if remainder_components:
        ax.plot(
            daily["hour_of_day"],
            daily[remainder_components].sum(axis=1),
            linewidth=1.7,
            linestyle="--",
            alpha=0.85,
            label=f"Other ({len(remainder_components)} types)",
        )
    ax.plot(
        daily["hour_of_day"], daily[total_column], color="black", linewidth=2.8, label="Total"
    )
    ax.set(
        title=(
            f"{curve_label} BEAM-derived SF Bay charging demand\n"
            f"({scenario_label})"
        ),
        xlabel="Hour of day",
        ylabel=y_label,
        xticks=range(0, 24, 2),
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    output_path = output_dir / "sfbay_mean_daily_charging_curve.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def build_beam_hourly_charging_demand(
    events_path: str | Path,
    *,
    force_all_vehicle_types_electric: bool = False,
    seed: int = 42,
    energy_kwh_per_km: float = 0.17,
    default_battery_kwh: float = 75.0,
    min_parking_minutes: float = 15.0,
    one_charge_share: float = 0.45,
    two_charge_share: float = 0.04,
    home_charging_share: float = 0.80,
    minimum_allowed_kwh: float = 2.0,
    critical_soc_kwh: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Build an annual charging profile directly from BEAM PathTraversal events.

    This is the events-only adapter for the charging notebook.  In the absence
    of plans, charger inventory, and CAISO inputs, the longest parking interval
    is treated as home, home has L1/L2 access, public stops have L1/L2/DCFC
    access, and the grid-load weight is neutral.
    """
    if energy_kwh_per_km <= 0 or default_battery_kwh <= 0:
        raise ValueError("BEAM charging energy rate and default battery capacity must be positive.")
    if min_parking_minutes < 0 or not 0 <= one_charge_share <= 1 or not 0 <= two_charge_share <= 1:
        raise ValueError("Invalid BEAM charging parking/session parameters.")
    if one_charge_share + two_charge_share > 1:
        raise ValueError("BEAM one-charge and two-charge shares cannot exceed one in total.")

    raw = _read_beam_charging_events(events_path)
    paths = raw.loc[raw["type"].astype(str).eq("PathTraversal")].copy()
    # BEAM represents a walking person as a BODY-TYPE-DEFAULT pseudo-vehicle.
    # Exclude only that sentinel so force-all-electric genuinely covers every
    # simulated vehicle type, including road and fixed-route transit vehicles.
    paths = paths.loc[
        ~paths["vehicleType"].astype(str).isin(BEAM_NON_VEHICLE_TYPES)
    ]
    if not force_all_vehicle_types_electric:
        paths = paths.loc[paths["primaryFuelType"].astype(str).str.casefold().eq("electricity")]
    paths = paths.loc[paths["vehicle"].notna()].copy()
    for column in ("departureTime", "arrivalTime", "length"):
        paths[column] = pd.to_numeric(paths[column], errors="coerce")
    paths = paths.dropna(subset=["departureTime", "arrivalTime", "length"])
    paths = paths.loc[paths["length"].ge(0)].copy()
    if paths.empty:
        raise ValueError("No eligible BEAM vehicle PathTraversal events remained for charging.")
    paths["vehicle"] = paths["vehicle"].astype(str)
    paths["vehicleType"] = paths["vehicleType"].fillna("UNKNOWN").astype(str)
    paths = paths.sort_values(["vehicle", "departureTime", "arrivalTime"], kind="stable").reset_index(drop=True)
    paths["trip_energy_kwh"] = paths["length"] / 1000.0 * float(energy_kwh_per_km)
    paths["daily_energy_kwh"] = paths.groupby("vehicle")["trip_energy_kwh"].transform("sum")
    paths["cumulative_energy_kwh"] = paths.groupby("vehicle")["trip_energy_kwh"].cumsum()

    next_departure = paths.groupby("vehicle")["departureTime"].shift(-1)
    first_departure = paths.groupby("vehicle")["departureTime"].transform("first")
    next_departure = next_departure.fillna(first_departure + 24.0 * 3600.0)
    parking_seconds = (next_departure - paths["arrivalTime"]).to_numpy(dtype=float)
    # Adjacent PathTraversal segments often have exactly zero dwell. Never turn
    # those into a full-day charging opportunity; negative gaps are not parking
    # either. Multi-day itineraries need a separate cycle-aware model.
    parking_seconds = np.maximum(parking_seconds, 0.0)
    paths["parkingTime_s"] = parking_seconds

    vehicle_info = paths.groupby("vehicle", sort=False).agg(
        vehicleType=("vehicleType", "first"),
        daily_energy_kwh=("daily_energy_kwh", "first"),
    )
    vehicle_info["battery_kwh"] = float(default_battery_kwh)
    relative_daily_range = vehicle_info["daily_energy_kwh"] / vehicle_info["battery_kwh"]
    alpha = 1.99 + 48.05 * relative_daily_range
    beta_param = 2.77 + 9.30 * relative_daily_range
    rng = np.random.default_rng(int(seed))
    vehicle_info["initial_soc_kwh"] = (
        rng.beta(alpha.to_numpy(dtype=float), beta_param.to_numpy(dtype=float))
        * vehicle_info["battery_kwh"].to_numpy(dtype=float)
    )
    paths = paths.join(vehicle_info[["battery_kwh", "initial_soc_kwh"]], on="vehicle")
    paths["projected_soc_kwh"] = paths["initial_soc_kwh"] - paths["cumulative_energy_kwh"]

    candidates = paths.loc[paths["parkingTime_s"].ge(float(min_parking_minutes) * 60.0)].copy()
    if candidates.empty:
        raise ValueError("No BEAM parking intervals satisfy the minimum charging duration.")
    home_indices = candidates.groupby("vehicle")["parkingTime_s"].idxmax()
    paths["is_home_inferred"] = False
    paths.loc[home_indices, "is_home_inferred"] = True
    candidates = paths.loc[candidates.index].copy()

    minimum_soc = paths.groupby("vehicle")["projected_soc_kwh"].min().reindex(vehicle_info.index)
    vehicle_info["minimum_projected_soc_kwh"] = minimum_soc
    vehicle_info["charging_need"] = (
        1.0 - vehicle_info["minimum_projected_soc_kwh"] / vehicle_info["battery_kwh"]
    ).clip(lower=1e-9)
    critical = vehicle_info["minimum_projected_soc_kwh"].lt(float(critical_soc_kwh))
    noncritical_positions = np.flatnonzero(~critical.to_numpy())
    n1 = min(int(round(one_charge_share * len(vehicle_info))), len(noncritical_positions))
    selected_one = _weighted_choice_without_replacement(
        rng,
        noncritical_positions,
        vehicle_info["charging_need"].to_numpy()[noncritical_positions],
        n1,
    )
    remaining = np.setdiff1d(noncritical_positions, selected_one, assume_unique=False)
    n2 = min(int(round(two_charge_share * len(vehicle_info))), len(remaining))
    two_weights = (
        vehicle_info["charging_need"]
        * candidates.groupby("vehicle").size().reindex(vehicle_info.index, fill_value=0)
    ).to_numpy()[remaining]
    selected_two = _weighted_choice_without_replacement(rng, remaining, two_weights, n2)
    planned_counts = pd.Series(0, index=vehicle_info.index, dtype=np.int8)
    planned_counts.iloc[selected_one] = 1
    planned_counts.iloc[selected_two] = 2
    home_selected = pd.Series(
        rng.random(len(vehicle_info)) < float(home_charging_share), index=vehicle_info.index
    )

    total_daily_profile = np.zeros(24, dtype=float)
    profiles_by_type: dict[str, np.ndarray] = {}
    session_records: list[dict[str, Any]] = []
    discarded = 0
    vehicles_charged = 0
    candidate_groups = {
        vehicle: group for vehicle, group in candidates.groupby("vehicle", sort=False)
    }
    empty_candidates = candidates.iloc[0:0]

    for vehicle, vehicle_paths in paths.groupby("vehicle", sort=False):
        info = vehicle_info.loc[vehicle]
        vehicle_candidates = candidate_groups.get(vehicle, empty_candidates)
        selected: dict[int, str] = {}
        if bool(critical.loc[vehicle]) and not vehicle_candidates.empty:
            options = vehicle_candidates.copy()
            options["charger"] = np.where(options["is_home_inferred"], "L2", "_F")
            options["capacity_kwh"] = (
                options["parkingTime_s"] / 3600.0
                * options["charger"].map(BEAM_CHARGER_POWER_KW)
            )
            options = options.sort_values("capacity_kwh", ascending=False)
            deficit = max(float(critical_soc_kwh) - float(info["minimum_projected_soc_kwh"]), 0.0)
            take = int(np.searchsorted(options["capacity_kwh"].cumsum().to_numpy(), deficit) + 1)
            for idx, row in options.iloc[:take].iterrows():
                selected[int(idx)] = str(row["charger"])
        else:
            need = int(planned_counts.loc[vehicle])
            pool = vehicle_candidates
            if need == 1 and not pool.empty:
                desired_home = bool(home_selected.loc[vehicle])
                restricted = pool.loc[pool["is_home_inferred"].eq(desired_home)]
                if not restricted.empty:
                    pool = restricted
            for _ in range(min(need, len(pool))):
                option_rows: list[tuple[int, str, float]] = []
                for idx, row in pool.iterrows():
                    gap = max(float(row["battery_kwh"] - row["projected_soc_kwh"]), 1e-3)
                    hours = float(row["parkingTime_s"]) / 3600.0
                    types = ("L1", "L2") if bool(row["is_home_inferred"]) else ("L1", "L2", "_F")
                    for charger in types:
                        fill = hours * BEAM_CHARGER_POWER_KW[charger] / gap
                        score = float(np.exp(-((fill - 1.0) ** 2))) * gap
                        if bool(row["is_home_inferred"]):
                            score *= 0.20 if charger == "L1" else 0.80
                        option_rows.append((int(idx), charger, score))
                if not option_rows:
                    break
                option_weights = np.asarray([item[2] for item in option_rows], dtype=float)
                chosen_pos = int(_weighted_choice_without_replacement(
                    rng, np.arange(len(option_rows)), option_weights, 1
                )[0])
                chosen_idx, charger, _ = option_rows[chosen_pos]
                selected[chosen_idx] = charger
                pool = pool.drop(index=chosen_idx)

        local_profile = np.zeros(24, dtype=float)
        local_sessions: list[dict[str, Any]] = []
        soc = float(info["initial_soc_kwh"])
        minimum_actual_soc = soc
        for idx, row in vehicle_paths.iterrows():
            soc -= float(row["trip_energy_kwh"])
            minimum_actual_soc = min(minimum_actual_soc, soc)
            charger = selected.get(int(idx))
            if charger and soc < float(info["battery_kwh"]):
                power_kw = BEAM_CHARGER_POWER_KW[charger]
                delivered = min(
                    power_kw * float(row["parkingTime_s"]) / 3600.0,
                    float(info["battery_kwh"]) - soc,
                )
                if delivered > 0:
                    duration_hours = delivered / power_kw
                    allocate_charging_event(
                        local_profile,
                        start_hour=float(row["arrivalTime"]) / 3600.0,
                        duration_hours=duration_hours,
                        energy_kwh=delivered,
                    )
                    soc += delivered
                    local_sessions.append({
                        "vehicle": vehicle,
                        "vehicleType": str(info["vehicleType"]),
                        "time": float(row["arrivalTime"]) % (24.0 * 3600.0),
                        "charge_time_minutes": duration_hours * 60.0,
                        "energy_needed_kwh": delivered,
                        "charger_type": charger,
                        "is_home_inferred": bool(row["is_home_inferred"]),
                    })
            minimum_actual_soc = min(minimum_actual_soc, soc)
        if minimum_actual_soc < float(minimum_allowed_kwh):
            discarded += 1
            continue
        if local_sessions:
            vehicles_charged += 1
        total_daily_profile += local_profile
        type_token = _safe_profile_token(info["vehicleType"])
        profiles_by_type.setdefault(type_token, np.zeros(24, dtype=float))
        profiles_by_type[type_token] += local_profile
        session_records.extend(local_sessions)

    yearly_total = np.tile(total_daily_profile / 1000.0, 365)
    demand = pd.DataFrame({"hour_of_year": np.arange(8760, dtype=np.int64)})
    for token, profile in sorted(profiles_by_type.items()):
        demand[f"charging_demand_{token}_MWh"] = np.tile(profile / 1000.0, 365)
    demand["charging_demand_MWh"] = yearly_total
    sessions = pd.DataFrame.from_records(session_records)
    summary = {
        "charging_demand_source": "beam_events",
        "source_geography": "BEAM events (debug; coordinates are not used for plant allocation)",
        "target_geography": "SFBAY power plants",
        "events_file": str(Path(events_path).resolve()),
        "force_all_vehicle_types_electric": bool(force_all_vehicle_types_electric),
        "eligible_modes": sorted(paths["mode"].dropna().astype(str).unique().tolist()),
        "eligible_vehicle_types": {
            str(k): int(v) for k, v in paths.groupby("vehicleType")["vehicle"].nunique().items()
        },
        "eligible_vehicles": int(len(vehicle_info)),
        "vehicles_charged": int(vehicles_charged),
        "discarded_vehicles_below_minimum_soc": int(discarded),
        "charging_sessions": int(len(sessions)),
        "daily_energy_MWh": float(total_daily_profile.sum() / 1000.0),
        "annual_energy_MWh": float(yearly_total.sum()),
        "annual_peak_MW": float(yearly_total.max()),
        "assumptions": {
            "energy_kwh_per_km": float(energy_kwh_per_km),
            "default_battery_kwh": float(default_battery_kwh),
            "min_parking_minutes": float(min_parking_minutes),
            "one_charge_share": float(one_charge_share),
            "two_charge_share": float(two_charge_share),
            "home_charging_share": float(home_charging_share),
            "home_definition": "longest eligible parking interval per vehicle",
            "charger_availability": "home=L1/L2; non-home=L1/L2/DCFC",
            "grid_load_weight": "neutral (CAISO input unavailable)",
        },
    }
    return demand, summary, sessions


def _ground_cell_count(sr: Any) -> int:
    # W/S/E/N describe receptors/source cells once; matrix source dimensions use
    # the same ground-grid cell count.
    return int(np.asarray(sr["W"]).shape[0])


def _map_plants_to_source_cells(plants: pd.DataFrame, sr: Any) -> pd.DataFrame:
    n_cells = _ground_cell_count(sr)
    w, s, e, n = (np.asarray(sr[key][:n_cells]) for key in ("W", "S", "E", "N"))
    transformer = Transformer.from_crs("EPSG:4326", sr.attrs.get("crs", "+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=33 +lat_2=45 +ellps=sphere +units=m"), always_xy=True)
    x, y = transformer.transform(plants["LON"].to_numpy(), plants["LAT"].to_numpy())
    area = (e - w) * (n - s)
    cell_ids = np.full(len(plants), -1, dtype=np.int64)
    for idx, (px, py) in enumerate(zip(x, y)):
        candidates = np.flatnonzero((w <= px) & (px <= e) & (s <= py) & (py <= n))
        if candidates.size:
            cell_ids[idx] = int(candidates[np.argmin(area[candidates])])
    result = plants.copy()
    result["isrm_source_cell"] = cell_ids
    return result


def _weighted_response(matrix: Any, weights: pd.Series, layer_index: int, receptors: np.ndarray) -> np.ndarray:
    result = np.zeros(len(receptors), dtype=np.float64)
    weights = weights[weights.ne(0)]
    chunk = int(matrix.chunks[1]) if getattr(matrix, "chunks", None) else max(len(weights), 1)
    for chunk_no in sorted(set((weights.index.to_numpy(dtype=int) // chunk).tolist())):
        start = chunk_no * chunk
        stop = min(start + chunk, int(matrix.shape[1]))
        local = weights[(weights.index >= start) & (weights.index < stop)]
        dense = np.zeros(stop - start, dtype=np.float64)
        dense[local.index.to_numpy(dtype=int) - start] = local.to_numpy(dtype=float)
        if hasattr(matrix, "oindex"):
            block = matrix.oindex[layer_index, slice(start, stop), receptors]
        else:
            block = np.asarray(matrix[layer_index, start:stop, :])[:, receptors]
        result += dense @ np.asarray(block, dtype=np.float32)
    return result


def compute_power_generation_isrm(
    plants: pd.DataFrame,
    *,
    isrm_url: str,
    receptor_cells: np.ndarray,
    source_layer_index: int = 1,
) -> pd.DataFrame:
    """Compute power-generation concentration at IMPACTS receptors.

    Defaults to elevated layer 1; layer 0 is an explicit temporary alternative.
    Compact stores must cover every emitting source cell; missing rows are errors.
    """
    missing = {"LAT", "LON", *POWER_COMPONENTS}.difference(plants.columns)
    if missing:
        raise ValueError(f"Power-plant emissions are missing columns: {sorted(missing)}")
    sr = _load_isrm_store(isrm_url)
    layers = np.asarray(sr["layers"][:], dtype=float) if "layers" in sr else np.array([0.0])
    if source_layer_index not in (0, 1) or source_layer_index >= len(layers):
        raise ValueError(f"ISRM source layer {source_layer_index} is unavailable; layers={layers.tolist()}")
    if source_layer_index == 1 and not layers[1] > layers[0]:
        raise ValueError(
            "ISRM source layer index 1 must be the first layer above ground. "
            f"ISRM layers={layers.tolist()}"
        )
    mapped = _map_plants_to_source_cells(plants, sr)
    emissions = mapped[list(POWER_COMPONENTS)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(emissions.to_numpy()).all() or (emissions < 0).any().any():
        raise ValueError("Plant emissions must be finite and nonnegative.")
    valid = mapped.loc[emissions.gt(0).any(axis=1)].copy()
    if valid["isrm_source_cell"].lt(0).any():
        raise ValueError("Some emitting plants fall outside the ISRM source grid.")
    source_ids = np.asarray(sr["source_ids"][:], dtype=np.int64) if "source_ids" in sr else np.arange(sr["PrimaryPM25"].shape[1])
    if len(np.unique(source_ids)) != len(source_ids):
        raise ValueError("Duplicate ISRM source_ids.")
    absent = np.setdiff1d(valid["isrm_source_cell"].unique(), source_ids)
    if len(absent):
        raise ValueError(f"ISRM matrix is missing {len(absent)} emitting source cells: {absent.tolist()}. Cannot compute complete power concentration.")
    source_index = pd.Series(np.arange(len(source_ids)), index=source_ids)
    valid["matrix_source_index"] = valid["isrm_source_cell"].map(source_index)
    result = pd.DataFrame({"inmap_cell_id": np.asarray(receptor_cells, dtype=np.int64)})
    for emission_col, (matrix_name, output_col) in POWER_COMPONENTS.items():
        tons = pd.to_numeric(valid[emission_col], errors="coerce").fillna(0.0)
        tons = tons / KG_PER_SHORT_TON
        if sr[matrix_name].shape[1] != len(source_ids) or source_layer_index >= sr[matrix_name].shape[0]:
            raise ValueError(f"Inconsistent source axis/layer shape for {matrix_name}.")
        weights = tons.groupby(valid["matrix_source_index"]).sum()
        result[output_col] = _weighted_response(
            sr[matrix_name], weights, source_layer_index, np.asarray(receptor_cells, dtype=np.int64)
        ) * float(tons_per_year_to_ug_per_s)
    result["power_SecondaryPM25"] = result[["power_SOA", "power_pNO3", "power_pNH4", "power_pSO4"]].sum(axis=1)
    result["power_TotalPM25"] = result["power_PrimaryPM25"] + result["power_SecondaryPM25"]
    return result


def plot_power_generation_concentration(
    concentrations: gpd.GeoDataFrame,
    plants: pd.DataFrame,
    output_path: Path,
    *,
    source_layer_index: int = 1,
    method_note: str | None = None,
) -> Path:
    """Plot power-generation-only PM2.5 with emitting plant locations."""
    import matplotlib.pyplot as plt

    column = "power_TotalPM25"
    values = pd.to_numeric(concentrations[column], errors="coerce")
    positive = values.loc[values.gt(0)]
    if positive.empty:
        raise ValueError("Power-generation ISRM output contains no positive PM2.5 concentrations.")

    plant_frame = plants.copy()
    emission_columns = [
        name for name in ("PM2_5_kg", "NOx_kg", "SOx_kg", "NH3_kg", "VOC_kg")
        if name in plant_frame.columns
    ]
    if emission_columns:
        emitting = (
            plant_frame[emission_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .gt(0.0)
            .any(axis=1)
        )
        plant_frame = plant_frame.loc[emitting].copy()
    plant_frame["LON"] = pd.to_numeric(plant_frame["LON"], errors="coerce")
    plant_frame["LAT"] = pd.to_numeric(plant_frame["LAT"], errors="coerce")
    plant_frame = plant_frame.dropna(subset=["LON", "LAT"])
    plant_points = gpd.GeoDataFrame(
        plant_frame,
        geometry=gpd.points_from_xy(plant_frame["LON"], plant_frame["LAT"]),
        crs="EPSG:4326",
    ).to_crs(concentrations.crs)

    vmax = float(positive.max())
    fig, ax = plt.subplots(figsize=(12, 11))
    concentrations.assign(**{column: values}).plot(
        column=column,
        ax=ax,
        cmap="RdYlGn_r",
        vmin=0.0,
        vmax=vmax,
        linewidth=0.08,
        edgecolor=(0.2, 0.2, 0.2, 0.16),
        alpha=0.86,
        zorder=1,
    )
    xmin, ymin, xmax, ymax = concentrations.total_bounds
    visible_plants = plant_points.cx[xmin:xmax, ymin:ymax]
    if not visible_plants.empty:
        visible_plants.plot(
            ax=ax,
            marker="^",
            color="#1976d2",
            edgecolor="white",
            linewidth=0.5,
            markersize=30,
            label="Emitting power plant",
            zorder=3,
        )
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_facecolor("#f4f2ed")
    ax.set_title(
        "SF Bay Power-Generation ISRM Concentration\n"
        + ("Ground source layer 0" if source_layer_index == 0 else f"Elevated source layer {source_layer_index}")
        + "; no local-to-local removal"
        + (f"\n{method_note}" if method_note else ""),
        fontsize=18,
        pad=14,
    )
    ax.set_axis_off()
    scalar = plt.cm.ScalarMappable(
        cmap="RdYlGn_r", norm=plt.Normalize(vmin=0.0, vmax=vmax)
    )
    scalar.set_array([])
    colorbar = fig.colorbar(scalar, ax=ax, fraction=0.035, pad=0.025, shrink=0.78)
    colorbar.set_label("Power-generation total PM2.5 concentration (µg/m³)", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
