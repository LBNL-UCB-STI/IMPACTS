from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from impacts.emfac.config import read_table
from impacts.emfac.config import resolve_workflow_path
from impacts.emfac.config import ATLAS_VEHICLES_SCHEMA
from impacts.emfac.config import VEHICLE_CATEGORY_METADATA_SCHEMA


LIGHT_DUTY_VEHICLE_CATEGORIES = {"LDA", "LDT1", "LDT2"}
VEHICLE_TYPE_ID_HASH_HEX_LENGTH = 6


def _read_full_input_table(path_like: str) -> pd.DataFrame:
    resolved = Path(resolve_workflow_path(path_like))
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved, low_memory=False)


def read_atlas_vehicles_input(path_like: str) -> pd.DataFrame:
    vehicles = _read_full_input_table(path_like)
    prepared = vehicles.copy()

    for column_name, dtype_name in ATLAS_VEHICLES_SCHEMA.items():
        if column_name not in prepared.columns:
            raise ValueError(f"ATLAS vehicles file is missing required column '{column_name}'")
        if dtype_name == "string":
            prepared[column_name] = prepared[column_name].astype("string")
        elif dtype_name == "Float64":
            prepared[column_name] = pd.to_numeric(prepared[column_name], errors="raise").astype("Float64")
        elif dtype_name == "Int64":
            prepared[column_name] = pd.to_numeric(prepared[column_name], errors="raise").astype("Int64")
        else:
            raise ValueError(f"Unsupported ATLAS vehicles dtype '{dtype_name}' for column '{column_name}'")
    return prepared


def read_frism_carriers_input(path_like: str) -> pd.DataFrame:
    carriers = _read_full_input_table(path_like)
    for column_name in ["tourId", "vehicleTypeId"]:
        if column_name not in carriers.columns:
            raise ValueError(f"FRISM carriers file is missing required column '{column_name}'")
    prepared = carriers.copy()
    prepared["tourId"] = prepared["tourId"].astype("string")
    prepared["vehicleTypeId"] = prepared["vehicleTypeId"].astype("string")
    return prepared


def normalize_probabilities_to_fixed_precision(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    weight_column: str,
    output_column: str,
    decimals: int = 6,
) -> pd.DataFrame:
    prepared = frame.copy()
    if prepared.empty:
        prepared[output_column] = pd.Series(dtype="float64")
        return prepared

    scale = 10**decimals
    values = pd.to_numeric(prepared[weight_column], errors="coerce").fillna(0.0)
    prepared[weight_column] = values
    prepared[output_column] = 0.0

    if group_columns:
        grouping_key: str | list[str]
        grouping_key = group_columns[0] if len(group_columns) == 1 else group_columns
        group_iterator = prepared.groupby(grouping_key, dropna=False, sort=False).groups.items()
    else:
        group_iterator = [((), prepared.index)]

    for _, group_index in group_iterator:
        group_positions = list(group_index)
        group_weights = prepared.loc[group_positions, weight_column].to_numpy(dtype=float)
        total_weight = float(group_weights.sum())
        if total_weight <= 0.0:
            prepared.loc[group_positions, output_column] = 0.0
            continue
        raw_scaled = group_weights / total_weight * scale
        base_units = np.floor(raw_scaled).astype(int)
        remainder_units = int(scale - base_units.sum())
        if remainder_units > 0:
            fractional_units = raw_scaled - base_units
            allocation_order = np.argsort(-fractional_units, kind="mergesort")
            base_units[allocation_order[:remainder_units]] += 1
        elif remainder_units < 0:
            fractional_units = raw_scaled - base_units
            allocation_order = np.argsort(fractional_units, kind="mergesort")
            for idx in allocation_order:
                if remainder_units == 0:
                    break
                if base_units[idx] <= 0:
                    continue
                base_units[idx] -= 1
                remainder_units += 1
        prepared.loc[group_positions, output_column] = base_units.astype(float) / scale
    return prepared


def build_hashed_vehicle_type_ids(
    frame: pd.DataFrame,
    *,
    frame_name: str,
    source_column: str = "vehicleTypeId",
    mapping_column: str = "mappingVehicleTypeId",
    prefix: str,
    hash_hex_length: int = VEHICLE_TYPE_ID_HASH_HEX_LENGTH,
) -> pd.DataFrame:
    prepared = frame.copy()
    if source_column not in prepared.columns:
        raise ValueError(f"{frame_name} is missing required column '{source_column}'")
    if prepared[source_column].isna().any():
        raise ValueError(f"{frame_name} contains null values in required column '{source_column}'")

    prepared[mapping_column] = prepared[source_column].astype(str)
    seen_short_to_long: dict[str, str] = {}
    hashed_ids: list[str] = []

    for mapping_vehicle_type_id in prepared[mapping_column].tolist():
        mapping_vehicle_type_id = str(mapping_vehicle_type_id)
        digest = hashlib.sha256(mapping_vehicle_type_id.encode("utf-8")).hexdigest()
        hashed_vehicle_type_id = f"{prefix}-{digest[:hash_hex_length]}"
        existing = seen_short_to_long.get(hashed_vehicle_type_id)
        if existing is not None and existing != mapping_vehicle_type_id:
            raise ValueError(
                f"{frame_name} hash collision detected for "
                f"{mapping_vehicle_type_id} and {existing} using prefix={prefix} length={hash_hex_length}"
            )
        seen_short_to_long[hashed_vehicle_type_id] = mapping_vehicle_type_id
        hashed_ids.append(hashed_vehicle_type_id)

    if len(set(hashed_ids)) != len(hashed_ids):
        raise ValueError(f"{frame_name} generated duplicate hashed {source_column} values")

    prepared[source_column] = hashed_ids
    return prepared


def ensure_trace_dir(workflow: dict[str, object]) -> Path:
    trace_dir = Path(workflow["paths"]["trace_dir"]).expanduser()
    trace_dir.mkdir(parents=True, exist_ok=True)
    return trace_dir


def write_trace(workflow: dict[str, object], name: str, payload: dict[str, Any]) -> Path:
    target = ensure_trace_dir(workflow) / f"{name}.json"
    with target.open("w") as handle:
        json.dump(_to_json_ready(payload), handle, indent=2, sort_keys=True)
    return target


def write_failure_trace(
    workflow: dict[str, object],
    *,
    step: str,
    error: Exception,
    payload: dict[str, Any] | None = None,
) -> Path:
    failure_payload: dict[str, Any] = {
        "step": step,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
    }
    if payload:
        failure_payload.update(payload)
    return write_trace(workflow, f"{step}_failure", failure_payload)


def raise_runtime_error(step: str, error: Exception) -> None:
    raise RuntimeError(f"EMFAC workflow failed during {step}: {type(error).__name__}: {error}") from error


def frame_summary(frame: pd.DataFrame, *, name: str) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": name,
        "row_count": int(len(frame)),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
    }
    if "process" in frame.columns:
        summary["process_counts"] = {
            str(key): int(value)
            for key, value in frame["process"].astype(str).value_counts(dropna=False).sort_index().items()
        }
    if "pollutant" in frame.columns:
        summary["pollutant_counts"] = {
            str(key): int(value)
            for key, value in frame["pollutant"].astype(str).value_counts(dropna=False).sort_index().items()
        }
    if "county" in frame.columns:
        summary["county_count"] = int(frame["county"].nunique(dropna=True))
    return summary


def _to_json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_model_year_group(group: dict[str, object] | str) -> dict[str, object]:
    if isinstance(group, dict):
        return group
    token = str(group).strip()
    if not token:
        raise ValueError("Configured model_year_groups entries must not be empty.")
    if token.isdigit() and len(token) == 4:
        year = int(token)
        return {"min_year": year, "max_year": year}
    if token.startswith("<="):
        return {"max_year": int(token[2:].strip())}
    if token.startswith(">="):
        return {"min_year": int(token[2:].strip())}
    if "-" in token:
        min_year, max_year = token.split("-", 1)
        return {"min_year": int(min_year.strip()), "max_year": int(max_year.strip())}
    raise ValueError(
        "Configured model_year_groups string entries must use one of: '<=YYYY', 'YYYY-YYYY', '>=YYYY'."
    )


def model_year_group_string(group: dict[str, object] | str) -> str:
    if isinstance(group, str):
        token = str(group).strip()
        if not token:
            raise ValueError("Configured model_year_groups entries must not be empty.")
        _normalize_model_year_group(token)
        return token
    normalized = _normalize_model_year_group(group)
    min_year = normalized.get("min_year")
    max_year = normalized.get("max_year")
    if min_year is None:
        return f"<={int(max_year)}"
    if max_year is None:
        return f">={int(min_year)}"
    if int(min_year) == int(max_year):
        return f"{int(min_year)}"
    return f"{int(min_year)}-{int(max_year)}"


def model_year_group_id_component(group: dict[str, object] | str) -> str:
    group = _normalize_model_year_group(group)
    min_year = group.get("min_year")
    max_year = group.get("max_year")
    if min_year is None:
        return f"pre{int(max_year) + 1:02d}"
    if max_year is None:
        return f"post{int(min_year) - 1:02d}"
    if int(min_year) == int(max_year):
        return f"{int(min_year)}"
    return f"{int(min_year)}to{int(max_year)}"


def parse_model_year_group_interval(group: dict[str, object] | str) -> tuple[float, float]:
    normalized = _normalize_model_year_group(group)
    min_year = normalized.get("min_year")
    max_year = normalized.get("max_year")
    left = float("-inf") if min_year is None else float(int(min_year))
    right = float("inf") if max_year is None else float(int(max_year))
    return left, right


def model_year_interval_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    a_min, a_max = a
    b_min, b_max = b
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def vehicle_group(vehicle_category: object) -> str:
    vehicle_category = str(vehicle_category).strip()
    if vehicle_category in LIGHT_DUTY_VEHICLE_CATEGORIES:
        return "light_duty"
    return "medium_heavy_duty"


def _require_column(frame: pd.DataFrame, column_name: str, frame_name: str) -> None:
    if column_name not in frame.columns:
        raise ValueError(f"{frame_name} is missing required column '{column_name}'")


def load_idle_time_fraction_lookup(metadata_path: str) -> dict[str, float]:
    metadata = read_table(metadata_path, schema=VEHICLE_CATEGORY_METADATA_SCHEMA)
    required = {"emfac_vehicle_category", "idle_time_fraction"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(
            "Vehicle category metadata file is missing required columns for idleTimeFraction: "
            f"{missing}"
        )
    prepared = metadata[["emfac_vehicle_category", "idle_time_fraction"]].copy()
    prepared["emfac_vehicle_category"] = prepared["emfac_vehicle_category"].fillna("").str.strip()
    prepared = prepared.loc[
        prepared["emfac_vehicle_category"].ne("") & prepared["idle_time_fraction"].notna()
    ].copy()
    return (
        prepared.drop_duplicates(subset=["emfac_vehicle_category"], keep="first")
        .set_index("emfac_vehicle_category")["idle_time_fraction"]
        .to_dict()
    )


def attach_idle_time_fraction(
    vehicle_types: pd.DataFrame,
    *,
    idle_time_fraction_lookup: dict[str, float],
) -> pd.DataFrame:
    prepared = vehicle_types.copy()
    if "emfacVehicleCategory" not in prepared.columns:
        raise ValueError(
            "Vehicle types table is missing required column 'emfacVehicleCategory' for idleTimeFraction"
        )
    categories = prepared["emfacVehicleCategory"].fillna("").astype(str).str.strip()

    prepared["idleTimeFraction"] = categories.map(idle_time_fraction_lookup)
    prepared.loc[categories.ne("") & prepared["idleTimeFraction"].isna(), "idleTimeFraction"] = 0.0
    return prepared


def attach_idle_time_fraction_from_config(
    vehicle_types: pd.DataFrame,
    *,
    config: dict[str, Any],
    step_label: str,
) -> pd.DataFrame:
    metadata_path = config.get("vehicle_category_attributes_file")
    if not metadata_path:
        raise ValueError(f"{step_label} requires vehicle_category_attributes_file in the EMFAC config.")
    return attach_idle_time_fraction(
        vehicle_types,
        idle_time_fraction_lookup=load_idle_time_fraction_lookup(str(metadata_path)),
    )


def load_rates_store_relative_paths(
    *,
    config: dict[str, Any],
    scenario: object,
    output_root: str | Path,
    required_emfac_ids: list[str],
    require_duckdb: bool = False,
) -> dict[str, str]:
    activities_output_root = Path(str(config["activities"]["outputs"])).expanduser().resolve()
    frism_year = config["frism"]["year"]
    scenario_name = str(scenario).strip()
    store_root = activities_output_root / "emissions" / f"{frism_year}-{scenario_name}"
    parquet_root = store_root / "dataset"
    duckdb_path = store_root / "dataset.duckdb"
    output_root_path = Path(str(output_root)).expanduser().resolve()

    relative_paths: dict[str, str] = {}
    for partition_dir in sorted(parquet_root.glob("emfacId=*")):
        if not partition_dir.is_dir():
            continue
        emfac_id = partition_dir.name.removeprefix("emfacId=")
        parquet_path = partition_dir / f"{emfac_id}.parquet"
        if parquet_path.exists():
            relative_paths[emfac_id] = os.path.relpath(parquet_path, output_root_path)

    missing_store_ids = [emfac_id for emfac_id in required_emfac_ids if emfac_id and emfac_id not in relative_paths]
    if missing_store_ids:
        preview = ", ".join(missing_store_ids[:10])
        raise FileNotFoundError(
            "EMFAC rates store is missing parquet partitions for required emfacId values: "
            f"{preview}{' ...' if len(missing_store_ids) > 10 else ''}"
        )
    if require_duckdb and not duckdb_path.exists():
        raise FileNotFoundError(f"EMFAC rates store database not found: {duckdb_path}")
    return relative_paths


def attach_emissions_rates_filepaths(
    vehicle_types: pd.DataFrame,
    *,
    relative_paths: dict[str, str],
) -> pd.DataFrame:
    prepared = vehicle_types.copy()
    if "emfacId" not in prepared.columns:
        raise ValueError("Vehicle types table is missing required column 'emfacId' for emissionsRatesFile.")
    emfac_ids = prepared["emfacId"].fillna("").astype(str)
    prepared["emfacId"] = emfac_ids
    prepared["emissionsRatesFile"] = emfac_ids.map(relative_paths).fillna("")
    return prepared


def attach_emissions_rates_filepaths_from_config(
    vehicle_types: pd.DataFrame,
    *,
    config: dict[str, Any],
    scenario: object,
    output_root: str | Path,
    step_label: str,
) -> pd.DataFrame:
    if "emfacId" not in vehicle_types.columns:
        raise ValueError(f"{step_label} requires vehicle types with required column 'emfacId'.")
    required_emfac_ids = sorted(
        [
            str(emfac_id).strip()
            for emfac_id in vehicle_types["emfacId"].fillna("").astype(str).unique().tolist()
            if str(emfac_id).strip()
        ]
    )
    relative_paths = load_rates_store_relative_paths(
        config=config,
        scenario=scenario,
        output_root=output_root,
        required_emfac_ids=required_emfac_ids,
    )
    try:
        return attach_emissions_rates_filepaths(
            vehicle_types,
            relative_paths=relative_paths,
        )
    except Exception as error:
        raise ValueError(f"{step_label} could not attach emissionsRatesFile from the EMFAC rates store: {error}") from error


def assign_model_year_groups(
    frame: pd.DataFrame,
    model_year_groups: dict[str, list[dict[str, object] | str]],
    *,
    year_column: str = "modelYear",
    category_column: str = "vehicleCategory",
    output_column: str | None = None,
) -> pd.DataFrame:
    result = frame.copy()
    target_column = output_column or year_column
    labels = pd.Series(pd.NA, index=result.index, dtype="object")
    model_year = pd.to_numeric(result[year_column], errors="raise").astype(int)
    vehicle_groups = result[category_column].map(vehicle_group)
    for current_group, groups in model_year_groups.items():
        group_mask = vehicle_groups == current_group
        if not group_mask.any():
            continue
        for group in groups:
            group = _normalize_model_year_group(group)
            min_year = group.get("min_year")
            max_year = group.get("max_year")
            mask = group_mask.copy()
            if min_year is not None:
                mask &= model_year >= int(min_year)
            if max_year is not None:
                mask &= model_year <= int(max_year)
            labels.loc[mask] = model_year_group_string(group)
    if labels.isna().any():
        missing_rows = result.loc[labels.isna(), [category_column, year_column]].drop_duplicates()
        raise ValueError(
            "Some vehicleCategory/modelYear rows are not covered by the configured model_year_groups: "
            f"{missing_rows.to_dict(orient='records')[:20]}"
        )
    result[target_column] = labels
    return result
