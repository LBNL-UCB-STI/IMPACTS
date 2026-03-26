from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Optional

import pandas as pd

from .contract_utils import load_structured_file
from .contract_utils import parquet_available
from .contract_utils import write_structured_file
from .manifest_models import PostprocessManifest
from .manifest_models import RunManifest

logger = logging.getLogger(__name__)


def _read_table(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        return pd.read_parquet(path)
    if lower.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    as_set = set(columns)
    for name in candidates:
        if name in as_set:
            return name
    return None


def _resolve_column_config(config: Optional[Dict[str, str]], defaults: Dict[str, str]) -> Dict[str, str]:
    resolved = defaults.copy()
    if config:
        resolved.update({k: v for k, v in config.items() if v})
    return resolved


def _ordered_unique(items: Iterable[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _population_by_cell(
    persons_path: Optional[str],
    households_path: Optional[str],
    persons_columns: Optional[Dict[str, str]] = None,
    households_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    if not persons_path and not households_path:
        return pd.DataFrame(columns=["cell_id", "population_total", "households_total", "population_mix"])

    households = _read_table(households_path) if households_path else pd.DataFrame()
    persons = _read_table(persons_path) if persons_path else pd.DataFrame()
    persons_cfg = _resolve_column_config(persons_columns, {"household_id": "household_id", "cell_id": "cell_id", "age": "age", "sex": "sex", "income": "income"})
    households_cfg = _resolve_column_config(households_columns, {"household_id": "household_id", "cell_id": "cell_id", "income": "income", "income_category": "income_category"})

    hh_id_col = _first_existing(households.columns, [households_cfg["household_id"], "household_id", "householdId", "hh_id"])
    person_hh_col = _first_existing(persons.columns, [persons_cfg["household_id"], "household_id", "householdId", "hh_id"])
    cell_col = _first_existing(households.columns, [households_cfg["cell_id"], "cell_id", "GRID", "grid", "zone", "home_cell_id", "TAZ"])

    if cell_col is None and _first_existing(persons.columns, [persons_cfg["cell_id"], "cell_id", "GRID", "grid", "zone", "home_cell_id", "TAZ"]):
        cell_col = _first_existing(persons.columns, [persons_cfg["cell_id"], "cell_id", "GRID", "grid", "zone", "home_cell_id", "TAZ"])
        persons = persons.rename(columns={cell_col: "cell_id"})
        cell_col = "cell_id"

    if cell_col is None:
        return pd.DataFrame(columns=["cell_id", "population_total", "households_total", "population_mix"])

    households = households.rename(columns={cell_col: "cell_id"})
    if not persons.empty and person_hh_col and hh_id_col:
        household_extra_cols = [
            c
            for c in _ordered_unique(
                [
                    households_cfg["income"],
                    households_cfg["income_category"],
                    "income",
                    "income_category",
                ]
            )
            if c in households.columns
        ]
        persons = persons.merge(
            households[[hh_id_col, "cell_id"] + household_extra_cols],
            how="left",
            left_on=person_hh_col,
            right_on=hh_id_col,
        )

    household_counts = (
        households.dropna(subset=["cell_id"])
        .groupby("cell_id", dropna=False)
        .size()
        .rename("households_total")
        .reset_index()
    )

    if persons.empty:
        population = household_counts.copy()
        population["population_total"] = 0
        population["population_mix"] = "{}"
        return population[["cell_id", "population_total", "households_total", "population_mix"]]

    persons = persons.dropna(subset=["cell_id"]).copy()
    try:
        persons["cell_id"] = pd.to_numeric(persons["cell_id"])
    except Exception:
        pass
    grouped = persons.groupby("cell_id", dropna=False)

    mix_payload = []
    for cell_id, frame in grouped:
        payload: Dict[str, Any] = {
            "population_total": int(len(frame)),
        }
        age_col = _first_existing(frame.columns, [persons_cfg["age"], "age", "age_years"])
        if age_col:
            age_bins = pd.cut(
                pd.to_numeric(frame[age_col], errors="coerce"),
                bins=[-1, 17, 64, 200],
                labels=["child", "adult", "senior"],
            )
            payload["age_group_counts"] = {
                str(k): int(v) for k, v in age_bins.value_counts(dropna=True).sort_index().items()
            }
        income_col = _first_existing(frame.columns, [persons_cfg["income"], households_cfg["income"], households_cfg["income_category"], "income", "income_category"])
        if income_col:
            payload["income_counts"] = {
                str(k): int(v) for k, v in frame[income_col].fillna("unknown").value_counts().to_dict().items()
            }
        sex_col = _first_existing(frame.columns, [persons_cfg["sex"], "sex", "gender"])
        if sex_col:
            payload["sex_counts"] = {
                str(k): int(v) for k, v in frame[sex_col].fillna("unknown").value_counts().to_dict().items()
            }
        mix_payload.append(
            {
                "cell_id": cell_id,
                "population_total": int(len(frame)),
                "population_mix": json.dumps(payload, sort_keys=True),
            }
        )

    mix_df = pd.DataFrame(mix_payload)
    return household_counts.merge(mix_df, how="outer", on="cell_id").fillna(
        {"population_total": 0, "households_total": 0, "population_mix": "{}"}
    )


def create_canonical_exposure_table(
    *,
    concentration_path: str,
    persons_path: Optional[str] = None,
    households_path: Optional[str] = None,
    concentration_columns: Optional[Dict[str, str]] = None,
    persons_columns: Optional[Dict[str, str]] = None,
    households_columns: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    concentration = _read_table(concentration_path).copy()
    concentration_cfg = _resolve_column_config(concentration_columns, {"grid_id": "GRID"})
    cell_col = _first_existing(concentration.columns, [concentration_cfg["grid_id"], "GRID", "cell_id", "grid", "zone", "Location"])
    if cell_col is None:
        raise ValueError("Raw concentration output must contain a grid/cell identifier column.")

    concentration = concentration.rename(columns={cell_col: "cell_id"})
    concentration["geometry_reference"] = concentration["cell_id"].map(lambda value: f"GRID:{value}")

    pop_by_cell = _population_by_cell(
        persons_path=persons_path,
        households_path=households_path,
        persons_columns=persons_columns,
        households_columns=households_columns,
    )
    merged = concentration.merge(pop_by_cell, how="left", on="cell_id")
    merged["population_total"] = merged["population_total"].fillna(0).astype(int)
    merged["households_total"] = merged["households_total"].fillna(0).astype(int)
    merged["population_mix"] = merged["population_mix"].fillna("{}")
    return merged.sort_values("cell_id").reset_index(drop=True)


def postprocess_from_run_manifest(
    run_manifest_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    run_manifest = RunManifest.from_dict(load_structured_file(run_manifest_path)).to_dict()
    logger.info("Postprocess: loaded run manifest %s", Path(run_manifest_path).resolve())
    raw_outputs = run_manifest.get("raw_outputs", {}) or {}
    concentration_path = raw_outputs.get("grid_concentration")
    if not concentration_path or not Path(concentration_path).exists():
        raise FileNotFoundError("Required raw output missing: grid_concentration")

    output_root = Path(output_dir).resolve()
    canonical_dir = output_root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)

    population_inputs = run_manifest.get("population_inputs", {}) or {}
    pipeline = run_manifest.get("pipeline", {}) or {}
    logger.info("Postprocess: building canonical exposure table from %s", concentration_path)
    canonical = create_canonical_exposure_table(
        concentration_path=concentration_path,
        persons_path=population_inputs.get("persons_path"),
        households_path=population_inputs.get("households_path"),
        concentration_columns=pipeline.get("dispersion_emissions_columns"),
        persons_columns=population_inputs.get("persons_columns"),
        households_columns=population_inputs.get("households_columns"),
    )
    canonical_path = canonical_dir / (
        "impacts_exposure_table.parquet" if parquet_available() else "impacts_exposure_table.csv.gz"
    )
    if canonical_path.suffix == ".parquet":
        canonical.to_parquet(canonical_path, index=False)
    else:
        canonical.to_csv(canonical_path, index=False, compression="gzip")

    postprocess_manifest = {
        "contract_version": run_manifest.get("contract_version", "1"),
        "model": "impacts",
        "run_manifest_path": str(Path(run_manifest_path).resolve()),
        "output_dir": str(output_root),
        "canonical_artifact": {
            "name": "impacts_exposure_table",
            "path": str(canonical_path),
            "rows": int(len(canonical)),
            "columns": list(canonical.columns),
        },
        "validation": {
            "grid_concentration_exists": True,
            "population_inputs_present": bool(
                population_inputs.get("persons_path") or population_inputs.get("households_path")
            ),
        },
        "notes": [
            "Population mix is derived from staged ActivitySim-like tables when those tables carry usable cell ids.",
            "Geometry reference is currently a stable cell identifier placeholder.",
        ],
    }
    output_manifest = Path(manifest_path) if manifest_path else output_root / "postprocess_manifest.yaml"
    postprocess_manifest["postprocess_manifest_path"] = str(output_manifest)
    typed_manifest = PostprocessManifest.from_dict(postprocess_manifest)
    write_structured_file(output_manifest, typed_manifest.to_dict())
    logger.info("Postprocess complete: canonical artifact %s", canonical_path)
    logger.info("Postprocess manifest written: %s", output_manifest)
    return typed_manifest.to_dict()


def postprocess_from_runtime_config(
    runtime_config_path: str | Path,
    workspace: str | Path,
    manifest_path: str | Path | None = None,
) -> Dict[str, Any]:
    from impacts.runner import run_from_runtime_config

    workspace_root = Path(workspace).resolve()
    run_manifest = run_from_runtime_config(
        runtime_config_path=runtime_config_path,
        workspace=workspace_root,
        run_dispersion=True,
    )
    return postprocess_from_run_manifest(
        run_manifest_path=run_manifest["run_manifest_path"],
        output_dir=workspace_root / "output",
        manifest_path=manifest_path,
    )
