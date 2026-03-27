"""Step 5 — InMAP dispersion workflow."""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from .defaults import DEFAULT_CONCENTRATION_FACTOR
from .defaults import DEFAULT_DISPERSION_EMISSIONS_COLUMNS as DEFAULT_EMISSIONS_COLUMNS
from .manifest_models import PipelineConfig

logger = logging.getLogger(__name__)


def _step5_label(step: str) -> str:
    return f"Step 5.{step}"


def _trace_columns(step: str, label: str, columns: list[str]) -> None:
    preview = columns[:20]
    suffix = "" if len(columns) <= 20 else " ..."
    logger.info("%s trace %s columns(%d): %s%s", _step5_label(step), label, len(columns), preview, suffix)


def _trace_frame(step: str, label: str, df: pd.DataFrame, *, key_cols: Optional[list[str]] = None) -> None:
    logger.info("%s trace %s shape=%s", _step5_label(step), label, df.shape)
    _trace_columns(step, label, list(df.columns))
    if key_cols:
        present = [col for col in key_cols if col in df.columns]
        if present and not df.empty:
            sample = df[present].head(5).to_dict(orient="records")
            logger.info("%s trace %s sample_keys=%s", _step5_label(step), label, sample)


def _trace_array(step: str, label: str, values: np.ndarray) -> None:
    arr = np.asarray(values).reshape(-1)
    size = int(arr.shape[0])
    if size == 0:
        logger.info("%s trace %s length=0", _step5_label(step), label)
        return
    logger.info(
        "%s trace %s length=%d min=%s max=%s sum=%s nonzero=%d sample=%s",
        _step5_label(step),
        label,
        size,
        float(arr.min()),
        float(arr.max()),
        float(arr.sum()),
        int(np.count_nonzero(arr)),
        arr[:5].tolist(),
    )


def _read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".csv.gz"):
        return pd.read_csv(path, compression="gzip")
    raise ValueError(f"Unsupported table format: {path}. Use .csv, .csv.gz, or .parquet")


def _load_isrm_store(isrm_url: str):
    import zarr

    if isrm_url.startswith("s3://"):
        try:
            import s3fs  # noqa: F401
        except ImportError as exc:
            raise ImportError("s3fs is required to read ISRM zarr from s3:// URLs") from exc
        return zarr.open(
            isrm_url,
            mode="r",
            storage_options={"anon": True, "client_kwargs": {"region_name": "us-east-2"}},
        )

    return zarr.open(isrm_url, mode="r")


def _prepare_grid_emissions(emissions_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize emissions to ISRM input species and aggregate by grid."""
    _trace_frame("0", "raw_emissions", emissions_df, key_cols=["inmap_srm_cell_id"])
    if "inmap_srm_cell_id" not in emissions_df.columns:
        raise ValueError("No grid id column found. Expected inmap_srm_cell_id")

    emission_cols = [c for c in DEFAULT_EMISSIONS_COLUMNS if c != "inmap_srm_cell_id"]
    source_column_map: dict[str, str] = {}
    for col in emission_cols:
        if col in emissions_df.columns:
            source_column_map[col] = col
            continue
        labeled = f"{col}_inmap_allocated"
        if labeled in emissions_df.columns:
            source_column_map[col] = labeled

    df = emissions_df.copy()
    df["GRID"] = pd.to_numeric(df["inmap_srm_cell_id"], errors="coerce")
    df = df[df["GRID"].notna()].copy()
    df["GRID"] = df["GRID"].astype(int)

    for col in emission_cols:
        source_col = source_column_map.get(col)
        if source_col is None:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0)

    grouped = df.groupby("GRID", dropna=False)[emission_cols].sum().reset_index()
    logger.info("%s trace source_column_map=%s", _step5_label("0"), source_column_map)
    _trace_frame("0", "prepared_grid_emissions", grouped, key_cols=["GRID"])
    return grouped


def _matrix_response(sr, species_key: str, source_cells: np.ndarray, source_values: np.ndarray) -> np.ndarray:
    transfer = sr[species_key].oindex[[0], source_cells, :]
    return transfer[0, :, :].T.dot(source_values)


def _align_emissions_to_isrm_sources(
    emis: pd.DataFrame,
    sr,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Step 5.1: align emissions source cells to the ISRM source index."""
    source_dim = int(sr["SOA"].shape[1])
    receptor_dim = int(sr["SOA"].shape[2])
    raw_source_cells = np.sort(emis["GRID"].unique().astype(int))
    valid_mask = (raw_source_cells >= 0) & (raw_source_cells < source_dim)
    source_cells = raw_source_cells[valid_mask]
    dropped = int(raw_source_cells.size - source_cells.size)
    source_indexed = emis.set_index("GRID").reindex(source_cells).fillna(0.0)
    logger.info(
        "%s aligned emissions to ISRM sources: raw_source_cells=%d kept=%d dropped=%d source_dim=%d",
        _step5_label("1"),
        int(raw_source_cells.size),
        int(source_cells.size),
        dropped,
        source_dim,
    )
    logger.info(
        "%s trace receptor_dim=%d raw_source_head=%s kept_source_head=%s",
        _step5_label("1"),
        receptor_dim,
        raw_source_cells[:10].tolist(),
        source_cells[:10].tolist(),
    )
    _trace_frame("1", "source_indexed", source_indexed.reset_index(), key_cols=["GRID"])
    for col in source_indexed.columns:
        _trace_array("1", f"source_vector[{col}]", source_indexed[col].to_numpy())
    return source_cells, source_indexed


def _compute_species_response(
    *,
    sr,
    species_key: str,
    emissions_key: str,
    source_cells: np.ndarray,
    source_indexed: pd.DataFrame,
) -> np.ndarray:
    started = time.perf_counter()
    source_values = source_indexed[emissions_key].to_numpy()
    logger.info(
        "%s computing %s from %s over %d source cells",
        _step5_label("2"),
        species_key,
        emissions_key,
        int(source_cells.size),
    )
    _trace_array("2", f"{species_key}.source_values", source_values)
    arr = _matrix_response(sr, species_key, source_cells, source_values)
    logger.info(
        "%s computed %s in %.2fs",
        _step5_label("2"),
        species_key,
        time.perf_counter() - started,
    )
    _trace_array("2", f"{species_key}.response", arr)
    return arr


def _read_receptor_vector(sr, key: str, receptor_dim: int) -> np.ndarray:
    values = np.asarray(sr[key][:]).reshape(-1)
    logger.info(
        "%s trace receptor_vector[%s] raw_length=%d expected=%d sample=%s",
        _step5_label("3"),
        key,
        int(values.shape[0]),
        receptor_dim,
        values[:5].tolist(),
    )
    if values.shape[0] < receptor_dim:
        raise ValueError(
            f"{_step5_label('3')} {key} length {values.shape[0]} is shorter than receptor_dim {receptor_dim}"
        )
    return values[:receptor_dim]


def _assemble_concentration_results(
    *,
    sr,
    factor: float,
    include_health: bool,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Step 5.3: assemble concentration outputs on the receptor dimension."""
    receptor_dim = int(sr["SOA"].shape[2])
    logger.info("%s assembling concentration DataFrame for %d receptors", _step5_label("3"), receptor_dim)
    for name, arr in arrays.items():
        logger.info("%s %s response length=%d", _step5_label("3"), name, int(np.asarray(arr).shape[0]))
        _trace_array("3", name, arr)
    bad = {
        name: int(np.asarray(arr).shape[0])
        for name, arr in arrays.items()
        if int(np.asarray(arr).shape[0]) != receptor_dim
    }
    if bad:
        raise ValueError(
            f"{_step5_label('3')} produced mismatched receptor lengths: expected {receptor_dim}, got {bad}"
        )

    results = pd.DataFrame(
        {
            "GRID": np.arange(receptor_dim, dtype=int),
            "SOA": factor * arrays["SOA"],
            "pNO3": factor * arrays["pNO3"],
            "pNH4": factor * arrays["pNH4"],
            "pSO4": factor * arrays["pSO4"],
            "PrimaryPM25": factor * arrays["PrimaryPM25"],
            "BCh": factor * arrays["BCh"],
            "TotalPM25": factor * arrays["TotalPM25"],
        }
    )

    if include_health and "TotalPop" in sr and "MortalityRate" in sr:
        total_pop = _read_receptor_vector(sr, "TotalPop", receptor_dim)
        mortality_rate = _read_receptor_vector(sr, "MortalityRate", receptor_dim)
        results["deathsK"] = (
            (np.exp(np.log(1.06) / 10.0 * results["TotalPM25"]) - 1.0)
            * total_pop
            * 1.096163
            * mortality_rate
            / 100000.0
            * 0.960899254
        )
        results["deathsL"] = (
            (np.exp(np.log(1.14) / 10.0 * results["TotalPM25"]) - 1.0)
            * total_pop
            * 1.096163
            * mortality_rate
            / 100000.0
            * 0.960899254
        )
        _trace_array("3", "TotalPop.trimmed", total_pop)
        _trace_array("3", "MortalityRate.trimmed", mortality_rate)
        _trace_array("3", "deathsK", results["deathsK"].to_numpy())
        _trace_array("3", "deathsL", results["deathsL"].to_numpy())

    _trace_frame("3", "concentrations", results, key_cols=["GRID"])
    return results


def _write_concentration_outputs(concentrations: pd.DataFrame, output_path: str) -> Path:
    """Step 5.4: write concentration outputs."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    logger.info("%s writing concentration outputs to %s", _step5_label("4"), out)
    logger.info(
        "%s trace output_shape=%s suffix=%s columns=%s",
        _step5_label("4"),
        concentrations.shape,
        suffix,
        list(concentrations.columns),
    )
    if suffix == ".parquet":
        concentrations.to_parquet(out, index=False)
    elif suffix == ".gz" and out.name.lower().endswith(".csv.gz"):
        concentrations.to_csv(out, index=False, compression="gzip")
    elif suffix == ".csv":
        concentrations.to_csv(out, index=False)
    else:
        raise ValueError("Output path must end with .parquet, .csv, or .csv.gz")
    logger.info("%s trace write_complete=%s", _step5_label("4"), out)
    return out


def compute_isrm_concentrations(
    *,
    grid_emissions_df: pd.DataFrame,
    sr,
    factor: float,
    include_health: bool,
) -> pd.DataFrame:
    logger.info(
        "%s trace compute_start factor=%s bch_required=%s include_health=%s",
        _step5_label("0"),
        factor,
        True,
        include_health,
    )
    emis = _prepare_grid_emissions(grid_emissions_df)
    if emis.empty:
        raise ValueError("No emissions rows found after GRID normalization.")

    source_cells, source_indexed = _align_emissions_to_isrm_sources(emis, sr)
    if source_cells.size == 0:
        raise ValueError("No emissions source cells remain after ISRM source alignment.")

    species_plan = [
        ("SOA", "tons_per_year_ROG", "SOA"),
        ("pNO3", "tons_per_year_NOx", "pNO3"),
        ("pNH4", "tons_per_year_NH3", "pNH4"),
        ("pSO4", "tons_per_year_SOx", "pSO4"),
        ("PrimaryPM25", "tons_per_year_PM2_5", "PrimaryPM25"),
    ]
    arrays: dict[str, np.ndarray] = {}
    for species_key, emissions_key, result_key in tqdm(
        species_plan,
        desc="Step 5.2 pollutant responses",
        unit="species",
    ):
        arrays[result_key] = _compute_species_response(
            sr=sr,
            species_key=species_key,
            emissions_key=emissions_key,
            source_cells=source_cells,
            source_indexed=source_indexed,
        )

    arrays["TotalPM25"] = (
        arrays["SOA"] + arrays["pNO3"] + arrays["pNH4"] + arrays["pSO4"] + arrays["PrimaryPM25"]
    )
    _trace_array("2", "TotalPM25.pre_bc", arrays["TotalPM25"])
    arrays["BCh"] = _compute_species_response(
        sr=sr,
        species_key="PrimaryPM25",
        emissions_key="tons_per_year_BCh",
        source_cells=source_cells,
        source_indexed=source_indexed,
    )
    arrays["TotalPM25"] = arrays["TotalPM25"] + arrays["BCh"]
    _trace_array("2", "TotalPM25.post_bc", arrays["TotalPM25"])

    return _assemble_concentration_results(
        sr=sr,
        factor=factor,
        include_health=include_health,
        arrays=arrays,
    )


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    emissions_input_path: str,
    output_path: str,
) -> pd.DataFrame:
    """Run InMAP dispersion from corrected inMAP grid emissions."""
    del raw_dir

    if not pipeline.isrm_url:
        raise ValueError(
            "isrm_url must be configured. "
            "Set dispersions.inmap.isrm_zarr (or isrm_zarr_directory / isrm_zarr_s3bucket) in runtime.yaml."
        )

    logger.info("%s loading corrected grid emissions from %s", _step5_label("0"), emissions_input_path)
    emissions_df = _read_table(emissions_input_path)
    _trace_frame("0", "loaded_emissions", emissions_df, key_cols=["inmap_srm_cell_id"])

    logger.info("%s loading ISRM store from %s", _step5_label("0"), pipeline.isrm_url)
    sr = _load_isrm_store(pipeline.isrm_url)
    logger.info(
        "%s trace isrm_shapes SOA=%s TotalPop=%s MortalityRate=%s",
        _step5_label("0"),
        getattr(sr["SOA"], "shape", None),
        getattr(sr["TotalPop"], "shape", None) if "TotalPop" in sr else None,
        getattr(sr["MortalityRate"], "shape", None) if "MortalityRate" in sr else None,
    )

    concentrations = compute_isrm_concentrations(
        grid_emissions_df=emissions_df,
        sr=sr,
        factor=float(pipeline.concentration_factor or DEFAULT_CONCENTRATION_FACTOR),
        include_health=bool(pipeline.include_health),
    )
    _write_concentration_outputs(concentrations, output_path)
    return concentrations
