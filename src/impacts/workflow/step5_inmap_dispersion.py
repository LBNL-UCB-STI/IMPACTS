"""Step 5 — InMAP concentration workflow."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..config.defaults import DEFAULT_TONS_PER_YEAR_TO_UG_PER_S
from ..config.defaults import DEFAULT_DISPERSION_EMISSIONS_COLUMNS as DEFAULT_EMISSIONS_COLUMNS
from ..manifest.schema import PipelineConfig

logger = logging.getLogger(__name__)


def _step_label(step_num: int, step: str) -> str:
    return f"Step {step_num}.{step}"


def _trace_columns(step: str, label: str, columns: list[str]) -> None:
    preview = columns[:20]
    suffix = "" if len(columns) <= 20 else " ..."
    logger.info("%s trace %s columns(%d): %s%s", _step_label(5, step), label, len(columns), preview, suffix)


def _trace_frame(step: str, label: str, df: pd.DataFrame, *, key_cols: Optional[list[str]] = None) -> None:
    logger.info("%s trace %s shape=%s", _step_label(5, step), label, df.shape)
    _trace_columns(step, label, list(df.columns))
    if key_cols:
        present = [col for col in key_cols if col in df.columns]
        if present and not df.empty:
            sample = df[present].head(5).to_dict(orient="records")
            logger.info("%s trace %s sample_keys=%s", _step_label(5, step), label, sample)


def _trace_array(step: str, label: str, values: np.ndarray) -> None:
    arr = np.asarray(values).reshape(-1)
    size = int(arr.shape[0])
    if size == 0:
        logger.info("%s trace %s length=0", _step_label(5, step), label)
        return
    logger.info(
        "%s trace %s length=%d min=%s max=%s sum=%s nonzero=%d sample=%s",
        _step_label(5, step),
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


def _read_vector(path: str) -> gpd.GeoDataFrame:
    target = Path(path)
    if target.suffix.lower() == ".parquet":
        return gpd.read_parquet(target)
    return gpd.read_file(target)


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


def _read_rdata(path: str) -> dict[str, object]:
    try:
        import pyreadr
    except ImportError as exc:
        raise ImportError("pyreadr is required to read NOx/NO2 RData inputs") from exc
    return pyreadr.read_r(path)


def _read_square_transfer_matrix(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pd.read_parquet(path)
    elif lower.endswith(".csv"):
        df = pd.read_csv(path, index_col=0)
    elif lower.endswith(".rdata"):
        rdata = _read_rdata(path)
        if "res.dat" in rdata:
            df = rdata["res.dat"].copy()
        elif "res" in rdata:
            df = rdata["res"].copy()
        else:
            df = next(iter(rdata.values())).copy()
    else:
        raise ValueError(f"Unsupported NOx transfer matrix format: {path}")

    df.index = pd.to_numeric(pd.Index(df.index), errors="coerce")
    df.columns = pd.to_numeric(pd.Index(df.columns), errors="coerce")
    df = df.loc[df.index.notna(), df.columns.notna()].copy()
    df.index = df.index.astype(int)
    df.columns = df.columns.astype(int)
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def _build_no2_transfer_matrix(
    *,
    isrm_nox_to_no2_matrix_path: Optional[str],
) -> Optional[pd.DataFrame]:
    if isrm_nox_to_no2_matrix_path:
        matrix = _read_square_transfer_matrix(isrm_nox_to_no2_matrix_path)
        logger.info(
            "%s loaded NOx->NO2 transfer matrix from %s shape=%s",
            _step_label(5, "2"),
            isrm_nox_to_no2_matrix_path,
            matrix.shape,
        )
        return matrix

    return None


def _compute_custom_receptor_response(
    *,
    transfer_matrix: pd.DataFrame,
    emissions_key: str,
    source_cells: np.ndarray,
    source_indexed: pd.DataFrame,
    receptor_dim: int,
    result_key: str,
) -> np.ndarray:
    started = time.perf_counter()
    source_values = source_indexed[emissions_key].to_numpy()
    logger.info(
        "%s computing %s from %s over %d source cells using custom transfer matrix",
        _step_label(5, "2"),
        result_key,
        emissions_key,
        int(source_cells.size),
    )
    _trace_array("2", f"{result_key}.source_values", source_values)
    aligned = transfer_matrix.reindex(index=source_cells, fill_value=0.0)
    response_subset = aligned.to_numpy().T.dot(source_values)
    response = np.zeros(receptor_dim, dtype=float)
    receptor_ids = np.asarray(aligned.columns, dtype=int)
    valid = (receptor_ids >= 0) & (receptor_ids < receptor_dim)
    response[receptor_ids[valid]] = response_subset[valid]
    logger.info(
        "%s computed %s in %.2fs",
        _step_label(5, "2"),
        result_key,
        time.perf_counter() - started,
    )
    _trace_array("2", f"{result_key}.response", response)
    return response


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
    logger.info("%s trace source_column_map=%s", _step_label(5, "0"), source_column_map)
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
        _step_label(5, "1"),
        int(raw_source_cells.size),
        int(source_cells.size),
        dropped,
        source_dim,
    )
    logger.info(
        "%s trace receptor_dim=%d raw_source_head=%s kept_source_head=%s",
        _step_label(5, "1"),
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
        _step_label(5, "2"),
        species_key,
        emissions_key,
        int(source_cells.size),
    )
    _trace_array("2", f"{species_key}.source_values", source_values)
    arr = _matrix_response(sr, species_key, source_cells, source_values)
    logger.info(
        "%s computed %s in %.2fs",
        _step_label(5, "2"),
        species_key,
        time.perf_counter() - started,
    )
    _trace_array("2", f"{species_key}.response", arr)
    return arr


def _compute_no2_response(
    *,
    sr,
    source_cells: np.ndarray,
    source_indexed: pd.DataFrame,
    receptor_dim: int,
    isrm_nox_to_no2_matrix_path: Optional[str],
) -> Optional[np.ndarray]:
    logger.info("%s resolving NO2 source from ISRM zarr or configured fallback matrix", _step_label(5, "1"))
    if "NO2" in sr:
        logger.info("%s using NO2 transfer matrix from ISRM zarr", _step_label(5, "1"))
        return _compute_species_response(
            sr=sr,
            species_key="NO2",
            emissions_key="tons_per_year_NOx",
            source_cells=source_cells,
            source_indexed=source_indexed,
        )

    no2_transfer_matrix = _build_no2_transfer_matrix(
        isrm_nox_to_no2_matrix_path=isrm_nox_to_no2_matrix_path,
    )
    if no2_transfer_matrix is None:
        logger.info(
            "%s NO2 skipped: ISRM zarr has no NO2 and no ISRM NOx->NO2 fallback matrix was configured",
            _step_label(5, "1"),
        )
        return None

    logger.info("%s using configured ISRM NOx->NO2 fallback transfer matrix", _step_label(5, "1"))
    return _compute_custom_receptor_response(
        transfer_matrix=no2_transfer_matrix,
        emissions_key="tons_per_year_NOx",
        source_cells=source_cells,
        source_indexed=source_indexed,
        receptor_dim=receptor_dim,
        result_key="NO2",
    )


def _has_no2_fallback_matrix(
    *,
    isrm_nox_to_no2_matrix_path: Optional[str],
) -> bool:
    return bool(isrm_nox_to_no2_matrix_path)


def _warn_missing_requested_concentrations(
    *,
    requested_pollutants: list[str],
    sr,
    isrm_nox_to_no2_matrix_path: Optional[str],
) -> None:
    species_requirements = {
        "ROG": ["SOA"],
        "NH3": ["pNH4"],
        "SOx": ["pSO4"],
        "PM2_5": ["PrimaryPM25"],
        "BCh": ["PrimaryPM25"],
        "NOx": ["pNO3"],
    }
    requested = [pollutant for pollutant in requested_pollutants if pollutant]
    for pollutant in requested:
        for species_key in species_requirements.get(pollutant, []):
            if species_key not in sr:
                logger.warning(
                    "%s requested pollutant %s is missing required concentration source %s in ISRM zarr",
                    _step_label(5, "1"),
                    pollutant,
                    species_key,
                )

        if pollutant == "NOx":
            has_no2 = "NO2" in sr or _has_no2_fallback_matrix(
                isrm_nox_to_no2_matrix_path=isrm_nox_to_no2_matrix_path,
            )
            if not has_no2:
                logger.warning(
                    "%s requested pollutant NOx has no NO2 concentration source: ISRM zarr has no NO2 and "
                    "no isrm_nox_to_no2_matrix fallback was configured",
                    _step_label(5, "1"),
                )


def _read_receptor_vector(sr, key: str, receptor_dim: int) -> np.ndarray:
    values = np.asarray(sr[key][:]).reshape(-1)
    logger.info(
        "%s trace receptor_vector[%s] raw_length=%d expected=%d sample=%s",
        _step_label(5, "3"),
        key,
        int(values.shape[0]),
        receptor_dim,
        values[:5].tolist(),
    )
    if values.shape[0] < receptor_dim:
        raise ValueError(
            f"{_step_label(5, '3')} {key} length {values.shape[0]} is shorter than receptor_dim {receptor_dim}"
        )
    return values[:receptor_dim]


def _assemble_concentration_results(
    *,
    sr,
    factor: float,
    arrays: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Step 5.3: assemble concentration outputs on the receptor dimension."""
    receptor_dim = int(sr["SOA"].shape[2])
    logger.info("%s assembling concentration DataFrame for %d receptors", _step_label(5, "3"), receptor_dim)
    for name, arr in arrays.items():
        logger.info("%s %s response length=%d", _step_label(5, "3"), name, int(np.asarray(arr).shape[0]))
        _trace_array("3", name, arr)
    bad = {
        name: int(np.asarray(arr).shape[0])
        for name, arr in arrays.items()
        if int(np.asarray(arr).shape[0]) != receptor_dim
    }
    if bad:
        raise ValueError(
            f"{_step_label(5, '3')} produced mismatched receptor lengths: expected {receptor_dim}, got {bad}"
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
    if "NO2" in arrays:
        results["NO2"] = factor * arrays["NO2"]

    _trace_frame("3", "concentrations", results, key_cols=["GRID"])
    return results


def _build_beam_inmap_concentrations_gdf(
    *,
    concentrations: pd.DataFrame,
    inmap_grid_path: str,
    grid_id_col: str,
    included_grid_ids: np.ndarray,
) -> gpd.GeoDataFrame:
    logger.info(
        "%s building BEAM InMAP concentrations GeoDataFrame from %s using grid_id_col=%s",
        _step_label(5, "4"),
        inmap_grid_path,
        grid_id_col,
    )
    grid = _read_vector(inmap_grid_path)
    if grid_id_col not in grid.columns:
        raise ValueError(
            f"{_step_label(5, '4')} grid id column '{grid_id_col}' not found in {inmap_grid_path}. "
            f"Available columns: {list(grid.columns)}"
        )
    grid = grid.rename(columns={grid_id_col: "GRID"}).copy()
    grid["GRID"] = pd.to_numeric(grid["GRID"], errors="raise").astype(int)
    included_grid_ids = np.asarray(included_grid_ids, dtype=int)
    included_grid_set = set(included_grid_ids.tolist())
    grid = grid[grid["GRID"].isin(included_grid_set)].copy()
    concentration_cols = [col for col in concentrations.columns if col != "GRID" and col not in grid.columns]
    merged = grid.merge(concentrations[["GRID"] + concentration_cols], how="left", on="GRID")
    for col in concentration_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=grid.crs)


def _write_concentration_outputs(
    beam_inmap_concentrations_gdf: gpd.GeoDataFrame,
    output_path: str,
) -> tuple[Path, Path]:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() != ".parquet":
        raise ValueError("Output path must end with .parquet for Step 5 BEAM InMAP concentrations")
    gpkg_path = out.with_suffix(".gpkg")
    logger.info("%s writing BEAM InMAP concentrations GeoParquet to %s", _step_label(5, "5"), out)
    beam_inmap_concentrations_gdf.to_parquet(out, index=False)
    logger.info("%s writing BEAM InMAP concentrations GPKG to %s", _step_label(5, "5"), gpkg_path)
    beam_inmap_concentrations_gdf.to_file(gpkg_path, driver="GPKG")
    return out, gpkg_path


def compute_isrm_concentrations(
    *,
    grid_emissions_df: pd.DataFrame,
    sr,
    factor: float,
    request_no2: bool = True,
    isrm_nox_to_no2_matrix_path: Optional[str] = None,
) -> pd.DataFrame:
    logger.info(
        "%s trace compute_start factor=%s bch_required=%s",
        _step_label(5, "0"),
        factor,
        True,
    )
    emis = _prepare_grid_emissions(grid_emissions_df)
    if emis.empty:
        raise ValueError("No emissions rows found after GRID normalization.")

    source_cells, source_indexed = _align_emissions_to_isrm_sources(emis, sr)
    if source_cells.size == 0:
        raise ValueError("No emissions source cells remain after ISRM source alignment.")
    receptor_dim = int(sr["SOA"].shape[2])

    species_plan = [
        ("SOA", "tons_per_year_ROG", "SOA"),
        ("pNO3", "tons_per_year_NOx", "pNO3"),
        ("pNH4", "tons_per_year_NH3", "pNH4"),
        ("pSO4", "tons_per_year_SOx", "pSO4"),
        ("PrimaryPM25", "tons_per_year_PM2_5", "PrimaryPM25"),
    ]
    arrays: dict[str, np.ndarray] = {}
    with logging_redirect_tqdm():
        for species_key, emissions_key, result_key in tqdm(
            species_plan,
            desc="Step 5.2 pollutant responses",
            unit="species",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
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

    if request_no2:
        no2_response = _compute_no2_response(
            sr=sr,
            source_cells=source_cells,
            source_indexed=source_indexed,
            receptor_dim=receptor_dim,
            isrm_nox_to_no2_matrix_path=isrm_nox_to_no2_matrix_path,
        )
        if no2_response is not None:
            arrays["NO2"] = no2_response

    return _assemble_concentration_results(
        sr=sr,
        factor=factor,
        arrays=arrays,
    )


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    emissions_input_path: str,
) -> tuple[gpd.GeoDataFrame, np.ndarray, Path]:
    """Step 5: compute and export receptor-side InMAP concentrations from BEAM InMAP emissions."""
    if not pipeline.isrm_url:
        raise ValueError(
            "isrm_url must be configured. "
            "Set dispersions.inmap.isrm_zarr (or isrm_zarr_directory / isrm_zarr_s3bucket) in runtime.yaml."
        )

    logger.info("%s loading BEAM emissions for InMAP from %s", _step_label(5, "0"), emissions_input_path)
    emissions_df = _read_table(emissions_input_path)
    _trace_frame("0", "loaded_emissions", emissions_df, key_cols=["inmap_srm_cell_id"])
    beam_inmap_grid_ids = (
        pd.to_numeric(emissions_df["inmap_srm_cell_id"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
    )
    logger.info(
        "%s trace beam_inmap_grid_ids count=%d sample=%s",
        _step_label(5, "0"),
        int(beam_inmap_grid_ids.shape[0]),
        beam_inmap_grid_ids[:10].tolist(),
    )

    logger.info("%s loading ISRM store from %s", _step_label(5, "0"), pipeline.isrm_url)
    sr = _load_isrm_store(pipeline.isrm_url)
    logger.info(
        "%s trace isrm_shapes SOA=%s TotalPop=%s MortalityRate=%s",
        _step_label(5, "0"),
        getattr(sr["SOA"], "shape", None),
        getattr(sr["TotalPop"], "shape", None) if "TotalPop" in sr else None,
        getattr(sr["MortalityRate"], "shape", None) if "MortalityRate" in sr else None,
    )
    _warn_missing_requested_concentrations(
        requested_pollutants=pipeline.pollutants,
        sr=sr,
        isrm_nox_to_no2_matrix_path=pipeline.isrm_nox_to_no2_matrix_path,
    )

    concentrations = compute_isrm_concentrations(
        grid_emissions_df=emissions_df,
        sr=sr,
        factor=float(pipeline.concentration_factor or DEFAULT_TONS_PER_YEAR_TO_UG_PER_S),
        request_no2="NOx" in set(pipeline.pollutants),
        isrm_nox_to_no2_matrix_path=pipeline.isrm_nox_to_no2_matrix_path,
    )
    output_path = raw_dir / "beam_inmap_concentrations.parquet"
    beam_inmap_concentrations_gdf = _build_beam_inmap_concentrations_gdf(
        concentrations=concentrations,
        inmap_grid_path=pipeline.inmap_grid_path,
        grid_id_col=pipeline.mapping_columns.get("grid_id", "srm_cell_id"),
        included_grid_ids=beam_inmap_grid_ids,
    )
    _write_concentration_outputs(beam_inmap_concentrations_gdf, str(output_path))
    return beam_inmap_concentrations_gdf, beam_inmap_grid_ids, output_path
