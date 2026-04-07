"""Step 2 — InMAP concentration workflow."""
from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Any
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from ..common import log_step_banner
from ..common import log_substep_banner
from ..common import read_table
from ..common import read_vector
from ..config.defaults import concentrations
from ..config.defaults import pollutants as default_pollutants
from ..config.defaults import tons_per_year_to_ug_per_s
from ..manifest.schema import PipelineConfig
from . import _step_label

logger = logging.getLogger(__name__)


def _trace_columns(step: str, label: str, columns: list[str]) -> None:
    preview = columns[:20]
    suffix = "" if len(columns) <= 20 else " ..."
    logger.info("%s trace %s columns(%d): %s%s", _step_label(f"2.{step}"), label, len(columns), preview, suffix)


def _trace_frame(step: str, label: str, df: pd.DataFrame, *, key_cols: Optional[list[str]] = None) -> None:
    logger.info("%s trace %s shape=%s", _step_label(f"2.{step}"), label, df.shape)
    _trace_columns(step, label, list(df.columns))
    if key_cols:
        present = [col for col in key_cols if col in df.columns]
        if present and not df.empty:
            sample = df[present].head(5).to_dict(orient="records")
            logger.info("%s trace %s sample_keys=%s", _step_label(f"2.{step}"), label, sample)


def _trace_array(step: str, label: str, values: np.ndarray) -> None:
    arr = np.asarray(values).reshape(-1)
    size = int(arr.shape[0])
    if size == 0:
        logger.info("%s trace %s length=0", _step_label(f"2.{step}"), label)
        return
    logger.info(
        "%s trace %s length=%d min=%s max=%s sum=%s nonzero=%d sample=%s",
        _step_label(f"2.{step}"),
        label,
        size,
        float(arr.min()),
        float(arr.max()),
        float(arr.sum()),
        int(np.count_nonzero(arr)),
        arr[:5].tolist(),
    )


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


def _read_sparse_transfer_matrix_npz(path: str) -> dict[str, np.ndarray | int]:
    fields = ["source_ids", "receptor_ids", "values", "source_dim", "receptor_dim"]
    progress = tqdm(
        total=len(fields),
        desc="Loading NOx->NO2 npz",
        unit="field",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
    )
    try:
        with np.load(path) as data:
            required = set(fields)
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Sparse NOx transfer matrix {path} is missing fields {sorted(missing)}")
            source_ids = np.asarray(data["source_ids"], dtype=np.int64)
            progress.update(1)
            receptor_ids = np.asarray(data["receptor_ids"], dtype=np.int64)
            progress.update(1)
            values = np.asarray(data["values"], dtype=np.float64)
            progress.update(1)
            source_dim = int(np.asarray(data["source_dim"]).item())
            progress.update(1)
            receptor_dim = int(np.asarray(data["receptor_dim"]).item())
            progress.update(1)
    finally:
        progress.close()
    return {
        "source_ids": source_ids,
        "receptor_ids": receptor_ids,
        "values": values,
        "source_dim": source_dim,
        "receptor_dim": receptor_dim,
    }


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


def _canonical_pollutant_from_emissions_column(column_name: str) -> str:
    pollutant = column_name.removeprefix("tons_per_year_")
    if pollutant.endswith("_inmap_allocated"):
        pollutant = pollutant.removesuffix("_inmap_allocated")
    return pollutant


def _expected_emissions_columns(pollutants_map: Optional[dict[str, str]] = None) -> list[str]:
    canonical_pollutants = list((pollutants_map or {}).keys()) or list(default_pollutants)
    ordered_pollutants = [pollutant for pollutant in default_pollutants if pollutant in canonical_pollutants]
    return [f"tons_per_year_{pollutant}_inmap_allocated" for pollutant in ordered_pollutants]


def _concentration_specs() -> dict[str, dict[str, str]]:
    return {
        "SOA": {
            "output": "SOA",
            "pollutant": "ROG",
            "zarr": "SOA",
            "emissions": "tons_per_year_ROG_inmap_allocated",
        },
        "pNO3": {
            "output": "pNO3",
            "pollutant": "NOx",
            "zarr": "pNO3",
            "emissions": "tons_per_year_NOx_inmap_allocated",
        },
        "pNH4": {
            "output": "pNH4",
            "pollutant": "NH3",
            "zarr": "pNH4",
            "emissions": "tons_per_year_NH3_inmap_allocated",
        },
        "pSO4": {
            "output": "pSO4",
            "pollutant": "SOx",
            "zarr": "pSO4",
            "emissions": "tons_per_year_SOx_inmap_allocated",
        },
        "PrimaryPM25": {
            "output": "PrimaryPM25",
            "pollutant": "PM2_5",
            "zarr": "PrimaryPM25",
            "emissions": "tons_per_year_PM2_5_inmap_allocated",
        },
        "BC": {
            "output": "BC",
            "pollutant": "BC",
            "zarr": "PrimaryPM25",
            "emissions": "tons_per_year_BC_inmap_allocated",
        },
        "NO2": {
            "output": "NO2",
            "pollutant": "NOx",
            "zarr": "NO2",
            "emissions": "tons_per_year_NOx_inmap_allocated",
        },
    }


def _build_no2_transfer_matrix(
    *,
    isrm_nox_to_no2_ratios_file: Optional[str],
) -> Optional[Any]:
    if isrm_nox_to_no2_ratios_file:
        if isrm_nox_to_no2_ratios_file.lower().endswith(".npz"):
            matrix = _read_sparse_transfer_matrix_npz(isrm_nox_to_no2_ratios_file)
            shape = (matrix["source_dim"], matrix["receptor_dim"])
        else:
            matrix = _read_square_transfer_matrix(isrm_nox_to_no2_ratios_file)
            shape = matrix.shape
        logger.info(
            "%s loaded NOx->NO2 transfer matrix from %s shape=%s",
            _step_label("2.2"),
            isrm_nox_to_no2_ratios_file,
            shape,
        )
        return matrix

    return None


def _compute_custom_receptor_response(
    *,
    transfer_matrix: Any,
    emissions_key: str,
    source_cells: np.ndarray,
    source_indexed: pd.DataFrame,
    receptor_cells: np.ndarray,
    result_key: str,
) -> np.ndarray:
    started = time.perf_counter()
    source_values = source_indexed[emissions_key].to_numpy()
    n_receptors = int(receptor_cells.size)
    logger.info(
        "%s computing %s from %s over %d source cells → %d receptor cells using custom transfer matrix",
        _step_label("2.2"),
        result_key,
        emissions_key,
        int(source_cells.size),
        n_receptors,
    )
    _trace_array("2", f"{result_key}.source_values", source_values)
    response = np.zeros(n_receptors, dtype=float)

    if isinstance(transfer_matrix, pd.DataFrame):
        aligned = transfer_matrix.reindex(index=source_cells, fill_value=0.0)
        # keep only columns that are in receptor_cells (which is sorted)
        cols_in_rc = [c for c in aligned.columns if np.searchsorted(receptor_cells, c) < n_receptors and receptor_cells[np.searchsorted(receptor_cells, c)] == c]
        if cols_in_rc:
            sub = aligned[cols_in_rc].to_numpy().T.dot(source_values)
            positions = np.searchsorted(receptor_cells, np.asarray(cols_in_rc, dtype=int))
            response[positions] = sub
    else:
        triplet_sources = np.asarray(transfer_matrix["source_ids"], dtype=np.int64)
        triplet_receptors = np.asarray(transfer_matrix["receptor_ids"], dtype=np.int64)
        triplet_values = np.asarray(transfer_matrix["values"], dtype=np.float64)
        source_positions = np.searchsorted(source_cells, triplet_sources)
        in_bounds = source_positions < source_cells.size
        source_valid = np.zeros_like(in_bounds, dtype=bool)
        if np.any(in_bounds):
            bounded_positions = source_positions[in_bounds]
            source_valid[in_bounds] = source_cells[bounded_positions] == triplet_sources[in_bounds]
        receptor_positions = np.searchsorted(receptor_cells, triplet_receptors)
        in_rc_bounds = receptor_positions < n_receptors
        receptor_valid = np.zeros_like(in_rc_bounds, dtype=bool)
        if np.any(in_rc_bounds):
            receptor_valid[in_rc_bounds] = receptor_cells[receptor_positions[in_rc_bounds]] == triplet_receptors[in_rc_bounds]
        valid = source_valid & receptor_valid
        if np.any(valid):
            weights = triplet_values[valid] * source_values[source_positions[valid]]
            response = np.bincount(
                receptor_positions[valid],
                weights=weights,
                minlength=n_receptors,
            ).astype(float, copy=False)

    logger.info(
        "%s computed %s in %.2fs",
        _step_label("2.2"),
        result_key,
        time.perf_counter() - started,
    )
    _trace_array("2", f"{result_key}.response", response)
    return response


def _prepare_grid_emissions(
    emissions_df: pd.DataFrame,
    source_id_col: str,
    pollutants_map: Optional[dict[str, str]] = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Normalize emissions to ISRM input species and aggregate by grid."""
    _trace_frame("0", "raw_emissions", emissions_df, key_cols=[source_id_col])
    if source_id_col not in emissions_df.columns:
        raise ValueError(f"No grid id column found. Expected {source_id_col}")

    emission_cols = _expected_emissions_columns(pollutants_map)
    source_column_map: dict[str, str] = {}
    for col in emission_cols:
        if col in emissions_df.columns:
            source_column_map[col] = col

    df = emissions_df.copy()
    df[source_id_col] = pd.to_numeric(df[source_id_col], errors="coerce")
    df = df[df[source_id_col].notna()].copy()
    df[source_id_col] = df[source_id_col].astype(int)

    for col in emission_cols:
        source_col = source_column_map.get(col)
        if source_col is None:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[source_col], errors="coerce").fillna(0.0)

    grouped = df.groupby(source_id_col, dropna=False)[emission_cols].sum().reset_index()
    available_pollutants = {
        _canonical_pollutant_from_emissions_column(col)
        for col, source_col in source_column_map.items()
        if source_col is not None
    }
    logger.info("%s trace source_column_map=%s", _step_label("2.0"), source_column_map)
    _trace_frame("0", "prepared_grid_emissions", grouped, key_cols=[source_id_col])
    logger.info("%s trace available_source_pollutants=%s", _step_label("2.0"), sorted(available_pollutants))
    return grouped, available_pollutants


def _matrix_response(
    sr,
    species_key: str,
    source_cells: np.ndarray,
    source_values: np.ndarray,
    receptor_cells: np.ndarray,
) -> np.ndarray:
    transfer = sr[species_key].oindex[[0], source_cells, receptor_cells]
    return transfer[0, :, :].T.dot(source_values)


def _align_emissions_to_isrm_sources(
    emis: pd.DataFrame,
    sr,
    source_id_col: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Step 2.1: align emissions source cells to the ISRM source index."""
    source_dim = int(sr["SOA"].shape[1])
    raw_source_cells = np.sort(emis[source_id_col].unique().astype(int))
    valid_mask = (raw_source_cells >= 0) & (raw_source_cells < source_dim)
    source_cells = raw_source_cells[valid_mask]
    dropped = int(raw_source_cells.size - source_cells.size)
    source_indexed = emis.set_index(source_id_col).reindex(source_cells).fillna(0.0)
    source_indexed.index.name = source_id_col
    logger.info(
        "%s aligned emissions to ISRM sources: raw_source_cells=%d kept=%d dropped=%d source_dim=%d",
        _step_label("2.1"),
        int(raw_source_cells.size),
        int(source_cells.size),
        dropped,
        source_dim,
    )
    _trace_frame("1", "source_indexed", source_indexed.reset_index(), key_cols=[source_id_col])
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
    receptor_cells: np.ndarray,
) -> np.ndarray:
    started = time.perf_counter()
    source_values = source_indexed[emissions_key].to_numpy()
    logger.info(
        "%s computing %s from %s over %d source cells → %d receptor cells",
        _step_label("2.2"),
        species_key,
        emissions_key,
        int(source_cells.size),
        int(receptor_cells.size),
    )
    _trace_array("2", f"{species_key}.source_values", source_values)
    arr = _matrix_response(sr, species_key, source_cells, source_values, receptor_cells)
    logger.info(
        "%s computed %s in %.2fs",
        _step_label("2.2"),
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
    receptor_cells: np.ndarray,
    isrm_nox_to_no2_ratios_file: Optional[str],
) -> Optional[np.ndarray]:
    logger.info("%s resolving NO2 source from ISRM zarr or configured NOx->NO2 transfer matrix", _step_label("2.1"))
    if "NO2" in sr:
        logger.info("%s using NO2 transfer matrix from ISRM zarr", _step_label("2.1"))
        return _compute_species_response(
            sr=sr,
            species_key="NO2",
            emissions_key="tons_per_year_NOx_inmap_allocated",
            source_cells=source_cells,
            source_indexed=source_indexed,
            receptor_cells=receptor_cells,
        )

    no2_transfer_matrix = _build_no2_transfer_matrix(
        isrm_nox_to_no2_ratios_file=isrm_nox_to_no2_ratios_file,
    )
    if no2_transfer_matrix is None:
        logger.warning(
            "%s NO2 skipped: ISRM zarr has no NO2 and no configured ISRM NOx->NO2 transfer matrix was available",
            _step_label("2.1"),
        )
        return None

    logger.info("%s using configured ISRM NOx->NO2 transfer matrix", _step_label("2.1"))
    return _compute_custom_receptor_response(
        transfer_matrix=no2_transfer_matrix,
        emissions_key="tons_per_year_NOx_inmap_allocated",
        source_cells=source_cells,
        source_indexed=source_indexed,
        receptor_cells=receptor_cells,
        result_key="NO2",
    )


def _assemble_concentration_results(
    *,
    receptor_cells: np.ndarray,
    factor: float,
    arrays: dict[str, np.ndarray],
    source_id_col: str,
) -> pd.DataFrame:
    """Step 2.3: assemble concentration outputs on the receptor dimension."""
    n_receptors = int(receptor_cells.size)
    logger.info("%s assembling concentration DataFrame for %d receptors", _step_label("2.3"), n_receptors)
    for name, arr in arrays.items():
        logger.info("%s %s response length=%d", _step_label("2.3"), name, int(np.asarray(arr).shape[0]))
        _trace_array("3", name, arr)
    bad = {
        name: int(np.asarray(arr).shape[0])
        for name, arr in arrays.items()
        if int(np.asarray(arr).shape[0]) != n_receptors
    }
    if bad:
        raise ValueError(
            f"{_step_label('2.3')} produced mismatched receptor lengths: expected {n_receptors}, got {bad}"
        )

    results = pd.DataFrame({source_id_col: receptor_cells})
    ordered_output_keys = [
        _concentration_specs()[name]["output"]
        for name in concentrations
        if name in _concentration_specs()
    ] + ["TotalPM25"]
    for output_key in ordered_output_keys:
        if output_key in arrays:
            results[output_key] = factor * arrays[output_key]

    _trace_frame("3", "concentrations", results, key_cols=[source_id_col])
    return results


def _build_beam_inmap_concentrations_gdf(
    *,
    concentrations: pd.DataFrame,
    inmap_grid_path: str,
    grid_id_col: str,
    source_id_col: str,
) -> gpd.GeoDataFrame:
    logger.info(
        "%s building BEAM InMAP concentrations GeoDataFrame from %s using grid_id_col=%s",
        _step_label("2.4"),
        inmap_grid_path,
        grid_id_col,
    )
    grid = read_vector(inmap_grid_path)
    if grid_id_col not in grid.columns:
        raise ValueError(
            f"{_step_label('2.4')} grid id column '{grid_id_col}' not found in {inmap_grid_path}. "
            f"Available columns: {list(grid.columns)}"
        )
    grid = grid.copy()
    grid[grid_id_col] = pd.to_numeric(grid[grid_id_col], errors="raise").astype(int)
    concentration_cols = [col for col in concentrations.columns if col != source_id_col and col not in grid.columns]
    merged = grid.merge(
        concentrations[[source_id_col] + concentration_cols],
        how="left",
        left_on=grid_id_col,
        right_on=source_id_col,
    )
    if source_id_col not in merged.columns:
        merged[source_id_col] = pd.to_numeric(merged[grid_id_col], errors="raise").astype(int)
    elif grid_id_col != source_id_col:
        merged[source_id_col] = pd.to_numeric(merged[source_id_col], errors="raise").astype(int)
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
        raise ValueError("Output path must end with .parquet for Step 2 BEAM InMAP concentrations")
    gpkg_path = out.with_suffix(".gpkg")
    logger.info("%s writing BEAM InMAP concentrations GeoParquet to %s", _step_label("2.5"), out)
    beam_inmap_concentrations_gdf.to_parquet(out, index=False)
    logger.info("%s writing BEAM InMAP concentrations GPKG to %s", _step_label("2.5"), gpkg_path)
    beam_inmap_concentrations_gdf.to_file(gpkg_path, driver="GPKG")
    return out, gpkg_path


def compute_isrm_concentrations(
    *,
    grid_emissions_df: pd.DataFrame,
    sr,
    factor: float,
    requested_pollutants: list[str],
    receptor_cells: np.ndarray,
    source_id_col: str = "inmap_cell_id",
    isrm_nox_to_no2_ratios_file: Optional[str] = None,
    pollutants_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    emis, available_pollutants = _prepare_grid_emissions(
        grid_emissions_df,
        source_id_col=source_id_col,
        pollutants_map=pollutants_map,
    )
    if emis.empty:
        raise ValueError(f"No emissions rows found after {source_id_col} normalization.")

    source_cells, source_indexed = _align_emissions_to_isrm_sources(emis, sr, source_id_col=source_id_col)
    if source_cells.size == 0:
        raise ValueError("No emissions source cells remain after ISRM source alignment.")

    requested_pollutant_set = set(requested_pollutants)
    specs = _concentration_specs()
    arrays: dict[str, np.ndarray] = {}
    concentration_plan: list[tuple[str, dict[str, str]]] = []
    for concentration_name in concentrations:
        spec = specs.get(concentration_name)
        if spec is None:
            logger.warning(
                "%s concentration %s is listed in defaults.concentrations but has no computation spec",
                _step_label("2.1"),
                concentration_name,
            )
            continue
        concentration_output = spec["output"]
        required_pollutant = spec["pollutant"]
        if required_pollutant not in requested_pollutant_set:
            continue
        if required_pollutant not in available_pollutants:
            logger.warning(
                "%s requested pollutant %s is missing source emissions needed for concentration %s",
                _step_label("2.1"),
                required_pollutant,
                concentration_output,
            )
            continue
        if concentration_output == "NO2":
            has_no2 = "NO2" in sr or bool(isrm_nox_to_no2_ratios_file)
            if not has_no2:
                logger.warning(
                    "%s requested pollutant NOx has no NO2 concentration source: ISRM zarr has no NO2 and "
                    "no isrm_nox_to_no2_ratios_file fallback was configured",
                    _step_label("2.1"),
                )
                continue
        elif spec["zarr"] not in sr:
            logger.warning(
                "%s requested pollutant %s is missing required concentration source %s in ISRM zarr",
                _step_label("2.1"),
                required_pollutant,
                concentration_output,
            )
            continue
        concentration_plan.append((concentration_name, spec))

    with logging_redirect_tqdm():
        for concentration_name, spec in tqdm(
            concentration_plan,
            desc="Step 2.2 pollutant responses",
            unit="species",
            file=sys.stdout,
            dynamic_ncols=True,
            leave=True,
        ):
            concentration_output = spec["output"]
            if concentration_output == "NO2":
                no2_response = _compute_no2_response(
                    sr=sr,
                    source_cells=source_cells,
                    source_indexed=source_indexed,
                    receptor_cells=receptor_cells,
                    isrm_nox_to_no2_ratios_file=isrm_nox_to_no2_ratios_file,
                )
                if no2_response is not None:
                    arrays["NO2"] = no2_response
                continue

            arrays[concentration_output] = _compute_species_response(
                sr=sr,
                species_key=spec["zarr"],
                emissions_key=spec["emissions"],
                source_cells=source_cells,
                source_indexed=source_indexed,
                receptor_cells=receptor_cells,
            )

    total_pm_components = ["SOA", "pNO3", "pNH4", "pSO4", "PrimaryPM25"]
    missing_total_pm_components = [component for component in total_pm_components if component not in arrays]
    if missing_total_pm_components:
        logger.warning(
            "%s TotalPM25 was not calculated because required concentration components are missing: %s",
            _step_label("2.2"),
            missing_total_pm_components,
        )
    else:
        arrays["TotalPM25"] = sum(arrays[component] for component in total_pm_components)
        _trace_array("2", "TotalPM25", arrays["TotalPM25"])

    return _assemble_concentration_results(
        receptor_cells=receptor_cells,
        factor=factor,
        arrays=arrays,
        source_id_col=source_id_col,
    )


def run(
    *,
    pipeline: PipelineConfig,
    raw_dir: Path,
    emissions_input_path: str,
    inmap_study_area_grid_path: Optional[str] = None,
) -> tuple[gpd.GeoDataFrame, np.ndarray, Path]:
    """Step 2: compute and export receptor-side InMAP concentrations from BEAM InMAP emissions."""
    if not pipeline.inmap_enabled:
        raise ValueError("InMAP concentration step was called but pipeline.inmap_enabled is false.")
    if not pipeline.isrm_url:
        raise ValueError(
            "isrm_url must be configured. "
            "Set impacts.dispersions.inmap.isrm_zarr in settings.yaml."
        )
    if "grid_id" not in pipeline.mapping_columns or not str(pipeline.mapping_columns["grid_id"]).strip():
        raise ValueError("pipeline.mapping_columns.grid_id must be configured before running InMAP concentrations.")

    log_step_banner("Step 2", "Compute InMAP Concentrations", logger=logger)
    log_substep_banner("2.0", "load emissions input", logger=logger)
    logger.info("%s loading BEAM emissions for InMAP from %s", _step_label("2.0"), emissions_input_path)
    emissions_df = read_table(emissions_input_path)
    _trace_frame("0", "loaded_emissions", emissions_df, key_cols=["inmap_cell_id"])
    beam_inmap_grid_ids = (
        pd.to_numeric(emissions_df["inmap_cell_id"], errors="coerce")
        .dropna()
        .astype(int)
        .sort_values()
        .unique()
    )
    logger.info(
        "%s trace beam_inmap_grid_ids count=%d sample=%s",
        _step_label("2.0"),
        int(beam_inmap_grid_ids.shape[0]),
        beam_inmap_grid_ids[:10].tolist(),
    )

    log_substep_banner("2.1", "load ISRM source data", logger=logger)
    logger.info("%s loading ISRM store from %s", _step_label("2.0"), pipeline.isrm_url)
    sr = _load_isrm_store(pipeline.isrm_url)
    logger.info(
        "%s trace isrm_shapes SOA=%s TotalPop=%s MortalityRate=%s",
        _step_label("2.0"),
        getattr(sr["SOA"], "shape", None),
        getattr(sr["TotalPop"], "shape", None) if "TotalPop" in sr else None,
        getattr(sr["MortalityRate"], "shape", None) if "MortalityRate" in sr else None,
    )
    log_substep_banner("2.2", "compute concentration responses", logger=logger)
    concentrations = compute_isrm_concentrations(
        grid_emissions_df=emissions_df,
        sr=sr,
        factor=float(tons_per_year_to_ug_per_s),
        requested_pollutants=pipeline.pollutants,
        receptor_cells=beam_inmap_grid_ids,
        source_id_col="inmap_cell_id",
        isrm_nox_to_no2_ratios_file=pipeline.isrm_nox_to_no2_ratios_file,
        pollutants_map=pipeline.pollutants_map,
    )
    output_path = raw_dir / "beam_inmap_concentrations.parquet"
    log_substep_banner("2.3", "build receptor geodataframe", logger=logger)
    grid_path = inmap_study_area_grid_path or pipeline.inmap_grid_path
    beam_inmap_concentrations_gdf = _build_beam_inmap_concentrations_gdf(
        concentrations=concentrations,
        inmap_grid_path=grid_path,
        grid_id_col=str(pipeline.mapping_columns["grid_id"]),
        source_id_col="inmap_cell_id",
    )
    log_substep_banner("2.4", "write concentration outputs", logger=logger)
    _write_concentration_outputs(beam_inmap_concentrations_gdf, str(output_path))
    return beam_inmap_concentrations_gdf, beam_inmap_grid_ids, output_path
