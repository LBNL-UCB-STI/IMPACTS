from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from typing import Dict

import geopandas as gpd
import osm_chordify

from ..manifest.file_ops import file_entry
from .common import required_local_path
from .common import stage_local_input

logger = logging.getLogger(__name__)


def _is_remote_path(path: str) -> bool:
    return path.startswith(("gs://", "s3://", "http://", "https://"))


def _validate_local_path(path: str, label: str) -> None:
    if _is_remote_path(path):
        return
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _save_geodataframe(gdf: gpd.GeoDataFrame, output_path: str) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix == ".parquet":
        gdf.to_parquet(out, index=False)
        return
    if suffix == ".geojson":
        gdf.to_file(out, driver="GeoJSON")
        return
    if suffix == ".gpkg":
        gdf.to_file(out, driver="GPKG")
        return
    gdf.to_file(out.with_suffix(".geojson"), driver="GeoJSON")


def _load_geodataframe(path_or_gdf):
    if isinstance(path_or_gdf, gpd.GeoDataFrame):
        return path_or_gdf.copy()
    path = Path(path_or_gdf)
    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def map_beam_network_to_osm(
    osm_path: str,
    network_path: str,
    output_path: str,
    network_osm_id_col: str = "attributeOrigId",
    output_epsg: int | None = None,
):
    _validate_local_path(osm_path, "OSM network path")
    _validate_local_path(network_path, "BEAM network path")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    mapped = osm_chordify.map_osm_with_beam_network(
        osm_path=osm_path,
        network_path=network_path,
        network_osm_id_col=network_osm_id_col,
        output_path=output_path,
    )
    if output_epsg is not None:
        mapped_gdf = _load_geodataframe(output_path).to_crs(epsg=output_epsg)
        _save_geodataframe(mapped_gdf, output_path)
        return mapped_gdf
    return mapped


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    processing,
    beam_network_source: str,
    osm_source: str,
    local_output_epsg: int,
) -> dict[str, str]:
    staged_osm = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="osm_network",
        source_path=required_local_path(osm_source, "inputs.osm_links"),
        relative_target=f"osm/{Path(osm_source).name}",
    )
    staged_network = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="network",
        source_path=beam_network_source,
        relative_target=f"network/{Path(beam_network_source).name}",
    )

    staged_beam_osm_mapped_path = str((input_root / "network" / "beam_osm_mapped.parquet").resolve())
    staged_beam_osm_mapped_gpkg = str(Path(staged_beam_osm_mapped_path).with_suffix(".gpkg"))
    if Path(staged_beam_osm_mapped_path).exists():
        logger.info("Preprocess: reusing Step 1.1 BEAM/OSM mapping %s", staged_beam_osm_mapped_path)
    else:
        logger.info("Preprocess: Step 1.1 mapping BEAM network to OSM using %s", staged_osm)
        beam_osm_mapped = map_beam_network_to_osm(
            osm_path=staged_osm,
            network_path=staged_network,
            output_path=staged_beam_osm_mapped_path,
            network_osm_id_col=processing.beam_osm_id_col,
            output_epsg=int(local_output_epsg),
        )
        beam_osm_mapped.to_file(staged_beam_osm_mapped_gpkg, driver="GPKG")
        logger.info("Preprocess: Step 1.1 complete: wrote %s", staged_beam_osm_mapped_path)
    logger.info("Preprocess: Step 1.1 buffered network generation skipped for line-based workflow")
    manifest_inputs["beam_osm_mapped"] = file_entry(
        kind="local",
        path=staged_beam_osm_mapped_path,
        staged_path=staged_beam_osm_mapped_path,
        optional=True,
    )
    return {
        "staged_osm": staged_osm,
        "staged_network": staged_network,
        "staged_beam_osm_mapped_path": staged_beam_osm_mapped_path,
    }
