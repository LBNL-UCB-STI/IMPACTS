from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


def _required_string(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required value: {label}")
    return text


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_int(value: Any, label: str) -> int:
    if value is None:
        raise ValueError(f"Missing required value: {label}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer for {label}: {value}") from exc


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value: {value}") from exc


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float value: {value}") from exc


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _normalize_runtime_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "shared" in payload and "shared_context" not in payload:
        payload = dict(payload)
        payload["shared_context"] = payload["shared"]
    if "processing" in payload:
        return payload

    emissions = payload.get("emissions", {}) or {}
    dispersions = payload.get("dispersions", {}) or {}
    outputs = payload.get("outputs", {}) or {}
    inmap = dispersions.get("inmap", {}) or {}
    aermod = dispersions.get("aermod", {}) or {}
    inputs = dict(payload.get("inputs", {}) or {})

    if "activity_corrections" not in inputs and emissions.get("activity_correction_factors_file") is not None:
        inputs["activity_corrections"] = emissions.get("activity_correction_factors_file")
    isrm_directory = _optional_string(inmap.get("isrm_zarr_directory"))
    isrm_s3bucket = _optional_string(inmap.get("isrm_zarr_s3bucket"))
    isrm_direct = _optional_string(inmap.get("isrm_zarr"))
    isrm_nox_to_no2_matrix = _optional_string(inmap.get("isrm_nox_to_no2_matrix"))
    if "isrm_zarr" not in inputs:
        if isrm_directory and Path(isrm_directory).exists():
            inputs["isrm_zarr"] = isrm_directory
        else:
            inputs["isrm_zarr"] = isrm_direct or isrm_s3bucket or isrm_directory
    if "isrm_nox_to_no2_matrix" not in inputs:
        inputs["isrm_nox_to_no2_matrix"] = isrm_nox_to_no2_matrix

    normalized = dict(payload)
    normalized["inputs"] = inputs
    normalized["processing"] = {
        "annualization_days": emissions.get("annualization_days"),
        "pollutants": emissions.get("pollutants"),
        "grid": {
            "inmap_grid_path": inmap.get("grid_path"),
            "inmap_grid_epsg": inmap.get("grid_epsg"),
            "inmap_grid_id": inmap.get("grid_id"),
            "aermod_grid_path": aermod.get("grid_path"),
            "aermod_grid_epsg": aermod.get("grid_epsg"),
            "aermod_grid_id": aermod.get("grid_id"),
        },
        "mapping_columns": {
            "grid_id": inmap.get("grid_id"),
        },
    }
    return normalized


@dataclass(frozen=True)
class GeographyFips:
    state: str
    counties: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeographyFips":
        return cls(
            state=_required_string(payload.get("state"), "shared_context.geography.fips.state"),
            counties=_coerce_string_list(payload.get("counties")),
        )


@dataclass(frozen=True)
class GeographyZones:
    zone_type: str
    source_file: str
    source_crs: Optional[str] = None
    canonical_id_col: Optional[str] = None
    activitysim_index_col: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeographyZones":
        return cls(
            zone_type=_required_string(payload.get("zone_type"), "shared_context.geography.zones.zone_type"),
            source_file=_required_string(payload.get("source_file"), "shared_context.geography.zones.source_file"),
            source_crs=_optional_string(payload.get("source_crs")),
            canonical_id_col=_optional_string(payload.get("canonical_id_col")),
            activitysim_index_col=_optional_string(payload.get("activitysim_index_col")),
        )


@dataclass(frozen=True)
class GeographyContext:
    fips: GeographyFips
    local_crs: Optional[str] = None
    zones: Optional[GeographyZones] = None
    alternative_zones: Optional[GeographyZones] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeographyContext":
        zones_payload = payload.get("zones")
        alternative_zones_payload = payload.get("alternative_zones")
        fips_payload = payload.get("fips", payload.get("FIPS", {})) or {}
        return cls(
            fips=GeographyFips.from_dict(fips_payload),
            local_crs=_optional_string(payload.get("local_crs")),
            zones=GeographyZones.from_dict(zones_payload) if zones_payload else None,
            alternative_zones=GeographyZones.from_dict(alternative_zones_payload)
            if alternative_zones_payload
            else None,
        )


@dataclass(frozen=True)
class SharedSkims:
    zone_type: Optional[str] = None
    fname: Optional[str] = None
    origin_fname: Optional[str] = None
    geoms_fname: Optional[str] = None
    geoms_index_col: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SharedSkims":
        return cls(
            zone_type=_optional_string(payload.get("zone_type")),
            fname=_optional_string(payload.get("fname")),
            origin_fname=_optional_string(payload.get("origin_fname")),
            geoms_fname=_optional_string(payload.get("geoms_fname")),
            geoms_index_col=_optional_string(payload.get("geoms_index_col")),
        )


@dataclass(frozen=True)
class SharedContext:
    region: Optional[str]
    start_year: Optional[int]
    geography: GeographyContext
    skims: Optional[SharedSkims] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SharedContext":
        return cls(
            region=_optional_string(payload.get("region")),
            start_year=_optional_int(payload.get("start_year")),
            geography=GeographyContext.from_dict(payload.get("geography", {}) or {}),
            skims=SharedSkims.from_dict(payload.get("skims", {}) or {}) if payload.get("skims") else None,
        )


@dataclass(frozen=True)
class RuntimeInputs:
    beam_network: str
    emissions_skims: str
    osm_pbf: str
    activity_corrections: Optional[str] = None
    isrm_zarr: Optional[str] = None
    isrm_nox_to_no2_matrix: Optional[str] = None
    osm_links: Optional[str] = None
    beam_mapdb: Optional[str] = None
    households_asim_out: Optional[str] = None
    persons_asim_out: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RuntimeInputs":
        return cls(
            beam_network=_required_string(payload.get("beam_network"), "inputs.beam_network"),
            emissions_skims=_required_string(payload.get("emissions_skims"), "inputs.emissions_skims"),
            osm_pbf=_required_string(payload.get("osm_pbf"), "inputs.osm_pbf"),
            activity_corrections=_optional_string(payload.get("activity_corrections")),
            isrm_zarr=_optional_string(payload.get("isrm_zarr")),
            isrm_nox_to_no2_matrix=_optional_string(payload.get("isrm_nox_to_no2_matrix")),
            osm_links=_optional_string(payload.get("osm_links")),
            beam_mapdb=_optional_string(payload.get("beam_mapdb")),
            households_asim_out=_optional_string(payload.get("households_asim_out")),
            persons_asim_out=_optional_string(payload.get("persons_asim_out")),
        )


@dataclass(frozen=True)
class GridProcessing:
    inmap_grid_path: str
    aermod_grid_path: Optional[str] = None
    inmap_grid_epsg: Optional[int] = None
    aermod_grid_epsg: Optional[int] = None
    inmap_grid_id: Optional[str] = None
    aermod_grid_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GridProcessing":
        return cls(
            inmap_grid_path=_required_string(payload.get("inmap_grid_path"), "processing.grid.inmap_grid_path"),
            aermod_grid_path=_optional_string(payload.get("aermod_grid_path")),
            inmap_grid_epsg=_optional_int(payload.get("inmap_grid_epsg")),
            aermod_grid_epsg=_optional_int(payload.get("aermod_grid_epsg")),
            inmap_grid_id=_optional_string(payload.get("inmap_grid_id")),
            aermod_grid_id=_optional_string(payload.get("aermod_grid_id")),
        )


@dataclass(frozen=True)
class ActivityCorrectionsColumns:
    county_fips: str = "countyfp"
    vmt_factor: str = "vmt_factor"
    trips_factor: str = "trips_factor"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActivityCorrectionsColumns":
        return cls(
            county_fips=_optional_string(payload.get("county_fips")) or "countyfp",
            vmt_factor=_optional_string(payload.get("vmt_factor")) or "vmt_factor",
            trips_factor=_optional_string(payload.get("trips_factor")) or "trips_factor",
        )


@dataclass(frozen=True)
class MappingColumns:
    link_id: str = "edge_linkId"
    proportion: str = "zone_edge_proportion"
    grid_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MappingColumns":
        return cls(
            link_id=_optional_string(payload.get("link_id")) or "edge_linkId",
            proportion=_optional_string(payload.get("proportion")) or "zone_edge_proportion",
            grid_id=_optional_string(payload.get("grid_id")),
        )


@dataclass(frozen=True)
class SkimsColumns:
    hour: str = "hour"
    link_id: str = "linkId"
    vehicle_type: str = "vehicleTypeId"
    process: str = "process"
    emissions: str = "emissions"
    observations: str = "observations"
    iterations: str = "iterations"
    travel_time: str = "travelTimeInSecond"
    parking_duration: str = "parkingDurationInSecond"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SkimsColumns":
        return cls(
            hour=_optional_string(payload.get("hour")) or "hour",
            link_id=_optional_string(payload.get("link_id")) or "linkId",
            vehicle_type=_optional_string(payload.get("vehicle_type")) or "vehicleTypeId",
            process=_optional_string(payload.get("process")) or "process",
            emissions=_optional_string(payload.get("emissions")) or "emissions",
            observations=_optional_string(payload.get("observations")) or "observations",
            iterations=_optional_string(payload.get("iterations")) or "iterations",
            travel_time=_optional_string(payload.get("travel_time")) or "travelTimeInSecond",
            parking_duration=_optional_string(payload.get("parking_duration")) or "parkingDurationInSecond",
        )


@dataclass(frozen=True)
class DispersionEmissionsColumns:
    grid_id: str = "GRID"
    rog: str = "tons_per_year_ROG"
    nox: str = "tons_per_year_NOx"
    nh3: str = "tons_per_year_NH3"
    sox: str = "tons_per_year_SOx"
    pm25: str = "tons_per_year_PM2_5"
    bch: str = "tons_per_year_BCh"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DispersionEmissionsColumns":
        return cls(
            grid_id=_optional_string(payload.get("grid_id")) or "GRID",
            rog=_optional_string(payload.get("rog")) or "tons_per_year_ROG",
            nox=_optional_string(payload.get("nox")) or "tons_per_year_NOx",
            nh3=_optional_string(payload.get("nh3")) or "tons_per_year_NH3",
            sox=_optional_string(payload.get("sox")) or "tons_per_year_SOx",
            pm25=_optional_string(payload.get("pm25")) or "tons_per_year_PM2_5",
            bch=_optional_string(payload.get("bch")) or "tons_per_year_BCh",
        )


@dataclass(frozen=True)
class DispersionSettings:
    concentration_factor: float = 28766.639
    include_health: bool = False
    emissions_columns: DispersionEmissionsColumns = field(default_factory=DispersionEmissionsColumns)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DispersionSettings":
        return cls(
            concentration_factor=_optional_float(payload.get("concentration_factor")) or 28766.639,
            include_health=bool(payload.get("include_health", False)),
            emissions_columns=DispersionEmissionsColumns.from_dict(payload.get("emissions_columns", {}) or {}),
        )


@dataclass(frozen=True)
class ProcessingSettings:
    pollutants: List[str]
    annualization_days: int
    grid: GridProcessing
    beam_osm_id_col: str = "attributeOrigId"
    beam_length_col: str = "linkLength"
    county_area_name: str = "county"
    prepared_skims_group_cols: List[str] = field(default_factory=lambda: ["linkId", "vehicleTypeId", "process"])
    skims_columns: SkimsColumns = field(default_factory=SkimsColumns)
    mapping_columns: MappingColumns = field(default_factory=MappingColumns)
    activity_corrections_columns: ActivityCorrectionsColumns = field(default_factory=ActivityCorrectionsColumns)
    dispersion: DispersionSettings = field(default_factory=DispersionSettings)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProcessingSettings":
        pollutants = _coerce_string_list(payload.get("pollutants"))
        if not pollutants:
            raise ValueError("processing.pollutants must contain at least one pollutant")
        return cls(
            pollutants=pollutants,
            annualization_days=_required_int(payload.get("annualization_days"), "processing.annualization_days"),
            grid=GridProcessing.from_dict(payload.get("grid", {}) or {}),
            beam_osm_id_col=_optional_string(payload.get("beam_osm_id_col")) or "attributeOrigId",
            beam_length_col=_optional_string(payload.get("beam_length_col")) or "linkLength",
            county_area_name=_optional_string(payload.get("county_area_name")) or "county",
            prepared_skims_group_cols=(
                _coerce_string_list(payload.get("prepared_skims_group_cols"))
                or ["linkId", "vehicleTypeId", "process"]
            ),
            skims_columns=SkimsColumns.from_dict(payload.get("skims_columns", {}) or {}),
            mapping_columns=MappingColumns.from_dict(payload.get("mapping_columns", {}) or {}),
            activity_corrections_columns=ActivityCorrectionsColumns.from_dict(
                payload.get("activity_corrections_columns", {}) or {}
            ),
            dispersion=DispersionSettings.from_dict(payload.get("dispersion", {}) or {}),
        )


@dataclass(frozen=True)
class OutputSettings:
    output_dir: str
    exposure_table: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OutputSettings":
        return cls(
            output_dir=_required_string(payload.get("output_dir"), "outputs.output_dir"),
            exposure_table=_optional_string(payload.get("exposure_table")),
        )


@dataclass(frozen=True)
class ImpactsRuntimeConfig:
    shared_context: SharedContext
    inputs: RuntimeInputs
    processing: ProcessingSettings
    outputs: OutputSettings

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImpactsRuntimeConfig":
        payload = _normalize_runtime_payload(payload)
        return cls(
            shared_context=SharedContext.from_dict(payload.get("shared_context", {}) or {}),
            inputs=RuntimeInputs.from_dict(payload.get("inputs", {}) or {}),
            processing=ProcessingSettings.from_dict(payload.get("processing", {}) or {}),
            outputs=OutputSettings.from_dict(payload.get("outputs", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return {
            "shared": payload["shared_context"],
            "inputs": {
                "beam_network": payload["inputs"]["beam_network"],
                "emissions_skims": payload["inputs"]["emissions_skims"],
                "osm_pbf": payload["inputs"]["osm_pbf"],
                "households_asim_out": payload["inputs"]["households_asim_out"],
                "persons_asim_out": payload["inputs"]["persons_asim_out"],
            },
            "emissions": {
                "annualization_days": payload["processing"]["annualization_days"],
                "activity_correction_factors_file": payload["inputs"]["activity_corrections"],
                "pollutants": payload["processing"]["pollutants"],
            },
            "dispersions": {
                "inmap": {
                    "isrm_zarr_directory": payload["inputs"]["isrm_zarr"],
                    "isrm_zarr_s3bucket": None,
                    "isrm_nox_to_no2_matrix": payload["inputs"].get("isrm_nox_to_no2_matrix"),
                    "grid_path": payload["processing"]["grid"]["inmap_grid_path"],
                    "grid_epsg": payload["processing"]["grid"]["inmap_grid_epsg"],
                    "grid_id": payload["processing"]["mapping_columns"]["grid_id"],
                },
                "aermod": {
                    "grid_path": payload["processing"]["grid"]["aermod_grid_path"],
                    "grid_epsg": payload["processing"]["grid"]["aermod_grid_epsg"],
                },
            },
            "outputs": {
                "output_dir": payload["outputs"]["output_dir"],
                "exposure_table": payload["outputs"]["exposure_table"],
            },
        }

    def output_root(self) -> Path:
        return Path(self.outputs.output_dir)
