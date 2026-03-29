from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .defaults import pollutants


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


def _coerce_string_map(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    resolved: Dict[str, str] = {}
    for key, mapped in value.items():
        key_text = str(key).strip()
        mapped_text = str(mapped).strip()
        if key_text and mapped_text:
            resolved[key_text] = mapped_text
    return resolved


def _build_pollutants_map(pollutants_value: Any, pollutants_map_value: Any) -> Dict[str, str]:
    configured_pollutants = _coerce_string_list(pollutants_value) or list(pollutants)
    if len(configured_pollutants) > len(pollutants):
        raise ValueError(
            "processing.pollutants cannot contain more entries than defaults.pollutants "
            f"({len(pollutants)}): got {len(configured_pollutants)}"
        )

    mapping = {
        canonical: configured
        for canonical, configured in zip(pollutants, configured_pollutants)
    }
    explicit_map = _coerce_string_map(pollutants_map_value)
    for canonical, configured in explicit_map.items():
        if canonical not in pollutants:
            raise ValueError(
                f"processing.pollutants_map contains unsupported canonical pollutant '{canonical}'. "
                f"Expected one of {pollutants}"
            )
        mapping[canonical] = configured
    return mapping


def _normalize_runtime_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if "shared" in payload and "shared_context" not in payload:
        payload = dict(payload)
        payload["shared_context"] = payload["shared"]

    emissions = payload.get("emissions", {}) or {}
    dispersions = payload.get("dispersions", {}) or {}
    processing_payload = payload.get("processing", {}) or {}
    processing_emissions = processing_payload.get("emissions", {}) or {}
    processing_dispersions = processing_payload.get("dispersions", {}) or {}
    outputs = payload.get("outputs", {}) or {}
    inmap = dispersions.get("inmap", {}) or {}
    aermod = dispersions.get("aermod", {}) or {}
    processing_dispersion_inmap = processing_dispersions.get("inmap", {}) or {}
    processing_dispersion_aermod = processing_dispersions.get("aermod", {}) or {}
    processing_inmap = processing_payload.get("inmap", {}) or {}
    processing_aermod = processing_payload.get("aermod", {}) or {}
    inputs = dict(payload.get("inputs", {}) or {})

    if "activity_totals_file" not in inputs:
        inputs["activity_totals_file"] = (
            _optional_string(processing_emissions.get("activity_totals_file"))
            or _optional_string(emissions.get("activity_totals_file"))
            or _optional_string(processing_emissions.get("county_activity_totals_target_file"))
            or _optional_string(emissions.get("county_activity_totals_target_file"))
            or _optional_string(processing_emissions.get("activity_totals_target_file"))
            or _optional_string(emissions.get("activity_totals_target_file"))
            or _optional_string(processing_emissions.get("activity_corrections"))
            or _optional_string(emissions.get("activity_corrections"))
            or _optional_string(emissions.get("activity_correction_factors_file"))
        )
    if "simulation_network_folder" not in inputs:
        inputs["simulation_network_folder"] = (
            _optional_string(processing_emissions.get("simulation_network_folder"))
            or _optional_string(emissions.get("simulation_network_folder"))
        )
    if "osm_network_folder" not in inputs:
        inputs["osm_network_folder"] = (
            _optional_string(processing_emissions.get("osm_network_folder"))
            or _optional_string(emissions.get("osm_network_folder"))
        )
    isrm_directory = _optional_string(inmap.get("isrm_zarr_directory"))
    isrm_s3bucket = _optional_string(inmap.get("isrm_zarr_s3bucket"))
    isrm_direct = (
        _optional_string(inmap.get("isrm_zarr"))
        or _optional_string(processing_dispersion_inmap.get("isrm_zarr"))
        or _optional_string(processing_inmap.get("isrm_zarr"))
    )
    isrm_nox_to_no2_matrix_npz = _optional_string(inmap.get("isrm_nox_to_no2_matrix_npz")) or _optional_string(
        processing_dispersion_inmap.get("isrm_nox_to_no2_matrix_npz")
    ) or _optional_string(processing_inmap.get("isrm_nox_to_no2_matrix_npz"))
    if "isrm_zarr" not in inputs:
        if isrm_directory and Path(isrm_directory).exists():
            inputs["isrm_zarr"] = isrm_directory
        else:
            inputs["isrm_zarr"] = isrm_direct or isrm_s3bucket or isrm_directory
    if "isrm_nox_to_no2_matrix_npz" not in inputs:
        inputs["isrm_nox_to_no2_matrix_npz"] = isrm_nox_to_no2_matrix_npz

    if "processing" in payload:
        normalized = dict(payload)
        normalized["inputs"] = inputs
        processing = dict(normalized.get("processing", {}) or {})
        emissions_processing = dict(processing_emissions)
        dispersions_processing = dict(processing_dispersions)
        inmap_processing = dict(dispersions_processing.get("inmap", {}) or processing_inmap)
        aermod_processing = dict(dispersions_processing.get("aermod", {}) or processing_aermod)
        if "annualization_days" not in processing:
            processing["annualization_days"] = emissions_processing.get("annualization_days")
        if "population_sample" not in processing:
            processing["population_sample"] = emissions_processing.get("population_sample")
        if "pollutants" not in processing:
            processing["pollutants"] = emissions_processing.get("pollutants")
        if "pollutants_map" not in processing:
            processing["pollutants_map"] = emissions_processing.get("pollutants_map")
        if "grid" not in processing:
            processing["grid"] = {
                "inmap_grid_path": inmap_processing.get("grid_path"),
                "inmap_grid_epsg": inmap_processing.get("grid_epsg"),
                "inmap_grid_id": inmap_processing.get("grid_id"),
                "aermod_grid_path": aermod_processing.get("grid_path"),
                "aermod_grid_epsg": aermod_processing.get("grid_epsg"),
                "aermod_grid_id": aermod_processing.get("grid_id"),
            }
        if "mapping_columns" not in processing:
            processing["mapping_columns"] = {"grid_id": inmap_processing.get("grid_id")}
        if "concentration_factor" not in processing:
            processing["concentration_factor"] = (
                _optional_float(dispersions_processing.get("concentration_factor"))
                or _optional_float(processing.get("dispersion", {}).get("concentration_factor"))
            )
        normalized["processing"] = processing
        return normalized

    normalized = dict(payload)
    normalized["inputs"] = inputs
    normalized["processing"] = {
        "annualization_days": emissions.get("annualization_days"),
        "population_sample": emissions.get("population_sample"),
        "concentration_factor": None,
        "pollutants": emissions.get("pollutants"),
        "pollutants_map": emissions.get("pollutants_map"),
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
    simulation_network_folder: str
    osm_network_folder: str
    activity_totals_file: Optional[str] = None
    isrm_zarr: Optional[str] = None
    isrm_nox_to_no2_matrix_npz: Optional[str] = None
    households_asim_out: Optional[str] = None
    persons_asim_out: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RuntimeInputs":
        return cls(
            simulation_network_folder=_required_string(
                payload.get("simulation_network_folder"),
                "inputs.simulation_network_folder",
            ),
            osm_network_folder=_required_string(
                payload.get("osm_network_folder"),
                "inputs.osm_network_folder",
            ),
            activity_totals_file=_optional_string(
                payload.get("activity_totals_file") or payload.get("county_activity_totals_target_file")
            ),
            isrm_zarr=_optional_string(payload.get("isrm_zarr")),
            isrm_nox_to_no2_matrix_npz=_optional_string(payload.get("isrm_nox_to_no2_matrix_npz")),
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
    bc: str = "tons_per_year_BC"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "DispersionEmissionsColumns":
        return cls(
            grid_id=_optional_string(payload.get("grid_id")) or "GRID",
            rog=_optional_string(payload.get("rog")) or "tons_per_year_ROG",
            nox=_optional_string(payload.get("nox")) or "tons_per_year_NOx",
            nh3=_optional_string(payload.get("nh3")) or "tons_per_year_NH3",
            sox=_optional_string(payload.get("sox")) or "tons_per_year_SOx",
            pm25=_optional_string(payload.get("pm25")) or "tons_per_year_PM2_5",
            bc=_optional_string(payload.get("bc")) or "tons_per_year_BC",
        )


@dataclass(frozen=True)
class ConcentrationSettings:
    concentration_factor: float = 28766.639
    emissions_columns: DispersionEmissionsColumns = field(default_factory=DispersionEmissionsColumns)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConcentrationSettings":
        return cls(
            concentration_factor=_optional_float(payload.get("concentration_factor")) or 28766.639,
            emissions_columns=DispersionEmissionsColumns.from_dict(payload.get("emissions_columns", {}) or {}),
        )


@dataclass(frozen=True)
class ProcessingSettings:
    pollutants: List[str]
    pollutants_map: Dict[str, str]
    annualization_days: int
    grid: GridProcessing
    population_sample: float = 1.0
    beam_osm_id_col: str = "attributeOrigId"
    beam_length_col: str = "linkLength"
    county_area_name: str = "county"
    prepared_skims_group_cols: List[str] = field(default_factory=lambda: ["linkId", "vehicleTypeId", "process"])
    skims_columns: SkimsColumns = field(default_factory=SkimsColumns)
    mapping_columns: MappingColumns = field(default_factory=MappingColumns)
    activity_totals_columns: Dict[str, str] = field(
        default_factory=lambda: {
            "county_fips": "countyfp",
            "tot_vmt": "totVMT",
            "tot_trips": "totTrips",
        }
    )
    concentrations: ConcentrationSettings = field(default_factory=ConcentrationSettings)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProcessingSettings":
        pollutants_map = _build_pollutants_map(
            payload.get("pollutants"),
            payload.get("pollutants_map"),
        )
        configured_pollutants = [pollutant for pollutant in pollutants if pollutant in pollutants_map]
        if not pollutants_map:
            raise ValueError("processing.pollutants must contain at least one pollutant")
        return cls(
            pollutants=configured_pollutants,
            pollutants_map=pollutants_map,
            annualization_days=_required_int(payload.get("annualization_days"), "processing.annualization_days"),
            population_sample=float(payload.get("population_sample") or 1.0),
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
            activity_totals_columns={
                "county_fips": _optional_string(
                    (payload.get("activity_totals_columns", {}) or payload.get("county_activity_totals_columns", {}) or {}).get("county_fips")
                ) or "countyfp",
                "tot_vmt": _optional_string(
                    (payload.get("activity_totals_columns", {}) or payload.get("county_activity_totals_columns", {}) or {}).get("tot_vmt")
                ) or "totVMT",
                "tot_trips": _optional_string(
                    (payload.get("activity_totals_columns", {}) or payload.get("county_activity_totals_columns", {}) or {}).get("tot_trips")
                ) or "totTrips",
            },
            concentrations=ConcentrationSettings.from_dict(
                {
                    **(payload.get("dispersion", {}) or {}),
                    "concentration_factor": payload.get("concentration_factor", None),
                }
            ),
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
                "simulation_network_folder": payload["inputs"]["simulation_network_folder"],
                "osm_network_folder": payload["inputs"]["osm_network_folder"],
                "households_asim_out": payload["inputs"]["households_asim_out"],
                "persons_asim_out": payload["inputs"]["persons_asim_out"],
            },
            "processing": {
                "emissions": {
                    "annualization_days": payload["processing"]["annualization_days"],
                    "population_sample": payload["processing"]["population_sample"],
                    "activity_totals_file": payload["inputs"]["activity_totals_file"],
                    "activity_totals_columns": payload["processing"]["activity_totals_columns"],
                    "pollutants": list(payload["processing"]["pollutants_map"].values()),
                    "pollutants_map": payload["processing"]["pollutants_map"],
                },
                "dispersions": {
                    "inmap": {
                        "isrm_zarr": payload["inputs"]["isrm_zarr"],
                        "isrm_nox_to_no2_matrix_npz": payload["inputs"].get("isrm_nox_to_no2_matrix_npz"),
                        "grid_path": payload["processing"]["grid"]["inmap_grid_path"],
                        "grid_epsg": payload["processing"]["grid"]["inmap_grid_epsg"],
                        "grid_id": payload["processing"]["mapping_columns"]["grid_id"],
                    },
                    "aermod": {
                        "grid_path": payload["processing"]["grid"]["aermod_grid_path"],
                        "grid_epsg": payload["processing"]["grid"]["aermod_grid_epsg"],
                        "grid_id": payload["processing"]["grid"]["aermod_grid_id"],
                    },
                },
            },
            "outputs": {
                "output_dir": payload["outputs"]["output_dir"],
                "exposure_table": payload["outputs"]["exposure_table"],
            },
        }

    def output_root(self) -> Path:
        return Path(self.outputs.output_dir)
