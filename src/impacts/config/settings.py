from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .defaults import pollutants as canonical_pollutants
from ._coerce import _required_string, _optional_string, _required_int, _optional_int, _required_float, _optional_float, _required_bool, _coerce_string_list, _reject_unknown_keys


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


def _build_pollutants_map(value: Any) -> Dict[str, str]:
    mapping = _coerce_string_map(value)
    if not mapping:
        raise ValueError("Missing required value: impacts.emissions.pollutants_map")
    unknown = sorted(set(mapping.keys()) - set(canonical_pollutants))
    if unknown:
        raise ValueError(
            f"impacts.emissions.pollutants_map contains unsupported canonical pollutants: {unknown}. "
            f"Expected only {canonical_pollutants}"
        )
    return mapping


def _normalize_settings_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    _reject_unknown_keys(payload, {"run", "shared", "beam", "impacts", "__source_root__"}, "root")
    if "run" not in payload or "shared" not in payload or "beam" not in payload or "impacts" not in payload:
        raise ValueError("Settings file must use the maintained YAML shape: run, shared, beam, impacts")
    return {
        "run": dict(payload.get("run", {}) or {}),
        "shared": dict(payload.get("shared", {}) or {}),
        "beam": dict(payload.get("beam", {}) or {}),
        "impacts": dict(payload.get("impacts", {}) or {}),
    }


@dataclass(frozen=True)
class GeographyFips:
    state: str
    counties: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GeographyFips":
        _reject_unknown_keys(payload, {"state", "counties"}, "shared.geography.FIPS")
        return cls(
            state=_required_string(payload.get("state"), "shared.geography.FIPS.state"),
            counties=_coerce_string_list(payload.get("counties")),
        )


@dataclass(frozen=True)
class Geography:
    fips: GeographyFips
    local_crs: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Geography":
        _reject_unknown_keys(payload, {"FIPS", "local_crs"}, "shared.geography")
        return cls(
            fips=GeographyFips.from_dict(dict(payload.get("FIPS", {}) or {})),
            local_crs=_required_string(payload.get("local_crs"), "shared.geography.local_crs"),
        )


@dataclass(frozen=True)
class Shared:
    geography: Geography

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Shared":
        _reject_unknown_keys(payload, {"geography"}, "shared")
        return cls(
            geography=Geography.from_dict(dict(payload.get("geography", {}) or {})),
        )


@dataclass(frozen=True)
class Run:
    region: str
    scenario: str
    start_year: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Run":
        _reject_unknown_keys(payload, {"region", "scenario", "start_year"}, "run")
        return cls(
            region=_required_string(payload.get("region"), "run.region"),
            scenario=_required_string(payload.get("scenario"), "run.scenario"),
            start_year=_required_int(payload.get("start_year"), "run.start_year"),
        )


@dataclass(frozen=True)
class Beam:
    local_input_folder: str
    local_output_folder: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Beam":
        _reject_unknown_keys(payload, {"local_input_folder", "local_output_folder"}, "beam")
        return cls(
            local_input_folder=_required_string(payload.get("local_input_folder"), "beam.local_input_folder"),
            local_output_folder=_required_string(payload.get("local_output_folder"), "beam.local_output_folder"),
        )


@dataclass(frozen=True)
class InmapDispersion:
    enabled: bool
    isrm_zarr: Optional[str] = None
    isrm_nox_to_no2_ratios_file: Optional[str] = None
    grid_path: Optional[str] = None
    grid_id: Optional[str] = None
    grid_epsg: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InmapDispersion":
        _reject_unknown_keys(
            payload,
            {
                "enabled",
                "isrm_zarr",
                "isrm_nox_to_no2_ratios_file",
                "grid_path",
                "grid_id",
                "grid_epsg",
            },
            "impacts.dispersions.inmap",
        )
        enabled = _required_bool(payload.get("enabled"), "impacts.dispersions.inmap.enabled")
        result = cls(
            enabled=enabled,
            isrm_zarr=_optional_string(payload.get("isrm_zarr")),
            isrm_nox_to_no2_ratios_file=_optional_string(payload.get("isrm_nox_to_no2_ratios_file")),
            grid_path=_optional_string(payload.get("grid_path")),
            grid_id=_optional_string(payload.get("grid_id")),
            grid_epsg=_optional_int(payload.get("grid_epsg")),
        )
        if result.enabled:
            if not result.isrm_zarr:
                raise ValueError("Missing required value: impacts.dispersions.inmap.isrm_zarr")
            if not result.isrm_nox_to_no2_ratios_file:
                raise ValueError("Missing required value: impacts.dispersions.inmap.isrm_nox_to_no2_ratios_file")
            if not result.grid_path:
                raise ValueError("Missing required value: impacts.dispersions.inmap.grid_path")
        return result


@dataclass(frozen=True)
class AermodDispersion:
    enabled: bool
    grid_size_meters: Optional[float] = None
    asrv_patterns_file: Optional[str] = None
    asrv_patterns_epsg: Optional[int] = None
    asrv_nox_to_no2_ratios_file: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AermodDispersion":
        _reject_unknown_keys(
            payload,
            {"enabled", "grid_size_meters", "asrv_patterns_file", "asrv_patterns_epsg", "asrv_nox_to_no2_ratios_file"},
            "impacts.dispersions.aermod",
        )
        enabled = _required_bool(payload.get("enabled"), "impacts.dispersions.aermod.enabled")
        result = cls(
            enabled=enabled,
            grid_size_meters=_optional_float(payload.get("grid_size_meters")),
            asrv_patterns_file=_optional_string(payload.get("asrv_patterns_file")),
            asrv_patterns_epsg=_optional_int(payload.get("asrv_patterns_epsg")),
            asrv_nox_to_no2_ratios_file=_optional_string(payload.get("asrv_nox_to_no2_ratios_file")),
        )
        if result.enabled:
            if result.grid_size_meters is None:
                raise ValueError("Missing required value: impacts.dispersions.aermod.grid_size_meters")
            if not result.asrv_patterns_file:
                raise ValueError("Missing required value: impacts.dispersions.aermod.asrv_patterns_file")
        return result


@dataclass(frozen=True)
class Dispersions:
    inmap: InmapDispersion
    aermod: AermodDispersion

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Dispersions":
        _reject_unknown_keys(payload, {"inmap", "aermod"}, "impacts.dispersions")
        return cls(
            inmap=InmapDispersion.from_dict(dict(payload.get("inmap", {}) or {})),
            aermod=AermodDispersion.from_dict(dict(payload.get("aermod", {}) or {})),
        )


@dataclass(frozen=True)
class Emissions:
    osm_network_folder: str
    emissions_rates_folder: str
    annualization_days: int
    population_sample: float
    pollutants_map: Dict[str, str]
    activity_totals_file: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Emissions":
        _reject_unknown_keys(
            payload,
            {
                "osm_network_folder",
                "emissions_rates_folder",
                "activity_totals_file",
                "annualization_days",
                "population_sample",
                "pollutants_map",
            },
            "impacts.emissions",
        )
        return cls(
            osm_network_folder=_required_string(
                payload.get("osm_network_folder"),
                "impacts.emissions.osm_network_folder",
            ),
            emissions_rates_folder=_required_string(
                payload.get("emissions_rates_folder"),
                "impacts.emissions.emissions_rates_folder",
            ),
            annualization_days=_required_int(
                payload.get("annualization_days"),
                "impacts.emissions.annualization_days",
            ),
            population_sample=_required_float(
                payload.get("population_sample"),
                "impacts.emissions.population_sample",
            ),
            pollutants_map=_build_pollutants_map(payload.get("pollutants_map")),
            activity_totals_file=_optional_string(payload.get("activity_totals_file")),
        )

    @property
    def pollutants(self) -> List[str]:
        return [pollutant for pollutant in canonical_pollutants if pollutant in self.pollutants_map]

    @property
    def beam_osm_id_col(self) -> str:
        return "attributeOrigId"

    @property
    def beam_length_col(self) -> str:
        return "linkLength"

    @property
    def prepared_skims_group_cols(self) -> List[str]:
        return ["linkId", "vehicleTypeId", "process"]

    @property
    def activity_totals_columns(self) -> Dict[str, str]:
        return {
            "county_fips": "countyfp",
            "tot_vmt": "totVMT",
            "tot_trips": "totTrips",
        }

    @property
    def mapping_columns(self) -> Dict[str, str]:
        return {
            "link_id": "edge_linkId",
            "proportion": "zone_edge_proportion",
        }

@dataclass(frozen=True)
class Impacts:
    local_input_folder: str
    local_output_folder: str
    emissions: Emissions
    dispersions: Dispersions
    exposure: "Exposure"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Impacts":
        _reject_unknown_keys(
            payload,
            {"local_input_folder", "local_output_folder", "emissions", "dispersions", "exposure"},
            "impacts",
        )
        return cls(
            local_input_folder=_required_string(payload.get("local_input_folder"), "impacts.local_input_folder"),
            local_output_folder=_required_string(payload.get("local_output_folder"), "impacts.local_output_folder"),
            emissions=Emissions.from_dict(dict(payload.get("emissions", {}) or {})),
            dispersions=Dispersions.from_dict(dict(payload.get("dispersions", {}) or {})),
            exposure=Exposure.from_dict(dict(payload.get("exposure", {}) or {})),
        )


@dataclass(frozen=True)
class Exposure:
    enabled: bool
    population_folder: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Exposure":
        _reject_unknown_keys(payload, {"enabled", "population_folder"}, "impacts.exposure")
        result = cls(
            enabled=_required_bool(payload.get("enabled"), "impacts.exposure.enabled"),
            population_folder=_optional_string(payload.get("population_folder")),
        )
        if result.enabled and not result.population_folder:
            raise ValueError("Missing required value: impacts.exposure.population_folder")
        return result


@dataclass(frozen=True)
class ImpactsSettings:
    run: Run
    shared: Shared
    beam: Beam
    impacts: Impacts

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImpactsSettings":
        payload = _normalize_settings_payload(payload)
        return cls(
            run=Run.from_dict(payload.get("run", {}) or {}),
            shared=Shared.from_dict(payload.get("shared", {}) or {}),
            beam=Beam.from_dict(payload.get("beam", {}) or {}),
            impacts=Impacts.from_dict(payload.get("impacts", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": {
                "region": self.run.region,
                "scenario": self.run.scenario,
                "start_year": self.run.start_year,
            },
            "shared": {
                "geography": {
                    "FIPS": {
                        "state": self.shared.geography.fips.state,
                        "counties": list(self.shared.geography.fips.counties),
                    },
                    "local_crs": self.shared.geography.local_crs,
                },
            },
            "beam": {
                "local_input_folder": self.beam.local_input_folder,
                "local_output_folder": self.beam.local_output_folder,
            },
            "impacts": {
                "local_input_folder": self.impacts.local_input_folder,
                "local_output_folder": self.impacts.local_output_folder,
                "emissions": {
                    "osm_network_folder": self.impacts.emissions.osm_network_folder,
                    "emissions_rates_folder": self.impacts.emissions.emissions_rates_folder,
                    "activity_totals_file": self.impacts.emissions.activity_totals_file,
                    "annualization_days": self.impacts.emissions.annualization_days,
                    "population_sample": self.impacts.emissions.population_sample,
                    "pollutants_map": dict(self.impacts.emissions.pollutants_map),
                },
                "dispersions": {
                    "inmap": {
                        "enabled": self.impacts.dispersions.inmap.enabled,
                        "isrm_zarr": self.impacts.dispersions.inmap.isrm_zarr,
                        "isrm_nox_to_no2_ratios_file": self.impacts.dispersions.inmap.isrm_nox_to_no2_ratios_file,
                        "grid_path": self.impacts.dispersions.inmap.grid_path,
                        "grid_id": self.impacts.dispersions.inmap.grid_id,
                        "grid_epsg": self.impacts.dispersions.inmap.grid_epsg,
                    },
                    "aermod": {
                        "enabled": self.impacts.dispersions.aermod.enabled,
                        "grid_size_meters": self.impacts.dispersions.aermod.grid_size_meters,
                        "asrv_patterns_file": self.impacts.dispersions.aermod.asrv_patterns_file,
                        "asrv_patterns_epsg": self.impacts.dispersions.aermod.asrv_patterns_epsg,
                        "asrv_nox_to_no2_ratios_file": self.impacts.dispersions.aermod.asrv_nox_to_no2_ratios_file,
                    },
                },
                "exposure": {
                    "enabled": self.impacts.exposure.enabled,
                    "population_folder": self.impacts.exposure.population_folder,
                },
            },
        }

    def output_root(self) -> Path:
        return Path(self.impacts.local_output_folder)
