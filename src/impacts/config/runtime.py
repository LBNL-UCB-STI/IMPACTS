from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .defaults import pollutants as canonical_pollutants


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
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value: {value}") from exc


def _required_float(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"Missing required value: {label}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {label}: {value}") from exc


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


def _reject_unknown_keys(payload: Dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unsupported keys under {label}: {unknown}")


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


def _normalize_runtime_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    _reject_unknown_keys(payload, {"run", "shared", "impacts", "__source_root__"}, "root")
    if "run" not in payload or "shared" not in payload or "impacts" not in payload:
        raise ValueError("Runtime config must use the maintained YAML shape: run, shared, impacts")
    return {
        "run": dict(payload.get("run", {}) or {}),
        "shared": dict(payload.get("shared", {}) or {}),
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
class InmapDispersion:
    isrm_zarr: str
    isrm_nox_to_no2_matrix_npz: str
    grid_path: str
    grid_id: Optional[str] = None
    grid_epsg: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InmapDispersion":
        _reject_unknown_keys(
            payload,
            {"isrm_zarr", "isrm_nox_to_no2_matrix_npz", "grid_path", "grid_id", "grid_epsg"},
            "impacts.dispersions.inmap",
        )
        return cls(
            isrm_zarr=_required_string(payload.get("isrm_zarr"), "impacts.dispersions.inmap.isrm_zarr"),
            isrm_nox_to_no2_matrix_npz=_required_string(
                payload.get("isrm_nox_to_no2_matrix_npz"),
                "impacts.dispersions.inmap.isrm_nox_to_no2_matrix_npz",
            ),
            grid_path=_required_string(payload.get("grid_path"), "impacts.dispersions.inmap.grid_path"),
            grid_id=_optional_string(payload.get("grid_id")),
            grid_epsg=_optional_int(payload.get("grid_epsg")),
        )


@dataclass(frozen=True)
class Dispersions:
    inmap: InmapDispersion

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Dispersions":
        _reject_unknown_keys(payload, {"inmap"}, "impacts.dispersions")
        return cls(
            inmap=InmapDispersion.from_dict(dict(payload.get("inmap", {}) or {})),
        )


@dataclass(frozen=True)
class Emissions:
    simulation_network_folder: str
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
                "simulation_network_folder",
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
            simulation_network_folder=_required_string(
                payload.get("simulation_network_folder"),
                "impacts.emissions.simulation_network_folder",
            ),
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

    @property
    def concentration_factor(self) -> float:
        return 28766.639


@dataclass(frozen=True)
class Impacts:
    local_input_folder: str
    local_output_folder: str
    emissions: Emissions
    dispersions: Dispersions

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Impacts":
        _reject_unknown_keys(payload, {"local_input_folder", "local_output_folder", "emissions", "dispersions"}, "impacts")
        return cls(
            local_input_folder=_required_string(payload.get("local_input_folder"), "impacts.local_input_folder"),
            local_output_folder=_required_string(payload.get("local_output_folder"), "impacts.local_output_folder"),
            emissions=Emissions.from_dict(dict(payload.get("emissions", {}) or {})),
            dispersions=Dispersions.from_dict(dict(payload.get("dispersions", {}) or {})),
        )


@dataclass(frozen=True)
class ImpactsRuntimeConfig:
    run: Run
    shared: Shared
    impacts: Impacts

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImpactsRuntimeConfig":
        payload = _normalize_runtime_payload(payload)
        return cls(
            run=Run.from_dict(payload.get("run", {}) or {}),
            shared=Shared.from_dict(payload.get("shared", {}) or {}),
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
            "impacts": {
                "local_input_folder": self.impacts.local_input_folder,
                "local_output_folder": self.impacts.local_output_folder,
                "emissions": {
                    "simulation_network_folder": self.impacts.emissions.simulation_network_folder,
                    "osm_network_folder": self.impacts.emissions.osm_network_folder,
                    "emissions_rates_folder": self.impacts.emissions.emissions_rates_folder,
                    "activity_totals_file": self.impacts.emissions.activity_totals_file,
                    "annualization_days": self.impacts.emissions.annualization_days,
                    "population_sample": self.impacts.emissions.population_sample,
                    "pollutants_map": dict(self.impacts.emissions.pollutants_map),
                },
                "dispersions": {
                    "inmap": {
                        "isrm_zarr": self.impacts.dispersions.inmap.isrm_zarr,
                        "isrm_nox_to_no2_matrix_npz": self.impacts.dispersions.inmap.isrm_nox_to_no2_matrix_npz,
                        "grid_path": self.impacts.dispersions.inmap.grid_path,
                        "grid_id": self.impacts.dispersions.inmap.grid_id,
                        "grid_epsg": self.impacts.dispersions.inmap.grid_epsg,
                    },
                },
            },
        }

    def output_root(self) -> Path:
        return Path(self.impacts.local_output_folder)
