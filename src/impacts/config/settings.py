from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from copy import deepcopy
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import pandas as pd
import yaml

from .defaults import pollutants as canonical_pollutants
from ._coerce import _required_string, _optional_string, _required_int, _optional_int, _required_float, _optional_float, _required_bool, _coerce_string_list, _reject_unknown_keys



_POLLUTANT_SOURCE_ALIASES = {
    "NH3": "NH3",
    "NOX": "NOx",
    "PM25": "PM2_5",
    "PM2_5": "PM2_5",
    "PM2.5": "PM2_5",
    "SOX": "SOx",
    "ROG": "ROG",
    "BC": "BC",
    "BCH": "BC",
    "BCM": "BC",
}


def _canonical_pollutant_from_source(value: str) -> str:
    token = str(value).strip()
    if not token:
        raise ValueError("Configured pollutant names must be non-empty strings.")
    canonical = _POLLUTANT_SOURCE_ALIASES.get(token.upper())
    if canonical is None:
        raise ValueError(
            f"Unsupported pollutant '{token}' in impacts.emissions.pollutants. "
            f"Expected one of {sorted(_POLLUTANT_SOURCE_ALIASES)}"
        )
    return canonical


def _build_source_pollutants(value: Any) -> List[str]:
    pollutants = _coerce_string_list(value)
    if not pollutants:
        raise ValueError("Missing required value: impacts.emissions.pollutants")
    return pollutants


def build_pollutants_map_from_sources(source_pollutants: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for source_pollutant in source_pollutants:
        canonical = _canonical_pollutant_from_source(source_pollutant)
        if canonical in mapping and mapping[canonical] != source_pollutant:
            raise ValueError(
                f"Multiple source pollutants map to canonical pollutant '{canonical}': "
                f"{mapping[canonical]!r}, {source_pollutant!r}"
            )
        mapping[canonical] = source_pollutant
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


def _mapping_payload(value: Any, *, path: str) -> Dict[str, Any]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for {path}")
    return dict(value)


def _validate_activities_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    activities = _mapping_payload(payload, path="impacts.activities")
    _reject_unknown_keys(
        activities,
        {"region_label", "model_year_groups", "project_analysis", "emissions_inventory"},
        "impacts.activities",
    )
    project_analysis = _mapping_payload(activities.get("project_analysis"), path="impacts.activities.project_analysis")
    _reject_unknown_keys(
        project_analysis,
        {"main", "black_carbon", "paved_road_dust"},
        "impacts.activities.project_analysis",
    )
    main = _mapping_payload(project_analysis.get("main"), path="impacts.activities.project_analysis.main")
    _reject_unknown_keys(main, {"folder", "pto_as_process"}, "impacts.activities.project_analysis.main")
    black_carbon = _mapping_payload(
        project_analysis.get("black_carbon"),
        path="impacts.activities.project_analysis.black_carbon",
    )
    _reject_unknown_keys(
        black_carbon,
        {"folder", "pollutant"},
        "impacts.activities.project_analysis.black_carbon",
    )
    paved_road_dust = _mapping_payload(
        project_analysis.get("paved_road_dust"),
        path="impacts.activities.project_analysis.paved_road_dust",
    )
    _reject_unknown_keys(
        paved_road_dust,
        {"folder", "road_category_map"},
        "impacts.activities.project_analysis.paved_road_dust",
    )
    emissions_inventory = _mapping_payload(
        activities.get("emissions_inventory"),
        path="impacts.activities.emissions_inventory",
    )
    _reject_unknown_keys(
        emissions_inventory,
        {"inventory_folder", "fallback_folder", "fuel_map"},
        "impacts.activities.emissions_inventory",
    )
    return activities


def _validate_fleet_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    fleet = _mapping_payload(payload, path="impacts.fleet")
    _reject_unknown_keys(fleet, {"assignment_model", "atlas"}, "impacts.fleet")
    atlas = _mapping_payload(fleet.get("atlas"), path="impacts.fleet.atlas")
    _reject_unknown_keys(atlas, {"fuel_map", "income_bins"}, "impacts.fleet.atlas")
    if atlas:
        fleet["atlas"] = atlas
    return fleet


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
    router_directory: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Beam":
        _reject_unknown_keys(payload, {"local_input_folder", "local_output_folder", "router_directory"}, "beam")
        return cls(
            local_input_folder=_required_string(payload.get("local_input_folder"), "beam.local_input_folder"),
            local_output_folder=_required_string(payload.get("local_output_folder"), "beam.local_output_folder"),
            router_directory=_optional_string(payload.get("router_directory")),
        )


@dataclass(frozen=True)
class InmapDispersion:
    isrm_zarr: Optional[str] = None
    isrm_nox_to_no2_ratios_file: Optional[str] = None
    grid_path: Optional[str] = None
    grid_id: Optional[str] = None
    grid_epsg: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InmapDispersion":
        _reject_unknown_keys(
            payload,
            {"isrm_zarr", "isrm_nox_to_no2_ratios_file", "grid_path", "grid_id", "grid_epsg"},
            "impacts.dispersions.inmap",
        )
        return cls(
            isrm_zarr=_optional_string(payload.get("isrm_zarr")),
            isrm_nox_to_no2_ratios_file=_optional_string(payload.get("isrm_nox_to_no2_ratios_file")),
            grid_path=_optional_string(payload.get("grid_path")),
            grid_id=_optional_string(payload.get("grid_id")),
            grid_epsg=_optional_int(payload.get("grid_epsg")),
        )


@dataclass(frozen=True)
class AermodDispersion:
    grid_size_meters: Optional[float] = None
    asrv_patterns_file: Optional[str] = None
    asrv_patterns_epsg: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "AermodDispersion":
        _reject_unknown_keys(
            payload,
            {"grid_size_meters", "asrv_patterns_file", "asrv_patterns_epsg"},
            "impacts.dispersions.aermod",
        )
        return cls(
            grid_size_meters=_optional_float(payload.get("grid_size_meters")),
            asrv_patterns_file=_optional_string(payload.get("asrv_patterns_file")),
            asrv_patterns_epsg=_optional_int(payload.get("asrv_patterns_epsg")),
        )


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
class EmissionsInventory:
    passenger_file: Optional[str]
    freight_file: Optional[str]
    inventory_folder: Optional[str] = None
    enable_passenger_activity_correction: bool = True
    enable_freight_activity_correction: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EmissionsInventory":
        _reject_unknown_keys(
            payload,
            {
                "inventory_folder",
                "passenger_file",
                "freight_file",
                "enable_passenger_activity_correction",
                "enable_freight_activity_correction",
            },
            "impacts.emissions.inventory",
        )
        inventory_folder = _optional_string(payload.get("inventory_folder"))
        passenger_file = _optional_string(payload.get("passenger_file"))
        freight_file = _optional_string(payload.get("freight_file"))
        if inventory_folder and not passenger_file:
            passenger_file = _find_matching_file(inventory_folder, ("passenger",), required=False)
        if inventory_folder and not freight_file:
            freight_file = _find_matching_file(inventory_folder, ("freight",), required=False)
        return cls(
            inventory_folder=inventory_folder,
            passenger_file=passenger_file,
            freight_file=freight_file,
            enable_passenger_activity_correction=_required_bool(
                payload.get("enable_passenger_activity_correction", True),
                "impacts.emissions.inventory.enable_passenger_activity_correction",
            ),
            enable_freight_activity_correction=_required_bool(
                payload.get("enable_freight_activity_correction", True),
                "impacts.emissions.inventory.enable_freight_activity_correction",
            ),
        )


@dataclass(frozen=True)
class Emissions:
    @dataclass(frozen=True)
    class Defaults:
        @dataclass(frozen=True)
        class AnnualizationDays:
            light_duty: float
            medium_heavy_duty: float

            @classmethod
            def from_dict(cls, payload: Dict[str, Any]) -> "Emissions.Defaults.AnnualizationDays":
                if not isinstance(payload, dict):
                    raise ValueError("Expected mapping for impacts.emissions.default_annualization_days")
                _reject_unknown_keys(
                    payload,
                    {"light_duty", "medium_heavy_duty"},
                    "impacts.emissions.default_annualization_days",
                )
                return cls(
                    light_duty=_required_float(
                        payload.get("light_duty"),
                        "impacts.emissions.default_annualization_days.light_duty",
                    ),
                    medium_heavy_duty=_required_float(
                        payload.get("medium_heavy_duty"),
                        "impacts.emissions.default_annualization_days.medium_heavy_duty",
                    ),
                )
        default_annualization_days: "Emissions.Defaults.AnnualizationDays"

    osm_network_folder: Optional[str]
    rates_folder: str
    inventory: EmissionsInventory
    vehicle_category_metadata_file: Optional[str]
    defaults: "Emissions.Defaults"
    source_pollutants: List[str]
    beam: "ImpactsBeamProcessing"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Emissions":
        _reject_unknown_keys(
            payload,
            {
                "osm_network_folder",
                "rates_folder",
                "inventory_folder",
                "vehicle_category_metadata_file",
                "enable_passenger_activity_correction",
                "enable_freight_activity_correction",
                "pollutants",
                "passenger_vehicle_types_file",
                "freight_vehicle_types_file",
                "include_non_osm_car_links",
                "include_passenger",
                "include_freight",
                "default_annualization_days",
            },
            "impacts.emissions",
        )
        source_pollutants = _build_source_pollutants(payload.get("pollutants"))
        build_pollutants_map_from_sources(source_pollutants)
        return cls(
            osm_network_folder=_optional_string(payload.get("osm_network_folder")),
            rates_folder=_required_string(
                payload.get("rates_folder"),
                "impacts.emissions.rates_folder",
            ),
            inventory=EmissionsInventory.from_dict(
                {
                    "inventory_folder": payload.get("inventory_folder"),
                    "enable_passenger_activity_correction": payload.get("enable_passenger_activity_correction"),
                    "enable_freight_activity_correction": payload.get("enable_freight_activity_correction"),
                }
            ),
            vehicle_category_metadata_file=_optional_string(payload.get("vehicle_category_metadata_file")),
            defaults=Emissions.Defaults(
                default_annualization_days=Emissions.Defaults.AnnualizationDays.from_dict(
                    dict(payload.get("default_annualization_days", {}) or {})
                )
            ),
            source_pollutants=source_pollutants,
            beam=ImpactsBeamProcessing.from_dict(payload),
        )

    @property
    def pollutants(self) -> List[str]:
        mapping = self.pollutants_map
        return [pollutant for pollutant in canonical_pollutants if pollutant in self.pollutants_map]

    @property
    def pollutants_map(self) -> Dict[str, str]:
        return build_pollutants_map_from_sources(list(self.source_pollutants))

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
    def mapping_columns(self) -> Dict[str, str]:
        return {
            "link_id": "linkId",
            "proportion": "proportion",
        }


@dataclass(frozen=True)
class ImpactsBeamProcessing:
    passenger_vehicle_types_file: Optional[str]
    freight_vehicle_types_file: Optional[str]
    include_non_osm_car_links: bool = False
    include_passenger: bool = True
    include_freight: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImpactsBeamProcessing":
        return cls(
            passenger_vehicle_types_file=_optional_string(payload.get("passenger_vehicle_types_file")),
            freight_vehicle_types_file=_optional_string(payload.get("freight_vehicle_types_file")),
            include_non_osm_car_links=_required_bool(
                payload.get("include_non_osm_car_links", False), "impacts.emissions.include_non_osm_car_links"
            ),
            include_passenger=_required_bool(
                payload.get("include_passenger", True), "impacts.emissions.include_passenger"
            ),
            include_freight=_required_bool(
                payload.get("include_freight", True), "impacts.emissions.include_freight"
            ),
        )


@dataclass(frozen=True)
class AnalysisPollutantTarget:
    columns: List[str] = field(default_factory=list)
    prefixes: List[str] = field(default_factory=list)
    exclude_columns: List[str] = field(default_factory=list)
    exclude_prefixes: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any], *, path: str) -> "AnalysisPollutantTarget":
        _reject_unknown_keys(payload, {"columns", "prefixes", "exclude_columns", "exclude_prefixes"}, path)
        columns = _coerce_string_list(payload.get("columns"))
        prefixes = _coerce_string_list(payload.get("prefixes"))
        exclude_columns = _coerce_string_list(payload.get("exclude_columns"))
        exclude_prefixes = _coerce_string_list(payload.get("exclude_prefixes"))
        if not columns and not prefixes:
            raise ValueError(f"{path} must define at least one of columns or prefixes")
        return cls(
            columns=columns,
            prefixes=prefixes,
            exclude_columns=exclude_columns,
            exclude_prefixes=exclude_prefixes,
        )


@dataclass(frozen=True)
class AnalysisTarget:
    name: str
    pollutants: Dict[str, AnalysisPollutantTarget]

    @classmethod
    def from_dict(cls, name: str, payload: Dict[str, Any], *, path: str) -> "AnalysisTarget":
        raw_pollutants = dict(payload or {})
        if not raw_pollutants:
            raise ValueError(f"Missing required value: {path}")
        pollutants = {
            str(pollutant).strip(): AnalysisPollutantTarget.from_dict(
                dict(selector or {}),
                path=f"{path}.{str(pollutant).strip()}",
            )
            for pollutant, selector in raw_pollutants.items()
        }
        return cls(
            name=_required_string(name, f"{path} key"),
            pollutants=pollutants,
        )


@dataclass(frozen=True)
class AnalysisSectorTarget:
    source: str
    sector: str
    annual_pm25_short_tons: Optional[float] = None
    annual_nox_short_tons: Optional[float] = None


def _parse_analysis_sector_targets(
    *,
    raw_pm25_targets: Any,
    raw_nox_targets: Any,
    path_prefix: str,
) -> List[AnalysisSectorTarget]:
    sector_targets_by_key: Dict[tuple[str, str], AnalysisSectorTarget] = {}
    for pollutant_name, raw_sector_targets in (
        ("pm25", raw_pm25_targets),
        ("nox", raw_nox_targets),
    ):
        if raw_sector_targets is None:
            continue
        if not isinstance(raw_sector_targets, dict):
            raise ValueError(
                f"{path_prefix}.{pollutant_name}_annual_short_tons must be a mapping of source -> sector -> annual target"
            )
        for source, sectors in raw_sector_targets.items():
            source_name = str(source).strip()
            source_path = f"{path_prefix}.{pollutant_name}_annual_short_tons.{source_name}"
            if not isinstance(sectors, dict):
                raise ValueError(f"{source_path} must be a mapping of sector -> annual target")
            for sector, value in sectors.items():
                sector_name = str(sector).strip()
                key = (source_name, sector_name)
                existing = sector_targets_by_key.get(key)
                if existing is None:
                    existing = AnalysisSectorTarget(source=source_name, sector=sector_name)
                if pollutant_name == "pm25":
                    existing = AnalysisSectorTarget(
                        source=existing.source,
                        sector=existing.sector,
                        annual_pm25_short_tons=_required_float(value, f"{source_path}.{sector_name}"),
                        annual_nox_short_tons=existing.annual_nox_short_tons,
                    )
                else:
                    existing = AnalysisSectorTarget(
                        source=existing.source,
                        sector=existing.sector,
                        annual_pm25_short_tons=existing.annual_pm25_short_tons,
                        annual_nox_short_tons=_required_float(value, f"{source_path}.{sector_name}"),
                    )
                sector_targets_by_key[key] = existing
    return list(sector_targets_by_key.values())


@dataclass(frozen=True)
class Analysis:
    inventory_file: Optional[str] = None
    inventory_label: Optional[str] = None
    inventory_targets: List[AnalysisTarget] = field(default_factory=list)
    sector_targets: List[AnalysisSectorTarget] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Analysis":
        _reject_unknown_keys(
            payload,
            {
                "inventory_file",
                "inventory_label",
                "targets",
                "inventory_targets",
            },
            "impacts.analysis",
        )
        raw_targets = payload.get("targets")
        raw_inventory_targets = payload.get("inventory_targets")
        if raw_targets is not None and not isinstance(raw_targets, dict):
            raise ValueError(
                "impacts.analysis.targets must be a mapping containing pm25_annual_short_tons and/or "
                "nox_annual_short_tons"
            )
        raw_targets_dict = dict(raw_targets or {})
        _reject_unknown_keys(
            raw_targets_dict,
            {"pm25_annual_short_tons", "nox_annual_short_tons"},
            "impacts.analysis.targets",
        )
        raw_pm25_targets = raw_targets_dict.get("pm25_annual_short_tons")
        raw_nox_targets = raw_targets_dict.get("nox_annual_short_tons")
        if (
            raw_targets is None
            and raw_inventory_targets is None
            and raw_pm25_targets is None
            and raw_nox_targets is None
            and payload.get("inventory_file") is None
            and payload.get("inventory_label") is None
        ):
            return cls()
        if raw_inventory_targets is None:
            inventory_targets: List[AnalysisTarget] = []
        elif not isinstance(raw_inventory_targets, dict):
            raise ValueError("impacts.analysis.inventory_targets must be a mapping of target_name -> pollutant selectors")
        else:
            inventory_targets = [
                AnalysisTarget.from_dict(
                    str(name).strip(),
                    dict(target or {}),
                    path=f"impacts.analysis.inventory_targets.{str(name).strip()}",
                )
                for name, target in raw_inventory_targets.items()
            ]
        sector_targets = _parse_analysis_sector_targets(
            raw_pm25_targets=raw_pm25_targets,
            raw_nox_targets=raw_nox_targets,
            path_prefix="impacts.analysis.targets",
        )
        inventory_file = _optional_string(payload.get("inventory_file"))
        inventory_label = _optional_string(payload.get("inventory_label"))
        if inventory_targets and (not inventory_file or not inventory_label):
            raise ValueError(
                "impacts.analysis.inventory_file and impacts.analysis.inventory_label are required when impacts.analysis.inventory_targets is configured"
            )
        return cls(
            inventory_file=inventory_file,
            inventory_label=inventory_label,
            inventory_targets=inventory_targets,
            sector_targets=sector_targets,
        )

@dataclass(frozen=True)
class Impacts:
    local_output_folder: str
    local_input_folder: str
    seed: int
    scenario: Optional[str]
    pipeline: "Pipeline"
    activities: Dict[str, Any]
    fleet: Dict[str, Any]
    emissions: Emissions
    dispersions: Dispersions
    population: "Population"
    analysis: Analysis = field(default_factory=Analysis)

    @classmethod
    def from_dict(cls, impacts_settings: Dict[str, Any]) -> "Impacts":
        _reject_unknown_keys(
            impacts_settings,
            {
                "local_output_folder", "local_input_folder", "seed", "scenario",
                "pipeline", "population",
                "activities", "fleet", "emissions", "dispersions", "analysis",
            },
            "impacts",
        )
        population = Population.from_dict(dict(impacts_settings.get("population", {}) or {}))
        activities = _validate_activities_settings(impacts_settings.get("activities", {}) or {})
        fleet = _validate_fleet_settings(impacts_settings.get("fleet", {}) or {})
        emissions_payload = dict(impacts_settings.get("emissions", {}) or {})
        scenario = _optional_string(impacts_settings.get("scenario"))
        local_input_folder = impacts_settings.get("local_input_folder")
        emissions_folder = (
            str(Path(population.vehicle_folder) / "emissions") if population.vehicle_folder else None
        )
        if population.vehicle_folder:
            if not emissions_payload.get("passenger_vehicle_types_file"):
                derived = _find_em_vehicle_types_file(population.vehicle_folder, "atlas")
                if derived:
                    emissions_payload["passenger_vehicle_types_file"] = derived
            if not emissions_payload.get("freight_vehicle_types_file"):
                derived = _find_em_vehicle_types_file(population.vehicle_folder, "frism")
                if derived:
                    emissions_payload["freight_vehicle_types_file"] = derived
        if not emissions_payload.get("vehicle_category_metadata_file"):
            derived = _default_vehicle_category_metadata_file(emissions_folder)
            if derived:
                emissions_payload["vehicle_category_metadata_file"] = derived
        if not emissions_payload.get("rates_folder"):
            derived = _default_emissions_rates_folder(emissions_folder, scenario)
            if derived:
                emissions_payload["rates_folder"] = derived
        if not emissions_payload.get("inventory_folder"):
            derived = _default_emissions_inventory_folder(local_input_folder, scenario)
            if derived:
                emissions_payload["inventory_folder"] = derived
        result = cls(
            local_output_folder=_required_string(impacts_settings.get("local_output_folder"), "impacts.local_output_folder"),
            local_input_folder=_required_string(impacts_settings.get("local_input_folder"), "impacts.local_input_folder"),
            seed=int(impacts_settings.get("seed") or 0),
            scenario=scenario,
            pipeline=Pipeline.from_dict(dict(impacts_settings.get("pipeline", {}) or {})),
            activities=activities,
            fleet=fleet,
            emissions=Emissions.from_dict(emissions_payload),
            dispersions=Dispersions.from_dict(dict(impacts_settings.get("dispersions", {}) or {})),
            population=population,
            analysis=Analysis.from_dict(dict(impacts_settings.get("analysis", {}) or {})),
        )
        if result.pipeline.inmap:
            if not result.dispersions.inmap.isrm_zarr:
                raise ValueError("Missing required value: impacts.dispersions.inmap.isrm_zarr")
            if not result.dispersions.inmap.isrm_nox_to_no2_ratios_file:
                raise ValueError("Missing required value: impacts.dispersions.inmap.isrm_nox_to_no2_ratios_file")
            if not result.dispersions.inmap.grid_path:
                raise ValueError("Missing required value: impacts.dispersions.inmap.grid_path")
        if result.pipeline.aermod:
            if result.dispersions.aermod.grid_size_meters is None:
                raise ValueError("Missing required value: impacts.dispersions.aermod.grid_size_meters")
            if not result.dispersions.aermod.asrv_patterns_file:
                raise ValueError("Missing required value: impacts.dispersions.aermod.asrv_patterns_file")
        if result.analysis.sector_targets and not result.emissions.vehicle_category_metadata_file:
            raise ValueError(
                "impacts.emissions.vehicle_category_metadata_file is required when annual sector targets are configured"
            )
        return result


@dataclass(frozen=True)
class Pipeline:
    @dataclass(frozen=True)
    class Presim:
        activities: bool = True
        fleet: bool = True

        @classmethod
        def from_dict(cls, payload: Dict[str, Any]) -> "Pipeline.Presim":
            _reject_unknown_keys(payload, {"activities", "fleet"}, "impacts.pipeline.presim")
            def _flag(key: str) -> bool:
                val = payload.get(key)
                return True if val is None else bool(val)
            return cls(
                activities=_flag("activities"),
                fleet=_flag("fleet"),
            )

    @dataclass(frozen=True)
    class Postsim:
        emissions: bool = True
        inmap: bool = True
        aermod: bool = True
        exposure: bool = True

        @classmethod
        def from_dict(cls, payload: Dict[str, Any]) -> "Pipeline.Postsim":
            _reject_unknown_keys(
                payload,
                {"emissions", "inmap", "aermod", "exposure"},
                "impacts.pipeline.postsim",
            )
            def _flag(key: str) -> bool:
                val = payload.get(key)
                return True if val is None else bool(val)
            return cls(
                emissions=_flag("emissions"),
                inmap=_flag("inmap"),
                aermod=_flag("aermod"),
                exposure=_flag("exposure"),
            )

    presim: "Pipeline.Presim" = field(default_factory=Presim)
    postsim: "Pipeline.Postsim" = field(default_factory=Postsim)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Pipeline":
        _reject_unknown_keys(payload, {"presim", "postsim"}, "impacts.pipeline")
        return cls(
            presim=Pipeline.Presim.from_dict(dict(payload.get("presim", {}) or {})),
            postsim=Pipeline.Postsim.from_dict(dict(payload.get("postsim", {}) or {})),
        )

    @property
    def activities(self) -> bool:
        return self.presim.activities

    @property
    def fleet(self) -> bool:
        return self.presim.fleet

    @property
    def emissions(self) -> bool:
        return self.postsim.emissions

    @property
    def inmap(self) -> bool:
        return self.postsim.inmap

    @property
    def aermod(self) -> bool:
        return self.postsim.aermod

    @property
    def exposure(self) -> bool:
        return self.postsim.exposure



@dataclass(frozen=True)
class Population:
    passenger_folder: Optional[str] = None
    freight_folder: Optional[str] = None
    vehicle_folder: Optional[str] = None
    atlas_year: Optional[int] = None
    frism_year: Optional[int] = None
    population_sample: float = 1.0
    transit_sample: float = 1.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Population":
        _reject_unknown_keys(
            payload,
            {
                "passenger_folder",
                "freight_folder",
                "vehicle_folder",
                "atlas_year",
                "frism_year",
                "population_sample",
                "transit_sample",
            },
            "impacts.population",
        )
        return cls(
            passenger_folder=_optional_string(payload.get("passenger_folder")),
            freight_folder=_optional_string(payload.get("freight_folder")),
            vehicle_folder=_optional_string(payload.get("vehicle_folder")),
            atlas_year=_optional_int(payload.get("atlas_year")),
            frism_year=_optional_int(payload.get("frism_year")),
            population_sample=_required_float(payload.get("population_sample", 1.0), "impacts.population.population_sample"),
            transit_sample=_optional_float(payload.get("transit_sample"), default=1.0),
        )


@dataclass(frozen=True)
class ImpactsSettings:
    run: Run
    shared: Shared
    beam: Beam
    impacts: Impacts

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImpactsSettings":
        payload = _normalize_settings_payload(payload)
        beam_payload = dict(payload.get("beam", {}) or {})
        impacts_payload = dict(payload.get("impacts", {}) or {})
        router_directory = beam_payload.get("router_directory")
        if router_directory:
            emissions_payload = dict(impacts_payload.get("emissions", {}) or {})
            if not emissions_payload.get("osm_network_folder"):
                emissions_payload["osm_network_folder"] = router_directory
                impacts_payload = {**impacts_payload, "emissions": emissions_payload}
        return cls(
            run=Run.from_dict(payload.get("run", {}) or {}),
            shared=Shared.from_dict(payload.get("shared", {}) or {}),
            beam=Beam.from_dict(beam_payload),
            impacts=Impacts.from_dict(impacts_payload),
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
                "router_directory": self.beam.router_directory,
            },
            "impacts": {
                "local_output_folder": self.impacts.local_output_folder,
                "local_input_folder": self.impacts.local_input_folder,
                "seed": self.impacts.seed,
                "scenario": self.impacts.scenario,
                "pipeline": {
                    "presim": {
                        "activities": self.impacts.pipeline.presim.activities,
                        "fleet": self.impacts.pipeline.presim.fleet,
                    },
                    "postsim": {
                        "emissions": self.impacts.pipeline.postsim.emissions,
                        "inmap": self.impacts.pipeline.postsim.inmap,
                        "aermod": self.impacts.pipeline.postsim.aermod,
                        "exposure": self.impacts.pipeline.postsim.exposure,
                    },
                },
                "population": {
                    "passenger_folder": self.impacts.population.passenger_folder,
                    "freight_folder": self.impacts.population.freight_folder,
                    "vehicle_folder": self.impacts.population.vehicle_folder,
                    "atlas_year": self.impacts.population.atlas_year,
                    "frism_year": self.impacts.population.frism_year,
                    "population_sample": self.impacts.population.population_sample,
                    "transit_sample": self.impacts.population.transit_sample,
                },
                "activities": self.impacts.activities,
                "fleet": self.impacts.fleet,
                "emissions": {
                        "include_non_osm_car_links": self.impacts.emissions.beam.include_non_osm_car_links,
                        "include_passenger": self.impacts.emissions.beam.include_passenger,
                        "include_freight": self.impacts.emissions.beam.include_freight,
                        "enable_passenger_activity_correction": self.impacts.emissions.inventory.enable_passenger_activity_correction,
                        "enable_freight_activity_correction": self.impacts.emissions.inventory.enable_freight_activity_correction,
                        "default_annualization_days": {
                            "light_duty": self.impacts.emissions.defaults.default_annualization_days.light_duty,
                            "medium_heavy_duty": self.impacts.emissions.defaults.default_annualization_days.medium_heavy_duty,
                        },
                        "pollutants": list(self.impacts.emissions.source_pollutants),
                    },
                    "dispersions": {
                        "inmap": {
                            "isrm_zarr": self.impacts.dispersions.inmap.isrm_zarr,
                            "isrm_nox_to_no2_ratios_file": self.impacts.dispersions.inmap.isrm_nox_to_no2_ratios_file,
                            "grid_path": self.impacts.dispersions.inmap.grid_path,
                            "grid_id": self.impacts.dispersions.inmap.grid_id,
                            "grid_epsg": self.impacts.dispersions.inmap.grid_epsg,
                        },
                        "aermod": {
                            "grid_size_meters": self.impacts.dispersions.aermod.grid_size_meters,
                            "asrv_patterns_file": self.impacts.dispersions.aermod.asrv_patterns_file,
                            "asrv_patterns_epsg": self.impacts.dispersions.aermod.asrv_patterns_epsg,
                        },
                    },
                    "analysis": {
                        **(
                            {
                                "targets": {
                                    "pm25_annual_short_tons": {
                                        source: {
                                            sector: target.annual_pm25_short_tons
                                            for sector, target in sectors.items()
                                            if target.annual_pm25_short_tons is not None
                                        }
                                        for source, sectors in {
                                            source: {
                                                target.sector: target
                                                for target in self.impacts.analysis.sector_targets
                                                if target.source == source
                                            }
                                            for source in dict.fromkeys(target.source for target in self.impacts.analysis.sector_targets)
                                        }.items()
                                        if any(target.annual_pm25_short_tons is not None for target in sectors.values())
                                    },
                                    "nox_annual_short_tons": {
                                        source: {
                                            sector: target.annual_nox_short_tons
                                            for sector, target in sectors.items()
                                            if target.annual_nox_short_tons is not None
                                        }
                                        for source, sectors in {
                                            source: {
                                                target.sector: target
                                                for target in self.impacts.analysis.sector_targets
                                                if target.source == source
                                            }
                                            for source in dict.fromkeys(target.source for target in self.impacts.analysis.sector_targets)
                                        }.items()
                                        if any(target.annual_nox_short_tons is not None for target in sectors.values())
                                    },
                                },
                            }
                            if self.impacts.analysis.sector_targets
                            else {}
                        ),
                        **(
                            {
                                "inventory_file": self.impacts.analysis.inventory_file,
                                "inventory_label": self.impacts.analysis.inventory_label,
                                "inventory_targets": {
                                    target.name: {
                                        pollutant: {
                                            "columns": list(selector.columns),
                                            "prefixes": list(selector.prefixes),
                                            "exclude_columns": list(selector.exclude_columns),
                                            "exclude_prefixes": list(selector.exclude_prefixes),
                                        }
                                        for pollutant, selector in target.pollutants.items()
                                    }
                                    for target in self.impacts.analysis.inventory_targets
                                },
                            }
                            if self.impacts.analysis.inventory_targets
                            else {}
                        ),
                },
            },
        }

    def output_root(self) -> Path:
        return Path(self.impacts.local_output_folder)

# EMFAC workflow helpers
CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parents[2]
DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"

EMFAC_KEY_SCHEMA = {
    "vehicleCategory": "string",
    "fuel": "string",
    "modelYear": "string",
}
EMFAC_ACTIVITY_SCHEMA = {
    **EMFAC_KEY_SCHEMA,
    "population_vehicles": "Float64",
    "total_vmt_vehicle_miles_per_year": "Float64",
}
ATLAS_VEHICLES_SCHEMA = {
    "vehicle_id": "Int64",
    "household_id": "Int64",
    "bodytype": "string",
    "modelyear": "Int64",
    "adopt_fuel": "string",
}
ATLAS_HOUSEHOLDS_SCHEMA = {
    "household_id": "Int64",
    "income_segment": "Float64",
    "income_in_thousands": "Float64",
}
ATLAS_PERSONS_SCHEMA = {
    "household_id": "Int64",
}
FUEL_CONSUMPTION_CATALOG_SCHEMA = {
    "fastsim_id": "string",
    "model_year": "Float64",
    "fuel": "string",
    "charge_behavior": "string",
    "model_trim": "string",
    "msrp_usd": "Float64",
    "fastsim_relative_path": "string",
}
VEHICLE_CATEGORY_METADATA_SCHEMA = {
    "emfac_vehicle_category": "string",
    "idle_time_fraction": "Float64",
}


def _schema_columns(schema: dict[str, str] | None) -> list[str] | None:
    if not schema:
        return None
    return list(schema.keys())


def _apply_table_schema(frame: pd.DataFrame, schema: dict[str, str] | None, *, frame_name: str) -> pd.DataFrame:
    if not schema:
        return frame
    result = frame.copy()
    missing = [column for column in schema if column not in result.columns]
    index_names = [name for name in result.index.names if name is not None]
    if missing and any(column in index_names for column in missing):
        result = result.reset_index()
        missing = [column for column in schema if column not in result.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing schema columns: {missing}")
    for column, dtype_name in schema.items():
        if dtype_name == "string":
            result[column] = result[column].astype("string")
            continue
        if dtype_name == "Float64":
            result[column] = pd.to_numeric(result[column], errors="raise").astype("Float64")
            continue
        if dtype_name == "Int64":
            result[column] = pd.to_numeric(result[column], errors="raise").astype("Int64")
            continue
        raise ValueError(f"Unsupported EMFAC table schema dtype '{dtype_name}' for column '{column}'")
    return result


def _load_yaml_path(path: Path, *sections: str) -> dict:
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    current = data
    for section in sections:
        if not isinstance(current, dict):
            return {}
        current = current.get(section, {})
    return current if isinstance(current, dict) else {}


def _merge_dicts(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _build_emfac_root_from_settings_file(path: Path) -> dict[str, object]:
    from .settings_builder import load_settings_from_yaml
    from ..manifest.file_ops import resolve_path

    settings = load_settings_from_yaml(path)
    activities = settings.impacts.activities if isinstance(settings.impacts.activities, dict) else {}
    fleet_settings = settings.impacts.fleet if isinstance(settings.impacts.fleet, dict) else {}
    fleet: dict[str, object] = {}
    local_input_folder = Path(resolve_path(settings.impacts.local_input_folder, path)).resolve()
    scenario_id = str(settings.impacts.scenario)
    region_label = str(activities.get("region_label") or settings.run.region.upper())
    population_folder = settings.impacts.population.passenger_folder
    atlas = dict(fleet_settings.get("atlas", {}) or {})
    atlas_year = settings.impacts.population.atlas_year
    if atlas_year is not None:
        atlas["year"] = atlas_year
    if population_folder:
        atlas.setdefault("population_folder", population_folder)
    if atlas:
        fleet["atlas"] = atlas
    carriers_folder = settings.impacts.population.freight_folder
    frism = dict(fleet_settings.get("frism", {}) or {})
    frism_year = settings.impacts.population.frism_year
    if frism_year is not None:
        frism["year"] = frism_year
    if carriers_folder:
        frism.setdefault("carriers_folder", carriers_folder)
    if frism:
        fleet["frism"] = frism
    vehicle_folder = settings.impacts.population.vehicle_folder
    scenario = settings.impacts.scenario
    if vehicle_folder and scenario:
        passenger_vehicle_types_file = _find_em_vehicle_types_file(vehicle_folder, "atlas")
        if passenger_vehicle_types_file:
            fleet.setdefault("passenger_vehicle_types_file", passenger_vehicle_types_file)
        freight_vehicle_types_file = _find_em_vehicle_types_file(vehicle_folder, "frism")
        if freight_vehicle_types_file:
            fleet.setdefault("freight_vehicle_types_file", freight_vehicle_types_file)
        else:
            fleet.setdefault("freight_vehicle_types_file", f"{vehicle_folder}/vehicletypes--frism--{scenario}.csv")
        fleet.setdefault("fuel_consumption_catalog", _default_fuel_consumption_catalog(vehicle_folder))
    assignment_model = fleet_settings.get("assignment_model")
    if assignment_model not in (None, ""):
        fleet["assignment_model"] = assignment_model
    return {
        "region": {
            "name": settings.run.region,
            "label": region_label,
        },
        "scenario": {
            "year": settings.run.start_year,
            "name": scenario_id,
        },
        "seed": settings.impacts.seed,
        "output": str(local_input_folder / "emfac"),
        "activities": dict(activities),
        "fleet": dict(fleet),
    }


def _build_activities_config_from_root(emfac_root: dict[str, object]) -> dict[str, object]:
    activities = emfac_root.get("activities", {})
    if not isinstance(activities, dict):
        activities = {}
    defaults: dict[str, object] = {}
    root_region = emfac_root.get("region", {})
    if isinstance(root_region, dict) and "label" in root_region:
        defaults["region_label"] = deepcopy(root_region["label"])
    root_scenario = emfac_root.get("scenario", {})
    if isinstance(root_scenario, dict) and "year" in root_scenario:
        defaults["calendar_year"] = deepcopy(root_scenario["year"])
    if isinstance(root_scenario, dict) and "name" in root_scenario:
        defaults["scenario"] = deepcopy(root_scenario["name"])
    if "output" in emfac_root:
        defaults["outputs"] = deepcopy(emfac_root["output"])
    fleet = emfac_root.get("fleet", {})
    if isinstance(fleet, dict) and "assignment_model" in fleet:
        defaults["assignment_model"] = deepcopy(fleet["assignment_model"])
    return _merge_dicts(defaults, activities)


def _build_fleet_config_from_root(emfac_root: dict[str, object]) -> dict[str, object]:
    fleet = emfac_root.get("fleet", {})
    if not isinstance(fleet, dict):
        fleet = {}
    defaults: dict[str, object] = {}
    root_region = emfac_root.get("region", {})
    root_scenario = emfac_root.get("scenario", {})
    if isinstance(root_region, dict) and "name" in root_region:
        defaults["region"] = deepcopy(root_region["name"])
    if isinstance(root_scenario, dict):
        if "year" in root_scenario:
            defaults["year"] = deepcopy(root_scenario["year"])
        if "name" in root_scenario:
            defaults["scenario"] = deepcopy(root_scenario["name"])
    for key in ("seed", "output"):
        if key in emfac_root:
            defaults[key] = deepcopy(emfac_root[key])
    activities_defaults = _build_activities_config_from_root(emfac_root)
    merged = _merge_dicts(defaults, fleet)
    nested_activities = merged.get("activities", {})
    if not isinstance(nested_activities, dict):
        nested_activities = {}
    merged["activities"] = _merge_dicts(activities_defaults, nested_activities)
    return merged


def resolve_workflow_path(path_like: str | None) -> str:
    if path_like in (None, ""):
        raise ValueError("Expected a configured path value, got an empty value")
    source = Path(str(path_like)).expanduser()
    if not source.is_absolute():
        source = REPO_ROOT / source
    return str(source.resolve())


def _expand_optional_path(value: str | None) -> str | None:
    if value in (None, ""):
        return value
    return resolve_workflow_path(value)


def _find_em_vehicle_types_file(vehicle_folder: str | None, source: str) -> str | None:
    if not vehicle_folder:
        return None
    try:
        folder = Path(resolve_workflow_path(vehicle_folder))
    except ValueError:
        return None
    if not folder.is_dir():
        return None
    matches = sorted(
        str(f) for f in folder.iterdir()
        if f.is_file() and source in f.name.lower() and "--em" in f.name.lower()
    )
    return matches[0] if matches else None


def _default_vehicle_category_metadata_file(rates_root: str | None) -> str | None:
    if not rates_root:
        return None
    return str(Path(rates_root) / "emissions_vehicle_categories.csv")


def _default_emissions_rates_folder(emissions_root: str | None, scenario: str | None) -> str | None:
    if not emissions_root or not scenario:
        return None
    return str(Path(emissions_root) / "activities" / scenario / "rates")


def _default_emissions_inventory_folder(emissions_root: str | None, scenario: str | None) -> str | None:
    if not emissions_root or not scenario:
        return None
    return str(Path(emissions_root) / "activities" / scenario / "inventory")


def _default_fuel_consumption_catalog(vehicle_folder: str | None) -> str | None:
    if not vehicle_folder:
        return None
    return str(Path(vehicle_folder) / "fuel" / "fuel_vehicle_categories.csv")


def _normalize_configured_path(
    path_like: str | None,
    *,
    path_label: str,
    expect_directory: bool = False,
    must_exist: bool = True,
) -> str | None:
    if path_like in (None, ""):
        return None
    resolved = Path(resolve_workflow_path(path_like))
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Configured fleet path '{path_label}' does not exist: {resolved}")
    if expect_directory and must_exist and not resolved.is_dir():
        raise NotADirectoryError(f"Configured fleet path '{path_label}' is not a directory: {resolved}")
    if not expect_directory and must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Configured fleet path '{path_label}' is not a file: {resolved}")
    return str(resolved)


def read_table(
    path_like: str,
    *,
    dtype: str | None = "str",
    columns: list[str] | tuple[str, ...] | None = None,
    schema: dict[str, str] | None = None,
) -> pd.DataFrame:
    resolved = Path(resolve_workflow_path(path_like))
    read_columns = list(columns) if columns is not None else _schema_columns(schema)
    if resolved.suffix.lower() == ".parquet":
        frame = pd.read_parquet(resolved, columns=read_columns)
        if schema is not None:
            return _apply_table_schema(frame, schema, frame_name=str(resolved))
        if dtype == "str":
            return frame.fillna("").astype(str)
        return frame
    read_kwargs = {"dtype": dtype}
    if read_columns is not None:
        read_kwargs["usecols"] = read_columns
    if schema is not None:
        read_kwargs.pop("dtype", None)
        return _apply_table_schema(pd.read_csv(resolved, **read_kwargs), schema, frame_name=str(resolved))
    return pd.read_csv(resolved, **read_kwargs).fillna("")


def _mapping_from_entry(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _folder_from_entry(value: object) -> str | None:
    mapping = _mapping_from_entry(value)
    if mapping is not None:
        folder = mapping.get("folder")
        return None if folder in (None, "") else str(folder)
    return None


def _value_from_entry(value: object, key: str) -> object | None:
    mapping = _mapping_from_entry(value)
    if mapping is not None:
        result = mapping.get(key)
        return None if result in (None, "") else result
    return None


def _find_matching_file(path: str | None, patterns: tuple[str, ...], *, required: bool = True) -> str | None:
    if path in (None, ""):
        return None
    target = Path(resolve_workflow_path(path))
    if target.is_file():
        return str(target)
    if not target.exists():
        if required:
            raise FileNotFoundError(f"Configured path does not exist: {target}")
        return None
    matches = sorted(
        candidate
        for candidate in target.iterdir()
        if candidate.is_file() and any(pattern in candidate.name.lower() for pattern in patterns)
    )
    if matches:
        return str(matches[0])
    if required:
        raise FileNotFoundError(f"No files matching {patterns} found under: {target}")
    return None


def _normalize_pto_as_process(raw: dict) -> dict[str, object]:
    project_analysis = raw.get("project_analysis", {})
    project_analysis_main = _mapping_from_entry(project_analysis.get("main")) or {}
    config = deepcopy(project_analysis_main.get("pto_as_process") or {})
    enabled = bool(config.get("enabled", False))
    targets = [str(value).strip() for value in config.get("vehicle_category", [])]
    return {
        "enabled": enabled,
        "targets": targets,
    }


def _normalize_activities_inputs(raw: dict) -> dict:
    project_analysis = raw.get("project_analysis", {})
    emissions_inventory = raw.get("emissions_inventory", {})

    project_analysis_root = _folder_from_entry(project_analysis.get("main"))
    black_carbon_root = _folder_from_entry(project_analysis.get("black_carbon"))
    black_carbon_pollutant = _value_from_entry(project_analysis.get("black_carbon"), "pollutant")
    road_dust_root = _folder_from_entry(project_analysis.get("paved_road_dust"))
    road_category_map = _normalize_string_mapping(
        _value_from_entry(project_analysis.get("paved_road_dust"), "road_category_map"),
        lower_keys=True,
    )
    emissions_inventory_main = _folder_from_entry(emissions_inventory.get("inventory_folder"))
    emissions_inventory_fallback = _folder_from_entry(emissions_inventory.get("fallback_folder"))
    vehicle_category_metadata_file = emissions_inventory.get("vehicle_category_metadata_file")

    normalized = {
        "project_analysis_raw": project_analysis_root,
        "black_carbon_raw": _find_matching_file(black_carbon_root, ("bc",), required=False),
        "black_carbon_pollutant": black_carbon_pollutant,
        "assignment_model": _normalize_model_spec_path(
            raw.get("assignment_model"),
            path_label="assignment_model",
        ),
        "vehicle_category_metadata_file": _expand_optional_path(
            None
            if vehicle_category_metadata_file in (None, "")
            else str(vehicle_category_metadata_file)
        ),
        "statewide_inventory_raw": _find_matching_file(emissions_inventory_fallback, ("statewide",), required=False),
        "population_raw": _find_matching_file(emissions_inventory_main, ("population",), required=False),
        "trips_raw": _find_matching_file(emissions_inventory_main, ("trips",), required=False),
        "vmt_raw": _find_matching_file(emissions_inventory_main, ("vmt",), required=False),
        "emission_raw": _find_matching_file(emissions_inventory_main, ("emission",), required=True),
        "ghg_raw": _find_matching_file(emissions_inventory_main, ("ghg",), required=False),
        "rainy_days_file": _find_matching_file(road_dust_root, ("rainy_days",), required=False),
        "silt_loading_file": _find_matching_file(road_dust_root, ("silt_loading",), required=False),
        "road_category_map": road_category_map,
    }
    path_keys = {
        "project_analysis_raw",
        "black_carbon_raw",
        "statewide_inventory_raw",
        "population_raw",
        "trips_raw",
        "vmt_raw",
        "emission_raw",
        "ghg_raw",
        "rainy_days_file",
        "silt_loading_file",
    }
    return {
        key: _expand_optional_path(value) if key in path_keys else value
        for key, value in normalized.items()
    }


def _normalize_string_mapping(mapping: object, *, lower_keys: bool = False, lower_values: bool = False) -> dict[str, str]:
    if mapping in (None, ""):
        return {}
    if not isinstance(mapping, dict):
        raise ValueError("Expected a mapping")
    normalized: dict[str, str] = {}
    for source, target in mapping.items():
        source_token = str(source).strip()
        target_token = str(target).strip()
        if not source_token or not target_token:
            continue
        if lower_keys:
            source_token = source_token.lower()
        if lower_values:
            target_token = target_token.lower()
        normalized[source_token] = target_token
    return normalized


def _normalize_alias_mapping(
    mapping: object,
    *,
    normalize_keys=None,
    normalize_values=None,
) -> dict[str, str]:
    if mapping in (None, ""):
        return {}
    if not isinstance(mapping, dict):
        raise ValueError("Expected a mapping")
    if normalize_keys is None:
        normalize_keys = lambda value: str(value).strip()
    if normalize_values is None:
        normalize_values = lambda value: str(value).strip()

    normalized: dict[str, str] = {}
    for canonical_value, aliases in mapping.items():
        canonical_token = normalize_keys(canonical_value)
        if not canonical_token:
            continue
        candidates = aliases if isinstance(aliases, (list, tuple, set)) else [aliases]
        for alias in candidates:
            alias_token = normalize_values(alias)
            if not alias_token:
                continue
            normalized[alias_token] = canonical_token
    return normalized


def _expand_activities_paths(raw: dict) -> dict:
    raw = deepcopy(raw)
    raw["pto_as_process"] = _normalize_pto_as_process(raw)
    raw["inputs"] = _normalize_activities_inputs(raw)
    raw.update(raw["inputs"])
    outputs = raw["outputs"].format(calendar_year=raw["calendar_year"])
    raw["outputs"] = _expand_optional_path(outputs)
    return raw


def _required(raw: dict, path: tuple[str, ...]) -> object:
    current = raw
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_activities(raw: dict, source_path: Path) -> None:
    required = [
        ("region_label",),
        ("calendar_year",),
        ("outputs",),
        ("model_year_groups",),
        ("project_analysis_raw",),
        ("assignment_model",),
        ("vehicle_category_metadata_file",),
        ("statewide_inventory_raw",),
        ("vmt_raw",),
        ("population_raw",),
        ("trips_raw",),
        ("emission_raw",),
        ("black_carbon_raw",),
        ("black_carbon_pollutant",),
        ("rainy_days_file",),
        ("silt_loading_file",),
    ]
    missing = [".".join(path) for path in required if _required(raw, path) in (None, "")]
    if missing:
        raise ValueError(f"EMFAC config at {source_path} is missing required keys: {', '.join(missing)}.")
    model_year_groups = raw.get("model_year_groups")
    if not isinstance(model_year_groups, dict):
        raise ValueError(
            f"EMFAC config at {source_path} must define model_year_groups as a mapping with light_duty and medium_heavy_duty."
        )
    required_group_keys = {"light_duty", "medium_heavy_duty"}
    missing_group_keys = sorted(required_group_keys - set(model_year_groups))
    if missing_group_keys:
        raise ValueError(
            f"EMFAC config at {source_path} is missing model_year_groups keys: {', '.join(missing_group_keys)}."
        )


def _build_activities_workflow(raw: dict[str, object], source_path: Path) -> dict[str, object]:
    raw = _expand_activities_paths(raw)
    _validate_activities(raw, source_path)
    emissions_inventory = raw.get("emissions_inventory", {})
    if not isinstance(emissions_inventory, dict):
        emissions_inventory = {}
    normalized_fuel_map = _normalize_alias_mapping(emissions_inventory.get("fuel_map", {}))

    year = int(raw["calendar_year"])
    region = str(raw["region_label"])
    scenario_id = str(raw["scenario"])
    emissions_store_name = scenario_id
    outputs_root = Path(str(raw["outputs"])).expanduser()
    activities_output_root = outputs_root / "activities" / emissions_store_name / "inventory"
    tmp_root = outputs_root / "activities" / "_tmp"
    trace_dir = tmp_root / "traces"
    region_slug = region.lower()
    base_name = f"{region_slug}-emfac-{year}"
    final_name = f"{base_name}-project-analysis-final"
    inventory_final_name = f"{base_name}-inventory-final"

    return {
        "run": {
            "region_label": region,
            "calendar_year": year,
            "scenario": scenario_id,
            "outputs": str(outputs_root),
            "model_year_groups": {
                "light_duty": list(raw["model_year_groups"]["light_duty"]),
                "medium_heavy_duty": list(raw["model_year_groups"]["medium_heavy_duty"]),
            },
            "pto_as_process": raw["pto_as_process"],
            "mappings": {
                **deepcopy(raw.get("mappings", {})),
                "fuel_map": normalized_fuel_map,
                "road_category_map": deepcopy(raw.get("road_category_map", {})),
            },
        },
        "inputs": {
            key: raw[key]
            for key in (
                "project_analysis_raw",
                "assignment_model",
                "vehicle_category_metadata_file",
                "black_carbon_raw",
                "black_carbon_pollutant",
                "statewide_inventory_raw",
                "population_raw",
                "trips_raw",
                "vmt_raw",
                "emission_raw",
                "ghg_raw",
                "rainy_days_file",
                "silt_loading_file",
            )
            if key in raw
        },
        "paths": {
            "outputs_root": str(outputs_root),
            "activities_output_root": str(activities_output_root),
            "tmp_root": str(tmp_root),
            "trace_dir": str(trace_dir),
            "project_analysis_source": str(tmp_root / f"{base_name}-project-analysis-source.parquet"),
            "project_analysis_passenger": str(tmp_root / f"{base_name}-project-analysis-passenger.parquet"),
            "project_analysis_freight": str(tmp_root / f"{base_name}-project-analysis-freight.parquet"),
            "project_analysis_bc": str(tmp_root / f"{base_name}-project-analysis-bc.parquet"),
            "project_analysis_prdust": str(tmp_root / f"{base_name}-project-analysis-prdust.parquet"),
            "project_analysis_nh3_rates": str(tmp_root / f"{base_name}-project-analysis-nh3-rates.parquet"),
            "emissions_inventory": str(tmp_root / f"{base_name}-inventory-intermediate-with-activity.parquet"),
            "statewide_inventory": str(tmp_root / f"statewide-emfac-{year}-emissions-inventory.parquet"),
            "matching_activity_output_passenger": str(tmp_root / f"{base_name}-inventory-matching-passenger-activity.parquet"),
            "matching_activity_output_freight": str(tmp_root / f"{base_name}-inventory-matching-freight-activity.parquet"),
            "final_output_passenger": str(activities_output_root / f"{final_name}-passenger-rates.parquet"),
            "final_activity_by_model_year_output_passenger": str(activities_output_root / f"{inventory_final_name}-passenger-activity-by-model-year.parquet"),
            "final_activity_by_emfacid_output_passenger": str(
                activities_output_root / f"{inventory_final_name}-passenger-activity-by-emfacid.parquet"
            ),
            "final_fleet_output_passenger": str(activities_output_root / f"{inventory_final_name}-passenger-fleet.parquet"),
            "final_output_freight": str(activities_output_root / f"{final_name}-freight-rates.parquet"),
            "final_activity_by_model_year_output_freight": str(activities_output_root / f"{inventory_final_name}-freight-activity-by-model-year.parquet"),
            "final_activity_by_emfacid_output_freight": str(
                activities_output_root / f"{inventory_final_name}-freight-activity-by-emfacid.parquet"
            ),
            "final_fleet_output_freight": str(activities_output_root / f"{inventory_final_name}-freight-fleet.parquet"),
            "emissions_store_root": str(outputs_root / "activities" / emissions_store_name / "rates"),
        },
    }



def _load_model_spec(model_spec_path: str) -> dict:
    spec_path = Path(model_spec_path)
    with spec_path.open() as handle:
        return yaml.safe_load(handle) or {}
def _extract_fleet_assignment_root(model_spec: dict, *, model_spec_path: Path) -> dict[str, object]:
    root = model_spec.get("fleet_assignment")
    if not isinstance(root, dict):
        raise ValueError(
            f"Configured fleet path has an invalid model spec file at {model_spec_path}. "
            "It must contain top-level fleet_assignment."
        )
    return root


def _extract_named_model(model_spec: dict, *, model_name: str, model_spec_path: Path) -> dict[str, object]:
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=model_spec_path)
    models = fleet_assignment.get("models")
    if not isinstance(models, dict):
        raise ValueError(
            f"Configured fleet path has an invalid model spec file at {model_spec_path}. "
            "It must contain fleet_assignment.models."
        )
    model_section = models.get(model_name)
    if not isinstance(model_section, dict):
        raise ValueError(
            f"Configured fleet path has an invalid model spec file at {model_spec_path}. "
            f"It must contain fleet_assignment.models.{model_name}."
        )
    return model_section


def _normalized_string_list(values: object, *, lower: bool = False) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for value in values:
        token = str(value).strip()
        if not token:
            continue
        normalized.append(token.lower() if lower else token)
    return normalized


def build_model_category_fuel_mapping(model_spec_path: str | Path) -> pd.DataFrame:
    spec_path = Path(model_spec_path)
    model_spec = _load_model_spec(str(spec_path))
    rows: list[dict[str, str]] = []
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=spec_path)
    mappings = fleet_assignment.get("mappings", {})
    if not isinstance(mappings, dict):
        raise ValueError(f"Vehicle type assignment model file at {spec_path} must define fleet_assignment.mappings.")

    passenger_allowed_fuels_by_emfac_category = {
        "MCY": {"Gas"},
        "UBUS": {"Dsl", "Elec", "Gas"},
    }

    passenger_mapping = mappings.get("passenger", {})
    if not isinstance(passenger_mapping, dict):
        raise ValueError(f"Vehicle type assignment model file at {spec_path} must define mappings.passenger.")
    passenger_vehicle_categories = passenger_mapping.get("vehicle_categories", {})
    passenger_fuel_types = passenger_mapping.get("fuel_types", {})

    for beam_category, emfac_categories in passenger_vehicle_categories.items():
        beam_category_token = str(beam_category).strip()
        if not beam_category_token:
            continue
        for emfac_category in _normalized_string_list(emfac_categories):
            allowed_emfac_fuels = passenger_allowed_fuels_by_emfac_category.get(emfac_category)
            for adopt_fuel, emfac_fuels in passenger_fuel_types.items():
                adopt_fuel_token = str(adopt_fuel).strip().lower()
                if not adopt_fuel_token:
                    continue
                for emfac_fuel in _normalized_string_list(emfac_fuels):
                    if allowed_emfac_fuels is not None and emfac_fuel not in allowed_emfac_fuels:
                        continue
                    rows.append(
                        {
                            "group": "passenger",
                            "emfac_vehicle_category": emfac_category,
                            "emfac_fuel": emfac_fuel,
                            "beam_category": beam_category_token,
                            "adopt_fuel": adopt_fuel_token,
                        }
                    )

    freight_mapping = mappings.get("freight", {})
    if not isinstance(freight_mapping, dict):
        raise ValueError(f"Vehicle type assignment model file at {spec_path} must define mappings.freight.")
    freight_vehicle_categories = freight_mapping.get("vehicle_categories", {})
    freight_fuel_types = freight_mapping.get("fuel_types", {})
    for beam_category, emfac_categories in freight_vehicle_categories.items():
        beam_category_token = str(beam_category).strip()
        if not beam_category_token:
            continue
        for emfac_category in _normalized_string_list(emfac_categories):
            for adopt_fuel, emfac_fuels in freight_fuel_types.items():
                adopt_fuel_token = str(adopt_fuel).strip().lower()
                if not adopt_fuel_token:
                    continue
                for emfac_fuel in _normalized_string_list(emfac_fuels):
                    rows.append(
                        {
                            "group": "freight",
                            "emfac_vehicle_category": emfac_category,
                            "emfac_fuel": emfac_fuel,
                            "beam_category": beam_category_token,
                            "adopt_fuel": adopt_fuel_token,
                        }
                    )

    frame = pd.DataFrame(rows, columns=["group", "emfac_vehicle_category", "emfac_fuel", "beam_category", "adopt_fuel"])
    if frame.empty:
        raise ValueError(f"Vehicle type assignment model file at {spec_path} produced no EMFAC category/fuel mapping rows.")
    return frame.drop_duplicates().reset_index(drop=True)


def build_fuel_consumption_emfac_assignment_catalog(
    model_spec_path: str | Path,
    breakdown_path: str | Path,
) -> pd.DataFrame:
    spec_path = Path(model_spec_path)
    model_spec = _load_model_spec(str(spec_path))
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=spec_path)
    mappings = fleet_assignment.get("mappings", {})
    assignments = mappings.get("fuel_consumption", []) if isinstance(mappings, dict) else []
    if not isinstance(assignments, list) or not assignments:
        raise ValueError(
            f"Vehicle type assignment model file at {spec_path} has no mappings.fuel_consumption rows."
        )

    breakdown = read_table(
        str(breakdown_path),
        schema=FUEL_CONSUMPTION_CATALOG_SCHEMA,
    ).copy()
    breakdown["fastsim_id"] = breakdown["fastsim_id"].map(lambda value: str(value or "").strip())
    breakdown["fuel"] = breakdown["fuel"].fillna("").str.strip().str.lower()
    breakdown["charge_behavior"] = breakdown["charge_behavior"].fillna("").str.strip().str.lower()
    breakdown["model_trim"] = breakdown["model_trim"].fillna("").str.strip()
    breakdown["fastsim_relative_path"] = breakdown["fastsim_relative_path"].map(lambda value: str(value or "").strip())

    rows: list[dict[str, object]] = []
    for item in assignments:
        if not isinstance(item, dict):
            continue
        fastsim_id = str(item.get("fastsim_id", "") or "").strip()
        emfac_vehicle_categories = _normalized_string_list(item.get("vehicle_categories"))
        emfac_fuels = _normalized_string_list(item.get("fuel_types"))
        if not fastsim_id or not emfac_vehicle_categories or not emfac_fuels:
            continue
        matched = breakdown[breakdown["fastsim_id"] == fastsim_id].copy()
        if matched.empty:
            raise ValueError(
                f"Fuel-consumption assignment row in {spec_path} could not be resolved in {breakdown_path}: "
                f"fastsim_id={fastsim_id}"
            )
        for _, matched_row in matched.iterrows():
            relative_path = str(matched_row.get("fastsim_relative_path", "")).strip()
            matched_charge_behavior = str(matched_row.get("charge_behavior", "") or "").strip().lower()
            matched_fuel = str(matched_row.get("fuel", "")).strip().lower()
            if not relative_path:
                continue
            for emfac_vehicle_category in emfac_vehicle_categories:
                for emfac_fuel in emfac_fuels:
                    rows.append(
                        {
                            "fastsim_id": fastsim_id,
                            "fastsim_relative_path": relative_path,
                            "emfac_vehicle_category": str(emfac_vehicle_category).strip(),
                            "emfac_fuel": str(emfac_fuel).strip(),
                            "model_year": matched_row.get("model_year"),
                            "fuel": matched_fuel,
                            "charge_behavior": matched_charge_behavior,
                            "model_trim": matched_row.get("model_trim"),
                            "msrp_usd": matched_row.get("msrp_usd"),
                        }
                    )
    frame = pd.DataFrame(
        rows,
        columns=[
            "fastsim_id",
            "fastsim_relative_path",
            "emfac_vehicle_category",
            "emfac_fuel",
            "model_year",
            "fuel",
            "charge_behavior",
            "model_trim",
            "msrp_usd",
        ],
    )
    if frame.empty:
        raise ValueError(
            f"Vehicle type assignment model file at {spec_path} produced no fuel-consumption assignment rows."
        )
    return frame.drop_duplicates().reset_index(drop=True)


def _normalize_model_spec_path(path_like: str | None, *, path_label: str) -> str | None:
    resolved = _normalize_configured_path(
        path_like,
        path_label=path_label,
        expect_directory=False,
        must_exist=True,
    )
    if resolved is None:
        return None
    model_spec_path = Path(resolved)
    model_spec = _load_model_spec(str(model_spec_path))
    model_section = _extract_named_model(
        model_spec,
        model_name="freight_bayesian_dag",
        model_spec_path=model_spec_path,
    )
    freight_scoring = model_section.get("scoring")
    if not isinstance(freight_scoring, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.freight_bayesian_dag.scoring."
        )
    freight_weights = freight_scoring.get("weights")
    if not isinstance(freight_weights, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.freight_bayesian_dag.scoring.weights."
        )
    missing_freight_weights = [
        key
        for key in ("fleet_vmt_prior", "fleet_population_prior", "naics_sector", "payload_mass", "port_location")
        if key not in freight_weights
    ]
    if missing_freight_weights:
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.freight_bayesian_dag.scoring.weights is missing: "
            + ", ".join(sorted(missing_freight_weights))
        )
    freight_evidence = model_section.get("evidence", {})
    if freight_evidence not in (None, "") and not isinstance(freight_evidence, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.freight_bayesian_dag.evidence must be a mapping when provided."
        )
    payload_mass_evidence = freight_evidence.get("payload_mass") if isinstance(freight_evidence, dict) else None
    if payload_mass_evidence not in (None, ""):
        if not isinstance(payload_mass_evidence, dict):
            raise ValueError(
                f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
                "models.freight_bayesian_dag.evidence.payload_mass must be a mapping when provided."
            )
        missing_payload_mass_evidence = [
            key
            for key in ("source", "unit", "overload_penalty_power")
            if key not in payload_mass_evidence
        ]
        if missing_payload_mass_evidence:
            raise ValueError(
                f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
                "models.freight_bayesian_dag.evidence.payload_mass is missing: "
                + ", ".join(sorted(missing_payload_mass_evidence))
            )
    fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=model_spec_path)
    mappings = fleet_assignment.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain fleet_assignment.mappings."
        )
    freight_mapping = mappings.get("freight")
    if not isinstance(freight_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain mappings.freight."
        )
    freight_evidence = model_section.get("evidence", {})
    if not isinstance(freight_evidence, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.freight_bayesian_dag.evidence must be a mapping."
        )
    naics_evidence = freight_evidence.get("naics_sector")
    if not isinstance(naics_evidence, list) or not naics_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no models.freight_bayesian_dag.evidence.naics_sector entries in {model_spec_path}. "
            "It should contain the NAICS-sector-to-vehicle-category evidence mappings."
        )
    freight_vehicle_categories = freight_mapping.get("vehicle_categories")
    if not isinstance(freight_vehicle_categories, dict) or not freight_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.freight.vehicle_categories entries in {model_spec_path}. "
            "It should contain the FRISM-to-EMFAC vehicle-category evidence mappings."
        )
    invalid_freight_vehicle_categories = [
        key
        for key, value in freight_vehicle_categories.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_freight_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.freight.vehicle_categories entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_freight_vehicle_categories))
        )
    freight_fuel_types = freight_mapping.get("fuel_types")
    if not isinstance(freight_fuel_types, dict) or not freight_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.freight.fuel_types entries in {model_spec_path}. "
            "It should contain the FRISM-to-EMFAC fuel evidence mappings."
        )
    invalid_freight_fuel_types = [
        key
        for key, value in freight_fuel_types.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_freight_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.freight.fuel_types entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_freight_fuel_types))
        )
    port_evidence = freight_evidence.get("port_location")
    if not isinstance(port_evidence, list) or not port_evidence:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no models.freight_bayesian_dag.evidence.port_location entries in {model_spec_path}. "
            "It should contain the zone-to-vehicle-category evidence mappings for port assignments."
        )
    passenger_model = _extract_named_model(
        model_spec,
        model_name="passenger_bayesian_dag",
        model_spec_path=model_spec_path,
    )
    passenger_scoring = passenger_model.get("scoring")
    if not isinstance(passenger_scoring, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.passenger_bayesian_dag.scoring."
        )
    passenger_weights = passenger_scoring.get("weights")
    if not isinstance(passenger_weights, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain models.passenger_bayesian_dag.scoring.weights."
        )
    missing_passenger_weights = [
        key for key in ("fleet_vmt_prior", "fleet_population_prior", "income") if key not in passenger_weights
    ]
    if missing_passenger_weights:
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.passenger_bayesian_dag.scoring.weights is missing: "
            + ", ".join(sorted(missing_passenger_weights))
        )
    passenger_evidence = passenger_model.get("evidence")
    if passenger_evidence is not None and not isinstance(passenger_evidence, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "models.passenger_bayesian_dag.evidence must be a mapping when provided."
        )
    income_evidence = passenger_evidence.get("income") if isinstance(passenger_evidence, dict) else None
    if income_evidence is not None:
        if not isinstance(income_evidence, dict):
            raise ValueError(
                f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
                "models.passenger_bayesian_dag.evidence.income must be a mapping when provided."
            )
        missing_income_evidence = [
            key for key in ("center_ratio", "sigma_ratio") if key not in income_evidence
        ]
        if missing_income_evidence:
            raise ValueError(
                f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
                "models.passenger_bayesian_dag.evidence.income is missing: "
                + ", ".join(sorted(missing_income_evidence))
            )
    mappings = fleet_assignment.get("mappings")
    if not isinstance(mappings, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain fleet_assignment.mappings."
        )
    freight_mapping = mappings.get("freight")
    if not isinstance(freight_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain mappings.freight."
        )
    passenger_mapping = mappings.get("passenger")
    if not isinstance(passenger_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has an invalid model spec file at {model_spec_path}. "
            "It must contain mappings.passenger."
        )
    passenger_vehicle_categories = passenger_mapping.get("body_types")
    if not isinstance(passenger_vehicle_categories, dict) or not passenger_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.passenger.body_types entries in {model_spec_path}. "
            "It should contain the ATLAS-bodytype-to-EMFAC-category support mappings."
        )
    invalid_vehicle_categories = [
        key
        for key, value in passenger_vehicle_categories.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.body_types entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_vehicle_categories))
        )
    passenger_fuel_types = passenger_mapping.get("fuel_types")
    if not isinstance(passenger_fuel_types, dict) or not passenger_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.passenger.fuel_types entries in {model_spec_path}. "
            "It should contain the passenger BEAM-fuel-to-EMFAC fuel evidence mappings."
        )
    invalid_passenger_fuel_types = [
        key
        for key, value in passenger_fuel_types.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_passenger_fuel_types:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.fuel_types entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_passenger_fuel_types))
        )
    passenger_fuel_fallbacks = passenger_mapping.get("fuel_fallbacks", [])
    if passenger_fuel_fallbacks not in (None, ""):
        if not isinstance(passenger_fuel_fallbacks, list):
            raise ValueError(
                f"Configured fleet path '{path_label}' has invalid mappings.passenger.fuel_fallbacks in {model_spec_path}. "
                "It must be a list when provided."
            )
        invalid_fallbacks = []
        for index, item in enumerate(passenger_fuel_fallbacks):
            if not isinstance(item, dict):
                invalid_fallbacks.append(str(index))
                continue
            source_fuel = str(item.get("source_fuel", "")).strip()
            model_year = str(item.get("if_model_year", "")).strip()
            fallback_fuels = item.get("fallback_emfac_fuels")
            if (
                source_fuel == ""
                or model_year == ""
                or not isinstance(fallback_fuels, list)
                or not [str(value).strip() for value in fallback_fuels if str(value).strip()]
            ):
                invalid_fallbacks.append(str(index))
        if invalid_fallbacks:
            raise ValueError(
                f"Configured fleet path '{path_label}' has invalid mappings.passenger.fuel_fallbacks rows in {model_spec_path}: "
                + ", ".join(invalid_fallbacks)
            )
    passenger_vehicle_categories = passenger_mapping.get("vehicle_categories")
    if not isinstance(passenger_vehicle_categories, dict) or not passenger_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.passenger.vehicle_categories entries in {model_spec_path}. "
            "It should contain the BEAM-category-to-EMFAC-category support mappings for passenger bus/bike mapping."
        )
    invalid_passenger_vehicle_categories = [
        key
        for key, value in passenger_vehicle_categories.items()
        if not isinstance(value, list) or not value
    ]
    if invalid_passenger_vehicle_categories:
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.vehicle_categories entries in {model_spec_path}: "
            + ", ".join(sorted(str(key) for key in invalid_passenger_vehicle_categories))
        )
    passenger_model_year_mapping = passenger_mapping.get("model_year", {})
    if passenger_model_year_mapping not in (None, "") and not isinstance(passenger_model_year_mapping, dict):
        raise ValueError(
            f"Configured fleet path '{path_label}' has invalid mappings.passenger.model_year in {model_spec_path}. "
            "It must be a mapping when provided."
        )
    assignment_rows = mappings.get("fuel_consumption")
    if not isinstance(assignment_rows, list) or not assignment_rows:
        raise ValueError(
            f"Configured fleet path '{path_label}' has no mappings.fuel_consumption rows in {model_spec_path}. "
            "It should contain the fuel-consumption assignment rows."
        )
    return str(model_spec_path)


def _derive_emfac_output_paths(emfac: dict[str, object]) -> dict[str, object]:
    outputs = emfac.get("outputs")
    region_label = emfac.get("region_label")
    calendar_year = emfac.get("calendar_year")
    scenario = emfac.get("scenario")
    if outputs in (None, "") or region_label in (None, "") or calendar_year in (None, "") or scenario in (None, ""):
        return emfac
    region_slug = str(region_label).strip().lower()
    project_analysis_name = f"{region_slug}-emfac-{int(calendar_year)}-project-analysis-final"
    inventory_final_name = f"{region_slug}-emfac-{int(calendar_year)}-inventory-final"
    inventory_matching_name = f"{region_slug}-emfac-{int(calendar_year)}-inventory-matching"
    outputs_root = Path(str(outputs))
    activities_output_root = outputs_root / "activities" / str(scenario) / "inventory"
    tmp_root = outputs_root / "activities" / "_tmp"
    emfac["passenger_rates_file"] = str((activities_output_root / f"{project_analysis_name}-passenger-rates.parquet").resolve())
    emfac["passenger_activity_file"] = str((tmp_root / f"{inventory_matching_name}-passenger-activity.parquet").resolve())
    emfac["passenger_fleet_file"] = str((activities_output_root / f"{inventory_final_name}-passenger-fleet.parquet").resolve())
    emfac["freight_rates_file"] = str((activities_output_root / f"{project_analysis_name}-freight-rates.parquet").resolve())
    emfac["freight_activity_file"] = str((tmp_root / f"{inventory_matching_name}-freight-activity.parquet").resolve())
    emfac["freight_fleet_file"] = str((activities_output_root / f"{inventory_final_name}-freight-fleet.parquet").resolve())
    emfac["emissions_store_root"] = str((outputs_root / "activities" / str(scenario) / "rates").resolve())
    return emfac


def _ingest_fleet_sources(config: dict) -> dict:
    config = deepcopy(config)
    model_spec: dict | None = None
    model_file = config.get("assignment_model")
    if model_file not in (None, ""):
        model_spec = _load_model_spec(str(model_file))
    config["output"] = _normalize_configured_path(config.get("output"), path_label="output", must_exist=False)
    activities = config.get("activities", {})
    if isinstance(activities, dict):
        activities = _expand_activities_paths(activities)
        activities["outputs"] = _normalize_configured_path(
            activities.get("outputs"),
            path_label="activities.outputs",
            must_exist=False,
        )
        mappings = activities.get("mappings", {})
        if mappings in (None, ""):
            mappings = {}
        if not isinstance(mappings, dict):
            raise ValueError("activities.mappings must be a mapping")
        activities["mappings"] = mappings
        emissions_inventory = activities.get("emissions_inventory", {})
        if isinstance(emissions_inventory, dict):
            raw_fuel_map = emissions_inventory.get("fuel_map", {})
            if raw_fuel_map in (None, ""):
                raw_fuel_map = {}
            if not isinstance(raw_fuel_map, dict):
                raise ValueError(
                    "activities.emissions_inventory.fuel_map must be a mapping of "
                    "normalized fuel tokens to one or more raw EMFAC fuel labels"
                )
            emissions_inventory["fuel_map"] = _normalize_alias_mapping(raw_fuel_map)
            activities["emissions_inventory"] = emissions_inventory
        activities = _derive_emfac_output_paths(activities)
        config["activities"] = activities
        config["vehicle_category_metadata_file"] = activities.get("vehicle_category_metadata_file")
    frism = config.get("frism", {})
    if isinstance(frism, dict):
        carriers_folder = _normalize_configured_path(
            frism.get("carriers_folder"),
            path_label="frism.carriers_folder",
            expect_directory=True,
        )
        if carriers_folder:
            frism["carriers_file"] = _find_matching_file(carriers_folder, ("carrier",))
            frism["payloads_file"] = _find_matching_file(carriers_folder, ("payload",))
            frism["tours_file"] = _find_matching_file(carriers_folder, ("tour",))
        frism["carriers_folder"] = carriers_folder
        config["frism"] = frism
    config["passenger_vehicle_types_file"] = _normalize_configured_path(
        config.get("passenger_vehicle_types_file"),
        path_label="passenger_vehicle_types_file",
    )
    config["freight_vehicle_types_file"] = _normalize_configured_path(
        config.get("freight_vehicle_types_file"),
        path_label="freight_vehicle_types_file",
    )
    config["fuel_consumption_catalog"] = _normalize_configured_path(
        config.get("fuel_consumption_catalog"),
        path_label="fuel_consumption_catalog",
    )
    model_file = _normalize_model_spec_path(config.get("assignment_model"), path_label="assignment_model")
    if model_file is not None:
        config["assignment_model"] = model_file
        model_spec = _load_model_spec(model_file)
        freight_model = _extract_named_model(
            model_spec,
            model_name="freight_bayesian_dag",
            model_spec_path=Path(model_file),
        )
        scoring = freight_model.get("scoring", {})
        evidence = freight_model.get("evidence", {}) if isinstance(freight_model.get("evidence", {}), dict) else {}
        if "likelihood_floor" not in scoring:
            raise ValueError(
                "assignment_model must define models.freight_bayesian_dag.scoring.likelihood_floor"
            )
        floor_value = float(scoring["likelihood_floor"])
        if not (0.0 < floor_value < 1.0):
            raise ValueError(
                f"assignment_model likelihood_floor must be between 0 and 1 exclusive, got {floor_value}"
            )
        weights = scoring.get("weights", {})
        missing_weights = [
            key
            for key in ("fleet_vmt_prior", "fleet_population_prior", "naics_sector", "payload_mass", "port_location")
            if key not in weights
        ]
        if missing_weights:
            raise ValueError(
                "assignment_model must define models.freight_bayesian_dag.scoring.weights for: "
                + ", ".join(missing_weights)
            )
        for key in ("fleet_vmt_prior", "fleet_population_prior", "naics_sector", "payload_mass", "port_location"):
            value = float(weights[key])
            if value < 0.0:
                raise ValueError(f"assignment_model {key} weight must be non-negative, got {value}")
            config[key] = value
        fleet_assignment = _extract_fleet_assignment_root(model_spec, model_spec_path=Path(model_file))
        mappings = fleet_assignment.get("mappings", {})
        freight_mapping = mappings.get("freight", {}) if isinstance(mappings, dict) else {}
        freight_vehicle_categories = freight_mapping.get("vehicle_categories", {})
        freight_fuel_types = freight_mapping.get("fuel_types", {})
        config["freight_bayesian_dag"] = {
            "likelihood_floor": floor_value,
            "fleet_vmt_prior": float(weights["fleet_vmt_prior"]),
            "fleet_population_prior": float(weights["fleet_population_prior"]),
            "naics_sector": float(weights["naics_sector"]),
            "payload_mass": float(weights["payload_mass"]),
            "port_location": float(weights["port_location"]),
            "payload_mass_enabled": isinstance(evidence.get("payload_mass"), dict),
            "payload_mass_source": str(evidence.get("payload_mass", {}).get("source", "")).strip(),
            "payload_mass_unit": str(evidence.get("payload_mass", {}).get("unit", "")).strip(),
            "payload_mass_overload_penalty_power": float(
                evidence.get("payload_mass", {}).get("overload_penalty_power", 2.0)
            ),
        }
        config["freight_mapping"] = {
            "vehicle_categories": {
                str(category).strip(): [
                    str(emfac_category).strip()
                    for emfac_category in emfac_categories
                    if str(emfac_category).strip()
                ]
                for category, emfac_categories in freight_vehicle_categories.items()
                if str(category).strip()
            },
            "fuel_types": {
                str(fuel).strip(): [
                    str(emfac_fuel).strip()
                    for emfac_fuel in emfac_fuels
                    if str(emfac_fuel).strip()
                ]
                for fuel, emfac_fuels in freight_fuel_types.items()
                if str(fuel).strip()
            },
        }
        passenger_model = _extract_named_model(
            model_spec,
            model_name="passenger_bayesian_dag",
            model_spec_path=Path(model_file),
        )
        passenger_scoring = passenger_model.get("scoring", {})
        passenger_weights = passenger_scoring.get("weights", {})
        passenger_evidence = passenger_model.get("evidence", {})
        passenger_income_evidence = passenger_evidence.get("income", {}) if isinstance(passenger_evidence, dict) else {}
        passenger_mapping = mappings.get("passenger", {}) if isinstance(mappings, dict) else {}
        passenger_vehicle_categories = passenger_mapping.get("body_types", {})
        passenger_fuel_types = passenger_mapping.get("fuel_types", {})
        config["passenger_bayesian_dag"] = {
            "likelihood_floor": float(passenger_scoring.get("likelihood_floor", 1e-3)),
            "fleet_vmt_prior_weight": float(passenger_weights.get("fleet_vmt_prior", 1.0)),
            "fleet_population_prior_weight": float(passenger_weights.get("fleet_population_prior", 1.0)),
            "income_weight": float(passenger_weights.get("income", 1.0)),
            "income_enabled": isinstance(passenger_income_evidence, dict) and bool(passenger_income_evidence),
            "income_center_ratio": float(passenger_income_evidence.get("center_ratio", 0.30)),
            "income_sigma_ratio": float(passenger_income_evidence.get("sigma_ratio", 0.10)),
        }
        config["passenger_mapping"] = {
            "body_types": {
                str(bodytype).strip().lower(): [
                    str(category).strip()
                    for category in categories
                    if str(category).strip()
                ]
                for bodytype, categories in passenger_vehicle_categories.items()
                if str(bodytype).strip()
            },
            "fuel_types": {
                str(fuel).strip(): [
                    str(emfac_fuel).strip()
                    for emfac_fuel in emfac_fuels
                    if str(emfac_fuel).strip()
                ]
                for fuel, emfac_fuels in passenger_fuel_types.items()
                if str(fuel).strip()
            },
            "fuel_fallbacks": [
                {
                    "source_fuel": str(item.get("source_fuel", "")).strip().lower(),
                    "if_model_year": str(item.get("if_model_year", "")).strip(),
                    "fallback_emfac_fuels": [
                        str(emfac_fuel).strip()
                        for emfac_fuel in item.get("fallback_emfac_fuels", [])
                        if str(emfac_fuel).strip()
                    ],
                    "reason": str(item.get("reason", "")).strip(),
                }
                for item in passenger_mapping.get("fuel_fallbacks", [])
                if isinstance(item, dict) and str(item.get("source_fuel", "")).strip()
            ],
            "vehicle_categories": {
                str(beam_category).strip(): [
                    str(category).strip()
                    for category in categories
                    if str(category).strip()
                ]
                for beam_category, categories in passenger_mapping.get("vehicle_categories", {}).items()
                if str(beam_category).strip()
            },
        }
        config["fuel_consumption_mapping"] = [
            {
                "fastsim_id": str(item.get("fastsim_id", "")).strip(),
                "vehicle_categories": [
                    str(category).strip()
                    for category in item.get("vehicle_categories", [])
                    if str(category).strip()
                ],
                "fuel_types": [
                    str(fuel).strip()
                    for fuel in item.get("fuel_types", [])
                    if str(fuel).strip()
                ],
            }
            for item in mappings.get("fuel_consumption", [])
            if isinstance(item, dict)
        ]
    atlas = config.get("atlas", {})
    if isinstance(atlas, dict):
        population_folder = _normalize_configured_path(
            atlas.get("population_folder"),
            path_label="atlas.population_folder",
            expect_directory=True,
        )
        if population_folder:
            pop_path = Path(population_folder)
            atlas["vehicles_file"] = str(pop_path / "vehicles.parquet")
            atlas["households_file"] = str(pop_path / "households.parquet")
            atlas["persons_file"] = str(pop_path / "persons.parquet")
        atlas["population_folder"] = population_folder
        if atlas.get("income_bins") is not None:
            atlas["income_bins"] = list(atlas["income_bins"])
        fuel_map = atlas.get("fuel_map", {})
        if fuel_map in (None, ""):
            fuel_map = {}
        if not isinstance(fuel_map, dict):
            raise ValueError(
                "atlas.fuel_map must be a mapping of normalized BEAM fuel tokens "
                "to one or more source ATLAS fuel tokens"
            )
        atlas["fuel_map"] = _normalize_alias_mapping(
            fuel_map,
            normalize_keys=lambda value: str(value).strip().lower(),
            normalize_values=lambda value: str(value).strip().lower(),
        )
        config["atlas"] = atlas
    return config


def _required_value(raw: dict, path: tuple[str, ...]):
    current = raw
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _validate_fleet(raw: dict, source_path: Path) -> None:
    required_paths = [
        ("region",),
        ("scenario",),
        ("seed",),
        ("output",),
        ("assignment_model",),
        ("activities",),
        ("activities", "outputs"),
        ("activities", "region_label"),
        ("activities", "calendar_year"),
        ("activities", "model_year_groups"),
        ("activities", "project_analysis"),
        ("activities", "emissions_inventory"),
        ("atlas",),
        ("atlas", "population_folder"),
        ("frism",),
        ("frism", "carriers_folder"),
        ("passenger_vehicle_types_file",),
        ("fuel_consumption_catalog",),
    ]
    missing = []
    for path in required_paths:
        value = _required_value(raw, path)
        if value is None or value == "":
            missing.append(".".join(path))
    if missing:
        raise ValueError(f"Fleet config at {source_path} is missing required keys: {', '.join(missing)}.")


def load_fleet_workflow(config_path: str | Path | None = None) -> dict:
    source_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    raw = _build_fleet_config_from_root(_build_emfac_root_from_settings_file(source_path))
    _validate_fleet(raw, source_path)
    raw = _ingest_fleet_sources(raw)
    return _build_loaded_fleet_workflow(raw)


def _build_loaded_fleet_workflow(raw: dict) -> dict:
    output_root = raw["output"]
    config = {
        "seed": raw["seed"],
        "output": output_root,
        "vehicle_category_metadata_file": raw.get("vehicle_category_metadata_file"),
        "activities": raw["activities"],
        "atlas": raw["atlas"],
        "frism": raw["frism"],
        "passenger_vehicle_types_file": raw.get("passenger_vehicle_types_file"),
        "freight_vehicle_types_file": raw.get("freight_vehicle_types_file"),
        "fuel_consumption_catalog": raw.get("fuel_consumption_catalog"),
        "rates": raw.get("rates", {}),
        "assignment_model": raw.get("assignment_model"),
        "freight_bayesian_dag": raw.get("freight_bayesian_dag", {}),
        "passenger_bayesian_dag": raw.get("passenger_bayesian_dag", {}),
        "freight_mapping": raw.get("freight_mapping", {}),
        "passenger_mapping": raw.get("passenger_mapping", {}),
        "fuel_consumption_mapping": raw.get("fuel_consumption_mapping", []),
    }
    return {
        "area": raw["region"],
        "scenario": raw["scenario"],
        "config": config,
        "paths": {
            "trace_dir": str(Path(str(output_root)).expanduser() / "_tmp" / "traces"),
        },
    }


def load_fleet_workflow_from_activities_manifest(activities_manifest_path: str | Path) -> dict:
    from ..manifest.file_ops import load_structured_file
    from ..manifest.schema import ActivitiesManifest

    manifest = ActivitiesManifest.from_dict(load_structured_file(activities_manifest_path)).to_dict()
    settings_source = Path(manifest["settings_source"]).resolve()
    raw = _build_fleet_config_from_root(_build_emfac_root_from_settings_file(settings_source))
    _validate_fleet(raw, settings_source)
    raw = _ingest_fleet_sources(raw)
    raw["activities"] = {
        "outputs": manifest["outputs"]["outputs_root"],
        "passenger_rates_file": manifest["outputs"]["passenger_rates_file"],
        "passenger_activity_file": manifest["outputs"]["passenger_activity_file"],
        "passenger_fleet_file": manifest["outputs"]["passenger_fleet_file"],
        "freight_rates_file": manifest["outputs"]["freight_rates_file"],
        "freight_activity_file": manifest["outputs"]["freight_activity_file"],
        "freight_fleet_file": manifest["outputs"]["freight_fleet_file"],
        "emissions_store_root": manifest["outputs"]["emissions_store_root"],
    }
    raw["vehicle_category_metadata_file"] = manifest["vehicle_category_metadata_file"]
    return _build_loaded_fleet_workflow(raw)


def load_default_fleet_workflow() -> dict:
    return load_fleet_workflow(DEFAULT_CONFIG_PATH)
