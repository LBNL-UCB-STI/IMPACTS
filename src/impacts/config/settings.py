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
            {
                "enabled",
                "grid_size_meters",
                "asrv_patterns_file",
                "asrv_patterns_epsg",
                "asrv_nox_to_no2_ratios_file",
            },
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
class EmissionsInventory:
    passenger_file: str
    freight_file: str
    enable_passenger_activity_correction: bool = True
    enable_freight_activity_correction: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EmissionsInventory":
        _reject_unknown_keys(
            payload,
            {
                "passenger_file",
                "freight_file",
                "enable_passenger_activity_correction",
                "enable_freight_activity_correction",
            },
            "impacts.emissions.inventory",
        )
        return cls(
            passenger_file=_required_string(
                payload.get("passenger_file"),
                "impacts.emissions.inventory.passenger_file",
            ),
            freight_file=_required_string(
                payload.get("freight_file"),
                "impacts.emissions.inventory.freight_file",
            ),
            enable_passenger_activity_correction=(
                _required_bool(
                    payload.get("enable_passenger_activity_correction"),
                    "impacts.emissions.inventory.enable_passenger_activity_correction",
                )
                if payload.get("enable_passenger_activity_correction") is not None
                else True
            ),
            enable_freight_activity_correction=(
                _required_bool(
                    payload.get("enable_freight_activity_correction"),
                    "impacts.emissions.inventory.enable_freight_activity_correction",
                )
                if payload.get("enable_freight_activity_correction") is not None
                else True
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
                    raise ValueError("Expected mapping for impacts.emissions.defaults.annualization_days")
                _reject_unknown_keys(
                    payload,
                    {"light_duty", "medium_heavy_duty"},
                    "impacts.emissions.defaults.annualization_days",
                )
                return cls(
                    light_duty=_required_float(
                        payload.get("light_duty"),
                        "impacts.emissions.defaults.annualization_days.light_duty",
                    ),
                    medium_heavy_duty=_required_float(
                        payload.get("medium_heavy_duty"),
                        "impacts.emissions.defaults.annualization_days.medium_heavy_duty",
                    ),
                )

        annualization_days: "Emissions.Defaults.AnnualizationDays"

        @classmethod
        def from_dict(cls, payload: Dict[str, Any]) -> "Emissions.Defaults":
            _reject_unknown_keys(payload, {"annualization_days"}, "impacts.emissions.defaults")
            return cls(
                annualization_days=Emissions.Defaults.AnnualizationDays.from_dict(
                    dict(payload.get("annualization_days", {}) or {})
                )
            )

    osm_network_folder: str
    emissions_rates_folder: str
    inventory: EmissionsInventory
    vehicle_category_metadata_file: Optional[str]
    defaults: "Emissions.Defaults"
    source_pollutants: List[str]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Emissions":
        _reject_unknown_keys(
            payload,
            {
                "osm_network_folder",
                "emissions_rates_folder",
                "inventory",
                "vehicle_category_metadata_file",
                "defaults",
                "pollutants",
            },
            "impacts.emissions",
        )
        source_pollutants = _build_source_pollutants(payload.get("pollutants"))
        build_pollutants_map_from_sources(source_pollutants)
        return cls(
            osm_network_folder=_required_string(
                payload.get("osm_network_folder"),
                "impacts.emissions.osm_network_folder",
            ),
            emissions_rates_folder=_required_string(
                payload.get("emissions_rates_folder"),
                "impacts.emissions.emissions_rates_folder",
            ),
            inventory=EmissionsInventory.from_dict(
                dict(payload.get("inventory", {}) or {})
            ),
            vehicle_category_metadata_file=_optional_string(
                payload.get("vehicle_category_metadata_file")
            ),
            defaults=Emissions.Defaults.from_dict(
                dict(payload.get("defaults", {}) or {})
            ),
            source_pollutants=source_pollutants,
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
            "link_id": "edge_linkId",
            "proportion": "zone_edge_proportion",
        }


@dataclass(frozen=True)
class ImpactsBeamProcessing:
    passenger_vehicle_types_file: str
    freight_vehicle_types_file: str
    population_sample: float
    transit_sample: float = 1.0
    include_non_osm_car_links: bool = False
    include_passenger: bool = True
    include_freight: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ImpactsBeamProcessing":
        _reject_unknown_keys(
            payload,
            {
                "passenger_vehicle_types_file",
                "freight_vehicle_types_file",
                "population_sample",
                "transit_sample",
                "include_non_osm_car_links",
                "include_passenger",
                "include_freight",
            },
            "impacts.beam",
        )
        return cls(
            passenger_vehicle_types_file=_required_string(
                payload.get("passenger_vehicle_types_file"),
                "impacts.beam.passenger_vehicle_types_file",
            ),
            freight_vehicle_types_file=_required_string(
                payload.get("freight_vehicle_types_file"),
                "impacts.beam.freight_vehicle_types_file",
            ),
            population_sample=_required_float(
                payload.get("population_sample"),
                "impacts.beam.population_sample",
            ),
            transit_sample=(
                _optional_float(payload.get("transit_sample"))
                if payload.get("transit_sample") is not None
                else 1.0
            ),
            include_non_osm_car_links=(
                _required_bool(
                    payload.get("include_non_osm_car_links"),
                    "impacts.beam.include_non_osm_car_links",
                )
                if payload.get("include_non_osm_car_links") is not None
                else False
            ),
            include_passenger=(
                _required_bool(payload.get("include_passenger"), "impacts.beam.include_passenger")
                if payload.get("include_passenger") is not None
                else True
            ),
            include_freight=(
                _required_bool(payload.get("include_freight"), "impacts.beam.include_freight")
                if payload.get("include_freight") is not None
                else True
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
    beam: ImpactsBeamProcessing
    emissions: Emissions
    dispersions: Dispersions
    exposure: "Exposure"
    analysis: Analysis = field(default_factory=Analysis)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Impacts":
        _reject_unknown_keys(
            payload,
            {"local_output_folder", "beam", "emissions", "dispersions", "exposure", "analysis"},
            "impacts",
        )
        result = cls(
            local_output_folder=_required_string(payload.get("local_output_folder"), "impacts.local_output_folder"),
            beam=ImpactsBeamProcessing.from_dict(dict(payload.get("beam", {}) or {})),
            emissions=Emissions.from_dict(dict(payload.get("emissions", {}) or {})),
            dispersions=Dispersions.from_dict(dict(payload.get("dispersions", {}) or {})),
            exposure=Exposure.from_dict(dict(payload.get("exposure", {}) or {})),
            analysis=Analysis.from_dict(dict(payload.get("analysis", {}) or {})),
        )
        if result.analysis.sector_targets and not result.emissions.vehicle_category_metadata_file:
            raise ValueError(
                "impacts.emissions.vehicle_category_metadata_file is required when annual sector targets are configured"
            )
        return result


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
                "local_output_folder": self.impacts.local_output_folder,
                "beam": {
                    "passenger_vehicle_types_file": self.impacts.beam.passenger_vehicle_types_file,
                    "freight_vehicle_types_file": self.impacts.beam.freight_vehicle_types_file,
                    "population_sample": self.impacts.beam.population_sample,
                    "transit_sample": self.impacts.beam.transit_sample,
                    "include_non_osm_car_links": self.impacts.beam.include_non_osm_car_links,
                    "include_passenger": self.impacts.beam.include_passenger,
                    "include_freight": self.impacts.beam.include_freight,
                },
                "emissions": {
                    "osm_network_folder": self.impacts.emissions.osm_network_folder,
                    "emissions_rates_folder": self.impacts.emissions.emissions_rates_folder,
                    "inventory": {
                        "passenger_file": self.impacts.emissions.inventory.passenger_file,
                        "freight_file": self.impacts.emissions.inventory.freight_file,
                        "enable_passenger_activity_correction": self.impacts.emissions.inventory.enable_passenger_activity_correction,
                        "enable_freight_activity_correction": self.impacts.emissions.inventory.enable_freight_activity_correction,
                    },
                    "vehicle_category_metadata_file": self.impacts.emissions.vehicle_category_metadata_file,
                    "defaults": {
                        "annualization_days": {
                            "light_duty": self.impacts.emissions.defaults.annualization_days.light_duty,
                            "medium_heavy_duty": self.impacts.emissions.defaults.annualization_days.medium_heavy_duty,
                        }
                    },
                    "pollutants": list(self.impacts.emissions.source_pollutants),
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
