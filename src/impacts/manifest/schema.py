from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from impacts.config.defaults import annualization_days_by_vehicle_group as default_annualization_days_by_vehicle_group
from impacts.config.settings import build_pollutants_map_from_sources
from ..config._coerce import _required_string, _optional_string, _required_int, _optional_int, _required_float, _optional_float, _required_bool, _coerce_string_list, _reject_unknown_keys


def _required_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for {label}")
    return value


@dataclass(frozen=True)
class PipelineConfig:
    emissions_enabled: bool
    beam_osm_id_col: str
    beam_length_col: str
    output_epsg: int
    mapping_columns: Dict[str, Any]
    inmap_enabled: bool
    aermod_enabled: bool
    exposure_enabled: bool
    grid_size_meters: Optional[float] = None
    inmap_grid_path: Optional[str] = None
    inmap_grid_epsg: Optional[int] = None
    isrm_url: Optional[str] = None
    isrm_nox_to_no2_ratios_file: Optional[str] = None
    asrv_patterns_file: Optional[str] = None
    asrv_patterns_epsg: Optional[int] = None
    aermod_full_grid_path: Optional[str] = None
    aermod_grid_path: Optional[str] = None
    aermod_grid_epsg: Optional[int] = None
    aermod_grid_id: Optional[str] = None
    region: Optional[str] = None
    start_year: Optional[int] = None
    county_state_fips: Optional[str] = None
    county_fips_codes: List[str] = field(default_factory=list)
    passenger_inventory_file: Optional[str] = None
    freight_inventory_file: Optional[str] = None
    enable_passenger_inventory_activity_correction: bool = True
    enable_freight_inventory_activity_correction: bool = True
    passenger_vehicle_types_file: Optional[str] = None
    freight_vehicle_types_file: Optional[str] = None
    vehicle_category_metadata_file: Optional[str] = None
    prepared_skims_group_cols: List[str] = field(default_factory=list)
    pollutants: List[str] = field(default_factory=list)
    source_pollutants: List[str] = field(default_factory=list)
    annualization_days: Dict[str, float] = field(
        default_factory=lambda: dict(default_annualization_days_by_vehicle_group)
    )
    population_sample: float = 1.0
    transit_sample: float = 1.0
    include_non_osm_car_links: bool = False
    include_passenger: bool = True
    include_freight: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineConfig":
        _reject_unknown_keys(
            payload,
            {
                "beam_osm_id_col",
                "beam_length_col",
                "output_epsg",
                "emissions_enabled",
                "inmap_enabled",
                "aermod_enabled",
                "exposure_enabled",
                "inmap_grid_path",
                "inmap_grid_epsg",
                "mapping_columns",
                "isrm_url",
                "isrm_nox_to_no2_ratios_file",
                "asrv_patterns_file",
                "asrv_patterns_epsg",
                "grid_size_meters",
                "aermod_full_grid_path",
                "aermod_grid_path",
                "aermod_grid_epsg",
                "aermod_grid_id",
                "region",
                "start_year",
                "county_state_fips",
                "county_fips_codes",
                "passenger_inventory_file",
                "freight_inventory_file",
                "enable_passenger_inventory_activity_correction",
                "enable_freight_inventory_activity_correction",
                "passenger_vehicle_types_file",
                "freight_vehicle_types_file",
                "vehicle_category_metadata_file",
                "prepared_skims_group_cols",
                "pollutants",
                "source_pollutants",
                "annualization_days",
                "population_sample",
                "transit_sample",
                "include_non_osm_car_links",
                "include_passenger",
                "include_freight",
            },
            "pipeline",
        )
        result = cls(
            emissions_enabled=_required_bool(payload.get("emissions_enabled"), "pipeline.emissions_enabled"),
            beam_osm_id_col=_required_string(payload.get("beam_osm_id_col"), "pipeline.beam_osm_id_col"),
            beam_length_col=_required_string(payload.get("beam_length_col"), "pipeline.beam_length_col"),
            output_epsg=int(_required_string(payload.get("output_epsg"), "pipeline.output_epsg")),
            mapping_columns=_required_dict(payload.get("mapping_columns"), "pipeline.mapping_columns"),
            inmap_enabled=_required_bool(payload.get("inmap_enabled"), "pipeline.inmap_enabled"),
            aermod_enabled=_required_bool(payload.get("aermod_enabled"), "pipeline.aermod_enabled"),
            exposure_enabled=_required_bool(payload.get("exposure_enabled"), "pipeline.exposure_enabled"),
            grid_size_meters=_optional_float(payload.get("grid_size_meters")),
            inmap_grid_path=_optional_string(payload.get("inmap_grid_path")),
            inmap_grid_epsg=_optional_int(payload.get("inmap_grid_epsg")),
            isrm_url=_optional_string(payload.get("isrm_url")),
            isrm_nox_to_no2_ratios_file=_optional_string(payload.get("isrm_nox_to_no2_ratios_file")),
            asrv_patterns_file=_optional_string(payload.get("asrv_patterns_file")),
            asrv_patterns_epsg=_optional_int(payload.get("asrv_patterns_epsg")),
            aermod_full_grid_path=_optional_string(payload.get("aermod_full_grid_path")),
            aermod_grid_path=_optional_string(payload.get("aermod_grid_path")),
            aermod_grid_epsg=_optional_int(payload.get("aermod_grid_epsg")),
            aermod_grid_id=_optional_string(payload.get("aermod_grid_id")),
            region=_required_string(payload.get("region"), "pipeline.region"),
            start_year=_required_int(payload.get("start_year"), "pipeline.start_year"),
            county_state_fips=_required_string(payload.get("county_state_fips"), "pipeline.county_state_fips"),
            county_fips_codes=_coerce_string_list(payload.get("county_fips_codes")),
            enable_passenger_inventory_activity_correction=_required_bool(
                payload.get("enable_passenger_inventory_activity_correction", True),
                "pipeline.enable_passenger_inventory_activity_correction",
            ),
            enable_freight_inventory_activity_correction=_required_bool(
                payload.get("enable_freight_inventory_activity_correction", True),
                "pipeline.enable_freight_inventory_activity_correction",
            ),
            passenger_inventory_file=(
                _required_string(
                    payload.get("passenger_inventory_file"),
                    "pipeline.passenger_inventory_file",
                )
                if payload.get("enable_passenger_inventory_activity_correction", True)
                else _optional_string(payload.get("passenger_inventory_file"))
            ),
            freight_inventory_file=(
                _required_string(
                    payload.get("freight_inventory_file"),
                    "pipeline.freight_inventory_file",
                )
                if payload.get("enable_freight_inventory_activity_correction", True)
                else _optional_string(payload.get("freight_inventory_file"))
            ),
            passenger_vehicle_types_file=_required_string(
                payload.get("passenger_vehicle_types_file"),
                "pipeline.passenger_vehicle_types_file",
            ),
            freight_vehicle_types_file=_required_string(
                payload.get("freight_vehicle_types_file"),
                "pipeline.freight_vehicle_types_file",
            ),
            vehicle_category_metadata_file=_optional_string(payload.get("vehicle_category_metadata_file")),
            prepared_skims_group_cols=_coerce_string_list(payload.get("prepared_skims_group_cols")),
            pollutants=_coerce_string_list(payload.get("pollutants")),
            source_pollutants=_coerce_string_list(payload.get("source_pollutants")),
            annualization_days=_required_dict(payload.get("annualization_days"), "pipeline.annualization_days"),
            population_sample=_required_float(payload.get("population_sample"), "pipeline.population_sample"),
            transit_sample=_optional_float(payload.get("transit_sample"), default=1.0),
            include_non_osm_car_links=_required_bool(
                payload.get("include_non_osm_car_links", False), "pipeline.include_non_osm_car_links"
            ),
            include_passenger=_required_bool(payload.get("include_passenger", True), "pipeline.include_passenger"),
            include_freight=_required_bool(payload.get("include_freight", True), "pipeline.include_freight"),
        )
        for key in ("light_duty", "medium_heavy_duty"):
            if key not in result.annualization_days:
                raise ValueError(f"Missing required value: pipeline.annualization_days.{key}")
            result.annualization_days[key] = float(result.annualization_days[key])
        if not result.source_pollutants:
            raise ValueError("Missing required value: pipeline.source_pollutants")
        if not result.pollutants:
            raise ValueError("Missing required value: pipeline.pollutants")
        if result.inmap_enabled:
            if not result.inmap_grid_path:
                raise ValueError("Missing required value: pipeline.inmap_grid_path")
            if not result.isrm_url:
                raise ValueError("Missing required value: pipeline.isrm_url")
            if not result.isrm_nox_to_no2_ratios_file:
                raise ValueError("Missing required value: pipeline.isrm_nox_to_no2_ratios_file")
        if result.aermod_enabled:
            if result.grid_size_meters is None:
                raise ValueError("Missing required value: pipeline.grid_size_meters")
            if not result.aermod_full_grid_path:
                raise ValueError("Missing required value: pipeline.aermod_full_grid_path")
            if not result.aermod_grid_path:
                raise ValueError("Missing required value: pipeline.aermod_grid_path")
            if not result.asrv_patterns_file:
                raise ValueError("Missing required value: pipeline.asrv_patterns_file")
        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def pollutants_map(self) -> Dict[str, str]:
        return build_pollutants_map_from_sources(list(self.source_pollutants))


@dataclass(frozen=True)
class InputsManifest:
    contract_version: str
    model: str
    settings_source: str
    staging_dir: str
    input_dir: str
    inputs_manifest_path: str
    maintained_execution_path: List[str]
    inputs: Dict[str, Any]
    pipeline: Dict[str, Any]
    pilates_contract: Dict[str, Any]
    population_inputs: Dict[str, Any]
    notes: List[str]

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InputsManifest":
        _reject_unknown_keys(
            payload,
            {
                "contract_version",
                "model",
                "settings_source",
                "staging_dir",
                "input_dir",
                "inputs_manifest_path",
                "maintained_execution_path",
                "inputs",
                "pipeline",
                "pilates_contract",
                "population_inputs",
                "notes",
            },
            "inputs manifest",
        )
        pipeline = _required_dict(payload.get("pipeline"), "pipeline")
        # Validate the embedded pipeline payload against the maintained schema.
        PipelineConfig.from_dict(pipeline)
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            settings_source=_required_string(payload.get("settings_source"), "settings_source"),
            staging_dir=_required_string(payload.get("staging_dir"), "staging_dir"),
            input_dir=_required_string(payload.get("input_dir"), "input_dir"),
            inputs_manifest_path=_required_string(payload.get("inputs_manifest_path"), "inputs_manifest_path"),
            maintained_execution_path=_coerce_string_list(payload.get("maintained_execution_path")),
            inputs=_required_dict(payload.get("inputs"), "inputs"),
            pipeline=pipeline,
            pilates_contract=_required_dict(payload.get("pilates_contract"), "pilates_contract"),
            population_inputs=_required_dict(payload.get("population_inputs"), "population_inputs"),
            notes=_coerce_string_list(payload.get("notes")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def path(self) -> Path:
        return Path(self.inputs_manifest_path)


@dataclass(frozen=True)
class RunManifest:
    contract_version: str
    model: str
    input_manifest_path: str
    output_dir: str
    command: str
    image: str
    outputs: Dict[str, Any]
    pipeline: Dict[str, Any]
    population_inputs: Dict[str, Any]
    deterministic_contract: Dict[str, Any]
    execution: Dict[str, Any]
    run_manifest_path: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunManifest":
        _reject_unknown_keys(
            payload,
            {
                "contract_version",
                "model",
                "input_manifest_path",
                "output_dir",
                "command",
                "image",
                "outputs",
                "pipeline",
                "population_inputs",
                "deterministic_contract",
                "execution",
                "run_manifest_path",
            },
            "run manifest",
        )
        outputs = _required_dict(payload.get("outputs"), "outputs")
        if "skims_emissions" not in outputs:
            raise ValueError("Run manifest missing outputs.skims_emissions")
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            input_manifest_path=_required_string(payload.get("input_manifest_path"), "input_manifest_path"),
            output_dir=_required_string(payload.get("output_dir"), "output_dir"),
            command=_required_string(payload.get("command"), "command"),
            image=_required_string(payload.get("image"), "image"),
            outputs=outputs,
            pipeline=_required_dict(payload.get("pipeline"), "pipeline"),
            population_inputs=_required_dict(payload.get("population_inputs"), "population_inputs"),
            deterministic_contract=_required_dict(payload.get("deterministic_contract"), "deterministic_contract"),
            execution=_required_dict(payload.get("execution"), "execution"),
            run_manifest_path=_required_string(payload.get("run_manifest_path"), "run_manifest_path"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def path(self) -> Path:
        return Path(self.run_manifest_path)


@dataclass(frozen=True)
class ActivitiesManifest:
    contract_version: str
    model: str
    settings_source: str
    output_dir: str
    region_label: str
    calendar_year: int
    scenario: str
    vehicle_category_metadata_file: str
    outputs: Dict[str, Any]
    notes: List[str]
    activities_manifest_path: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ActivitiesManifest":
        _reject_unknown_keys(
            payload,
            {
                "contract_version",
                "model",
                "settings_source",
                "output_dir",
                "region_label",
                "calendar_year",
                "scenario",
                "vehicle_category_metadata_file",
                "outputs",
                "notes",
                "activities_manifest_path",
            },
            "activities manifest",
        )
        outputs = _required_dict(payload.get("outputs"), "outputs")
        required_outputs = {
            "outputs_root",
            "passenger_rates_file",
            "passenger_activity_file",
            "passenger_fleet_file",
            "freight_rates_file",
            "freight_activity_file",
            "freight_fleet_file",
            "emissions_store_root",
        }
        missing = sorted(key for key in required_outputs if key not in outputs or outputs.get(key) in (None, ""))
        if missing:
            raise ValueError("Activities manifest missing " + ", ".join(f"outputs.{key}" for key in missing))
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            settings_source=_required_string(payload.get("settings_source"), "settings_source"),
            output_dir=_required_string(payload.get("output_dir"), "output_dir"),
            region_label=_required_string(payload.get("region_label"), "region_label"),
            calendar_year=_required_int(payload.get("calendar_year"), "calendar_year"),
            scenario=_required_string(payload.get("scenario"), "scenario"),
            vehicle_category_metadata_file=_required_string(
                payload.get("vehicle_category_metadata_file"),
                "vehicle_category_metadata_file",
            ),
            outputs=outputs,
            notes=_coerce_string_list(payload.get("notes")),
            activities_manifest_path=_required_string(
                payload.get("activities_manifest_path"),
                "activities_manifest_path",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def path(self) -> Path:
        return Path(self.activities_manifest_path)


@dataclass(frozen=True)
class PostprocessManifest:
    contract_version: str
    model: str
    run_manifest_path: str
    output_dir: str
    canonical_artifact: Dict[str, Any]
    analysis_outputs: Dict[str, Any]
    validation: Dict[str, Any]
    notes: List[str]
    postprocess_manifest_path: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PostprocessManifest":
        _reject_unknown_keys(
            payload,
            {
                "contract_version",
                "model",
                "run_manifest_path",
                "output_dir",
                "canonical_artifact",
                "analysis_outputs",
                "validation",
                "notes",
                "postprocess_manifest_path",
            },
            "postprocess manifest",
        )
        canonical_artifact = _required_dict(payload.get("canonical_artifact"), "canonical_artifact")
        if "path" not in canonical_artifact:
            raise ValueError("Postprocess manifest missing canonical_artifact.path")
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            run_manifest_path=_required_string(payload.get("run_manifest_path"), "run_manifest_path"),
            output_dir=_required_string(payload.get("output_dir"), "output_dir"),
            canonical_artifact=canonical_artifact,
            analysis_outputs=_required_dict(payload.get("analysis_outputs"), "analysis_outputs"),
            validation=_required_dict(payload.get("validation"), "validation"),
            notes=_coerce_string_list(payload.get("notes")),
            postprocess_manifest_path=_required_string(
                payload.get("postprocess_manifest_path"),
                "postprocess_manifest_path",
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def path(self) -> Path:
        return Path(self.postprocess_manifest_path)
