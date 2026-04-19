from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from impacts.config.defaults import representative_days_per_year as default_representative_days_per_year
from impacts.config.settings import build_pollutants_map_from_sources
from ..config._coerce import _required_string, _optional_string, _required_int, _optional_int, _required_float, _optional_float, _required_bool, _coerce_string_list, _reject_unknown_keys, _required_float_or_string


def _required_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for {label}")
    return value


@dataclass(frozen=True)
class PipelineConfig:
    beam_osm_id_col: str
    beam_length_col: str
    output_epsg: int
    mapping_columns: Dict[str, Any]
    inmap_enabled: bool
    aermod_enabled: bool
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
    asrv_nox_to_no2_ratios_file: Optional[str] = None
    region: Optional[str] = None
    start_year: Optional[int] = None
    county_state_fips: Optional[str] = None
    county_fips_codes: List[str] = field(default_factory=list)
    activity_totals_file: Optional[str] = None
    activity_totals_columns: Dict[str, Any] = field(default_factory=dict)
    prepared_skims_group_cols: List[str] = field(default_factory=list)
    pollutants: List[str] = field(default_factory=list)
    source_pollutants: List[str] = field(default_factory=list)
    annualization_days_or_file: float | str = default_representative_days_per_year
    population_sample: float = 1.0
    transit_sample: float = 1.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineConfig":
        _reject_unknown_keys(
            payload,
            {
                "beam_osm_id_col",
                "beam_length_col",
                "output_epsg",
                "inmap_enabled",
                "aermod_enabled",
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
                "asrv_nox_to_no2_ratios_file",
                "region",
                "start_year",
                "county_state_fips",
                "county_fips_codes",
                "activity_totals_file",
                "activity_totals_columns",
                "prepared_skims_group_cols",
                "pollutants",
                "source_pollutants",
                "annualization_days_or_file",
                "population_sample",
                "transit_sample",
            },
            "pipeline",
        )
        result = cls(
            beam_osm_id_col=_required_string(payload.get("beam_osm_id_col"), "pipeline.beam_osm_id_col"),
            beam_length_col=_required_string(payload.get("beam_length_col"), "pipeline.beam_length_col"),
            output_epsg=int(_required_string(payload.get("output_epsg"), "pipeline.output_epsg")),
            mapping_columns=_required_dict(payload.get("mapping_columns"), "pipeline.mapping_columns"),
            inmap_enabled=_required_bool(payload.get("inmap_enabled"), "pipeline.inmap_enabled"),
            aermod_enabled=_required_bool(payload.get("aermod_enabled"), "pipeline.aermod_enabled"),
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
            asrv_nox_to_no2_ratios_file=_optional_string(payload.get("asrv_nox_to_no2_ratios_file")),
            region=_required_string(payload.get("region"), "pipeline.region"),
            start_year=_required_int(payload.get("start_year"), "pipeline.start_year"),
            county_state_fips=_required_string(payload.get("county_state_fips"), "pipeline.county_state_fips"),
            county_fips_codes=_coerce_string_list(payload.get("county_fips_codes")),
            activity_totals_file=_optional_string(payload.get("activity_totals_file")),
            activity_totals_columns=_required_dict(payload.get("activity_totals_columns"), "pipeline.activity_totals_columns"),
            prepared_skims_group_cols=_coerce_string_list(payload.get("prepared_skims_group_cols")),
            pollutants=_coerce_string_list(payload.get("pollutants")),
            source_pollutants=_coerce_string_list(payload.get("source_pollutants")),
            annualization_days_or_file=_required_float_or_string(payload.get("annualization_days_or_file"), "pipeline.annualization_days_or_file"),
            population_sample=_required_float(payload.get("population_sample"), "pipeline.population_sample"),
            transit_sample=(
                _optional_float(payload.get("transit_sample"))
                if payload.get("transit_sample") is not None
                else 1.0
            ),
        )
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
    outputs_dir: str
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
                "outputs_dir",
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
            outputs_dir=_required_string(payload.get("outputs_dir"), "outputs_dir"),
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
