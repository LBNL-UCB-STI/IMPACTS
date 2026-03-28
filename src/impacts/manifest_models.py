from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from impacts.defaults import DEFAULT_ANNUALIZATION_DAYS
from impacts.defaults import DEFAULT_CONCENTRATION_FACTOR


def _required_string(value: Any, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required value: {label}")
    return text


def _required_dict(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping for {label}")
    return value


def _coerce_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


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


@dataclass(frozen=True)
class PipelineConfig:
    beam_network_path: str
    beam_osm_id_col: str
    beam_length_col: str
    beam_osm_epsg: int
    output_epsg: int
    inmap_grid_path: str
    inmap_grid_epsg: int
    mapping_columns: Dict[str, Any]
    isrm_url: Optional[str] = None
    isrm_nox_to_no2_matrix_path: Optional[str] = None
    aermod_grid_path: Optional[str] = None
    aermod_grid_epsg: Optional[int] = None
    aermod_grid_id: Optional[str] = None
    osm_links_path: Optional[str] = None
    osm_pbf_path: Optional[str] = None
    region: Optional[str] = None
    start_year: Optional[int] = None
    county_state_fips: Optional[str] = None
    county_fips_codes: List[str] = field(default_factory=list)
    county_area_name: str = "county"
    county_boundaries_path: Optional[str] = None
    mapping_input_path: Optional[str] = None
    prepared_skims_input_path: Optional[str] = None
    skims_input_path: Optional[str] = None
    activity_corrections_path: Optional[str] = None
    activity_corrections_columns: Dict[str, Any] = field(default_factory=dict)
    concentration_factor: Optional[float] = None
    include_health: bool = False
    events_path: Optional[str] = None
    rates_dir: Optional[str] = None
    link_length_path: Optional[str] = None
    iterations: int = 0
    use_rates: bool = True
    pollutants: List[str] = field(default_factory=list)
    annualization_days: float = DEFAULT_ANNUALIZATION_DAYS

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            beam_network_path=_required_string(payload.get("beam_network_path"), "pipeline.beam_network_path"),
            beam_osm_id_col=_required_string(payload.get("beam_osm_id_col"), "pipeline.beam_osm_id_col"),
            beam_length_col=_required_string(payload.get("beam_length_col"), "pipeline.beam_length_col"),
            beam_osm_epsg=int(_required_string(payload.get("beam_osm_epsg"), "pipeline.beam_osm_epsg")),
            output_epsg=int(_required_string(payload.get("output_epsg"), "pipeline.output_epsg")),
            inmap_grid_path=_required_string(payload.get("inmap_grid_path"), "pipeline.inmap_grid_path"),
            inmap_grid_epsg=int(_required_string(payload.get("inmap_grid_epsg"), "pipeline.inmap_grid_epsg")),
            mapping_columns=_required_dict(payload.get("mapping_columns"), "pipeline.mapping_columns"),
            isrm_url=_optional_string(payload.get("isrm_url")),
            isrm_nox_to_no2_matrix_path=_optional_string(payload.get("isrm_nox_to_no2_matrix_path")),
            aermod_grid_path=_optional_string(payload.get("aermod_grid_path")),
            aermod_grid_epsg=_optional_int(payload.get("aermod_grid_epsg")),
            aermod_grid_id=_optional_string(payload.get("aermod_grid_id")),
            osm_links_path=_optional_string(payload.get("osm_links_path")),
            osm_pbf_path=_optional_string(payload.get("osm_pbf_path")),
            region=_optional_string(payload.get("region")),
            start_year=_optional_int(payload.get("start_year")),
            county_state_fips=_optional_string(payload.get("county_state_fips")),
            county_fips_codes=_coerce_string_list(payload.get("county_fips_codes")),
            county_area_name=_optional_string(payload.get("county_area_name")) or "county",
            county_boundaries_path=_optional_string(payload.get("county_boundaries_path")),
            mapping_input_path=_optional_string(payload.get("mapping_input_path")),
            prepared_skims_input_path=_optional_string(payload.get("prepared_skims_input_path")),
            skims_input_path=_optional_string(payload.get("skims_input_path")),
            activity_corrections_path=_optional_string(payload.get("activity_corrections_path")),
            activity_corrections_columns=dict(payload.get("activity_corrections_columns", {}) or {}),
            concentration_factor=_optional_float(payload.get("concentration_factor")),
            include_health=bool(payload.get("include_health", False)),
            events_path=_optional_string(payload.get("events_path")),
            rates_dir=_optional_string(payload.get("rates_dir")),
            link_length_path=_optional_string(payload.get("link_length_path")),
            iterations=int(payload.get("iterations", 0) or 0),
            use_rates=bool(payload.get("use_rates", True)),
            pollutants=_coerce_string_list(payload.get("prepared_pollutants") or payload.get("pollutants")),
            annualization_days=float(payload.get("annualization_days") or DEFAULT_ANNUALIZATION_DAYS),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputsManifest:
    contract_version: str
    model: str
    runtime_config_source: str
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
        pipeline = _required_dict(payload.get("pipeline"), "pipeline")
        PipelineConfig.from_dict(pipeline)
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            runtime_config_source=_required_string(payload.get("runtime_config_source"), "runtime_config_source"),
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
    raw_output_dir: str
    command: str
    image: str
    raw_outputs: Dict[str, Any]
    pipeline: Dict[str, Any]
    population_inputs: Dict[str, Any]
    deterministic_contract: Dict[str, Any]
    execution: Dict[str, Any]
    run_manifest_path: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RunManifest":
        raw_outputs = _required_dict(payload.get("raw_outputs"), "raw_outputs")
        if "skims_emissions" not in raw_outputs:
            raise ValueError("Run manifest missing raw_outputs.skims_emissions")
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            input_manifest_path=_required_string(payload.get("input_manifest_path"), "input_manifest_path"),
            output_dir=_required_string(payload.get("output_dir"), "output_dir"),
            raw_output_dir=_required_string(payload.get("raw_output_dir"), "raw_output_dir"),
            command=_required_string(payload.get("command"), "command"),
            image=_required_string(payload.get("image"), "image"),
            raw_outputs=raw_outputs,
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
    validation: Dict[str, Any]
    notes: List[str]
    postprocess_manifest_path: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PostprocessManifest":
        canonical_artifact = _required_dict(payload.get("canonical_artifact"), "canonical_artifact")
        if "path" not in canonical_artifact:
            raise ValueError("Postprocess manifest missing canonical_artifact.path")
        return cls(
            contract_version=_required_string(payload.get("contract_version"), "contract_version"),
            model=_required_string(payload.get("model"), "model"),
            run_manifest_path=_required_string(payload.get("run_manifest_path"), "run_manifest_path"),
            output_dir=_required_string(payload.get("output_dir"), "output_dir"),
            canonical_artifact=canonical_artifact,
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
