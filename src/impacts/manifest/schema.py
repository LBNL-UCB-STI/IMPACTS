from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from impacts.config.defaults import annualization_days as default_annualization_days


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
    beam_osm_id_col: str
    beam_length_col: str
    beam_osm_epsg: int
    output_epsg: int
    inmap_grid_path: str
    inmap_grid_epsg: int
    mapping_columns: Dict[str, Any]
    isrm_url: Optional[str] = None
    isrm_nox_to_no2_matrix_npz_path: Optional[str] = None
    aermod_grid_path: Optional[str] = None
    aermod_grid_epsg: Optional[int] = None
    aermod_grid_id: Optional[str] = None
    region: Optional[str] = None
    start_year: Optional[int] = None
    county_state_fips: Optional[str] = None
    county_fips_codes: List[str] = field(default_factory=list)
    county_area_name: str = "county"
    activity_totals_file: Optional[str] = None
    activity_totals_columns: Dict[str, Any] = field(default_factory=dict)
    concentration_factor: Optional[float] = None
    iterations: int = 0
    pollutants: List[str] = field(default_factory=list)
    pollutants_map: Dict[str, str] = field(default_factory=dict)
    annualization_days: float = default_annualization_days
    population_sample: float = 1.0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            beam_osm_id_col=_required_string(payload.get("beam_osm_id_col"), "pipeline.beam_osm_id_col"),
            beam_length_col=_required_string(payload.get("beam_length_col"), "pipeline.beam_length_col"),
            beam_osm_epsg=int(_required_string(payload.get("beam_osm_epsg"), "pipeline.beam_osm_epsg")),
            output_epsg=int(_required_string(payload.get("output_epsg"), "pipeline.output_epsg")),
            inmap_grid_path=_required_string(payload.get("inmap_grid_path"), "pipeline.inmap_grid_path"),
            inmap_grid_epsg=int(_required_string(payload.get("inmap_grid_epsg"), "pipeline.inmap_grid_epsg")),
            mapping_columns=_required_dict(payload.get("mapping_columns"), "pipeline.mapping_columns"),
            isrm_url=_optional_string(payload.get("isrm_url")),
            isrm_nox_to_no2_matrix_npz_path=_optional_string(payload.get("isrm_nox_to_no2_matrix_npz_path")),
            aermod_grid_path=_optional_string(payload.get("aermod_grid_path")),
            aermod_grid_epsg=_optional_int(payload.get("aermod_grid_epsg")),
            aermod_grid_id=_optional_string(payload.get("aermod_grid_id")),
            region=_optional_string(payload.get("region")),
            start_year=_optional_int(payload.get("start_year")),
            county_state_fips=_optional_string(payload.get("county_state_fips")),
            county_fips_codes=_coerce_string_list(payload.get("county_fips_codes")),
            county_area_name=_optional_string(payload.get("county_area_name")) or "county",
            activity_totals_file=_optional_string(
                payload.get("activity_totals_file")
                or payload.get("activity_totals_path")
                or payload.get("county_activity_totals_target_path")
            ),
            activity_totals_columns=dict(
                payload.get("activity_totals_columns") or payload.get("county_activity_totals_columns", {}) or {}
            ),
            concentration_factor=_optional_float(payload.get("concentration_factor")),
            iterations=int(payload.get("iterations", 0) or 0),
            pollutants=_coerce_string_list(payload.get("pollutants")),
            pollutants_map=dict(payload.get("pollutants_map", {}) or {}),
            annualization_days=float(payload.get("annualization_days") or default_annualization_days),
            population_sample=float(payload.get("population_sample") or 1.0),
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
