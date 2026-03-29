from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Dict

from ..manifest.file_ops import file_entry
from .common import annualize_prepared_skims_for_grid_allocation
from .common import prepare_skims_for_grid_allocation
from .common import prepared_table_target
from .common import stage_local_input


def run(
    *,
    manifest_inputs: Dict[str, Any],
    input_root: Path,
    processing,
    skims_input_source: str,
    network_path: str,
) -> dict[str, str]:
    staged_skims_input = stage_local_input(
        manifest_inputs=manifest_inputs,
        input_root=input_root,
        key="skims_input",
        source_path=skims_input_source,
        relative_target=f"skims/{Path(skims_input_source).name}",
        optional=True,
    )

    prepared_grouped_skims_path = prepared_table_target(input_root, "prepared_skims_grouped_for_grid_allocation")
    canonical_pollutants = list(processing.pollutants)
    prepare_skims_for_grid_allocation(
        skims_path=staged_skims_input,
        output_path=str(prepared_grouped_skims_path),
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=canonical_pollutants,
        pollutants_map=dict(processing.pollutants_map),
    )
    prepared_skims_path = prepared_table_target(input_root, "prepared_skims_for_grid_allocation")
    annualize_prepared_skims_for_grid_allocation(
        prepared_skims_path=str(prepared_grouped_skims_path),
        output_path=str(prepared_skims_path),
        network_path=network_path,
        beam_length_col=processing.beam_length_col,
        group_cols=list(processing.prepared_skims_group_cols),
        required_pollutants=canonical_pollutants,
        annualization_days=float(processing.annualization_days),
        population_sample=float(processing.population_sample),
    )
    manifest_inputs["prepared_skims_grouped"] = file_entry(
        kind="local",
        path=str(prepared_grouped_skims_path),
        staged_path=str(prepared_grouped_skims_path),
        optional=True,
    )
    manifest_inputs["prepared_skims_input"] = file_entry(
        kind="local",
        path=str(prepared_skims_path),
        staged_path=str(prepared_skims_path),
        optional=True,
    )
    return {
        "staged_skims_input": str(staged_skims_input),
        "prepared_grouped_skims_path": str(prepared_grouped_skims_path),
        "prepared_skims_path": str(prepared_skims_path),
    }
