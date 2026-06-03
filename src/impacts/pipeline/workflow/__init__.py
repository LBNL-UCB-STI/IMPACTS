"""Workflow step modules."""
from __future__ import annotations

from typing import Optional

__all__ = [
    "prepare_emissions",
    "step1_process_emissions",
    "step2_compute_inmap_concentrations",
    "step3_compute_aermod_concentrations",
    "_step_label",
]


def _step_label(step: str, zone_label: Optional[str] = None) -> str:
    if zone_label:
        return f"Step {step} [{zone_label}]"
    return f"Step {step}"
