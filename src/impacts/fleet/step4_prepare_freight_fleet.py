"""Fleet Step 4: prepare freight fleet inputs.

This step is intentionally minimal for now and acts as the dedicated hook for
freight-specific preparation between passenger mapping and freight mapping.
"""

from __future__ import annotations

from typing import Any


def run_step4(workflow: dict[str, Any]) -> dict[str, Any]:
    """Step 4: placeholder freight-preparation hook."""
    return workflow
