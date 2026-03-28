"""Dispersion utilities retained outside the main pipeline path.

The active InMAP workflow uses:
- ``impacts.step5_inmap_dispersion`` for InMAP concentrations and export

This package currently contains only auxiliary utilities such as NOx-to-NO2 conversion.
"""

__all__ = ["isrm_nox_to_no2"]
