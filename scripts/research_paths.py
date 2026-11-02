"""Optional external assets for historical research drivers.

Native IMPACTS uses YAML/manifests. These environment variables only replace
machine-specific defaults in archived scenario/diagnostic drivers.
"""
import os
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
ISRM_PATH = os.environ.get('IMPACTS_ISRM_PATH', str(CODE_ROOT / 'data/isrm/isrm_v1.2.1_SFBAY_compact.zarr'))
BREATHE_ROOT = Path(os.environ.get('IMPACTS_BREATHE_ROOT', str(CODE_ROOT / 'external/PowerGeneration')))
BEAM_DATA_ROOT = Path(os.environ.get('IMPACTS_BEAM_DATA_ROOT', str(CODE_ROOT / 'external/beam-data-sfbay')))
CHARGING_REFERENCE = Path(os.environ.get('IMPACTS_CHARGING_REFERENCE', str(CODE_ROOT / 'external/charging_reference.pdf')))
