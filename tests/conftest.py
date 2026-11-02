from __future__ import annotations

import sys
import os
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT / 'scripts'))


@pytest.fixture
def charging_notebooks():
    """External collaborator reference code is not redistributed in this repo."""
    path = Path(os.environ.get('IMPACTS_CHARGING_NOTEBOOKS', str(ROOT / 'chargingcode')))
    if not (path / 'AssignChargingEvents.ipynb').exists():
        pytest.skip('Set IMPACTS_CHARGING_NOTEBOOKS to trusted collaborator notebooks for parity tests')
    return path
