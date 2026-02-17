#!/usr/bin/env python
"""Compatibility wrapper for renamed module `beam_emfac_freight.py`."""

from pathlib import Path
from runpy import run_path

TARGET = Path(__file__).with_name('beam_emfac_freight.py')

if __name__ == '__main__':
    run_path(str(TARGET), run_name='__main__')
