# ISRM Wrapper

A Python toolkit for working with InMAP Source-Receptor Matrices (ISRM) and converting BEAM emission outputs into pollutant concentrations.

## Overview

This package provides utilities to:
- Convert InMAP outputs to NOx-to-NO2 source-receptor relationships
- Map these relationships to ISRM grid cells
- Process BEAM emission data for air quality analysis

## Installation #WIP

```bash
# Clone the repository
git clone <repository-url>
cd isrm-wrapper

# Install dependencies with Poetry
poetry install

# Or activate the environment
poetry shell
```

## Project Structure #WIP

```
isrm-wrapper/
├── src/isrm_wrapper/    # Main package code
├── data/
│   ├── raw/            # Input BEAM emissions and InMAP outputs
│   └── processed/      # Generated concentration outputs
├── tests/              # Unit tests
└── docs/               # Documentation
```

## Requirements #WIP

- Python 3.8+
- Dependencies managed via Poetry (see `pyproject.toml`)

## Contributing #WIP

See `CONTRIBUTING.rst` for guidelines.

## License #WIP

See `LICENSE.txt` for details.


