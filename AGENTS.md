# Repository Guidelines

## Project Structure & Module Organization
- `src/impacts/`: Primary Python package plus supporting R scripts (`*.R`) and notebooks (`*.ipynb`).
- `data/`: Input and reference data (e.g., shapefiles, `*.RData`); see `data/Readme.md` for details.
- `tests/`: Pytest-based unit tests (example: `tests/test_skeleton.py`).
- `docs/`: Sphinx documentation sources and build artifacts.

## Build, Test, and Development Commands
- `pip install -e .`: Install the package in editable mode for local development.
- `pytest`: Run the test suite directly.
- `tox -e default`: Run tests in an isolated tox environment (invokes pytest).
- `tox -e docs`: Build Sphinx documentation into `docs/_build`.
- `tox -e build`: Build source and wheel distributions via `python -m build`.

## Coding Style & Naming Conventions
- Python style follows standard PEP 8; use 4-space indentation.
- `flake8` configuration lives in `setup.cfg` (88-char line length; ignores `E203`, `W503`).
- Prefer `snake_case` for functions/variables and `PascalCase` for classes.
- R scripts use a numeric prefix for ordering (e.g., `0_convert_*.R`, `1_generate_*.R`).

## Testing Guidelines
- Tests use `pytest` with `pytest-cov` enabled by default (see `setup.cfg` addopts).
- Place new tests under `tests/` and follow the `test_*.py` naming pattern.
- Run focused tests with `pytest tests/test_skeleton.py` during development.

## Commit & Pull Request Guidelines
- Existing commit messages are short, action-focused, and lowercase; no conventional-commit pattern is enforced.
- Keep commits scoped to a single logical change and update tests/docs when behavior changes.
- PRs should include a clear summary, a note of tests run, and links to any related issues.

## Data & Configuration Notes
- The ISRM matrix is expected from `s3://inmap-model/isrm_v1.2.1.zarr/` (see `README.md`).
- Avoid committing large, derived data artifacts unless they are required for reproducibility.
