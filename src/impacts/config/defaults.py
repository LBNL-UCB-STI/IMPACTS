"""Central registry of all configurable default values for the impacts pipeline.

If you need to know what a default value is, look here first.
No other module should define defaults independently.
"""

# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

annualization_days: float = 330.0
"""Number of representative days per year used to convert per-day gram
emissions to annual tons.  Matches the EMFAC/CARB modelling convention."""

pollutants: list = ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BC"]
"""Pollutants processed by default when no explicit list is configured."""

grams_per_ton: float = 1_000_000.0
"""Grams in one metric ton — used for unit conversion."""

# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

concentrations: list = ["SOA", "pNO3", "pNH4", "pSO4", "PrimaryPM25", "BC", "NO2"]
"""Concentration fields computed directly from ISRM/zarr or the NOx->NO2 fallback."""

tons_per_year_to_ug_per_s: float = 28766.639
"""Unit conversion from tons/year emissions to micrograms/second emission rate."""

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

chunk_size: int = 50_000
"""Default batch size used to chunk large workflow tables for processing and parallelization."""
