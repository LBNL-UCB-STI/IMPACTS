"""Central registry of all configurable default values for the impacts pipeline.

If you need to know what a default value is, look here first.
No other module should define defaults independently.
"""

# ---------------------------------------------------------------------------
# Emissions
# ---------------------------------------------------------------------------

annualization_days_by_vehicle_group: dict[str, float] = {
    "light_duty": 327.0,
    "medium_heavy_duty": 312.0,
}
"""Default annualization-day fallbacks by EMFAC vehicle group when the vehicle
metadata catalog does not provide operation_days_per_year for a vehicle category."""

pollutants: list = ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BC"]
"""Pollutants processed by default when no explicit list is configured."""

grams_per_short_ton: float = 907_184.74
"""Grams in one U.S. short ton. Maintained preprocess/settings-driven ton conversions use this basis."""

aermod_active_days_per_year: float = 365.0
"""Days per year used when normalizing AERMOD pattern kernels from grams/second/m^2 to annual tons."""

# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

concentrations: list = ["SOA", "pNO3", "pNH4", "pSO4", "PrimaryPM25", "BC", "NO2"]
"""Concentration fields computed directly from ISRM/zarr or the NOx->NO2 fallback."""

tons_per_year_to_ug_per_s: float = grams_per_short_ton * 1_000_000.0 / (365.0 * 24.0 * 3600.0)
"""Unit conversion from short tons/year emissions to micrograms/second emission rate."""

# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------

primary_pm25_strategy_inmap_only: str = "inmap_only"
"""Use InMAP PrimaryPM25 only when constructing the exposure concentration surface."""

primary_pm25_strategy_impute_inmap_primary_with_aermod: str = "impute_inmap_primary_with_aermod"
"""Use AERMOD PrimaryPM25 where available, and fall back to InMAP PrimaryPM25 elsewhere."""

primary_pm25_strategy_scale_aermod_to_inmap_primary: str = "scale_aermod_to_inmap_primary"
"""Scale AERMOD PrimaryPM25 within each InMAP cell to conserve the InMAP PrimaryPM25 budget."""

primary_pm25_integration_strategies: tuple[str, ...] = (
    primary_pm25_strategy_inmap_only,
    primary_pm25_strategy_impute_inmap_primary_with_aermod,
    primary_pm25_strategy_scale_aermod_to_inmap_primary,
)
"""Allowed strategies for integrating primary PM2.5 into the exposure concentration surface."""

primary_pm25_integration_strategy: str = primary_pm25_strategy_impute_inmap_primary_with_aermod
"""Default primary PM2.5 integration strategy."""

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

chunk_size: int = 50_000
"""Default batch size used to chunk large workflow tables for processing and parallelization."""

skims_chunk_size: int = 250_000
"""Default batch size used when streaming large skims tables during preprocessing."""

meters_per_mile: float = 1609.344
"""Meters in one mile — used for distance unit conversion."""

running_processes: list = ["RUNEX", "RUNLOSS", "PMBW", "PMTW", "PRDUST"]
"""Emission processes associated with vehicle movement (per-link traversal)."""

parked_processes: list = ["DIURN", "HOTSOAK", "STREX", "IDLEX"]
"""Emission processes associated with vehicle parking/departure events."""
