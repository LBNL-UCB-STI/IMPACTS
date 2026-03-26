"""Central registry of all configurable default values for the impacts pipeline.

If you need to know what a default value is, look here first.
No other module should define defaults independently.
"""

# ---------------------------------------------------------------------------
# Emissions / annualization
# ---------------------------------------------------------------------------

DEFAULT_ANNUALIZATION_DAYS: float = 330.0
"""Number of representative days per year used to convert per-day gram
emissions to annual tons.  Matches the EMFAC/CARB modelling convention."""

DEFAULT_POLLUTANTS: list = ["NH3", "NOx", "PM2_5", "SOx", "ROG", "BCh"]
"""Pollutants processed by default when no explicit list is configured."""

GRAMS_PER_TON: float = 1_000_000.0
"""Grams in one metric ton — used for unit conversion."""

# ---------------------------------------------------------------------------
# Dispersion
# ---------------------------------------------------------------------------

DEFAULT_CONCENTRATION_FACTOR: float = 28766.639
"""Scaling factor applied when converting ISRM source fractions to
concentration units (μg/m³ per ton/year)."""

DEFAULT_DISPERSION_EMISSIONS_COLUMNS: list = [
    "inmap_srm_cell_id",
    "tons_per_year_ROG",
    "tons_per_year_NOx",
    "tons_per_year_NH3",
    "tons_per_year_SOx",
    "tons_per_year_PM2_5",
    "tons_per_year_BCh",
]

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 50_000
"""Number of unique linkIds processed per chunk in Step 4.1 allocation."""

DEFAULT_SAMPLING_CHUNK_SIZE: int = 200_000
"""Number of rows processed per chunk during sampling/compaction."""

DEFAULT_COMPACT_WORKERS: int = 4
"""Number of parallel workers used during skims compaction."""

# ---------------------------------------------------------------------------
# Column name defaults — events / network
# ---------------------------------------------------------------------------

DEFAULT_EVENTS_COLUMNS: list = [
    "type",
    "vehicle",
    "vehicleType",
    "departureTime",
    "links",
    "linkTravelTime",
    "length",
]

DEFAULT_BEAM_NETWORK_COLUMNS: list = [
    "linkId",
    "linkLength",
    "attributeOrigId",
]

# ---------------------------------------------------------------------------
# Column name defaults — persons / households
# ---------------------------------------------------------------------------

DEFAULT_PERSONS_COLUMNS: list = [
    "household_id",
    "cell_id",
    "age",
    "sex",
    "income",
]

DEFAULT_HOUSEHOLDS_COLUMNS: list = [
    "household_id",
    "cell_id",
    "income",
    "income_category",
]

# ---------------------------------------------------------------------------
# Column name defaults — skims
# ---------------------------------------------------------------------------

DEFAULT_SKIMS_COLUMNS: list = [
    "linkId",
    "vehicleTypeId",
    "process",
    "CH4",
    "CO",
    "CO2",
    "HC",
    "NH3",
    "NOx",
    "PM",
    "PM10",
    "PM2_5",
    "ROG",
    "SOx",
    "TOG",
]

# ---------------------------------------------------------------------------
# Column name defaults — county corrections
# ---------------------------------------------------------------------------

DEFAULT_COUNTY_CORRECTION_COLUMNS: dict = {
    "county_fips": "countyfp",
    "vmt_factor": "vmt_factor",
    "trips_factor": "trips_factor",
}

# ---------------------------------------------------------------------------
# Network / BEAM defaults
# ---------------------------------------------------------------------------

DEFAULT_LANE_WIDTH_M: float = 4.0