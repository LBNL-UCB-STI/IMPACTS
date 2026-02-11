import os

STADIAMAPS_API_KEY = "REDACTED_STADIA_KEY"
CENSUS_API_KEY = "REDACTED_CENSUS_KEY"


def register_stadiamaps(api_key: str = STADIAMAPS_API_KEY) -> None:
    os.environ.setdefault("STADIA_MAPS_API_KEY", api_key)


def register_census(api_key: str = CENSUS_API_KEY) -> None:
    os.environ.setdefault("CENSUS_API_KEY", api_key)
