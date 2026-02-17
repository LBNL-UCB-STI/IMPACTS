import os

def _get_required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def register_stadiamaps(api_key: str | None = None) -> None:
    key = api_key if api_key else _get_required_env("STADIA_MAPS_API_KEY")
    os.environ.setdefault("STADIA_MAPS_API_KEY", key)


def register_census(api_key: str | None = None) -> None:
    key = api_key if api_key else _get_required_env("CENSUS_API_KEY")
    os.environ.setdefault("CENSUS_API_KEY", key)
