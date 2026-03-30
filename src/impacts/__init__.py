from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "impacts"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .config.runtime import ImpactsRuntimeConfig
from .config.runtime import Dispersions
from .config.runtime import Emissions
from .config.runtime import Geography
from .config.runtime import GeographyFips
from .config.runtime import Impacts
from .config.runtime import InmapDispersion
from .config.runtime import Run
from .config.runtime import Shared

__all__ = [
    "__version__",
    "Dispersions",
    "Emissions",
    "Geography",
    "GeographyFips",
    "ImpactsRuntimeConfig",
    "Impacts",
    "InmapDispersion",
    "Run",
    "Shared",
]
