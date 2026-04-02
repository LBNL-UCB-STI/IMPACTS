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

from .config.settings import ImpactsSettings
from .config.settings import Dispersions
from .config.settings import Emissions
from .config.settings import Geography
from .config.settings import GeographyFips
from .config.settings import Impacts
from .config.settings import InmapDispersion
from .config.settings import Run
from .config.settings import Shared

__all__ = [
    "__version__",
    "Dispersions",
    "Emissions",
    "Geography",
    "GeographyFips",
    "ImpactsSettings",
    "Impacts",
    "InmapDispersion",
    "Run",
    "Shared",
]
