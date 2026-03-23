from __future__ import annotations

from os import PathLike

from . import coverage as coverage
from . import geometry as geometry
from . import localization as localization
from . import plotting as plotting
from . import propagation as propagation
from . import time_utils as time_utils
from . import transforms as transforms

__version__: str

def initialize_orekit(*, data_path: str | PathLike[str] | None = None) -> None: ...
def set_orekit_data_path(data_path: str | PathLike[str]) -> None: ...

__all__ = [
    "__version__",
    "initialize_orekit",
    "set_orekit_data_path",
    "coverage",
    "localization",
    "geometry",
    "transforms",
    "propagation",
    "plotting",
    "time_utils",
]
