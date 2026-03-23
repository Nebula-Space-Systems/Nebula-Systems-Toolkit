from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import coverage as coverage
    from . import geometry as geometry
    from . import localization as localization
    from . import plotting as plotting
    from . import propagation as propagation
    from . import time_utils as time_utils
    from . import transforms as transforms

try:
    __version__ = version("nstk")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

_SUBPACKAGES = {
    "coverage",
    "localization",
    "geometry",
    "transforms",
    "propagation",
    "plotting",
}
_MODULE_EXPORTS = {"time_utils"}

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


def initialize_orekit(*, data_path: str | os.PathLike[str] | None = None) -> None:
    """Optionally force eager Orekit/JVM initialization for Nebula Space Toolkit.

    Most users do not need to call this. Nebula Space Toolkit initializes Orekit-backed
    features lazily on first use. This method remains available for advanced
    workflows that prefer eager startup and/or want to provide a custom
    Orekit data directory.
    """

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime(data_path=data_path)


def set_orekit_data_path(data_path: str | os.PathLike[str]) -> None:
    """Configure a custom Orekit data directory for future lazy initialization.

    Call this before using Orekit-backed Nebula Space Toolkit features when you want nstk
    to use a user-managed Orekit data folder instead of the bundled
    ``nstk-data`` package.
    """

    from nstk._orekit_runtime import configure_orekit_data_path

    configure_orekit_data_path(data_path)


def __getattr__(name: str) -> Any:
    if name in _SUBPACKAGES or name in _MODULE_EXPORTS:
        mod = import_module(f"nstk.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'nstk' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _SUBPACKAGES | _MODULE_EXPORTS)
