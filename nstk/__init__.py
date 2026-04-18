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

_ASTROPY_IERS_DEFAULTS: dict[str, Any] | None = None

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
    "initialize",
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


def initialize(
    *,
    orekit_data_path: str | os.PathLike[str] | None = None,
    cartopy_data_path: str | os.PathLike[str] | None = None,
    offline: bool = False,
) -> None:
    """Eagerly prepare NSTK's Orekit and plotting runtime integrations.

    Call this once near the top of a notebook or script when you want all
    Orekit-backed wrappers plus NSTK's offline Cartopy plotting assets ready
    before first use. When no explicit paths are provided, NSTK uses the
    bundled ``nstk-data`` package. Set ``offline=True`` to disable Astropy IERS
    network fetches and relax IERS age checks for offline-safe workflows. This
    function is safe to call more than once; repeated calls reuse the
    initialized runtime and refresh the requested NSTK runtime options without
    reinitializing the JVM.
    """

    from nstk._orekit_runtime import ensure_orekit_runtime

    _configure_astropy_iers_runtime(offline=bool(offline))
    ensure_orekit_runtime(data_path=orekit_data_path)

    try:
        from nstk.plotting._cartopy_data import configure_cartopy_data_dir
    except ModuleNotFoundError as exc:
        if exc.name == "cartopy" and cartopy_data_path is None:
            return
        raise

    configure_cartopy_data_dir(data_dir=cartopy_data_path)


def _configure_astropy_iers_runtime(*, offline: bool) -> None:
    """Apply or restore Astropy IERS settings for offline-safe NSTK usage."""

    global _ASTROPY_IERS_DEFAULTS

    if not offline and _ASTROPY_IERS_DEFAULTS is None:
        return

    try:
        from astropy.utils import iers
    except ModuleNotFoundError:
        return

    if _ASTROPY_IERS_DEFAULTS is None:
        _ASTROPY_IERS_DEFAULTS = {
            "auto_download": iers.conf.auto_download,
            "auto_max_age": iers.conf.auto_max_age,
            "remote_timeout": iers.conf.remote_timeout,
            "iers_degraded_accuracy": iers.conf.iers_degraded_accuracy,
        }

    if offline:
        iers.conf.auto_download = False
        iers.conf.auto_max_age = None
        iers.conf.remote_timeout = 1.0
        iers.conf.iers_degraded_accuracy = "warn"
        return

    iers.conf.auto_download = _ASTROPY_IERS_DEFAULTS["auto_download"]
    iers.conf.auto_max_age = _ASTROPY_IERS_DEFAULTS["auto_max_age"]
    iers.conf.remote_timeout = _ASTROPY_IERS_DEFAULTS["remote_timeout"]
    iers.conf.iers_degraded_accuracy = _ASTROPY_IERS_DEFAULTS["iers_degraded_accuracy"]


def initialize_orekit(*, data_path: str | os.PathLike[str] | None = None) -> None:
    """Optionally force eager Orekit/JVM initialization for Nebula Space Toolkit.

    Most users do not need to call this. Nebula Space Toolkit initializes Orekit-backed
    features lazily on first use. Prefer ``nstk.initialize()`` when you want a
    single root-level setup call for Orekit plus NSTK's bundled Cartopy data.
    Use ``nstk.initialize(offline=True)`` for offline-friendly notebook/script
    setup. This method remains available for advanced workflows that only want
    eager Orekit startup and/or want to provide a custom Orekit data directory.
    """

    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime(data_path=data_path)


def set_orekit_data_path(data_path: str | os.PathLike[str]) -> None:
    """Configure a custom Orekit data directory for future lazy initialization.

    Call this before using Orekit-backed Nebula Space Toolkit features when you want nstk
    to use a user-managed Orekit data folder instead of the bundled
    ``nstk-data`` package. ``nstk.initialize(orekit_data_path=...)`` is the
    one-call eager alternative.
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
