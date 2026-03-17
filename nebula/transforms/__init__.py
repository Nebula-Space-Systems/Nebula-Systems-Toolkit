"""Public API for coordinate transforms.

Conventions used in this package:
- Modules prefixed with ``_`` are internal implementation modules.
- Public functions are re-exported from ``nebula.transforms``.
- Timed-rotation internals live in ``_timed_rotations`` and are exposed via
  package-level ``transform`` / ``initialize_timed_rotations``.
"""

from importlib import import_module
from typing import Any

from nebula.transforms._aer2ecef import (
    aer2ecef,
    aer2ecef_vec_aer,
    aer2ecef_vec_aer3,
    aer2enu,
)
from nebula.transforms._aer2geodetic import (
    aer2geodetic,
    aer2geodetic_vec_aer,
    aer2geodetic_vec_aer3,
)
from nebula.transforms._coarse_eci2geodetic import (
    coarse_eci2geodetic,
    coarse_eci2geodetic_deg,
    coarse_eci2geodetic_vec,
    coarse_eci2geodetic_vec_deg,
)
from nebula.transforms._coarse_eci2itrf import (
    DAS2R,
    EARTH_OMEGA,
    J2000_JD,
    coarse_ecef2eci,
    coarse_ecef2eci_pos,
    coarse_ecef2eci_pos_vec,
    coarse_ecef2eci_pos_vel,
    coarse_ecef2eci_pos_vel_vec,
    coarse_ecef2eci_vec,
    coarse_eci2ecef_pos,
    coarse_eci2ecef_pos_vec,
    coarse_eci2ecef_pos_vel,
    coarse_eci2ecef_pos_vel_vec,
)
from nebula.transforms._ecef2aer import ecef2aer, ecef2aer_vec_xyz
from nebula.transforms._ecef2enu import (
    ecef2enu,
    ecef2enu_delta,
    ecef2enu_vec_ecef,
    ecef2enu_vec_xyz,
    enu_basis_from_ecef_xyz,
)
from nebula.transforms._ecef2geodetic import (
    ecef2geodetic,
    ecef2geodetic_deg,
    ecef2geodetic_vec_ecef,
    ecef2geodetic_vec_ecef_deg,
    ecef2geodetic_vec_xyz,
)
from nebula.transforms._enu2ecef import (
    enu2ecef,
    enu2ecef_delta,
    enu2ecef_vec_enu,
    enu2ecef_vec_enu3,
)
from nebula.transforms._enu2geodetic import (
    enu2geodetic,
    enu2geodetic_vec_enu,
    enu2geodetic_vec_enu3,
)
from nebula.transforms._geodetic2aer import enu2aer, geodetic2aer, geodetic2aer_vec_llh
from nebula.transforms._geodetic2ecef import (
    geodetic2ecef,
    geodetic2ecef_vec_lla,
    geodetic2ecef_vec_llh,
)
from nebula.transforms._geodetic2enu import (
    enu_basis_from_latlon,
    geodetic2enu,
    geodetic2enu_vec_lla,
    geodetic2enu_vec_llh,
)
from nebula.transforms.constants import (
    DEG2RAD,
    HALF_PI,
    PI,
    RAD2DEG,
    TWO_PI,
    WGS84_A,
    WGS84_A2,
    WGS84_B,
    WGS84_B2,
    WGS84_B2_OVER_A2,
    WGS84_E2,
    WGS84_EP2,
)

__all__ = [
    "aer2enu",
    "aer2ecef",
    "aer2ecef_vec_aer",
    "aer2ecef_vec_aer3",
    "aer2geodetic",
    "aer2geodetic_vec_aer",
    "aer2geodetic_vec_aer3",
    "coarse_eci2geodetic",
    "coarse_eci2geodetic_deg",
    "coarse_eci2geodetic_vec",
    "coarse_eci2geodetic_vec_deg",
    "coarse_eci2ecef_pos",
    "coarse_eci2ecef_pos_vel",
    "coarse_eci2ecef_pos_vec",
    "coarse_eci2ecef_pos_vel_vec",
    "coarse_ecef2eci",
    "coarse_ecef2eci_pos",
    "coarse_ecef2eci_pos_vel",
    "coarse_ecef2eci_vec",
    "coarse_ecef2eci_pos_vec",
    "coarse_ecef2eci_pos_vel_vec",
    "ecef2aer",
    "ecef2aer_vec_xyz",
    "enu_basis_from_ecef_xyz",
    "ecef2enu_delta",
    "ecef2enu",
    "ecef2enu_vec_xyz",
    "ecef2enu_vec_ecef",
    "ecef2geodetic",
    "ecef2geodetic_deg",
    "ecef2geodetic_vec_xyz",
    "ecef2geodetic_vec_ecef",
    "ecef2geodetic_vec_ecef_deg",
    "enu2ecef_delta",
    "enu2ecef",
    "enu2ecef_vec_enu",
    "enu2ecef_vec_enu3",
    "enu2geodetic",
    "enu2geodetic_vec_enu",
    "enu2geodetic_vec_enu3",
    "enu2aer",
    "geodetic2aer",
    "geodetic2aer_vec_llh",
    "geodetic2ecef",
    "geodetic2ecef_vec_llh",
    "geodetic2ecef_vec_lla",
    "enu_basis_from_latlon",
    "geodetic2enu",
    "geodetic2enu_vec_llh",
    "geodetic2enu_vec_lla",
    "transform",
    "initialize_timed_rotations",
    "WGS84_A",
    "WGS84_B",
    "WGS84_A2",
    "WGS84_B2",
    "WGS84_B2_OVER_A2",
    "WGS84_E2",
    "WGS84_EP2",
    "DEG2RAD",
    "RAD2DEG",
    "PI",
    "TWO_PI",
    "HALF_PI",
    "J2000_JD",
    "EARTH_OMEGA",
    "DAS2R",
]

_TIMED_ROTATION_EXPORTS = {
    "transform",
    "initialize_timed_rotations",
}


def __getattr__(name: str) -> Any:
    if name in _TIMED_ROTATION_EXPORTS:
        mod = import_module("nebula.transforms._timed_rotations")
        value = getattr(mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'nebula.transforms' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(__all__))
