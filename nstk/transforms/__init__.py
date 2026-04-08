"""Public API for coordinate transforms.

Conventions used in this package:
- Modules prefixed with ``_`` are internal implementation modules.
- Public functions are re-exported from ``nstk.transforms``.
- Timed-rotation internals live in ``_timed_rotations`` and are exposed via
  package-level ``transform``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nstk.transforms._aer2ecef import (
    aer2ecef,
    aer2enu,
)
from nstk.transforms._aer2geodetic import (
    aer2geodetic,
)
from nstk.transforms._coarse_eci2geodetic import (
    coarse_eci2geodetic,
)
from nstk.transforms._coarse_eci2itrf import (
    DAS2R,
    EARTH_OMEGA,
    J2000_JD,
    coarse_ecef2eci_pos,
    coarse_ecef2eci_pos_vel,
    coarse_eci2ecef_pos,
    coarse_eci2ecef_pos_vel,
)
from nstk.transforms._ecef2aer import ecef2aer
from nstk.transforms._ecef2enu import (
    ecef2enu,
    ecef2enu_delta,
    enu_basis_from_ecef_xyz,
)
from nstk.transforms._ecef2geodetic import (
    ecef2geodetic,
)
from nstk.transforms._enu2ecef import (
    enu2ecef,
    enu2ecef_delta,
)
from nstk.transforms._enu2geodetic import (
    enu2geodetic,
)
from nstk.transforms._geodetic2aer import enu2aer, geodetic2aer
from nstk.transforms._geodetic2ecef import (
    geodetic2ecef,
)
from nstk.transforms._geodetic2enu import (
    enu_basis_from_latlon,
    geodetic2enu,
)
from nstk.transforms.constants import (
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

if TYPE_CHECKING:
    import astropy.units as u
    import numpy as np

__all__ = [
    "aer2enu",
    "aer2ecef",
    "aer2geodetic",
    "coarse_eci2geodetic",
    "coarse_eci2ecef_pos",
    "coarse_eci2ecef_pos_vel",
    "coarse_ecef2eci_pos",
    "coarse_ecef2eci_pos_vel",
    "ecef2aer",
    "enu_basis_from_ecef_xyz",
    "ecef2enu_delta",
    "ecef2enu",
    "ecef2geodetic",
    "enu2ecef_delta",
    "enu2ecef",
    "enu2geodetic",
    "enu2aer",
    "geodetic2aer",
    "geodetic2ecef",
    "enu_basis_from_latlon",
    "geodetic2enu",
    "transform",
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


def transform(
    from_frame: Any,
    to_frame: Any,
    time: Any,
    position: np.ndarray | u.Quantity,
    velocity: np.ndarray | u.Quantity | None = None,
    acceleration: np.ndarray | u.Quantity | None = None,
    *,
    iers_convention: Any = None,
    simple_eop: bool = True,
) -> tuple[
    np.ndarray | u.Quantity,
    np.ndarray | u.Quantity | None,
    np.ndarray | u.Quantity | None,
]:
    """Transform Cartesian state vectors between arbitrary Orekit frames.

    Parameters
    ----------
    from_frame, to_frame : Frame | str
        Source and destination frames. Strings resolve common Orekit frames
        such as ``"gcrf"``, ``"itrf"``, ``"itrf2014"``, ``"tod"``, or
        versioned variants like ``"TOD_CONVENTIONS_2010_SIMPLE_EOP"``.
        For custom frames, pass an Orekit ``Frame`` object directly.
    time : Any
        Scalar or array-like time input. Supported forms include
        ``astropy.time.Time``, Orekit ``AbsoluteDate`` objects, unix seconds
        as numeric values, and time quantities in seconds.
    position : np.ndarray | astropy.units.Quantity
        Cartesian positions with shape ``(..., 3)`` in meters.
    velocity : np.ndarray | astropy.units.Quantity | None, optional
        Cartesian velocities with shape ``(..., 3)`` in meters per second.
    acceleration : np.ndarray | astropy.units.Quantity | None, optional
        Cartesian accelerations with shape ``(..., 3)`` in meters per second
        squared. If provided without ``velocity``, a zero velocity is assumed
        internally so the acceleration transform can be evaluated.
    iers_convention : optional
        Orekit IERS convention used when resolving Earth-fixed frames that do
        not already encode a specific convention/version in the frame string.
        Defaults to the latest convention supported by the installed Orekit.
    simple_eop : bool, default True
        ``simpleEOP`` flag used when resolving Earth-fixed frames.

    Returns
    -------
    tuple[np.ndarray | Quantity, np.ndarray | Quantity | None, np.ndarray | Quantity | None]
        ``(position, velocity, acceleration)`` transformed into ``to_frame``.
        Missing optional inputs are returned as ``None``. Output arrays follow
        the broadcasted input shape with a trailing ``(3,)`` component axis.

    Notes
    -----
    This wrapper preserves the public ``nstk.transforms.transform`` symbol
    while importing the Orekit-backed implementation only when the function
    is first called.
    """

    from nstk.transforms._timed_rotations import transform as _transform

    return _transform(
        from_frame=from_frame,
        to_frame=to_frame,
        time=time,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        iers_convention=iers_convention,
        simple_eop=simple_eop,
    )


def __dir__() -> list[str]:
    return sorted(set(__all__))
