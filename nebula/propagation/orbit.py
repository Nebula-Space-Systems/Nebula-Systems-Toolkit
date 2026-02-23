from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Sequence, Any
import math
import threading
from importlib import import_module

import numpy as np
from nebula.transforms._ecef2geodetic import (
    ecef2geodetic_deg,
    ecef2geodetic_vec_ecef_deg,
)

try:
    from astropy.time import Time as AstropyTime  # type: ignore
except Exception:  # pragma: no cover
    AstropyTime = None  # type: ignore

_FastOrbit = None  # type: ignore
_FAST_ORBIT_IMPORT_ERROR: Optional[Exception] = None

import jdk4py
import os
import orekit_jpype

_OREKIT_READY = False
_OREKIT_INIT_LOCK = threading.Lock()
_OREKIT_DATA_PATH: Optional[str] = None
_OREKIT_FAULTHANDLER_DISABLED = False


class _LazyJavaProxy:
    """
    Lightweight proxy for lazily bound JPype Java classes.
    """

    __slots__ = ("_name", "_target")

    def __init__(self, name: str) -> None:
        self._name = name
        self._target = None

    def bind(self, target) -> None:
        self._target = target

    def _ensure_target(self):
        if self._target is None:
            initialize_orekit()
        if self._target is None:
            raise RuntimeError(
                f"{self._name} is not available (Orekit JVM not initialized)"
            )
        return self._target

    def __getattr__(self, item):
        return getattr(self._ensure_target(), item)

    def __call__(self, *args, **kwargs):
        return self._ensure_target()(*args, **kwargs)


# Stable exported symbols; targets are bound lazily after JVM startup.
FramesFactory = _LazyJavaProxy("FramesFactory")
TimeScalesFactory = _LazyJavaProxy("TimeScalesFactory")
AbsoluteDate = _LazyJavaProxy("AbsoluteDate")
IERSConventions = _LazyJavaProxy("IERSConventions")


def initialize_orekit(*, data_path: Optional[str] = None) -> None:
    """
    Initialize the Orekit JVM runtime and data providers for the current process.

    This function is idempotent: repeated calls are safe and return immediately
    after the first successful initialization.

    Parameters
    ----------
    data_path : str | None, optional
        Path to an Orekit data directory. If ``None``, Nebula uses its bundled
        repository path under ``data/orekit-data``.

    Notes
    -----
    - Public entry point for explicit runtime setup in scripts/services.
    - Most Orbit constructors call this automatically.
    """
    global _OREKIT_READY, _OREKIT_DATA_PATH, _OREKIT_FAULTHANDLER_DISABLED
    global FramesFactory, TimeScalesFactory, AbsoluteDate, IERSConventions

    if _OREKIT_READY:
        return

    with _OREKIT_INIT_LOCK:
        if _OREKIT_READY:
            return

        # On Windows, CPython faulthandler + embedded JVM (JPype/Orekit) can
        # emit spurious fatal access-violation dumps at process teardown.
        # Disable faulthandler before starting the JVM to keep shutdown stable.
        if os.name == "nt":
            try:
                import faulthandler

                if faulthandler.is_enabled():
                    faulthandler.disable()
                    _OREKIT_FAULTHANDLER_DISABLED = True
            except Exception:
                pass

        os.environ.setdefault("JAVA_HOME", str(jdk4py.JAVA_HOME))
        orekit_jpype.initVM()
        from orekit_jpype.pyhelpers import setup_orekit_curdir

        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "data",
                "orekit-data",
            )
        data_path = os.path.abspath(data_path)

        # setup_orekit_curdir clears/replaces providers in the default DataContext.
        setup_orekit_curdir(filename=data_path)
        _OREKIT_DATA_PATH = data_path

        # Bind Java classes only after JVM is guaranteed up.
        from org.orekit.frames import FramesFactory as _FramesFactory  # type: ignore
        from org.orekit.time import (  # type: ignore
            TimeScalesFactory as _TimeScalesFactory,
            AbsoluteDate as _AbsoluteDate,
        )
        from org.orekit.utils import IERSConventions as _IERSConventions  # type: ignore

        FramesFactory.bind(_FramesFactory)
        TimeScalesFactory.bind(_TimeScalesFactory)
        AbsoluteDate.bind(_AbsoluteDate)
        IERSConventions.bind(_IERSConventions)
        _OREKIT_READY = True


def _resolve_iers(iers: Optional[object]) -> object:
    initialize_orekit()
    if iers is None:
        return IERSConventions.IERS_2010  # type: ignore[union-attr]
    return iers


def _resolve_fast_orbit_class():
    global _FastOrbit, _FAST_ORBIT_IMPORT_ERROR
    if _FastOrbit is not None:
        return _FastOrbit
    try:
        mod = import_module("nebula.propagation._fast_orbit_backend")
        _FastOrbit = getattr(mod, "FastOrbit")
        return _FastOrbit
    except Exception as exc:  # pragma: no cover
        _FAST_ORBIT_IMPORT_ERROR = exc
        raise RuntimeError("Fast orbit backend is unavailable") from exc


FrameKind = Literal["native", "itrf"]
ITRFQueryMode = Literal["cached", "transform"]
InterpolationMode = Literal["cubic", "quintic"]
OrbitPropagationMode = Literal["precision", "efficiency"]

# NumericalPropagator default integrator controls (balanced for stability/speed).
DEFAULT_POSITION_TOLERANCE_M = 0.1
DEFAULT_MIN_STEP_S = 0.001
DEFAULT_MAX_STEP_S = 180.0
DEFAULT_INITIAL_STEP_S = 20.0


# -----------------------------------------------------------------------------
# Time conversion helpers
# -----------------------------------------------------------------------------


def _absdate_to_astropy_utc(abs_date: "AbsoluteDate") -> "AstropyTime":  # type: ignore
    if AstropyTime is None:
        raise RuntimeError("astropy is required for Orbit")
    initialize_orekit()

    utc = TimeScalesFactory.getUTC()
    c = abs_date.getComponents(utc)  # DateTimeComponents
    d = c.getDate()
    tm = c.getTime()

    return AstropyTime(
        {
            "year": int(d.getYear()),
            "month": int(d.getMonth()),
            "day": int(d.getDay()),
            "hour": int(tm.getHour()),
            "minute": int(tm.getMinute()),
            "second": float(tm.getSecond()),
        },
        format="ymdhms",
        scale="utc",
    )


def _astropy_to_absdate_utc(t: "AstropyTime") -> "AbsoluteDate":  # type: ignore
    if AstropyTime is None:
        raise RuntimeError("astropy is required for Orbit")
    initialize_orekit()
    if not isinstance(t, AstropyTime):
        raise TypeError("epoch must be an astropy.time.Time")
    if getattr(t, "shape", None) not in ((), None):
        raise TypeError("epoch must be a scalar astropy.time.Time")

    utc = TimeScalesFactory.getUTC()

    # ymdhms supports leap seconds (second can be 60.x)
    c = t.utc.ymdhms  # type: ignore
    return AbsoluteDate(
        int(c.year),
        int(c.month),
        int(c.day),
        int(c.hour),
        int(c.minute),
        float(c.second),
        utc,
    )


def _dt_seconds_from_epoch(
    t: "AstropyTime", epoch_astropy: "AstropyTime"  # type: ignore
) -> Tuple[np.ndarray, bool]:
    if AstropyTime is None:
        raise RuntimeError("astropy is required for Orbit")
    if not isinstance(t, AstropyTime):
        raise TypeError("t must be an astropy.time.Time")

    is_scalar = getattr(t, "shape", None) == ()
    dt = (t.utc - epoch_astropy.utc).to_value("s")
    dt_arr = np.atleast_1d(np.asarray(dt, dtype=np.float64))
    return dt_arr, is_scalar


def _dt_seconds_from_time_like(
    t: Any, epoch_astropy: "AstropyTime"  # type: ignore
) -> Tuple[np.ndarray, bool]:
    """
    Normalize query time to seconds-from-epoch.

    Accepted:
    - scalar float/int (seconds from epoch)
    - numpy float/int ndarray (seconds from epoch)
    - astropy.time.Time (scalar or vector)
    """
    if isinstance(t, (float, int, np.floating, np.integer)) and not isinstance(t, bool):
        return np.asarray([float(t)], dtype=np.float64), True

    if isinstance(t, np.ndarray):
        if t.dtype.kind not in ("f", "i"):
            raise TypeError(
                "numpy array time input must be float/int seconds from epoch"
            )
        dt_arr = np.asarray(t, dtype=np.float64)
        if dt_arr.ndim == 0:
            return np.asarray([float(dt_arr)], dtype=np.float64), True
        return np.atleast_1d(dt_arr), False

    if AstropyTime is not None and isinstance(t, AstropyTime):
        return _dt_seconds_from_epoch(t, epoch_astropy)

    raise TypeError(
        "t must be astropy.time.Time, float seconds, int seconds, or numpy float/int seconds"
    )


def _pv_acceleration_xyz(pv) -> np.ndarray:
    """
    Best-effort acceleration extraction from Orekit PVCoordinates.
    Falls back to zeros if acceleration is unavailable or non-finite.
    """
    try:
        a = np.asarray(pv.getAcceleration().toArray(), dtype=np.float64)
        if a.shape == (3,) and np.all(np.isfinite(a)):
            return a
    except Exception:
        pass
    return np.zeros(3, dtype=np.float64)


# -----------------------------------------------------------------------------
# Hermite interpolation (unchanged)
# -----------------------------------------------------------------------------
def _hermite_pv_uniform_twosided(
    t_query: np.ndarray,
    k_min: int,
    dt: float,
    r_samples: np.ndarray,
    v_samples: np.ndarray,
    a_samples: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n = int(r_samples.shape[0])
    if n < 2:
        raise ValueError("Need at least 2 samples for Hermite interpolation")

    h = float(dt)
    tq = np.asarray(t_query, dtype=np.float64)

    k = np.floor(tq / h).astype(np.int64)
    k = np.clip(k, k_min, k_min + n - 2)
    i = (k - k_min).astype(np.int64)

    t_k = k.astype(np.float64) * h
    u = (tq - t_k) / h

    r0 = r_samples[i]
    r1 = r_samples[i + 1]
    v0 = v_samples[i]
    v1 = v_samples[i + 1]

    u2 = u * u
    u3 = u2 * u

    if a_samples is None:
        h00 = 2.0 * u3 - 3.0 * u2 + 1.0
        h10 = u3 - 2.0 * u2 + u
        h01 = -2.0 * u3 + 3.0 * u2
        h11 = u3 - u2

        r = (
            h00[:, None] * r0
            + h10[:, None] * (h * v0)
            + h01[:, None] * r1
            + h11[:, None] * (h * v1)
        )

        dh00 = 6.0 * u2 - 6.0 * u
        dh10 = 3.0 * u2 - 4.0 * u + 1.0
        dh01 = -6.0 * u2 + 6.0 * u
        dh11 = 3.0 * u2 - 2.0 * u

        v = (
            (dh00[:, None] * r0) / h
            + dh10[:, None] * v0
            + (dh01[:, None] * r1) / h
            + dh11[:, None] * v1
        )
        return r, v

    a0 = a_samples[i]
    a1 = a_samples[i + 1]
    h2 = h * h
    u4 = u2 * u2
    u5 = u4 * u

    # Quintic Hermite basis with endpoint position/velocity/acceleration.
    h00 = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5
    h10 = u - 6.0 * u3 + 8.0 * u4 - 3.0 * u5
    h20 = 0.5 * u2 - 1.5 * u3 + 1.5 * u4 - 0.5 * u5
    h01 = 10.0 * u3 - 15.0 * u4 + 6.0 * u5
    h11 = -4.0 * u3 + 7.0 * u4 - 3.0 * u5
    h21 = 0.5 * u3 - u4 + 0.5 * u5

    r = (
        h00[:, None] * r0
        + h10[:, None] * (h * v0)
        + h20[:, None] * (h2 * a0)
        + h01[:, None] * r1
        + h11[:, None] * (h * v1)
        + h21[:, None] * (h2 * a1)
    )

    dh00 = -30.0 * u2 + 60.0 * u3 - 30.0 * u4
    dh10 = 1.0 - 18.0 * u2 + 32.0 * u3 - 15.0 * u4
    dh20 = u - 4.5 * u2 + 6.0 * u3 - 2.5 * u4
    dh01 = 30.0 * u2 - 60.0 * u3 + 30.0 * u4
    dh11 = -12.0 * u2 + 28.0 * u3 - 15.0 * u4
    dh21 = 1.5 * u2 - 4.0 * u3 + 2.5 * u4

    v = (
        (dh00[:, None] * r0) / h
        + dh10[:, None] * v0
        + dh20[:, None] * (h * a0)
        + (dh01[:, None] * r1) / h
        + dh11[:, None] * v1
        + dh21[:, None] * (h * a1)
    )

    return r, v


# -----------------------------------------------------------------------------
# Propagator construction helpers (NEW)
# -----------------------------------------------------------------------------
AngleType = Literal["true", "mean", "eccentric"]
InertialFrameName = str
PVInputFrameName = str
SolarActivityStrength = Literal["average", "weak", "strong"]
ThirdBodyName = Literal["sun", "moon", "venus", "mars", "jupiter", "saturn"]

_INERTIAL_FRAME_NAMES: tuple[str, ...] = (
    "gcrf",
    "icrf",
    "eme2000",
    "mod",
    "tod",
    "teme",
    "cirf",
    "veis1950",
    "ecliptic",
)

_INERTIAL_FRAME_ALIASES: dict[str, str] = {
    "j2000": "eme2000",
    "meanofdate": "mod",
    "trueofdate": "tod",
    "veis": "veis1950",
    "veis50": "veis1950",
}


def _normalize_frame_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _supported_inertial_frames_text() -> str:
    return ", ".join(f"'{name}'" for name in _INERTIAL_FRAME_NAMES)


def _resolve_inertial_frame(name: InertialFrameName, *, iers: object, simple_eop: bool):
    key = _normalize_frame_name(name)
    canonical = _INERTIAL_FRAME_ALIASES.get(key, key)

    if canonical == "gcrf":
        frame = FramesFactory.getGCRF()
    elif canonical == "icrf":
        frame = FramesFactory.getICRF()
    elif canonical == "eme2000":
        frame = FramesFactory.getEME2000()
    elif canonical == "mod":
        frame = FramesFactory.getMOD(iers)  # type: ignore
    elif canonical == "tod":
        frame = FramesFactory.getTOD(iers, bool(simple_eop))  # type: ignore
    elif canonical == "teme":
        frame = FramesFactory.getTEME()
    elif canonical == "cirf":
        frame = FramesFactory.getCIRF(iers, bool(simple_eop))  # type: ignore
    elif canonical == "veis1950":
        frame = FramesFactory.getVeis1950()
    elif canonical == "ecliptic":
        frame = FramesFactory.getEcliptic(iers)  # type: ignore
    else:
        raise ValueError(
            f"Unsupported inertial frame '{name}'. "
            f"Supported inertial frames: {_supported_inertial_frames_text()}."
        )

    if not bool(frame.isPseudoInertial()):
        raise ValueError(
            f"Resolved frame '{frame.getName()}' from '{name}' is not pseudo-inertial."
        )
    return frame


def _resolve_pv_input_frame(name: PVInputFrameName, *, iers: object, simple_eop: bool):
    key = _normalize_frame_name(name)
    if key in ("itrf", "ecef"):
        return FramesFactory.getITRF(iers, bool(simple_eop))  # type: ignore

    canonical = _INERTIAL_FRAME_ALIASES.get(key, key)
    if canonical not in _INERTIAL_FRAME_NAMES:
        raise ValueError(
            f"Unsupported frame '{name}'. "
            f"Supported frames: {_supported_inertial_frames_text()}, 'itrf'."
        )

    return _resolve_inertial_frame(
        canonical,
        iers=iers,
        simple_eop=bool(simple_eop),
    )


def _resolve_itrf_query_mode(mode: str) -> ITRFQueryMode:
    key = _normalize_frame_name(mode)
    if key == "cached":
        return "cached"
    if key == "transform":
        return "transform"
    raise ValueError("itrf_query_mode must be 'cached' or 'transform'")


def _resolve_interpolation_mode(mode: str) -> InterpolationMode:
    m = str(mode).strip().lower()
    if m in ("cubic", "quintic"):
        return m  # type: ignore[return-value]
    raise ValueError("interpolation_mode must be 'cubic' or 'quintic'")


def _resolve_position_angle_type(angle_type: AngleType):
    from org.orekit.orbits import PositionAngleType  # type: ignore

    if angle_type == "true":
        return PositionAngleType.TRUE
    if angle_type == "mean":
        return PositionAngleType.MEAN
    if angle_type == "eccentric":
        return PositionAngleType.ECCENTRIC
    raise ValueError("anomaly_type must be 'true', 'mean', or 'eccentric'")


def _resolve_solar_activity_strength(level: SolarActivityStrength):
    from org.orekit.models.earth.atmosphere.data import (  # type: ignore
        MarshallSolarActivityFutureEstimation,
    )

    if level == "average":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.AVERAGE
    if level == "weak":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.WEAK
    if level == "strong":
        return MarshallSolarActivityFutureEstimation.StrengthLevel.STRONG
    raise ValueError("solar_activity_strength must be 'average', 'weak', or 'strong'")


def _resolve_third_body(name: ThirdBodyName):
    from org.orekit.bodies import CelestialBodyFactory  # type: ignore

    if name == "sun":
        return CelestialBodyFactory.getSun()
    if name == "moon":
        return CelestialBodyFactory.getMoon()
    if name == "venus":
        return CelestialBodyFactory.getVenus()
    if name == "mars":
        return CelestialBodyFactory.getMars()
    if name == "jupiter":
        return CelestialBodyFactory.getJupiter()
    if name == "saturn":
        return CelestialBodyFactory.getSaturn()
    raise ValueError(f"Unsupported third body: {name}")


def _build_earth_shape(itrf):
    from org.orekit.bodies import OneAxisEllipsoid  # type: ignore
    from org.orekit.utils import Constants  # type: ignore

    return OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        itrf,
    )


def _build_numerical_propagator(
    *,
    initial_orbit,
    initial_state,
    position_tolerance_m: float,
    min_step_s: float,
    max_step_s: float,
    initial_step_s: float,
):
    from org.hipparchus.ode.nonstiff import DormandPrince853Integrator  # type: ignore
    from org.orekit.orbits import OrbitType  # type: ignore
    from org.orekit.propagation.numerical import NumericalPropagator  # type: ignore

    tolerances = NumericalPropagator.tolerances(
        float(position_tolerance_m),
        initial_orbit,
        OrbitType.CARTESIAN,
    )

    integrator = DormandPrince853Integrator(
        float(min_step_s),
        float(max_step_s),
        tolerances[0],
        tolerances[1],
    )
    integrator.setInitialStepSize(float(initial_step_s))

    propagator = NumericalPropagator(integrator)
    # Keep propagation variables consistent with CARTESIAN tolerances above.
    propagator.setOrbitType(OrbitType.CARTESIAN)
    propagator.setInitialState(initial_state)
    return propagator


def _configure_force_models(
    *,
    propagator,
    # frames/scales/constants
    itrf,
    inertial_frame,
    utc,
    iers: object,
    simple_eop: bool,
    mu: float,
    ae: float,
    earth_shape,
    mass_kg: float,
    # gravity
    gravity_model: Literal["newtonian", "harmonic"] = "newtonian",
    gravity_degree: int = 20,
    gravity_order: int = 20,
    # drag
    enable_drag: bool = False,
    drag_area_m2: float = 1.0,
    drag_cd: float = 2.2,
    solar_activity_strength: SolarActivityStrength = "average",
    # third body
    enable_third_body: bool = False,
    third_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
    # tides
    enable_solid_tides: bool = False,
    solid_tides_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
    enable_ocean_tides: bool = False,
    ocean_degree: int = 8,
    ocean_order: int = 8,
    # relativity
    enable_relativity: bool = False,  # Schwarzschild
    enable_de_sitter: bool = False,
    enable_lense_thirring: bool = False,
    # SRP
    enable_srp: bool = False,
    srp_area_m2: float = 1.0,
    srp_cr: float = 1.2,
    srp_occult_moon: bool = True,
    # ERP
    enable_erp: bool = False,
    erp_angular_resolution_deg: float = 1.0,
):
    """
    Attach Orekit force models to `propagator` using a user-facing boolean/parameter interface.

    Notes on gravity choice:
      - gravity_model="newtonian": adds NewtonianAttraction(mu)
      - gravity_model="harmonic": adds HolmesFeatherstoneAttractionModel(ITRF, provider)
        and does NOT add NewtonianAttraction to avoid double counting.
    """
    from org.orekit.forces.gravity import NewtonianAttraction  # type: ignore
    from org.orekit.forces.gravity import (  # type: ignore
        HolmesFeatherstoneAttractionModel,
        ThirdBodyAttraction,
        SolidTides,
        OceanTides,
        Relativity,
        DeSitterRelativity,
        LenseThirringRelativity,
    )
    from org.orekit.forces.gravity.potential import GravityFieldFactory, TideSystem  # type: ignore
    from org.orekit.forces.drag import DragForce, IsotropicDrag  # type: ignore
    from org.orekit.models.earth.atmosphere import NRLMSISE00  # type: ignore
    from org.orekit.models.earth.atmosphere.data import (  # type: ignore
        MarshallSolarActivityFutureEstimation,
    )
    from org.orekit.bodies import CelestialBodyFactory  # type: ignore
    from org.orekit.utils import Constants  # type: ignore
    from org.orekit.time import TimeScalesFactory  # type: ignore
    from org.orekit.forces.radiation import (  # type: ignore
        SolarRadiationPressure,
        IsotropicRadiationSingleCoefficient,
        KnockeRediffusedForceModel,
    )

    # ---- Gravity (choose one)
    if gravity_model == "newtonian":
        propagator.addForceModel(NewtonianAttraction(float(mu)))
    elif gravity_model == "harmonic":
        provider = GravityFieldFactory.getNormalizedProvider(
            int(gravity_degree), int(gravity_order)
        )
        propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, provider))
    else:
        raise ValueError("gravity_model must be 'newtonian' or 'harmonic'")

    # ---- Common bodies
    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()
    earth_body = CelestialBodyFactory.getEarth()

    # ---- Drag
    if enable_drag:
        msafe = MarshallSolarActivityFutureEstimation(
            MarshallSolarActivityFutureEstimation.DEFAULT_SUPPORTED_NAMES,
            _resolve_solar_activity_strength(solar_activity_strength),
        )
        atmosphere = NRLMSISE00(msafe, sun, earth_shape)
        drag_sensitive = IsotropicDrag(float(drag_area_m2), float(drag_cd))
        propagator.addForceModel(DragForce(atmosphere, drag_sensitive))

    # ---- Third-body gravity
    if enable_third_body:
        for name in third_bodies:
            body = _resolve_third_body(name)
            propagator.addForceModel(ThirdBodyAttraction(body))

    # ---- Solid tides / Ocean tides
    if enable_solid_tides or enable_ocean_tides:
        ut1 = TimeScalesFactory.getUT1(iers, bool(simple_eop))  # type: ignore
        tide_system = TideSystem.ZERO_TIDE

        if enable_solid_tides:
            for name in solid_tides_bodies:
                body = _resolve_third_body(name)
                propagator.addForceModel(
                    SolidTides(
                        itrf,
                        float(ae),
                        float(mu),
                        tide_system,
                        iers,  # type: ignore
                        ut1,
                        body,
                    )
                )

        if enable_ocean_tides:
            propagator.addForceModel(
                OceanTides(
                    itrf,
                    float(ae),
                    float(mu),
                    int(ocean_degree),
                    int(ocean_order),
                    iers,  # type: ignore
                    ut1,
                )
            )

    # ---- Relativistic corrections
    if enable_relativity:
        propagator.addForceModel(Relativity(float(mu)))
    if enable_de_sitter:
        propagator.addForceModel(DeSitterRelativity(earth_body, sun))
    if enable_lense_thirring:
        propagator.addForceModel(LenseThirringRelativity(float(mu), itrf))

    # ---- SRP (with optional Moon occultation)
    # Note: Orekit SRP uses the provided Earth shape for eclipse geometry.
    radiation_sensitive = None
    if enable_srp or enable_erp:
        radiation_sensitive = IsotropicRadiationSingleCoefficient(
            float(srp_area_m2), float(srp_cr)
        )

    if enable_srp:
        srp = SolarRadiationPressure(sun, earth_shape, radiation_sensitive)  # type: ignore
        if bool(srp_occult_moon):
            srp.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)
        propagator.addForceModel(srp)

    # ---- Earth radiation pressure (Knocke model)
    if enable_erp:
        if radiation_sensitive is None:
            radiation_sensitive = IsotropicRadiationSingleCoefficient(
                float(srp_area_m2), float(srp_cr)
            )
        ang_res_rad = math.radians(float(erp_angular_resolution_deg))
        propagator.addForceModel(
            KnockeRediffusedForceModel(
                sun,
                radiation_sensitive,
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                ang_res_rad,
                utc,
            )
        )


# -----------------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------------
@dataclass
class Orbit:
    """
    Unified orbital propagation/query interface for precision and fast modes.

    The class exposes a consistent query API (`pv`, `pos`, `vel`, `pv_itrf`,
    `lla`, ...) regardless of backend:

    - ``precision`` mode uses Orekit numerical propagation with configurable
        force models.
    - ``efficiency`` mode uses the fast numpy/numba backend for higher
        throughput with approximate dynamics.

    Preferred constructors
    ----------------------
    - ``from_kepler_precise`` for high-fidelity Orekit propagation.
    - ``from_kepler_fast`` for high-speed approximate propagation.
    - ``from_pv`` when starting from Cartesian position/velocity states.

    Notes
    -----
    Query methods accept either scalar or vector time inputs and preserve that
    shape in outputs.
    """

    propagator: object
    dt_save_s: float = 60.0
    iers: Optional[object] = None
    simple_eop: bool = True
    itrf_query_mode: ITRFQueryMode = "cached"
    interpolation_mode: InterpolationMode = "cubic"

    def __post_init__(self) -> None:
        self._mode: OrbitPropagationMode = "precision"
        self._fast_impl = None
        initialize_orekit()
        if AstropyTime is None:
            raise RuntimeError("astropy is required for Orbit")
        if float(self.dt_save_s) <= 0.0:
            raise ValueError("dt_save_s must be > 0")

        self._dt = float(self.dt_save_s)
        self.iers = _resolve_iers(self.iers)
        self._itrf_query_mode = _resolve_itrf_query_mode(self.itrf_query_mode)
        self._interpolation_mode = _resolve_interpolation_mode(self.interpolation_mode)
        self._use_quintic = self._interpolation_mode == "quintic"

        self._state0 = self.propagator.getInitialState()  # type: ignore
        self._t0_abs = self._state0.getDate()
        self._epoch_ast = _absdate_to_astropy_utc(self._t0_abs)

        self._frame_native = self._state0.getFrame()
        self._itrf = FramesFactory.getITRF(self.iers, bool(self.simple_eop))  # type: ignore

        self._k_min = 0
        self._k_max = 0

        self._first_state = self._state0
        self._last_state = self._state0

        pv0 = self._state0.getPVCoordinates(self._frame_native)
        pvi = self._state0.getPVCoordinates(self._itrf)

        self._r_native = np.array([pv0.getPosition().toArray()], dtype=np.float64)
        self._v_native = np.array([pv0.getVelocity().toArray()], dtype=np.float64)
        if self._use_quintic:
            self._a_native = np.array([_pv_acceleration_xyz(pv0)], dtype=np.float64)
        else:
            self._a_native = np.zeros((1, 3), dtype=np.float64)

        self._r_itrf = np.array([pvi.getPosition().toArray()], dtype=np.float64)
        self._v_itrf = np.array([pvi.getVelocity().toArray()], dtype=np.float64)
        if self._use_quintic:
            self._a_itrf = np.array([_pv_acceleration_xyz(pvi)], dtype=np.float64)
        else:
            self._a_itrf = np.zeros((1, 3), dtype=np.float64)

        self._ephem_generator = None

    @property
    def epoch(self) -> "AstropyTime":  # type: ignore
        """Reference epoch as scalar UTC ``astropy.time.Time``."""
        if self._mode == "efficiency":
            return self._fast_impl.epoch  # type: ignore[return-value]
        return self._epoch_ast

    @property
    def dt(self) -> float:
        """Cache sample interval in seconds."""
        if self._mode == "efficiency":
            return float(self._fast_impl.dt)
        return self._dt

    @property
    def mode(self) -> OrbitPropagationMode:
        """Active propagation mode: ``\"precision\"`` or ``\"efficiency\"``."""
        return self._mode

    @property
    def is_precision(self) -> bool:
        """True when this instance uses the Orekit precision backend."""
        return self._mode == "precision"

    @property
    def is_efficiency(self) -> bool:
        """True when this instance uses the fast numpy/numba backend."""
        return self._mode == "efficiency"

    def coverage(self) -> Tuple[float, float]:
        """Return currently cached time coverage as ``(t_min_s, t_max_s)``."""
        if self._mode == "efficiency":
            return self._fast_impl.coverage()
        return float(self._k_min) * self._dt, float(self._k_max) * self._dt

    def precompute(self, t_min_s: float, t_max_s: float) -> None:
        """
        Expand the internal interpolation cache to cover a target time window.

        Parameters
        ----------
        t_min_s, t_max_s : float
            Window bounds in seconds from ``epoch``.
        """
        if self._mode == "efficiency":
            self._fast_impl.precompute(float(t_min_s), float(t_max_s))
            return
        self._ensure_covered(
            np.asarray([float(t_min_s), float(t_max_s)], dtype=np.float64)
        )

    def _new_ephemeris_generator(self):
        # Prevent unbounded generator growth in repeated cache extensions.
        try:
            self.propagator.clearEphemerisGenerators()  # type: ignore
        except Exception:
            pass
        self._ephem_generator = self.propagator.getEphemerisGenerator()  # type: ignore
        return self._ephem_generator

    def _transform_native_to_itrf(
        self, dt_s: np.ndarray, r_native: np.ndarray, v_native: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
        from org.orekit.utils import PVCoordinates  # type: ignore

        n = int(dt_s.size)
        r_itrf = np.empty((n, 3), dtype=np.float64)
        v_itrf = np.empty((n, 3), dtype=np.float64)

        for j in range(n):
            abs_t = self._t0_abs.shiftedBy(float(dt_s[j]))
            tr = self._frame_native.getTransformTo(self._itrf, abs_t)
            pv_n = PVCoordinates(
                Vector3D(*r_native[j].tolist()),
                Vector3D(*v_native[j].tolist()),
            )
            pv_i = tr.transformPVCoordinates(pv_n)
            r_itrf[j, :] = pv_i.getPosition().toArray()
            v_itrf[j, :] = pv_i.getVelocity().toArray()

        return r_itrf, v_itrf

    def pv(self, t: Any, frame: FrameKind = "native") -> Tuple[np.ndarray, np.ndarray]:
        """
        Query position/velocity at one or more times.

        Parameters
        ----------
        t : astropy.time.Time | float | int | np.ndarray
            Query time(s). Numeric values are seconds from ``epoch``.
        frame : {"native", "itrf"}, default "native"
            Output frame. ``native`` is the propagation frame.

        Returns
        -------
        (r, v) : tuple[np.ndarray, np.ndarray]
            Position and velocity in SI units. Scalar input returns two
            ``(3,)`` arrays; vector input returns ``(N, 3)`` arrays.
        """
        if self._mode == "efficiency":
            return self._fast_impl.pv(t, frame=frame)

        dt_s, is_scalar = _dt_seconds_from_time_like(t, self._epoch_ast)
        # If cache has only the initial sample, allow exact-knot queries without interpolation.
        if (self._k_max - self._k_min) < 1:
            if dt_s.size == 1:
                t0 = float(self._k_min) * self._dt
                if abs(float(dt_s[0]) - t0) <= 1e-9:
                    if frame == "native":
                        return self._r_native[0].copy(), self._v_native[0].copy()
                    if frame == "itrf":
                        if self._itrf_query_mode == "cached":
                            return self._r_itrf[0].copy(), self._v_itrf[0].copy()
                        r_i, v_i = self._transform_native_to_itrf(
                            dt_s,
                            self._r_native[:1],
                            self._v_native[:1],
                        )
                        return r_i[0], v_i[0]
                    raise ValueError("frame must be 'native' or 'itrf'")

        self._ensure_covered(dt_s)

        if frame == "native":
            r, v = _hermite_pv_uniform_twosided(
                dt_s,
                self._k_min,
                self._dt,
                self._r_native,
                self._v_native,
                self._a_native if self._use_quintic else None,
            )
        elif frame == "itrf":
            if self._itrf_query_mode == "cached":
                r, v = _hermite_pv_uniform_twosided(
                    dt_s,
                    self._k_min,
                    self._dt,
                    self._r_itrf,
                    self._v_itrf,
                    self._a_itrf if self._use_quintic else None,
                )
            else:
                r_native, v_native = _hermite_pv_uniform_twosided(
                    dt_s,
                    self._k_min,
                    self._dt,
                    self._r_native,
                    self._v_native,
                    self._a_native if self._use_quintic else None,
                )
                r, v = self._transform_native_to_itrf(dt_s, r_native, v_native)
        else:
            raise ValueError("frame must be 'native' or 'itrf'")

        if is_scalar:
            return r[0], v[0]
        return r, v

    def pos(self, t: Any, frame: FrameKind = "native") -> np.ndarray:
        """Position query helper; equivalent to ``pv(...)[0]``."""
        r, _ = self.pv(t, frame=frame)
        return r

    def vel(self, t: Any, frame: FrameKind = "native") -> np.ndarray:
        """Velocity query helper; equivalent to ``pv(...)[1]``."""
        _, v = self.pv(t, frame=frame)
        return v

    def pv_itrf(self, t: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience wrapper for ``pv(t, frame=\"itrf\")``."""
        return self.pv(t, frame="itrf")

    def pos_itrf(self, t: Any) -> np.ndarray:
        """Convenience wrapper for ``pos(t, frame=\"itrf\")``."""
        return self.pos(t, frame="itrf")

    def vel_itrf(self, t: Any) -> np.ndarray:
        """Convenience wrapper for ``vel(t, frame=\"itrf\")``."""
        return self.vel(t, frame="itrf")

    def lla(self, t: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Query geodetic latitude/longitude/altitude from ITRF position.

        Parameters
        ----------
        t : astropy.time.Time | float | int | np.ndarray
            Query time(s). Numeric values are seconds from ``epoch``.

        Returns
        -------
        (lat_deg, lon_deg, alt_m) : tuple[np.ndarray, np.ndarray, np.ndarray]
            Geodetic latitude/longitude in degrees and altitude in meters.
            Scalar input returns scalars; vector input returns ``(N,)`` arrays.
        """
        if self._mode == "efficiency":
            return self._fast_impl.lla(t)

        r_itrf = self.pos_itrf(t)
        if r_itrf.ndim == 1:
            lat, lon, alt = ecef2geodetic_deg(r_itrf[0], r_itrf[1], r_itrf[2])
        else:
            lat, lon, alt = ecef2geodetic_vec_ecef_deg(r_itrf, wrap_lon=True)
        return lat, lon, alt  # type: ignore

    def _ensure_covered(self, dt_s: np.ndarray) -> None:
        dt_s = np.asarray(dt_s, dtype=np.float64)
        if dt_s.size == 0:
            return
        if np.any(~np.isfinite(dt_s)):
            raise ValueError("Non-finite query times are not supported")

        lo = float(dt_s.min())
        hi = float(dt_s.max())

        k_need_lo = int(math.floor(lo / self._dt))
        k_need_hi = int(math.ceil(hi / self._dt))

        if k_need_lo < self._k_min:
            self._extend_backward_to(k_need_lo)
        if k_need_hi > self._k_max:
            self._extend_forward_to(k_need_hi)

        if (self._k_max - self._k_min) < 1:
            raise RuntimeError("Cache has insufficient samples for interpolation")

    def _extend_forward_to(self, k_target: int) -> None:
        if k_target <= self._k_max:
            return

        dt = self._dt
        t0 = self._t0_abs

        prev_init = self.propagator.getInitialState()  # type: ignore
        try:
            self.propagator.resetInitialState(self._last_state)  # type: ignore

            gen = self._new_ephemeris_generator()
            target_date = t0.shiftedBy(float(k_target) * dt)
            new_last_state = self.propagator.propagate(target_date)  # type: ignore
            ephem = gen.getGeneratedEphemeris()

            ks = np.arange(self._k_max + 1, k_target + 1, dtype=np.int64)
            n_new = int(ks.size)

            rN = np.empty((n_new, 3), dtype=np.float64)
            vN = np.empty((n_new, 3), dtype=np.float64)
            use_quintic = self._use_quintic
            if use_quintic:
                aN = np.empty((n_new, 3), dtype=np.float64)
            cache_itrf = self._itrf_query_mode == "cached"
            if cache_itrf:
                rI = np.empty((n_new, 3), dtype=np.float64)
                vI = np.empty((n_new, 3), dtype=np.float64)
                if use_quintic:
                    aI = np.empty((n_new, 3), dtype=np.float64)

            for j, k in enumerate(ks):
                d = t0.shiftedBy(float(k) * dt)
                st = ephem.propagate(d)

                pv = st.getPVCoordinates(self._frame_native)
                rN[j, :] = pv.getPosition().toArray()
                vN[j, :] = pv.getVelocity().toArray()
                if use_quintic:
                    aN[j, :] = _pv_acceleration_xyz(pv)

                if cache_itrf:
                    pvi = st.getPVCoordinates(self._itrf)
                    rI[j, :] = pvi.getPosition().toArray()
                    vI[j, :] = pvi.getVelocity().toArray()
                    if use_quintic:
                        aI[j, :] = _pv_acceleration_xyz(pvi)

            self._r_native = np.vstack([self._r_native, rN])
            self._v_native = np.vstack([self._v_native, vN])
            if use_quintic:
                self._a_native = np.vstack([self._a_native, aN])
            if cache_itrf:
                self._r_itrf = np.vstack([self._r_itrf, rI])
                self._v_itrf = np.vstack([self._v_itrf, vI])
                if use_quintic:
                    self._a_itrf = np.vstack([self._a_itrf, aI])

            self._k_max = int(k_target)
            self._last_state = new_last_state
        finally:
            self.propagator.resetInitialState(prev_init)  # type: ignore

    def _extend_backward_to(self, k_target: int) -> None:
        if k_target >= self._k_min:
            return

        dt = self._dt
        t0 = self._t0_abs

        prev_init = self.propagator.getInitialState()  # type: ignore
        try:
            self.propagator.resetInitialState(self._first_state)  # type: ignore

            gen = self._new_ephemeris_generator()
            target_date = t0.shiftedBy(float(k_target) * dt)
            new_first_state = self.propagator.propagate(target_date)  # type: ignore
            ephem = gen.getGeneratedEphemeris()

            ks = np.arange(k_target, self._k_min, dtype=np.int64)
            n_new = int(ks.size)

            rN = np.empty((n_new, 3), dtype=np.float64)
            vN = np.empty((n_new, 3), dtype=np.float64)
            use_quintic = self._use_quintic
            if use_quintic:
                aN = np.empty((n_new, 3), dtype=np.float64)
            cache_itrf = self._itrf_query_mode == "cached"
            if cache_itrf:
                rI = np.empty((n_new, 3), dtype=np.float64)
                vI = np.empty((n_new, 3), dtype=np.float64)
                if use_quintic:
                    aI = np.empty((n_new, 3), dtype=np.float64)

            for j, k in enumerate(ks):
                d = t0.shiftedBy(float(k) * dt)
                st = ephem.propagate(d)

                pv = st.getPVCoordinates(self._frame_native)
                rN[j, :] = pv.getPosition().toArray()
                vN[j, :] = pv.getVelocity().toArray()
                if use_quintic:
                    aN[j, :] = _pv_acceleration_xyz(pv)

                if cache_itrf:
                    pvi = st.getPVCoordinates(self._itrf)
                    rI[j, :] = pvi.getPosition().toArray()
                    vI[j, :] = pvi.getVelocity().toArray()
                    if use_quintic:
                        aI[j, :] = _pv_acceleration_xyz(pvi)

            self._r_native = np.vstack([rN, self._r_native])
            self._v_native = np.vstack([vN, self._v_native])
            if use_quintic:
                self._a_native = np.vstack([aN, self._a_native])
            if cache_itrf:
                self._r_itrf = np.vstack([rI, self._r_itrf])
                self._v_itrf = np.vstack([vI, self._v_itrf])
                if use_quintic:
                    self._a_itrf = np.vstack([aI, self._a_itrf])

            self._k_min = int(k_target)
            self._first_state = new_first_state
        finally:
            self.propagator.resetInitialState(prev_init)  # type: ignore

    # -------------------------------------------------------------------------
    # Classmethod constructors (IMPLEMENTED)
    # -------------------------------------------------------------------------
    @classmethod
    def _from_fast_impl(cls, fast_impl) -> "Orbit":
        obj = cls.__new__(cls)
        # Dataclass fields
        obj.propagator = None
        obj.dt_save_s = float(fast_impl.dt)
        obj.iers = None
        obj.simple_eop = True
        obj.itrf_query_mode = "cached"
        obj.interpolation_mode = "cubic"
        # Unified-mode internals
        obj._mode = "efficiency"
        obj._fast_impl = fast_impl
        obj._dt = float(fast_impl.dt)
        obj._epoch_ast = fast_impl.epoch
        return obj

    @classmethod
    def from_kepler_precise(
        cls,
        *,
        epoch: "AstropyTime",  # type: ignore
        a_m: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type: AngleType = "true",
        degrees: bool = False,
        inertial_frame: InertialFrameName = "gcrf",
        mass_kg: float = 1000.0,
        mu: Optional[float] = None,
        # cache
        dt_save_s: float = 60.0,
        # frames/EOP used for ITRF + force models
        iers: Optional[object] = None,
        simple_eop: bool = True,
        itrf_query_mode: ITRFQueryMode = "cached",
        interpolation_mode: InterpolationMode = "cubic",
        # integrator
        position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M,
        min_step_s: float = DEFAULT_MIN_STEP_S,
        max_step_s: float = DEFAULT_MAX_STEP_S,
        initial_step_s: float = DEFAULT_INITIAL_STEP_S,
        # force model selection (expose all you used)
        gravity_model: Literal["newtonian", "harmonic"] = "newtonian",
        gravity_degree: int = 20,
        gravity_order: int = 20,
        enable_drag: bool = False,
        drag_area_m2: float = 1.0,
        drag_cd: float = 2.2,
        solar_activity_strength: SolarActivityStrength = "average",
        enable_third_body: bool = False,
        third_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
        enable_solid_tides: bool = False,
        solid_tides_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
        enable_ocean_tides: bool = False,
        ocean_degree: int = 8,
        ocean_order: int = 8,
        enable_relativity: bool = False,
        enable_de_sitter: bool = False,
        enable_lense_thirring: bool = False,
        enable_srp: bool = False,
        srp_area_m2: float = 1.0,
        srp_cr: float = 1.2,
        srp_occult_moon: bool = True,
        enable_erp: bool = False,
        erp_angular_resolution_deg: float = 1.0,
    ) -> "Orbit":
        """
        Build a precision-mode ``Orbit`` from classical Keplerian elements.

        Parameters
        ----------
        epoch : astropy.time.Time
                Epoch of the element set.
        a_m, e, i, raan, argp, anomaly : float
                Standard Keplerian elements. Angles are radians unless
                ``degrees=True``.
        anomaly_type : {"true", "mean", "eccentric"}, default "true"
                Interpretation of ``anomaly``.
        inertial_frame : str, default "gcrf"
                Pseudo-inertial frame for the input elements and propagation state.
        dt_save_s : float, default 60
                Cache sampling interval used by interpolation-backed query methods.

        Returns
        -------
        Orbit
                Precision-mode wrapper around an Orekit ``NumericalPropagator``.

        Notes
        -----
        - ``itrf_query_mode=\"cached\"`` is faster for repeated Earth-fixed queries.
            ``\"transform\"`` computes ITRF from native-frame interpolation at query
            time for stronger frame consistency.
        - ``interpolation_mode=\"cubic\"`` is the default speed/accuracy tradeoff;
            ``\"quintic\"`` uses acceleration-aware interpolation.
        - Force model flags expose common Orekit force options without requiring
            direct Orekit setup in user code.
        """
        if AstropyTime is None:
            raise RuntimeError("astropy is required for Orbit")
        initialize_orekit()
        iers = _resolve_iers(iers)

        from org.orekit.orbits import KeplerianOrbit  # type: ignore
        from org.orekit.propagation import SpacecraftState  # type: ignore
        from org.orekit.utils import Constants  # type: ignore

        if degrees:
            i = math.radians(float(i))
            raan = math.radians(float(raan))
            argp = math.radians(float(argp))
            anomaly = math.radians(float(anomaly))

        date0 = _astropy_to_absdate_utc(epoch)
        utc = TimeScalesFactory.getUTC()

        inertial = _resolve_inertial_frame(
            inertial_frame,
            iers=iers,
            simple_eop=bool(simple_eop),
        )
        itrf = FramesFactory.getITRF(iers, bool(simple_eop))  # type: ignore

        mu0 = float(Constants.WGS84_EARTH_MU if mu is None else mu)
        ae = float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

        earth_shape = _build_earth_shape(itrf)

        pos_angle = _resolve_position_angle_type(anomaly_type)

        orbit0 = KeplerianOrbit(
            float(a_m),
            float(e),
            float(i),
            float(argp),
            float(raan),
            float(anomaly),
            pos_angle,
            inertial,
            date0,
            mu0,
        )
        state0 = SpacecraftState(orbit0, float(mass_kg))

        propagator = _build_numerical_propagator(
            initial_orbit=orbit0,
            initial_state=state0,
            position_tolerance_m=float(position_tolerance_m),
            min_step_s=float(min_step_s),
            max_step_s=float(max_step_s),
            initial_step_s=float(initial_step_s),
        )

        _configure_force_models(
            propagator=propagator,
            itrf=itrf,
            inertial_frame=inertial,
            utc=utc,
            iers=iers,
            simple_eop=bool(simple_eop),
            mu=mu0,
            ae=ae,
            earth_shape=earth_shape,
            mass_kg=float(mass_kg),
            gravity_model=gravity_model,
            gravity_degree=int(gravity_degree),
            gravity_order=int(gravity_order),
            enable_drag=bool(enable_drag),
            drag_area_m2=float(drag_area_m2),
            drag_cd=float(drag_cd),
            solar_activity_strength=solar_activity_strength,
            enable_third_body=bool(enable_third_body),
            third_bodies=tuple(third_bodies),
            enable_solid_tides=bool(enable_solid_tides),
            solid_tides_bodies=tuple(solid_tides_bodies),
            enable_ocean_tides=bool(enable_ocean_tides),
            ocean_degree=int(ocean_degree),
            ocean_order=int(ocean_order),
            enable_relativity=bool(enable_relativity),
            enable_de_sitter=bool(enable_de_sitter),
            enable_lense_thirring=bool(enable_lense_thirring),
            enable_srp=bool(enable_srp),
            srp_area_m2=float(srp_area_m2),
            srp_cr=float(srp_cr),
            srp_occult_moon=bool(srp_occult_moon),
            enable_erp=bool(enable_erp),
            erp_angular_resolution_deg=float(erp_angular_resolution_deg),
        )

        return cls(
            propagator,
            dt_save_s=float(dt_save_s),
            iers=iers,
            simple_eop=bool(simple_eop),
            itrf_query_mode=itrf_query_mode,
            interpolation_mode=interpolation_mode,
        )

    @classmethod
    def from_kepler_fast(
        cls,
        *,
        epoch: "AstropyTime",  # type: ignore
        a_m: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type: Literal["true", "mean", "eccentric"] = "true",
        degrees: bool = False,
        mu: float = 3.986004418e14,
        dt_save_s: float = 60.0,
        enable_j2: bool = False,
        j2_mode: Literal["secular", "osculating"] = "secular",
        j2_substeps: int = 1,
        J2: float = 1.08262668e-3,
        Re_m: float = 6378137.0,
        use_polar_motion: bool = True,
    ) -> "Orbit":
        """
        Build an efficiency-mode ``Orbit`` backed by the fast numpy/numba engine.

        The returned object preserves the same public query interface as
        precision mode (`pv`, `pos`, `vel`, `pv_itrf`, `lla`, ...).

        Notes
        -----
        This path is optimized for throughput and may trade physical fidelity
        relative to ``from_kepler_precise`` depending on configuration.
        """
        fast_orbit_cls = _resolve_fast_orbit_class()
        fast_impl = fast_orbit_cls.from_kepler(
            epoch=epoch,
            a_m=float(a_m),
            e=float(e),
            i=float(i),
            raan=float(raan),
            argp=float(argp),
            anomaly=float(anomaly),
            anomaly_type=anomaly_type,
            degrees=bool(degrees),
            mu=float(mu),
            dt_save_s=float(dt_save_s),
            enable_j2=bool(enable_j2),
            j2_mode=j2_mode,
            j2_substeps=int(j2_substeps),
            J2=float(J2),
            Re_m=float(Re_m),
            use_polar_motion=bool(use_polar_motion),
        )
        return cls._from_fast_impl(fast_impl)

    @classmethod
    def from_spacecraft_state(
        cls,
        state,
        *,
        dt_save_s: float = 60.0,
        # frames/EOP used for ITRF in the cache + force models that need it
        iers: Optional[object] = None,
        simple_eop: bool = True,
        itrf_query_mode: ITRFQueryMode = "cached",
        interpolation_mode: InterpolationMode = "cubic",
        # integrator
        position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M,
        min_step_s: float = DEFAULT_MIN_STEP_S,
        max_step_s: float = DEFAULT_MAX_STEP_S,
        initial_step_s: float = DEFAULT_INITIAL_STEP_S,
        # override mass / mu (optional)
        mass_kg: Optional[float] = None,
        mu: Optional[float] = None,
        # force models
        gravity_model: Literal["newtonian", "harmonic"] = "newtonian",
        gravity_degree: int = 20,
        gravity_order: int = 20,
        enable_drag: bool = False,
        drag_area_m2: float = 1.0,
        drag_cd: float = 2.2,
        solar_activity_strength: SolarActivityStrength = "average",
        enable_third_body: bool = False,
        third_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
        enable_solid_tides: bool = False,
        solid_tides_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
        enable_ocean_tides: bool = False,
        ocean_degree: int = 8,
        ocean_order: int = 8,
        enable_relativity: bool = False,
        enable_de_sitter: bool = False,
        enable_lense_thirring: bool = False,
        enable_srp: bool = False,
        srp_area_m2: float = 1.0,
        srp_cr: float = 1.2,
        srp_occult_moon: bool = True,
        enable_erp: bool = False,
        erp_angular_resolution_deg: float = 1.0,
    ) -> "Orbit":
        """
        Build a precision-mode ``Orbit`` from an existing Orekit ``SpacecraftState``.

        This constructor is intended for advanced Orekit-integrated workflows.
        If you do not already have Orekit state objects, prefer
        ``from_kepler_precise`` or ``from_pv``.
        """
        initialize_orekit()
        iers = _resolve_iers(iers)
        from org.orekit.utils import Constants  # type: ignore
        from org.orekit.propagation import SpacecraftState  # type: ignore

        orbit0 = state.getOrbit()
        frame0 = orbit0.getFrame()

        mu0 = float(mu if mu is not None else orbit0.getMu())
        ae = float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

        itrf = FramesFactory.getITRF(iers, bool(simple_eop))  # type: ignore
        utc = TimeScalesFactory.getUTC()
        earth_shape = _build_earth_shape(itrf)

        # Mass handling: preserve attitude if possible when overriding mass.
        if mass_kg is None:
            state0 = state
            mass_used = float(state.getMass())
        else:
            try:
                att = state.getAttitude()
                state0 = SpacecraftState(orbit0, att, float(mass_kg))
            except Exception:
                state0 = SpacecraftState(orbit0, float(mass_kg))
            mass_used = float(mass_kg)

        propagator = _build_numerical_propagator(
            initial_orbit=orbit0,
            initial_state=state0,
            position_tolerance_m=float(position_tolerance_m),
            min_step_s=float(min_step_s),
            max_step_s=float(max_step_s),
            initial_step_s=float(initial_step_s),
        )

        _configure_force_models(
            propagator=propagator,
            itrf=itrf,
            inertial_frame=frame0,  # may be inertial; force models use their own frames anyway
            utc=utc,
            iers=iers,
            simple_eop=bool(simple_eop),
            mu=mu0,
            ae=ae,
            earth_shape=earth_shape,
            mass_kg=mass_used,
            gravity_model=gravity_model,
            gravity_degree=int(gravity_degree),
            gravity_order=int(gravity_order),
            enable_drag=bool(enable_drag),
            drag_area_m2=float(drag_area_m2),
            drag_cd=float(drag_cd),
            solar_activity_strength=solar_activity_strength,
            enable_third_body=bool(enable_third_body),
            third_bodies=tuple(third_bodies),
            enable_solid_tides=bool(enable_solid_tides),
            solid_tides_bodies=tuple(solid_tides_bodies),
            enable_ocean_tides=bool(enable_ocean_tides),
            ocean_degree=int(ocean_degree),
            ocean_order=int(ocean_order),
            enable_relativity=bool(enable_relativity),
            enable_de_sitter=bool(enable_de_sitter),
            enable_lense_thirring=bool(enable_lense_thirring),
            enable_srp=bool(enable_srp),
            srp_area_m2=float(srp_area_m2),
            srp_cr=float(srp_cr),
            srp_occult_moon=bool(srp_occult_moon),
            enable_erp=bool(enable_erp),
            erp_angular_resolution_deg=float(erp_angular_resolution_deg),
        )

        return cls(
            propagator,
            dt_save_s=float(dt_save_s),
            iers=iers,
            simple_eop=bool(simple_eop),
            itrf_query_mode=itrf_query_mode,
            interpolation_mode=interpolation_mode,
        )

    @classmethod
    def from_pv(
        cls,
        r: np.ndarray,
        v: np.ndarray,
        epoch: "AstropyTime",  # type: ignore
        *,
        frame: PVInputFrameName = "gcrf",
        propagate_inertial_frame: InertialFrameName = "gcrf",
        mass_kg: float = 1000.0,
        mu: Optional[float] = None,
        dt_save_s: float = 60.0,
        iers: Optional[object] = None,
        simple_eop: bool = True,
        itrf_query_mode: ITRFQueryMode = "cached",
        interpolation_mode: InterpolationMode = "cubic",
        # integrator
        position_tolerance_m: float = DEFAULT_POSITION_TOLERANCE_M,
        min_step_s: float = DEFAULT_MIN_STEP_S,
        max_step_s: float = DEFAULT_MAX_STEP_S,
        initial_step_s: float = DEFAULT_INITIAL_STEP_S,
        # force models (same surface)
        gravity_model: Literal["newtonian", "harmonic"] = "newtonian",
        gravity_degree: int = 20,
        gravity_order: int = 20,
        enable_drag: bool = False,
        drag_area_m2: float = 1.0,
        drag_cd: float = 2.2,
        solar_activity_strength: SolarActivityStrength = "average",
        enable_third_body: bool = False,
        third_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
        enable_solid_tides: bool = False,
        solid_tides_bodies: Sequence[ThirdBodyName] = ("sun", "moon"),
        enable_ocean_tides: bool = False,
        ocean_degree: int = 8,
        ocean_order: int = 8,
        enable_relativity: bool = False,
        enable_de_sitter: bool = False,
        enable_lense_thirring: bool = False,
        enable_srp: bool = False,
        srp_area_m2: float = 1.0,
        srp_cr: float = 1.2,
        srp_occult_moon: bool = True,
        enable_erp: bool = False,
        erp_angular_resolution_deg: float = 1.0,
    ) -> "Orbit":
        """
        Build a precision-mode ``Orbit`` from Cartesian position/velocity state.

        Parameters
        ----------
        r, v : np.ndarray
            Position and velocity vectors of shape ``(3,)`` in meters and
            meters/second.
        epoch : astropy.time.Time
            Epoch of the state vector.
        frame : str, default "gcrf"
            Frame of input vectors. Supports both pseudo-inertial frame names
            and Earth-fixed aliases (``itrf``/``ecef``).
        propagate_inertial_frame : str, default "gcrf"
            Inertial frame used internally by the numerical propagator.

        Notes
        -----
        Input PV is transformed to ``propagate_inertial_frame`` at ``epoch``
        before propagation; if frames already match this is a no-op.
        """
        if AstropyTime is None:
            raise RuntimeError("astropy is required for Orbit")
        initialize_orekit()
        iers = _resolve_iers(iers)

        from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
        from org.orekit.utils import PVCoordinates, Constants  # type: ignore
        from org.orekit.orbits import CartesianOrbit  # type: ignore
        from org.orekit.propagation import SpacecraftState  # type: ignore

        r = np.asarray(r, dtype=np.float64).reshape(3)
        v = np.asarray(v, dtype=np.float64).reshape(3)

        date0 = _astropy_to_absdate_utc(epoch)

        mu0 = float(Constants.WGS84_EARTH_MU if mu is None else mu)
        ae = float(Constants.WGS84_EARTH_EQUATORIAL_RADIUS)

        itrf = FramesFactory.getITRF(iers, bool(simple_eop))  # type: ignore
        utc = TimeScalesFactory.getUTC()
        earth_shape = _build_earth_shape(itrf)

        in_frame = _resolve_pv_input_frame(
            frame, iers=iers, simple_eop=bool(simple_eop)
        )
        inertial = _resolve_inertial_frame(
            propagate_inertial_frame,
            iers=iers,
            simple_eop=bool(simple_eop),
        )

        pv_in = PVCoordinates(Vector3D(*r.tolist()), Vector3D(*v.tolist()))

        tr = in_frame.getTransformTo(inertial, date0)
        pv0 = tr.transformPVCoordinates(pv_in)
        frame0 = inertial

        orbit0 = CartesianOrbit(pv0, frame0, date0, mu0)
        state0 = SpacecraftState(orbit0, float(mass_kg))

        propagator = _build_numerical_propagator(
            initial_orbit=orbit0,
            initial_state=state0,
            position_tolerance_m=float(position_tolerance_m),
            min_step_s=float(min_step_s),
            max_step_s=float(max_step_s),
            initial_step_s=float(initial_step_s),
        )

        _configure_force_models(
            propagator=propagator,
            itrf=itrf,
            inertial_frame=inertial,
            utc=utc,
            iers=iers,
            simple_eop=bool(simple_eop),
            mu=mu0,
            ae=ae,
            earth_shape=earth_shape,
            mass_kg=float(mass_kg),
            gravity_model=gravity_model,
            gravity_degree=int(gravity_degree),
            gravity_order=int(gravity_order),
            enable_drag=bool(enable_drag),
            drag_area_m2=float(drag_area_m2),
            drag_cd=float(drag_cd),
            solar_activity_strength=solar_activity_strength,
            enable_third_body=bool(enable_third_body),
            third_bodies=tuple(third_bodies),
            enable_solid_tides=bool(enable_solid_tides),
            solid_tides_bodies=tuple(solid_tides_bodies),
            enable_ocean_tides=bool(enable_ocean_tides),
            ocean_degree=int(ocean_degree),
            ocean_order=int(ocean_order),
            enable_relativity=bool(enable_relativity),
            enable_de_sitter=bool(enable_de_sitter),
            enable_lense_thirring=bool(enable_lense_thirring),
            enable_srp=bool(enable_srp),
            srp_area_m2=float(srp_area_m2),
            srp_cr=float(srp_cr),
            srp_occult_moon=bool(srp_occult_moon),
            enable_erp=bool(enable_erp),
            erp_angular_resolution_deg=float(erp_angular_resolution_deg),
        )

        return cls(
            propagator,
            dt_save_s=float(dt_save_s),
            iers=iers,
            simple_eop=bool(simple_eop),
            itrf_query_mode=itrf_query_mode,
            interpolation_mode=interpolation_mode,
        )
