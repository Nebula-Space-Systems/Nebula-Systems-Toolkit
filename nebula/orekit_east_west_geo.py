"""
Geosynchronous / Geostationary Stationkeeping (Orekit + orekit-jpype, version-tolerant)

Implements operationally-relevant, practical stationkeeping for GEO/GSO:
- East/West (E/W): controls longitude deadband by commanding a small constant drift rate
  (deadband drift strategy; "box traversal" with hysteresis), achieved via paired tangential
  impulsive burns separated by ~half a day / half an orbit (ASC/DES node pair when available).
- North/South (N/S): controls inclination about a target using cross-track impulses at
  ascending nodes (optionally also at descending if you enable it).

Key design choices vs your prior versions
-----------------------------------------
1) E/W control uses MEAN MOTION relative to Earth rotation, not d/dt of ground longitude:
     drift_now ≈ (n - ω_E) [rad/s]  ->  deg/day for plots/logs
   This is stable for both inclined GSO and near-equatorial GEO.

2) NodeDetector is used when inclination is not ~0 (best for realism).
   For near-zero inclination (geostationary-ish), a fallback DateDetector provides
   regular maneuver opportunities every 12 hours (keeps the script working).

3) Burn minimization knobs:
   - ew_cooldown_days, ns_cooldown_days
   - drift_tolerance_deg_per_day (bigger => fewer E/W burns)
   - min_pair_dv_mps, min_ns_dv_mps (bigger => ignore tiny burns)
   - drift_mag_deg_per_day (bigger => fewer flips but larger corrections)

Realism notes
-------------
- Drag is NOT included (invalid/negligible at GEO/GSO).
- Full eccentricity-vector control is not modeled; the paired tangential burns reduce
  induced eccentricity but this is not a full "longitude at epoch" controller.
- N/S uses cross-track (W) impulses at ascending node to regulate inclination magnitude.
  This is a common operational simplification.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import radians
from typing import List, Optional, Tuple

import jdk4py
import orekit_jpype
import orekitdata

# -----------------------------------------------------------------------------
# 1) Setup JVM and Orekit data
# -----------------------------------------------------------------------------
os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
orekit_jpype.initVM()

from orekit_jpype.pyhelpers import setup_orekit_curdir

setup_orekit_curdir(filename=orekitdata.__path__[0])

from jpype import JClass, JProxy
from org.hipparchus.geometry.euclidean.threed import Vector3D  # type: ignore
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator  # type: ignore

# Bodies / shapes
from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid  # type: ignore

# Gravity
from org.orekit.forces.gravity import (  # type: ignore
    DeSitterRelativity,
    HolmesFeatherstoneAttractionModel,
    LenseThirringRelativity,
    OceanTides,
    Relativity,
    SolidTides,
    ThirdBodyAttraction,
)
from org.orekit.forces.gravity.potential import (  # type: ignore
    GravityFieldFactory,
    TideSystem,
)

# Radiation
from org.orekit.forces.radiation import (  # type: ignore
    IsotropicRadiationSingleCoefficient,
    KnockeRediffusedForceModel,
    SolarRadiationPressure,
)

# Orekit core
from org.orekit.frames import FramesFactory  # type: ignore
from org.orekit.orbits import (  # type: ignore
    CartesianOrbit,
    KeplerianOrbit,
    OrbitType,
    PositionAngleType,
)
from org.orekit.propagation import SpacecraftState  # type: ignore
from org.orekit.propagation.events import NodeDetector  # type: ignore
from org.orekit.propagation.events.handlers import EventHandler  # type: ignore
from org.orekit.propagation.numerical import NumericalPropagator  # type: ignore
from org.orekit.propagation.sampling import OrekitFixedStepHandler  # type: ignore
from org.orekit.time import AbsoluteDate, TimeScalesFactory  # type: ignore
from org.orekit.utils import Constants, IERSConventions, PVCoordinates  # type: ignore


# -----------------------------------------------------------------------------
# Compatibility helpers
# -----------------------------------------------------------------------------
def _get_action_enum():
    """
    Return an enum/class with CONTINUE and RESET_STATE members.

    Orekit versions differ:
      - Some use org.orekit...EventHandler$Action (nested enum)
      - Some use org.hipparchus.ode.events.Action
    """
    try:
        return JClass("org.orekit.propagation.events.handlers.EventHandler$Action")
    except Exception:
        pass
    try:
        return JClass("org.hipparchus.ode.events.Action")
    except Exception as e:
        raise RuntimeError(
            "Could not locate an Action enum class. Tried EventHandler$Action and hipparchus Action."
        ) from e


ACTION = _get_action_enum()


def _set_fixed_step_handler(propagator, step_s: float, handler_proxy):
    """
    Set a fixed-step handler across Orekit versions.
    """
    if hasattr(propagator, "setStepHandler"):
        propagator.setStepHandler(float(step_s), handler_proxy)
        return
    if hasattr(propagator, "setMasterMode"):
        propagator.setMasterMode(float(step_s), handler_proxy)
        return
    raise AttributeError(
        "This NumericalPropagator has neither setStepHandler nor setMasterMode."
    )


def _attach_handler_to_detector(det, handler_proxy):
    """
    Attach an EventHandler proxy to a detector across versions.
    """
    if hasattr(det, "withHandler"):
        return det.withHandler(handler_proxy)
    if hasattr(det, "setHandler"):
        det.setHandler(handler_proxy)
        return det
    raise AttributeError("Detector does not support withHandler or setHandler.")


# -----------------------------------------------------------------------------
# Angle helpers (radians)
# -----------------------------------------------------------------------------
def wrap_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    import math

    return (x + math.pi) % (2.0 * math.pi) - math.pi


def unwrap_relative(new_wrapped: float, ref_unwrapped: float) -> float:
    """
    Unwrap `new_wrapped` (in (-pi, pi]) to be continuous with `ref_unwrapped`.
    """
    ref_wrapped = wrap_pi(ref_unwrapped)
    delta = wrap_pi(new_wrapped - ref_wrapped)
    return ref_unwrapped + delta


def _earth_omega() -> float:
    """
    Earth rotation rate (rad/s). Prefer Orekit constant, fall back if needed.
    """
    try:
        return float(Constants.WGS84_EARTH_ANGULAR_VELOCITY)
    except Exception:
        # IERS conventional value ~ 7.292115e-5 rad/s
        return 7.292115e-5


# -----------------------------------------------------------------------------
# Logs
# -----------------------------------------------------------------------------
@dataclass
class BurnLog:
    date: AbsoluteDate
    opp: str  # "ASC", "DES", or "OPP"
    ew_mode: str  # "EAST" or "WEST"
    lon_mean_deg: float
    lon_err_deg: float
    drift_now_deg_per_day: float
    drift_cmd_deg_per_day: float
    dv_t_pair_mps: float
    dv_t_this_mps: float
    dv_w_mps: float
    inc_deg: float
    inc_err_deg: float


@dataclass
class SampleLog:
    date: AbsoluteDate
    t_days: float
    lon_deg: float
    lat_deg: float
    alt_m: float
    lon_mean_deg: float
    lon_err_deg: float
    drift_now_deg_per_day: float
    drift_cmd_deg_per_day: float
    ew_mode: str
    inc_deg: float
    inc_err_deg: float


# -----------------------------------------------------------------------------
# Combined Stationkeeping Handler (E/W + N/S)
# -----------------------------------------------------------------------------
class GeoSynStationKeepingHandler:
    """
    Stationkeeping event handler called at maneuver opportunities.

    E/W:
      - maintain drift_cmd = +/- drift_mag (deg/day) inside lon deadband
      - flip sign when mean lon error crosses +/- lon_switch (hysteresis)
      - achieve drift via paired tangential impulses separated by one opportunity

    N/S:
      - regulate inclination about inc_target with a deadband
      - apply cross-track (W) impulses at ASC opportunities (optionally DES too)

    Burns are applied via RESET_STATE and resetState().
    """

    def __init__(
        self,
        *,
        earth: OneAxisEllipsoid,
        inertial_frame,
        mu: float,
        # longitude control
        lon_target_deg: float,
        lon_deadband_deg: float = 2.0,
        lon_switch_deg: float = 1.6,
        drift_mag_deg_per_day: float = 0.02,
        drift_tolerance_deg_per_day: float = 0.01,
        tau_lon_mean_days: float = 10.0,
        ew_cooldown_days: float = 7.0,
        # dv limits (E/W)
        max_pair_dv_mps: float = 1.0,
        min_pair_dv_mps: float = 0.05,
        # optional very slow bias trim on drift_cmd
        enable_trim: bool = False,
        trim_tau_days: float = 300.0,
        # inclination control
        inc_target_deg: float = 0.0,
        inc_deadband_deg: float = 0.05,
        ns_cooldown_days: float = 14.0,
        max_ns_dv_mps: float = 1.0,
        min_ns_dv_mps: float = 0.02,
        ns_apply_on_des: bool = False,  # usually False (burn at one node only)
    ):
        import math

        self.earth = earth
        self.inertial_frame = inertial_frame
        self.mu = float(mu)

        # Lon targets
        self.lon_target = math.radians(float(lon_target_deg))
        self.lon_deadband = math.radians(float(lon_deadband_deg))
        self.lon_switch = math.radians(float(lon_switch_deg))
        if self.lon_switch > self.lon_deadband:
            raise ValueError("lon_switch_deg must be <= lon_deadband_deg")

        # Mean longitude smoothing
        self.tau_lon_mean = float(tau_lon_mean_days) * 86400.0

        # Drift strategy
        self.drift_mag = math.radians(float(drift_mag_deg_per_day)) / 86400.0  # rad/s
        self.drift_tol = (
            math.radians(float(drift_tolerance_deg_per_day)) / 86400.0
        )  # rad/s
        self._omega_e = _earth_omega()

        # Burn limits
        self.max_pair_dv = float(max_pair_dv_mps)
        self.min_pair_dv = float(min_pair_dv_mps)

        # Cooldowns
        self.ew_cooldown = float(ew_cooldown_days) * 86400.0

        # Trim
        self.enable_trim = bool(enable_trim)
        self.trim_tau = float(trim_tau_days) * 86400.0

        # Inclination control
        self.inc_target = math.radians(float(inc_target_deg))
        self.inc_deadband = math.radians(float(inc_deadband_deg))
        self.ns_cooldown = float(ns_cooldown_days) * 86400.0
        self.max_ns_dv = float(max_ns_dv_mps)
        self.min_ns_dv = float(min_ns_dv_mps)
        self.ns_apply_on_des = bool(ns_apply_on_des)

        # E/W mode state (sawtooth traversal)
        self._ew_mode = "EAST"  # or "WEST"

        # Mean longitude estimator state
        self._have_lon_mean = False
        self._t_last_lon: Optional[AbsoluteDate] = None
        self._lon_mean_unwrapped: float = 0.0

        # Pending E/W paired burn state (tangential half to apply on next opportunity)
        self._pending_tangential = False
        self._dv_t_half_pending = 0.0

        # Last times we initiated maneuvers
        self._t_last_ew_pair_start: Optional[AbsoluteDate] = None
        self._t_last_ns_burn: Optional[AbsoluteDate] = None

        # What to apply at resetState() for this event
        self._dv_t_to_apply_now = 0.0
        self._dv_w_to_apply_now = 0.0

        # Fallback parity for non-node opportunities
        self._opp_parity = 0  # toggles 0/1 each event if not NodeDetector

        self.burns: List[BurnLog] = []

    # ---- EventHandler interface (signatures differ across versions) ----
    def init(self, *args):
        return

    def finish(self, *args):
        return

    # ---- core ----
    def eventOccurred(self, s, detector, increasing):
        """
        Decide whether to apply maneuver. If yes: return RESET_STATE.
        """
        date = s.getDate()
        is_node, opp = self._classify_opportunity(detector, increasing)

        # Update mean longitude estimate
        lon_wrapped, _lat, _alt = self._subsat_lla(s)
        self._update_lon_mean(date, lon_wrapped)

        lon_err = wrap_pi(self._lon_mean_unwrapped - self.lon_target)

        # Determine current mean motion / drift
        a = float(s.getOrbit().getA())
        n_now = (self.mu / (a * a * a)) ** 0.5  # rad/s
        drift_now = n_now - self._omega_e  # rad/s (Earth-fixed longitude drift approx)

        # --- E/W commanded drift: deadband drift strategy ---
        if self._ew_mode == "EAST" and lon_err > +self.lon_switch:
            self._ew_mode = "WEST"
        elif self._ew_mode == "WEST" and lon_err < -self.lon_switch:
            self._ew_mode = "EAST"

        drift_cmd = (+self.drift_mag) if self._ew_mode == "EAST" else (-self.drift_mag)

        # Optional slow trim (very weak)
        if self.enable_trim and self.trim_tau > 0.0:
            drift_cmd += -lon_err / self.trim_tau

        # Safety: if outside deadband, force drift back inward
        if lon_err > self.lon_deadband:
            drift_cmd = -abs(self.drift_mag)
            self._ew_mode = "WEST"
        elif lon_err < -self.lon_deadband:
            drift_cmd = +abs(self.drift_mag)
            self._ew_mode = "EAST"

        # --- N/S inclination regulation ---
        inc_now = float(s.getOrbit().getI())  # rad
        inc_err = inc_now - self.inc_target

        # Decide burn components for THIS opportunity
        dv_t_now = 0.0
        dv_w_now = 0.0
        dv_t_pair_total = 0.0

        # 1) If we have a pending tangential half, apply it now (always)
        if self._pending_tangential:
            dv_t_now = float(self._dv_t_half_pending)
            dv_t_pair_total = 2.0 * dv_t_now
            self._pending_tangential = False
            self._dv_t_half_pending = 0.0

        # 2) Possibly start a new E/W pair (prefer ASC only if node, else every-other opp)
        if dv_t_now == 0.0:  # don't start a new pair on the same event we finish one
            if self._ew_start_allowed(date, is_node=is_node, opp=opp):
                delta_n = (self._omega_e + drift_cmd) - n_now  # rad/s
                dv_pair = (
                    -(a / 3.0) * delta_n
                )  # m/s total across pair (small-change approx)

                if abs(dv_pair) >= self.min_pair_dv:
                    if abs(dv_pair) > self.max_pair_dv:
                        dv_pair = self.max_pair_dv if dv_pair > 0 else -self.max_pair_dv

                    dv_half = 0.5 * dv_pair
                    dv_t_now = dv_half
                    dv_t_pair_total = dv_pair

                    # queue second half for next opportunity
                    self._pending_tangential = True
                    self._dv_t_half_pending = dv_half

                    self._t_last_ew_pair_start = date

        # 3) Possibly apply N/S burn (usually at ASC node only)
        if self._ns_allowed(date, is_node=is_node, opp=opp, inc_err=inc_err):
            # Cross-track impulse at node changes inclination:
            #   Δi ≈ (r/h) * Δv_w  at ascending node (u≈0)
            # => Δv_w ≈ (h/r) * (i_target - i_now)
            pv = s.getPVCoordinates(self.inertial_frame)
            r_vec = pv.getPosition()
            v_vec = pv.getVelocity()
            h_vec = Vector3D.crossProduct(r_vec, v_vec)
            h = float(h_vec.getNorm())
            r = float(r_vec.getNorm())
            if h > 0.0 and r > 0.0:
                di_needed = self.inc_target - inc_now
                dvw = (h / r) * di_needed  # m/s along +W
                if abs(dvw) >= self.min_ns_dv:
                    if abs(dvw) > self.max_ns_dv:
                        dvw = self.max_ns_dv if dvw > 0 else -self.max_ns_dv
                    dv_w_now = dvw
                    self._t_last_ns_burn = date

        # If nothing to do, continue
        if dv_t_now == 0.0 and dv_w_now == 0.0:
            return ACTION.CONTINUE

        # Store what to apply in resetState
        self._dv_t_to_apply_now = float(dv_t_now)
        self._dv_w_to_apply_now = float(dv_w_now)

        # Log
        self._log_burn(
            s=s,
            opp=opp,
            lon_err=lon_err,
            drift_now=drift_now,
            drift_cmd=drift_cmd,
            dv_t_pair=dv_t_pair_total,
            dv_t_this=dv_t_now,
            dv_w=dv_w_now,
            inc_now=inc_now,
            inc_err=inc_err,
        )

        return ACTION.RESET_STATE

    def resetState(self, detector, oldState):
        """
        Apply the instantaneous velocity change and return updated SpacecraftState.
        """
        dv_t = float(self._dv_t_to_apply_now)
        dv_w = float(self._dv_w_to_apply_now)

        self._dv_t_to_apply_now = 0.0
        self._dv_w_to_apply_now = 0.0

        if dv_t == 0.0 and dv_w == 0.0:
            return oldState

        pv = oldState.getPVCoordinates(self.inertial_frame)
        r = pv.getPosition()
        v = pv.getVelocity()

        v_norm = float(v.getNorm())
        if v_norm == 0.0:
            return oldState

        # Tangential unit vector
        t_hat = v.scalarMultiply(1.0 / v_norm)

        # Orbit normal (W) unit vector
        h_vec = Vector3D.crossProduct(r, v)
        h_norm = float(h_vec.getNorm())
        if h_norm == 0.0:
            return oldState
        w_hat = h_vec.scalarMultiply(1.0 / h_norm)

        v_new = v
        if dv_t != 0.0:
            v_new = v_new.add(t_hat.scalarMultiply(dv_t))
        if dv_w != 0.0:
            v_new = v_new.add(w_hat.scalarMultiply(dv_w))

        pv_new = PVCoordinates(r, v_new)
        orbit_new = CartesianOrbit(
            pv_new, self.inertial_frame, oldState.getDate(), self.mu
        )
        return SpacecraftState(orbit_new, float(oldState.getMass()))

    # ---- internals ----
    def _subsat_lla(self, s) -> Tuple[float, float, float]:
        date = s.getDate()
        pos = s.getPVCoordinates(self.inertial_frame).getPosition()
        gp = self.earth.transform(pos, self.inertial_frame, date)
        return (
            float(gp.getLongitude()),
            float(gp.getLatitude()),
            float(gp.getAltitude()),
        )

    def _update_lon_mean(self, date: AbsoluteDate, lon_wrapped: float) -> None:
        import math

        if not self._have_lon_mean:
            self._have_lon_mean = True
            self._t_last_lon = date
            self._lon_mean_unwrapped = lon_wrapped
            return

        assert self._t_last_lon is not None
        dt = float(date.durationFrom(self._t_last_lon))
        if dt <= 0.0:
            return

        lon_unwrapped = unwrap_relative(lon_wrapped, self._lon_mean_unwrapped)
        alpha = 1.0 - math.exp(-dt / self.tau_lon_mean)
        self._lon_mean_unwrapped = (
            1.0 - alpha
        ) * self._lon_mean_unwrapped + alpha * lon_unwrapped
        self._t_last_lon = date

    def _classify_opportunity(self, detector, increasing) -> Tuple[bool, str]:
        """
        Returns (is_node, opp_string).
        """
        try:
            name = detector.getClass().getName()
        except Exception:
            name = ""

        if name.endswith("NodeDetector"):
            return True, ("ASC" if bool(increasing) else "DES")

        # Fallback opportunity (DateDetector or others)
        self._opp_parity = 1 - self._opp_parity
        return False, ("OPP" if self._opp_parity == 0 else "OPP")

    def _ew_start_allowed(self, date: AbsoluteDate, *, is_node: bool, opp: str) -> bool:
        """
        Gate E/W pair initiation:
          - If node opportunities exist: start only on ASC
          - Otherwise: start every-other opportunity (coarsely)
          - Enforce cooldown
          - Only if drift error is beyond tolerance
        """
        # Cooldown
        if self._t_last_ew_pair_start is not None:
            if float(date.durationFrom(self._t_last_ew_pair_start)) < self.ew_cooldown:
                return False

        if is_node:
            return opp == "ASC"

        # Non-node: allow initiation on parity 0 only (reduces spam)
        return self._opp_parity == 0

    def _ns_allowed(
        self, date: AbsoluteDate, *, is_node: bool, opp: str, inc_err: float
    ) -> bool:
        """
        Gate N/S burns:
          - Only if |inc_err| outside deadband
          - Only on ASC (and optionally DES)
          - Enforce cooldown
        """
        if abs(inc_err) <= self.inc_deadband:
            return False

        if self._t_last_ns_burn is not None:
            if float(date.durationFrom(self._t_last_ns_burn)) < self.ns_cooldown:
                return False

        if not is_node:
            # Without true node knowledge, skip N/S (avoids wrong geometry)
            return False

        if opp == "ASC":
            return True
        if opp == "DES" and self.ns_apply_on_des:
            return True
        return False

    def _log_burn(
        self,
        *,
        s,
        opp: str,
        lon_err: float,
        drift_now: float,
        drift_cmd: float,
        dv_t_pair: float,
        dv_t_this: float,
        dv_w: float,
        inc_now: float,
        inc_err: float,
    ):
        import math

        lon_mean_deg = math.degrees(wrap_pi(self._lon_mean_unwrapped))
        lon_err_deg = math.degrees(lon_err)
        drift_now_deg_per_day = math.degrees(drift_now) * 86400.0
        drift_cmd_deg_per_day = math.degrees(drift_cmd) * 86400.0

        self.burns.append(
            BurnLog(
                date=s.getDate(),
                opp=str(opp),
                ew_mode=str(self._ew_mode),
                lon_mean_deg=float(lon_mean_deg),
                lon_err_deg=float(lon_err_deg),
                drift_now_deg_per_day=float(drift_now_deg_per_day),
                drift_cmd_deg_per_day=float(drift_cmd_deg_per_day),
                dv_t_pair_mps=float(dv_t_pair),
                dv_t_this_mps=float(dv_t_this),
                dv_w_mps=float(dv_w),
                inc_deg=float(math.degrees(inc_now)),
                inc_err_deg=float(math.degrees(inc_err)),
            )
        )

    # ---- getters for logger ----
    def has_lon_mean(self) -> bool:
        return self._have_lon_mean

    def lon_mean_unwrapped(self) -> float:
        return self._lon_mean_unwrapped

    def ew_mode(self) -> str:
        return self._ew_mode

    def get_omega_e(self) -> float:
        return self._omega_e

    def drift_cmd_rad_s(self, lon_err: float) -> float:
        """
        For logger: compute current drift_cmd given current lon_err and mode.
        """
        drift_cmd = (+self.drift_mag) if self._ew_mode == "EAST" else (-self.drift_mag)
        if self.enable_trim and self.trim_tau > 0.0:
            drift_cmd += -lon_err / self.trim_tau
        if lon_err > self.lon_deadband:
            drift_cmd = -abs(self.drift_mag)
        elif lon_err < -self.lon_deadband:
            drift_cmd = +abs(self.drift_mag)
        return drift_cmd


# -----------------------------------------------------------------------------
# Fixed-step logger
# -----------------------------------------------------------------------------
class FixedStepLogger:
    def __init__(
        self,
        *,
        earth: OneAxisEllipsoid,
        inertial_frame,
        sk: GeoSynStationKeepingHandler,
        t0: AbsoluteDate,
    ):
        self.earth = earth
        self.inertial_frame = inertial_frame
        self.sk = sk
        self.t0 = t0
        self.samples: List[SampleLog] = []

    def init(self, *args):
        return

    def finish(self, *args):
        return

    def handleStep(self, s):
        import math

        date = s.getDate()
        t_days = float(date.durationFrom(self.t0)) / 86400.0

        pos = s.getPVCoordinates(self.inertial_frame).getPosition()
        gp = self.earth.transform(pos, self.inertial_frame, date)

        lon = float(gp.getLongitude())
        lat = float(gp.getLatitude())
        alt = float(gp.getAltitude())

        # mean lon from SK estimator (low-passed)
        if self.sk.has_lon_mean():
            lon_mean = self.sk.lon_mean_unwrapped()
        else:
            lon_mean = lon

        lon_err = wrap_pi(lon_mean - self.sk.lon_target)

        # mean motion drift estimate from a(t)
        a = float(s.getOrbit().getA())
        n_now = (self.sk.mu / (a * a * a)) ** 0.5
        drift_now = n_now - self.sk.get_omega_e()

        drift_cmd = self.sk.drift_cmd_rad_s(lon_err)

        inc_now = float(s.getOrbit().getI())
        inc_err = inc_now - self.sk.inc_target

        self.samples.append(
            SampleLog(
                date=date,
                t_days=float(t_days),
                lon_deg=float(math.degrees(lon)),
                lat_deg=float(math.degrees(lat)),
                alt_m=float(alt),
                lon_mean_deg=float(math.degrees(wrap_pi(lon_mean))),
                lon_err_deg=float(math.degrees(lon_err)),
                drift_now_deg_per_day=float(math.degrees(drift_now) * 86400.0),
                drift_cmd_deg_per_day=float(math.degrees(drift_cmd) * 86400.0),
                ew_mode=str(self.sk.ew_mode()),
                inc_deg=float(math.degrees(inc_now)),
                inc_err_deg=float(math.degrees(inc_err)),
            )
        )


# -----------------------------------------------------------------------------
# Propagator builder (tuned for GEO/GSO)
# -----------------------------------------------------------------------------
def build_geo_propagator(
    *,
    initial_orbit,
    mass_kg: float,
    earth: OneAxisEllipsoid,
    itrf,
    inertial_frame,
    utc,
    mu: float,
    ae: float,
    # integrator config
    min_step: float = 10.0,
    max_step: float = 7200.0,
    initial_step: float = 600.0,
    position_tolerance_m: float = 10.0,
    # gravity
    gravity_degree: int = 20,
    gravity_order: int = 20,
    # toggles
    include_ocean_tides: bool = True,
    include_erp: bool = False,
) -> NumericalPropagator:
    tolerances = NumericalPropagator.tolerances(
        position_tolerance_m, initial_orbit, OrbitType.KEPLERIAN
    )
    integrator = DormandPrince853Integrator(
        float(min_step), float(max_step), tolerances[0], tolerances[1]
    )
    integrator.setInitialStepSize(float(initial_step))

    propagator = NumericalPropagator(integrator)
    propagator.setInitialState(SpacecraftState(initial_orbit, float(mass_kg)))

    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()
    earth_body = CelestialBodyFactory.getEarth()

    # A) Earth gravity field
    gravity_provider = GravityFieldFactory.getNormalizedProvider(
        int(gravity_degree), int(gravity_order)
    )
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))

    # Relativity
    propagator.addForceModel(DeSitterRelativity(earth_body, sun))
    propagator.addForceModel(LenseThirringRelativity(mu, itrf))
    propagator.addForceModel(Relativity(mu))

    # Third bodies
    propagator.addForceModel(ThirdBodyAttraction(sun))
    propagator.addForceModel(ThirdBodyAttraction(moon))
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getVenus()))
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getMars()))
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getJupiter()))
    propagator.addForceModel(ThirdBodyAttraction(CelestialBodyFactory.getSaturn()))

    # Solid tides (+ optional ocean tides)
    ut1 = TimeScalesFactory.getUT1(IERSConventions.IERS_2010, True)
    tidesystem = TideSystem.ZERO_TIDE
    propagator.addForceModel(
        SolidTides(itrf, ae, mu, tidesystem, IERSConventions.IERS_2010, ut1, sun)
    )
    propagator.addForceModel(
        SolidTides(itrf, ae, mu, tidesystem, IERSConventions.IERS_2010, ut1, moon)
    )

    if include_ocean_tides:
        propagator.addForceModel(
            OceanTides(itrf, ae, mu, 8, 8, IERSConventions.IERS_2010, ut1)
        )

    # SRP
    radiation_sensitive = IsotropicRadiationSingleCoefficient(1.0, 1.2)
    srp = SolarRadiationPressure(sun, earth, radiation_sensitive)
    srp.addOccultingBody(moon, Constants.MOON_EQUATORIAL_RADIUS)
    propagator.addForceModel(srp)

    # ERP (optional; expensive)
    if include_erp:
        propagator.addForceModel(
            KnockeRediffusedForceModel(
                sun,
                radiation_sensitive,
                Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                radians(1.0),
                utc,
            )
        )

    return propagator


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def plot_stationkeeping(
    samples: List[SampleLog],
    *,
    lon_deadband_deg: float,
    lon_switch_deg: float,
    inc_target_deg: float,
    inc_deadband_deg: float,
):
    import matplotlib.pyplot as plt
    import numpy as np

    t = np.array([s.t_days for s in samples], dtype=np.float64)

    lon_err = np.array([s.lon_err_deg for s in samples], dtype=np.float64)
    drift_now = np.array([s.drift_now_deg_per_day for s in samples], dtype=np.float64)
    drift_cmd = np.array([s.drift_cmd_deg_per_day for s in samples], dtype=np.float64)

    inc = np.array([s.inc_deg for s in samples], dtype=np.float64)
    inc_err = np.array([s.inc_err_deg for s in samples], dtype=np.float64)
    lat = np.array([s.lat_deg for s in samples], dtype=np.float64)

    # Longitude error
    plt.figure()
    plt.plot(t, lon_err)
    plt.axhline(+lon_deadband_deg, linestyle="--")
    plt.axhline(-lon_deadband_deg, linestyle="--")
    plt.axhline(+lon_switch_deg, linestyle=":")
    plt.axhline(-lon_switch_deg, linestyle=":")
    plt.title("Mean longitude error (E/W station-keeping)")
    plt.xlabel("Time (days)")
    plt.ylabel("Mean longitude error (deg)")
    plt.grid(True)

    # Drift
    plt.figure()
    plt.plot(t, drift_now, label="drift_now (n - ωE)")
    plt.plot(t, drift_cmd, label="drift_cmd", linestyle="--")
    plt.axhline(0.0, linestyle="--")
    plt.title("Earth-fixed drift (deg/day)")
    plt.xlabel("Time (days)")
    plt.ylabel("Drift (deg/day)")
    plt.grid(True)
    plt.legend()

    # Inclination
    plt.figure()
    plt.plot(t, inc, label="inclination")
    plt.axhline(inc_target_deg + inc_deadband_deg, linestyle="--")
    plt.axhline(inc_target_deg - inc_deadband_deg, linestyle="--")
    plt.title("Inclination (N/S station-keeping)")
    plt.xlabel("Time (days)")
    plt.ylabel("Inclination (deg)")
    plt.grid(True)
    plt.legend()

    # Subsat latitude (diagnostic; for inclined GSO it will oscillate daily)
    plt.figure()
    plt.plot(t, lat)
    plt.title("Subsatellite latitude (diagnostic)")
    plt.xlabel("Time (days)")
    plt.ylabel("Latitude (deg)")
    plt.grid(True)

    plt.show()


def plot_orbit_basemap(
    lat_deg,
    lon_deg,
    *,
    projection="cyl",
    figsize=(12, 6),
    title="GSO ground track",
    linewidth=0.8,
    wrap_lon=True,
):
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from mpl_toolkits.basemap import Basemap
    except Exception as e:
        raise ImportError(
            "Basemap is not installed. Install 'basemap basemap-data-hires' or switch to Cartopy."
        ) from e

    lat = np.asarray(lat_deg, dtype=np.float64).ravel()
    lon = np.asarray(lon_deg, dtype=np.float64).ravel()
    if lat.shape != lon.shape:
        raise ValueError("lat_deg and lon_deg must have same shape")

    if wrap_lon:
        lon = (lon + 180.0) % 360.0 - 180.0

    # Split at dateline
    dlon = np.abs(np.diff(lon))
    breaks = np.where(dlon > 180.0)[0] + 1
    segments = np.split(np.arange(lon.size), breaks)

    fig, ax = plt.subplots(figsize=figsize)
    m = Basemap(projection=projection, lon_0=0, ax=ax)
    m.drawmapboundary(fill_color="white")
    try:
        m.drawlsmask(land_color="0.95", ocean_color="white")
    except Exception:
        pass
    m.drawcoastlines(linewidth=0.6)
    m.drawcountries(linewidth=0.4)

    for seg in segments:
        if seg.size < 2:
            continue
        x, y = m(lon[seg], lat[seg])
        ax.plot(x, y, linewidth=linewidth)

    ax.set_title(title)
    plt.show()
    return fig, ax, m


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    # Time / frames / constants
    utc = TimeScalesFactory.getUTC()
    t0 = AbsoluteDate(2026, 1, 16, 12, 0, 0.0, utc)

    inertial_frame = FramesFactory.getGCRF()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

    mu = Constants.WGS84_EARTH_MU
    ae = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf
    )

    # -----------------------------
    # Initial orbit (geosynchronous / geostationary-ish)
    # -----------------------------
    # For geostationary-ish, set i very small (e.g., 0.05 deg) so nodes still exist numerically.
    # For inclined geosynchronous, i can be several degrees.
    initial_orbit = KeplerianOrbit(
        42164000.0,  # a (m)
        0.0015,  # e
        radians(10.0),  # i (rad)  (10 deg = inclined geosynchronous)
        0.0,  # RAAN
        0.0,  # argp
        0.0,  # nu
        PositionAngleType.TRUE,
        inertial_frame,
        t0,
        mu,
    )

    mass_kg = 1000.0

    propagator = build_geo_propagator(
        initial_orbit=initial_orbit,
        mass_kg=mass_kg,
        earth=earth,
        itrf=itrf,
        inertial_frame=inertial_frame,
        utc=utc,
        mu=mu,
        ae=ae,
        min_step=10.0,
        max_step=7200.0,
        initial_step=600.0,
        position_tolerance_m=10.0,
        gravity_degree=20,
        gravity_order=20,
        include_ocean_tides=True,
        include_erp=False,
    )

    # -----------------------------
    # Targets
    # -----------------------------
    s0 = propagator.getInitialState()
    gp0 = earth.transform(
        s0.getPVCoordinates(inertial_frame).getPosition(),
        inertial_frame,
        s0.getDate(),
    )
    lon_target_deg = float(gp0.getLongitude()) * 180.0 / 3.141592653589793

    # Inclination target:
    # - If you're simulating a GEO slot, set inc_target_deg ~ 0.0 (or 0.05 deg).
    # - If you're simulating an inclined geosynchronous mission, set it to your desired inclination.
    inc0_deg = float(s0.getOrbit().getI()) * 180.0 / 3.141592653589793
    inc_target_deg = inc0_deg  # default: "hold what you started with"

    # -----------------------------
    # Controller tuning (start here)
    # -----------------------------
    deadband_deg = 2  # allowable +/- around target longitude
    switch_deg = 0.75  # must be reachable on both sides
    drift_mag_deg_per_day = 0.04  # traverse 4 deg in ~100 days
    drift_tol_deg_per_day = 0.01  # fewer burns; allow mismatch
    tau_mean_days = 20.0  # smoother drift estimate
    ew_cooldown_days = 25.0  # bigger => fewer E/W burns
    max_pair_dv_mps = 1.0
    min_pair_dv_mps = 0.05
    enable_trim = True
    trim_tau_days = 180.0
    inc_deadband_deg = 0.1  # allowable +/- around target inclination
    ns_cooldown_days = 40.0  # bigger => fewer N/S burns

    sk_impl = GeoSynStationKeepingHandler(
        earth=earth,
        inertial_frame=inertial_frame,
        mu=mu,
        lon_target_deg=lon_target_deg,
        lon_deadband_deg=deadband_deg,
        lon_switch_deg=switch_deg,
        drift_mag_deg_per_day=drift_mag_deg_per_day,
        drift_tolerance_deg_per_day=drift_tol_deg_per_day,
        tau_lon_mean_days=tau_mean_days,
        ew_cooldown_days=ew_cooldown_days,
        max_pair_dv_mps=max_pair_dv_mps,
        min_pair_dv_mps=min_pair_dv_mps,
        enable_trim=enable_trim,
        trim_tau_days=trim_tau_days,
        inc_target_deg=inc_target_deg,
        inc_deadband_deg=inc_deadband_deg,
        ns_cooldown_days=ns_cooldown_days,
        max_ns_dv_mps=1.0,
        min_ns_dv_mps=0.02,
        ns_apply_on_des=False,
    )

    sk_proxy = JProxy(EventHandler, inst=sk_impl)

    # -----------------------------
    # Maneuver opportunities: prefer NodeDetector
    # -----------------------------
    node_det = NodeDetector(inertial_frame)
    if hasattr(node_det, "withMaxCheck"):
        node_det = node_det.withMaxCheck(7200.0)
    if hasattr(node_det, "withThreshold"):
        node_det = node_det.withThreshold(1.0)

    node_det = _attach_handler_to_detector(node_det, sk_proxy)
    propagator.addEventDetector(node_det)

    # -----------------------------
    # Fixed-step logging (hourly)
    # -----------------------------
    logger_impl = FixedStepLogger(
        earth=earth, inertial_frame=inertial_frame, sk=sk_impl, t0=t0
    )
    logger_proxy = JProxy(OrekitFixedStepHandler, inst=logger_impl)
    _set_fixed_step_handler(propagator, 3600.0, logger_proxy)

    # -----------------------------
    # Run
    # -----------------------------
    duration_s = 86400.0 * 365.0 * 3.0  # 3 years
    tf = t0.shiftedBy(duration_s)

    import time

    t_start = time.time()
    final_state = propagator.propagate(tf)
    t_end = time.time()

    print(f"Propagation runtime: {t_end - t_start:.1f} s")
    print(f"Target lon (deg): {lon_target_deg:.6f}")
    print(f"Inc target (deg): {inc_target_deg:.6f}")
    print(f"Final date: {final_state.getDate()}")
    print(f"Samples: {len(logger_impl.samples)}")
    print(f"Burns: {len(sk_impl.burns)}")

    if logger_impl.samples:
        lon_err = [s.lon_err_deg for s in logger_impl.samples]
        drift_now = [s.drift_now_deg_per_day for s in logger_impl.samples]
        drift_cmd = [s.drift_cmd_deg_per_day for s in logger_impl.samples]
        inc = [s.inc_deg for s in logger_impl.samples]
        inc_err = [s.inc_err_deg for s in logger_impl.samples]

        print(f"Lon err  min/max (deg): {min(lon_err):+.4f} / {max(lon_err):+.4f}")
        print(
            f"Drift now min/max (deg/day): {min(drift_now):+.4f} / {max(drift_now):+.4f}"
        )
        print(
            f"Drift cmd min/max (deg/day): {min(drift_cmd):+.4f} / {max(drift_cmd):+.4f}"
        )
        print(f"Inclination min/max (deg): {min(inc):+.4f} / {max(inc):+.4f}")
        print(
            f"Inclination err min/max (deg): {min(inc_err):+.4f} / {max(inc_err):+.4f}"
        )

    if sk_impl.burns:
        print("\nFirst 20 burns:")
        for b in sk_impl.burns[:20]:
            print(
                f"  {b.date}  {b.opp}  EW={b.ew_mode:<4}  "
                f"err={b.lon_err_deg:+.3f} deg  "
                f"drift_now={b.drift_now_deg_per_day:+.3f}  drift_cmd={b.drift_cmd_deg_per_day:+.3f}  "
                f"dv_t_pair={b.dv_t_pair_mps:+.4f}  dv_t_now={b.dv_t_this_mps:+.4f}  dv_w={b.dv_w_mps:+.4f}  "
                f"inc={b.inc_deg:+.3f}  inc_err={b.inc_err_deg:+.3f}"
            )

    # -----------------------------
    # Plots
    # -----------------------------
    try:
        plot_stationkeeping(
            logger_impl.samples,
            lon_deadband_deg=deadband_deg,
            lon_switch_deg=switch_deg,
            inc_target_deg=inc_target_deg,
            inc_deadband_deg=inc_deadband_deg,
        )
    except Exception as e:
        print(f"Plotting skipped: {e}")

    try:
        plot_orbit_basemap(
            [s.lat_deg for s in logger_impl.samples],
            [s.lon_deg for s in logger_impl.samples],
            title="GSO ground track (hourly samples)",
        )
    except Exception as e:
        print(f"Basemap skipped: {e}")


if __name__ == "__main__":
    main()
