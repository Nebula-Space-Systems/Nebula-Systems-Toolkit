# this script represents the interface i would like for my orbit class, which is fully implemented in java using orekit.
# the python code here is just a thin wrapper/interface around the faster java code.
from typing import Union
from astropy.time import Time
import astropy.units as u
import numpy as np

# from org.orekit.utils import PVCoordinates
from org.orekit.propagation import Propagator, SpacecraftState
from org.orekit.propagation.analytical import EcksteinHechlerPropagator
from org.orekit.orbits import PositionAngleType
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.frames import FramesFactory, Frame
from org.orekit.utils import Constants, IERSConventions
from org.orekit.orbits import KeplerianOrbit
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.attitudes import LofOffset


WGS84_ELLIPSOID = OneAxisEllipsoid(
    6378137.0, 6356752.314245, FramesFactory.getITRF(IERSConventions.IERS_2010, True)
)


def astropy_time_to_orekit_date(time: Time) -> AbsoluteDate:
    # convert astropy Time to orekit AbsoluteDate
    utc = TimeScalesFactory.getUTC()
    return AbsoluteDate(time.utc.datetime, utc)


class OrbitCreationMixin:
    @classmethod
    def from_spacecraft_state(
        cls,
        state: SpacecraftState,
        iers_convention: IERSConventions = IERSConventions.IERS_2010,
        simple_eop: bool = True,
    ) -> "Orbit": ...

    @classmethod
    def from_kepler_two_body(
        cls,
        epoch: Time,
        a: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type: PositionAngleType = PositionAngleType.MEAN,
        mass: float = 1000.0,
        inertial_frame: Frame = FramesFactory.getGCRF(),
        iers_convention: IERSConventions = IERSConventions.IERS_2010,
        simple_eop: bool = True,
    ) -> "Orbit":
        # construct the propagator using analytical two body motion
        ...

    @classmethod
    def from_kepler(
        cls,
        epoch: Time,
        a: float,
        e: float,
        i: float,
        raan: float,
        argp: float,
        anomaly: float,
        anomaly_type: PositionAngleType = PositionAngleType.MEAN,
        mass: float = 1000.0,
        inertial_frame: Frame = FramesFactory.getGCRF(),
        iers_convention: IERSConventions = IERSConventions.IERS_2010,
        simple_eop: bool = True,
    ) -> "Orbit":
        initial_orbit = KeplerianOrbit(
            a,
            e,
            i,
            argp,
            raan,
            anomaly,
            positionAngleType=anomaly_type,
            frame=inertial_frame,
            absoluteDate=astropy_time_to_orekit_date(epoch),
            mu=Constants.EIGEN5C_EARTH_MU,
        )
        propagator = EcksteinHechlerPropagator(
            initial_orbit,
            Constants.EIGEN5C_EARTH_EQUATORIAL_RADIUS,
            Constants.EIGEN5C_EARTH_MU,
            Constants.EIGEN5C_EARTH_C20,
            Constants.EIGEN5C_EARTH_C30,
            Constants.EIGEN5C_EARTH_C40,
            Constants.EIGEN5C_EARTH_C50,
            Constants.EIGEN5C_EARTH_C60,
        )
        return cls(propagator, iers_convention, simple_eop)


class Orbit(OrbitCreationMixin):

    def __init__(
        self,
        propagator: Propagator,
        iers: IERSConventions = IERSConventions.IERS_2010,
        simple_eop: bool = True,
    ):
        # orekit_propagator is the underlying orekit propagator object that does the actual propagation.
        # iers is the IERS data used for Earth orientation parameters, which is needed for accurate transformations between inertial and Earth-fixed frames.
        # simple_eop is a flag to use a simplified Earth orientation model that does not require IERS data, which can be faster but less accurate for long-term propagation.
        self.propagator = propagator
        self.iers = iers
        self.simple_eop = simple_eop

    def get_native_frame(self) -> Frame:
        return self.propagator.getFrame()

    def get_p(self, time: Time, frame: Union[Frame, None] = None) -> u.Quantity:
        # return the position at the requested times and frame.
        # if the orbit has not been propagated to that time, then it is propagated
        # and the state is cached for later use.
        # The returned position is in meters.
        # If time is scalar, return shape (3,). If time is array-like of shape (N,), return shape (N, 3).
        ...

    def get_v(self, time: Time, frame: Union[Frame, None] = None) -> u.Quantity:
        # return the velocity at the requested times and frame.
        # if the orbit has not been propagated to that time, then it is propagated
        # and the state is cached for later use.
        # The returned velocity is in meters per second.
        # If time is scalar, return shape (3,). If time is array-like of shape (N,), return shape (N, 3).
        ...

    def get_a(self, time: Time, frame: Union[Frame, None] = None) -> u.Quantity:
        # return the acceleration at the requested times and frame.
        # if the orbit has not been propagated to that time, then it is propagated
        # and the state is cached for later use.
        # The returned acceleration is in meters per second squared.
        # If time is scalar, return shape (3,). If time is array-like of shape (N,), return shape (N, 3).
        ...

    def get_pv(
        self, time: Time, frame: Union[Frame, None] = None
    ) -> tuple[u.Quantity, u.Quantity]:
        # return the position and velocity at the requested times and frame.
        # if the orbit has not been propagated to that time, then it is propagated
        # and the state is cached for later use.
        # The returned position is in meters and velocity is in meters per second.
        # If time is scalar, return shapes (3,) for position and velocity. If time is array-like of shape (N,), return shapes (N, 3) for position and velocity.
        ...

    def get_pva(
        self, time: Time, frame: Union[Frame, None] = None
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        # return the position, velocity, and acceleration at the requested times and frame.
        # if the orbit has not been propagated to that time, then it is propagated
        # and the state is cached for later use.
        # The returned position is in meters, velocity is in meters per second, and acceleration is in meters per second squared.
        # If time is scalar, return shapes (3,) for position, velocity, and acceleration. If time is array-like of shape (N,), return shapes (N, 3) for position, velocity, and acceleration.
        ...

    def get_geodetic(
        self, time: Time, ellipsoid: OneAxisEllipsoid = WGS84_ELLIPSOID
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        # return the geodetic latitude, longitude, and altitude at the requested times and ellipsoid.
        # if the orbit has not been propagated to that time, then it is propagated
        # and the state is cached for later use.
        # The returned latitude and longitude are in degrees, and altitude is in meters.
        # If time is scalar, return shapes (3,) for latitude, longitude, and altitude. If time is array-like of shape (N,), return shapes (N, 3) for latitude, longitude, and altitude.
        ...

    def get_attitude(self, time: Time) -> np.ndarray:
        # interface in work, not sure exactly what to return here or what is required.
        # I want all Orbit objects to have an attitude component though, defaulted to LofOffset with zero offset from the local orbital frame.
        ...


if __name__ == "__main__":
    # example analysis usage
    epoch = Time("2026-01-01T00:00:00", scale="utc")

    seed_orbit = Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
    )
