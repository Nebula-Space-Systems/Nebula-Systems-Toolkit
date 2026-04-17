from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time

import nstk.propagation.orbit as orbit_module
from nstk.propagation import (
    Orbit,
    RateLimitedYawSteeringProvider,
    build_numerical_propagator,
    build_two_body_propagator,
)


def _make_two_body_propagator(
    epoch: Time,
    *,
    a: float = 7000e3,
    e: float = 0.001,
    i: float = np.deg2rad(53.0),
    raan: float = np.deg2rad(20.0),
    argp: float = np.deg2rad(15.0),
    anomaly: float = np.deg2rad(10.0),
):
    return build_two_body_propagator(
        epoch=epoch,
        a=a,
        e=e,
        i=i,
        raan=raan,
        argp=argp,
        anomaly=anomaly,
        inertial_frame="gcrf",
    )


def _make_low_beta_propagator() -> tuple[Time, object]:
    epoch = Time("2026-03-20T12:00:00", scale="utc")
    propagator = _make_two_body_propagator(
        epoch,
        i=np.deg2rad(1.0),
        raan=0.0,
        argp=0.0,
        anomaly=0.0,
    )
    return epoch, propagator


def _find_low_beta_peak_time_s(epoch: Time, pv_provider) -> float:
    probe = RateLimitedYawSteeringProvider(
        inertial_frame="gcrf",
        reference_epoch=epoch,
        max_yaw_rate_rad_s=10.0,
        max_yaw_acceleration_rad_s2=10.0,
        kp=0.0,
        kd=0.0,
        finite_difference_step_s=0.05,
    )
    coarse_times_s = np.linspace(0.0, 6000.0, 15)
    reference = np.vstack(
        [probe.get_reference_yaw_state(pv_provider, float(t)) for t in coarse_times_s]
    )
    peak_idx = int(np.argmax(np.abs(reference[:, 1])))
    return float(coarse_times_s[peak_idx])


def _make_seeded_provider(
    pv_provider,
    epoch: Time,
    *,
    max_yaw_rate_rad_s: float,
    max_yaw_acceleration_rad_s2: float,
    kp: float,
    kd: float,
    finite_difference_step_s: float,
    initial_yaw_rate_rad_s: float | None = None,
    enable_cache: bool = True,
    cache_step_s: float = 10.0,
) -> RateLimitedYawSteeringProvider:
    bootstrap = RateLimitedYawSteeringProvider(
        inertial_frame="gcrf",
        reference_epoch=epoch,
        max_yaw_rate_rad_s=max(10.0, max_yaw_rate_rad_s),
        max_yaw_acceleration_rad_s2=max(10.0, max_yaw_acceleration_rad_s2),
        kp=1.0,
        kd=1.0,
        finite_difference_step_s=finite_difference_step_s,
        enable_cache=enable_cache,
        cache_step_s=cache_step_s,
    )
    ref0 = bootstrap.get_reference_yaw_state(pv_provider, 0.0)
    omega0 = float(ref0[1] if initial_yaw_rate_rad_s is None else initial_yaw_rate_rad_s)
    omega0 = float(np.clip(omega0, -max_yaw_rate_rad_s, max_yaw_rate_rad_s))
    return RateLimitedYawSteeringProvider(
        inertial_frame="gcrf",
        reference_epoch=epoch,
        max_yaw_rate_rad_s=max_yaw_rate_rad_s,
        max_yaw_acceleration_rad_s2=max_yaw_acceleration_rad_s2,
        kp=kp,
        kd=kd,
        initial_yaw_rad=float(ref0[0]),
        initial_yaw_rate_rad_s=omega0,
        finite_difference_step_s=finite_difference_step_s,
        enable_cache=enable_cache,
        cache_step_s=cache_step_s,
    )


def _rotation_distance(rot_a, rot_b) -> float:
    delta = rot_a.applyInverseTo(rot_b)
    return float(abs(delta.getAngle()))


def test_relative_yaw_extraction_and_sign_convention() -> None:
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Rotation, RotationConvention, Vector3D

    base = Rotation.IDENTITY
    positive = Rotation(Vector3D.PLUS_K, 0.3, RotationConvention.FRAME_TRANSFORM)
    negative = Rotation(Vector3D.PLUS_K, -0.45, RotationConvention.FRAME_TRANSFORM)

    assert RateLimitedYawSteeringProvider.extract_relative_yaw(base, positive) == pytest.approx(
        0.3,
        abs=1.0e-15,
    )
    assert RateLimitedYawSteeringProvider.extract_relative_yaw(base, negative) == pytest.approx(
        -0.45,
        abs=1.0e-15,
    )


def test_rate_limited_yaw_provider_matches_ideal_yaw_steering_with_large_limits() -> None:
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Vector3D
    from org.orekit.attitudes import NadirPointing, YawSteering
    from org.orekit.bodies import CelestialBodyFactory

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)
    gcrf = orbit_module.FramesFactory.getGCRF()
    sun = CelestialBodyFactory.getSun()

    base_law = NadirPointing(gcrf, orbit_module.WGS84_ELLIPSOID)
    ideal_law = YawSteering(gcrf, base_law, sun, Vector3D.PLUS_I)
    provider = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=100.0,
        max_yaw_acceleration_rad_s2=100.0,
        kp=0.0,
        kd=0.0,
        finite_difference_step_s=0.05,
    )
    actual_law = provider.to_orekit()
    epoch_date = raw_propagator.getInitialState().getDate()

    for dt_s in (0.0, 30.0, 60.0, 120.0):
        date = epoch_date.shiftedBy(float(dt_s))
        actual = actual_law.getAttitude(raw_propagator, date, gcrf)
        ideal = ideal_law.getAttitude(raw_propagator, date, gcrf)

        assert _rotation_distance(actual.getRotation(), ideal.getRotation()) < 5.0e-5

        spin_delta = actual.getSpin().subtract(ideal.getSpin())
        assert float(spin_delta.getNorm()) < 5.0e-4


def test_rate_limited_yaw_provider_cache_matches_uncached_solution() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)

    cached = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
        enable_cache=True,
        cache_step_s=5.0,
    )
    uncached = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
        enable_cache=False,
    )

    for dt_s in (0.0, 20.0, 60.0, 120.0):
        np.testing.assert_allclose(
            cached.get_actual_yaw_state(raw_propagator, dt_s),
            uncached.get_actual_yaw_state(raw_propagator, dt_s),
            rtol=0.0,
            atol=5.0e-6,
        )


def test_rate_limited_yaw_provider_enforces_max_yaw_rate() -> None:
    epoch, raw_propagator = _make_low_beta_propagator()
    peak_time_s = _find_low_beta_peak_time_s(epoch, raw_propagator)
    reference_epoch = Time(epoch.unix + peak_time_s, format="unix", scale="utc")
    provider = _make_seeded_provider(
        raw_propagator,
        reference_epoch,
        max_yaw_rate_rad_s=2.0e-3,
        max_yaw_acceleration_rad_s2=5.0e-3,
        kp=0.5,
        kd=2.0,
        finite_difference_step_s=0.05,
    )

    times_s = np.linspace(-60.0, 60.0, 11)
    actual = np.vstack([provider.get_actual_yaw_state(raw_propagator, float(t)) for t in times_s])

    assert np.max(np.abs(actual[:, 1])) <= 2.0e-3 + 1.0e-10


def test_rate_limited_yaw_provider_enforces_max_yaw_acceleration() -> None:
    epoch, raw_propagator = _make_low_beta_propagator()
    peak_time_s = _find_low_beta_peak_time_s(epoch, raw_propagator)
    reference_epoch = Time(epoch.unix + peak_time_s, format="unix", scale="utc")
    provider = _make_seeded_provider(
        raw_propagator,
        reference_epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=7.5e-5,
        kp=0.4,
        kd=1.0,
        finite_difference_step_s=0.05,
    )

    times_s = np.linspace(-60.0, 60.0, 11)
    actual = np.vstack([provider.get_actual_yaw_state(raw_propagator, float(t)) for t in times_s])

    assert np.max(np.abs(actual[:, 2])) <= 7.5e-5 + 1.0e-10


def test_rate_limited_yaw_provider_is_call_order_deterministic() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)
    provider = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
    )

    query_times = [0.0, 120.0, 60.0]
    first_pass = {
        dt_s: provider.get_actual_yaw_state(raw_propagator, dt_s) for dt_s in query_times
    }
    second_pass = {
        dt_s: provider.get_actual_yaw_state(raw_propagator, dt_s) for dt_s in reversed(query_times)
    }

    for dt_s in query_times:
        np.testing.assert_allclose(first_pass[dt_s], second_pass[dt_s], rtol=0.0, atol=0.0)


def test_rate_limited_yaw_provider_stays_bounded_in_aggressive_low_beta_region() -> None:
    epoch, raw_propagator = _make_low_beta_propagator()
    peak_time_s = _find_low_beta_peak_time_s(epoch, raw_propagator)
    reference_epoch = Time(epoch.unix + peak_time_s, format="unix", scale="utc")
    reference_probe = _make_seeded_provider(
        raw_propagator,
        reference_epoch,
        max_yaw_rate_rad_s=10.0,
        max_yaw_acceleration_rad_s2=10.0,
        kp=0.0,
        kd=0.0,
        finite_difference_step_s=0.05,
    )
    coarse_times_s = np.linspace(-120.0, 120.0, 15)
    reference = np.vstack(
        [reference_probe.get_reference_yaw_state(raw_propagator, float(t)) for t in coarse_times_s]
    )

    rate_limit = 2.0e-3
    provider = _make_seeded_provider(
        raw_propagator,
        reference_epoch,
        max_yaw_rate_rad_s=rate_limit,
        max_yaw_acceleration_rad_s2=5.0e-3,
        kp=0.5,
        kd=2.0,
        finite_difference_step_s=0.05,
    )

    dense_times_s = np.arange(-60.0, 60.1, 8.0)
    actual = np.vstack([provider.get_actual_yaw_state(raw_propagator, float(t)) for t in dense_times_s])

    assert np.max(np.abs(reference[:, 1])) > 5.0 * rate_limit
    assert np.all(np.isfinite(actual))
    assert np.max(np.abs(actual[:, 1])) <= rate_limit + 1.0e-10

    unwrapped_yaw = np.unwrap(actual[:, 0])
    dt_s = float(dense_times_s[1] - dense_times_s[0])
    assert np.max(np.abs(np.diff(unwrapped_yaw))) <= rate_limit * dt_s * 1.1 + 1.0e-6


def test_rate_limited_yaw_provider_spin_and_acceleration_are_self_consistent() -> None:
    orbit_module._bind_orbit_java()

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)
    provider = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=10.0,
        max_yaw_acceleration_rad_s2=10.0,
        kp=0.0,
        kd=0.0,
        finite_difference_step_s=0.05,
    )
    gcrf = orbit_module.FramesFactory.getGCRF()
    epoch_date = raw_propagator.getInitialState().getDate()
    date = epoch_date.shiftedBy(60.0)
    dt_s = 0.05

    law = provider.to_orekit()
    attitude = law.getAttitude(raw_propagator, date, gcrf)
    future = law.getAttitude(raw_propagator, date.shiftedBy(dt_s), gcrf)
    past = law.getAttitude(raw_propagator, date.shiftedBy(-dt_s), gcrf)

    predicted_future = attitude.getOrientation().shiftedBy(dt_s)
    predicted_past = attitude.getOrientation().shiftedBy(-dt_s)

    assert _rotation_distance(predicted_future.getRotation(), future.getRotation()) < 2.0e-6
    assert _rotation_distance(predicted_past.getRotation(), past.getRotation()) < 2.0e-6


def test_rate_limited_yaw_provider_can_be_used_with_propagator_builders() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)
    provider = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
    )

    analytical = build_two_body_propagator(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame="gcrf",
        attitude_provider=provider,
    )
    numerical = build_numerical_propagator(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame="gcrf",
        attitude_provider=provider,
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=False,
        enable_srp=False,
    )

    analytical_state = analytical.propagate(analytical.getInitialState().getDate().shiftedBy(60.0))
    numerical_state = numerical.propagate(numerical.getInitialState().getDate().shiftedBy(60.0))

    assert analytical_state.getAttitude() is not None
    assert numerical_state.getAttitude() is not None


def test_rate_limited_yaw_provider_bulk_orbit_attitude_vectors_match_propagated_states() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)
    provider = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
    )

    propagator = build_two_body_propagator(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame="gcrf",
        attitude_provider=provider,
    )
    orbit = Orbit(propagator)

    dt_s = np.array([0.0, 120.0, 30.0, 240.0], dtype=np.float64)
    sampled = orbit.sample(dt_s, attitude_spin=True, attitude_acceleration=True)

    expected_rate = np.empty((dt_s.size, 3), dtype=np.float64)
    expected_accel = np.empty((dt_s.size, 3), dtype=np.float64)
    date0 = propagator.getInitialState().getDate()
    for i, dt in enumerate(dt_s):
        state = propagator.propagate(date0.shiftedBy(float(dt)))
        spin = state.getAttitude().getSpin()
        accel = state.getAttitude().getRotationAcceleration()
        expected_rate[i] = [float(spin.getX()), float(spin.getY()), float(spin.getZ())]
        expected_accel[i] = [float(accel.getX()), float(accel.getY()), float(accel.getZ())]

    np.testing.assert_allclose(sampled.attitude_spin_body_rad_s, expected_rate, rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(
        sampled.attitude_accel_body_rad_s2,
        expected_accel,
        rtol=0.0,
        atol=1.0e-6,
    )
