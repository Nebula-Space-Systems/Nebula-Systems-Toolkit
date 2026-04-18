from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from astropy.time import Time

import nstk.propagation.orbit as orbit_module
from nstk.propagation import (
    Orbit,
    RateLimitedYawSteeringProvider,
    build_ideal_nadir_sun_constrained_attitude_provider,
    build_nadir_sun_constrained_attitude_provider,
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


def _wrap_minus_pi_to_pi(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _reconstruct_yaw_orientation(date, base_orientation, psi: float, omega: float, alpha: float):
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Rotation, RotationConvention, Vector3D
    from org.orekit.utils import TimeStampedAngularCoordinates

    yaw_offset = TimeStampedAngularCoordinates(
        date,
        Rotation(Vector3D.PLUS_K, float(psi), RotationConvention.FRAME_TRANSFORM),
        Vector3D(0.0, 0.0, float(omega)),
        Vector3D(0.0, 0.0, float(alpha)),
    )
    return yaw_offset.addOffset(base_orientation)


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


def test_relative_yaw_reconstruction_for_non_identity_base_attitude() -> None:
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Rotation, RotationConvention, Vector3D
    from org.orekit.time import AbsoluteDate
    from org.orekit.utils import AngularCoordinates

    date = AbsoluteDate.ARBITRARY_EPOCH
    base_orientation = AngularCoordinates(
        Rotation(Vector3D.PLUS_I, 0.37, RotationConvention.FRAME_TRANSFORM),
        Vector3D.ZERO,
        Vector3D.ZERO,
    )

    for known_yaw in (0.32, -0.41):
        target_orientation = _reconstruct_yaw_orientation(
            date,
            base_orientation,
            known_yaw,
            0.0,
            0.0,
        )
        extracted = RateLimitedYawSteeringProvider.extract_relative_yaw(
            base_orientation.getRotation(),
            target_orientation.getRotation(),
        )
        reconstructed = _reconstruct_yaw_orientation(
            date,
            base_orientation,
            extracted,
            0.0,
            0.0,
        )
        assert extracted == pytest.approx(known_yaw, abs=1.0e-15)
        assert _rotation_distance(
            reconstructed.getRotation(),
            target_orientation.getRotation(),
        ) < 1.0e-14


def test_zero_offset_reconstruction_returns_base_attitude_exactly() -> None:
    orbit_module._bind_orbit_java()

    from org.hipparchus.geometry.euclidean.threed import Rotation, RotationConvention, Vector3D
    from org.orekit.time import AbsoluteDate
    from org.orekit.utils import AngularCoordinates

    date = AbsoluteDate.ARBITRARY_EPOCH
    base_orientation = AngularCoordinates(
        Rotation(Vector3D.PLUS_J, -0.28, RotationConvention.FRAME_TRANSFORM),
        Vector3D.ZERO,
        Vector3D.ZERO,
    )
    reconstructed = _reconstruct_yaw_orientation(date, base_orientation, 0.0, 0.0, 0.0)

    assert _rotation_distance(reconstructed.getRotation(), base_orientation.getRotation()) < 1.0e-15
    assert float(reconstructed.getRotationRate().getNorm()) == pytest.approx(0.0, abs=1.0e-15)
    assert float(reconstructed.getRotationAcceleration().getNorm()) == pytest.approx(0.0, abs=1.0e-15)


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


def test_ideal_reference_reconstruction_matches_ideal_yaw_steering() -> None:
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
    provider = RateLimitedYawSteeringProvider(
        inertial_frame="gcrf",
        reference_epoch=epoch,
        max_yaw_rate_rad_s=10.0,
        max_yaw_acceleration_rad_s2=10.0,
        kp=0.0,
        kd=0.0,
        finite_difference_step_s=0.05,
    )
    epoch_date = raw_propagator.getInitialState().getDate()

    for dt_s in (0.0, 60.0, 120.0, 240.0):
        date = epoch_date.shiftedBy(float(dt_s))
        base_attitude = base_law.getAttitude(raw_propagator, date, gcrf)
        ideal_attitude = ideal_law.getAttitude(raw_propagator, date, gcrf)
        psi_ref, omega_ref, alpha_ref = provider.get_reference_yaw_state(raw_propagator, float(dt_s))
        reconstructed = _reconstruct_yaw_orientation(
            date,
            base_attitude.getOrientation(),
            float(psi_ref),
            float(omega_ref),
            float(alpha_ref),
        )

        assert _rotation_distance(reconstructed.getRotation(), ideal_attitude.getRotation()) < 2.0e-6
        spin_delta = reconstructed.getRotationRate().subtract(ideal_attitude.getSpin())
        accel_delta = reconstructed.getRotationAcceleration().subtract(
            ideal_attitude.getRotationAcceleration()
        )
        assert float(spin_delta.getNorm()) < 2.0e-5
        assert float(accel_delta.getNorm()) < 5.0e-4


def test_relative_yaw_rate_and_acceleration_match_scalar_finite_differences() -> None:
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
    epoch_date = raw_propagator.getInitialState().getDate()
    h = 0.05

    for dt_s in (0.0, 60.0, 120.0, 240.0):
        date = epoch_date.shiftedBy(float(dt_s))
        base_attitude = base_law.getAttitude(raw_propagator, date, gcrf)
        ideal_attitude = ideal_law.getAttitude(raw_propagator, date, gcrf)
        relative = ideal_attitude.getOrientation().subtractOffset(base_attitude.getOrientation())
        direct_rate = relative.getRotationRate()
        direct_accel = relative.getRotationAcceleration()

        def scalar_yaw(offset_s: float) -> float:
            shifted = epoch_date.shiftedBy(float(dt_s + offset_s))
            base_shifted = base_law.getAttitude(raw_propagator, shifted, gcrf)
            ideal_shifted = ideal_law.getAttitude(raw_propagator, shifted, gcrf)
            return RateLimitedYawSteeringProvider.extract_relative_yaw(
                base_shifted.getRotation(),
                ideal_shifted.getRotation(),
            )

        psi_m = scalar_yaw(-h)
        psi_0 = scalar_yaw(0.0)
        psi_p = scalar_yaw(h)
        psi_m_unwrapped = psi_0 - _wrap_minus_pi_to_pi(psi_0 - psi_m)
        psi_p_unwrapped = psi_0 + _wrap_minus_pi_to_pi(psi_p - psi_0)
        omega_fd = (psi_p_unwrapped - psi_m_unwrapped) / (2.0 * h)
        alpha_fd = (psi_p_unwrapped - 2.0 * psi_0 + psi_m_unwrapped) / (h * h)

        assert float(np.hypot(direct_rate.getX(), direct_rate.getY())) < 1.0e-8
        assert float(np.hypot(direct_accel.getX(), direct_accel.getY())) < 1.0e-6
        assert float(direct_rate.getZ()) == pytest.approx(omega_fd, abs=2.0e-5)
        assert float(direct_accel.getZ()) == pytest.approx(alpha_fd, abs=3.0e-3)


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


def test_rate_limited_yaw_provider_cache_matches_uncached_for_negative_and_shuffled_queries() -> None:
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

    query_times = np.array([120.0, -30.0, 0.0, 240.0, -60.0, 60.0], dtype=np.float64)
    cached_states = np.vstack([cached.get_actual_yaw_state(raw_propagator, float(t)) for t in query_times])
    uncached_states = np.vstack(
        [uncached.get_actual_yaw_state(raw_propagator, float(t)) for t in reversed(query_times)]
    )[::-1]
    np.testing.assert_allclose(cached_states, uncached_states, rtol=0.0, atol=1.0e-5)


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


def test_rate_limited_yaw_provider_is_deterministic_for_repeated_shuffled_queries() -> None:
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
    query_times = np.linspace(-120.0, 240.0, 13)
    baseline = {
        float(dt_s): provider.get_actual_yaw_state(raw_propagator, float(dt_s)) for dt_s in query_times
    }
    rng = np.random.default_rng(12345)

    for _ in range(5):
        for dt_s in rng.permutation(query_times):
            actual = provider.get_actual_yaw_state(raw_propagator, float(dt_s))
            np.testing.assert_allclose(
                actual,
                baseline[float(dt_s)],
                rtol=0.0,
                atol=0.0,
            )


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
    assert np.max(np.abs(np.diff(actual[:, 1]))) <= provider.max_yaw_acceleration_rad_s2 * dt_s * 1.2 + 1.0e-6


def test_rate_limited_yaw_provider_spin_and_acceleration_are_self_consistent() -> None:
    orbit_module._bind_orbit_java()

    from org.orekit.utils import AngularCoordinates

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

    estimated_spin = AngularCoordinates.estimateRate(
        past.getRotation(),
        future.getRotation(),
        2.0 * dt_s,
    )
    spin_fd = future.getSpin().subtract(past.getSpin()).scalarMultiply(1.0 / (2.0 * dt_s))
    attitude_spin = attitude.getSpin()
    attitude_accel = attitude.getRotationAcceleration()

    np.testing.assert_allclose(
        np.asarray(attitude_spin.toArray(), dtype=np.float64),
        np.asarray(estimated_spin.toArray(), dtype=np.float64),
        rtol=0.0,
        atol=5.0e-5,
    )
    np.testing.assert_allclose(
        np.asarray(attitude_accel.toArray(), dtype=np.float64),
        np.asarray(spin_fd.toArray(), dtype=np.float64),
        rtol=0.0,
        atol=5.0e-3,
    )


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


def test_rate_limited_yaw_provider_is_deterministic_with_numerical_pv_provider() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = build_numerical_propagator(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(20.0),
        argp=np.deg2rad(15.0),
        anomaly=np.deg2rad(10.0),
        inertial_frame="gcrf",
        gravity_degree=8,
        gravity_order=8,
        enable_drag=False,
        enable_third_body=False,
        enable_srp=False,
    )
    provider = _make_seeded_provider(
        raw_propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
    )
    query_times = np.array([120.0, 0.0, 60.0, 180.0, -30.0], dtype=np.float64)
    forward = np.vstack([provider.get_actual_yaw_state(raw_propagator, float(t)) for t in query_times])
    reverse = np.vstack(
        [provider.get_actual_yaw_state(raw_propagator, float(t)) for t in reversed(query_times)]
    )[::-1]
    np.testing.assert_allclose(forward, reverse, rtol=0.0, atol=1.0e-14)


def test_rate_limited_yaw_wrapper_normalizes_phasing_axis_inputs() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    raw_propagator = _make_two_body_propagator(epoch)
    axis_string = RateLimitedYawSteeringProvider(
        inertial_frame="gcrf",
        reference_epoch=epoch,
        phasing_axis="x",
        finite_difference_step_s=0.05,
    )
    axis_vector = RateLimitedYawSteeringProvider(
        inertial_frame="gcrf",
        reference_epoch=epoch,
        phasing_axis=[4.0, 0.0, 0.0],
        finite_difference_step_s=0.05,
    )
    np.testing.assert_allclose(
        axis_string.get_reference_yaw_state(raw_propagator, 60.0),
        axis_vector.get_reference_yaw_state(raw_propagator, 60.0),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_ideal_nadir_sun_constrained_helper_matches_compatibility_alias() -> None:
    ideal = build_ideal_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")
    compat = build_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")
    assert ideal.getClass().getName() == compat.getClass().getName()


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


def test_bridge_pv_proxy_uses_orbit_only_bridge_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jpype

    monkeypatch.setattr(orbit_module, "_bind_orbit_java", lambda: None)
    monkeypatch.setattr(jpype, "JImplements", lambda *args, **kwargs: (lambda cls: cls))
    monkeypatch.setattr(jpype, "JOverride", lambda func=None, **kwargs: func)
    monkeypatch.setattr(orbit_module, "Vector3D", lambda x, y, z: (float(x), float(y), float(z)))

    class _FakeTimeStampedPVCoordinates:
        def __init__(self, date, position, velocity):
            self.date = date
            self.position = position
            self.velocity = velocity

    monkeypatch.setattr(orbit_module, "TimeStampedPVCoordinates", _FakeTimeStampedPVCoordinates)

    class _FakeBridge:
        def __init__(self):
            self.calls: list[tuple[str, float, object, bool]] = []

        def queryOrbitOnlyPosition(self, dt_s, frame, strict):
            self.calls.append(("pos", float(dt_s), frame, bool(strict)))
            return [1.0, 2.0, 3.0]

        def queryOrbitOnlyVelocity(self, dt_s, frame, strict):
            self.calls.append(("vel", float(dt_s), frame, bool(strict)))
            return [4.0, 5.0, 6.0]

        def queryOrbitOnlyPV(self, dt_s, frame, strict):
            self.calls.append(("pv", float(dt_s), frame, bool(strict)))
            return SimpleNamespace(p=[7.0, 8.0, 9.0], v=[10.0, 11.0, 12.0])

        def queryPosition(self, *args, **kwargs):
            raise AssertionError("generic queryPosition should not be used")

        def queryVelocity(self, *args, **kwargs):
            raise AssertionError("generic queryVelocity should not be used")

        def queryPV(self, *args, **kwargs):
            raise AssertionError("generic queryPV should not be used")

    class _FakeDate:
        def __init__(self, dt_s: float):
            self._dt_s = float(dt_s)

        def durationFrom(self, epoch):
            return self._dt_s

    bridge = _FakeBridge()
    orbit = SimpleNamespace(_bridge=bridge, _epoch_orekit=object())
    proxy = orbit_module._build_bridge_pv_provider_proxy(orbit)
    date = _FakeDate(42.0)
    frame = object()

    assert proxy.getPosition(date, frame) == (1.0, 2.0, 3.0)
    assert proxy.getVelocity(date, frame) == (4.0, 5.0, 6.0)
    pv = proxy.getPVCoordinates(date, frame)
    assert pv.position == (7.0, 8.0, 9.0)
    assert pv.velocity == (10.0, 11.0, 12.0)
    assert bridge.calls == [
        ("pos", 42.0, frame, True),
        ("vel", 42.0, frame, True),
        ("pv", 42.0, frame, True),
    ]


def test_bridge_pv_proxy_orbit_only_helpers_respect_strict_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jpype

    monkeypatch.setattr(orbit_module, "_bind_orbit_java", lambda: None)
    monkeypatch.setattr(jpype, "JImplements", lambda *args, **kwargs: (lambda cls: cls))
    monkeypatch.setattr(jpype, "JOverride", lambda func=None, **kwargs: func)
    monkeypatch.setattr(orbit_module, "Vector3D", lambda x, y, z: (float(x), float(y), float(z)))
    monkeypatch.setattr(
        orbit_module,
        "TimeStampedPVCoordinates",
        lambda date, position, velocity: SimpleNamespace(
            date=date,
            position=position,
            velocity=velocity,
        ),
    )

    class _FailingBridge:
        def queryOrbitOnlyPosition(self, dt_s, frame, strict):
            if strict:
                raise RuntimeError("position failure")
            return None

        def queryOrbitOnlyVelocity(self, dt_s, frame, strict):
            if strict:
                raise RuntimeError("velocity failure")
            return None

        def queryOrbitOnlyPV(self, dt_s, frame, strict):
            if strict:
                raise RuntimeError("pv failure")
            return None

    class _FakeDate:
        def durationFrom(self, epoch):
            return 12.0

    orbit = SimpleNamespace(_bridge=_FailingBridge(), _epoch_orekit=object())
    proxy = orbit_module._build_bridge_pv_provider_proxy(orbit)
    date = _FakeDate()
    frame = object()

    with pytest.raises(RuntimeError, match="position failure"):
        proxy._query_orbit_only_position(date, frame, strict=True)
    assert proxy._query_orbit_only_position(date, frame, strict=False) is None

    with pytest.raises(RuntimeError, match="velocity failure"):
        proxy._query_orbit_only_velocity(date, frame, strict=True)
    assert proxy._query_orbit_only_velocity(date, frame, strict=False) is None

    with pytest.raises(RuntimeError, match="pv failure"):
        proxy._query_orbit_only_pv(date, frame, strict=True)
    assert proxy._query_orbit_only_pv(date, frame, strict=False) is None


def test_orbit_set_attitude_provider_updates_sampling_without_manual_cache_clear() -> None:
    orbit_module._bind_orbit_java()

    epoch = Time("2026-01-01T00:00:00", scale="utc")
    propagator = _make_two_body_propagator(epoch)
    orbit = Orbit(propagator, should_cache=True)
    gcrf = orbit_module.FramesFactory.getGCRF()

    orbit.precompute(0.0, 900.0)
    before = orbit.get_attitude_matrix(600.0, reference_frame=gcrf)

    provider = build_ideal_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")
    orbit.set_attitude_provider(provider)
    after = orbit.get_attitude_matrix(600.0, reference_frame=gcrf)

    assert not np.allclose(before, after, rtol=0.0, atol=1.0e-10)

    state = orbit.propagator.propagate(orbit.propagator.getInitialState().getDate().shiftedBy(600.0))
    expected = np.asarray(
        state.getAttitude().withReferenceFrame(gcrf).getRotation().getMatrix(),
        dtype=np.float64,
    ).reshape(1, 3, 3)
    np.testing.assert_allclose(after, expected, rtol=0.0, atol=1.0e-12)


def test_orbit_set_attitude_provider_accepts_raw_orekit_provider_and_getter_returns_it() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = Orbit(_make_two_body_propagator(epoch))
    raw_provider = build_ideal_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")

    orbit.set_attitude_provider(raw_provider)
    installed = orbit.get_attitude_provider()

    assert installed == raw_provider


def test_orbit_set_attitude_provider_accepts_wrapper_provider() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    propagator = _make_two_body_propagator(epoch)
    orbit = Orbit(propagator)
    wrapper = _make_seeded_provider(
        orbit.propagator,
        epoch,
        max_yaw_rate_rad_s=0.05,
        max_yaw_acceleration_rad_s2=0.01,
        kp=0.5,
        kd=1.5,
        finite_difference_step_s=0.05,
    )

    orbit.set_attitude_provider(wrapper)

    installed = orbit.get_attitude_provider()
    expected = wrapper.to_orekit(pv_provider=orbit.propagator)
    assert installed.getClass().getName() == expected.getClass().getName()

    sampled = orbit.get_attitude_spin(np.array([0.0, 60.0, 120.0], dtype=np.float64))
    assert sampled.shape == (3, 3)


def test_orbit_set_attitude_provider_binds_wrapper_to_orbit_propagator() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = Orbit(_make_two_body_propagator(epoch))
    raw_provider = build_ideal_nadir_sun_constrained_attitude_provider(inertial_frame="gcrf")
    calls: list[object] = []

    class _TrackingWrapper:
        def to_orekit(self, *, pv_provider=None):
            calls.append(pv_provider)
            return raw_provider

    wrapper = _TrackingWrapper()
    orbit.set_attitude_provider(wrapper)

    assert calls == [orbit.propagator]
    assert orbit.get_attitude_provider() == raw_provider


@pytest.mark.parametrize(
    ("spin_requested", "accel_requested", "bad_length", "match_text"),
    [
        (True, False, 5, "attitude spin"),
        (False, True, 4, "attitude acceleration"),
    ],
)
def test_custom_attitude_vector_shape_validation_raises_clear_error(
    spin_requested: bool,
    accel_requested: bool,
    bad_length: int,
    match_text: str,
) -> None:
    class _BadProvider:
        def sampleBodyAngularVectors(
            self,
            propagator,
            epoch,
            dt_s,
            attitude_reference_frame,
            include_acceleration,
        ):
            return np.arange(bad_length, dtype=np.float64)

    class _FakePropagator:
        def __init__(self, provider):
            self._provider = provider

        def getAttitudeProvider(self):
            return self._provider

    orbit = Orbit.__new__(Orbit)
    orbit.propagator = _FakePropagator(_BadProvider())
    orbit._epoch_orekit = object()

    with pytest.raises(
        ValueError,
        match=rf"{match_text}.*expected exactly 9.*3 samples",
    ):
        orbit._sample_custom_attitude_vectors(
            np.array([0.0, 60.0, 120.0], dtype=np.float64),
            attitude_reference_frame=object(),
            attitude_spin=spin_requested,
            attitude_acceleration=accel_requested,
        )


def test_mixed_cartesian_and_custom_attitude_vectors_use_full_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    real_sampler = Orbit._sample_custom_attitude_vectors
    calls = {"count": 0}

    def _tracking_sampler(self, *args, **kwargs):
        calls["count"] += 1
        return real_sampler(self, *args, **kwargs)

    monkeypatch.setattr(Orbit, "_sample_custom_attitude_vectors", _tracking_sampler)

    dt_s = np.array([0.0, 60.0, 120.0], dtype=np.float64)
    sampled = orbit.sample(dt_s, position=True, attitude_spin=True)

    assert calls["count"] == 0
    np.testing.assert_allclose(sampled.position_m, orbit.get_position(dt_s), rtol=0.0, atol=1.0e-8)
    np.testing.assert_allclose(
        sampled.attitude_spin_body_rad_s,
        orbit.get_attitude_spin(dt_s),
        rtol=0.0,
        atol=1.0e-6,
    )
