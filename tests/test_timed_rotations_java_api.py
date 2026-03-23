from __future__ import annotations

import numpy as np
import pytest

pytestmark = [
    pytest.mark.filterwarnings("ignore::erfa.core.ErfaWarning"),
    pytest.mark.filterwarnings(
        "ignore:Tried to get polar motions for times after IERS data is valid.*:astropy.utils.exceptions.AstropyWarning"
    ),
]

astropy_time = pytest.importorskip("astropy.time")
astropy_coordinates = pytest.importorskip("astropy.coordinates")
u = pytest.importorskip("astropy.units")

from nstk.time_utils import astropy_time_to_orekit_date
from nstk.transforms import transform

Time = astropy_time.Time
GCRS = astropy_coordinates.GCRS
ITRS = astropy_coordinates.ITRS
CartesianRepresentation = astropy_coordinates.CartesianRepresentation


def _random_states(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    d = rng.standard_normal((n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    rmag = rng.uniform(6.6e6, 4.2e7, size=n)
    r = d * rmag[:, None]

    v = rng.standard_normal((n, 3))
    v *= 7800.0 / np.linalg.norm(v, axis=1, keepdims=True)

    a = rng.standard_normal((n, 3)) * 1.0e-3
    return r.astype(np.float64), v.astype(np.float64), a.astype(np.float64)


def _astropy_transform_pos_gcrf_to_itrf(r_m: np.ndarray, times: Time) -> np.ndarray:
    rep = CartesianRepresentation(r_m[:, 0] * u.m, r_m[:, 1] * u.m, r_m[:, 2] * u.m)
    c0 = GCRS(rep, obstime=times)
    c1 = c0.transform_to(ITRS(obstime=times))
    return c1.cartesian.xyz.to_value(u.m).T.astype(np.float64)


def _latest_itrf_version():
    from org.orekit.frames import ITRFVersion

    years = sorted(
        int(name.removeprefix("ITRF_"))
        for name in dir(ITRFVersion)
        if name.startswith("ITRF_") and name.removeprefix("ITRF_").isdigit()
    )
    return getattr(ITRFVersion, f"ITRF_{years[-1]}")


def _latest_iers_convention():
    from org.orekit.utils import IERSConventions

    years = sorted(
        int(name.removeprefix("IERS_"))
        for name in dir(IERSConventions)
        if name.startswith("IERS_") and name.removeprefix("IERS_").isdigit()
    )
    return getattr(IERSConventions, f"IERS_{years[-1]}")


def test_transform_identity_with_broadcast_scalar_time() -> None:
    r, v, a = _random_states(8, seed=11)
    t = Time("2026-01-01T00:00:00", scale="utc")

    p_out, v_out, a_out = transform(
        from_frame="gcrf",
        to_frame="gcrf",
        time=t,
        position=r,
        velocity=v,
        acceleration=a,
    )

    assert p_out.shape == (8, 3)
    assert v_out is not None and v_out.shape == (8, 3)
    assert a_out is not None and a_out.shape == (8, 3)
    np.testing.assert_allclose(p_out, r, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(v_out, v, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(a_out, a, atol=1e-12, rtol=0.0)


def test_transform_pos_matches_astropy_and_pos_vel_roundtrip() -> None:
    n = 20
    r, v, _ = _random_states(n, seed=12)
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    times = Time(
        t0.unix + np.linspace(0.0, 2.0 * 3600.0, n, dtype=np.float64),
        format="unix",
        scale="utc",
    )

    p_new, v_new, a_new = transform(
        from_frame="gcrf",
        to_frame="itrf",
        time=times,
        position=r,
        velocity=v,
    )
    p_ast = _astropy_transform_pos_gcrf_to_itrf(r, times)
    err = np.linalg.norm(p_new - p_ast, axis=1)

    assert a_new is None
    assert float(np.max(err)) < 120.0
    assert float(np.mean(err)) < 40.0

    p_back, v_back, a_back = transform(
        from_frame="itrf",
        to_frame="gcrf",
        time=times,
        position=p_new,
        velocity=v_new,
    )
    assert a_back is None
    np.testing.assert_allclose(p_back, r, atol=2e-6, rtol=0.0)
    np.testing.assert_allclose(v_back, v, atol=2e-9, rtol=0.0)


def test_transform_accepts_absolutedate_vectors() -> None:
    n = 7
    r, _, _ = _random_states(n, seed=13)
    epoch = Time("2026-02-01T12:00:00", scale="utc")
    base = astropy_time_to_orekit_date(epoch)
    dates = [base.shiftedBy(float(dt)) for dt in np.linspace(0.0, 180.0, n)]

    p_abs, _, _ = transform(
        from_frame="teme",
        to_frame="mod",
        time=dates,
        position=r,
    )

    t_ast = Time(epoch.unix + np.linspace(0.0, 180.0, n), format="unix", scale="utc")
    p_ast, _, _ = transform(
        from_frame="teme",
        to_frame="mod",
        time=t_ast,
        position=r,
    )

    np.testing.assert_allclose(p_abs, p_ast, atol=1e-8, rtol=0.0)


def test_transform_quantity_and_acceleration_without_velocity() -> None:
    n = 5
    r, _, a = _random_states(n, seed=14)
    t0 = Time("2026-03-01T00:00:00", scale="utc")
    times = Time(t0.unix + np.arange(n, dtype=np.float64), format="unix", scale="utc")

    p_q, v_none, a_q = transform(
        from_frame="gcrf",
        to_frame="itrf",
        time=times,
        position=r * u.km,
        acceleration=a * (u.km / (u.s**2)),
    )

    assert v_none is None
    assert hasattr(p_q, "unit") and p_q.unit == u.m
    assert hasattr(a_q, "unit") and a_q.unit == (u.m / (u.s**2))
    assert p_q.shape == (n, 3)
    assert a_q.shape == (n, 3)
    assert np.all(np.isfinite(p_q.to_value(u.m)))
    assert np.all(np.isfinite(a_q.to_value(u.m / (u.s**2))))


def test_transform_scalar_position_and_scalar_time_shape() -> None:
    t = Time("2026-03-01T00:00:00", scale="utc")
    p = np.array([7000e3, 0.0, 0.0], dtype=np.float64)

    p_out, v_out, a_out = transform("gcrf", "itrf", t, p)

    assert p_out.shape == (3,)
    assert v_out is None
    assert a_out is None


def test_transform_accepts_versioned_itrf_string_aliases() -> None:
    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.frames import FramesFactory, ITRFVersion
    from org.orekit.utils import IERSConventions

    t = Time("2026-03-01T00:00:00", scale="utc")
    p, v, _ = _random_states(4, seed=15)
    itrf2014 = FramesFactory.getITRF(
        ITRFVersion.ITRF_2014,
        IERSConventions.IERS_2010,
        True,
    )

    p_ref, v_ref, _ = transform("gcrf", itrf2014, t, p, velocity=v)

    for name in ("itrf2014", "ITRF_2014", "itrs2014", "ecef2014"):
        p_out, v_out, a_out = transform("gcrf", name, t, p, velocity=v)

        assert a_out is None
        np.testing.assert_allclose(p_out, p_ref, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(v_out, v_ref, atol=1e-12, rtol=0.0)


def test_transform_uses_latest_itrf_when_no_version_is_specified() -> None:
    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.frames import FramesFactory

    t = Time("2026-03-01T00:00:00", scale="utc")
    p, v, _ = _random_states(4, seed=18)
    iers = _latest_iers_convention()
    latest_itrf = FramesFactory.getITRF(_latest_itrf_version(), iers, True)
    p_ref, v_ref, _ = transform("gcrf", latest_itrf, t, p, velocity=v)

    for name in ("itrf", "itrs", "ecef"):
        p_out, v_out, a_out = transform("gcrf", name, t, p, velocity=v)

        assert a_out is None
        np.testing.assert_allclose(p_out, p_ref, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(v_out, v_ref, atol=1e-12, rtol=0.0)


def test_transform_rejects_unknown_itrf_version_string() -> None:
    t = Time("2026-03-01T00:00:00", scale="utc")
    p = np.array([7000e3, 0.0, 0.0], dtype=np.float64)

    with pytest.raises(ValueError, match="Supported ITRF versions"):
        transform("gcrf", "itrf2099", t, p)


def test_transform_accepts_iers_versioned_frame_shorthands() -> None:
    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.frames import FramesFactory
    from org.orekit.utils import IERSConventions

    t = Time("2026-03-01T00:00:00", scale="utc")
    p, v, _ = _random_states(4, seed=16)

    cases = [
        ("tod2010", FramesFactory.getTOD(IERSConventions.IERS_2010, True)),
        ("mod2003", FramesFactory.getMOD(IERSConventions.IERS_2003)),
        ("cirf1996", FramesFactory.getCIRF(IERSConventions.IERS_1996, True)),
        ("gtod2010", FramesFactory.getGTOD(IERSConventions.IERS_2010, True)),
        ("tirf2003", FramesFactory.getTIRF(IERSConventions.IERS_2003, True)),
        ("ecliptic1996", FramesFactory.getEcliptic(IERSConventions.IERS_1996)),
        ("itrfcio2010", FramesFactory.getITRF(IERSConventions.IERS_2010, True)),
        (
            "itrfequinox2003",
            FramesFactory.getITRFEquinox(IERSConventions.IERS_2003, True),
        ),
    ]

    for name, frame in cases:
        p_out, v_out, a_out = transform("gcrf", name, t, p, velocity=v)
        p_ref, v_ref, a_ref = transform("gcrf", frame, t, p, velocity=v)

        assert a_out is None
        assert a_ref is None
        np.testing.assert_allclose(p_out, p_ref, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(v_out, v_ref, atol=1e-12, rtol=0.0)


def test_transform_uses_latest_iers_convention_when_no_year_is_specified() -> None:
    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.frames import FramesFactory

    t = Time("2026-03-01T00:00:00", scale="utc")
    p, v, _ = _random_states(4, seed=19)
    iers = _latest_iers_convention()

    cases = [
        ("tod", FramesFactory.getTOD(iers, True)),
        ("mod", FramesFactory.getMOD(iers)),
        ("cirf", FramesFactory.getCIRF(iers, True)),
        ("gtod", FramesFactory.getGTOD(iers, True)),
        ("tirf", FramesFactory.getTIRF(iers, True)),
        ("ecliptic", FramesFactory.getEcliptic(iers)),
        ("itrfcio", FramesFactory.getITRF(iers, True)),
        ("itrfequinox", FramesFactory.getITRFEquinox(iers, True)),
    ]

    for name, frame in cases:
        p_out, v_out, a_out = transform("gcrf", name, t, p, velocity=v)
        p_ref, v_ref, a_ref = transform("gcrf", frame, t, p, velocity=v)

        assert a_out is None
        assert a_ref is None
        np.testing.assert_allclose(p_out, p_ref, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(v_out, v_ref, atol=1e-12, rtol=0.0)


def test_transform_accepts_orekit_predefined_frame_names() -> None:
    from nstk._orekit_runtime import ensure_orekit_runtime

    ensure_orekit_runtime()

    from org.orekit.frames import FramesFactory, Predefined

    t = Time("2026-03-01T00:00:00", scale="utc")
    p, v, _ = _random_states(4, seed=17)

    cases = [
        "TOD_CONVENTIONS_2010_SIMPLE_EOP",
        "GTOD_WITHOUT_EOP_CORRECTIONS",
        "ITRF_EQUINOX_CONV_2003_ACCURATE_EOP",
    ]

    for name in cases:
        frame = FramesFactory.getFrame(getattr(Predefined, name))
        p_out, v_out, a_out = transform("gcrf", name, t, p, velocity=v)
        p_ref, v_ref, a_ref = transform("gcrf", frame, t, p, velocity=v)

        assert a_out is None
        assert a_ref is None
        np.testing.assert_allclose(p_out, p_ref, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(v_out, v_ref, atol=1e-12, rtol=0.0)
