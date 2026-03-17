from __future__ import annotations

import numpy as np
import pytest

import nebula.transforms as transforms
from nebula.transforms import transform

pytestmark = pytest.mark.filterwarnings(
    "ignore:Tried to get polar motions for times after IERS data is valid.*:astropy.utils.exceptions.AstropyWarning"
)

astropy_time = pytest.importorskip("astropy.time")

Time = astropy_time.Time


def test_transform_broadcasts_scalar_time_over_state_rows() -> None:
    t = Time("2026-01-01T00:00:00", scale="utc")
    r = np.array(
        [
            [7000e3, 0.0, 0.0],
            [6990e3, 80e3, 10e3],
            [6970e3, 160e3, 20e3],
        ],
        dtype=np.float64,
    )

    p_out, v_out, a_out = transform("gcrf", "itrf", t, r)

    assert p_out.shape == (3, 3)
    assert v_out is None
    assert a_out is None


def test_transform_broadcasts_scalar_state_over_time_array() -> None:
    t0 = Time("2026-01-01T00:00:00", scale="utc")
    times = Time(t0.unix + np.arange(4, dtype=np.float64), format="unix", scale="utc")
    p = np.array([7000e3, 0.0, 0.0], dtype=np.float64)

    p_out, v_out, a_out = transform("gcrf", "itrf", times, p)

    assert p_out.shape == (4, 3)
    assert v_out is None
    assert a_out is None


def test_transform_namespace_exports_only_new_timed_api() -> None:
    assert callable(transforms.transform)
    assert callable(transforms.initialize_timed_rotations)
    assert not hasattr(transforms, "transform_timed")
    assert not hasattr(transforms, "transform_positions_timed")
    assert not hasattr(transforms, "transform_pos_vel_timed")
