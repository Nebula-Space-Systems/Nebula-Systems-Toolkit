from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.time import Time

from nebula.time_utils import (
    astropy_time_to_orekit_date,
    normalize_time_to_epoch_seconds,
    orekit_date_to_astropy_time,
)


def test_astropy_orekit_roundtrip() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    date = astropy_time_to_orekit_date(epoch)
    epoch_back = orekit_date_to_astropy_time(date)

    dt_s = float((epoch_back - epoch).to_value(u.s))
    assert abs(dt_s) < 1.0e-9


def test_normalize_time_to_epoch_seconds_accepts_absolutedate() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    base = astropy_time_to_orekit_date(epoch)
    dates = [base.shiftedBy(0.0), base.shiftedBy(60.0), base.shiftedBy(120.0)]

    dt_dates, is_scalar_dates = normalize_time_to_epoch_seconds(dates, epoch)
    dt_seconds, is_scalar_seconds = normalize_time_to_epoch_seconds(
        np.array([0.0, 60.0, 120.0], dtype=np.float64),
        epoch,
    )

    assert is_scalar_dates is False
    assert is_scalar_seconds is False
    np.testing.assert_allclose(dt_dates, dt_seconds, atol=1.0e-12, rtol=0.0)
