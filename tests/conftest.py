from __future__ import annotations

import numpy as np
import pytest
from astropy.time import Time


def pytest_configure(config) -> None:
    try:
        from astropy.utils import iers

        iers.conf.auto_download = False
        iers.conf.remote_timeout = 1
        iers.conf.auto_max_age = None
        iers.conf.iers_degraded_accuracy = "warn"
    except Exception:
        pass

    config.addinivalue_line(
        "markers",
        "slow: marks slower Orekit/JPype propagation checks",
    )


@pytest.fixture(scope="session")
def epoch_utc() -> Time:
    return Time("2025-01-01T00:00:00", scale="utc")


@pytest.fixture(scope="session")
def leo_case() -> dict[str, float]:
    return {
        "a_m": 7000e3,
        "e": 1e-6,
        "i_rad": float(np.deg2rad(25.0)),
        "raan_rad": 0.0,
        "argp_rad": 0.0,
        "anomaly_rad": 0.0,
    }
