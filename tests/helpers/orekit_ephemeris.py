from __future__ import annotations

import numpy as np
from astropy.time import Time

from nstk.propagation.orbit import Orbit, _astropy_to_absdate_utc


def stats(err: np.ndarray) -> dict[str, float]:
    x = np.asarray(err, dtype=np.float64)
    return {
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
    }


def direct_pv_from_propagator(
    ephem_obj: Orbit, t: Time, *, frame: str
) -> tuple[np.ndarray, np.ndarray]:
    abs_t = _astropy_to_absdate_utc(t)
    st = ephem_obj.propagator.propagate(abs_t)  # type: ignore[attr-defined]

    if frame == "native":
        fr = ephem_obj._frame_native  # type: ignore[attr-defined]
        pv = st.getPVCoordinates(fr)
    elif frame == "itrf":
        fr = ephem_obj._itrf  # type: ignore[attr-defined]
        pv = st.getPVCoordinates(fr)
    else:
        raise ValueError("frame must be 'native' or 'itrf'")

    r = np.asarray(pv.getPosition().toArray(), dtype=np.float64)
    v = np.asarray(pv.getVelocity().toArray(), dtype=np.float64)
    return r, v


def make_time_grid(epoch: Time, *, t_min_s: float, t_max_s: float, n: int) -> Time:
    import astropy.units as u

    secs = np.linspace(float(t_min_s), float(t_max_s), int(n), dtype=np.float64) * u.s
    return epoch + secs
