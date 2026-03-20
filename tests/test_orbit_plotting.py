from __future__ import annotations

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import astropy.units as u
import numpy as np
from astropy.time import Time
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from nstk.plotting import plot_orbits
from nstk.propagation.orbit import Orbit


def _make_orbit(*, epoch: Time, raan_deg: float, anomaly_deg: float) -> Orbit:
    return Orbit.from_kepler_two_body(
        epoch=epoch,
        a=7000e3,
        e=0.001,
        i=np.deg2rad(53.0),
        raan=np.deg2rad(raan_deg),
        argp=np.deg2rad(20.0),
        anomaly=np.deg2rad(anomaly_deg),
    )


def test_plot_orbits_multi_orbit_3d_defaults() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit_a = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)
    orbit_b = _make_orbit(epoch=epoch, raan_deg=45.0, anomaly_deg=120.0)

    fig, ax = plot_orbits(
        [orbit_a, orbit_b],
        labels=["Alpha", "Bravo"],
        view="3d",
        samples=96,
        show=False,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert ax.name == "3d"

    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Alpha", "Bravo"]
    assert len(ax.collections) >= 6
    assert any("Keplerian period" in text.get_text() for text in fig.texts)

    plt.close(fig)


def test_orbit_plot_supports_explicit_time_window_in_3d() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=20.0, anomaly_deg=10.0)

    fig, ax = orbit.plot(
        start_time=epoch + 12.0 * u.min,
        duration=42.0 * u.min,
        view="3d",
        samples=72,
        title="Single Orbit Plot",
        show=False,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert ax.name == "3d"
    assert ax.get_title() == "Single Orbit Plot"
    assert len(ax.collections) >= 4
    assert ax.get_legend() is None

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Initial Keplerian Elements" in text_blob
    assert "Epoch" in text_blob
    assert "Evaluation" in text_blob
    assert "RAAN" in text_blob
    assert "Marker shows the evaluated spacecraft state" in text_blob

    plt.close(fig)


def test_plot_orbits_single_orbit_2d_ground_track() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=10.0, anomaly_deg=25.0)

    fig, ax = plot_orbits(
        orbit,
        view="2d",
        duration=35.0 * u.min,
        samples=120,
        title="Ground Track View",
        show=False,
    )

    assert isinstance(fig, Figure)
    assert fig.axes[0] is ax
    assert isinstance(ax, GeoAxes)
    assert not hasattr(ax, "zaxis")
    assert ax.get_title() == "Ground Track View"
    assert ax.get_legend() is None
    assert len(ax.lines) >= 1
    assert len(ax.collections) >= 4

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Initial Keplerian Elements" in text_blob
    assert "WGS84 geodetic" in text_blob
    assert "Evaluation" in text_blob
    assert "Marker shows the evaluated spacecraft ground-track position" in text_blob

    plt.close(fig)


def test_plot_orbits_multi_orbit_2d_legend() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit_a = _make_orbit(epoch=epoch, raan_deg=5.0, anomaly_deg=0.0)
    orbit_b = _make_orbit(epoch=epoch, raan_deg=95.0, anomaly_deg=140.0)

    fig, ax = plot_orbits(
        [orbit_a, orbit_b],
        labels=["Alpha", "Bravo"],
        view="2d",
        samples=90,
        show=False,
    )

    assert isinstance(ax, GeoAxes)
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Alpha", "Bravo"]

    text_blob = "\n".join(text.get_text() for text in fig.texts)
    assert "Initial Keplerian Elements" not in text_blob
    assert "Keplerian period" in text_blob

    plt.close(fig)


def test_plot_orbits_rejects_invalid_view() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)

    try:
        plot_orbits(orbit, view="sideways", show=False)
        assert False, "expected ValueError for invalid view"
    except ValueError as exc:
        assert "view" in str(exc)


def test_plot_orbits_rejects_mismatched_axis_type() -> None:
    epoch = Time("2026-01-01T00:00:00", scale="utc")
    orbit = _make_orbit(epoch=epoch, raan_deg=0.0, anomaly_deg=0.0)

    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    try:
        plot_orbits(orbit, view="2d", ax=ax_3d, show=False)
        assert False, "expected TypeError for 2d plot on 3d axes"
    except TypeError as exc:
        assert "view='2d'" in str(exc)
    finally:
        plt.close(fig_3d)

    fig_2d = plt.figure()
    ax_2d = fig_2d.add_subplot(111, projection=ccrs.PlateCarree())
    try:
        plot_orbits(orbit, view="3d", ax=ax_2d, show=False)
        assert False, "expected TypeError for 3d plot on 2d axes"
    except TypeError as exc:
        assert "view='3d'" in str(exc)
    finally:
        plt.close(fig_2d)
