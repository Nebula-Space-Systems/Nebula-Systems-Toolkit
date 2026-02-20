from __future__ import annotations

import numpy as np
import pytest

from nebula.localization.particle_initialization import (
    make_coarse_candidates_ecef,
    los_mask_all_observers,
    spacing_radius,
)
from nebula.transform._ecef2geodetic import ecef2geodetic_vec_xyz
from nebula.transform._geodetic2ecef import geodetic2ecef


_DIST_PARAMS = dict(
    h_min=-200.0,
    h_fade_start=8000.0,
    h_dense_max=15000.0,
    h_max=60000.0,
    dh_surface=2000.0,
    dh0=3000.0,
    growth=1.25,
    dh_max=30000.0,
    occupancy=0.8,
    min_per_band=250,
    max_per_band=4000,
    keep_inflate=1.6,
    r_terrain=6000.0,
    r_air=10000.0,
    h_grow=60000.0,
    beta=0.75,
    chunk_size=20000,
    seed=123,
)


@pytest.fixture(scope="module")
def particle_distribution_sample():
    pts = make_coarse_candidates_ecef(**_DIST_PARAMS)
    lat, lon, h = ecef2geodetic_vec_xyz(
        pts[:, 0].astype(np.float64),
        pts[:, 1].astype(np.float64),
        pts[:, 2].astype(np.float64),
    )
    return pts, lat, lon, h


def _shell_volume(a_m: float, h_lo: float, h_hi: float) -> float:
    h_c = 0.5 * (h_lo + h_hi)
    r = a_m + (h_c if h_c > 0.0 else 0.0)
    return 4.0 * np.pi * r * r * (h_hi - h_lo)


def test_particle_altitude_coverage_and_band_presence(
    particle_distribution_sample,
) -> None:
    pts, _lat, _lon, h = particle_distribution_sample

    assert pts.dtype == np.float32
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] > 10000
    assert np.isfinite(pts).all()

    assert float(h.min()) >= _DIST_PARAMS["h_min"] - 2.0
    assert float(h.max()) <= _DIST_PARAMS["h_max"] + 2.0

    bands = np.array(
        [
            _DIST_PARAMS["h_min"],
            _DIST_PARAMS["h_fade_start"],
            _DIST_PARAMS["h_dense_max"],
            35000.0,
            _DIST_PARAMS["h_max"],
        ],
        dtype=np.float64,
    )
    counts, _ = np.histogram(h, bins=bands)
    assert np.all(counts > 0)


def test_particle_latitude_equal_area_distribution(
    particle_distribution_sample,
) -> None:
    _pts, lat, _lon, _h = particle_distribution_sample
    sin_lat = np.sin(lat)
    counts, _ = np.histogram(sin_lat, bins=np.linspace(-1.0, 1.0, 13))

    # Equal-area sampling should produce an almost-uniform sin(lat) histogram.
    assert float(counts.max() / counts.min()) < 1.25

    expected = float(np.mean(counts))
    chi2_per_bin = float(np.sum((counts - expected) ** 2 / expected) / counts.size)
    assert chi2_per_bin < 3.0


def test_particle_global_lat_lon_cell_coverage(
    particle_distribution_sample,
) -> None:
    _pts, lat, lon, _h = particle_distribution_sample

    n_sin_bins = 12
    n_lon_bins = 24
    sin_lat = np.sin(lat)

    s_edges = np.linspace(-1.0, 1.0, n_sin_bins + 1)
    lon_edges = np.linspace(-np.pi, np.pi, n_lon_bins + 1)

    s_idx = np.clip(np.searchsorted(s_edges, sin_lat, side="right") - 1, 0, n_sin_bins - 1)
    l_idx = np.clip(np.searchsorted(lon_edges, lon, side="right") - 1, 0, n_lon_bins - 1)

    grid = np.zeros((n_sin_bins, n_lon_bins), dtype=np.int64)
    for a, b in zip(s_idx, l_idx):
        grid[a, b] += 1

    # Every equal-area x longitude cell should be populated for this sample size.
    assert np.all(grid > 0)


def test_particle_longitude_uniformity_within_latitude_slabs(
    particle_distribution_sample,
) -> None:
    _pts, lat, lon, _h = particle_distribution_sample
    sin_lat = np.sin(lat)
    lon_edges = np.linspace(-np.pi, np.pi, 25)

    slabs = [(-0.3, 0.3), (0.3, 0.7), (-0.7, -0.3)]
    for lo, hi in slabs:
        m = (sin_lat >= lo) & (sin_lat < hi)
        counts, _ = np.histogram(lon[m], bins=lon_edges)

        assert np.all(counts > 0)
        assert float(counts.max() / counts.min()) < 1.45

        expected = float(np.mean(counts))
        chi2_per_bin = float(np.sum((counts - expected) ** 2 / expected) / counts.size)
        assert chi2_per_bin < 3.5


def test_particle_density_decreases_with_altitude(
    particle_distribution_sample,
) -> None:
    _pts, _lat, _lon, h = particle_distribution_sample

    bins = np.array([-200.0, 8000.0, 15000.0, 35000.0, 60000.0], dtype=np.float64)
    counts, _ = np.histogram(h, bins=bins)

    a = 6378137.0
    densities = np.array(
        [
            counts[i] / _shell_volume(a, float(bins[i]), float(bins[i + 1]))
            for i in range(bins.size - 1)
        ],
        dtype=np.float64,
    )

    # Density should not increase materially with altitude.
    for i in range(densities.size - 1):
        assert densities[i + 1] <= densities[i] * 1.10

    # Upper-atmosphere density should be substantially lower than near-surface.
    assert densities[-1] < 0.3 * densities[0]


def test_particle_band_voxel_uniqueness_from_spacing_model(
    particle_distribution_sample,
) -> None:
    pts, _lat, _lon, h = particle_distribution_sample
    p = _DIST_PARAMS

    def splitmix64_py(x: int) -> int:
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def u01_from_uint64_py(x: int) -> float:
        return float(x >> 11) * 1.1102230246251565e-16

    edges = []
    l0 = p["h_fade_start"] - p["h_min"]
    n0 = max(1, int(np.ceil(l0 / p["dh_surface"])))
    step0 = l0 / n0
    for i in range(n0 + 1):
        edges.append(p["h_min"] + step0 * i)

    l1 = p["h_dense_max"] - p["h_fade_start"]
    n1 = max(1, int(np.ceil(l1 / p["dh_surface"])))
    step1 = l1 / n1 if n1 > 0 else 0.0
    for i in range(1, n1 + 1):
        edges.append(p["h_fade_start"] + step1 * i)

    hh = edges[-1]
    dh = p["dh0"]
    while hh < p["h_max"]:
        step = min(dh, p["dh_max"])
        hh2 = min(hh + step, p["h_max"])
        edges.append(hh2)
        hh = hh2
        dh *= p["growth"]
        if step <= 0.0:
            break

    edges = np.asarray(edges, dtype=np.float64)
    xyz = pts.astype(np.float64)
    seed = int(p["seed"])

    for b in range(edges.size - 1):
        h0 = float(edges[b])
        h1 = float(edges[b + 1])

        # Trim thin boundary layers to avoid numerical band-edge classification noise.
        m = (h >= h0 + 1.0) & (h < h1 - 1.0)
        if int(np.count_nonzero(m)) < 10:
            continue

        h_c = 0.5 * (h0 + h1)
        voxel = spacing_radius(
            h_c,
            p["h_fade_start"],
            p["h_dense_max"],
            p["r_terrain"],
            p["r_air"],
            p["h_grow"],
            p["beta"],
        )

        s0 = splitmix64_py(seed ^ (b * 3 + 0))
        s1 = splitmix64_py(seed ^ (b * 3 + 1))
        s2 = splitmix64_py(seed ^ (b * 3 + 2))
        ox = u01_from_uint64_py(s0) * voxel
        oy = u01_from_uint64_py(s1) * voxel
        oz = u01_from_uint64_py(s2) * voxel

        q = xyz[m]
        ix = np.floor((q[:, 0] + ox) / voxel).astype(np.int64)
        iy = np.floor((q[:, 1] + oy) / voxel).astype(np.int64)
        iz = np.floor((q[:, 2] + oz) / voxel).astype(np.int64)
        keys = np.stack((ix, iy, iz), axis=1)
        uniq = np.unique(keys, axis=0)

        assert uniq.shape[0] == keys.shape[0]


def test_particle_nearest_neighbor_not_clumpy(
    particle_distribution_sample,
) -> None:
    pts, _lat, _lon, _h = particle_distribution_sample
    p = pts.astype(np.float64)
    rng = np.random.default_rng(0)
    q_idx = rng.choice(p.shape[0], size=120, replace=False)

    nn = np.empty(q_idx.size, dtype=np.float64)
    for i, qi in enumerate(q_idx):
        d = np.linalg.norm(p - p[qi], axis=1)
        d[qi] = np.inf
        nn[i] = float(np.min(d))

    # Keep a healthy lower bound against pathological clumping.
    assert float(np.percentile(nn, 5)) > 15000.0
    assert float(np.percentile(nn, 95) / np.percentile(nn, 5)) < 8.0


def test_particle_seed_reproducibility_and_variation() -> None:
    p1 = make_coarse_candidates_ecef(**_DIST_PARAMS)
    p2 = make_coarse_candidates_ecef(**_DIST_PARAMS)
    assert np.array_equal(p1, p2)

    p3_params = dict(_DIST_PARAMS)
    p3_params["seed"] = 124
    p3 = make_coarse_candidates_ecef(**p3_params)
    assert not np.array_equal(p1, p3)


def test_particle_los_mask_filters_occluded_points(
    particle_distribution_sample,
) -> None:
    pts, _lat, _lon, _h = particle_distribution_sample
    observer = np.array([geodetic2ecef(0.0, 0.0, 0.0)], dtype=np.float64)

    mask = los_mask_all_observers(pts.astype(np.float64), observer)
    keep = int(mask.sum())
    total = int(mask.size)

    assert mask.dtype == np.bool_
    assert 0 < keep < total
    assert (keep / total) < 0.2


def test_spacing_profile_matches_expected_shape() -> None:
    p = _DIST_PARAMS
    s_surface = spacing_radius(
        0.0,
        p["h_fade_start"],
        p["h_dense_max"],
        p["r_terrain"],
        p["r_air"],
        p["h_grow"],
        p["beta"],
    )
    s_fade_start = spacing_radius(
        p["h_fade_start"],
        p["h_fade_start"],
        p["h_dense_max"],
        p["r_terrain"],
        p["r_air"],
        p["h_grow"],
        p["beta"],
    )
    s_dense_max = spacing_radius(
        p["h_dense_max"],
        p["h_fade_start"],
        p["h_dense_max"],
        p["r_terrain"],
        p["r_air"],
        p["h_grow"],
        p["beta"],
    )
    s_high = spacing_radius(
        50000.0,
        p["h_fade_start"],
        p["h_dense_max"],
        p["r_terrain"],
        p["r_air"],
        p["h_grow"],
        p["beta"],
    )

    assert abs(s_surface - p["r_terrain"]) < 1e-12
    assert abs(s_fade_start - p["r_terrain"]) < 1e-12
    assert abs(s_dense_max - p["r_air"]) < 1e-12
    assert s_high > p["r_air"]
