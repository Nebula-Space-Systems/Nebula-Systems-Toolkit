import numpy as np
from numba import njit, prange

DEG2RAD = np.pi / 180.0


# ----------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------


@njit(inline="always")
def _bilinear_sample(z: np.ndarray, r: float, c: float) -> float:
    """
    Bilinear interpolation of z at fractional row r, col c.
    Assumes 0 <= r <= nrows-1, 0 <= c <= ncols-1.
    """
    nrows, ncols = z.shape

    # Clamp to valid range to be safe
    if r < 0.0:
        r = 0.0
    elif r > nrows - 1:
        r = nrows - 1.0

    if c < 0.0:
        c = 0.0
    elif c > ncols - 1:
        c = ncols - 1.0

    i0 = int(np.floor(r))
    j0 = int(np.floor(c))

    i1 = i0 + 1
    if i1 >= nrows:
        i1 = nrows - 1

    j1 = j0 + 1
    if j1 >= ncols:
        j1 = ncols - 1

    tr = r - i0
    tc = c - j0

    z00 = z[i0, j0]
    z10 = z[i1, j0]
    z01 = z[i0, j1]
    z11 = z[i1, j1]

    # bilinear interpolation
    z0 = z00 * (1.0 - tc) + z01 * tc
    z1 = z10 * (1.0 - tc) + z11 * tc
    return z0 * (1.0 - tr) + z1 * tr


@njit
def _prepare_viewshed_inputs(
    lat_grid_deg: np.ndarray,
    lon_grid_deg: np.ndarray,
    z_grid: np.ndarray,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_h_ellip_m: float,
):
    """
    Precompute observer index, grid spacing, and observer height for the viewshed.
    """
    nrows, ncols = z_grid.shape

    # Find observer's reference cell (nearest in lat/lon)
    dlat = lat_grid_deg - obs_lat_deg
    dlon = (lon_grid_deg - obs_lon_deg) * np.cos(obs_lat_deg * DEG2RAD)
    dist2 = dlat * dlat + dlon * dlon

    # Flatten argmin
    obs_idx_flat = 0
    min_val = dist2.ravel()[0]
    flat = dist2.ravel()
    for k in range(flat.size):
        if flat[k] < min_val:
            min_val = flat[k]
            obs_idx_flat = k

    obs_i = obs_idx_flat // ncols
    obs_j = obs_idx_flat % ncols

    # Approximate horizontal spacing at central latitude
    lat0_rad = lat_grid_deg.mean() * DEG2RAD
    m_per_deg_lat = 111_132.954
    m_per_deg_lon = 111_132.954 * np.cos(lat0_rad)

    # Assume regular grid
    if nrows > 1:
        dlat_row = lat_grid_deg[1, 0] - lat_grid_deg[0, 0]
    else:
        dlat_row = 0.0

    if ncols > 1:
        dlon_col = lon_grid_deg[0, 1] - lon_grid_deg[0, 0]
    else:
        dlon_col = 0.0

    # Use absolute spacing so step length is positive
    dy = abs(dlat_row) * m_per_deg_lat
    dx = abs(dlon_col) * m_per_deg_lon

    # Use the actual observer ellipsoidal height (NOT the DEM value)
    z_obs = float(obs_h_ellip_m)

    return obs_i, obs_j, dx, dy, z_obs


# ----------------------------------------------------------------------
# Core kernels
# ----------------------------------------------------------------------


@njit(parallel=True)
def _viewshed_kernel(
    z: np.ndarray,
    obs_i: int,
    obs_j: int,
    z_obs: float,
    dx: float,
    dy: float,
    step_factor: float,
    z_margin: float,
) -> np.ndarray:
    """
    Per-cell, independent LOS viewshed using bilinear sampling.

    z          : 2D DEM heights (ellipsoidal) [m]
    obs_i,obs_j: observer cell indices
    z_obs      : observer ellipsoidal height [m]
    dx, dy     : horizontal spacing [m] per column/row step
    step_factor: step length = step_factor * min(dx, dy)
    z_margin   : tolerance (m) to treat near-ties as visible/blocked

    Returns:
      visible : 2D bool array, True if cell is visible.
    """
    nrows, ncols = z.shape
    visible = np.zeros((nrows, ncols), dtype=np.bool_)

    visible[obs_i, obs_j] = True

    min_step = dx if dx < dy else dy
    if min_step <= 0.0:
        min_step = 1.0  # fallback

    step_len = step_factor * min_step
    if step_len <= 0.0:
        step_len = min_step

    # Parallelize over rows
    for i in prange(nrows):
        for j in range(ncols):
            if i == obs_i and j == obs_j:
                continue

            # Vector from observer cell index to target cell index (grid coords)
            di = i - obs_i
            dj = j - obs_j

            # Horizontal distance in meters
            hx = dj * dx
            hy = di * dy
            R = np.sqrt(hx * hx + hy * hy)
            if R == 0.0:
                visible[i, j] = True
                continue

            z_tgt = z[i, j]

            # Number of steps along LOS
            n_steps = int(R / step_len)
            if n_steps < 1:
                # Very close cell; treat as visible
                visible[i, j] = True
                continue

            blocked = False

            for k in range(1, n_steps):
                t = k / n_steps  # 0 < t < 1

                # Fractional position in grid indices
                r = obs_i + di * t
                c = obs_j + dj * t

                # Height along the straight LOS line (linear in t)
                z_line = z_obs + (z_tgt - z_obs) * t

                # Terrain height at this point (bilinear interpolation)
                z_terr = _bilinear_sample(z, r, c)

                if z_terr > z_line + z_margin:
                    blocked = True
                    break

            visible[i, j] = not blocked

    return visible


@njit(parallel=True)
def _viewshed_kernel_masked(
    z: np.ndarray,
    obs_i: int,
    obs_j: int,
    z_obs: float,
    dx: float,
    dy: float,
    step_factor: float,
    z_margin: float,
    candidate_mask: np.ndarray,
) -> np.ndarray:
    """
    Same as _viewshed_kernel, but only performs LOS checks where
    candidate_mask[i,j] is True. Other cells remain False (except
    the observer cell, which is forced True).
    """
    nrows, ncols = z.shape
    visible = np.zeros((nrows, ncols), dtype=np.bool_)

    # Always visible to itself
    visible[obs_i, obs_j] = True

    min_step = dx if dx < dy else dy
    if min_step <= 0.0:
        min_step = 1.0

    step_len = step_factor * min_step
    if step_len <= 0.0:
        step_len = min_step

    for i in prange(nrows):
        for j in range(ncols):
            # Skip cells not in candidate_mask, except the observer itself
            if not candidate_mask[i, j] and not (i == obs_i and j == obs_j):
                continue

            if i == obs_i and j == obs_j:
                # Already marked True
                continue

            di = i - obs_i
            dj = j - obs_j

            hx = dj * dx
            hy = di * dy
            R = np.sqrt(hx * hx + hy * hy)
            if R == 0.0:
                visible[i, j] = True
                continue

            z_tgt = z[i, j]

            n_steps = int(R / step_len)
            if n_steps < 1:
                visible[i, j] = True
                continue

            blocked = False

            for k in range(1, n_steps):
                t = k / n_steps

                r = obs_i + di * t
                c = obs_j + dj * t

                z_line = z_obs + (z_tgt - z_obs) * t
                z_terr = _bilinear_sample(z, r, c)

                if z_terr > z_line + z_margin:
                    blocked = True
                    break

            visible[i, j] = not blocked

    return visible


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def compute_viewshed(
    lat_grid_deg: np.ndarray,
    lon_grid_deg: np.ndarray,
    ellip_height_grid_m: np.ndarray,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_h_ellip_m: float,
    step_factor: float = 1.0,
    z_margin: float = 0.0,
) -> np.ndarray:
    """
    High-accuracy per-cell LOS viewshed (local planar approximation).

    Inputs
    ------
    lat_grid_deg, lon_grid_deg : 2D arrays of same shape (deg)
    ellip_height_grid_m        : 2D array (m), DEM ellipsoidal heights
    obs_lat_deg, obs_lon_deg   : observer geodetic position (deg)
    obs_h_ellip_m              : observer ellipsoidal height (m)
    step_factor                : step length = step_factor * min(dx, dy),
                                 1.0 ~ one sample per cell along LOS,
                                 <1.0 -> more samples (slower, more accurate).
    z_margin                   : small positive (m) tolerance to avoid
                                 numerical ties (e.g. 0.01).

    Returns
    -------
    visible_mask : 2D bool array, True if that DEM cell is visible.
    """
    lat_grid_deg = np.asarray(lat_grid_deg, dtype=np.float64)
    lon_grid_deg = np.asarray(lon_grid_deg, dtype=np.float64)
    z_grid = np.asarray(ellip_height_grid_m, dtype=np.float64)

    if lat_grid_deg.shape != lon_grid_deg.shape or lat_grid_deg.shape != z_grid.shape:
        raise ValueError(
            "lat_grid_deg, lon_grid_deg, and ellip_height_grid_m must have the same shape"
        )

    obs_i, obs_j, dx, dy, z_obs = _prepare_viewshed_inputs(
        lat_grid_deg,
        lon_grid_deg,
        z_grid,
        float(obs_lat_deg),
        float(obs_lon_deg),
        float(obs_h_ellip_m),
    )

    visible_mask = _viewshed_kernel(
        z_grid,
        obs_i,
        obs_j,
        z_obs,
        dx,
        dy,
        float(step_factor),
        float(z_margin),
    )
    return visible_mask


def compute_viewshed_masked(
    lat_grid_deg: np.ndarray,
    lon_grid_deg: np.ndarray,
    ellip_height_grid_m: np.ndarray,
    obs_lat_deg: float,
    obs_lon_deg: float,
    obs_h_ellip_m: float,
    candidate_mask: np.ndarray,
    step_factor: float = 1.0,
    z_margin: float = 0.0,
) -> np.ndarray:
    """
    Viewshed from an observer, but only performing LOS checks where
    candidate_mask[i,j] is True.

    Parameters
    ----------
    lat_grid_deg, lon_grid_deg : 2D arrays of same shape (deg)
    ellip_height_grid_m        : 2D array (m), DEM ellipsoidal heights
    obs_lat_deg, obs_lon_deg   : observer geodetic position (deg)
    obs_h_ellip_m              : observer ellipsoidal height (m)
    candidate_mask             : 2D bool array, same shape as DEM
                                 True => compute LOS to that cell,
                                 False => skip (cell will stay False,
                                 except observer cell which is forced True).
    step_factor                : same as compute_viewshed
    z_margin                   : same as compute_viewshed

    Returns
    -------
    visible_mask : 2D bool array, True where candidate_mask AND LOS.
    """
    lat_grid_deg = np.asarray(lat_grid_deg, dtype=np.float64)
    lon_grid_deg = np.asarray(lon_grid_deg, dtype=np.float64)
    z_grid = np.asarray(ellip_height_grid_m, dtype=np.float64)
    candidate_mask = np.asarray(candidate_mask, dtype=np.bool_)

    if lat_grid_deg.shape != lon_grid_deg.shape or lat_grid_deg.shape != z_grid.shape:
        raise ValueError(
            "lat_grid_deg, lon_grid_deg, and ellip_height_grid_m must have the same shape"
        )

    if candidate_mask.shape != z_grid.shape:
        raise ValueError("candidate_mask must have the same shape as the DEM grid")

    obs_i, obs_j, dx, dy, z_obs = _prepare_viewshed_inputs(
        lat_grid_deg,
        lon_grid_deg,
        z_grid,
        float(obs_lat_deg),
        float(obs_lon_deg),
        float(obs_h_ellip_m),
    )

    visible_mask = _viewshed_kernel_masked(
        z_grid,
        obs_i,
        obs_j,
        z_obs,
        dx,
        dy,
        float(step_factor),
        float(z_margin),
        candidate_mask,
    )
    return visible_mask
