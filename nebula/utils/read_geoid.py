import struct

import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------------
# 1. Raw-grid reader (Fortran unformatted records)
# ---------------------------------------------------------------------


def _detect_fortran_endian(path: str, expected_record_bytes: int) -> str:
    with open(path, "rb") as f:
        marker = f.read(4)

    if len(marker) != 4:
        raise ValueError(f"File too short to read Fortran record marker: {path}")

    little = struct.unpack("<i", marker)[0]
    big = struct.unpack(">i", marker)[0]

    if little == expected_record_bytes:
        return "<"
    if big == expected_record_bytes:
        return ">"

    if little > 0 and little % 4 == 0 and not (big > 0 and big % 4 == 0):
        return "<"
    if big > 0 and big % 4 == 0 and not (little > 0 and little % 4 == 0):
        return ">"

    raise ValueError(
        "Could not determine Fortran record endianness from first record marker. "
        f"little={little}, big={big}, expected={expected_record_bytes}"
    )


def _read_fortran_record_f32(
    f, *, endian: str, expected_count: int, row_index: int
) -> np.ndarray:
    marker = f.read(4)
    if len(marker) != 4:
        raise ValueError(f"Unexpected EOF at row {row_index}: missing record marker")

    nbytes = struct.unpack(f"{endian}i", marker)[0]
    expected_bytes = expected_count * 4
    if nbytes != expected_bytes:
        raise ValueError(
            f"Row {row_index}: expected {expected_bytes} bytes, got {nbytes}"
        )

    payload = f.read(nbytes)
    if len(payload) != nbytes:
        raise ValueError(f"Unexpected EOF at row {row_index}: incomplete record body")

    trailer = f.read(4)
    if len(trailer) != 4:
        raise ValueError(f"Unexpected EOF at row {row_index}: missing record trailer")

    nbytes_trailer = struct.unpack(f"{endian}i", trailer)[0]
    if nbytes_trailer != nbytes:
        raise ValueError(
            f"Row {row_index}: marker/trailer mismatch ({nbytes} vs {nbytes_trailer})"
        )

    return np.frombuffer(payload, dtype=np.dtype(f"{endian}f4"), count=expected_count)


def read_egm2008_grid_raw(path):
    """
    Read the EGM2008 undulation grid as used by the provided Fortran code.

    Returns
    -------
    grid : (nrows, ncols) float32
        grid[0, :] is the northernmost latitude (90 deg),
        grid[-1, :] is the southernmost latitude (-90 deg).
    """
    nrows = 4321
    ncols = 8640

    grid = np.zeros((nrows, ncols), dtype=np.float32)
    endian = _detect_fortran_endian(path, expected_record_bytes=ncols * 4)

    with open(path, "rb") as f:
        for i in range(nrows):
            # read this latitude band as float32
            row = _read_fortran_record_f32(
                f, endian=endian, expected_count=ncols, row_index=i
            )

            # Fortran: ii = nrows + 1 - i   (1-based) → ii0 = nrows - i  (0-based)
            ii = nrows - 1 - i
            grid[ii, :] = row

    return grid


# ---------------------------------------------------------------------
# 2. Build padded grid the same way the Fortran code does
# ---------------------------------------------------------------------


def build_padded_grid(raw_grid, iwind=6):
    """
    Reproduce the padded Fortran grid H(nriw2, nciw2) from raw_grid.

    Parameters
    ----------
    raw_grid : (4321, 8640) array_like
        Raw EGM2008 undulation grid: [lat, lon] with
        lat 0 =  90 deg, lat -1 = -90 deg,
        lon 0 = 0 deg, lon increasing eastward.
    iwind : int
        Fortran's iwind; code uses iw = iwind + 1.

    Returns
    -------
    grid : (nriw2, nciw2) float32
        Padded grid as in Fortran.
    slat : float
        PHIS in Fortran: latitude of SW grid point (deg).
    wlon : float
        DLAW in Fortran: longitude of SW grid point (deg).
    dlat : float
        Latitude spacing (deg).
    dlon : float
        Longitude spacing (deg).
    """
    raw_grid = np.asarray(raw_grid, dtype=np.float32)

    nrows, ncols = raw_grid.shape
    if nrows != 4321 or ncols != 8640:
        raise ValueError(f"Expected raw_grid shape (4321, 8640), got {raw_grid.shape}")

    # From Fortran
    dlat = 2.5 / 60.0  # deg
    dlon = 2.5 / 60.0  # deg

    iw = iwind + 1  # Fortran iw
    nriw2 = nrows + 2 * iw
    nciw2 = ncols + 2 * iw

    grid = np.zeros((nriw2, nciw2), dtype=np.float32)

    # Place the physical grid into the padded interior:
    # Fortran: grid(ii+iw, j+iw) where ii=1..nrows, j=1..ncols.
    # Here raw_grid[0,:] is northmost, corresponding to ii=1.
    grid[iw : iw + nrows, iw : iw + ncols] = raw_grid

    # ---------- longitude padding (periodic wrap) ----------
    # Fortran:
    # do i = iw+1, nrows+iw
    #   do j = 1, iw
    #     grid(i,j) = grid(i,j+ncols)
    #   enddo
    #   do j = ncols+iw+1, ncols+2*iw
    #     grid(i,j) = grid(i,j-ncols)
    #   enddo
    # enddo
    #
    # Python (0-based):
    # i in [iw .. iw+nrows-1]
    grid[iw : iw + nrows, 0:iw] = grid[iw : iw + nrows, ncols : ncols + iw]  # left
    grid[iw : iw + nrows, ncols + iw : ncols + 2 * iw] = grid[
        iw : iw + nrows, iw : 2 * iw
    ]  # right

    # ---------- polar padding ----------
    # Top (north side): i = 1..iw → i0 = 0..iw-1
    # ii0 = 2*iw - i0
    ncols_half = ncols // 2
    for i0 in range(iw):
        ii0 = 2 * iw - i0  # 0-based corresponding to Fortran ii = 2*iw+1-i
        for j0 in range(ncols + 2 * iw):
            jj0 = j0 + ncols_half
            if jj0 >= ncols + 2 * iw:
                jj0 -= ncols
            grid[i0, j0] = grid[ii0, jj0]

    # Bottom (south side): i = nrows+iw+1..nrows+2*iw
    for i1 in range(nrows + iw + 1, nrows + 2 * iw + 1):  # Fortran indices
        i0 = i1 - 1
        ii = 2 * (nrows + iw) + 1 - i1
        ii0 = ii - 1
        for j0 in range(ncols + 2 * iw):
            jj0 = j0 + ncols_half
            if jj0 >= ncols + 2 * iw:
                jj0 -= ncols
            grid[i0, j0] = grid[ii0, jj0]

    # Fortran main:
    # slat = -90.d0 - dlat*iw
    # wlon =        - dlon*iw
    slat = -90.0 - dlat * iw
    wlon = -dlon * iw

    return grid, slat, wlon, dlat, dlon


# ---------------------------------------------------------------------
# 3. 1D spline tools (INITSP, SPLINE) and BILIN
# ---------------------------------------------------------------------


@njit
def _initsp(y):
    """
    Python equivalent of INITSP(Y, N, R, Q) for equidistant natural spline.
    y: 1D array (length N)
    Returns: r (spline moments), length N
    """
    y = np.asarray(y, np.float64)
    n = y.size
    q = np.zeros(n, np.float64)
    r = np.zeros(n, np.float64)

    # Q(1)=0; R(1)=0 already
    for k in range(1, n - 1):
        p = q[k - 1] / 2.0 + 2.0
        q[k] = -0.5 / p
        r[k] = (3.0 * (y[k + 1] - 2.0 * y[k] + y[k - 1]) - r[k - 1] / 2.0) / p

    # R(N) = 0
    for k in range(n - 2, 0, -1):
        r[k] = q[k] * r[k + 1] + r[k]

    return r


@njit
def _spline(x, y, r):
    """
    Python equivalent of SPLINE(X, Y, N, R).
    y, r: arrays length N.
    x: scalar, with 1 = first data point, N = last data point.
    """
    y = np.asarray(y, np.float64)
    r = np.asarray(r, np.float64)
    n = y.size

    if x < 1.0:
        return y[0] + (x - 1.0) * (y[1] - y[0] - r[1] / 6.0)
    elif x > n:
        return y[-1] + (x - n) * (y[-1] - y[-2] + r[-2] / 6.0)
    else:
        j = int(x)  # IFRAC(x) effectively
        xx = x - j
        j0 = j - 1  # Python index
        return y[j0] + xx * (
            (y[j0 + 1] - y[j0] - r[j0] / 3.0 - r[j0 + 1] / 6.0)
            + xx * (r[j0] / 2.0 + xx * (r[j0 + 1] - r[j0]) / 6.0)
        )


@njit
def _bilin(ri, rj, a):
    """
    Python equivalent of BILIN(RI, RJ, A, IMAX, JMAX, IADIM, JADIM),
    with A as 2D numpy array.

    RI, RJ: scalar, with (1,1) = lower-left corner, (IMAX,JMAX) = upper-right.
    """
    imax, jmax = np.shape(a)

    # Fortran IFRAC + integer truncation → int(RI) truncated toward zero.
    # RI, RJ are positive inside domain, so this is floor().
    IN = int(ri)
    IE = int(rj)
    RN = ri - IN
    RE = rj - IE

    if IN < 1:
        IN = 1
        RN = 0.0
    elif IN >= imax:
        IN = imax - 1
        RN = 1.0

    if IE < 1:
        IE = 1
        RE = 0.0
    elif IE >= jmax:
        IE = jmax - 1
        RE = 1.0

    # convert to 0-based
    IN0 = IN - 1
    IE0 = IE - 1

    rnm1 = 1.0 - RN
    rem1 = 1.0 - RE

    return (
        rnm1 * rem1 * a[IN0, IE0]
        + RN * rem1 * a[IN0 + 1, IE0]
        + rnm1 * RE * a[IN0, IE0 + 1]
        + RN * RE * a[IN0 + 1, IE0 + 1]
    )


# ---------------------------------------------------------------------
# 4. INTERP equivalent: 2D spline (or bilinear if iwO<=2)
# ---------------------------------------------------------------------


@njit
def interp_point(
    lat_deg,
    lon_deg,
    grid,
    slat,
    wlon,
    dlat,
    dlon,
    iwind=6,
    dmin_km=0.0,
    invalid_value=999999.0,
):
    """
    Python equivalent of calling INTERP(iwind, dmin, grid, slat, wlon, dlat, dlon, ...).

    Parameters
    ----------
    lat_deg : float
        Geodetic latitude (deg). Valid range: [-90, 90].
    lon_deg : float
        Geodetic longitude (deg). Can be any value; will be wrapped to [0, 360].
    grid : 2D ndarray
        Padded grid, shape (nriw2, nciw2), from build_padded_grid().
    slat : float
        PHIS in Fortran: latitude of SW grid point (deg).
    wlon : float
        DLAW in Fortran: longitude of SW grid point (deg).
    dlat : float
        Grid spacing in latitude (deg).
    dlon : float
        Grid spacing in longitude (deg).
    iwind : int
        Fortran iwind; interpolation window size iwO = iwind.
    dmin_km : float
        Minimum distance from grid edge (km). Fortran uses 0.0.
    invalid_value : float
        Value to return if out of bounds, analogous to 999999. in Fortran.

    Returns
    -------
    val : float
        Interpolated geoid undulation, or invalid_value if outside bounds.
    """

    # Wrap longitude into [0, 360] like Fortran main
    lon = lon_deg
    if lon > 360.0:
        lon = lon - 360.0
    if lon < 0.0:
        lon = lon + 360.0

    # Coordinate validity check (same as main program)
    if lat_deg > 90.0 or lat_deg < -90.0 or lon > 360.0 or lon < 0.0:
        return invalid_value

    # Grid / interpolator parameters
    iwO = iwind
    if iwO < 2:
        iwO = 2
    IPA1 = 20
    if iwO > IPA1:
        iwO = IPA1

    nphi, ndla = np.shape(grid)  # NPHI, NDLA

    # DMIN logic from Fortran (not used when dmin_km=0)
    TWOPI = 6.28318530717959
    RHO = 360.0 / TWOPI
    REARTH = 6371000.0

    if dmin_km <= 0.0:
        ILIM = 0
        JLIM = 0
    else:
        ILIM = int(dmin_km * 1000.0 * RHO / (REARTH * dlat))
        JLIM = int(
            dmin_km
            * 1000.0
            * RHO
            / (REARTH * dlon * np.cos(np.deg2rad(slat + dlat * nphi / 2.0)))
        )

    # RI, RJ as in Fortran
    RI = (lat_deg - slat) / dlat
    RJ = (lon - wlon) / dlon

    # LODD = (iwO/2)*2 != iwO
    LODD = (iwO // 2) * 2 != iwO

    if LODD:
        I0 = int(RI - 0.5)  # int = trunc toward zero, RI>=0 so same as floor
        J0 = int(RJ - 0.5)
    else:
        I0 = int(RI)
        J0 = int(RJ)

    # Fortran: I0 = I0 - iwO/2 + 1  (integer arithmetic)
    I0 = I0 - iwO // 2 + 1
    J0 = J0 - iwO // 2 + 1

    II = I0 + iwO - 1
    JJ = J0 + iwO - 1

    # boundary checks (Fortran uses 0-based I0/J0 with arrays 1..N)
    if (I0 < 0) or (II >= nphi) or (J0 < 0) or (JJ >= ndla):
        return invalid_value

    if (I0 < ILIM) or (II > nphi - ILIM) or (J0 < JLIM) or (JJ > ndla - JLIM):
        # For dmin=0, ILIM=JLIM=0, this branch never triggers if above didn't.
        return invalid_value

    if iwO > 2:
        # 2D spline interpolation using "row then column" approach
        hc = np.empty(iwO, np.float64)
        for i in range(iwO):
            # A(J) = H(I0+I, J0+J), J=1..iwO  (Fortran)
            # Python slice: row I0+i, cols J0 .. J0+iwO-1
            a_row = grid[I0 + i, J0 : J0 + iwO].copy()
            r = _initsp(a_row)
            # x = RJ-J0+1. (Fortran), so x in spline is 1-based index
            x = (RJ - J0) + 1.0
            hc[i] = _spline(x, a_row, r)

        r_hc = _initsp(hc)
        x_row = (RI - I0) + 1.0
        val = _spline(x_row, hc, r_hc)
        return float(val)
    else:
        # Bilinear case (not used for iwind=6, but kept for completeness)
        # Fortran uses RI+1, RJ+1 with full grid as A.
        return float(_bilin(RI + 1.0, RJ + 1.0, grid))


@njit(parallel=True)
def interp_many(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    padded_grid: np.ndarray,
    slat: float,
    wlon: float,
    dlat: float,
    dlon: float,
    iwind: int,
    fill_value: float,
):
    """
    Faster batched interpolation using the same algorithm as interp_point,
    but with constants precomputed once and the logic inlined and parallelized.

    latitudes, longitudes: 1D arrays of same length (already flattened).
    Returns: 1D float32 array of undulations.
    """
    n_points = latitudes.size
    undulations = np.empty(n_points, dtype=np.float32)

    grid = padded_grid
    nphi, ndla = grid.shape

    # Precompute iwO (window size) once
    iwO = iwind
    if iwO < 2:
        iwO = 2
    IPA1 = 20
    if iwO > IPA1:
        iwO = IPA1

    # dmin_km is always 0.0 in your usage, so ILIM = JLIM = 0 and
    # the second edge check in interp_point is redundant; we omit it.

    for idx in prange(n_points):
        lat_deg = latitudes[idx]
        lon_deg = longitudes[idx]

        # ---- BEGIN: same logic as interp_point ----

        # Wrap longitude into [0, 360]
        lon = lon_deg
        if lon > 360.0:
            lon = lon - 360.0
        if lon < 0.0:
            lon = lon + 360.0

        # Coordinate validity check
        if lat_deg > 90.0 or lat_deg < -90.0 or lon > 360.0 or lon < 0.0:
            undulations[idx] = fill_value
            continue

        # RI, RJ as in Fortran
        RI = (lat_deg - slat) / dlat
        RJ = (lon - wlon) / dlon

        # LODD = (iwO/2)*2 != iwO
        LODD = (iwO // 2) * 2 != iwO

        if LODD:
            I0 = int(RI - 0.5)  # trunc toward zero, RI>=0 so floor
            J0 = int(RJ - 0.5)
        else:
            I0 = int(RI)
            J0 = int(RJ)

        # Fortran: I0 = I0 - iwO/2 + 1
        I0 = I0 - iwO // 2 + 1
        J0 = J0 - iwO // 2 + 1

        II = I0 + iwO - 1
        JJ = J0 + iwO - 1

        # First boundary check (with ILIM=JLIM=0 this is the only one needed)
        if (I0 < 0) or (II >= nphi) or (J0 < 0) or (JJ >= ndla):
            undulations[idx] = fill_value
            continue

        # Main interpolation
        if iwO > 2:
            # 2D spline interpolation (row then column), same as interp_point
            hc = np.empty(iwO, np.float64)
            for i in range(iwO):
                a_row = grid[I0 + i, J0 : J0 + iwO].astype(np.float64)
                r = _initsp(a_row)
                x = (RJ - J0) + 1.0  # 1-based index
                hc[i] = _spline(x, a_row, r)

            r_hc = _initsp(hc)
            x_row = (RI - I0) + 1.0
            val = _spline(x_row, hc, r_hc)
            undulations[idx] = float(val)
        else:
            # Bilinear fallback (not used for iwind=6, kept for completeness)
            val = _bilin(RI + 1.0, RJ + 1.0, grid)
            undulations[idx] = float(val)

        # ---- END: same logic as interp_point ----

    return undulations


# ---------------------------------------------------------------------
# 5. Example usage
# ---------------------------------------------------------------------


def run_self_tests(raw, grid_padded, slat, wlon, dlat, dlon):
    """
    Basic sanity tests against the Fortran behavior:
      - interpolation at grid nodes reproduces raw values (including edges)
      - longitude periodicity
      - valid vs invalid latitude/longitude boundaries
      - known min/max checks
    """
    invalid = 999999.0
    nrows, ncols = raw.shape
    atol_node = 1e-3  # tolerance for node reproduction
    atol_periodic = 1e-5

    # 1) Node reproduction tests (corners, equator, random samples)
    node_indices = [
        (0, 0),  # south-west corner (-90, 0)
        (0, ncols - 1),  # south-east corner (-90, ~360)
        (nrows - 1, 0),  # north-west corner (90, 0)
        (nrows - 1, ncols - 1),  # north-east corner (90, ~360)
    ]

    # equator row: -90 + i*dlat = 0  => i = 90/dlat = 2160
    i_eq = int(round(90.0 / dlat))
    node_indices.append((i_eq, 0))
    node_indices.append((i_eq, ncols - 1))

    rng = np.random.default_rng(0)
    for _ in range(10):
        i = int(rng.integers(0, nrows))
        j = int(rng.integers(0, ncols))
        node_indices.append((i, j))

    for i, j in node_indices:
        lat = -90.0 + i * dlat
        lon = j * dlon
        v_interp = interp_point(lat, lon, grid_padded, slat, wlon, dlat, dlon)
        v_true = float(raw[i, j])
        assert np.isfinite(v_interp), f"NaN at node ({i},{j})"
        diff = abs(v_interp - v_true)
        assert (
            diff <= atol_node
        ), f"Node mismatch at ({lat},{lon}) idx({i},{j}): interp={v_interp}, true={v_true}, diff={diff}"

    # 2) Longitude periodicity: f(lat, λ) = f(lat, λ+360) = f(lat, λ-360)
    lat_test = 0.0
    lon_test = 23.75
    v0 = interp_point(lat_test, lon_test, grid_padded, slat, wlon, dlat, dlon)
    v1 = interp_point(lat_test, lon_test + 360.0, grid_padded, slat, wlon, dlat, dlon)
    v2 = interp_point(lat_test, lon_test - 360.0, grid_padded, slat, wlon, dlat, dlon)
    assert abs(v0 - v1) <= atol_periodic, f"Periodicity fail: v0={v0}, v1={v1}"
    assert abs(v0 - v2) <= atol_periodic, f"Periodicity fail: v0={v0}, v2={v2}"

    # Also check explicit 0 vs 360 for a mid-latitude
    lat_mid = 45.0
    v_0 = interp_point(lat_mid, 0.0, grid_padded, slat, wlon, dlat, dlon)
    v_360 = interp_point(lat_mid, 360.0, grid_padded, slat, wlon, dlat, dlon)
    assert abs(v_0 - v_360) <= atol_periodic, f"0 vs 360 mismatch: {v_0} vs {v_360}"

    # 3) Boundary validity: exact edges are allowed, outside is invalid
    # valid:
    for lat, lon in [
        (-90.0, 0.0),
        (90.0, 0.0),
        (0.0, 0.0),
        (0.0, 360.0),
        (90.0, 360.0),
        (-90.0, 360.0),
    ]:
        v = interp_point(lat, lon, grid_padded, slat, wlon, dlat, dlon)
        assert v != invalid, f"Unexpected invalid at ({lat},{lon})"

    # invalid:
    for lat, lon in [
        (91.0, 0.0),
        (-91.0, 0.0),
    ]:
        v = interp_point(lat, lon, grid_padded, slat, wlon, dlat, dlon)
        assert v == invalid, f"Expected invalid at ({lat},{lon}), got {v}"

    # 4) Known min/max checks (from your examples)
    # These asserts assume your expected_min/expected_max are from the NGA test program.
    lat_min = 4.667
    lon_min = 78.750
    expected_min = -106.909
    val_min = interp_point(lat_min, lon_min, grid_padded, slat, wlon, dlat, dlon)
    assert (
        abs(val_min - expected_min) <= 1e-3
    ), f"Min check failed: got {val_min}, expected {expected_min}"

    lat_max = -8.417
    lon_max = 147.375
    expected_max = 85.824
    val_max = interp_point(lat_max, lon_max, grid_padded, slat, wlon, dlat, dlon)
    assert (
        abs(val_max - expected_max) <= 1e-3
    ), f"Max check failed: got {val_max}, expected {expected_max}"


class EGM2008Geoid:
    """
    Class to encapsulate EGM2008 geoid undulation interpolation.
    """

    def __init__(self, interpolated_grid_path: str, iwind: int = 6):
        raw = read_egm2008_grid_raw(interpolated_grid_path)
        self.grid_padded, self.slat, self.wlon, self.dlat, self.dlon = (
            build_padded_grid(raw, iwind=iwind)
        )
        self.iwind = iwind

    def get_undulation(self, latitudes, longitudes, fill_value=np.nan) -> np.ndarray:
        """
        Compute geoid undulation for arbitrary lat/lon inputs.

        Supports:
        - scalars
        - 1D arrays
        - 2D (e.g. meshgrid) and higher, as long as they can be broadcast.
        Broadcasting rules match numpy and RasterDEM.get_height.
        """

        # Convert and broadcast like RasterDEM.get_height
        latitudes = np.asarray(latitudes, dtype=np.float64)
        longitudes = np.asarray(longitudes, dtype=np.float64)
        latitudes, longitudes = np.broadcast_arrays(latitudes, longitudes)
        original_shape = latitudes.shape

        # Flatten to 1D for the numba kernel
        lat_flat = latitudes.ravel()
        lon_flat = longitudes.ravel()

        # Ensure fill_value is a plain float (numba-friendly)
        fv = float(fill_value)

        und_flat = interp_many(
            lat_flat,
            lon_flat,
            self.grid_padded,
            self.slat,
            self.wlon,
            self.dlat,
            self.dlon,
            self.iwind,
            fv,
        )

        # Reshape back to the original broadcast shape
        return und_flat.reshape(original_shape)


if __name__ == "__main__":
    geoid_path = r"J:\downloads\EGM2008_Interpolation_Grid\Und_min2.5x2.5_egm2008_isw=82_WGS84_TideFree_SE"

    geoid = EGM2008Geoid(geoid_path, iwind=6)
    test = geoid.get_undulation(38.790235, -79.308806)
    test2 = geoid.get_undulation(np.linspace(0, 10, 1000), np.linspace(0, 20, 1000))

    # 1) read raw grid
    raw = read_egm2008_grid_raw(geoid_path)

    # 2) build padded grid + metadata
    grid_padded, slat, wlon, dlat, dlon = build_padded_grid(raw, iwind=6)

    # 3) interpolate at a test point, e.g. lat=40N, lon=-105E
    lat = 40.0
    lon = -105.0  # will be wrapped to [0,360] inside interp_point
    val = interp_point(lat, lon, grid_padded, slat, wlon, dlat, dlon, iwind=6)

    print(f"EGM2008 geoid undulation at ({lat}, {lon}) = {val:.3f} m")

    lat_min = 4.667
    lon_min = 78.750
    expected_min = -106.909
    val_min = interp_point(
        lat_min, lon_min, grid_padded, slat, wlon, dlat, dlon, iwind=6
    )
    print(
        f"EGM2008 geoid undulation at ({lat_min}, {lon_min}) = {val_min:.3f} m (expected {expected_min} m)"
    )

    expected_max = 85.824
    lat_max = -8.417
    lon_max = 147.375
    val_max = interp_point(
        lat_max, lon_max, grid_padded, slat, wlon, dlat, dlon, iwind=6
    )
    print(
        f"EGM2008 geoid undulation at ({lat_max}, {lon_max}) = {val_max:.3f} m (expected {expected_max} m)"
    )

    val = interp_point(0, 0, grid_padded, slat, wlon, dlat, dlon, iwind=6)
    pass

    run_self_tests(raw, grid_padded, slat, wlon, dlat, dlon)

    pass
