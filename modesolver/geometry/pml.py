import numpy as np

def _solve_b(alpha, tol=1e-12, maxiter=50):
    """
    Solve exp(z) - 1 - alpha*z = 0 for real z (geometric stretching).
    """
    z = alpha if alpha != 0 else 1.0
    for _ in range(maxiter):
        f = np.exp(z) - 1.0 - alpha * z
        df = np.exp(z) - alpha
        dz = -f / df
        z += dz
        if abs(dz) < tol:
            break
    return z

def _maybe_promote(arr, new_values):
    """
    Prevents unnecessary promotion to complex.
    Only promotes arr to complex type if new_values contain imaginary parts.
    """
    if np.iscomplexobj(arr):
        return arr   # already complex

    if np.iscomplexobj(new_values):
        # Only promote if there is *actually* an imaginary component.
        if np.any(np.abs(np.imag(new_values)) > 0):
            return arr.astype(complex)

    return arr

def stretchmesh(x, y, nlayers, factor, method="PPPP"):
    """
    Continuously stretch the grid spacing near the edges of a computational
    window for finite-difference calculations.

    This function allows you to enlarge the physical domain without increasing
    the total number of grid points. Four different stretching methods are
    implemented—uniform, linear, parabolic (default), and geometric.  The first
    three methods support complex coordinate stretching, which is useful for
    implementing perfectly matched layers (PML) and non-reflective boundaries.

    Parameters

    x, y : array_like
        1D arrays specifying the original grid vertices, typically uniformly
        spaced.

    nlayers : int or array_like of length 4
        Number of grid layers to stretch at each boundary:
            nlayers[0] : north boundary   (top of y)
            nlayers[1] : south boundary   (bottom of y)
            nlayers[2] : east boundary    (right side of x)
            nlayers[3] : west boundary    (left side of x)
        A scalar value applies to all four boundaries.

    factor : scalar or array_like of length 4
        Cumulative expansion factor applied to each stretched region.  Similar
        to nlayers, a scalar expands all four boundaries equally.

    method : str of length 4, optional
        Stretching method at each boundary. Each character must be one of:
            'U' : uniform
            'L' : linear
            'P' : parabolic (default)
            'G' : geometric
        Example: method='LLLG' uses linear stretching on the north, south,
        and east boundaries, and geometric stretching on the west boundary.

    Returns

    x, y : ndarray
        The stretched grid vertices.

    xc, yc : ndarray (optional)
        Cell-center coordinates of the stretched grid:
            xc = 0.5 * (x[:-1] + x[1:])
            yc = 0.5 * (y[:-1] + y[1:])

    dx, dy : ndarray (optional)
        Grid spacing after stretching:
            dx = np.diff(x)
            dy = np.diff(y)

    Notes

    This function is commonly used in wave, electromagnetic, and acoustic
    finite-difference simulations to create absorbing boundaries (PML) or
    to extend the computational domain without increasing grid resolution.

    Author

    Thomas E. Murphy (tem@umd.edu)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    nlayers = np.asarray(nlayers, int)
    if nlayers.size == 1:
        nlayers = np.repeat(nlayers, 4)
    if nlayers.size != 4:
        raise ValueError("nlayers must be scalar or length-4")

    factor = np.asarray(factor)
    if factor.size == 1:
        factor = np.repeat(factor, 4)
    if factor.size != 4:
        raise ValueError("factor must be scalar or length-4")

    if len(method) != 4:
        raise ValueError("method must be length-4")
    method = method.upper()

    # --------------------------------------------
    # Internal helper to apply a stretch
    # --------------------------------------------
    def apply_stretch(arr, n, f, method_char):
        if n <= 0 or f == 1:
            return arr

        N = arr.size
        if method_char in "ULPG":
            pass
        else:
            raise ValueError("method must contain only U, L, P, G")

        # Determine slice indices
        if direction == "high":
            kv = slice(N - n - 1, N)
            q1 = arr[N - n - 1]
            q2 = arr[N - 1]
        else:
            kv = slice(0, n + 1)
            q1 = arr[n]
            q2 = arr[0]

        # Perform the appropriate mapping
        if method_char == "U":  # Uniform
            c = np.polyfit([q1, q2], [q1, q1 + f * (q2 - q1)], 1)
            new_vals = np.polyval(c, arr[kv])

        elif method_char == "L":  # Linear / polynomial
            c = (f - 1) / (q2 - q1)
            b = 1 - 2 * c * q1
            a = q1 - b * q1 - c * q1**2
            new_vals = a + b * arr[kv] + c * arr[kv] ** 2

        elif method_char == "P":  # Parabolic
            new_vals = arr[kv] + (f - 1) * (arr[kv] - q1) ** 3 / (q2 - q1) ** 2

        elif method_char == "G":  # Geometric
            alpha = np.real(f)
            B = _solve_b(alpha)
            A = (q2 - q1) / B
            new_vals = q1 + A * (np.exp((arr[kv] - q1) / A) - 1)

        # Promote arr only if needed
        arr = _maybe_promote(arr, new_vals)
        arr[kv] = new_vals
        return arr

    # --------------------------------------------
    # Y: north (0) and south (1)
    # --------------------------------------------
    for idx, direction in [(0, "high"), (1, "low")]:
        y = apply_stretch(y, nlayers[idx], factor[idx], method[idx])

    # --------------------------------------------
    # X: east (2) and west (3)
    # --------------------------------------------
    for idx, direction in [(2, "high"), (3, "low")]:
        x = apply_stretch(x, nlayers[idx], factor[idx], method[idx])

    # --------------------------------------------
    # Cell centers & increments
    # (they inherit real/complex type from x and y)
    # --------------------------------------------
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    dx = np.diff(x)
    dy = np.diff(y)

    return x, y, xc, yc, dx, dy

def padmesh(eps, x, y, nlayers):
    """
    Pad a 2D permittivity grid by adding extra layers of cells around
    the existing computational window.

    New cells are added by *extrapolating* (copying) the outermost
    values of eps; the interior coordinates and epsilon values are
    not modified.

    Parameters
    ----------
    eps : array_like, shape (ny, nx)
        Relative permittivity (or refractive-index squared) on the
        original cell-centered grid.
    x, y : array_like
        1D arrays of node coordinates along x and y.  If x has
        length nx_nodes and y has length ny_nodes, then
        eps.shape == (ny_nodes-1, nx_nodes-1).
    nlayers : sequence of 4 ints
        Number of additional *cells* to add on each side in the order
        (north, south, east, west):
            - north: +y side (top rows)
            - south: -y side (bottom rows)
            - east : +x side (right columns)
            - west : -x side (left columns)

    Returns
    -------
    x, y : ndarray
        Updated node coordinates including the padded region.
    xc, yc : ndarray
        Updated cell-center coordinates.
    nx, ny : int
        Number of cells in x and y after padding.
    eps : ndarray
        Padded permittivity array with shape (ny, nx).

    Notes
    -----
    The new node coordinates are generated by extending the existing
    grid with the same spacing as the outermost interval on each side.
    For example, if ``dx_w = x[1] - x[0]`` then ``nlayers[3]`` new
    nodes are inserted to the west at

        x[0] - dx_w, x[0] - 2*dx_w, ...

    similarly for the other sides.

    The outermost values of eps are replicated into the padded
    region (i.e., constant extrapolation).
    """
    eps = np.asarray(eps)
    x = np.asarray(x)
    y = np.asarray(y)

    if eps.ndim != 2:
        raise ValueError("eps must be a 2D array")

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    ny_eps, nx_eps = eps.shape
    if ny_eps != y.size - 1 or nx_eps != x.size - 1:
        raise ValueError(
            "Inconsistent shapes: eps.shape should be (len(y)-1, len(x)-1). "
            f"Got eps.shape={eps.shape}, len(x)={x.size}, len(y)={y.size}."
        )

    try:
        nN, nS, nE, nW = map(int, nlayers)
    except Exception as exc:
        raise ValueError("nlayers must be an iterable of four integers "
                         "(north, south, east, west)") from exc

    if min(nN, nS, nE, nW) < 0:
        raise ValueError("nlayers entries must be non-negative")

    # --------------------------
    # Pad eps in y (south, north)
    # --------------------------
    eps_padded = eps

    if nS > 0:
        # South: prepend rows copied from the first row
        south_row = eps_padded[0, :][np.newaxis, :]
        south_block = np.repeat(south_row, nS, axis=0)
        eps_padded = np.vstack([south_block, eps_padded])

    if nN > 0:
        # North: append rows copied from the last row
        north_row = eps_padded[-1, :][np.newaxis, :]
        north_block = np.repeat(north_row, nN, axis=0)
        eps_padded = np.vstack([eps_padded, north_block])

    # --------------------------
    # Pad eps in x (west, east)
    # --------------------------
    if nW > 0:
        # West: prepend columns copied from the first column
        west_col = eps_padded[:, 0][:, np.newaxis]
        west_block = np.repeat(west_col, nW, axis=1)
        eps_padded = np.hstack([west_block, eps_padded])

    if nE > 0:
        # East: append columns copied from the last column
        east_col = eps_padded[:, -1][:, np.newaxis]
        east_block = np.repeat(east_col, nE, axis=1)
        eps_padded = np.hstack([eps_padded, east_block])

    # --------------------------
    # Extend node coordinates
    # --------------------------
    # y: nodes, south (-y) first
    if y.size < 2:
        raise ValueError("y must contain at least two nodes")
    dy_south = y[1] - y[0]
    dy_north = y[-1] - y[-2]

    if nS > 0:
        add_y_s = y[0] - dy_south * np.arange(nS, 0, -1)
        y = np.concatenate([add_y_s, y])

    if nN > 0:
        add_y_n = y[-1] + dy_north * np.arange(1, nN + 1)
        y = np.concatenate([y, add_y_n])

    # x: nodes, west (-x) first
    if x.size < 2:
        raise ValueError("x must contain at least two nodes")
    dx_west = x[1] - x[0]
    dx_east = x[-1] - x[-2]

    if nW > 0:
        add_x_w = x[0] - dx_west * np.arange(nW, 0, -1)
        x = np.concatenate([add_x_w, x])

    if nE > 0:
        add_x_e = x[-1] + dx_east * np.arange(1, nE + 1)
        x = np.concatenate([x, add_x_e])

    # --------------------------
    # Cell centers and sizes
    # --------------------------
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    nx = xc.size
    ny = yc.size

    # Final consistency check
    if eps_padded.shape != (ny, nx):
        raise RuntimeError(
            "Shape mismatch after padding: eps.shape = "
            f"{eps_padded.shape}, expected ({ny}, {nx})."
        )

    return x, y, xc, yc, nx, ny, eps_padded