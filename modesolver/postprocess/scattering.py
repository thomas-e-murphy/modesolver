import numpy as np
from .collocate import collocate


def gsm_step(modes1, modes2, dx, dy, *, normalize=False):
    """
    Compute generalized scattering matrix for a dielectric waveguide step.

    This function implements the mode-matching technique to compute the
    scattering matrix at a step discontinuity between two dielectric
    waveguide regions. The method computes overlap integrals between modes
    from each region to determine how power couples between modes.

    USAGE:

        S11, S12, S21, S22 = gsm_step(modes1, modes2, dx, dy)

    INPUTS:

        modes1 : tuple
            Mode data for region 1 (z < 0), as returned by wgmodes():
            (neff, Ex, Ey, Ezj, Hx, Hy, Hzj)
            Contains N modes with fields of shape (ny, nx) or (ny, nx, N).

        modes2 : tuple
            Mode data for region 2 (z > 0), as returned by wgmodes():
            (neff, Ex, Ey, Ezj, Hx, Hy, Hzj)
            Contains M modes with fields of shape (ny, nx) or (ny, nx, M).
            Must use the same grid (dx, dy) and domain size as modes1.

        dx, dy : float or ndarray
            Grid spacing in x and y directions (same for both mode sets).
            Can be scalars for uniform grids, or 1D arrays of shape (nx,) and
            (ny,) respectively for non-uniform grids.

        normalize : bool, optional
            If True, return the power-normalized scattering matrix where
            |S_mn|^2 directly gives the power coupling efficiency. If False
            (default), return the generalized scattering matrix.

    OUTPUTS:

        S11 : ndarray (N, N)
            Reflection matrix for region 1. S11[n, m] is the amplitude
            coefficient for reflection from mode m to mode n in region 1.

        S12 : ndarray (N, M)
            Transmission matrix from region 2 to region 1. S12[n, m] is
            the amplitude coefficient for transmission from mode m in
            region 2 to mode n in region 1.

        S21 : ndarray (M, N)
            Transmission matrix from region 1 to region 2. S21[m, n] is
            the amplitude coefficient for transmission from mode n in
            region 1 to mode m in region 2.

        S22 : ndarray (M, M)
            Reflection matrix for region 2. S22[m, n] is the amplitude
            coefficient for reflection from mode n to mode m in region 2.

    THEORY:

        The mode-matching technique enforces continuity of tangential E and H
        fields at the junction (z = 0). The fields in each region are expanded
        as sums over eigenmodes:

            E_t = sum_n (a_n + b_n) * e_n    (forward + backward waves)
            H_t = sum_n (a_n - b_n) * h_n

        where a_n are incident wave amplitudes and b_n are scattered amplitudes.

        The scattering matrix relates scattered to incident waves:

            [b1]   [S11  S12] [a1]
            [b2] = [S21  S22] [a2]

        The S-matrix is computed from overlap integrals:

            A_mn = integral (e1_n x h2_m) . z_hat dA
            Q1_n = integral (e1_n x h1_n) . z_hat dA
            Q2_m = integral (e2_m x h2_m) . z_hat dA

        The normalization factors Q are related to mode power by Q = 2*P.

    NOTES:

        - Both mode sets must be computed on the same grid with identical
          spatial dimensions. Use the same dx, dy, and domain (cladding) size.

        - For power conservation in lossless waveguides, the S-matrix should
          satisfy S^H @ S = I (unitary).

        - The coupling efficiency from mode n in region 1 to mode m in region 2
          is |S21[m, n]|^2.

        - For lossy waveguides with complex modes, use normalize=False. The
          generalized S-matrix is computed correctly for complex fields, but
          the power-normalized form (normalize=True) is only valid for lossless
          waveguides where the mode fields are real.

    REFERENCES:

        G. V. Eleftheriades, A. S. Omar, L. P. B. Katehi, and G. M. Rebeiz,
        "Some Important Properties of Waveguide Junction Generalized
        Scattering Matrices in the Context of the Mode Matching Technique,"
        IEEE Trans. Microwave Theory Tech., vol. 42, no. 10, pp. 1896-1903, 1994.

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """
    # Unpack mode tuples
    neff1, Ex1, Ey1, Ezj1, Hx1, Hy1, Hzj1 = modes1
    neff2, Ex2, Ey2, Ezj2, Hx2, Hy2, Hzj2 = modes2

    # Convert to arrays and ensure 3D (ny, nx, nmodes)
    Ex1, Ey1, Hx1, Hy1 = _ensure_3d(Ex1, Ey1, Hx1, Hy1)
    Ex2, Ey2, Hx2, Hy2 = _ensure_3d(Ex2, Ey2, Hx2, Hy2)
    Ezj1, Hzj1 = _ensure_3d(Ezj1, Hzj1)
    Ezj2, Hzj2 = _ensure_3d(Ezj2, Hzj2)

    # Collocate fields to cell centers if needed
    Ex1, Ey1, Ezj1, Hx1, Hy1, Hzj1 = _try_collocate(Ex1, Ey1, Ezj1, Hx1, Hy1, Hzj1)
    Ex2, Ey2, Ezj2, Hx2, Hy2, Hzj2 = _try_collocate(Ex2, Ey2, Ezj2, Hx2, Hy2, Hzj2)

    # Get number of modes and grid dimensions
    ny, nx, N = Ex1.shape  # modes in region 1
    M = Ex2.shape[2]       # modes in region 2

    # Verify grid compatibility
    if Ex1.shape[:2] != Ex2.shape[:2]:
        raise ValueError(
            f"Mode grids must have same spatial dimensions. "
            f"Region 1: {Ex1.shape[:2]}, Region 2: {Ex2.shape[:2]}"
        )

    # Compute cell areas (handles both uniform and non-uniform grids)
    cell_areas = _get_cell_areas(dx, dy, ny, nx)

    # Compute overlap matrix A: A[m, n] = integral (e1_n x h2_m) . z dA
    A = _overlap_matrix(Ex1, Ey1, Hx2, Hy2, cell_areas)

    # Compute normalization vectors
    Q1 = _normalization_vector(Ex1, Ey1, Hx1, Hy1, cell_areas)
    Q2 = _normalization_vector(Ex2, Ey2, Hx2, Hy2, cell_areas)

    # Build diagonal inverse matrices
    Q1_inv = np.diag(1.0 / Q1)
    Q2_inv = np.diag(1.0 / Q2)

    # Compute W matrices for each region
    # W  = Q1^-1 A^T Q2^-1 A  (N x N) - for S11, S21
    # W' = Q2^-1 A Q1^-1 A^T  (M x M) - for S22, S12
    W = Q1_inv @ A.T @ Q2_inv @ A
    Wp = Q2_inv @ A @ Q1_inv @ A.T

    I_N = np.eye(N)
    I_M = np.eye(M)

    # Solve for S-matrix blocks using correct mode-matching formulas:
    # S11 = (I - W)(I + W)^-1
    W_plus_I_inv = np.linalg.inv(I_N + W)
    S11 = (I_N - W) @ W_plus_I_inv

    # S21 = 2 Q2^-1 A (I + W)^-1
    S21 = 2.0 * Q2_inv @ A @ W_plus_I_inv

    # S22 = (W' - I)(I + W')^-1
    Wp_plus_I_inv = np.linalg.inv(I_M + Wp)
    S22 = (Wp - I_M) @ Wp_plus_I_inv

    # S12 = 2 Q1^-1 A^T (I + W')^-1
    S12 = 2.0 * Q1_inv @ A.T @ Wp_plus_I_inv

    if normalize:
        # Convert to power-normalized S-matrix where |S_mn|^2 = power ratio
        # S_norm = Q_out^(1/2) @ S @ Q_in^(-1/2)
        # For S11: both in and out are Q1
        # For S21: in is Q1, out is Q2
        # For S12: in is Q2, out is Q1
        # For S22: both in and out are Q2

        # Check for complex modes (lossy waveguides)
        Q1_imag_frac = np.max(np.abs(Q1.imag)) / np.max(np.abs(Q1))
        Q2_imag_frac = np.max(np.abs(Q2.imag)) / np.max(np.abs(Q2))
        if Q1_imag_frac > 1e-6 or Q2_imag_frac > 1e-6:
            raise ValueError(
                "normalize=True is not supported for lossy waveguides with complex modes. "
                "Use normalize=False to get the generalized scattering matrix."
            )

        sqrt_Q1 = np.sqrt(np.abs(Q1.real))
        sqrt_Q2 = np.sqrt(np.abs(Q2.real))

        S11 = np.diag(sqrt_Q1) @ S11 @ np.diag(1.0 / sqrt_Q1)
        S21 = np.diag(sqrt_Q2) @ S21 @ np.diag(1.0 / sqrt_Q1)
        S12 = np.diag(sqrt_Q1) @ S12 @ np.diag(1.0 / sqrt_Q2)
        S22 = np.diag(sqrt_Q2) @ S22 @ np.diag(1.0 / sqrt_Q2)

    return S11, S12, S21, S22


def _get_cell_areas(dx, dy, ny, nx):
    """
    Compute cell areas for uniform or non-uniform grids.

    Parameters
    ----------
    dx : float or ndarray
        Grid spacing in x. Scalar for uniform grid, or array of shape (nx,)
        for non-uniform grid.
    dy : float or ndarray
        Grid spacing in y. Scalar for uniform grid, or array of shape (ny,)
        for non-uniform grid.
    ny, nx : int
        Number of grid cells in y and x directions.

    Returns
    -------
    cell_areas : float or ndarray (ny, nx)
        Cell areas. Scalar if both dx and dy are scalars, otherwise 2D array.

    Raises
    ------
    ValueError
        If dx or dy arrays have incorrect lengths.
    """
    dx_is_scalar = np.ndim(dx) == 0
    dy_is_scalar = np.ndim(dy) == 0

    if dx_is_scalar and dy_is_scalar:
        # Uniform grid - return scalar
        return float(dx) * float(dy)

    # Non-uniform grid - convert to arrays and validate
    dx = np.atleast_1d(np.asarray(dx))
    dy = np.atleast_1d(np.asarray(dy))

    if len(dx) != nx:
        raise ValueError(
            f"dx array length ({len(dx)}) must match number of x cells ({nx})"
        )
    if len(dy) != ny:
        raise ValueError(
            f"dy array length ({len(dy)}) must match number of y cells ({ny})"
        )

    # Create 2D array of cell areas: cell_areas[i,j] = dy[i] * dx[j]
    return np.outer(dy, dx)


def _overlap_matrix(Ex1, Ey1, Hx2, Hy2, cell_areas):
    """
    Compute overlap integral matrix A_mn = integral (e1_n x h2_m) . z dA.

    In component form:
        A[m, n] = integral (Ex1[:,:,n] * Hy2[:,:,m] - Ey1[:,:,n] * Hx2[:,:,m]) dx dy

    Parameters
    ----------
    Ex1, Ey1 : ndarray (ny, nx, N)
        E-field components for N modes in region 1.
    Hx2, Hy2 : ndarray (ny, nx, M)
        H-field components for M modes in region 2.
    cell_areas : float or ndarray (ny, nx)
        Cell areas for integration (scalar for uniform grid, 2D array for non-uniform).

    Returns
    -------
    A : ndarray (M, N)
        Overlap matrix.
    """
    N = Ex1.shape[2]
    M = Hx2.shape[2]

    A = np.zeros((M, N), dtype=complex)

    for n in range(N):
        for m in range(M):
            # (E1 x H2) . z = Ex1 * Hy2 - Ey1 * Hx2
            integrand = (Ex1[:, :, n] * np.conj(Hy2[:, :, m])
                         - Ey1[:, :, n] * np.conj(Hx2[:, :, m]))
            A[m, n] = np.sum(integrand * cell_areas)

    return A


def _normalization_vector(Ex, Ey, Hx, Hy, cell_areas):
    """
    Compute normalization factors Q_n = integral (e_n x h_n) . z dA.

    This equals twice the mode power for normalized modes.

    Parameters
    ----------
    Ex, Ey : ndarray (ny, nx, N)
        E-field components.
    Hx, Hy : ndarray (ny, nx, N)
        H-field components.
    cell_areas : float or ndarray (ny, nx)
        Cell areas for integration (scalar for uniform grid, 2D array for non-uniform).

    Returns
    -------
    Q : ndarray (N,)
        Normalization factors.
    """
    N = Ex.shape[2]

    Q = np.zeros(N, dtype=complex)

    for n in range(N):
        integrand = (Ex[:, :, n] * np.conj(Hy[:, :, n])
                     - Ey[:, :, n] * np.conj(Hx[:, :, n]))
        Q[n] = np.sum(integrand * cell_areas)

    return Q


def _ensure_3d(*arrays):
    """Ensure arrays are 3D by adding a trailing dimension if needed."""
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        result.append(arr)
    return result if len(result) > 1 else result[0]


def _try_collocate(Ex, Ey, Ezj, Hx, Hy, Hzj):
    """Collocate fields if they have different shapes, otherwise return as-is."""
    shapes = [f.shape[:2] for f in [Ex, Ey, Ezj, Hx, Hy, Hzj]]
    if len(set(shapes)) == 1:
        # Already collocated
        return Ex, Ey, Ezj, Hx, Hy, Hzj
    else:
        return collocate(Ex, Ey, Ezj, Hx, Hy, Hzj)
