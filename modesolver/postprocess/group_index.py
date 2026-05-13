import numpy as np
from .collocate import collocate


def group_index(ex, ey, ezj, hx, hy, hzj, ng, n, dx, dy):
    """
    Compute the group index of an optical waveguide mode using the
    weighted integral formula:

        ng_mode = ∫∫ [(2·ng·n − n²)·|E|² + |H|²] dA
                  ─────────────────────────────────────
                  ∫∫ (E×H* + E*×H)·ẑ dA

    where ng(x,y) and n(x,y) are the local group and phase refractive
    indices of the constituent materials at each grid point.

    ISOTROPIC MATERIALS ONLY

    This formula assumes that the permittivity at each point is a scalar
    (isotropic), so that n(x,y) = sqrt(eps(x,y)).  The numerator weight
    (2·ng·n − n²) = d(ω·n²)/dω is the dispersive correction to the
    electromagnetic energy density.  For anisotropic materials the correct
    expression requires separate ng and n values for each tensor component;
    this scalar formula does not apply in that case.

    FIELD GRID HANDLING

    This function accepts both staggered (Yee) and pre-collocated fields.
    If the six field components have different spatial shapes they are
    automatically collocated to cell centres via linear interpolation
    before the integrals are evaluated.  This matches the behaviour of
    poynting().

    INTEGRATION

    The integrals are evaluated as a Riemann (midpoint-rule) sum weighted
    by cell areas.  For cell-centred fields this is second-order accurate
    and is the natural choice for the finite-difference modesolver grid.
    For a non-uniform grid the exact cell widths supplied via dx and dy
    are used, so the result is correct regardless of grid uniformity.

    MULTI-MODE SUPPORT

    If the field arrays have a third dimension (nmodes), a group index is
    computed independently for each mode and an array of shape (nmodes,)
    is returned.

    INPUTS

    ex, ey, ezj : ndarray
        Electric-field components (Ex, Ey, j·Ez), shape (ny, nx) or
        (ny, nx, nmodes).

    hx, hy, hzj : ndarray
        Magnetic-field components (Hx, Hy, j·Hz).  May be staggered
        with shape (ny+1, nx+1) or already collocated with shape (ny, nx).

    ng : ndarray, shape (ny, nx)
        Local group index of each material at each grid point,
        ng = n + ω·dn/dω.

    n : ndarray, shape (ny, nx)
        Local phase index of each material at each grid point.
        For isotropic media, n = sqrt(eps).

    dx : scalar or 1D array of length nx
        Cell widths in the x-direction.

    dy : scalar or 1D array of length ny
        Cell heights in the y-direction.

    OUTPUT

    ng_mode : float or ndarray of shape (nmodes,)
        Group index of each mode.

    AUTHOR

        Thomas E. Murphy (tem@umd.edu)
    """
    ex  = np.asarray(ex)
    ey  = np.asarray(ey)
    ezj = np.asarray(ezj)
    hx  = np.asarray(hx)
    hy  = np.asarray(hy)
    hzj = np.asarray(hzj)
    ng  = np.asarray(ng)
    n   = np.asarray(n)

    # Collocate to cell centres if fields are on staggered grids
    shapes = [f.shape[:2] for f in [ex, ey, ezj, hx, hy, hzj]]
    if len(set(shapes)) > 1:
        ex, ey, ezj, hx, hy, hzj = collocate(ex, ey, ezj, hx, hy, hzj)

    ny, nx = ex.shape[:2]

    # Build 2-D cell-area array from (possibly non-uniform) grid spacings
    dx_vec = np.full(nx, dx) if np.isscalar(dx) else np.asarray(dx).ravel()
    dy_vec = np.full(ny, dy) if np.isscalar(dy) else np.asarray(dy).ravel()
    dA = np.outer(dy_vec, dx_vec)           # (ny, nx)

    # Broadcast ng, n, dA along mode axis for 3-D (multi-mode) fields
    if ex.ndim == 3:
        ng = ng[:, :, np.newaxis]
        n  = n[:, :, np.newaxis]
        dA = dA[:, :, np.newaxis]

    # |Ez|² = |j·Ez|² = |ezj|², and likewise |Hz|² = |hzj|²
    E2 = np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ezj)**2
    H2 = np.abs(hx)**2 + np.abs(hy)**2 + np.abs(hzj)**2

    # Numerator: ∫∫ [(2·ng·n − n²)·|E|² + |H|²] dA
    num = np.sum(((2*ng*n - n**2) * E2 + H2) * dA, axis=(0, 1))

    # Denominator: ∫∫ (E×H* + E*×H)·ẑ dA = 2·Re{Ex·Hy* − Ey·Hx*}
    #
    # Note: poynting() returns Sz = 0.5·Re{Ex·Hy* − Ey·Hx*}, so the
    # denominator integrand here equals 4·Sz.
    den = np.sum(2 * (ex * hy.conj() - ey * hx.conj()).real * dA, axis=(0, 1))

    return num / den
