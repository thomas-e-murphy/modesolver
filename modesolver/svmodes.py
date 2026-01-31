import numpy as np
import warnings
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigs

from .sparse_solve import make_shift_invert_operator


def svmodes(wavelength, guess, nmodes, dx, dy, eps, boundary, field, *, solver=None):
    """
    This function calculates the modes of a dielectric waveguide using the
    semivectorial finite difference method.

    USAGE:

        neff, phi = svmodes(wavelength, guess, nmodes, dx, dy, eps,
                            boundary, field)

    INPUT:

        wavelength :    optical wavelength

        guess      :    scalar shift to apply when calculating the eigenvalues.
                        This routine will return the eigenpairs which have an
                        effective index closest to this guess.

        nmodes     :    number of modes to calculate.

        dx         :    horizontal grid spacing.
                        May be a scalar (uniform grid) or a vector giving the
                        grid spacing along the second index of eps. For a
                        non-uniform grid, dx should have length equal to the
                        number of columns of eps.

        dy         :    vertical grid spacing.
                        May be a scalar (uniform grid) or a vector giving the
                        grid spacing along the first index of eps. For a
                        non-uniform grid, dy should have length equal to the
                        number of rows of eps.

        eps        :    index mesh (n^2(x,y)); a 2-D array with shape (ny, nx).

        boundary   :    4-letter string specifying boundary conditions to be
                        applied at the edges of the computation window.

                        boundary[0] = North boundary condition
                        boundary[1] = South boundary condition
                        boundary[2] = East  boundary condition
                        boundary[3] = West  boundary condition

                        The following boundary conditions are supported:

                            'A' : field is antisymmetric
                            'S' : field is symmetric
                            '0' : field is zero immediately outside the
                                  boundary (Dirichlet)

        field      :    which field to compute:
                            'EX'     - semivectorial Ex formulation
                            'EY'     - semivectorial Ey formulation
                            'scalar' - scalar Helmholtz approximation

        solver     :    Sparse linear solver for shift-invert eigenvalue problem.
                        If None (default), automatically selects the best available:
                            Real matrices:    PyPardiso > MUMPS > SuperLU
                            Complex matrices: MUMPS > SuperLU
                        May be explicitly set to 'pypardiso', 'mumps', or 'superlu'.

    OUTPUT:

        neff       :    vector of modal effective indices, shape (nmodes,)

        phi        :    three-dimensional array containing the requested field
                        component for each computed mode, shape (ny, nx, nmodes),
                        where (ny, nx) = eps.shape.

                        The semivectorial fields (Ex, Ey, or scalar) are
                        located at the *center* of each finite-difference cell
                        (not at the mesh nodes). 

                        If nmodes == 1, phi is returned as a 2-D array of
                        shape (ny, nx) and neff as a scalar.

    NOTES:

    1)  The units are arbitrary, but they must be self-consistent (e.g., if
        wavelength is in µm, then dx and dy should also be in µm).

    2)  The finite difference discretization yields a sparse eigenvalue problem
        of the form A·phi = β²·phi, where β is the propagation constant.
        The effective index is then neff = β / k, with k = 2π / wavelength.

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """

    k = 2.0 * np.pi / wavelength

    boundary = boundary.upper()
    if len(boundary) != 4:
        raise ValueError("boundary string must have exactly four characters")

    if any(ch not in {"A", "S", "0"} for ch in boundary):
        raise ValueError("boundary string may contain only 'A', 'S', or '0'")

    field = field.lower()
    if field not in {"ex", "ey", "scalar"}:
        raise ValueError("field must be 'EX', 'EY', or 'scalar'")

    eps = np.asarray(eps)
    if eps.ndim != 2:
        raise ValueError("eps must be a 2-D array")

    ny, nx = eps.shape
    N = nx * ny

    eps = np.pad(eps, pad_width=1, mode="edge")

    if np.isscalar(dx):   # uniform x grid
        dx = np.full(nx + 2, dx)
    else:                 # nonuniform x grid
        dx = np.asarray(dx)
        dx = np.pad(dx, pad_width=1, mode='edge')

    if np.isscalar(dy):   # uniform y grid
        dy = np.full(ny + 2, dy)
    else:                 # nonuniform y grid
        dy = np.asarray(dy)
        dy = np.pad(dy, pad_width=1, mode='edge')

    e = np.tile((dx[1:-1] + dx[2:])/2, ny)
    w = np.tile((dx[1:-1] + dx[0:-2])/2, ny)
    p = np.tile(dx[1:-1], ny)

    n = np.repeat((dy[1:-1] + dy[2:])/2, nx)
    s = np.repeat((dy[1:-1] + dy[0:-2])/2, nx)
    q = np.repeat(dy[1:-1], nx)

    en = eps[2:ny+2, 1:nx+1].ravel(order="C")  # north
    es = eps[ 0:ny,  1:nx+1].ravel(order="C")  # south
    ee = eps[1:ny+1, 2:nx+2].ravel(order="C")  # east
    ew = eps[1:ny+1, 0:nx  ].ravel(order="C")  # west
    ep = eps[1:ny+1, 1:nx+1].ravel(order="C")  # center

    if field == "ex":
        un = 2.0 / n / (n + s)
        us = 2.0 / s / (n + s)
        ue = (
            8.0 * (p * (ep - ew) + 2.0 * w * ew) * ee
            / (
                (p * (ep - ee) + 2.0 * e * ee) * (p**2 * (ep - ew) + 4.0 * w**2 * ew)
                + (p * (ep - ew) + 2.0 * w * ew) * (p**2 * (ep - ee) + 4.0 * e**2 * ee)
            )
        )
        uw = (
            8.0 * (p * (ep - ee) + 2.0 * e * ee) * ew
            / (
                (p * (ep - ee) + 2.0 * e * ee) * (p**2 * (ep - ew) + 4.0 * w**2 * ew)
                + (p * (ep - ew) + 2.0 * w * ew) * (p**2 * (ep - ee) + 4.0 * e**2 * ee)
            )
        )
        up = ep * k**2 - un - us - ue * ep / ee - uw * ep / ew

    elif field == "ey":
        un = (
            8.0 * (q * (ep - es) + 2.0 * s * es) * en
            / (
                (q * (ep - en) + 2.0 * n * en) * (q**2 * (ep - es) + 4.0 * s**2 * es)
                + (q * (ep - es) + 2.0 * s * es) * (q**2 * (ep - en) + 4.0 * n**2 * en)
            )
        )
        us = (
            8.0 * (q * (ep - en) + 2.0 * n * en) * es
            / (
                (q * (ep - en) + 2.0 * n * en) * (q**2 * (ep - es) + 4.0 * s**2 * es)
                + (q * (ep - es) + 2.0 * s * es) * (q**2 * (ep - en) + 4.0 * n**2 * en)
            )
        )
        ue = 2.0 / e / (e + w)
        uw = 2.0 / w / (e + w)
        up = ep * k**2 - un * ep / en - us * ep / es - ue - uw

    else:  # field == "scalar"
        un = 2.0 / n / (n + s)
        us = 2.0 / s / (n + s)
        ue = 2.0 / e / (e + w)
        uw = 2.0 / w / (e + w)
        up = ep * k**2 - un - us - ue - uw

    jj = np.arange(nx*ny).reshape((ny, nx), order="C")

    # NORTH BOUNDARY CONDITION
    jb = jj[ny-1,:].ravel(order="C")
    if boundary[0] == "S":
        up[jb] += un[jb]
    elif boundary[0] == "A":
        up[jb] -= un[jb]

    # SOUTH BOUNDARY CONDITION
    jb = jj[0,:].ravel(order="C")
    if boundary[1] == "S":
        up[jb] += us[jb]
    elif boundary[1] == "A":
        up[jb] -= us[jb]

    # EAST BOUNDARY CONDITION
    jb = jj[:,nx-1].ravel(order="C")
    if boundary[2] == "S":
        up[jb] += ue[jb]
    elif boundary[2] == "A":
        up[jb] -= ue[jb]

    # WEST BOUNDARY CONDITION
    jb = jj[:,0].ravel(order="C")
    if boundary[3] == "S":
        up[jb] += uw[jb]
    elif boundary[3] == "A":
        up[jb] -= uw[jb]

    jall = jj.ravel(order="C")
    jn = jj[1:ny,:].ravel(order="C")       # north neighbor indices
    js = jj[0:ny-1,:].ravel(order="C")     # south neighbor indices
    je = jj[:,1:nx].ravel(order="C")       # east neighbor indices
    jw = jj[:,0:nx-1].ravel(order="C")     # west neighbor indices

    rows = np.concatenate([jall, jw, je, js, jn])
    cols = np.concatenate([jall, je, jw, jn, js])
    data = np.concatenate([up[jall], ue[jw], uw[je], un[js], us[jn]])

    A = coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    A.eliminate_zeros()

    # Solve the sparse eigenvalue problem A·phi = β²·phi
    shift = (2.0 * np.pi * guess / wavelength) ** 2  # = (k*guess)^2

    OPinv, _ = make_shift_invert_operator(A, shift, solver=solver)
    vals, vecs = eigs(A, k=nmodes, sigma=shift, OPinv=OPinv, which="LM", tol=1e-8)

    if np.iscomplexobj(A.data):
        neff = np.zeros((nmodes,), dtype=complex)
        phi = np.zeros((ny, nx, nmodes), dtype=complex)
    else:
        neff = np.zeros((nmodes,), dtype=float)
        phi = np.zeros((ny, nx, nmodes), dtype=float)

        if not np.allclose(vals.imag, 0.0, 1e-12):
            warnings.warn(
                "Unexpected complex eigenvalues, discarding imaginary parts",
                RuntimeWarning,
            )
        vals = vals.real

        if not np.allclose(vecs.imag, 0.0, 1e-12):
            warnings.warn(
                "Unexpected complex eigenvectors, discarding imaginary parts",
                RuntimeWarning,
            )
        vecs = vecs.real

    # Effective indices neff from eigenvalues β²: neff = sqrt(β²)/k
    neff = np.sqrt(vals) / k

    # Sort modes by descending effective index
    order = np.argsort(-neff.real)
    neff = neff[order]
    vecs = vecs[:, order]

    for m in range(nmodes):
        v = vecs[:, m]
        vmax = np.max(np.abs(v))
        norm = 1.0 if vmax == 0 else vmax
        phi[:, :, m] = (v / norm).reshape((ny, nx), order="C")

    # If only one mode requested, return 2-D field and scalar neff
    if nmodes == 1:
        phi = phi[:, :, 0]
        neff = neff[0]

    return neff, phi
