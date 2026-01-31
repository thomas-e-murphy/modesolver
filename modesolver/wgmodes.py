import numpy as np
import warnings
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix, bmat

from .sparse_solve import make_shift_invert_operator

def wgmodes(
        wavelength, guess, nmodes, dx, dy, boundary,
        *,
        eps=None,
        epsxx=None, epsyy=None, epszz=None,
        epsxy=None, epsyx=None,
        solver=None
):    
    """
    This function computes the two transverse magnetic field components of a 
    dielectric waveguide, using the finite difference method. For details about the 
    method, please consult: 

        A. B. Fallahkhair, K. S. Li and T. E. Murphy,
        "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides",
        J. Lightwave Technol. 26(11), 1423-1431 (2008).
        https://doi.org/10.1109/JLT.2008.923643

    USAGE:

        neff, hx, hy, hzj = wgmodes(wavelength, guess, nmodes, dx, dy, boundary, 
                        eps=eps)

        neff, hx, hy, hzj = wgmodes(wavelength, guess, nmodes, dx, dy, boundary,
                        epsxx=xx, epsyy=yy, epszz=zz)

        neff, hx, hy, hzj = wgmodes(wavelength, guess, nmodes, dx, dy, boundary,
                        epsxx=xx, epsxy=xy, epsyx=yx, epsyy=yy, epszz=zz)

    INPUT:

        wavelength :    optical wavelength
        guess      :    scalar shift to apply when calculating the eigenvalues.
                        This routine will return the eigenpairs which have an
                        effective index closest to this guess.
        nmodes     :    number of modes to calculate.
        dx         :    horizontal grid spacing (scalar or vector).
        dy         :    vertical grid spacing (scalar or vector).
        boundary   :    4-letter string specifying boundary conditions to be
                        applied at the edges of the computation window.

                        boundary[0] = North boundary condition
                        boundary[1] = South boundary condition
                        boundary[2] = East  boundary condition
                        boundary[3] = West  boundary condition

                        The following boundary conditions are supported:
                            'A' : Hx is antisymmetric, Hy is symmetric.
                            'S' : Hx is symmetric, Hy is antisymmetric.
                            'E' : PEC boundary (H.n = 0, E x n = 0)
                            'M' : PMC boundary (E.n = 0, H x n = 0)
                            '0' : Hx and Hy are zero immediately outside of
                                the boundary (hard Dirichlet)
        Dielectric permittivity (specified with keyword arguments):
        1) eps (isotropic case)
        2) epsxx, epsyy, epszz (anisotropic, diagonal)
        3) epsxx, epsxy, epsyx, epsyy, epszz (fully anisotropic)

        solver  :   Sparse linear solver for shift-invert eigenvalue problem.
                    If None (default), automatically selects the best available:
                        Real matrices:    PyPardiso > MUMPS > SuperLU
                        Complex matrices: MUMPS > SuperLU
                    May be explicitly set to 'pypardiso', 'mumps', or 'superlu'.

    OUTPUT:

        hx         :    transverse magnetic field component (ny+1,nx+1,nmodes)
        hy         :    transverse magnetic field component (ny+1,nx+1,nmodes)
        hzj        :    j·Hz, longitudinal magnetic field component (ny+1,nx+1,nmodes)
        neff       :    effective indices of the modes (nmodes,)

    NOTES:

    1)  The units are arbitrary, but they must be self-consistent
        (e.g., if lambda is in µm, then dx and dy should also be in µm).

    2)  Unlike the E-field modesolvers, this method calculates the
        transverse MAGNETIC field components Hx and Hy. Also, it calculates
        the components at the edges (vertices) of each cell, rather than in
        the center of each cell. As a result, if np.shape(eps) is (ny,nx), 
        then the output fields will have a size of (ny+1,nx+1).

    3)  This version of the modesolver can optionally support non-uniform
        grid sizes. To use this feature, you may let dx and/or dy be vectors
        that match the number of columns/rows of eps

    4)  The modesolver can consider anisotropic materials, provided the
        permittivity of all constituent materials can be expressed in one of
        the following forms (in the principal axes):

        [eps  0   0 ]    [epsxx  0    0  ]    [epsxx epsxy   0  ]
        [ 0  eps  0 ] or [  0  epsyy  0  ] or [epsyx epsyy   0  ]
        [ 0   0  eps]    [  0    0  epszz]    [  0     0   epszz]

        The program will decide which form is appropriate based upon the
        input arguments supplied.

    5)  Perfectly matched boundary layers can be accommodated by using the
        complex coordinate stretching technique at the edges of the
        computation window. 

    6)  The longitudinal magnetic field j·Hz is calculated and returned along
        with Hx and Hy.  For passive waveguides, Hz is imaginary, and the product
        j·Hz is real-valued.

    AUTHORS (from original MATLAB code):

        Thomas E. Murphy (tem@umd.edu)
        Arman B. Fallahkhair
        Kai Sum Li
    """

    k = 2*np.pi/wavelength  # vacuum wavenumber

    if eps is not None:
        # isotropic case
        epsxx = epsyy = epszz = np.asarray(eps)
        epsxy = epsyx = np.zeros_like(epsxx)
    elif all(v is not None for v in [epsxx, epsxy, epsyx, epsyy, epszz]):
        # full tensor form
        epsxx = np.asarray(epsxx)
        epsxy = np.asarray(epsxy)
        epsyx = np.asarray(epsyx)
        epsyy = np.asarray(epsyy)
        epszz = np.asarray(epszz)
    elif all(v is not None for v in [epsxx, epsyy, epszz]):
        # diagonal anisotropy
        epsxx, epsyy, epszz = map(np.asarray, [epsxx, epsyy, epszz])
        epsxy = epsyx = np.zeros_like(epsxx)
    else:
        raise ValueError("Must supply either eps, or epsxx/yy/zz, or full tensor.")

    if not (epsxx.shape == epsxy.shape == epsyx.shape == epsyy.shape == epszz.shape):
        raise ValueError("All eps* components must have the same shape")
    
    boundary = boundary.upper()

    if len(boundary) != 4:
        raise ValueError("boundary string must have exactly four characters")

    if any(ch not in set("ASEM0") for ch in boundary):
        raise ValueError("boundary string may contain only A, S, E, M, or 0")
    
    ny, nx = epsxx.shape
    nx += 1
    ny += 1
    N = nx*ny  # number of grid-points at which to compute Hx or Hy

    epsxx = np.pad(epsxx, pad_width=1, mode='edge')
    epsyy = np.pad(epsyy, pad_width=1, mode='edge')
    epszz = np.pad(epszz, pad_width=1, mode='edge')
    epsxy = np.pad(epsxy, pad_width=1, mode='edge')
    epsyx = np.pad(epsyx, pad_width=1, mode='edge')

    if np.isscalar(dx):   # uniform x grid
        dx = np.full(nx + 1, dx)
    else:                 # nonuniform x grid
        dx = np.asarray(dx)
        dx = np.pad(dx, pad_width=1, mode='edge')

    if np.isscalar(dy):   # uniform y grid
        dy = np.full(ny + 1, dy)
    else:                 # nonuniform y grid
        dy = np.asarray(dy)
        dy = np.pad(dy, pad_width=1, mode='edge')

    n = np.repeat(dy[1:ny+1], nx)
    s = np.repeat(dy[0:ny],   nx)
    e = np.tile(  dx[1:nx+1], ny)
    w = np.tile(  dx[0:nx],   ny)

    exx1 = np.asarray(epsxx[1:ny+1, 0:nx  ]).ravel(order="C")
    exx2 = np.asarray(epsxx[0:ny  , 0:nx  ]).ravel(order="C")
    exx3 = np.asarray(epsxx[0:ny  , 1:nx+1]).ravel(order="C")
    exx4 = np.asarray(epsxx[1:ny+1, 1:nx+1]).ravel(order="C")

    eyy1 = np.asarray(epsyy[1:ny+1, 0:nx  ]).ravel(order="C")
    eyy2 = np.asarray(epsyy[0:ny  , 0:nx  ]).ravel(order="C")
    eyy3 = np.asarray(epsyy[0:ny  , 1:nx+1]).ravel(order="C")
    eyy4 = np.asarray(epsyy[1:ny+1, 1:nx+1]).ravel(order="C")

    ezz1 = np.asarray(epszz[1:ny+1, 0:nx  ]).ravel(order="C")
    ezz2 = np.asarray(epszz[0:ny  , 0:nx  ]).ravel(order="C")
    ezz3 = np.asarray(epszz[0:ny  , 1:nx+1]).ravel(order="C")
    ezz4 = np.asarray(epszz[1:ny+1, 1:nx+1]).ravel(order="C")

    exy1 = np.asarray(epsxy[1:ny+1, 0:nx  ]).ravel(order="C")
    exy2 = np.asarray(epsxy[0:ny  , 0:nx  ]).ravel(order="C")
    exy3 = np.asarray(epsxy[0:ny  , 1:nx+1]).ravel(order="C")
    exy4 = np.asarray(epsxy[1:ny+1, 1:nx+1]).ravel(order="C")

    eyx1 = np.asarray(epsyx[1:ny+1, 0:nx  ]).ravel(order="C")
    eyx2 = np.asarray(epsyx[0:ny  , 0:nx  ]).ravel(order="C")
    eyx3 = np.asarray(epsyx[0:ny  , 1:nx+1]).ravel(order="C")
    eyx4 = np.asarray(epsyx[1:ny+1, 1:nx+1]).ravel(order="C")

    ns21 = n*eyy2 + s*eyy1
    ns34 = n*eyy3 + s*eyy4
    ew14 = e*exx1 + w*exx4
    ew23 = e*exx2 + w*exx3

    axxn = (
        ((2*eyy4*e - eyx4*n)*(eyy3/ezz4)/ns34)
        + ((2*eyy1*w + eyx1*n)*(eyy2/ezz1)/ns21)
    ) / (n*(e + w))

    axxs = (
        ((2*eyy3*e + eyx3*s)*(eyy4/ezz3)/ns34)
        + ((2*eyy2*w - eyx2*s)*(eyy1/ezz2)/ns21)
    ) / (s*(e + w))

    ayye = (
        (2*n*exx4 - e*exy4)*exx1/ezz4/e/ew14/(n + s)
        + (2*s*exx3 + e*exy3)*exx2/ezz3/e/ew23/(n + s)
    )

    ayyw = (
        (2*exx1*n + exy1*w)*exx4/ezz1/w/ew14/(n + s)
        + (2*exx2*s - exy2*w)*exx3/ezz2/w/ew23/(n + s)
    )

    axxe = 2/(e*(e + w)) + (eyy4*eyx3/ezz3 - eyy3*eyx4/ezz4)/(e + w)/ns34

    axxw = 2/(w*(e + w)) + (eyy2*eyx1/ezz1 - eyy1*eyx2/ezz2)/(e + w)/ns21

    ayyn = 2/(n*(n + s)) + (exx4*exy1/ezz1 - exx1*exy4/ezz4)/(n + s)/ew14

    ayys = 2/(s*(n + s)) + (exx2*exy3/ezz3 - exx3*exy2/ezz2)/(n + s)/ew23

    axxne = +eyx4*eyy3/ezz4/(e + w)/ns34
    axxse = -eyx3*eyy4/ezz3/(e + w)/ns34
    axxnw = -eyx1*eyy2/ezz1/(e + w)/ns21
    axxsw = +eyx2*eyy1/ezz2/(e + w)/ns21

    ayyne = +exy4*exx1/ezz4/(n + s)/ew14
    ayyse = -exy3*exx2/ezz3/(n + s)/ew23
    ayynw = -exy1*exx4/ezz1/(n + s)/ew14
    ayysw = +exy2*exx3/ezz2/(n + s)/ew23

    axxp = (
        -axxn - axxs - axxe - axxw - axxne - axxse - axxnw - axxsw
        + k**2*(n + s)*(eyy4*eyy3*e/ns34 + eyy1*eyy2*w/ns21)/(e + w)
    )

    ayyp = (
        -ayyn - ayys - ayye - ayyw - ayyne - ayyse - ayynw - ayysw
        + k**2*(e + w)*(exx1*exx4*n/ew14 + exx2*exx3*s/ew23)/(n + s)
    )

    axyn = (
        (eyy3*eyy4/ezz4/ns34 - eyy2*eyy1/ezz1/ns21
        + s*(eyy2*eyy4 - eyy1*eyy3)/ns21/ns34)
        / (e + w)
    )

    axys = (
        (eyy1*eyy2/ezz2/ns21 - eyy4*eyy3/ezz3/ns34
        + n*(eyy2*eyy4 - eyy1*eyy3)/ns21/ns34)
        / (e + w)
    )

    ayxe = (
        (exx1*exx4/ezz4/ew14 - exx2*exx3/ezz3/ew23
        + w*(exx2*exx4 - exx1*exx3)/ew23/ew14)
        / (n + s)
    )

    ayxw = (
        (exx3*exx2/ezz2/ew23 - exx4*exx1/ezz1/ew14
        + e*(exx4*exx2 - exx1*exx3)/ew23/ew14)
        / (n + s)
    )

    axye = (
        (eyy4*(1 - eyy3/ezz3) - eyy3*(1 - eyy4/ezz4))/ns34/(e + w)
        - 2*(
            eyx1*eyy2/ezz1*n*w/ns21
            + eyx2*eyy1/ezz2*s*w/ns21
            + eyx4*eyy3/ezz4*n*e/ns34
            + eyx3*eyy4/ezz3*s*e/ns34
            + eyy1*eyy2*(1/ezz1 - 1/ezz2)*w**2/ns21
            + eyy3*eyy4*(1/ezz4 - 1/ezz3)*e*w/ns34
        )/e/(e + w)**2
    )

    axyw = (
        (eyy2*(1 - eyy1/ezz1) - eyy1*(1 - eyy2/ezz2))/ns21/(e + w)
        - 2*(
            eyx4*eyy3/ezz4*n*e/ns34
            + eyx3*eyy4/ezz3*s*e/ns34
            + eyx1*eyy2/ezz1*n*w/ns21
            + eyx2*eyy1/ezz2*s*w/ns21
            + eyy4*eyy3*(1/ezz3 - 1/ezz4)*e**2/ns34
            + eyy2*eyy1*(1/ezz2 - 1/ezz1)*w*e/ns21
        )/w/(e + w)**2
    )

    ayxn = (
        (exx4*(1 - exx1/ezz1) - exx1*(1 - exx4/ezz4))/ew14/(n + s)
        - 2*(
            exy3*exx2/ezz3*e*s/ew23
            + exy2*exx3/ezz2*w*s/ew23
            + exy4*exx1/ezz4*e*n/ew14
            + exy1*exx4/ezz1*w*n/ew14
            + exx3*exx2*(1/ezz3 - 1/ezz2)*s**2/ew23
            + exx1*exx4*(1/ezz4 - 1/ezz1)*n*s/ew14
        )/n/(n + s)**2
    )

    ayxs = (
        (exx2*(1 - exx3/ezz3) - exx3*(1 - exx2/ezz2))/ew23/(n + s)
        - 2*(
            exy4*exx1/ezz4*e*n/ew14
            + exy1*exx4/ezz1*w*n/ew14
            + exy3*exx2/ezz3*e*s/ew23
            + exy2*exx3/ezz2*w*s/ew23
            + exx4*exx1*(1/ezz1 - 1/ezz4)*n**2/ew14
            + exx2*exx3*(1/ezz2 - 1/ezz3)*s*n/ew23
        )/s/(n + s)**2
    )

    axyne = +eyy3*(1 - eyy4/ezz4)/(e + w)/ns34
    axyse = -eyy4*(1 - eyy3/ezz3)/(e + w)/ns34
    axynw = -eyy2*(1 - eyy1/ezz1)/(e + w)/ns21
    axysw = +eyy1*(1 - eyy2/ezz2)/(e + w)/ns21

    ayxne = +exx1*(1 - exx4/ezz4)/(n + s)/ew14
    ayxse = -exx2*(1 - exx3/ezz3)/(n + s)/ew23
    ayxnw = -exx4*(1 - exx1/ezz1)/(n + s)/ew14
    ayxsw = +exx3*(1 - exx2/ezz2)/(n + s)/ew23

    axyp = (
        -(axyn + axys + axye + axyw + axyne + axyse + axynw + axysw)
        - k**2*(
            w*(n*eyx1*eyy2 + s*eyx2*eyy1)/ns21
            + e*(s*eyx3*eyy4 + n*eyx4*eyy3)/ns34
        )/(e + w)
    )

    ayxp = (
        -(ayxn + ayxs + ayxe + ayxw + ayxne + ayxse + ayxnw + ayxsw)
        - k**2*(
            n*(w*exy1*exx4 + e*exy4*exx1)/ew14
            + s*(w*exy2*exx3 + e*exy3*exx2)/ew23
        )/(n + s)
    )

    bh12 =  e*w / ((2*e + 2*w) * ( ezz1*ezz2*s/(2*eyy2) + ezz1*ezz2*n/(2*eyy1)))
    bh34 = -e*w / ((2*e + 2*w) * (-ezz3*ezz4*n/(2*eyy4) - ezz3*ezz4*s/(2*eyy3)))
    bv14 = -n*s / ((2*n + 2*s) * (-e*ezz1*ezz4/(2*exx4) - ezz1*ezz4*w/(2*exx1)))
    bv23 =  n*s / ((2*n + 2*s) * (-e*ezz2*ezz3/(2*exx3) - ezz2*ezz3*w/(2*exx2)))

    bxn = (
        bh12 * (-eyx1 * ezz2 / (2 * eyy1 * w) - ezz2 / n)
        + bh34 * (ezz3 / n - eyx4 * ezz3 / (2 * e * eyy4))
        + bv14
        * (
            e * exy4 * ezz1 * s / (exx4 * n**3 + exx4 * n**2 * s)
            - e * exy4 * ezz1 / (exx4 * n**2)
            + exy1 * ezz4 * s * w / (exx1 * n**3 + exx1 * n**2 * s)
            - ezz1 * s / (n**2 + n * s)
            + ezz1 / (2 * n)
            + ezz4 * s / (n**2 + n * s)
            - ezz4 / (2 * n)
            - ezz1 * ezz4 / (2 * exx4 * n)
            - exy1 * ezz4 * w / (exx1 * n**2)
            + ezz1 * ezz4 / (2 * exx1 * n)
        )
        + bv23
        * (
            -e * exy3 * ezz2 / (exx3 * n**2 + exx3 * n * s)
            - exy2 * ezz3 * w / (exx2 * n**2 + exx2 * n * s)
            - ezz2 * s / (n**2 + n * s)
            + ezz3 * s / (n**2 + n * s)
        )
    )
    bxs = (
        bh12 * (eyx2 * ezz1 / (2 * eyy2 * w) - ezz1 / s)
        + bh34 * (ezz4 / s + eyx3 * ezz4 / (2 * e * eyy3))
        + bv14
        * (
            -e * exy4 * ezz1 / (exx4 * n * s + exx4 * s**2)
            - exy1 * ezz4 * w / (exx1 * n * s + exx1 * s**2)
            + ezz1 * n / (n * s + s**2)
            - ezz4 * n / (n * s + s**2)
        )
        + bv23
        * (
            e * exy3 * ezz2 * n / (exx3 * n * s**2 + exx3 * s**3)
            - e * exy3 * ezz2 / (exx3 * s**2)
            + exy2 * ezz3 * n * w / (exx2 * n * s**2 + exx2 * s**3)
            + ezz2 * n / (n * s + s**2)
            - ezz2 / (2 * s)
            - ezz3 * n / (n * s + s**2)
            + ezz3 / (2 * s)
            + ezz2 * ezz3 / (2 * exx3 * s)
            - exy2 * ezz3 * w / (exx2 * s**2)
            - ezz2 * ezz3 / (2 * exx2 * s)
        )
    )
    bxe = (
        bh34
        * (
            eyx3 * ezz4 / (2 * e * eyy3)
            - eyx4 * ezz3 / (2 * e * eyy4)
            + ezz3 * ezz4 * n / (e**2 * eyy4)
            + ezz3 * ezz4 * s / (e**2 * eyy3)
        )
        + bv14 * (ezz1 / (2 * n) - ezz1 * ezz4 / (2 * exx4 * n))
        + bv23 * (-ezz2 / (2 * s) + ezz2 * ezz3 / (2 * exx3 * s))
    )
    bxw = (
        bh12
        * (
            -eyx1 * ezz2 / (2 * eyy1 * w)
            + eyx2 * ezz1 / (2 * eyy2 * w)
            - ezz1 * ezz2 * s / (eyy2 * w**2)
            - ezz1 * ezz2 * n / (eyy1 * w**2)
        )
        + bv14 * (-ezz4 / (2 * n) + ezz1 * ezz4 / (2 * exx1 * n))
        + bv23 * (ezz3 / (2 * s) - ezz2 * ezz3 / (2 * exx2 * s))
    )

    bxne = bh34 * eyx4 * ezz3 / (2 * e * eyy4) + bv14 * (
        -ezz1 / (2 * n) + ezz1 * ezz4 / (2 * exx4 * n)
    )
    bxse = -bh34 * eyx3 * ezz4 / (2 * e * eyy3) + bv23 * (
        ezz2 / (2 * s) - ezz2 * ezz3 / (2 * exx3 * s)
    )
    bxnw = bh12 * eyx1 * ezz2 / (2 * eyy1 * w) + bv14 * (
        ezz4 / (2 * n) - ezz1 * ezz4 / (2 * exx1 * n)
    )
    bxsw = -bh12 * eyx2 * ezz1 / (2 * eyy2 * w) + bv23 * (
        -ezz3 / (2 * s) + ezz2 * ezz3 / (2 * exx2 * s)
    )

    bxp = (
        bh12
        * (
            eyx1 * ezz2 / (2 * eyy1 * w)
            - eyx2 * ezz1 / (2 * eyy2 * w)
            + ezz1 / s
            + ezz2 / n
            + ezz1 * ezz2 * s / (eyy2 * w**2)
            + ezz1 * ezz2 * n / (eyy1 * w**2)
        )
        + bh34
        * (
            -ezz3 / n
            - ezz4 / s
            - eyx3 * ezz4 / (2 * e * eyy3)
            + eyx4 * ezz3 / (2 * e * eyy4)
            - ezz3 * ezz4 * n / (e**2 * eyy4)
            - ezz3 * ezz4 * s / (e**2 * eyy3)
        )
        + bv14
        * (
            e * exy4 * ezz1 / (exx4 * n * s)
            - ezz1 / s
            + ezz1 / (2 * n)
            + ezz4 / s
            - ezz4 / (2 * n)
            + ezz1 * ezz4 / (2 * exx4 * n)
            + exy1 * ezz4 * w / (exx1 * n * s)
            - ezz1 * ezz4 / (2 * exx1 * n)
        )
        + bv23
        * (
            e * exy3 * ezz2 / (exx3 * n * s)
            - ezz2 / (2 * s)
            + ezz2 / n
            + ezz3 / (2 * s)
            - ezz3 / n
            - ezz2 * ezz3 / (2 * exx3 * s)
            + exy2 * ezz3 * w / (exx2 * n * s)
            + ezz2 * ezz3 / (2 * exx2 * s)
        )
    )
    bxp += k**2 * (
        bh12 * (-ezz1 * ezz2 * n / 2 - ezz1 * ezz2 * s / 2)
        + bh34 * (ezz3 * ezz4 * n / 2 + ezz3 * ezz4 * s / 2)
        + bv14
        * (-e * exy4 * ezz1 * ezz4 / (2 * exx4) - exy1 * ezz1 * ezz4 * w / (2 * exx1))
        + bv23
        * (-e * exy3 * ezz2 * ezz3 / (2 * exx3) - exy2 * ezz2 * ezz3 * w / (2 * exx2))
    )

    byn = (
        bh12 * (ezz2 / (2 * w) - ezz1 * ezz2 / (2 * eyy1 * w))
        + bh34 * (ezz3 / (2 * e) - ezz3 * ezz4 / (2 * e * eyy4))
        + bv14
        * (
            e * ezz1 * ezz4 / (exx4 * n**2)
            - exy4 * ezz1 / (2 * exx4 * n)
            + exy1 * ezz4 / (2 * exx1 * n)
            + ezz1 * ezz4 * w / (exx1 * n**2)
        )
    )
    bys = (
        bh12 * (-ezz1 / (2 * w) + ezz1 * ezz2 / (2 * eyy2 * w))
        + bh34 * (-ezz4 / (2 * e) + ezz3 * ezz4 / (2 * e * eyy3))
        + bv23
        * (
            e * ezz2 * ezz3 / (exx3 * s**2)
            + exy3 * ezz2 / (2 * exx3 * s)
            - exy2 * ezz3 / (2 * exx2 * s)
            + ezz2 * ezz3 * w / (exx2 * s**2)
        )
    )
    bye = (
        bh12
        * (
            eyx1 * ezz2 * n / (e**2 * eyy1 + e * eyy1 * w)
            + eyx2 * ezz1 * s / (e**2 * eyy2 + e * eyy2 * w)
            - ezz1 * w / (e**2 + e * w)
            + ezz2 * w / (e**2 + e * w)
        )
        + bh34
        * (
            eyx3 * ezz4 * s * w / (e**3 * eyy3 + e**2 * eyy3 * w)
            + eyx4 * ezz3 * n * w / (e**3 * eyy4 + e**2 * eyy4 * w)
            - ezz3 * w / (e**2 + e * w)
            + ezz4 * w / (e**2 + e * w)
            + ezz3 / (2 * e)
            - ezz4 / (2 * e)
            - ezz3 * ezz4 / (2 * e * eyy4)
            + ezz3 * ezz4 / (2 * e * eyy3)
            - eyx3 * ezz4 * s / (e**2 * eyy3)
            - eyx4 * ezz3 * n / (e**2 * eyy4)
        )
        + bv14 * (-exy4 * ezz1 / (2 * exx4 * n) + ezz1 / e)
        + bv23 * (exy3 * ezz2 / (2 * exx3 * s) + ezz2 / e)
    )
    byw = (
        bh12
        * (
            -e * eyx1 * ezz2 * n / (e * eyy1 * w**2 + eyy1 * w**3)
            - e * eyx2 * ezz1 * s / (e * eyy2 * w**2 + eyy2 * w**3)
            + e * ezz1 / (e * w + w**2)
            - e * ezz2 / (e * w + w**2)
            + eyx1 * ezz2 * n / (eyy1 * w**2)
            + eyx2 * ezz1 * s / (eyy2 * w**2)
            - ezz1 / (2 * w)
            + ezz2 / (2 * w)
            + ezz1 * ezz2 / (2 * eyy2 * w)
            - ezz1 * ezz2 / (2 * eyy1 * w)
        )
        + bh34
        * (
            e * ezz3 / (e * w + w**2)
            - e * ezz4 / (e * w + w**2)
            - eyx3 * ezz4 * s / (e * eyy3 * w + eyy3 * w**2)
            - eyx4 * ezz3 * n / (e * eyy4 * w + eyy4 * w**2)
        )
        + bv14 * (ezz4 / w + exy1 * ezz4 / (2 * exx1 * n))
        + bv23 * (ezz3 / w - exy2 * ezz3 / (2 * exx2 * s))
    )

    byne = bh34 * (-ezz3 / (2 * e) + ezz3 * ezz4 / (2 * e * eyy4)) + bv14 * exy4 * ezz1 / (
        2 * exx4 * n
    )
    byse = bh34 * (ezz4 / (2 * e) - ezz3 * ezz4 / (2 * e * eyy3)) - bv23 * exy3 * ezz2 / (
        2 * exx3 * s
    )
    bynw = bh12 * (-ezz2 / (2 * w) + ezz1 * ezz2 / (2 * eyy1 * w)) - bv14 * exy1 * ezz4 / (
        2 * exx1 * n
    )
    bysw = bh12 * (ezz1 / (2 * w) - ezz1 * ezz2 / (2 * eyy2 * w)) + bv23 * exy2 * ezz3 / (
        2 * exx2 * s
    )

    byp = (
        bh12
        * (
            -ezz1 / (2 * w)
            + ezz2 / (2 * w)
            - ezz1 * ezz2 / (2 * eyy2 * w)
            + ezz1 * ezz2 / (2 * eyy1 * w)
            - eyx1 * ezz2 * n / (e * eyy1 * w)
            - eyx2 * ezz1 * s / (e * eyy2 * w)
            + ezz1 / e
            - ezz2 / e
        )
        + bh34
        * (
            -ezz3 / w
            + ezz4 / w
            + eyx3 * ezz4 * s / (e * eyy3 * w)
            + eyx4 * ezz3 * n / (e * eyy4 * w)
            + ezz3 / (2 * e)
            - ezz4 / (2 * e)
            + ezz3 * ezz4 / (2 * e * eyy4)
            - ezz3 * ezz4 / (2 * e * eyy3)
        )
        + bv14
        * (
            -e * ezz1 * ezz4 / (exx4 * n**2)
            - ezz4 / w
            + exy4 * ezz1 / (2 * exx4 * n)
            - exy1 * ezz4 / (2 * exx1 * n)
            - ezz1 * ezz4 * w / (exx1 * n**2)
            - ezz1 / e
        )
        + bv23
        * (
            -e * ezz2 * ezz3 / (exx3 * s**2)
            - ezz3 / w
            - exy3 * ezz2 / (2 * exx3 * s)
            + exy2 * ezz3 / (2 * exx2 * s)
            - ezz2 * ezz3 * w / (exx2 * s**2)
            - ezz2 / e
        )
    )
    byp += k**2 * (
        bh12 * (eyx1 * ezz1 * ezz2 * n / (2 * eyy1) + eyx2 * ezz1 * ezz2 * s / (2 * eyy2))
        + bh34
        * (-eyx3 * ezz3 * ezz4 * s / (2 * eyy3) - eyx4 * ezz3 * ezz4 * n / (2 * eyy4))
        + bv14 * (e * ezz1 * ezz4 / 2 + ezz1 * ezz4 * w / 2)
        + bv23 * (e * ezz2 * ezz3 / 2 + ezz2 * ezz3 * w / 2)
    )

    jj = np.arange(nx*ny).reshape((ny, nx), order="C")
    
    xns_pairs = [ (axxn, axxs), (axxne, axxse), (axxnw, axxsw),
                  (ayxn, ayxs), (ayxne, ayxse), (ayxnw, ayxsw)  ]

    yns_pairs = [ (ayyn, ayys), (ayyne, ayyse), (ayynw, ayysw),
                  (axyn, axys), (axyne, axyse), (axynw, axysw), (byn, bys)  ]

    xew_pairs = [ (axxe, axxw), (axxne, axxnw), (axxse, axxsw),
                  (ayxe, ayxw), (ayxne, ayxnw), (ayxse, ayxsw), (bxe, bxw)  ]

    yew_pairs = [ (ayye, ayyw), (ayyne, ayynw), (ayyse, ayysw),
                  (axye, axyw), (axyne, axynw), (axyse, axysw)  ]

    ns_sign = {"S": +1.0,    # Hx symmetric (even), Hy antisymmetric (odd) at N/S boundary
               "A": -1.0,    # Hx antisymmetric (odd), Hy symmetric (even) at N/S boundary
               "0":  0.0,    # Hx = Hy = 0 just outside boundary
               "E": +1.0,    # PEC -> Hy(odd), Hx(even) at N/S boundary
               "M": -1.0 }   # PMC -> Hy(even), Hx(odd) at N/S boundary

    ew_sign = {"S": +1.0,    # Hx symmetric (even), Hy antisymmetric (odd) at E/W boundary
               "A": -1.0,    # Hx antisymmetric (odd), Hy symmetric (even) at E/W boundary
               "0":  0.0,    # Hx = Hy = 0 just outside boundary
               "E": -1.0,    # PEC -> Hx(odd), Hy(even) at E/W boundary
               "M": +1.0 }   # PMC -> Hx(even), Hy(odd) at E/W boundary

    # NORTH BOUNDARY CONDITION
    if boundary[0] != "0":
        jb = jj[ny-1,:]   # indices of nx points on north boundary 
        for n, s in xns_pairs:
            s[jb] += ns_sign[boundary[0]] * n[jb]
        for n, s in yns_pairs:
            s[jb] -= ns_sign[boundary[0]] * n[jb]

    # SOUTH BOUNDARY CONDITION
    if boundary[1] != "0":
        jb = jj[0,:]   # indices of nx points on south boundary
        for n, s in xns_pairs:
            n[jb] += ns_sign[boundary[1]] * s[jb]
        for n, s in yns_pairs:
            n[jb] -= ns_sign[boundary[1]] * s[jb]

    # EAST BOUNDARY CONDITION
    if boundary[2] != "0":
        jb = jj[:,nx-1]   # indices of ny points on east boundary
        for e, w in xew_pairs:
            w[jb] += ew_sign[boundary[2]] * e[jb]
        for e, w in yew_pairs:
            w[jb] -= ew_sign[boundary[2]] * e[jb]

    # WEST BOUNDARY CONDITION
    if boundary[3] != "0":
        jb = jj[:,0]   # indices of ny points on west boundary
        for e, w in xew_pairs:
            e[jb] += ew_sign[boundary[3]] * w[jb]
        for e, w in yew_pairs:
            e[jb] -= ew_sign[boundary[3]] * w[jb]

    jall = jj.ravel(order="C")

    js = jj[:-1,:].ravel(order="C")
    jn = jj[1:,:].ravel(order="C")
    jw = jj[:,:-1].ravel(order="C")
    je = jj[:,1:].ravel(order="C")

    jsw = jj[:-1,:-1].ravel(order="C")
    jse = jj[:-1,1:].ravel(order="C")
    jnw = jj[1:,:-1].ravel(order="C")
    jne = jj[1:,1:] .ravel(order="C")

    rows = np.concatenate([jall, jw, je, js, jn, jne, jse, jsw, jnw])
    cols = np.concatenate([jall, je, jw, jn, js, jsw, jnw, jne, jse])

    axx =  np.concatenate([axxp[jall], axxe[jw], axxw[je], axxn[js], axxs[jn], axxsw[jne], axxnw[jse], axxne[jsw], axxse[jnw]])
    axy =  np.concatenate([axyp[jall], axye[jw], axyw[je], axyn[js], axys[jn], axysw[jne], axynw[jse], axyne[jsw], axyse[jnw]])
    ayx =  np.concatenate([ayxp[jall], ayxe[jw], ayxw[je], ayxn[js], ayxs[jn], ayxsw[jne], ayxnw[jse], ayxne[jsw], ayxse[jnw]])
    ayy =  np.concatenate([ayyp[jall], ayye[jw], ayyw[je], ayyn[js], ayys[jn], ayysw[jne], ayynw[jse], ayyne[jsw], ayyse[jnw]])

    Axx = coo_matrix((axx, (rows, cols)), shape=(N, N))
    Axy = coo_matrix((axy, (rows, cols)), shape=(N, N))
    Ayx = coo_matrix((ayx, (rows, cols)), shape=(N, N))
    Ayy = coo_matrix((ayy, (rows, cols)), shape=(N, N))

    bx = np.concatenate([bxp[jall], bxe[jw], bxw[je], bxn[js], bxs[jn], bxsw[jne], bxnw[jse], bxne[jsw], bxse[jnw]])
    by = np.concatenate([byp[jall], bye[jw], byw[je], byn[js], bys[jn], bysw[jne], bynw[jse], byne[jsw], byse[jnw]])
    
    Bx = coo_matrix((bx, (rows, cols)), shape=(N, N))
    By = coo_matrix((by, (rows, cols)), shape=(N, N))

    A = bmat([[Axx, Axy],
              [Ayx, Ayy]], format="csr")
    A.eliminate_zeros()

    B = bmat([[Bx,By]], format="csr")
    B.eliminate_zeros()

    shift = (k*guess)**2
    OPinv, _ = make_shift_invert_operator(A, shift, solver=solver)
    vals, vecs = eigs(A, k=nmodes, sigma=shift, OPinv=OPinv, which="LM")

    # allocate space for result
    if np.iscomplexobj(A):
        # A is complex -> Keep (neff, hx, hy) complex
        neff = np.zeros((nmodes), dtype=complex)
        hx = np.zeros((ny, nx, nmodes), dtype=complex)
        hy = np.zeros((ny, nx, nmodes), dtype=complex)
        hzj = np.zeros((ny, nx, nmodes), dtype=complex)
    else:
        # A is real -> Downcast (neff, hx, hy) to real (discard imaginary)
        neff = np.zeros((nmodes,), dtype=float)
        hx = np.zeros((ny, nx, nmodes), dtype=float)
        hy = np.zeros((ny, nx, nmodes), dtype=float)
        hzj = np.zeros((ny, nx, nmodes), dtype=float)
        if not np.allclose(vals.imag, 0, 1e-12):
            warnings.warn("Unexpected complex eigenvalues, discarding imaginary parts", RuntimeWarning)
        vals = vals.real
        if not np.allclose(vecs.imag, 0, 1e-12):
            warnings.warn("Unexpected complex eigenvectors, discarding imaginary parts", RuntimeWarning)
        vecs = vecs.real

    # NOTE:  neff will downcast to real (discard imaginary components) if A was real
    neff = np.sqrt(vals)/k

    order = np.argsort(-neff.real)
    neff = neff[order]
    vecs = vecs[:, order]

    for m in range(nmodes):
        beta = k*neff[m]
        v = vecs[:, m]

        hzj_m = (B @ v) / beta
        hx_m = v[:N]
        hy_m = v[N:]

        # find index where |Hx|**2 + |Hy|**2 is largest
        imax = np.argmax(np.abs(hx_m)**2 + np.abs(hy_m)**2) 
        # pick larger of Hx or Hy at that point 
        norm = max([hx_m[imax], hy_m[imax]], key=abs)        # Normalization factor

        # NOTE:  hx,hy will downcast to real (discard imaginary components) if A was real
        hx[:, :, m] = (hx_m / norm).reshape((ny, nx), order="C")
        hy[:, :, m] = (hy_m / norm).reshape((ny, nx), order="C")
        hzj[:, :, m] = (hzj_m / norm).reshape((ny, nx), order="C")
        

    # if nmodes == 1, Collapse hx, hy to 2D arrays, and return scalar neff
    if nmodes == 1:  
        hx = hx[:, :, 0]
        hy = hy[:, :, 0]
        hzj = hzj[:, :, 0]
        neff = neff[0] 

    return neff, hx, hy, hzj