import numpy as np
import warnings
from scipy.sparse.linalg import eigs
from scipy.sparse import coo_matrix, bmat, diags, eye

def wgmodes_yee(
        wavelength, guess, nmodes, dx, dy, boundary,
        *,
        eps=None,
        epsxx=None, epsyy=None, epszz=None
):    
    """
    This function computes the eigenmodes of a dielectric waveguide, using the 
    finite difference method, applied on a transverse Yee mesh:

        │       │           FIELD COMPONENT     GRID DIMENSION
        ■───○───■────   ■ : Ez (node-centered)  (ny+1,nx+1)
        │       │       □ : Hz (cell-centered)  (ny,nx)
        ●   □   ●       ○ : Ex,Hy (x-edges)     (nx,ny+1)
        │       │       ● : Ey,Hx (y-edges)     (nx+1,ny)
        ■───○───■────       eps, epsxx/yy/zz    (ny,nx)
    
    The two transverse magnetic field components (Hx, Hy) are computed first, and 
    the remaining four field components are determined from these.

    USAGE:

        neff, ex, ey, ezj, hx, hy, hzj = wgmodes(wavelength, guess, nmodes, 
                        dx, dy, boundary, eps=eps)

        neff, ex, ey, ezj, hx, hy, hzj = wgmodes(wavelength, guess, nmodes, 
                        dx, dy, boundary, epsxx=xx, epsyy=yy, epszz=zz)

    INPUT:

        wavelength :    optical wavelength
        guess      :    scalar shift to apply when calculating the eigenvalues.
                        This routine will return the eigenpairs that have an
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
                            'E' : PEC boundary (H.n = 0, E x n = 0)
                            'M' : PMC boundary (E.n = 0, H x n = 0)
                            '0' : Hx and Hy are zero immediately outside of
                                the boundary (hard Dirichlet)

        Dielectric permittivity (specified with keyword arguments):
        1) eps (isotropic case), size (ny,nx)
        2) epsxx, epsyy, epszz (anisotropic, diagonal)

    OUTPUT:

        neff    :   effective indices of the modes (nmodes,)
        hx      :   transverse magnetic field component (ny,nx+1,nmodes)
        hy      :   transverse magnetic field component (ny+1,nx,nmodes)
        hzj     :   j·Hz, longitudinal magnetic field component (ny,nx,nmodes)
        ex      :   transverse electric field component (ny+1,nx,nmodes)
        ey      :   transverse electric field component (ny,nx+1,nmodes)
        ezj     :   j·Ez, longitudinal electric field component (ny+1,nx+1,nmodes)

    NOTES:

    1)  The units are arbitrary, but they must be self-consistent
        (e.g., if lambda is in µm, then dx and dy should also be in µm).

    2)  This modesolver can optionally support non-uniform grid sizes. 
        To use this feature, you may let dx and/or dy be vectors that 
        match the number of columns/rows of eps

    3)  The modesolver can consider anisotropic materials, provided the
        permittivity of all constituent materials can be expressed in one of
        the following forms (in the principal axes):

        [eps  0   0 ]      [epsxx  0    0  ]
        [ 0  eps  0 ]  or  [  0  epsyy  0  ]
        [ 0   0  eps]      [  0    0  epszz]

        The program will decide which form is appropriate based upon the
        input keyword arguments supplied.

    4)  eps (ny,nx) is taken to be piecewise uniform within each cell of 
        size dx*dy.  Hz is sampled at the center of each cell and therefore
        has the same dimensions as eps.  Ez is sampled at the corners or
        vertices of each cell, and therefore has dimensions (ny+1,nx+1).
        Ex and Hy are sampled at the top and bottom of each cell, and 
        Ey and Hx are sampled at the left and right edges of each cell

    5)  Perfectly matched boundary layers can be accommodated by using the
        complex coordinate stretching technique at the edges of the
        computation window. 

    6)  For a real-valued passive dielectric waveguide, the transverse field 
        components are real, and the longitudinal field components are imaginary.
        The longitudinal field components ez and hy are here returned as 
        ez·j and hz·j, which are real

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """

    k0 = 2*np.pi/wavelength  # vacuum wavenumber

    if eps is not None:
        # isotropic case
        epsxx = epsyy = epszz = np.asarray(eps)
    elif all(v is not None for v in [epsxx, epsyy, epszz]):
        # diagonal anisotropy
        epsxx, epsyy, epszz = map(np.asarray, [epsxx, epsyy, epszz])
    else:
        raise ValueError("Must supply either eps or epsxx/yy/zz.")

    if not (epsxx.shape == epsyy.shape == epszz.shape):
        raise ValueError("All eps* components must have the same shape")
    
    boundary = boundary.upper()

    if len(boundary) != 4:
        raise ValueError("boundary string must have exactly four characters")

    if any(ch not in set("EM0") for ch in boundary):
        raise ValueError("boundary string may contain only E, M, or 0")
    
    sign = {"0":  0.0,   # Hx = Hy = Hz = 0 just outside boundary
            "E": +1.0,   # PEC -> H_parallel is symmetric (even) 
            "M": -1.0 }  # PMC -> H_parallel is antisymmetric (odd)

    ny, nx = epsxx.shape

    epsxx = np.pad(epsxx, pad_width=1, mode='edge')
    epsyy = np.pad(epsyy, pad_width=1, mode='edge')
    epszz = np.pad(epszz, pad_width=1, mode='edge')

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

    # Yee grid (2D):
    #   │       │        FIELD COMPONENT     GRID DIMENSION     (x-index)       (y-index)
    #  ─■───○───■─   ■ : Ez (node-centered)  K = (ny+1)*(nx+1)  i = k % (nx+1)  j = k//(nx+1)
    #   │       │    □ : Hz (cell-centered)  L =  (ny) * (nx)   i = l % nx      j = l//nx
    #   ●   □   ●    ○ : Ex,Hy (x-edges)     M = (ny+1)* (nx)   i = m % nx      j = m//nx
    #   │       │    ● : Ey,Hx (y-edges)     N =  (ny) *(nx+1)  i = n % (nx+1)  j = n//(nx+1)
    #  ─■───○───■─       NOTE:  Fields are stored row-major (C-order)
    #   │       │    

    K = (ny+1)*(nx+1);  k = np.arange(K)
    L = ny*nx;          l = np.arange(L)
    M = (ny+1)*nx;      m = np.arange(M)
    N = ny*(nx+1);      n = np.arange(N)

    # At the PEC boundaries, the normal component of H must be zero.
    # Px, Py: sparse selection matrices that remove forced-zero Hx & Hy points
    # (at PEC boundaries).  They restrict the eigenproblem to the physical subspace
    # so that shift–invert factorization is nonsingular.  After computing 
    # eigenvalues in restricted space, multiplying by Px.T and Py.T will 
    # reinsert zeros.

    py = np.ones(M)             # Nonzero entries for Hy (and Ex)
    px = np.ones(N)             # Nonzero entries for Hx (and Ey)

    if (boundary[0]=='E'):      # If north boundary is PEC
        py[m//nx==ny] = 0       # Assume Hy = 0 on north boundary

    if (boundary[1]=='E'):      # If south boundary is PEC
        py[m//nx==0] = 0        # Assume Hy = 0 on south boundary

    if (boundary[2]=='E'):      # If east boundary is PEC
        px[n % (nx+1)==nx] = 0  # Assume Hx = 0 on east boundary

    if (boundary[3]=='E'):      # If west boundary is PEC
        px[n % (nx+1)==0] = 0   # Assume Hx = 0 on west boundary

    idx = np.nonzero(px)[0]     # indices where px == 1
    Px = eye(len(px), format="csr")[idx, :]
    idx = np.nonzero(py)[0]     # indices where py == 1
    Py = eye(len(py), format="csr")[idx, :]
    P = bmat([[Px, None],
              [None, Py]], format="csr")

    # Finite difference equations (Yee grid):
    #   ┌   ┐  ┌           ┐┌   ┐     ┌   ┐┌          ┐  ┌           ┐┌   ┐
    #   │Hx │  │ 0  -βI  Ay││Ex │     │Ex ││ϵx  0   0 │  │  0  βI -Cy││Hx │
    # k0│Hy │ =│βI   0  -Ax││Ey │,  k0│Ey ││ 0  ϵy  0 │ =│-βI  0   Cx││Hy │
    #   │Hzj│  │By  -Bx  0 ││Ezj│     │Ezj││ 0   0  ϵz│  │-Dy  Dx  0 ││Hzj│   
    #   └   ┘  └           ┘└   ┘     └   ┘└          ┘  └           ┘└   ┘
    
    # Finite difference operator: Ay·Ez ≈ ∂Ez/∂y
    # Ez is of length       K = (ny+1)*(nx+1)
    # Ay·Ez is of length    N = (ny)*(nx+1)
    # Ay is N x K sparse matrix

    v = np.repeat(dy[1:ny+1], nx+1) # distance between Ez(north) and Ez(south)
    j = k//(nx+1)                   # y-index for Ez grid
    ks = k[j<ny]                    # column index for Ez(south)
    kn = ks + (nx+1)                # column index for Ez(north)
    rows = np.concatenate([n, n])
    cols = np.concatenate([ks, kn])
    vals = np.concatenate([-1.0/v, +1.0/v])
    Ay = coo_matrix((vals, (rows, cols)), shape=(N, K)).tocsr()

    # Finite difference operator: Ax·Ez ≈ ∂Ez/∂x
    # Ez is of length       K = (ny+1)*(nx+1)
    # Ax·Ez is of length    M = (ny+1)*(nx)
    # Ax is M x K sparse matrix

    h = np.tile(dx[1:nx+1], ny+1)   # distance between Ez(east) and Ez(west)
    i = k % (nx+1)                  # x-index for Ez grid
    kw = k[i<nx]                    # column index for Ez(west)
    ke = kw + 1                     # column index for Ez(east)
    rows = np.concatenate([m, m])
    cols = np.concatenate([kw, ke])
    vals = np.concatenate([-1.0/h, +1.0/h])
    Ax = coo_matrix((vals, (rows, cols)), shape=(M, K)).tocsr()

    # Finite difference operator: By·Ex ≈ ∂Ex/∂y
    # Ex is of length     M = (ny+1)*(nx)
    # By·Ex is of length  L = (ny)*(nx)
    # By is L x M sparse matrix

    v = np.repeat(dy[1:ny+1], nx)   # distance between Ex(north) and Ex(south)
    j = m//nx                       # y index for Ex grid
    ms = m[j<ny]                    # column index for Ex(south)
    mn = ms + nx                    # column index for Ex(north)
    rows = np.concatenate([l, l])
    cols = np.concatenate([ms, mn])
    vals = np.concatenate([-1.0/v, +1.0/v])
    By = coo_matrix((vals, (rows, cols)), shape=(L, M)).tocsr()

    # Finite difference operator: Bx·Ey ≈ ∂Ey/∂x
    # Ey is of length     N = (ny)*(nx+1)
    # Bx·Ey is of length  L = (ny)*(nx)
    # Bx is L x N sparse matrix

    h = np.tile(dx[1:nx+1], ny)     # distance between Ey(east) and Ey(west)
    i = n % (nx+1)                  # x-index for Ey grid
    nw = n[i<nx]                    # column index for Ey(west)
    ne = nw + 1                     # column index for Ey(east)
    rows = np.concatenate([l, l])
    cols = np.concatenate([nw, ne])
    vals = np.concatenate([-1.0/h, +1.0/h])
    Bx = coo_matrix((vals, (rows, cols)), shape=(L, N)).tocsr()

    # Finite difference operator: Cy·Hz ≈ ∂Hz/∂y
    # Hz is of length     L = (ny)*(nx)
    # Cy·Hz is of length  M = (ny+1)*(nx)
    # Cy is M x L sparse matrix

    v = 0.5*(dy[0:ny+1] + dy[1:ny+2])   # v = (n+s)/2
    v = np.repeat(v, nx)            # distance between Hz(north) and Hz(south)
    j = m//nx                       # y-index for Ex (output) grid
    ms = m[j<ny]                    # rows (m) where north Hz neighbor exists
    mn = m[j>0]                     # rows (m) where south Hz neighbor exists
    vn = +1.0/v
    vs = -1.0/v
    vs[j==ny] += sign[boundary[0]]*vn[j==ny]
    vn[j==0]  += sign[boundary[1]]*vs[j==0]
    rows = np.concatenate([mn, ms])
    cols = np.concatenate([l, l])
    vals = np.concatenate([vs[mn], vn[ms]])
    Cy = coo_matrix((vals, (rows, cols)), shape=(M, L)).tocsr()

    # Finite difference operator: Cx·Hz ≈ ∂Hz/∂x
    # Hz is of length     L = (ny)*(nx)
    # Cx·Hz is of length  N = (ny)*(nx+1)
    # Cx is N X L sparse matrix

    h = 0.5*(dx[0:nx+1] + dx[1:nx+2])   # h = (e+w)/2
    h = np.tile(h, ny)              # distance between Hz(east) and Hz(west)
    i = n % (nx+1)                  # x-index for Ey (output) grid
    nw = n[i<nx]                    # rows (n) where east Hz neighbor exists
    ne = n[i>0]                     # rows (n) where west Hz neighbor exists
    vw = -1.0 / h                   # west coefficient
    ve = +1.0 / h                   # east coefficient
    vw[i==nx] += sign[boundary[2]]*ve[i==nx]
    ve[i==0]  += sign[boundary[3]]*vw[i==0]
    rows = np.concatenate([ne, nw])
    cols = np.concatenate([l, l])
    vals = np.concatenate([vw[ne], ve[nw]])
    Cx = coo_matrix((vals, (rows, cols)), shape=(N, L)).tocsr()

    # Finite difference operator: Dy·Hx ≈ ∂Hx/∂y
    # Hx is of length    N =  (ny) *(nx+1)
    # Dy·Hx is of length K = (ny+1)*(nx+1)
    # Dy is K x N sparse matrix

    v = 0.5*(dy[0:ny+1] + dy[1:ny+2])   # v = (n+s)/2
    v = np.repeat(v, nx + 1)        # distance between Hx(north) and Hx(south)
    j = k//(nx+1)                   # y-index for Ez (output) grid
    ks = k[j<ny]                    # rows (k) where north Hx neighbor exists
    kn = k[j>0]                     # rows (k) where south Hx neighbor exists
    vn = +1.0 / v                   # north coefficient
    vs = -1.0 / v                   # south coefficient
    vs[j==ny] += sign[boundary[0]]*vn[j==ny]
    vn[j==0]  += sign[boundary[1]]*vs[j==0]
    rows = np.concatenate([ks, kn])
    cols = np.concatenate([n, n])
    vals = np.concatenate([vn[ks], vs[kn]])
    Dy = coo_matrix((vals, (rows, cols)), shape=(K, N)).tocsr()

    # Finite difference operator: Dx·Hy ≈ ∂Hy/∂x
    # Hy is of length    M = (ny+1)* (nx)
    # Dx·Hy is of length K = (ny+1)*(nx+1)
    # Dx is K x M sparse matrix

    h = 0.5*(dx[0:nx+1] + dx[1:nx+2])   # h = (e+w)/2
    h = np.tile(h, ny + 1)          # distance between Hy(east) and Hy(west)
    i = k % (nx+1)                  # x-index for Ez (output) grid
    kw = k[i<nx]                    # rows (k) where east Hy neighbor exists
    ke = k[i>0]                     # rows (k) where west Hy neighbor exists
    ve = +1.0 / h                   # east coefficient
    vw = -1.0 / h                   # west coefficient
    vw[i==nx] += sign[boundary[2]]*ve[i==nx]
    ve[i==0]  += sign[boundary[3]]*vw[i==0]
    rows = np.concatenate([ke, kw])
    cols = np.concatenate([m, m])
    vals = np.concatenate([vw[ke], ve[kw]])
    Dx = coo_matrix((vals, (rows, cols)), shape=(K, M)).tocsr()
    
    # ϵx = geometric weighted average of adjacent north and south cells
    epsS = epsxx[0:ny+1, 1:nx+1]   # south-adjacent epsxx
    epsN = epsxx[1:ny+2, 1:nx+1]   # north-adjacent epsxx
    s = dy[0:ny+1]
    n = dy[1:ny+2]
    d = ((n[:, None]*epsN + s[:, None]*epsS) / (n + s)[:, None]).ravel(order="C")
    epsx = diags(d, 0, format="csr")
    epsx_inv = diags(1.0/d, 0, format="csr")

    # ϵy = geometric weighted average of adjacent east and west cells
    epsW = epsyy[1:ny+1, 0:nx+1]   # west-adjacent epsyy
    epsE = epsyy[1:ny+1, 1:nx+2]   # east-adjacent epsyy
    w = dx[0:nx+1]
    e = dx[1:nx+2]
    d = ((e[None, :]*epsE + w[None, :]*epsW) / (e + w)[None, :]).ravel(order="C")
    epsy = diags(d, 0, format="csr")
    epsy_inv = diags(1.0/d, 0, format="csr")

    # ϵz = geometric weighted average of adjacent SW, NW, NE, and SE cells
    epsSW = epszz[0:ny+1, 0:nx+1]
    epsSE = epszz[0:ny+1, 1:nx+2]
    epsNW = epszz[1:ny+2, 0:nx+1]
    epsNE = epszz[1:ny+2, 1:nx+2]
    d = ((
        (n[:, None] * w[None, :] * epsNW) +
        (s[:, None] * w[None, :] * epsSW) +
        (s[:, None] * e[None, :] * epsSE) +
        (n[:, None] * e[None, :] * epsNE)
    ) / ((n + s)[:, None] * (e + w)[None, :])).ravel(order="C")
    # epsz = diags(d, 0, format="csr")     # note: epsz is not needed
    epsz_inv = diags(1.0/d, 0, format="csr")

    Uxx = (Cx @ Bx + epsy @ Ay @ epsz_inv @ Dy + 
           (k0**2) * epsy + (1.0/k0**2) * Cx @ (Bx @ Ay - By @ Ax) @ epsz_inv @ Dy)
    Uxy = (Cx @ By - epsy @ Ay @ epsz_inv @ Dx - 
           (1.0/k0**2) * Cx @ (Bx @ Ay - By @ Ax) @ epsz_inv @ Dx)
    Uyx = (Cy @ Bx - epsx @ Ax @ epsz_inv @ Dy - 
           (1.0/k0**2) * Cy @ (By @ Ax - Bx @ Ay) @ epsz_inv @ Dy)
    Uyy = (Cy @ By + epsx @ Ax @ epsz_inv @ Dx + 
           (k0**2) * epsx + (1.0/k0**2) * Cy @ (By @ Ax - Bx @ Ay) @ epsz_inv @ Dx)

    # Eigenvalue equation:  U·Ht = β·Ht, where Ht = (Hx,Hy)
    U = bmat([[Uxx, Uxy],
              [Uyx, Uyy]], format="csr")

    # Use the selection matrix P to exclude points on PEC boundary for which Hx = 0 or Hy = 0 
    # This restricts the eigenproblem to the physical subspace where H fields are non-zero
    # The zero elements are later added back by multiplying the eigenvector by P.T

    U = P @ U @ P.T
    
    # Longitudinal component is computed from transverse components using:
    # jβHz = B·Ht, where B is sparse matrix

    B = bmat([[Bx - (1.0/k0**2) * (By @ Ax - Bx @ Ay) @ epsz_inv @ Dy,
               By + (1.0/k0**2) * (By @ Ax - Bx @ Ay) @ epsz_inv @ Dx]], format="csr")

    shift = (k0*guess)**2
    vals, vecs = eigs(U, k=nmodes, sigma=shift, which="LM")

    # allocate space for result
    if np.iscomplexobj(U):
        # U is complex -> Keep (neff, Hx, Hy) complex
        neff = np.zeros((nmodes), dtype=complex)
        Hx = np.zeros((ny, nx+1, nmodes), dtype=complex)
        Hy = np.zeros((ny+1, nx, nmodes), dtype=complex)
        Hzj = np.zeros((ny, nx, nmodes), dtype=complex)
        Ex = np.zeros((ny+1, nx, nmodes), dtype=complex)
        Ey = np.zeros((ny, nx+1, nmodes), dtype=complex)
        Ezj = np.zeros((ny+1, nx+1, nmodes), dtype=complex)
    else:
        # U is real -> Downcast (neff, Hx, Hy) to real (discard imaginary)
        neff = np.zeros((nmodes,), dtype=float)
        Hx = np.zeros((ny, nx+1, nmodes), dtype=float)
        Hy = np.zeros((ny+1, nx, nmodes), dtype=float)
        Hzj = np.zeros((ny, nx, nmodes), dtype=float)
        Ex = np.zeros((ny+1, nx, nmodes), dtype=float)
        Ey = np.zeros((ny, nx+1, nmodes), dtype=float)
        Ezj = np.zeros((ny+1, nx+1, nmodes), dtype=float)
        if not np.allclose(vals.imag, 0, 1e-12):
            warnings.warn("Unexpected complex eigenvalues, discarding imaginary parts", RuntimeWarning)
        vals = vals.real
        if not np.allclose(vecs.imag, 0, 1e-12):
            warnings.warn("Unexpected complex eigenvectors, discarding imaginary parts", RuntimeWarning)
        vecs = vecs.real

    # NOTE:  neff will downcast to real (discard imaginary components) if U was real
    neff = np.sqrt(vals)/k0

    order = np.argsort(-neff.real)
    neff = neff[order]
    vecs = vecs[:, order]

    for m in range(nmodes):
        beta = k0*neff[m]
        v = vecs[:, m]
        v = P.T @ v         # reinsert zeros back into eigenvector
        norm = v[np.argmax(np.abs(v))]
        v = v/norm          # normalize mode to maximum value of 1

        hx = v[:N]
        hy = v[N:]
        hzj = B @ v / beta

        ex = (1.0/k0) * epsx_inv * (beta * hy - Cy @ hzj)
        ey = (1.0/k0) * epsy_inv * (-beta * hx + Cx @ hzj)
        ezj = (1.0/k0) * epsz_inv * (Dx @ hy - Dy @ hx)

        # NOTE:  Fields will downcast to real (discard imaginary components) if U was real
        Hx[:, :, m]  = hx.reshape((ny, nx+1), order="C")
        Hy[:, :, m]  = hy.reshape((ny+1, nx), order="C")
        Hzj[:, :, m] = hzj.reshape((ny, nx), order="C")
        Ex[:, :, m]  = ex.reshape((ny+1, nx), order="C")
        Ey[:, :, m]  = ey.reshape((ny, nx+1), order="C")
        Ezj[:, :, m] = ezj.reshape((ny+1, nx+1), order="C")

    # if nmodes == 1, Collapse fields to 2D arrays, and return scalar neff
    if nmodes == 1:  
        Ex = Ex[:, :, 0]
        Ey = Ey[:, :, 0]
        Ezj = Ezj[:, :, 0]
        Hx = Hx[:, :, 0]
        Hy = Hy[:, :, 0]
        Hzj = Hzj[:, :, 0]
        neff = neff[0] 

    return neff, Ex, Ey, Ezj, Hx, Hy, Hzj