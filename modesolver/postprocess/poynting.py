import numpy as np
from .collocate import collocate


def poynting(ex, ey, ezj, hx, hy, hzj):
    """
    Compute the time-averaged Poynting vector ⟨S⟩ for an optical mode
    (or set of modes) using the electric- and magnetic-field components.

    This routine computes the time averaged intensity

        ⟨S⟩ = (1/2) Re{ E x H* }

    where the electric-field components (Ex, Ey, j·Ez) and magnetic-field
    components (Hx, Hy, j·Hz) are supplied as inputs.

    FIELD GRID HANDLING

    This function automatically detects whether the input fields are collocated
    (all at the same grid positions) or staggered (at different grid positions).

    • If all six fields have identical shapes, they are assumed to be collocated
      at cell centers, and the Poynting vector is computed directly.

    • If the fields have different shapes (differing by at most 1 in each
      dimension), the function automatically collocates them to cell centers
      using linear interpolation before computing the Poynting vector.

    This allows the function to work with fields from both wgmodes (H at
    vertices, E at cell centers) and wgmodes_yee (staggered Yee grid), as
    well as pre-collocated fields.

    FIELD CONVENTIONS

    • For passive, loss-free waveguides, the transverse fields (Ex, Ey,
      Hx, Hy) are real, while the longitudinal fields (Ez, Hz) are purely
      imaginary.  Accordingly, this routine uses Ezj = j·Ez and Hzj = j·Hz,
      which are real-valued in passive waveguides.

    • Units and normalization follow those of the input fields.  In typical
      usage with the vector finite-difference modesolver, the electric
      fields are scaled by η₀⁻¹ (η₀ = 377 Ω), so the returned Poynting
      vector is likewise normalized up to a constant factor.

    MULTI-MODE SUPPORT

    If the input arrays have a third dimension (nmodes), this routine
    automatically computes ⟨S⟩ for each mode and returns 3-D arrays
    (ny, nx, nmodes).

    INPUTS

    ex, ey, ezj : ndarray
        Electric-field components:
            ex  = Ex
            ey  = Ey
            ezj = j·Ez   (real for passive waveguides)

    hx, hy, hzj : ndarray
        Magnetic-field components. hzj = j·Hz.

    All fields may be 2D (single mode) or 3D (multiple modes with shape
    [..., nmodes]).  Field shapes may differ by at most 1 in each spatial
    dimension.

    OUTPUTS:

    Sx, Sy, Sz : ndarray
        Components of the time-averaged Poynting vector ⟨S⟩ evaluated at
        the center of each cell.

        Shape:
            (ny, nx)              — single mode
            (ny, nx, nmodes)      — multiple modes

    AUTHOR

        Thomas E. Murphy (tem@umd.edu)
    """
    ex = np.asarray(ex)
    ey = np.asarray(ey)
    ezj = np.asarray(ezj)
    hx = np.asarray(hx)
    hy = np.asarray(hy)
    hzj = np.asarray(hzj)

    # Check if fields are already collocated (all same spatial shape)
    shapes = [f.shape[:2] for f in [ex, ey, ezj, hx, hy, hzj]]
    already_collocated = (len(set(shapes)) == 1)

    if not already_collocated:
        # Collocate fields to cell centers
        ex, ey, ezj, hx, hy, hzj = collocate(ex, ey, ezj, hx, hy, hzj)

    # Now all fields are at cell centers with the same shape
    # Compute Poynting vector: ⟨S⟩ = (1/2) Re{ E x H* }
    #
    # S = E x H* gives:
    #   Sx = Ey*Hz* - Ez*Hy*
    #   Sy = Ez*Hx* - Ex*Hz*
    #   Sz = Ex*Hy* - Ey*Hx*
    #
    # With our convention that ezj = j*Ez and hzj = j*Hz (both real for
    # passive waveguides), we have Ez = ezj/j = -j*ezj and Hz = -j*hzj.
    #
    # Working through the algebra:
    #   Sx = Ey*(-j*hzj)* - (-j*ezj)*Hy* = Ey*(j*hzj) + j*ezj*Hy*
    #      = j*(Ey*hzj + ezj*Hy*)
    #   Re{Sx} = -Im{Ey*hzj + ezj*Hy*} = -(Ey*hzj + ezj*Hy*).imag
    #
    #   Sy = (-j*ezj)*Hx* - Ex*(-j*hzj)* = -j*ezj*Hx* + j*Ex*hzj
    #      = j*(Ex*hzj - ezj*Hx*)
    #   Re{Sy} = -Im{Ex*hzj - ezj*Hx*} = -(Ex*hzj - ezj*Hx*).imag
    #          = (ezj*Hx* - Ex*hzj).imag
    #
    #   Sz = Ex*Hy* - Ey*Hx*
    #   Re{Sz} = (Ex*Hy* - Ey*Hx*).real

    Sx = -0.5 * (ey * hzj.conj() + ezj * hy.conj()).imag
    Sy = +0.5 * (ezj * hx.conj() - ex * hzj.conj()).imag
    Sz = +0.5 * (ex * hy.conj() - ey * hx.conj()).real

    return Sx, Sy, Sz
