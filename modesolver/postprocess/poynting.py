import numpy as np

def poynting(ex, ey, ezj, hx, hy, hzj):
    """
    Compute the time-averaged Poynting vector ⟨S⟩ for an optical mode
    (or set of modes) using the electric- and magnetic-field components
    evaluated on a finite-difference grid.

    This routine computes the time averaged intensity

        ⟨S⟩ = (1/2) Re{ E x H* }

    where the electric-field components (Ex, Ey, j·Ez) and magnetic-field
    components (Hx, Hy, j·Hz) are supplied as inputs.  All H fields are
    assumed to be defined on the *vertices* of the finite-difference mesh,
    and this function returns the Poynting-vector components at the *center*
    of each cell.

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
    
    If the input arrays have the shape (ny, nx, nmodes) or (ny+1, nx+1, nmodes),
    this routine automatically computes ⟨S⟩ for each mode and returns
    3-D arrays (ny, nx, nmodes).

    INPUTS
    
    ex, ey, ezj : ndarray
        Electric-field components at the cell centers:
            ex  = Ex
            ey  = Ey
            ezj = j·Ez   (real for passive waveguides)

        Shape:
            (ny, nx)              — single mode
            (ny, nx, nmodes)      — multiple modes

    hx, hy, hzj : ndarray
        Magnetic-field components defined on the (ny+1, nx+1) vertex grid.
        hzj = j·Hz.  This routine averages these onto cell centers
        before computing ⟨S⟩.

        Shape:
            (ny+1, nx+1)              — single mode
            (ny+1, nx+1, nmodes)      — multiple modes

    OUTPUTS:

    Sx, Sy, Sz : ndarray
        Components of the time-averaged Poynting vector ⟨S⟩ evaluated at
        the center of each cell.

        Shape:
            (ny, nx)              — single mode
            (ny, nx, nmodes)      — multiple modes

    NOTES

    • All field components must be defined on consistent finite-difference
      grids (E fields cell-centered, H fields vertex-centered).

    • The returned Poynting vector components retain the same overall
      normalization as the supplied fields.

    AUTHOR

        Thomas E. Murphy (tem@umd.edu)
    """
    multi_mode = (hx.ndim == 3)
    if multi_mode:
        nmodes = hx.shape[2]
        if np.ndim(neff) == 0:
            raise ValueError("Multi-mode hx/hy but scalar neff. Pass neff as a length-nmodes array.")
        if len(neff) != nmodes:
            raise ValueError("Length of neff must match number of modes in hx, hy.")
    else:
        nmodes = 1

    # if hx, hy, and neff contain multiple modes, then process each mode recursively and return 
    # stacked multidimensional arrays 
    if multi_mode:
        Sx_all = []
        Sy_all = []
        Sz_all = []
        
        for m in range(nmodes):
            Sx_m, Sy_m, Sz_m = efields(ex[:,:,m], ey[:,:,m], ezj[:,:,m],hx[:,:,m], hy[:,:,m], hzj[:,:,m])
            Sx_all.append(Sx_m)
            Sy_all.append(Sy_m)
            Sz_all.append(Sz_m)

        # stack into (ny, nx, nmodes)
        return ( np.stack(Sx_all, axis=2),
                 np.stack(Sy_all, axis=2),
                 np.stack(Sz_all, axis=2))
    
    # the rest is for the single-mode case:

    hxc  = (hx[1:,:-1] + hx[:-1,:-1] + hx[:-1,1:] + hx[1:,1:])/4
    hyc  = (hy[1:,:-1] + hy[:-1,:-1] + hy[:-1,1:] + hy[1:,1:])/4
    hzcj = (hzj[1:,:-1] + hzj[:-1,:-1] + hzj[:-1,1:] + hzj[1:,1:])/4

    Sx = -0.5*(ey*hzcj.conj() + ezj*hyc.conj()).imag
    Sy = +0.5*(ezj*hxc.conj() + ex*hzcj.conj()).imag
    Sz = +0.5*(ex * hyc.conj() - ey * hxc.conj()).real

    return Sx, Sy, Sz