import numpy as np

def efields(
        neff, hx, hy, hzj,
        wavelength, dx, dy,
        *,
        eps=None,
        epsxx=None, epsyy=None, epszz=None,
        epsxy=None, epsyx=None      
):
    """
    Compute the transverse and longitudinal electric-field components
    (Ex, Ey, j·Ez) at the *center* of each finite-difference cell, using
    the magnetic-field components Hx, Hy, and j·Hz provided at the *nodes*
    (mesh vertices).

    This routine implements the field-reconstruction procedure described in:

        A. B. Fallahkhair, K. S. Li, and T. E. Murphy,
        "Vector Finite Difference Modesolver for Anisotropic Dielectric
        Waveguides," J. Lightwave Technol. 26(11), 1423-1431 (2008).

    FIELD CONVENTIONS
    
    • For passive, loss-free dielectric waveguides, the transverse fields  
      (Ex, Ey, Hx, Hy) are real, while the longitudinal fields (Ez, Hz) are
      purely imaginary.  To keep all returned quantities real, this function 
      returns j·Ez rather than Ez itself.

    • Normalization: this function returns electric fields scaled by η₀⁻¹  
      (η₀ = 377 Ω), so the output fields Ex, Ey, j·Ez have the same
      physical dimensions as H.  All components carry a common relative
      normalization.

    MULTI-MODE SUPPORT
    
    If neff, hx, hy, and hzj contain multiple modes, this function
    automatically processes each mode independently and returns a 3-D array
    of shape (ny, nx, nmodes) for each electric-field component.

    INPUTS
    
    neff : float or array
        Effective index of the mode(s).  Scalar for a single mode, or an
        array of length nmodes for the multi-mode case.

    hx, hy, hzj : ndarray
        Magnetic-field components defined on the (ny+1, nx+1) vertex grid.
        Shapes:
            (ny+1, nx+1)              — single mode
            (ny+1, nx+1, nmodes)      — multiple modes
        hzj is Hz·j (which is real for passive waveguides)

    wavelength : float
        Free-space wavelength of the mode (same units as dx and dy).

    dx, dy : float, 1-D array, or 2-D array
        Grid spacing in x and y.  Each is broadcast to match the
        cell-centered permittivity array.

    Permittivity specification (keyword-only):
        • eps                  — isotropic permittivity
        • epsxx, epsyy, epszz  — diagonal anisotropy
        • epsxx, epsxy, epsyx, epsyy, epszz — fully anisotropic tensor
      All permittivity arrays must have the same cell-centered shape (ny, nx).

    OUTPUTS
    
    Ex, Ey, Ezj : ndarray
        Electric-field components at the cell centers.
        Shapes:
            (ny, nx)              — single mode
            (ny, nx, nmodes)      — multiple modes
        Ezj is j·Ez, real-valued for passive waveguides.

    NOTES
    
    • All returned fields are cell-centered averages derived from the
      vertex-based magnetic fields using finite-difference curl relations.

    • The permittivity tensor is applied at cell centers, consistent with
      the definition of Ex, Ey, and j·Ez.

    • Units are arbitrary but must be self-consistent.

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
        Ex_all = []
        Ey_all = []
        Ez_all = []
        
        for m in range(nmodes):
            Ex_m, Ey_m, Ez_m = efields(
                neff[m],
                hx[:,:,m], hy[:,:,m], hzj[:,:,m],
                wavelength, dx, dy,
                eps=eps,
                epsxx=epsxx, epsyy=epsyy, epszz=epszz,
                epsxy=epsxy, epsyx=epsyx
            )
            Ex_all.append(Ex_m)
            Ey_all.append(Ey_m)
            Ez_all.append(Ez_m)

        # stack into (ny, nx, nmodes)
        return ( np.stack(Ex_all, axis=2),
                 np.stack(Ey_all, axis=2),
                 np.stack(Ez_all, axis=2))
    
    # the rest is for the single-mode case:

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
    
    dx = np.broadcast_to(dx, epsxx.shape)
    dy = np.broadcast_to(dy, epsxx.shape)

    k = 2*np.pi/wavelength
    beta = 2*np.pi*neff/wavelength

    hx1 = hx[1:,:-1]
    hx2 = hx[:-1,:-1]
    hx3 = hx[:-1,1:]
    hx4 = hx[1:,1:]

    hy1 = hy[1:,:-1]
    hy2 = hy[:-1,:-1]
    hy3 = hy[:-1,1:]
    hy4 = hy[1:,1:]

    hzj1 = hzj[1:,:-1]
    hzj2 = hzj[:-1,:-1]
    hzj3 = hzj[:-1,1:]
    hzj4 = hzj[1:,1:]

    Hx        = (hx1+hx2+hx3+hx4)/4
    dHx_dy    = (hx1+hx4-hx2-hx3)/(2*dy)

    Hy        = (hy1+hy2+hy3+hy4)/4
    dHy_dx    = (hy3+hy4-hy1-hy2)/(2*dx)

    dHzj_dx   = (hzj3+hzj4-hzj1-hzj2)/(2*dx)
    dHzj_dy   = (hzj1+hzj4-hzj2-hzj3)/(2*dy)

    # From jωD = ∇ x H, we find
    # Dx   = -(1/k) ∂(Hz∙j)/∂y + (β/k) Hy
    # Dy   =  (1/k) ∂(Hz∙j)/∂x - (β/k) Hy
    # Dz∙j =  (1/k) (∂Hy/∂x - ∂Hx/∂y)

    Dx = -dHzj_dy/k + (beta/k)*Hy
    Dy = +dHzj_dx/k - (beta/k)*Hx
    Dzj = (dHy_dx - dHx_dy)/k

    Ex = (epsyy*Dx - epsxy*Dy)/(epsxx*epsyy - epsxy*epsyx)
    Ey = (epsxx*Dy - epsyx*Dx)/(epsxx*epsyy - epsxy*epsyx)
    Ezj = Dzj/epszz

    return Ex, Ey, Ezj