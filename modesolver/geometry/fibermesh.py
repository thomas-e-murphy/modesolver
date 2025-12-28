import numpy as np

def fibermesh(n, r, dx, dy):
    """
    Construct a 2D finite-difference mesh for a multilayer circular fiber.

    This routine generates the vertex coordinates, cell-center coordinates,
    and permittivity distribution for a circularly symmetric fiber cross-section.
    The radial structure is specified by a sequence of concentric layers, each
    with its own refractive index and outer radius. An additional cladding
    region of width side is included outside the outermost layer to
    reduce boundary effects.

    The resulting mesh is suitable for finite-difference mode solvers operating 
    on a rectangular (x, y) grid.

    Parameters

    n : array_like
        Refractive indices of the concentric fiber layers. The last element
        corresponds to the outermost layer.
    r : array_like
        Outer radius of each layer (same length as n), in the same units
        as dx and dy.  The final element in r will define the rectangular limits
        of the computational window, and remaining points in the computational
        domain will be filled with this index.
    dx, dy : float
        Horizontal and vertical mesh spacings.

    Returns

    x, y : ndarray
        Vertex coordinates of the mesh in the x and y directions.
        x has length (nx + 1) and y has length ny + 1.
    xc, yc : ndarray
        Cell-center coordinates:
            xc has length nx and yc has length ny.
    nx, ny : int
        Number of interior cells in the x and y directions.
    eps : ndarray of shape (ny, nx)
        Permittivity distribution on the cell-centered grid, where each layer
        is assigned n[j]**2 inside its specified radius.

    Notes
    - The permittivity array eps uses the convention eps[iy, ix],
      where iy indexes the y-coordinate (rows) and ix indexes the
      x-coordinate (columns).
    - The radial coordinate is computed at cell centers, and each point is
      assigned to the innermost layer whose outer radius exceeds rho.
    """
    # Convert inputs to NumPy arrays
    n = np.asarray(n, dtype=complex)
    r = np.asarray(r, float)

    nsquared = n**2
    if np.allclose(nsquared.imag, 0.0):
        nsquared = nsquared.real.astype(float)

    if n.shape != r.shape:
        raise ValueError("n and r must have the same shape")

    max_r = float(np.max(r))

    # Number of cells in x and y
    nx = int(round((max_r) / dx))
    ny = int(round((max_r) / dy))

    # Cell-center coordinates (strictly positive)
    xc = (np.arange(1, nx + 1) * dx) - dx / 2.0
    yc = (np.arange(1, ny + 1) * dy) - dy / 2.0

    # Vertex coordinates
    x = np.arange(0, nx + 1) * dx
    y = np.arange(0, ny + 1) * dy

    # Build radial coordinate on the (y, x) cell-centered grid
    # Xc, Yc have shape (ny, nx) with rows=y, cols=x
    Xc, Yc = np.meshgrid(xc, yc)
    rho = np.sqrt(Xc**2 + Yc**2)

    nlayers = n.size

    # Start with outermost layer everywhere
    eps = np.full((ny, nx), n[-1]**2, dtype=complex)

    # Fill inner layers from outer-1 down to 0
    for jj in range(nlayers - 2, -1, -1):
        eps[rho < r[jj]] = nsquared[jj]

    # Downcast to real, if all elements of eps are real
    if np.iscomplexobj(eps) and np.allclose(eps.imag, 0.0, atol=1e-12):
        eps = eps.real.astype(float)

    return x, y, xc, yc, nx, ny, eps
