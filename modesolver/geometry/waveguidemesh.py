import numpy as np

def waveguidemesh(n, h, rh, rw, side, dx, dy, return_edges=False):
    """
    Construct a rectangular finite-difference mesh for a multilayer slab
    waveguide with an optional ridge region on top.

    This routine generates the vertex coordinates, cell-center coordinates,
    and permittivity distribution for a 2D waveguide cross-section. The
    vertical structure is specified by a sequence of material layers (each with
    its own refractive index and thickness), and the topmost layer may include
    a ridge of height rh and width rw. An optional left-hand side
    region expands the computational domain.

    The resulting mesh is suitable for finite-difference mode solvers or
    propagation solvers operating on a rectangular grid.

    Parameters
    
    n : array_like
        Refractive index of each vertical layer. Must have the same length
        as h. The permittivity profile is computed as eps = n**2`

    h : array_like
        Thickness of each vertical layer. The number of vertical
        layers is nlayers = len(h).

    rh : float
        Ridge height. Determines how many grid cells of the
        topmost layer are replaced with the ridge index.

    rw : float
        Ridge width. Determines the horizontal extent of the ridge.

    side : float
        Width of the left-hand “side region”, used to enlarge the
        computational window beyond the ridge.

    dx, dy : float
        Horizontal and vertical mesh spacings.

    return_edges : bool, optional
        If True, also return a list of polygonal edge curves outlining the
        waveguide structure (useful for plotting). If False (default), only
        return the mesh arrays and permittivity grid.

    Returns
    
    x, y : ndarray
        Vertex coordinates of the mesh in the x and y directions.

    xc, yc : ndarray
        Cell-center coordinates:
            xc = x[:-1] + dx/2
            yc = y[:-1] + dy/2

    nx, ny : int
        Number of interior cells in the x and y directions.

    eps : ndarray of shape (ny, nx)
        Permittivity distribution on the cell-centered grid, where each
        vertical layer is assigned n[j]**2 and the ridge region is drawn
        over the upper portion of the topmost layer.

    edges : list of ndarrays, optional
        Returned only if return_edges=True.
        Each entry is an array of shape (M, 2) containing polygon vertices
        outlining layer interfaces or ridge boundaries. Useful for plotting
        mesh overlays.

    Notes
    
    - Vertical layers are mapped to integer grid rows using
      ih = round(h / dy).
    - Ridge height and width are similarly converted to integer cell counts.
    - The permittivity matrix eps is assigned row-by-row from bottom to top.
    - The edges list contains line segments that trace the material
      boundaries for visualization.

    """

    n = np.asarray(n, dtype=complex) # index of vertical layers
    h = np.asarray(h) # thickness of each layer
    nlayers = len(h)

    nsquared = n**2
    if np.allclose(nsquared.imag, 0.0):
        nsquared = nsquared.real.astype(float)

    ih = np.rint(h / dy).astype(int)
    irh = int(round(rh / dy))
    irw = int(round(rw / dx))
    iside = int(round(side / dx))

    nx_nodes = irw + iside + 1
    ny_nodes = ih.sum() + 1

    x = np.arange(nx_nodes) * dx
    y = np.arange(ny_nodes) * dy
    xc = (np.arange(1, nx_nodes) * dx) - dx/2
    yc = (np.arange(1, ny_nodes) * dy) - dy/2

    nx = xc.size
    ny = yc.size

    eps = np.zeros((ny, nx), dtype=complex)

    iy = 0
    for jj in range(nlayers):
        for _ in range(ih[jj]):
            eps[iy,:] = nsquared[jj]
            iy += 1

    iy = ih.sum() - ih[-1] - 1
    for _ in range(irh):
        eps[iy, irw:irw+iside] = nsquared[-1]
        iy -= 1

    # Downcast to real, if all elements of eps are real
    if np.iscomplexobj(eps) and np.allclose(eps.imag, 0.0, atol=1e-12):
        eps = eps.real.astype(float)
        
    if not return_edges:
        return x, y, xc, yc, nx, ny, eps

    edges = []
    iyp = np.cumsum(ih)

    for jj in range(nlayers - 2):
        if iyp[jj] >= (iyp[nlayers-2] - irh):
            x_edge = dx * np.array([0, irw])
        else:
            x_edge = dx * np.array([0, irw + iside])
        y_edge = dy * np.array([iyp[jj], iyp[jj]])
        edges.append(np.column_stack((x_edge, y_edge)))

    jj = nlayers - 2
    x_edge = dx * np.array([0, irw, irw, irw + iside])
    y_edge = dy * np.array([iyp[jj], iyp[jj], iyp[jj] - irh, iyp[jj] - irh])
    edges.append(np.column_stack((x_edge, y_edge)))

    return x, y, xc, yc, nx, ny, eps, edges