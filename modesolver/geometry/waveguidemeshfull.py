import numpy as np

def waveguidemeshfull(n, h, rh, rw, side, dx, dy, return_edges=False):
    """
    Construct a rectangular finite-difference mesh for a multilayer slab
    waveguide with a central ridge and side regions on both the left and right.

    This routine generates the vertex coordinates, cell-center coordinates,
    and permittivity distribution for a 2D waveguide cross-section. The
    vertical structure is specified by a sequence of material layers (each with
    its own refractive index and thickness), and the structure includes a
    central waveguide region of half-width rw, plus excess “side” regions
    of width side1 (left) and side2 (right). The top of one layer can be
    etched by a height rh, leaving the central region as the ridge.

    The resulting mesh is suitable for finite-difference mode solvers or
    propagation solvers operating on a rectangular grid.

    Parameters
    
    n : array_like
        Refractive index of each vertical layer. Must have the same length
        as h. The permittivity profile is computed as eps = n**2.

    h : array_like
        Thickness of each vertical layer. The number of vertical
        layers is nlayers = len(h).

    rh : float
        Ridge (etch) height, measured vertically. This determines how many
        grid cells (near the top of the etched layer) participate in the
        ridge/side structure.

    rw : float
        Half-width of the central waveguide region.

    side : float or array_like
        Width of the excess regions to the left and right of the waveguide.
        If side is a scalar, the same width is used on both sides
        (side1 = side2 = side). If side has length 2, the first element is
        the left-hand side width and the second is the right-hand side width.

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
            xc = (x[:-1] + x[1:]) / 2
            yc = y[:-1] + dy/2

    nx, ny : int
        Number of interior cells in the x and y directions.

    eps : ndarray of shape (ny, nx)
        Permittivity distribution on the cell-centered grid, where each
        vertical layer is assigned n[j]**2 and the ridge/side regions are
        drawn over the etched portion of the appropriate layer.

    edges : list of ndarrays, optional
        Returned only if return_edges=True.
        Each entry is an array of shape (M, 2) containing polygon vertices
        outlining layer interfaces or ridge boundaries. Useful for plotting
        mesh overlays.

    Notes
    
    - Vertical layers are mapped to integer grid rows using
      ih = round(h / dy).
    - The ridge height rh and half-width rw are converted to integer cell
      counts, as are the side widths.
    - The permittivity matrix eps uses the convention eps[iy, ix] with
      iy indexing y (rows) and ix indexing x (columns).
    - The edges list contains line segments that trace the material
      boundaries for visualization.

    """

    # Convert inputs to arrays where appropriate
    n = np.asarray(n, dtype=complex)
    h = np.asarray(h)

    nsquared = n**2
    if np.allclose(nsquared.imag, 0.0):
        nsquared = nsquared.real.astype(float)

    # Handle side as either scalar or two-sided specification
    side_arr = np.asarray(side, float)
    if side_arr.size == 1:
        side1 = float(side_arr)
        side2 = float(side_arr)
    elif side_arr.size == 2:
        side1, side2 = map(float, side_arr)
    else:
        raise ValueError("side must be a scalar or an array-like of length 2")

    nlayers = len(h)

    # Integer layer/feature sizes
    ih = np.rint(h / dy).astype(int)
    irh = int(round(rh / dy))          # ridge (etch) height in grid cells
    irw = int(round(2 * rw / dx))      # full ridge width in cells
    iside1 = int(round(side1 / dx))    # left-hand side region (cells)
    iside2 = int(round(side2 / dx))    # right-hand side region (cells)

    # Number of nodes in x and y
    nx_nodes = irw + iside1 + iside2 + 1
    ny_nodes = ih.sum() + 1

    # Vertex coordinates: center the waveguide around x = 0
    x = dx * np.arange(-(irw / 2 + iside1), (irw / 2 + iside2) + 1)
    y = np.arange(ny_nodes) * dy

    # Cell-center coordinates
    xc = 0.5 * (x[:-1] + x[1:])
    yc = (np.arange(1, ny_nodes) * dy) - dy / 2

    nx = xc.size
    ny = yc.size

    # Permittivity grid: eps[iy, ix] with iy along y (rows), ix along x (cols)
    eps = np.zeros((ny, nx), dtype=complex)

    # Fill vertical layers from bottom to top
    iy = 0
    for jj in range(nlayers):
        for _ in range(ih[jj]):
            eps[iy, :] = nsquared[jj]
            iy += 1

    # Apply ridge/side modification over the etched portion of the appropriate layer
    if irh > 0:
        iy = ih.sum() - ih[-1] - 1  # starting row index (0-based)
        for _ in range(irh):
            # Left side region
            if iside1 > 0:
                eps[iy, :iside1] = nsquared[-1]

            # Right side region
            if iside2 > 0:
                eps[iy, irw + iside1: irw + iside1 + iside2] = nsquared[-1]

            iy -= 1

    # Downcast to real, if all elements of eps are real
    if np.iscomplexobj(eps) and np.allclose(eps.imag, 0.0, atol=1e-12):
        eps = eps.real.astype(float)

    if not return_edges:
        return x, y, xc, yc, nx, ny, eps

    # Construct edges for plotting (translated from cell-array representation)
    edges = []
    iyp = np.cumsum(ih)  # cumulative layer thickness in grid cells

    # Horizontal interfaces for layers beneath the etched region
    for jj in range(nlayers - 2):
        if iyp[jj] >= (iyp[nlayers - 2] - irh):
            # Use only the central ridge width
            x_edge = dx * np.array([-irw / 2, irw / 2])
        else:
            # Use full width (side1 + ridge + side2)
            x_edge = dx * np.array([-(irw / 2 + iside1),
                                    (irw / 2 + iside2)])
        y_edge = dy * np.array([iyp[jj], iyp[jj]])
        edges.append(np.column_stack((x_edge, y_edge)))

    # Polygon outlining the etched region at the top of the relevant layer
    jj = nlayers - 2
    x_edge = dx * np.array([
        -(irw / 2 + iside1),
        -irw / 2,
        -irw / 2,
        +irw / 2,
        +irw / 2,
        (irw / 2 + iside2),
    ])
    y_edge = dy * np.array([
        iyp[jj] - irh,
        iyp[jj] - irh,
        iyp[jj],
        iyp[jj],
        iyp[jj] - irh,
        iyp[jj] - irh,
    ])
    edges.append(np.column_stack((x_edge, y_edge)))

    return x, y, xc, yc, nx, ny, eps, edges
