import numpy as np

def unfold(x, y, xc, yc, *fields, boundary, unfold='',
           eps=None, epsxx=None, epsyy=None, epszz=None, epsxy=None, epsyx=None):
    """
    Unfold eigenmode solutions or permittivity arrays computed with symmetry
    boundary conditions.

    When eigenmodes are computed using symmetric or antisymmetric boundary
    conditions (S, A, E, or M), the computational domain represents only a
    portion of the full waveguide structure. This function reconstructs the
    complete mode by mirroring the solution across specified boundaries,
    applying appropriate symmetries to each field component.

    This function can unfold either field arrays OR permittivity arrays, but
    not both in the same call.

    USAGE:

        # Unfold wgmodes field result across west PMC boundary:
        neff, hx, hy, hzj = mode.wgmodes(wavelength, n2, nmodes, dx, dy,
                                          eps=eps, boundary='000M')
        x, y, xc, yc, hx, hy, hzj = mode.unfold(x, y, xc, yc, hx, hy, hzj,
                                                 boundary='000M', unfold='W')

        # Unfold permittivity (isotropic):
        x, y, xc, yc, eps = mode.unfold(x, y, xc, yc,
                                         boundary='000M', unfold='W', eps=eps)

        # Unfold permittivity tensor (diagonal anisotropic):
        x, y, xc, yc, epsxx, epsyy, epszz = mode.unfold(
            x, y, xc, yc, boundary='000M', unfold='W',
            epsxx=epsxx, epsyy=epsyy, epszz=epszz
        )

        # Unfold permittivity tensor (full anisotropic):
        x, y, xc, yc, epsxx, epsxy, epsyx, epsyy, epszz = mode.unfold(
            x, y, xc, yc, boundary='000M', unfold='W',
            epsxx=epsxx, epsxy=epsxy, epsyx=epsyx,
            epsyy=epsyy, epszz=epszz
        )

    PARAMETERS:

        x, y : ndarray (1D)
            Vertex coordinate arrays from the original mesh.

        xc, yc : ndarray (1D)
            Cell-center coordinate arrays (these are recomputed after unfolding).

        *fields : ndarray
            Field arrays from any modesolver. The number of field arguments
            determines how they are interpreted:

                1 field  : Scalar field (Ex or Ey)
                3 fields : Magnetic fields (Hx, Hy, Hzj)
                6 fields : All fields (Ex, Ey, Ezj, Hx, Hy, Hzj)

            The grid location (cell-centered or node-centered) is detected
            independently for each field component from its array shape. This
            makes the function agnostic to the discretization scheme used.

            For multi-mode solutions, fields should have shape (ny, nx, nmodes)
            or (ny+1, nx+1, nmodes) depending on grid location.

            Cannot be used together with eps keyword arguments.

        boundary : str (keyword-only, required)
            4-character boundary condition string, same as used when computing
            the modes. Format: 'NSEW' where:
                boundary[0] = North boundary condition
                boundary[1] = South boundary condition
                boundary[2] = East boundary condition
                boundary[3] = West boundary condition

            Each character may be:
                'S' : symmetric (allowed only for scalar fields)
                'A' : antisymmetric (allowed only for scalar fields)
                'E' : PEC (perfect electric conductor, for vector fields)
                'M' : PMC (perfect magnetic conductor, for vector fields)
                '0' : Dirichlet (field = 0 outside boundary)

            For scalar fields (1 field argument), only 'S', 'A', and '0' are
            allowed, as the symmetry applies directly to the single component.

            For vector fields (3 or 6 field arguments), only 'E', 'M', and '0'
            are allowed. The 'S' and 'A' conditions are deprecated for vector
            fields because it's ambiguous which component they refer to.

        unfold : str (keyword-only, optional)
            String specifying which boundaries to unfold. May contain any
            combination of the letters 'N', 'S', 'E', 'W' (case insensitive).

            Examples:
                unfold='W'      - unfold across west boundary only
                unfold='WS'     - unfold across west and south
                unfold='NSEW'   - unfold across all four boundaries
                unfold=''       - no unfolding (returns input unchanged)

            Default: '' (no unfolding)

        eps, epsxx, epsyy, epszz, epsxy, epsyx : ndarray (keyword-only, optional)
            Permittivity arrays to unfold. These are always unfolded with
            symmetric (even) parity. The grid location (cell-centered,
            node-centered, or edge-centered) is automatically detected from
            the array shape.

            Supply either:
                - eps (isotropic)
                - epsxx, epsyy, epszz (diagonal anisotropic)
                - epsxx, epsxy, epsyx, epsyy, epszz (full anisotropic)

            Cannot be used together with positional field arguments.

    RETURNS:

        x, y : ndarray
            Unfolded vertex coordinates.

        xc, yc : ndarray
            Unfolded cell-center coordinates.

        *fields or *eps : ndarray
            Unfolded field arrays or permittivity arrays, returned in the
            same order as input.

    FIELD SYMMETRIES:

        For field arrays, the function applies the following symmetries:

        West/East boundaries (vertical, unfold across x):
            'M' or 'S' : Hx even, Hy odd, Hz odd; Ex odd, Ey even, Ez even
            'E' or 'A' : Hx odd, Hy even, Hz even; Ex even, Ey odd, Ez odd

        North/South boundaries (horizontal, unfold across y):
            'M' : Hx odd, Hy even, Hz odd; Ex even, Ey odd, Ez even
            'E' : Hx even, Hy odd, Hz odd; Ex odd, Ey even, Ez odd
            'S' : Hx even, Hy odd
            'A' : Hx odd, Hy even

        For permittivity arrays, all components are always symmetric (even).

    GRID LOCATION DETECTION:

        Both field and permittivity arrays are unfolded according to their
        grid location, which is detected independently for each component
        from its array shape. The grid location is determined separately
        for each dimension (y and x):

            ny dimension:  ny → cell-centered,  ny+1 → node-centered
            nx dimension:  nx → cell-centered,  nx+1 → node-centered

        This allows the function to work with any discretization scheme,
        including:
            - Cell-centered fields (ny, nx)
            - Node-centered fields (ny+1, nx+1)
            - Yee grid fields with mixed locations
            - Future discretization schemes

    NOTES:

        1) Boundaries are unfolded in the order: west, east, south, north.
           You can unfold across multiple boundaries in a single call by
           including multiple letters in the unfold string. However, you
           cannot unfold across opposing boundaries (both N+S or both E+W)
           in the same call, as this is not physically meaningful.

        2) The number of field arguments determines their interpretation:
           - 1 field: Scalar (Ex or Ey), only S/A/0 boundaries allowed
           - 3 fields: H-fields (Hx, Hy, Hzj), only E/M/0 boundaries allowed
           - 6 fields: All fields (Ex, Ey, Ezj, Hx, Hy, Hzj), only E/M/0 allowed

        3) Grid locations are detected per-component from array shapes. Each
           field can be cell-centered or node-centered independently in each
           dimension. This makes the function solver-agnostic.

        4) For multi-mode solutions, all modes are unfolded with the same
           symmetry rules. The mode dimension should be last, i.e., field
           arrays should have shape (ny, nx, nmodes) or (ny+1, nx+1, nmodes).

        5) The returned coordinates (xc, yc) are recomputed from (x, y).

        6) It is safe (though redundant) to unfold a boundary with '0'
           boundary condition. The function will handle this gracefully.

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Validate boundary string
    if not isinstance(boundary, str) or len(boundary) != 4:
        raise ValueError("boundary must be a 4-character string (NSEW order)")

    boundary = boundary.upper()
    unfold = unfold.upper()

    # Validate unfold string
    if not all(c in 'NSEW' for c in unfold):
        raise ValueError("unfold string may only contain letters N, S, E, W")

    # Check for opposing boundaries (not physically meaningful)
    if ('N' in unfold and 'S' in unfold) or ('E' in unfold and 'W' in unfold):
        raise ValueError(
            "Cannot unfold across opposing boundaries. "
            "Specify only one boundary per direction (not both N+S or both E+W)."
        )

    # Determine mode: fields or eps
    has_fields = len(fields) > 0
    has_eps = any(v is not None for v in [eps, epsxx, epsyy, epszz, epsxy, epsyx])

    if has_fields and has_eps:
        raise ValueError("Cannot unfold both fields and eps in the same call. "
                         "Use separate calls for fields and permittivity.")

    if not has_fields and not has_eps:
        raise ValueError("No fields or permittivity arrays provided to unfold")

    # UNFOLD FIELDS
    if has_fields:
        return _unfold_fields(x, y, xc, yc, fields, boundary, unfold)

    # UNFOLD EPS
    else:
        return _unfold_eps(x, y, xc, yc, boundary, unfold,
                          eps, epsxx, epsyy, epszz, epsxy, epsyx)


def _detect_grid_location(field, ny, nx):
    """
    Detect grid location of a field component in each dimension independently.

    Parameters
    ----------
    field : ndarray
        Field array with shape (Ny, Nx) or (Ny, Nx, nmodes)
    ny, nx : int
        Number of cells in y and x dimensions (ny_nodes - 1, nx_nodes - 1)

    Returns
    -------
    y_location : str
        'cell' if Ny == ny, 'node' if Ny == ny+1
    x_location : str
        'cell' if Nx == nx, 'node' if Nx == nx+1
    """
    field = np.asarray(field)
    Ny, Nx = field.shape[0], field.shape[1]

    if Ny == ny:
        y_location = 'cell'
    elif Ny == ny + 1:
        y_location = 'node'
    else:
        raise ValueError(
            f"Field y-dimension {Ny} doesn't match ny={ny} (cell) or ny+1={ny+1} (node)"
        )

    if Nx == nx:
        x_location = 'cell'
    elif Nx == nx + 1:
        x_location = 'node'
    else:
        raise ValueError(
            f"Field x-dimension {Nx} doesn't match nx={nx} (cell) or nx+1={nx+1} (node)"
        )

    return y_location, x_location


def _unfold_fields(x, y, xc, yc, fields, boundary, unfold):
    """Unfold field arrays with proper E/H field symmetries."""

    # Convert fields to list for easier manipulation
    fields = list(fields)

    nx = len(x) - 1  # Number of cells
    ny = len(y) - 1

    # Interpret fields based on count
    num_fields = len(fields)
    if num_fields == 1:
        field_names = ['scalar']
        # Validate boundary conditions for scalar field (only S, A, 0 allowed)
        valid_bcs = {'S', 'A', '0'}
        for i, bc in enumerate(boundary):
            if bc not in valid_bcs:
                bc_labels = ['North', 'South', 'East', 'West']
                raise ValueError(
                    f"For single field, only 'S', 'A', and '0' boundary conditions "
                    f"are allowed. Got '{bc}' for {bc_labels[i]} boundary."
                )
    elif num_fields == 3:
        field_names = ['hx', 'hy', 'hzj']
        # Validate boundary conditions for vector fields (only E, M, 0 allowed)
        valid_bcs = {'E', 'M', '0'}
        for i, bc in enumerate(boundary):
            if bc not in valid_bcs:
                bc_labels = ['North', 'South', 'East', 'West']
                raise ValueError(
                    f"For multiple fields, only 'E', 'M', and '0' boundary conditions "
                    f"are allowed (S and A are deprecated for vector fields). "
                    f"Got '{bc}' for {bc_labels[i]} boundary."
                )
    elif num_fields == 6:
        field_names = ['ex', 'ey', 'ezj', 'hx', 'hy', 'hzj']
        # Validate boundary conditions for vector fields (only E, M, 0 allowed)
        valid_bcs = {'E', 'M', '0'}
        for i, bc in enumerate(boundary):
            if bc not in valid_bcs:
                bc_labels = ['North', 'South', 'East', 'West']
                raise ValueError(
                    f"For multiple fields, only 'E', 'M', and '0' boundary conditions "
                    f"are allowed (S and A are deprecated for vector fields). "
                    f"Got '{bc}' for {bc_labels[i]} boundary."
                )
    else:
        raise ValueError(
            f"Expected 1, 3, or 6 field arguments, got {num_fields}. "
            f"Use 1 for scalar field (Ex or Ey), 3 for H-fields (Hx, Hy, Hzj), "
            f"or 6 for all fields (Ex, Ey, Ezj, Hx, Hy, Hzj)."
        )

    # Detect grid location for each field component
    grid_locations = []
    for field in fields:
        y_loc, x_loc = _detect_grid_location(field, ny, nx)
        grid_locations.append((y_loc, x_loc))

    # Unfold each boundary in sequence (order: W, E, S, N)
    if 'W' in unfold:
        x, fields = _unfold_west(x, fields, field_names, grid_locations, boundary[3])
        nx = len(x) - 1
        # Update grid locations after unfolding in x
        grid_locations = [(y_loc, x_loc) for y_loc, x_loc in grid_locations]

    if 'E' in unfold:
        x, fields = _unfold_east(x, fields, field_names, grid_locations, boundary[2])
        nx = len(x) - 1
        grid_locations = [(y_loc, x_loc) for y_loc, x_loc in grid_locations]

    if 'S' in unfold:
        y, fields = _unfold_south(y, fields, field_names, grid_locations, boundary[1])
        ny = len(y) - 1
        grid_locations = [(y_loc, x_loc) for y_loc, x_loc in grid_locations]

    if 'N' in unfold:
        y, fields = _unfold_north(y, fields, field_names, grid_locations, boundary[0])
        ny = len(y) - 1
        grid_locations = [(y_loc, x_loc) for y_loc, x_loc in grid_locations]

    # Recompute cell centers
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])

    return x, y, xc, yc, *fields


def _unfold_eps(x, y, xc, yc, boundary, unfold,
                eps, epsxx, epsyy, epszz, epsxy, epsyx):
    """Unfold permittivity arrays with symmetric (even) parity."""

    # Determine which eps components are provided
    if eps is not None:
        # Isotropic case
        eps_components = {'eps': eps}
    elif all(v is not None for v in [epsxx, epsxy, epsyx, epsyy, epszz]):
        # Full tensor
        eps_components = {
            'epsxx': epsxx, 'epsxy': epsxy, 'epsyx': epsyx,
            'epsyy': epsyy, 'epszz': epszz
        }
    elif all(v is not None for v in [epsxx, epsyy, epszz]):
        # Diagonal anisotropic
        eps_components = {'epsxx': epsxx, 'epsyy': epsyy, 'epszz': epszz}
    else:
        raise ValueError("Must supply either eps, or epsxx/yy/zz, or full tensor")

    # Convert to list for processing
    component_names = list(eps_components.keys())
    component_arrays = [np.asarray(eps_components[name]) for name in component_names]

    nx = len(x) - 1  # Number of cells
    ny = len(y) - 1

    # Detect grid location for each eps component
    grid_locations = []
    for eps_array in component_arrays:
        y_loc, x_loc = _detect_grid_location(eps_array, ny, nx)
        grid_locations.append((y_loc, x_loc))

    # Unfold each boundary in sequence (eps is always symmetric/even)
    if 'W' in unfold:
        if boundary[3] != '0':  # Skip Dirichlet boundary
            x, component_arrays = _unfold_eps_boundary(
                x, component_arrays, grid_locations, 'west'
            )
            nx = len(x) - 1

    if 'E' in unfold:
        if boundary[2] != '0':
            x, component_arrays = _unfold_eps_boundary(
                x, component_arrays, grid_locations, 'east'
            )
            nx = len(x) - 1

    if 'S' in unfold:
        if boundary[1] != '0':
            y, component_arrays = _unfold_eps_boundary(
                y, component_arrays, grid_locations, 'south'
            )
            ny = len(y) - 1

    if 'N' in unfold:
        if boundary[0] != '0':
            y, component_arrays = _unfold_eps_boundary(
                y, component_arrays, grid_locations, 'north'
            )
            ny = len(y) - 1

    # Recompute cell centers
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])

    return x, y, xc, yc, *component_arrays


def _unfold_eps_boundary(coord, eps_arrays, grid_locations, direction):
    """
    Unfold eps arrays across a single boundary.

    All eps components are symmetric (even), but grid location varies.
    """

    # Mirror coordinate
    if direction == 'west':
        coord_mirror = 2 * coord[0] - coord[1:][::-1]
        coord_new = np.concatenate([coord_mirror, coord])
    elif direction == 'east':
        coord_mirror = 2 * coord[-1] - coord[:-1][::-1]
        coord_new = np.concatenate([coord, coord_mirror])
    elif direction == 'south':
        coord_mirror = 2 * coord[0] - coord[1:][::-1]
        coord_new = np.concatenate([coord_mirror, coord])
    else:  # north
        coord_mirror = 2 * coord[-1] - coord[:-1][::-1]
        coord_new = np.concatenate([coord, coord_mirror])

    # Unfold each eps component
    eps_new = []
    for eps_array, (y_loc, x_loc) in zip(eps_arrays, grid_locations):
        eps_array = np.asarray(eps_array)

        # Apply symmetric mirroring based on direction and grid location
        # eps is always even parity (is_odd=False)
        if direction in ['west', 'east']:
            # Unfolding in x-direction
            if x_loc == 'cell':
                if direction == 'west':
                    eps_unfolded = _mirror_x_cell(eps_array, is_odd=False)
                else:
                    eps_unfolded = _mirror_x_cell_east(eps_array, is_odd=False)
            else:  # 'node'
                if direction == 'west':
                    eps_unfolded = _mirror_x_node(eps_array, is_odd=False)
                else:
                    eps_unfolded = _mirror_x_node_east(eps_array, is_odd=False)
        else:
            # Unfolding in y-direction
            if y_loc == 'cell':
                if direction == 'south':
                    eps_unfolded = _mirror_y_cell(eps_array, is_odd=False)
                else:
                    eps_unfolded = _mirror_y_cell_north(eps_array, is_odd=False)
            else:  # 'node'
                if direction == 'south':
                    eps_unfolded = _mirror_y_node(eps_array, is_odd=False)
                else:
                    eps_unfolded = _mirror_y_node_north(eps_array, is_odd=False)

        eps_new.append(eps_unfolded)

    return coord_new, eps_new


def _unfold_west(x, fields, field_names, grid_locations, boundary_symmetry):
    """Unfold across west boundary (vertical line at x[0])."""

    # Skip if Dirichlet boundary (no symmetry to unfold)
    if boundary_symmetry == '0':
        return x, fields

    # Create mirrored x coordinates
    x_mirror = 2 * x[0] - x[1:][::-1]
    x_new = np.concatenate([x_mirror, x])

    # Unfold each field
    fields_new = []
    for field, field_name, (y_loc, x_loc) in zip(fields, field_names, grid_locations):
        field = np.asarray(field)

        # Determine symmetry for this field component
        if len(field_names) == 1:
            # Scalar field - use boundary condition directly
            if boundary_symmetry == 'S':
                is_odd = False  # even
            elif boundary_symmetry == 'A':
                is_odd = True   # odd
            else:  # '0' already checked above
                is_odd = False
        else:
            # Vector field - infer from field name and boundary type
            is_odd = _get_x_symmetry(field_name, boundary_symmetry)

        # Apply mirroring based on x grid location
        if x_loc == 'cell':
            field_new = _mirror_x_cell(field, is_odd)
        else:  # 'node'
            field_new = _mirror_x_node(field, is_odd)

        fields_new.append(field_new)

    return x_new, fields_new


def _unfold_east(x, fields, field_names, grid_locations, boundary_symmetry):
    """Unfold across east boundary (vertical line at x[-1])."""

    # Skip if Dirichlet boundary
    if boundary_symmetry == '0':
        return x, fields

    # Create mirrored x coordinates
    x_mirror = 2 * x[-1] - x[:-1][::-1]
    x_new = np.concatenate([x, x_mirror])

    # Unfold each field
    fields_new = []
    for field, field_name, (y_loc, x_loc) in zip(fields, field_names, grid_locations):
        field = np.asarray(field)

        # Determine symmetry for this field component
        if len(field_names) == 1:
            # Scalar field - use boundary condition directly
            if boundary_symmetry == 'S':
                is_odd = False  # even
            elif boundary_symmetry == 'A':
                is_odd = True   # odd
            else:
                is_odd = False
        else:
            # Vector field - infer from field name and boundary type
            is_odd = _get_x_symmetry(field_name, boundary_symmetry)

        # Apply mirroring based on x grid location
        if x_loc == 'cell':
            field_new = _mirror_x_cell_east(field, is_odd)
        else:  # 'node'
            field_new = _mirror_x_node_east(field, is_odd)

        fields_new.append(field_new)

    return x_new, fields_new


def _unfold_south(y, fields, field_names, grid_locations, boundary_symmetry):
    """Unfold across south boundary (horizontal line at y[0])."""

    # Skip if Dirichlet boundary
    if boundary_symmetry == '0':
        return y, fields

    # Create mirrored y coordinates
    y_mirror = 2 * y[0] - y[1:][::-1]
    y_new = np.concatenate([y_mirror, y])

    # Unfold each field
    fields_new = []
    for field, field_name, (y_loc, x_loc) in zip(fields, field_names, grid_locations):
        field = np.asarray(field)

        # Determine symmetry for this field component
        if len(field_names) == 1:
            # Scalar field - use boundary condition directly
            if boundary_symmetry == 'S':
                is_odd = False  # even
            elif boundary_symmetry == 'A':
                is_odd = True   # odd
            else:
                is_odd = False
        else:
            # Vector field - infer from field name and boundary type
            is_odd = _get_y_symmetry(field_name, boundary_symmetry)

        # Apply mirroring based on y grid location
        if y_loc == 'cell':
            field_new = _mirror_y_cell(field, is_odd)
        else:  # 'node'
            field_new = _mirror_y_node(field, is_odd)

        fields_new.append(field_new)

    return y_new, fields_new


def _unfold_north(y, fields, field_names, grid_locations, boundary_symmetry):
    """Unfold across north boundary (horizontal line at y[-1])."""

    # Skip if Dirichlet boundary
    if boundary_symmetry == '0':
        return y, fields

    # Create mirrored y coordinates
    y_mirror = 2 * y[-1] - y[:-1][::-1]
    y_new = np.concatenate([y, y_mirror])

    # Unfold each field
    fields_new = []
    for field, field_name, (y_loc, x_loc) in zip(fields, field_names, grid_locations):
        field = np.asarray(field)

        # Determine symmetry for this field component
        if len(field_names) == 1:
            # Scalar field - use boundary condition directly
            if boundary_symmetry == 'S':
                is_odd = False  # even
            elif boundary_symmetry == 'A':
                is_odd = True   # odd
            else:
                is_odd = False
        else:
            # Vector field - infer from field name and boundary type
            is_odd = _get_y_symmetry(field_name, boundary_symmetry)

        # Apply mirroring based on y grid location
        if y_loc == 'cell':
            field_new = _mirror_y_cell_north(field, is_odd)
        else:  # 'node'
            field_new = _mirror_y_node_north(field, is_odd)

        fields_new.append(field_new)

    return y_new, fields_new


def _get_x_symmetry(field_type, boundary_symmetry):
    """
    Determine if field is odd across x-boundary.

    West/East boundaries:
        'M' or 'S': Hx even, Hy odd, Hz odd; Ex odd, Ey even, Ez even
        'E' or 'A': Hx odd, Hy even, Hz even; Ex even, Ey odd, Ez odd
    """
    field_type = field_type.lower()

    if boundary_symmetry in ['M', 'S']:
        # PMC or symmetric H
        return field_type in ['hy', 'hzj', 'ex']
    else:  # 'E' or 'A'
        # PEC or antisymmetric H
        return field_type in ['hx', 'ey', 'ezj']


def _get_y_symmetry(field_type, boundary_symmetry):
    """
    Determine if field is odd across y-boundary.

    North/South boundaries:
        'M': Hx odd, Hy even, Hz odd; Ex even, Ey odd, Ez even
        'E': Hx even, Hy odd, Hz odd; Ex odd, Ey even, Ez odd
        'S': Hx even, Hy odd
        'A': Hx odd, Hy even
    """
    field_type = field_type.lower()

    if boundary_symmetry == 'M':
        return field_type in ['hx', 'hzj', 'ey']
    elif boundary_symmetry == 'E':
        return field_type in ['hy', 'hzj', 'ex']
    elif boundary_symmetry == 'S':
        return field_type in ['hy']
    else:  # 'A'
        return field_type in ['hx']


# Mirroring functions for different grid types and directions

def _mirror_x_cell(field, is_odd):
    """Mirror cell-centered field across west boundary."""
    mirror = field[:, ::-1]
    if is_odd:
        mirror = -mirror
    return np.concatenate([mirror, field], axis=1)


def _mirror_x_cell_east(field, is_odd):
    """Mirror cell-centered field across east boundary."""
    mirror = field[:, ::-1]
    if is_odd:
        mirror = -mirror
    return np.concatenate([field, mirror], axis=1)


def _mirror_x_node(field, is_odd):
    """Mirror node-centered field across west boundary."""
    mirror = field[:, 1:][:, ::-1]  # Exclude boundary column
    if is_odd:
        mirror = -mirror
    return np.concatenate([mirror, field], axis=1)


def _mirror_x_node_east(field, is_odd):
    """Mirror node-centered field across east boundary."""
    mirror = field[:, :-1][:, ::-1]  # Exclude boundary column
    if is_odd:
        mirror = -mirror
    return np.concatenate([field, mirror], axis=1)


def _mirror_y_cell(field, is_odd):
    """Mirror cell-centered field across south boundary."""
    mirror = field[::-1, :]
    if is_odd:
        mirror = -mirror
    return np.concatenate([mirror, field], axis=0)


def _mirror_y_cell_north(field, is_odd):
    """Mirror cell-centered field across north boundary."""
    mirror = field[::-1, :]
    if is_odd:
        mirror = -mirror
    return np.concatenate([field, mirror], axis=0)


def _mirror_y_node(field, is_odd):
    """Mirror node-centered field across south boundary."""
    mirror = field[1:, :][::-1, :]  # Exclude boundary row
    if is_odd:
        mirror = -mirror
    return np.concatenate([mirror, field], axis=0)


def _mirror_y_node_north(field, is_odd):
    """Mirror node-centered field across north boundary."""
    mirror = field[:-1, :][::-1, :]  # Exclude boundary row
    if is_odd:
        mirror = -mirror
    return np.concatenate([field, mirror], axis=0)
