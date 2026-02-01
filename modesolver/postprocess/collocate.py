import numpy as np


def collocate(Ex, Ey, Ezj, Hx, Hy, Hzj):
    """
    Collocate field components to cell centers via linear interpolation.

    This function takes six electromagnetic field components that may be defined
    on different grid locations and interpolates them all to a common cell-centered
    grid of size (ny, nx), where ny and nx are the minimum row and column dimensions
    across all input fields.

    USAGE:

        Ex, Ey, Ezj, Hx, Hy, Hzj = collocate(Ex, Ey, Ezj, Hx, Hy, Hzj)

    INPUT:

        Ex, Ey, Ezj, Hx, Hy, Hzj : ndarray
            Field components with shapes that differ by at most 1 in each dimension.
            May be 2D (single mode) or 3D (multiple modes with shape [..., nmodes]).

    OUTPUT:

        Ex, Ey, Ezj, Hx, Hy, Hzj : ndarray
            All field components interpolated to cell centers with shape
            (ny, nx) or (ny, nx, nmodes), where ny and nx are the minimum
            dimensions from the input fields.

    INTERPOLATION RULES:

        For each field component, based on its shape relative to (ny, nx):
            (ny, nx)     : Already at cell centers, returned unchanged
            (ny+1, nx+1) : Average of 4 corner values
            (ny+1, nx)   : Average of top and bottom values
            (ny, nx+1)   : Average of left and right values

    ERRORS:

        Raises ValueError if:
            - All six fields already have identical shapes (nothing to collocate)
            - Row dimensions vary by more than 1 across fields
            - Column dimensions vary by more than 1 across fields

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """

    Ex = np.asarray(Ex)
    Ey = np.asarray(Ey)
    Ezj = np.asarray(Ezj)
    Hx = np.asarray(Hx)
    Hy = np.asarray(Hy)
    Hzj = np.asarray(Hzj)

    fields = [Ex, Ey, Ezj, Hx, Hy, Hzj]
    names = ['Ex', 'Ey', 'Ezj', 'Hx', 'Hy', 'Hzj']

    # Get spatial shapes (first two dimensions)
    shapes = [f.shape[:2] for f in fields]

    # Check if already collocated (all same shape)
    if len(set(shapes)) == 1:
        raise ValueError(
            "All six fields already have identical shapes. Nothing to collocate."
        )

    # Find min and max rows and columns
    rows = [s[0] for s in shapes]
    cols = [s[1] for s in shapes]

    ny = min(rows)
    nx = min(cols)
    max_rows = max(rows)
    max_cols = max(cols)

    # Validate row dimensions
    if max_rows - ny > 1:
        raise ValueError(
            f"Row dimensions vary by more than 1: min={ny}, max={max_rows}. "
            f"Shapes: {dict(zip(names, shapes))}"
        )

    # Validate column dimensions
    if max_cols - nx > 1:
        raise ValueError(
            f"Column dimensions vary by more than 1: min={nx}, max={max_cols}. "
            f"Shapes: {dict(zip(names, shapes))}"
        )

    # Interpolate each field to (ny, nx)
    result = []
    for field, name in zip(fields, names):
        field_rows, field_cols = field.shape[:2]
        is_3d = field.ndim == 3

        row_extra = field_rows - ny  # 0 or 1
        col_extra = field_cols - nx  # 0 or 1

        if row_extra == 0 and col_extra == 0:
            # Already at cell centers
            result.append(field)
        elif row_extra == 1 and col_extra == 1:
            # At vertices: average 4 corners
            if is_3d:
                field = 0.25 * (field[:-1, :-1, :] + field[1:, :-1, :] +
                                field[:-1, 1:, :] + field[1:, 1:, :])
            else:
                field = 0.25 * (field[:-1, :-1] + field[1:, :-1] +
                                field[:-1, 1:] + field[1:, 1:])
            result.append(field)
        elif row_extra == 1 and col_extra == 0:
            # Extra row: average top and bottom
            if is_3d:
                field = 0.5 * (field[:-1, :, :] + field[1:, :, :])
            else:
                field = 0.5 * (field[:-1, :] + field[1:, :])
            result.append(field)
        elif row_extra == 0 and col_extra == 1:
            # Extra column: average left and right
            if is_3d:
                field = 0.5 * (field[:, :-1, :] + field[:, 1:, :])
            else:
                field = 0.5 * (field[:, :-1] + field[:, 1:])
            result.append(field)

    return tuple(result)
