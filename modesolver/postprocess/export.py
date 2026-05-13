import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from PIL import Image


def field_to_bitmap(field, x, y, dpx=None,
                    xmin=None, xmax=None, ymin=None, ymax=None,
                    cmap='hot', vmin=None, vmax=None, vcenter=None,
                    filename=None):
    """
    Render a 2D scalar field to an indexed-color bitmap with square pixels.

    Resamples the field from its (possibly non-uniform or staggered) input grid
    to a uniform square-pixel output grid using bilinear interpolation, then maps
    the result to a 256-entry indexed-color palette derived from a matplotlib
    colormap.

    USAGE:

        img = field_to_bitmap(field, x, y)
        img = field_to_bitmap(field, x, y, dpx=2.5, cmap='jet', vmin=0, vmax=1,
                              filename='Sz.png')
        img = field_to_bitmap(Ex, xc, yc, cmap='seismic', vcenter=0,
                              filename='Ex.png')

    INPUT:

        field : ndarray, shape (ny, nx)
            2D real-valued scalar field to render.  The grid location (cell-
            centered, vertex-centered, or Yee-staggered edge) is arbitrary;
            the caller supplies the matching coordinate arrays x and y.

        x : 1D array, length nx
            x-coordinates of the columns of field.  Must be strictly increasing.
            Use xc (cell centers) for cell-centered fields, or the vertex array x
            for vertex-centered fields.  For Yee-staggered components see NOTES.

        y : 1D array, length ny
            y-coordinates of the rows of field.  Must be strictly increasing.

        dpx : float, optional
            Square pixel size in the same units as x and y.  Defaults to
            min(median(diff(x)), median(diff(y))).

        xmin, xmax, ymin, ymax : float, optional
            Output window in physical coordinates.  Defaults to the full extent
            of x and y respectively.  The window may extend slightly beyond the
            input domain; see NOTES on boundary behavior.

        cmap : str, optional
            Name of any matplotlib colormap (default 'hot').

        vmin, vmax : float, optional
            Data values mapped to palette indices 0 and 255.  Default: the
            minimum and maximum of the interpolated output field.

        vcenter : float, optional
            If given, the center of the colormap (index 127.5) is forced to this
            data value, and the range is made symmetric about it.  Useful for
            signed fields with a diverging colormap (e.g. cmap='seismic',
            vcenter=0).  May be combined with vmin OR vmax (not both) to set the
            half-range explicitly:

                vcenter alone        half = max(|field_max - vcenter|,
                                                |vcenter - field_min|)
                vcenter + vmax only  vmin = 2*vcenter - vmax
                vcenter + vmin only  vmax = 2*vcenter - vmin
                vcenter + vmin + vmax  raises ValueError

        filename : str, optional
            Output file path.  The file format is inferred from the extension by
            Pillow: .png -> PNG, .tif / .tiff -> TIFF.  If None the image is
            returned but not saved.

    OUTPUT:

        img : PIL.Image.Image
            Indexed-color image (mode 'P') with 256 palette entries taken from
            cmap.  The image y-axis is flipped so that y=ymin appears at the
            bottom of the image when displayed.

    NOTES:

        Grid location and coordinate arrays
        ------------------------------------
        Pass whichever 1D coordinate arrays match the actual grid location of
        the supplied field:

            wgmodes() Ex, Ey, Ezj  (cell-centered, ny x nx)  ->  x=xc, y=yc
            wgmodes() Hx, Hy, Hzj  (vertex,  ny+1 x nx+1)   ->  x=x,  y=y
            wgmodes_yee() Ex        (ny+1 x nx)               ->  x=xc, y=y
            wgmodes_yee() Ey        (ny x nx+1)               ->  x=x,  y=yc
            collocate() output      (cell-centered, ny x nx)  ->  x=xc, y=yc
            poynting() Sz           (cell-centered, ny x nx)  ->  x=xc, y=yc

        To export several components with consistent coordinates, collocate
        first:

            Ex, Ey, Ezj, Hx, Hy, Hzj = collocate(Ex, Ey, Ezj, Hx, Hy, Hzj)
            field_to_bitmap(Ex, xc, yc, ...)
            field_to_bitmap(Hx, xc, yc, ...)

        Boundary behavior
        -----------------
        Pixel centers that fall outside [x[0], x[-1]] x [y[0], y[-1]] are
        clamped to the nearest domain edge before interpolating.  This matches
        the implicit behavior of pcolormesh (shading='flat'), which colors the
        half-cell gap between a domain vertex and the nearest cell center with
        the edge cell value.  Linear extrapolation is avoided because fields near
        PML boundaries may have steep gradients that produce unphysical overshoot.

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """
    field = np.asarray(field, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape {field.shape}")
    if x.ndim != 1 or len(x) != field.shape[1]:
        raise ValueError(
            f"x must be 1D with length field.shape[1]={field.shape[1]}, "
            f"got length {len(x)}"
        )
    if y.ndim != 1 or len(y) != field.shape[0]:
        raise ValueError(
            f"y must be 1D with length field.shape[0]={field.shape[0]}, "
            f"got length {len(y)}"
        )
    if vcenter is not None and vmin is not None and vmax is not None:
        raise ValueError(
            "Specify at most two of vcenter, vmin, vmax.  "
            "When vcenter is given, provide vmin or vmax (not both) to set the "
            "half-range, or omit both to auto-scale symmetrically."
        )

    # --- output domain defaults ---
    if xmin is None:
        xmin = x[0]
    if xmax is None:
        xmax = x[-1]
    if ymin is None:
        ymin = y[0]
    if ymax is None:
        ymax = y[-1]

    # --- default pixel size: finest median spacing of input grid ---
    if dpx is None:
        dpx = float(min(np.median(np.diff(x)), np.median(np.diff(y))))

    # --- output pixel-center coordinates ---
    ncols = max(1, int(np.floor((xmax - xmin) / dpx)))
    nrows = max(1, int(np.floor((ymax - ymin) / dpx)))
    x_out = xmin + dpx / 2 + dpx * np.arange(ncols)
    y_out = ymin + dpx / 2 + dpx * np.arange(nrows)

    # --- bilinear interpolation with clamped boundary handling ---
    interp = RegularGridInterpolator(
        (y, x), field, method='linear', bounds_error=False, fill_value=None
    )
    XX, YY = np.meshgrid(x_out, y_out)
    xi = np.clip(XX, x[0], x[-1])
    yi = np.clip(YY, y[0], y[-1])
    pts = np.stack([yi.ravel(), xi.ravel()], axis=-1)
    field_out = interp(pts).reshape(nrows, ncols)

    # --- resolve vmin / vmax / vcenter ---
    if vcenter is None:
        if vmin is None:
            vmin = float(field_out.min())
        if vmax is None:
            vmax = float(field_out.max())
    else:
        vcenter = float(vcenter)
        if vmin is None and vmax is None:
            half = max(abs(float(field_out.max()) - vcenter),
                       abs(vcenter - float(field_out.min())))
            vmin = vcenter - half
            vmax = vcenter + half
        elif vmax is not None:
            vmin = 2.0 * vcenter - vmax
        else:
            vmax = 2.0 * vcenter - vmin

    # guard against zero range
    if vmax == vmin:
        vmax = vmin + 1.0

    # --- map to palette indices 0-255 ---
    indices = np.clip(
        (field_out - vmin) / (vmax - vmin) * 255, 0, 255
    ).astype(np.uint8)

    # flip vertically: image row 0 is top, physics y=ymin is bottom
    indices = indices[::-1, :]

    # --- build palette from matplotlib colormap ---
    try:
        import matplotlib
        colormap = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap, 256)

    rgba = (colormap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
    palette = rgba[:, :3].flatten().tolist()

    # --- create indexed-color image ---
    img = Image.fromarray(indices, mode='P')
    img.putpalette(palette)

    if filename is not None:
        img.save(filename)

    return img


def field_to_contours(field, x, y, levels,
                      xmin=None, xmax=None, ymin=None, ymax=None,
                      colors='black', linewidths=0.5, linestyles='solid',
                      filename=None):
    """
    Render contour lines of a 2D scalar field to a transparent SVG.

    Produces a vector SVG suitable for overlaying on a bitmap generated by
    field_to_bitmap().  Specifying the same xmin/xmax/ymin/ymax in both calls
    ensures the SVG covers the same physical region as the PNG; the two files
    can then be scaled freely and overlaid in any vector editor (Adobe
    Illustrator, Inkscape, etc.) without loss of alignment.

    USAGE:

        fig = field_to_contours(field, xc, yc, levels)
        fig = field_to_contours(Ex_dB, xc, yc, np.arange(-40, 0, 5),
                                colors='white', linewidths=0.3,
                                filename='Ex_contours.svg')

    INPUT:

        field : ndarray, shape (ny, nx)
            2D scalar field whose contours are to be drawn.  The same grid-
            location rules apply as for field_to_bitmap: pass the 1D coordinate
            arrays that match the actual grid location of the field.  The caller
            is responsible for any pre-processing such as converting to dB or
            normalizing.

        x : 1D array, length nx
            x-coordinates of the columns of field (same convention as
            field_to_bitmap).

        y : 1D array, length ny
            y-coordinates of the rows of field.

        levels : array-like or int
            Contour levels passed directly to matplotlib ax.contour().  May be
            a sorted 1D array of specific values, or an integer requesting that
            many automatically chosen levels.

        xmin, xmax, ymin, ymax : float, optional
            Physical coordinate window for the SVG.  Should match the values
            used in the corresponding field_to_bitmap() call.  Defaults to the
            full extent of x and y.

        colors : color or list of colors, optional
            Contour line color(s), passed to ax.contour().  Default 'black'.

        linewidths : float or list of floats, optional
            Contour line width(s) in points, passed to ax.contour().
            Default 0.5.

        linestyles : str or list of str, optional
            Contour line style(s), passed to ax.contour().  Default 'solid'
            (overrides matplotlib's default of dashing negative-valued contours).

        filename : str, optional
            Output file path; should end in .svg.  If None the figure is
            returned but not saved.

    OUTPUT:

        fig : matplotlib.figure.Figure
            Figure containing the contour plot.  Call display(fig) for inline
            preview in a Jupyter notebook, fig.savefig(...) to save in another
            format, or plt.close(fig) to release memory when done.

    NOTES:

        Alignment with field_to_bitmap
        --------------------------------
        The SVG viewport aspect ratio is set from (xmax-xmin)/(ymax-ymin), so
        the SVG is geometrically correct at any display scale.  To overlay the
        SVG on a PNG in a vector editor, import both files and scale the SVG
        to match the PNG dimensions; the contours will align precisely as long
        as both were generated with the same xmin/xmax/ymin/ymax.

        No resampling
        -------------
        Unlike field_to_bitmap, no interpolation to a uniform grid is performed.
        matplotlib's contour() accepts 1D non-uniform coordinate arrays directly
        and computes contours on the original (possibly stretched) grid.

        Adobe Illustrator / Inkscape compatibility
        ------------------------------------------
        The output SVG contains only vector paths with a transparent background
        and no tick marks, labels, or embedded rasters.  It opens cleanly in
        Illustrator and Inkscape for appearance editing.

    AUTHOR:

        Thomas E. Murphy (tem@umd.edu)
    """
    field = np.asarray(field, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if field.ndim != 2:
        raise ValueError(f"field must be 2D, got shape {field.shape}")
    if x.ndim != 1 or len(x) != field.shape[1]:
        raise ValueError(
            f"x must be 1D with length field.shape[1]={field.shape[1]}, "
            f"got length {len(x)}"
        )
    if y.ndim != 1 or len(y) != field.shape[0]:
        raise ValueError(
            f"y must be 1D with length field.shape[0]={field.shape[0]}, "
            f"got length {len(y)}"
        )

    # --- output window defaults ---
    if xmin is None:
        xmin = float(x[0])
    if xmax is None:
        xmax = float(x[-1])
    if ymin is None:
        ymin = float(y[0])
    if ymax is None:
        ymax = float(y[-1])

    # --- figure sized to the physical aspect ratio of the window ---
    width_in  = 10.0
    height_in = width_in * (ymax - ymin) / (xmax - xmin)
    fig = plt.figure(figsize=(width_in, height_in))
    ax  = fig.add_axes([0, 0, 1, 1])   # axes fill figure, no margins

    # --- contours on the native (possibly non-uniform) grid ---
    ax.contour(x, y, field, levels,
               colors=colors, linewidths=linewidths, linestyles=linestyles)

    # --- set window and suppress all decorations ---
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axis('off')

    if filename is not None:
        fig.savefig(filename, transparent=True)

    return fig
