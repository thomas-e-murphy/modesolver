"""
layoutmesh.py — build a finite-difference permittivity grid from a GeometryModel.

Public API
----------
    layoutmesh(model, label_n, ...)
"""

from __future__ import annotations

import numpy as np
from .layout import GeometryModel


def layoutmesh(
    model: GeometryModel,
    label_n: dict,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    dx: float = None,
    dy: float = None,
    *,
    x: "np.ndarray | None" = None,
    y: "np.ndarray | None" = None,
    method: str = "center",
    yee: bool = False,
):
    """Build a finite-difference permittivity mesh from a labeled GeometryModel.

    Parameters
    ----------
    model : GeometryModel
        The labeled 2D geometry (background LayerStack plus finite Region objects).
    label_n : dict
        Maps each material label to a refractive index.  Each value may be:

        - A scalar *complex* ``n`` — isotropic: ``n_x = n_y = n_z = n``.
        - A 3-tuple ``(n_x, n_y, n_z)`` — diagonal anisotropy with axes aligned
          to the physical x, y, z coordinate directions.

    xmin, xmax, ymin, ymax : float, optional
        Computational domain boundaries.  Derived from *x* / *y* when those
        arrays are supplied.
    dx, dy : float, optional
        Uniform horizontal and vertical mesh spacings.  Required when *x* / *y*
        are not supplied.
    x, y : array_like, optional
        Pre-built vertex coordinate arrays (e.g. from :func:`stretchmesh`).
        When provided, the uniform grid generation step is skipped and
        *xmin*, *xmax*, *dx* (respectively *ymin*, *ymax*, *dy*) may be
        omitted — they are inferred from ``x[0]``, ``x[-1]`` etc.
        Pass the stretched vertex arrays from ``stretchmesh`` here so that
        sub-cell permittivity averages are computed on the correct
        (non-uniform) cell boundaries.
    method : ``"center"`` or ``"fill"``
        Cell assignment strategy.

        - ``"center"`` (default): query ``label_at_point`` at each cell centre.
          Fast; equivalent in spirit to the legacy ``waveguidemesh`` routine.
        - ``"fill"``: use ``fill_fractions`` for area-weighted sub-cell averaging
          at material interfaces only.  Interior (pure-material) cells are
          assigned by the fast centre-label path.  Implied when ``yee=True``.

    yee : bool
        If ``False`` (default), return cell-centred permittivity tensors::

            x, y, xc, yc, nx, ny, epsxx, epsyy, epszz

        where each tensor has shape ``(ny, nx)``.  For isotropic materials
        ``epsxx == epsyy == epszz``; for anisotropic materials each holds
        ``n_x²``, ``n_y²``, or ``n_z²`` respectively.

        If ``True``, return Yee-staggered permittivity tensors with shapes
        matching the Yee field components:

        ============  ==============  ========================================
        tensor        shape           Yee grid location
        ============  ==============  ========================================
        ``epsxx``     ``(ny+1, nx)``  x-edge midpoints (Ex / Hy locations)
        ``epsyy``     ``(ny, nx+1)``  y-edge midpoints (Ey / Hx locations)
        ``epszz``     ``(ny+1, nx+1)`` node corners (Ez locations)
        ============  ==============  ========================================

        Boundary rows/columns use half-cell windows clipped to the domain edge.
        When ``yee=True``, ``method='fill'`` is used regardless of the
        *method* argument.

    Returns
    -------
    tuple
        ``(x, y, xc, yc, nx, ny, epsxx, epsyy, epszz)`` in both cases.
        Tensor shapes are ``(ny, nx)`` when ``yee=False`` (cell-centred) and
        staggered when ``yee=True``.

    Notes
    -----
    **Performance**: the ``fill`` path uses a two-pass algorithm.  All cells
    are first assigned a label via a vectorised centre-point query
    (Shapely 2.0 ufuncs when available, scalar loop fallback otherwise).
    Only cells whose label differs from any of the 8 (axis-aligned or
    diagonal) neighbours are then refined with ``fill_fractions``.  The
    diagonal check is essential near dielectric corners: when a cell centre
    lands in a majority quadrant all four axis-aligned neighbours may share
    the same label even though the cell straddles the corner boundary.  For
    a typical rectangular waveguide in a 3-layer stack this reduces expensive
    Shapely intersection calls from O(nx * ny) to O(nx + ny), yielding
    ~100–300× speedup.
    """
    if method not in ("center", "fill"):
        raise ValueError(f"method must be 'center' or 'fill', got {method!r}.")

    # yee=True forces fill method (point sampling is meaningless for off-centre windows)
    if yee:
        method = "fill"

    # -----------------------------------------------------------------------
    # Grid construction
    # -----------------------------------------------------------------------
    if x is not None:
        x = np.asarray(x, dtype=float)
        if xmin is None:
            xmin = float(x[0])
        if xmax is None:
            xmax = float(x[-1])
    else:
        if xmin is None or xmax is None or dx is None:
            raise ValueError(
                "Either the keyword argument 'x' or all of xmin, xmax, dx must be provided."
            )
        x = np.arange(xmin, xmax + dx / 2, dx)

    if y is not None:
        y = np.asarray(y, dtype=float)
        if ymin is None:
            ymin = float(y[0])
        if ymax is None:
            ymax = float(y[-1])
    else:
        if ymin is None or ymax is None or dy is None:
            raise ValueError(
                "Either the keyword argument 'y' or all of ymin, ymax, dy must be provided."
            )
        y = np.arange(ymin, ymax + dy / 2, dy)

    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    nx_cells = xc.size
    ny_cells = yc.size

    # -----------------------------------------------------------------------
    # Parse label_n: normalise every value to (n_x, n_y, n_z)
    # -----------------------------------------------------------------------
    parsed: dict[str, tuple] = {lbl: _parse_n(val) for lbl, val in label_n.items()}

    # -----------------------------------------------------------------------
    # Cell-centred label query  (ny, nx)
    # -----------------------------------------------------------------------
    XC, YC = np.meshgrid(xc, yc)
    lbl_cc = _labels_from_model(XC, YC, model)

    if not yee:
        epsxx = _labels_to_eps(lbl_cc, parsed, component=0)
        epsyy = _labels_to_eps(lbl_cc, parsed, component=1)
        epszz = _labels_to_eps(lbl_cc, parsed, component=2)
        if method == "fill":
            _fill_boundary_cells(
                epsxx, lbl_cc, model, parsed,
                x_lo=x[:-1], x_hi=x[1:], y_lo=y[:-1], y_hi=y[1:], component=0)
            _fill_boundary_cells(
                epsyy, lbl_cc, model, parsed,
                x_lo=x[:-1], x_hi=x[1:], y_lo=y[:-1], y_hi=y[1:], component=1)
            _fill_boundary_cells(
                epszz, lbl_cc, model, parsed,
                x_lo=x[:-1], x_hi=x[1:], y_lo=y[:-1], y_hi=y[1:], component=2)
        epsxx = _maybe_real(epsxx)
        epsyy = _maybe_real(epsyy)
        epszz = _maybe_real(epszz)
        return x, y, xc, yc, nx_cells, ny_cells, epsxx, epsyy, epszz

    # -----------------------------------------------------------------------
    # Yee-staggered window edge arrays
    # -----------------------------------------------------------------------
    # epsxx uses half-cell y windows: [yc[i-1], yc[i]] with sentinels at domain edges
    y_lo_xx = np.r_[ymin, yc]       # (ny+1,)
    y_hi_xx = np.r_[yc, ymax]       # (ny+1,)
    yc_xx   = 0.5 * (y_lo_xx + y_hi_xx)

    # epsyy uses half-cell x windows: [xc[j-1], xc[j]] with sentinels at domain edges
    x_lo_yy = np.r_[xmin, xc]       # (nx+1,)
    x_hi_yy = np.r_[xc, xmax]       # (nx+1,)
    xc_yy   = 0.5 * (x_lo_yy + x_hi_yy)

    # -----------------------------------------------------------------------
    # epsxx  (ny+1, nx) — Ex / Hy locations
    # -----------------------------------------------------------------------
    XC_xx, YC_xx = np.meshgrid(xc, yc_xx)
    lbl_xx = _labels_from_model(XC_xx, YC_xx, model)
    epsxx = _labels_to_eps(lbl_xx, parsed, component=0)
    _fill_boundary_cells(
        epsxx, lbl_xx, model, parsed,
        x_lo=x[:-1], x_hi=x[1:],
        y_lo=y_lo_xx, y_hi=y_hi_xx,
        component=0,
    )
    epsxx = _maybe_real(epsxx)

    # -----------------------------------------------------------------------
    # epsyy  (ny, nx+1) — Ey / Hx locations
    # -----------------------------------------------------------------------
    XC_yy, YC_yy = np.meshgrid(xc_yy, yc)
    lbl_yy = _labels_from_model(XC_yy, YC_yy, model)
    epsyy = _labels_to_eps(lbl_yy, parsed, component=1)
    _fill_boundary_cells(
        epsyy, lbl_yy, model, parsed,
        x_lo=x_lo_yy, x_hi=x_hi_yy,
        y_lo=y[:-1], y_hi=y[1:],
        component=1,
    )
    epsyy = _maybe_real(epsyy)

    # -----------------------------------------------------------------------
    # epszz  (ny+1, nx+1) — Ez corner locations
    # -----------------------------------------------------------------------
    XC_zz, YC_zz = np.meshgrid(xc_yy, yc_xx)
    lbl_zz = _labels_from_model(XC_zz, YC_zz, model)
    epszz = _labels_to_eps(lbl_zz, parsed, component=2)
    _fill_boundary_cells(
        epszz, lbl_zz, model, parsed,
        x_lo=x_lo_yy, x_hi=x_hi_yy,
        y_lo=y_lo_xx, y_hi=y_hi_xx,
        component=2,
    )
    epszz = _maybe_real(epszz)

    return x, y, xc, yc, nx_cells, ny_cells, epsxx, epsyy, epszz


def boundary_segments(
    model: GeometryModel,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list:
    """Return material-interface line segments within the computational domain.

    Produces a minimal set of polylines tracing the boundaries between regions
    of dissimilar material, ready for overlay on a permittivity plot.
    Compatible with :class:`matplotlib.collections.LineCollection`.

    Boundaries between adjacent regions that share the same material label are
    suppressed.  Segments that coincide with the computational-domain boundary
    are excluded.  Priority-based overlap resolution mirrors :func:`layoutmesh`:
    when finite regions overlap, the one with higher ``(priority, insertion_index)``
    wins.

    Parameters
    ----------
    model : GeometryModel
        The labeled 2D geometry.
    xmin, xmax, ymin, ymax : float
        Computational domain extent.

    Returns
    -------
    list[np.ndarray]
        Each element has shape ``(M, 2)`` and represents a connected polyline
        of ``(x, y)`` coordinates.  Pass directly to ``LineCollection``.

    Examples
    --------
    >>> edges = boundary_segments(model, xmin, xmax, ymin, ymax)
    >>> from matplotlib.collections import LineCollection
    >>> lc = LineCollection(edges, colors='k', linewidths=1)
    >>> ax.add_collection(lc)
    """
    from collections import defaultdict
    from shapely.geometry import box as _shapely_box
    from shapely.ops import unary_union

    domain = _shapely_box(xmin, ymin, xmax, ymax)
    domain_boundary = domain.boundary

    # Step 1: priority-resolved effective pieces (mirrors fill_fractions logic)
    uncovered = domain
    effective = []  # list of (label, Shapely geometry)

    candidates = sorted(model._regions, key=lambda r: (r.priority, r._index), reverse=True)
    for region in candidates:
        clipped = region.geometry.intersection(domain)
        if clipped.is_empty:
            continue
        piece = clipped.intersection(uncovered)
        if not piece.is_empty and piece.area > 0:
            effective.append((region.label, piece))
        uncovered = uncovered.difference(clipped)
        if uncovered.is_empty:
            break

    # Step 2: fill remaining uncovered area with background LayerStack bands
    if not uncovered.is_empty:
        for layer in model.layer_stack.clip_to_domain(ymin, ymax):
            band = _shapely_box(xmin, layer.ymin, xmax, layer.ymax)
            bg_piece = band.intersection(uncovered)
            if not bg_piece.is_empty and bg_piece.area > 0:
                effective.append((layer.label, bg_piece))

    # Step 3: merge same-label zones — shared edges between identical materials vanish
    label_geoms: dict = defaultdict(list)
    for label, geom in effective:
        label_geoms[label].append(geom)
    zones = {label: unary_union(geoms) for label, geoms in label_geoms.items()}

    # Step 4: extract interior boundaries; strip domain edges; GEOS deduplicates
    interior_parts = []
    for zone_geom in zones.values():
        # unary_union can return a GeometryCollection when inputs include
        # degenerate lower-dimensional components (e.g. a LineString along a
        # shared edge produced by band.intersection(uncovered) when the polygon
        # exactly coincides with a layer boundary).  Discard non-areal parts.
        if zone_geom.geom_type == 'GeometryCollection':
            polys = [g for g in zone_geom.geoms if g.area > 0]
            if not polys:
                continue
            zone_geom = unary_union(polys)
        if zone_geom.is_empty or zone_geom.area <= 0:
            continue
        interior = zone_geom.boundary.difference(domain_boundary)
        if not interior.is_empty:
            interior_parts.append(interior)

    if not interior_parts:
        return []

    merged = unary_union(interior_parts)
    segments: list = []
    _collect_linestrings(merged, segments)
    return segments


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _labels_from_model(
    xc_2d: np.ndarray,
    yc_2d: np.ndarray,
    model: GeometryModel,
) -> np.ndarray:
    """Return a 2D object array of material labels for every (x, y) position.

    Algorithm
    ---------
    1. Background LayerStack assignment using pure numpy comparisons — O(ny*nx)
       with no Python overhead per cell.
    2. Finite regions using Shapely 2.0 vectorised ufuncs (``shapely.within``)
       when available; falls back to the scalar ``label_at_point`` loop
       (Shapely < 2.0 compatible).
    """
    ni, nj = xc_2d.shape
    xc_flat = xc_2d.ravel()
    yc_flat = yc_2d.ravel()

    labels = np.empty(xc_flat.size, dtype=object)

    # --- Step 1: background from LayerStack (no Shapely) ---
    # Use a small tolerance on the upper bound to absorb floating-point
    # accumulation when yc lands infinitesimally above a layer boundary.
    _eps = 1e-10
    for layer in model.layer_stack.layers:
        lo = layer.ymin if np.isfinite(layer.ymin) else -np.inf
        hi = (layer.ymax + _eps) if np.isfinite(layer.ymax) else np.inf
        labels[(yc_flat >= lo) & (yc_flat <= hi)] = layer.label
    if model.layer_stack.default_label is not None:
        unlabeled = np.frompyfunc(lambda v: v is None, 1, 1)(labels).astype(bool)
        if unlabeled.any():
            labels[unlabeled] = model.layer_stack.default_label

    # --- Step 2: finite regions in ascending (priority, index) order ---
    if model._regions:
        regions_sorted = sorted(model._regions, key=lambda r: (r.priority, r._index))
        try:
            import shapely as _sh
            # Shapely 2.0: build geometry array once, then use vectorised within()
            pts = _sh.points(xc_flat, yc_flat)
            for region in regions_sorted:
                inside = _sh.within(pts, region.geometry)
                labels[inside] = region.label
        except (ImportError, AttributeError, TypeError):
            # Shapely < 2.0 fallback: scalar label_at_point for all cells
            for k in range(xc_flat.size):
                labels[k] = model.label_at_point(
                    float(xc_flat[k]), float(yc_flat[k])
                )

    return labels.reshape(ni, nj)


def _labels_to_eps(
    labels: np.ndarray,
    parsed: dict,
    component: int,
) -> np.ndarray:
    """Convert a 2D object label array to a 2D complex eps array.

    Iterates over unique labels rather than individual cells to avoid
    per-cell Python loop overhead.
    """
    eps = np.empty(labels.shape, dtype=complex)
    for lbl in np.unique(labels.ravel()):
        if lbl not in parsed:
            raise KeyError(
                f"Label {lbl!r} returned by geometry not found in label_n."
            )
        eps[labels == lbl] = parsed[lbl][component] ** 2
    return eps


def _boundary_mask_from_labels(labels: np.ndarray) -> np.ndarray:
    """Boolean mask: True for cells adjacent to or straddling a label boundary.

    A cell is flagged when any axis-aligned or diagonal neighbour carries a
    different material label.  The diagonal check catches the corner case where
    a cell centre falls close to a dielectric corner and all four axis-aligned
    neighbours share the same label — yet the cell still straddles the boundary
    (e.g. three quadrants are material A and one is B; the centre lands in A but
    the opposing diagonal neighbour resolves to B).
    """
    mask = np.zeros(labels.shape, dtype=bool)
    # N/S neighbours
    diff_v = labels[:-1, :] != labels[1:, :]   # (ni-1, nj)
    mask[:-1, :] |= diff_v
    mask[1:, :] |= diff_v
    # E/W neighbours
    diff_h = labels[:, :-1] != labels[:, 1:]   # (ni, nj-1)
    mask[:, :-1] |= diff_h
    mask[:, 1:] |= diff_h
    # NE/SW diagonal neighbours
    diff_ne = labels[:-1, :-1] != labels[1:, 1:]  # (ni-1, nj-1)
    mask[:-1, :-1] |= diff_ne
    mask[1:, 1:] |= diff_ne
    # NW/SE diagonal neighbours
    diff_nw = labels[:-1, 1:] != labels[1:, :-1]  # (ni-1, nj-1)
    mask[:-1, 1:] |= diff_nw
    mask[1:, :-1] |= diff_nw
    return mask


def _fill_boundary_cells(
    eps: np.ndarray,
    labels: np.ndarray,
    model: GeometryModel,
    parsed: dict,
    x_lo: np.ndarray,
    x_hi: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    component: int,
) -> None:
    """In-place: replace boundary-cell values with accurate fill-fraction averages.

    Only the small fraction of cells flagged by
    :func:`_boundary_mask_from_labels` are updated; the vast majority of
    interior (pure-material) cells retain their fast label-assigned value.
    """
    mask = _boundary_mask_from_labels(labels)
    for i, j in zip(*np.where(mask)):
        fracs = model.fill_fractions(
            float(x_lo[j]), float(x_hi[j]),
            float(y_lo[i]), float(y_hi[i]),
        )
        eps[i, j] = _weighted_eps(fracs, parsed, component)


def _parse_n(val) -> tuple:
    """Return a (n_x, n_y, n_z) triple from a scalar or 3-tuple."""
    if hasattr(val, "__len__"):
        val = tuple(val)
        if len(val) != 3:
            raise ValueError(
                f"Anisotropic n must be a 3-tuple (n_x, n_y, n_z); got length {len(val)}."
            )
        return val
    return (val, val, val)


def _weighted_eps(
    fracs: dict[str, float],
    parsed: dict[str, tuple],
    component: int,
) -> complex:
    """Arithmetic area-weighted average of n_component² over fill fractions."""
    total = 0.0 + 0j
    for lbl, frac in fracs.items():
        if lbl not in parsed:
            raise KeyError(
                f"Label {lbl!r} returned by geometry not found in label_n."
            )
        total += frac * (parsed[lbl][component] ** 2)
    return total


def _maybe_real(arr: np.ndarray) -> np.ndarray:
    """Downcast to float64 if the imaginary part is negligible."""
    if np.iscomplexobj(arr) and np.allclose(arr.imag, 0.0, atol=1e-12):
        return arr.real.astype(float)
    return arr


def _collect_linestrings(geom, out: list) -> None:
    """Recursively extract (M, 2) coord arrays from any Shapely linear geometry."""
    gtype = geom.geom_type
    if gtype in ("LineString", "LinearRing"):
        coords = np.array(geom.coords)
        if len(coords) >= 2:
            out.append(coords)
    elif gtype in ("MultiLineString", "GeometryCollection"):
        for g in geom.geoms:
            _collect_linestrings(g, out)


__all__ = ["layoutmesh", "boundary_segments"]
