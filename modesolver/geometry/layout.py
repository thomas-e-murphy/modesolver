"""
layout.py — labeled 2D geometry for finite-difference eigenmode solvers.

Represents material-labeled geometry in real (physical) coordinates,
independent of the numerical mesh and refractive indices.  The external
geometry engine is Shapely; all coordinates are floating-point real-space
values.

Public API
----------
Data structures:
    Layer, LayerStack, Region, ClippedShape, GeometryModel

Factory functions (return Region objects):
    rectangle, polygon, disk, from_shapely

Optional GDS I/O (requires ``gdstk``):
    gds_import, gds_export
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from shapely.geometry import (
    Point,
    Polygon as ShapelyPolygon,
    box as shapely_box,
    MultiPolygon,
)
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Layer / LayerStack
# ---------------------------------------------------------------------------

@dataclass
class Layer:
    """A horizontal background slab defined by a y-interval and a label.

    Parameters
    ----------
    ymin, ymax : float
        Vertical extents of the layer (ymin < ymax).
    label : str
        Material identifier, e.g. ``"substrate"``, ``"core"``.
    """
    ymin: float
    ymax: float
    label: str

    def __post_init__(self):
        if self.ymin >= self.ymax:
            raise ValueError(
                f"Layer '{self.label}': ymin ({self.ymin}) must be < ymax ({self.ymax})."
            )


class LayerStack:
    """Ordered, non-overlapping vertical background layers.

    The stack fills the y-axis from the lowest ``ymin`` to the highest
    ``ymax``.  Gaps between layers are allowed only when *default_label*
    is supplied; otherwise a gap raises ``ValueError``.

    Parameters
    ----------
    layers : list[Layer]
        The layers.  They will be sorted by ``ymin`` automatically.
    default_label : str or None
        Label returned for y-values that fall outside all defined layers
        (including gaps and the regions above/below the stack).  If
        ``None``, querying an uncovered y raises ``ValueError``.

    Raises
    ------
    ValueError
        If two layers overlap in y, or if there is a gap and no
        *default_label* is provided.
    """

    def __init__(self, layers: list[Layer], default_label: Optional[str] = None):
        self.default_label = default_label
        self.layers: list[Layer] = sorted(layers, key=lambda l: l.ymin)
        self._validate()

    def _validate(self):
        for i in range(len(self.layers) - 1):
            a, b = self.layers[i], self.layers[i + 1]
            if a.ymax > b.ymin:
                raise ValueError(
                    f"Layers '{a.label}' and '{b.label}' overlap: "
                    f"[{a.ymin}, {a.ymax}] vs [{b.ymin}, {b.ymax}]."
                )
            if a.ymax < b.ymin and self.default_label is None:
                raise ValueError(
                    f"Gap between layers '{a.label}' (ymax={a.ymax}) and "
                    f"'{b.label}' (ymin={b.ymin}) with no default_label supplied."
                )

    def label_at_y(self, y: float) -> str:
        """Return the background material label at height *y*.

        Parameters
        ----------
        y : float

        Returns
        -------
        str

        Raises
        ------
        ValueError
            If *y* is not covered by any layer and no *default_label* was set.
        """
        for layer in self.layers:
            if layer.ymin <= y <= layer.ymax:
                return layer.label
        if self.default_label is not None:
            return self.default_label
        raise ValueError(
            f"y={y} is not covered by any layer and no default_label is set."
        )

    def clip_to_domain(self, ymin: float, ymax: float) -> list[Layer]:
        """Return layers clipped to [ymin, ymax], including any default gap fills.

        Parameters
        ----------
        ymin, ymax : float
            Vertical extent of the computational domain.

        Returns
        -------
        list[Layer]
            Layers whose extents are clipped to [ymin, ymax].  For gaps
            covered by *default_label*, synthetic layers are inserted.
            The returned list is sorted by ymin and covers [ymin, ymax]
            completely when *default_label* is set.
        """
        clipped: list[Layer] = []
        prev_y = ymin

        for layer in self.layers:
            lo = max(layer.ymin, ymin)
            hi = min(layer.ymax, ymax)
            if hi <= lo:
                continue
            # Fill any gap before this layer with default label
            if prev_y < lo and self.default_label is not None:
                clipped.append(Layer(prev_y, lo, self.default_label))
            clipped.append(Layer(lo, hi, layer.label))
            prev_y = hi

        # Fill trailing gap (after last layer) with default label
        if prev_y < ymax and self.default_label is not None:
            clipped.append(Layer(prev_y, ymax, self.default_label))

        return clipped


# ---------------------------------------------------------------------------
# Region / ClippedShape
# ---------------------------------------------------------------------------

@dataclass
class Region:
    """A bounded 2D region with a material label and priority.

    Use the factory functions :func:`rectangle`, :func:`polygon`,
    :func:`disk`, or :func:`from_shapely` to create instances.

    Parameters
    ----------
    geometry : shapely.geometry.base.BaseGeometry
        A Shapely geometry (typically a Polygon).
    label : str
        Material identifier.
    priority : int
        Override order when regions overlap.  Higher values take
        precedence.  When two regions share the same priority, the one
        added later to a :class:`GeometryModel` wins (last-writer-wins).
    """
    geometry: object  # shapely BaseGeometry
    label: str
    priority: int = 0
    _index: int = field(default=-1, repr=False, compare=False)


@dataclass
class ClippedShape:
    """A Shapely polygon clipped to the computational domain.

    Produced by :meth:`GeometryModel.clip_to_domain`.

    Attributes
    ----------
    polygon : shapely.geometry.Polygon
    label : str
    priority : int
        Background layers receive ``priority = -1``; finite regions
        use their own priority.
    """
    polygon: object  # shapely Polygon
    label: str
    priority: int


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def rectangle(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    label: str,
    priority: int = 0,
) -> Region:
    """Create an axis-aligned rectangular :class:`Region`.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Bounding coordinates (xmin < xmax, ymin < ymax).
    label : str
        Material label.
    priority : int
        Override priority (higher wins).

    Returns
    -------
    Region
    """
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("rectangle: require xmin < xmax and ymin < ymax.")
    return Region(geometry=shapely_box(xmin, ymin, xmax, ymax), label=label, priority=priority)


def polygon(
    vertices: list[tuple[float, float]],
    label: str,
    priority: int = 0,
) -> Region:
    """Create a polygonal :class:`Region` from a vertex list.

    Parameters
    ----------
    vertices : list of (x, y) tuples
        At least 3 vertices.
    label : str
        Material label.
    priority : int
        Override priority.

    Returns
    -------
    Region
    """
    if len(vertices) < 3:
        raise ValueError("polygon: at least 3 vertices required.")
    return Region(geometry=ShapelyPolygon(vertices), label=label, priority=priority)


def disk(
    cx: float,
    cy: float,
    radius: float,
    label: str,
    priority: int = 0,
    n_pts: int = 64,
) -> Region:
    """Create a circular disk :class:`Region` approximated by a polygon.

    Parameters
    ----------
    cx, cy : float
        Center coordinates.
    radius : float
        Radius (must be > 0).
    label : str
        Material label.
    priority : int
        Override priority.
    n_pts : int
        Number of polygon vertices used to approximate the circle.
        Default is 64.

    Returns
    -------
    Region
    """
    if radius <= 0:
        raise ValueError("disk: radius must be positive.")
    if n_pts < 3:
        raise ValueError("disk: n_pts must be >= 3.")
    angles = np.linspace(0.0, 2.0 * math.pi, n_pts, endpoint=False)
    verts = [(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles]
    return Region(geometry=ShapelyPolygon(verts), label=label, priority=priority)


def from_shapely(
    geom: object,
    label: str,
    priority: int = 0,
) -> Region:
    """Wrap an existing Shapely geometry as a :class:`Region`.

    Parameters
    ----------
    geom : shapely geometry
        Any Shapely geometry (Polygon, MultiPolygon, etc.).
    label : str
        Material label.
    priority : int
        Override priority.

    Returns
    -------
    Region
    """
    return Region(geometry=geom, label=label, priority=priority)


# ---------------------------------------------------------------------------
# GeometryModel
# ---------------------------------------------------------------------------

class GeometryModel:
    """A 2D geometry composed of a background :class:`LayerStack` and
    zero or more finite :class:`Region` objects.

    Background layers extend infinitely in x; finite regions are bounded.
    When regions overlap, the one with higher *priority* wins.  Ties are
    broken by insertion order: the region added *later* wins.

    Parameters
    ----------
    layer_stack : LayerStack
        The background material structure.
    regions : list[Region], optional
        Initial list of finite regions.  More can be added via
        :meth:`add_region`.

    Examples
    --------
    >>> from modesolver.geometry.layout import *
    >>> stack = LayerStack([Layer(0, 1, "substrate"), Layer(1, 2, "cladding")])
    >>> core = rectangle(-0.3, 0.3, 1.0, 1.4, "core", priority=1)
    >>> model = GeometryModel(stack, [core])
    >>> model.label_at_point(0.0, 1.2)
    'core'
    >>> model.label_at_point(0.0, 0.5)
    'substrate'
    """

    def __init__(
        self,
        layer_stack: LayerStack,
        regions: Optional[list[Region]] = None,
    ):
        self.layer_stack = layer_stack
        self._regions: list[Region] = []
        self._strtree: Optional[STRtree] = None  # rebuilt lazily on access

        for r in (regions or []):
            self.add_region(r)

    # ------------------------------------------------------------------
    # Region management
    # ------------------------------------------------------------------

    def add_region(self, region: Region) -> None:
        """Add a finite region to the model.

        The region is assigned an internal insertion index used as a
        tie-breaker when two regions share the same priority.

        Parameters
        ----------
        region : Region
        """
        r = Region(
            geometry=region.geometry,
            label=region.label,
            priority=region.priority,
            _index=len(self._regions),
        )
        self._regions.append(r)
        self._strtree = None  # invalidate cache

    @property
    def regions(self) -> list[Region]:
        """Read-only view of all added regions."""
        return list(self._regions)

    # ------------------------------------------------------------------
    # Spatial index (lazy)
    # ------------------------------------------------------------------

    def _get_strtree(self) -> STRtree:
        if self._strtree is None:
            self._strtree = STRtree([r.geometry for r in self._regions])
        return self._strtree

    # ------------------------------------------------------------------
    # Point query
    # ------------------------------------------------------------------

    def label_at_point(self, x: float, y: float) -> str:
        """Return the effective material label at position *(x, y)*.

        Finite regions override the background layer; among overlapping
        regions the highest *(priority, insertion_index)* wins.

        Parameters
        ----------
        x, y : float
            Query coordinates.

        Returns
        -------
        str
            Material label.
        """
        pt = Point(x, y)
        tree = self._get_strtree()
        candidates = tree.query(pt)
        best: Optional[Region] = None
        for idx in candidates:
            r = self._regions[idx]
            if r.geometry.contains(pt):
                if best is None or (r.priority, r._index) > (best.priority, best._index):
                    best = r
        if best is not None:
            return best.label
        return self.layer_stack.label_at_y(y)

    # ------------------------------------------------------------------
    # Fill-fraction query
    # ------------------------------------------------------------------

    def fill_fractions(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> dict[str, float]:
        """Return area fractions by label within a rectangular cell.

        The fractions sum to approximately 1.0.  Background layers fill
        whatever area is not covered by finite regions.  When finite
        regions overlap, higher *(priority, insertion_index)* wins.

        This is intended for sub-cell dielectric averaging on a Yee grid.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Cell boundaries.

        Returns
        -------
        dict[str, float]
            ``{label: fraction}`` where fractions sum to ~1.0.
        """
        cell = shapely_box(xmin, ymin, xmax, ymax)
        cell_area = cell.area
        if cell_area == 0.0:
            return {}

        fractions: dict[str, float] = {}

        # Process regions from *lowest* to *highest* priority so that
        # higher-priority regions overwrite lower ones.  We work with
        # a "remaining uncovered area" polygon and subtract each region's
        # contribution in descending priority order.
        tree = self._get_strtree()
        candidates_idx = tree.query(cell)
        candidates = [
            self._regions[i] for i in candidates_idx
            if not self._regions[i].geometry.intersection(cell).is_empty
        ]
        # Sort descending so highest priority is processed first
        candidates.sort(key=lambda r: (r.priority, r._index), reverse=True)

        uncovered = cell
        for r in candidates:
            clipped = r.geometry.intersection(uncovered)
            if clipped.is_empty:
                continue
            area = clipped.area
            if area > 0.0:
                fractions[r.label] = fractions.get(r.label, 0.0) + area / cell_area
            uncovered = uncovered.difference(r.geometry)
            if uncovered.is_empty:
                break

        # Remaining area belongs to the background layer stack
        if not uncovered.is_empty:
            bg_layers = self.layer_stack.clip_to_domain(ymin, ymax)
            for layer in bg_layers:
                layer_box = shapely_box(xmin, layer.ymin, xmax, layer.ymax)
                bg_piece = uncovered.intersection(layer_box)
                if bg_piece.is_empty:
                    continue
                area = bg_piece.area
                if area > 0.0:
                    fractions[layer.label] = fractions.get(layer.label, 0.0) + area / cell_area

        return fractions

    # ------------------------------------------------------------------
    # Domain clipping / compilation
    # ------------------------------------------------------------------

    def clip_to_domain(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> list[ClippedShape]:
        """Clip all geometry to a rectangular computational domain.

        Returns a list of :class:`ClippedShape` objects ready for mesh
        assignment or rendering.  Background layers receive
        ``priority = -1``; finite regions keep their own priority.
        The list is sorted ascending by priority so that background is
        rendered/assigned first and higher-priority regions appear last.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Domain boundaries.

        Returns
        -------
        list[ClippedShape]
            Clipped shapes sorted by ascending priority.
        """
        domain = shapely_box(xmin, ymin, xmax, ymax)
        shapes: list[ClippedShape] = []

        # Background layers
        for layer in self.layer_stack.clip_to_domain(ymin, ymax):
            layer_box = shapely_box(xmin, layer.ymin, xmax, layer.ymax)
            clipped = layer_box.intersection(domain)
            if not clipped.is_empty:
                for poly in _iter_polygons(clipped):
                    shapes.append(ClippedShape(polygon=poly, label=layer.label, priority=-1))

        # Finite regions
        for r in self._regions:
            clipped = r.geometry.intersection(domain)
            if not clipped.is_empty:
                for poly in _iter_polygons(clipped):
                    shapes.append(ClippedShape(polygon=poly, label=r.label, priority=r.priority))

        shapes.sort(key=lambda s: s.priority)
        return shapes

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        ax=None,
        colormap=None,
    ):
        """Plot the geometry within the computational domain.

        Requires ``matplotlib``.

        Parameters
        ----------
        xmin, xmax, ymin, ymax : float
            Domain extent.
        ax : matplotlib.axes.Axes, optional
            Existing axes to draw into.  A new figure is created if not
            provided.
        colormap : str or matplotlib Colormap, optional
            Colormap used to assign distinct colors to labels.  Defaults
            to ``'tab10'``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path

        if ax is None:
            _, ax = plt.subplots()

        shapes = self.clip_to_domain(xmin, xmax, ymin, ymax)

        # Assign a consistent color to each unique label
        labels_ordered = []
        seen = set()
        for s in shapes:
            if s.label not in seen:
                labels_ordered.append(s.label)
                seen.add(s.label)

        cmap = plt.get_cmap(colormap or "tab10")
        color_map = {lbl: cmap(i / max(len(labels_ordered), 1))
                     for i, lbl in enumerate(labels_ordered)}

        for s in shapes:
            _plot_polygon(ax, s.polygon, color=color_map[s.label])

        # Legend
        handles = [
            mpatches.Patch(facecolor=color_map[lbl], edgecolor="k", label=lbl)
            for lbl in labels_ordered
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=8)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return ax


# ---------------------------------------------------------------------------
# GDS import / export (optional — requires gdstk)
# ---------------------------------------------------------------------------

def gds_import(
    filename: str,
    layer_map: dict[tuple[int, int], str],
    default_priority: int = 0,
) -> list[Region]:
    """Import polygons from a GDSII file as :class:`Region` objects.

    Requires the ``gdstk`` package (``pip install gdstk``).

    Parameters
    ----------
    filename : str
        Path to the ``.gds`` or ``.oas`` file.
    layer_map : dict
        Maps ``(gds_layer, datatype)`` integer pairs to label strings,
        e.g. ``{(1, 0): "core", (2, 0): "trench"}``.
    default_priority : int
        Priority assigned to all imported regions.

    Returns
    -------
    list[Region]
    """
    try:
        import gdstk
    except ImportError as exc:
        raise ImportError(
            "gds_import requires 'gdstk'.  Install with: pip install gdstk"
        ) from exc

    lib = gdstk.read_gds(filename)
    regions: list[Region] = []
    for cell in lib.cells:
        for poly in cell.polygons:
            key = (poly.layer, poly.datatype)
            if key not in layer_map:
                continue
            label = layer_map[key]
            pts = [(float(x), float(y)) for x, y in poly.points]
            regions.append(
                Region(
                    geometry=ShapelyPolygon(pts),
                    label=label,
                    priority=default_priority,
                )
            )
    return regions


def gds_export(
    model: GeometryModel,
    filename: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    label_map: Optional[dict[str, tuple[int, int]]] = None,
) -> None:
    """Export a :class:`GeometryModel` clipped to a domain as a GDSII file.

    Requires the ``gdstk`` package (``pip install gdstk``).

    Parameters
    ----------
    model : GeometryModel
    filename : str
        Output filename (``*.gds`` or ``*.oas``).
    xmin, xmax, ymin, ymax : float
        Domain boundaries used for clipping.
    label_map : dict, optional
        Maps label strings to ``(gds_layer, datatype)`` integer pairs.
        If not supplied, labels are auto-assigned to consecutive layers
        starting at layer 1, datatype 0.
    """
    try:
        import gdstk
    except ImportError as exc:
        raise ImportError(
            "gds_export requires 'gdstk'.  Install with: pip install gdstk"
        ) from exc

    shapes = model.clip_to_domain(xmin, xmax, ymin, ymax)

    if label_map is None:
        unique_labels = list({s.label for s in shapes})
        label_map = {lbl: (i + 1, 0) for i, lbl in enumerate(sorted(unique_labels))}

    lib = gdstk.Library()
    cell = lib.new_cell("GEOMETRY")
    for s in shapes:
        if s.label not in label_map:
            continue
        gds_layer, datatype = label_map[s.label]
        coords = list(s.polygon.exterior.coords)
        pts = [(float(x), float(y)) for x, y in coords]
        cell.add(gdstk.Polygon(pts, layer=gds_layer, datatype=datatype))

    lib.write_gds(filename)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_polygons(geom):
    """Yield simple Polygon objects from any Shapely geometry."""
    if geom.geom_type == "Polygon":
        yield geom
    elif geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        for g in geom.geoms:
            yield from _iter_polygons(g)


def _plot_polygon(ax, polygon, color):
    """Fill a Shapely Polygon on *ax* using a matplotlib PathPatch."""
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path

    def _ring_to_codes(ring):
        n = len(ring.coords)
        codes = [Path.LINETO] * n
        codes[0] = Path.MOVETO
        codes[-1] = Path.CLOSEPOLY
        return list(ring.coords), codes

    verts, codes = _ring_to_codes(polygon.exterior)
    for interior in polygon.interiors:
        iv, ic = _ring_to_codes(interior)
        verts += iv
        codes += ic

    path = Path(verts, codes)
    patch = PathPatch(path, facecolor=color, edgecolor="k", linewidth=0.5, alpha=0.7)
    ax.add_patch(patch)


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "Layer",
    "LayerStack",
    "Region",
    "ClippedShape",
    "GeometryModel",
    "rectangle",
    "polygon",
    "disk",
    "from_shapely",
    "gds_import",
    "gds_export",
]
