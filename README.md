# modesolver

A Python library for computing the electromagnetic eigenmodes of dielectric optical waveguides using the finite difference method.  This library uses the method described in: 

A. B. Fallahkhair, K. S. Li and T. E. Murphy,
"Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides",
J. Lightwave Technol. 26(11), 1423-1431 (2008).
https://doi.org/10.1109/JLT.2008.923643

New in the latest release:

Automatic selection of the best available solver for shift-invert eigenvalue problems
  - PyPardiso support (fastest for real matrices, requires Intel MKL)
  - MUMPS support (efficient for both real and complex matrices)
  - Falls back to SciPy's SuperLU when optional packages are unavailable

In addition to the original node-centered formulation, the library now includes an alternative finite-difference eigenmode solver based on a Yee (staggered-grid) discretization. In this formulation, the electric and magnetic field components are defined on offset grids, analogous to the spatial staggering used in finite-difference time-domain (FDTD) methods.

The library also includes examples, as well as a set of tools for defining refractive index profiles, and post-processing the modes.

## Geometry and Layout

The library includes a geometry/layout subsystem (`modesolver.geometry.layout`)
for describing device cross-sections as labeled 2D regions in real (physical)
coordinates, completely independent of the numerical mesh and refractive
indices.

Material labels are plain strings (e.g. `"substrate"`, `"core"`, `"air"`).
The internal geometry engine is [Shapely](https://shapely.readthedocs.io/).

Key classes and factory functions:

| Symbol | Description |
|---|---|
| `Layer` / `LayerStack` | Ordered, non-overlapping horizontal background layers |
| `Region` | Bounded 2D shape with a label and priority |
| `GeometryModel` | Combines a `LayerStack` with finite `Region` objects |
| `rectangle(xmin,xmax,ymin,ymax,label)` | Axis-aligned rectangular region |
| `polygon(vertices, label)` | Arbitrary polygon region |
| `disk(cx, cy, radius, label)` | Circular region (polygon approximation) |
| `from_shapely(geom, label)` | Wrap any existing Shapely geometry |

`GeometryModel` provides:
- `label_at_point(x, y)` — effective material label at a point (background
  layers supply the default; finite regions override by priority)
- `fill_fractions(xmin, xmax, ymin, ymax)` — area fractions by label over a
  rectangular cell, for sub-cell dielectric averaging on a Yee grid
- `clip_to_domain(xmin, xmax, ymin, ymax)` — clip all geometry to the
  computational domain and return priority-sorted `ClippedShape` objects
- `plot(xmin, xmax, ymin, ymax)` — quick matplotlib visualization

Optional GDSII import/export is available via the `gds_import()` and
`gds_export()` functions when `gdstk` is installed.

## Building a Permittivity Grid

`layoutmesh()` (`modesolver.geometry.layoutmesh`) converts a `GeometryModel`
and a `{label → n}` mapping into a finite-difference permittivity grid:

```python
from modesolver.geometry.layoutmesh import layoutmesh, boundary_segments

x, y, xc, yc, nx, ny, epsxx, epsyy, epszz = layoutmesh(
    model, label_n, xmin=..., xmax=..., ymin=..., ymax=..., dx=..., dy=...
)
```

The function always returns three permittivity tensors — `epsxx`, `epsyy`,
`epszz` — corresponding to the `n_x²`, `n_y²`, and `n_z²` components.
Tensor shapes depend on the `yee` flag:

| `yee` | `epsxx` | `epsyy` | `epszz` |
|-------|---------|---------|---------|
| `False` (default) | `(ny, nx)` | `(ny, nx)` | `(ny, nx)` |
| `True` | `(ny+1, nx)` | `(ny, nx+1)` | `(ny+1, nx+1)` |

With `yee=True` the tensors are staggered to match the Yee field-component
grid locations used by `wgmodes_yee`, and sub-cell area averaging
(`method='fill'`) is applied automatically.

`boundary_segments()` returns a list of `(M, 2)` polyline arrays tracing
material-interface boundaries within the domain, with shared-label edges and
domain-boundary coincident segments suppressed.  Pass directly to
`matplotlib.collections.LineCollection` to overlay on any field or
permittivity plot.

## Installation

You can install directly from GitHub:

```
pip install git+https://github.com/thomas-e-murphy/modesolver.git
```

To also enable GDSII import/export:

```
pip install "modesolver[gds] @ git+https://github.com/thomas-e-murphy/modesolver.git"
```
