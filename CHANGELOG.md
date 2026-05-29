# Changelog

All notable changes to the modesolver library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.2] - 2026-05-28

### Added
- **`layoutmesh()` function** (`geometry/layoutmesh.py`): Build a finite-difference
  permittivity grid from a `GeometryModel` plus a `{label → n}` mapping, bridging the
  new label-based geometry representation with the numerical solver interface:
  - Accepts isotropic (`n` scalar) or diagonal-anisotropic (`(n_x, n_y, n_z)` 3-tuple)
    refractive indices per label; single value is broadcast to all three components.
  - `method='center'` (default): point-sample `label_at_point` at each cell centre —
    fast, equivalent in spirit to the legacy `waveguidemesh` routine.
  - `method='fill'`: area-weighted sub-cell averaging via `fill_fractions` — more
    accurate at material interfaces.
  - Always returns three separate permittivity tensors `epsxx`, `epsyy`, `epszz`
    (never a single isotropic `eps`):
    - `yee=False` (default): cell-centred tensors, each of shape ``(ny, nx)``.
      For isotropic materials ``epsxx == epsyy == epszz``; for anisotropic
      materials each holds ``n_x²``, ``n_y²``, or ``n_z²`` respectively.
    - `yee=True`: Yee-staggered tensors with shapes matching the field-component
      grid locations used by `wgmodes_yee`:
      - `epsxx` ``(ny+1, nx)`` — at Ex / Hy locations (x-edge midpoints)
      - `epsyy` ``(ny, nx+1)`` — at Ey / Hx locations (y-edge midpoints)
      - `epszz` ``(ny+1, nx+1)`` — at Ez corner locations
      - Boundary rows/columns use half-cell windows clipped to the domain edge,
        so no ghost cells are needed; `method='fill'` is used regardless of
        the *method* argument.
  - Return tuple: `(x, y, xc, yc, nx, ny, epsxx, epsyy, epszz)` in both cases.
- **`boundary_segments()` function** (`geometry/layoutmesh.py`): Returns a list of
  polyline coordinate arrays tracing the material-interface boundaries within the
  computational domain, suitable for overlaying on permittivity or field plots:
  - Boundaries between adjacent regions sharing the same material label are
    suppressed; segments coinciding with the domain boundary are excluded.
  - Priority-based overlap resolution mirrors `layoutmesh()`.
  - Each element has shape ``(M, 2)`` and represents a connected polyline of
    ``(x, y)`` coordinates; pass directly to
    `matplotlib.collections.LineCollection`.
- **`wgmodes_yee` staggered-eps support** (`wgmodes_yee.py`): Auto-detects whether
  the supplied `epsxx`/`epsyy`/`epszz` are in the legacy cell-centred format
  `(ny, nx)` or the new Yee-located format `(ny+1, nx)` / `(ny, nx+1)` /
  `(ny+1, nx+1)`.  Backward compatible — existing code using `(ny, nx)` tensors
  continues to work unchanged.  When staggered tensors are detected, the internal
  cell-averaging step is bypassed and the pre-computed values are used directly.
- **Geometry/layout subsystem** (`geometry/layout.py`): A new, clean module
  for representing labeled 2D material geometry in real (physical) coordinates,
  independent of the numerical mesh and refractive indices.  Key components:
  - **`Layer`** / **`LayerStack`**: define an ordered, non-overlapping vertical
    background structure with string material labels (e.g. `"substrate"`,
    `"cladding"`).  The stack sorts layers automatically, raises on overlap,
    and supports an optional `default_label` for gaps or regions outside the
    defined extent.
  - **`Region`** / **`GeometryModel`**: a bounded 2D object (backed by Shapely)
    with a label and integer priority.  `GeometryModel` combines a `LayerStack`
    background with zero or more finite `Region` objects and uses a Shapely
    `STRtree` for efficient spatial queries.
  - **Factory functions**: `rectangle()`, `polygon()`, `disk()` (circle
    approximated by an n-gon), and `from_shapely()` for wrapping any existing
    Shapely geometry.
  - **`label_at_point(x, y)`**: returns the effective material label at a
    point; finite regions override the background layer by priority, with
    last-added winning on ties.
  - **`fill_fractions(xmin, xmax, ymin, ymax)`**: returns a `{label: fraction}`
    dict over a rectangular cell for sub-cell dielectric averaging; fractions
    sum to 1.0 and use Shapely intersection areas.
  - **`clip_to_domain(xmin, xmax, ymin, ymax)`**: clips all geometry to a
    rectangular computational domain and returns a priority-sorted list of
    `ClippedShape` objects ready for mesh assignment or rendering.
  - **`plot()`**: simple matplotlib visualization of the clipped geometry with
    auto-assigned colors per label.
  - **Optional GDS I/O** (`gds_import()`, `gds_export()`): import polygons
    from GDSII files and export geometry back to GDSII, mapping string labels
    to GDS layer/datatype integer pairs.  Requires the optional `gdstk`
    package (`pip install modesolver[gds]`).
  - New required dependency: `shapely`.
  - New optional dependency group `[gds]`: `gdstk`.
- **`field_to_bitmap()` function** (`postprocess/export.py`): Export any 2D scalar field
  to an indexed-color PNG or TIFF with square pixels, resampling from the (possibly
  non-uniform or staggered) input grid to a user-specified uniform output grid:
  - Accepts any field grid location (cell-centered, vertex-centered, or Yee-staggered
    edge); caller passes the 1D coordinate arrays matching the field's actual grid
  - `dpx` parameter sets the square pixel size; defaults to the median input grid spacing
  - `xmin`/`xmax`/`ymin`/`ymax` window crops or extends the output domain; pixels outside
    the input domain are filled by clamping to the nearest input edge value (matching
    the implicit behavior of `pcolormesh` with flat shading)
  - `cmap` selects any matplotlib colormap; the 256-entry palette is embedded in the image
  - `vmin`/`vmax` control the data range mapped to palette indices 0–255
  - `vcenter` forces a symmetric colormap range about a specified value (e.g.
    `vcenter=0` with `cmap='seismic'` for signed field components); may be combined
    with `vmin` or `vmax` (but not both) to set the half-range explicitly
  - File format (PNG or TIFF) inferred automatically from the filename extension by Pillow
  - Returns a `PIL.Image` object (mode `'P'`); also saves to file if `filename` is given
  - New dependency: `Pillow`; `scipy` and `matplotlib` are also used
- **`field_to_contours()` function** (`postprocess/export.py`): Generate a transparent
  SVG contour overlay to pair with `field_to_bitmap` output:
  - Accepts the same `field`, `x`, `y`, `xmin`/`xmax`/`ymin`/`ymax` arguments as
    `field_to_bitmap`; specifying the same window in both calls ensures the SVG covers
    the same physical region as the PNG and can be scaled to overlay it precisely in
    any vector editor
  - Figure viewport aspect ratio is derived from the physical window, so the SVG is
    geometrically correct at any scale with no manual adjustment needed
  - `levels` passed directly to `matplotlib contour()` — caller supplies values in
    whatever units are desired (raw field, dB, normalized, etc.)
  - `colors`, `linewidths`, `linestyles` passed through to `ax.contour()`; defaults
    to solid black 0.5-pt lines
  - Output is a pure vector SVG with transparent background, compatible with Adobe
    Illustrator, Inkscape, and similar tools
  - Saves to file if `filename` is given; always returns the `Figure` for inline
    display or further customization (e.g. `plt.close(fig)` after `display(fig)`)
- **`group_index()` function** (`postprocess/group_index.py`): Compute the group index of a waveguide mode using the weighted integral formula:
  - Numerator: ∫∫ [(2·ng·n − n²)·|E|² + |H|²] dA, where ng(x,y) and n(x,y) are the local group and phase indices of the constituent materials
  - Denominator: ∫∫ (E×H* + E*×H)·ẑ dA (equal to 4·Sz from `poynting()`)
  - **Isotropic materials only**: the scalar ng/n formulation assumes each material has a scalar permittivity; anisotropic materials would require per-component group indices
  - Automatically collocates staggered fields to cell centres if needed (works with output from `wgmodes`, `wgmodes_yee`, or pre-collocated fields)
  - Supports non-uniform grids: cell areas computed from exact dx/dy vectors via `np.outer(dy, dx)`
  - Integration uses the Riemann (midpoint-rule) sum, which is second-order accurate for cell-centred fields
  - Multi-mode support: returns a scalar for 2-D field input (nmodes=1) and an array of shape (nmodes,) for 3-D field input

### Fixed
- **Memory leak in sparse solver interface** (`sparse_solve.py`, `wgmodes.py`, `wgmodes_yee.py`, `svmodes.py`): Solver factorization memory is now released immediately after each eigensolve, preventing `PyPardisoError: error code -2 (Not enough memory)` and equivalent accumulation in MUMPS when the modesolver is called hundreds of times in a loop
  - `make_shift_invert_operator()` now returns the solver object instead of its name string, giving callers explicit ownership and the ability to free resources deterministically
  - `SparseSolver` base class gains a no-op `free()` method so all subclasses are safe to call without a type check
  - `PyPardisoSolver.free()` calls PARDISO's `free_memory('all')` to release Intel MKL internal factorization memory
  - `MUMPSSolver.free()` explicitly drops the MUMPS context, triggering immediate deallocation of Fortran-allocated symbolic and numeric factorization memory
  - `SuperLUSolver.free()` releases the SciPy SuperLU factorization object
  - All three eigensolve call sites (`wgmodes`, `wgmodes_yee`, `svmodes`) now call `sparse_solver.free()` and `del OPinv, sparse_solver` immediately after `eigs()` returns

## [3.0.1] - 2026-02-28

### Added
- **`gsm_step()` function** (`postprocess/scattering.py`): Compute generalized scattering matrices at waveguide step discontinuities using the mode-matching technique
  - Calculates S11, S12, S21, S22 matrices relating incident and scattered mode amplitudes
  - Supports both uniform grids (scalar dx, dy) and non-uniform grids (array dx, dy)
  - `normalize=True` option returns power-normalized S-matrix where |S_mn|² gives direct power coupling efficiency
  - Automatically collocates fields if needed (works with output from `wgmodes`, `wgmodes_yee`, or pre-collocated fields)
  - Validates power conservation (S†S = I for lossless waveguides)
  - Based on: G. V. Eleftheriades et al., "Some Important Properties of Waveguide Junction Generalized Scattering Matrices," IEEE Trans. MTT, vol. 42, no. 10, 1994
- **`skip_coords` parameter for `unfold()`**: Allows skipping coordinate unfolding when making multiple calls
  - Useful for edge cases where separate field and eps unfolding calls are needed
  - Second call can use `skip_coords=True` to avoid double-unfolding coordinates
- **New example notebook** (`examples/waveguide_junction.ipynb`): Demonstrates scattering matrix calculation at a silicon nitride waveguide width transition
- **Cubic PML stretching method** (`geometry/pml.py`): New `'C'` method for `stretchmesh()`
  - Implements a quartic coordinate map corresponding to a cubic (m=3) conductivity profile, the standard used by Lumerical and Tidy3D
  - For PML use, supply `factor = 1 + 1j*A` where `A = sigma_max / (4*omega*eps)`; typical values of `sigma_max / (omega*eps)` are 1–2, so `A` ~ 0.25–0.5

### Changed
- **`unfold()` now supports combining fields and eps in a single call**:
  - Previously required two separate calls which caused errors (coordinates were unfolded twice)
  - New recommended usage: `x, y, xc, yc, hx, hy, hzj, eps = unfold(x, y, xc, yc, hx, hy, hzj, boundary='000M', unfold='W', eps=eps)`
  - When both are provided, eps arrays are returned after field arrays in the tuple
  - Backward compatible: existing single-purpose calls continue to work unchanged

## [3.0.0] - 2026-01-31

### Breaking Changes
- **`wgmodes()` now returns 7 values** instead of 4:
  - Old: `neff, Hx, Hy, Hzj = wgmodes(...)`
  - New: `neff, Ex, Ey, Ezj, Hx, Hy, Hzj = wgmodes(...)`
  - E-fields are now computed internally and returned directly, eliminating the need to call `efields()` separately
- **All example notebooks updated** for the new return signature

### Added
- **Field collocation support**: Both `wgmodes()` and `wgmodes_yee()` now accept `collocate=True` parameter
  - Interpolates all field components to cell centers using linear averaging
  - Returns six fields of identical shape `(ny, nx)` or `(ny, nx, nmodes)`
  - Useful for visualization and post-processing where co-located fields are needed
- **Generic `collocate()` function** (`postprocess/collocate.py`):
  - Works with fields from either `wgmodes` or `wgmodes_yee`
  - Automatically detects field grid positions from array shapes
  - Interpolates all components to cell centers `(ny, nx)`
  - Supports both single-mode (2D) and multi-mode (3D) arrays

### Changed
- **`poynting()` rewritten** to handle both collocated and non-collocated fields:
  - Automatically detects whether input fields are already collocated
  - If not collocated, calls `collocate()` internally before computing Poynting vector
  - Works with output from `wgmodes`, `wgmodes_yee`, or pre-collocated fields
  - Fixed bugs in previous version (undefined `neff` reference, incorrect recursive call)
- **`wgmodes()` now computes E-fields internally** using the same curl relations as `efields()`
  - Saves unpadded permittivity arrays before padding for accurate E-field calculation
  - E-fields computed at cell centers, H-fields at vertices (unless `collocate=True`)

## [2.2.0] - 2026-01-30

### Added
- **Unified sparse solver interface** (`sparse_solve.py`): Automatic selection of the best available solver for shift-invert eigenvalue problems
  - PyPardiso support (fastest for real matrices, requires Intel MKL)
  - MUMPS support (efficient for both real and complex matrices)
  - Falls back to SciPy's SuperLU when optional packages are unavailable
  - New `make_shift_invert_operator()` function for optimized eigenvalue solving
- **Mode unfolding utility** (`postprocess/unfold.py`): Reconstruct full mode solutions from symmetry-reduced computations
  - Supports unfolding fields computed with symmetric, antisymmetric, PEC, and PMC boundary conditions
  - Handles scalar fields (1 component), H-fields (3 components), and all fields (6 components)
  - Automatically detects grid location (cell-centered vs node-centered) for each field component
  - Also supports unfolding permittivity arrays (isotropic, diagonal anisotropic, and full tensor)
- **Optional dependencies for accelerated solvers**: Install with `pip install modesolver[fast]` to get pypardiso and python-mumps

### Changed
- Core solvers (`wgmodes.py`, `svmodes.py`, `wgmodes_yee.py`) updated to use the new sparse solver interface

## [2.1.0] - 2026-01-14

### Added
- **Yee-mesh discretization modesolver** (`wgmodes_yee.py`): Alternative staggered-grid formulation similar to FDTD discretization
  - Returns all six field components (Ex, Ey, jEz, Hx, Hy, jHz) directly
  - Supports isotropic, diagonal anisotropic, and fully anisotropic materials
  - Same boundary condition options as the standard solver
- New example notebook demonstrating the Yee mesh formulation (`examples/yee_mesh.ipynb`)

### Changed
- Updated README to describe v2.1.0 features

## [2.0.0] - 2025-12-28

### Added
- **Initial Python release** - Complete port of the MATLAB modesolver to Python
- **Vector finite-difference modesolver** (`wgmodes.py`):
  - Computes transverse magnetic field components (Hx, Hy, jHz)
  - Supports isotropic, diagonal anisotropic, and fully anisotropic permittivity tensors
  - Flexible boundary conditions: symmetric (S), antisymmetric (A), PEC (E), PMC (M), Dirichlet (0)
  - Non-uniform grid support
  - Shift-invert mode for finding eigenmodes near specified effective indices
- **Semivectorial modesolver** (`svmodes.py`):
  - Scalar Helmholtz equation solver
  - Semivectorial Ex and Ey formulations
  - Faster alternative for weakly-guiding waveguides
- **Geometry module** (`geometry/`):
  - `waveguidemesh.py`: Rectangular mesh generation for multilayer slab waveguides with optional ridge
  - `waveguidemeshfull.py`: Extended waveguide mesh construction
  - `fibermesh.py`: Circular/fiber geometry mesh generation
  - `pml.py`: Perfectly Matched Layer functions (stretchmesh, padmesh, trimmesh)
- **Post-processing module** (`postprocess/`):
  - `efields.py`: Reconstruct electric field components (Ex, Ey, jEz) from magnetic fields
  - `poynting.py`: Compute time-averaged Poynting vector (optical intensity)
- **Comprehensive example notebooks** (11 notebooks):
  - Basic full-vector and semi-vector examples
  - Directional coupler mode analysis
  - Fiber mode computation
  - Faraday and uniaxial waveguide examples
  - Non-uniform mesh demonstrations
  - PML boundary examples for leaky modes
  - Mesh construction tutorial
  - All-fields visualization

### Fixed
- Corrected several bugs in `wgmodes.py` from initial commit
- Fixed issues in `geometry/fibermesh.py`
- Improved `geometry/pml.py` routines
- Bug fixes in `geometry/waveguidemesh.py` and `waveguidemeshfull.py`
- Corrections in `postprocess/efields.py`

## [1.x] - MATLAB Legacy

The original MATLAB implementation was developed between 2008-2011. The MATLAB source code is preserved in the `matlab/` directory for reference.

### 2011-04-22
- Fixed bug in `stretchmesh` routine: complex coordinate stretching had incorrect sign for south and west boundaries
- Corrected bug in `wgmodes` and `svmodes` where `dy` was incorrectly conjugated
- Thanks to Jiri Petracek for discovering this bug

### 2011-03-28
- Modified `contourmode` and `imagemode` plotting routines for improved usability

### 2008-07-08
- Fixed compatibility issue: replaced `transp` with `transpose` for MATLAB version compatibility

### 2008-01-10
- Corrected algebraic error in expressions for Axy(E), Axy(W), Ayx(N), and Ayx(S)
- Previous version was only correct for uniform meshes near dielectric edges
- Corrected expressions now agree with isotropic equations of Lusse et al.
- Thanks to yuchunlu_china@hotmail.com for reporting this bug

---

## References

This library implements the vector finite-difference method described in:

A. B. Fallahkhair, K. S. Li, and T. E. Murphy, "Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides," *J. Lightwave Technol.* **26**(11), 1423-1431 (2008).
