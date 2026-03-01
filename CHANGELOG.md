# Changelog

All notable changes to the modesolver library are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
