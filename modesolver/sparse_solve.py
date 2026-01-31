"""
Sparse linear solver interface for shift-invert eigenvalue problems.

This module provides a unified interface for sparse matrix factorization and
solve operations, automatically selecting the best available solver based on
matrix type (real or complex) and installed packages.

Solver selection priority:
    Real matrices:    PyPardiso > MUMPS > SuperLU (SciPy default)
    Complex matrices: MUMPS > SuperLU (SciPy default)

PyPardiso is the fastest for real matrices but does not support complex.
MUMPS supports both real and complex matrices efficiently.
SuperLU (via SciPy) is the fallback that's always available.
"""

import warnings
import numpy as np
from scipy.sparse import issparse, csc_matrix
from scipy.sparse.linalg import LinearOperator, splu

# Check for available solvers
_HAS_PYPARDISO = False
_HAS_MUMPS = False

try:
    import pypardiso
    _HAS_PYPARDISO = True
except ImportError:
    pass

try:
    import mumps
    _HAS_MUMPS = True
except ImportError:
    pass


def get_available_solvers():
    """Return a dict of available solver names and their availability status."""
    return {
        "pypardiso": _HAS_PYPARDISO,
        "mumps": _HAS_MUMPS,
        "superlu": True,  # Always available via SciPy
    }


class SparseSolver:
    """
    Abstract base class for sparse linear solvers.

    All solvers implement the same interface:
        - factor(A): Compute factorization of matrix A
        - solve(b): Solve A·x = b using the factorization
        - name: Human-readable solver name
    """

    name = "base"

    def factor(self, A):
        """Compute factorization of sparse matrix A."""
        raise NotImplementedError

    def solve(self, b):
        """Solve A·x = b for x, where A was previously factored."""
        raise NotImplementedError


class SuperLUSolver(SparseSolver):
    """Sparse solver using SciPy's SuperLU (always available)."""

    name = "SuperLU"

    def factor(self, A):
        """Compute LU factorization using SuperLU."""
        # SuperLU requires CSC format
        A_csc = csc_matrix(A)
        self._lu = splu(A_csc)
        return self

    def solve(self, b):
        """Solve A·x = b using the LU factorization."""
        return self._lu.solve(b)


class PyPardisoSolver(SparseSolver):
    """Sparse solver using Intel MKL PARDISO via pypardiso (real matrices only)."""

    name = "PyPardiso"

    def factor(self, A):
        """Initialize PyPardiso solver with matrix A."""
        if not _HAS_PYPARDISO:
            raise RuntimeError("PyPardiso is not installed")

        # PyPardiso requires CSR format
        A_csr = A.tocsr() if not A.format == 'csr' else A
        self._A = A_csr
        self._solver = pypardiso.PyPardisoSolver()
        # Factorization happens on first solve, but we can trigger it
        self._solver.factorize(A_csr)
        return self

    def solve(self, b):
        """Solve A·x = b using PARDISO."""
        return self._solver.solve(self._A, b)


class MUMPSSolver(SparseSolver):
    """Sparse solver using MUMPS (supports real and complex matrices)."""

    name = "MUMPS"

    def factor(self, A):
        """Compute factorization using MUMPS."""
        if not _HAS_MUMPS:
            raise RuntimeError("python-mumps is not installed")

        self._ctx = mumps.Context()
        self._ctx.set_matrix(A)
        self._ctx.analyze()
        self._ctx.factor()
        return self

    def solve(self, b):
        """Solve A·x = b using MUMPS factorization."""
        return self._ctx.solve(b)


def get_optimal_solver(A, solver=None):
    """
    Get the optimal sparse solver for matrix A.

    Parameters
    ----------
    A : sparse matrix
        The matrix to be factored and solved.
    solver : str, optional
        Force a specific solver: 'pypardiso', 'mumps', or 'superlu'.
        If None, automatically selects the best available solver.

    Returns
    -------
    SparseSolver
        An instance of the appropriate solver class.

    Notes
    -----
    Issues warnings when using a suboptimal solver due to missing packages.
    """
    is_complex = np.iscomplexobj(A.data) if issparse(A) else np.iscomplexobj(A)

    # If user explicitly requests a solver, use it
    if solver is not None:
        solver = solver.lower()
        if solver == 'pypardiso':
            if not _HAS_PYPARDISO:
                raise RuntimeError("PyPardiso requested but not installed")
            if is_complex:
                raise ValueError("PyPardiso does not support complex matrices")
            return PyPardisoSolver()
        elif solver == 'mumps':
            if not _HAS_MUMPS:
                raise RuntimeError("MUMPS requested but not installed")
            return MUMPSSolver()
        elif solver == 'superlu':
            return SuperLUSolver()
        else:
            raise ValueError(f"Unknown solver: {solver}")

    # Auto-select based on matrix type and availability
    if is_complex:
        # Complex matrix: MUMPS > SuperLU
        if _HAS_MUMPS:
            return MUMPSSolver()
        else:
            warnings.warn(
                "Using SuperLU for complex matrix solve. "
                "Install python-mumps for better performance: pip install python-mumps",
                UserWarning
            )
            return SuperLUSolver()
    else:
        # Real matrix: PyPardiso > MUMPS > SuperLU
        if _HAS_PYPARDISO:
            return PyPardisoSolver()
        elif _HAS_MUMPS:
            warnings.warn(
                "Using MUMPS for real matrix solve. "
                "Install pypardiso for better performance: pip install pypardiso",
                UserWarning
            )
            return MUMPSSolver()
        else:
            warnings.warn(
                "Using SuperLU for real matrix solve. "
                "Install pypardiso or python-mumps for better performance: "
                "pip install pypardiso (recommended) or pip install python-mumps",
                UserWarning
            )
            return SuperLUSolver()


def make_shift_invert_operator(A, sigma, solver=None):
    """
    Create a LinearOperator for shift-invert eigenvalue problems.

    This creates a LinearOperator that applies (A - sigma*I)^{-1} to vectors,
    using the optimal available sparse solver.

    Parameters
    ----------
    A : sparse matrix
        The matrix for the eigenvalue problem.
    sigma : complex
        The shift value for shift-invert mode.
    solver : str, optional
        Force a specific solver: 'pypardiso', 'mumps', or 'superlu'.
        If None, automatically selects the best available solver.

    Returns
    -------
    OPinv : LinearOperator
        A LinearOperator that applies (A - sigma*I)^{-1}.
    solver_name : str
        The name of the solver being used.

    Notes
    -----
    The returned LinearOperator is suitable for use with scipy.sparse.linalg.eigs
    as the OPinv parameter for shift-invert mode.

    Example
    -------
    >>> from scipy.sparse.linalg import eigs
    >>> OPinv, solver_name = make_shift_invert_operator(A, sigma)
    >>> vals, vecs = eigs(A, k=nmodes, sigma=sigma, OPinv=OPinv, which="LM")
    """
    n = A.shape[0]
    dtype = complex if np.iscomplexobj(A.data) else float

    # Create shifted matrix: A - sigma*I
    from scipy.sparse import eye
    A_shifted = A - sigma * eye(n, format=A.format, dtype=A.dtype)

    # Get optimal solver and factor the shifted matrix
    sparse_solver = get_optimal_solver(A_shifted, solver)
    sparse_solver.factor(A_shifted)

    # Create LinearOperator that applies the inverse
    def matvec(v):
        return sparse_solver.solve(v)

    OPinv = LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=dtype
    )

    return OPinv, sparse_solver.name
