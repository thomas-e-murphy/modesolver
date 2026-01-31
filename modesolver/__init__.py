"""
modesolver: finite-difference eigenmode solvers for optical waveguides.

This package currently provides:

- wgmodes: 
    Vector finite-difference modesolver for dielectric waveguides, implementing the method described in 
    A. B. Fallahkhair, K. S. Li and T. E. Murphy,
    Vector Finite Difference Modesolver for Anisotropic Dielectric Waveguides
    J. Lightwave Technol. 26(11), 1423-1431, (2008)
    https://doi.org/10.1109/JLT.2008.923643
"""

__version__ = "2.0.0"
__author__ = "Thomas E. Murphy"

from .wgmodes_matrix import wgmodes_matrix
from .wgmodes import wgmodes
from .wgmodes_yee import wgmodes_yee
from .svmodes import svmodes
from .geometry import waveguidemesh, waveguidemeshfull, fibermesh, stretchmesh, padmesh, trimmesh
from .postprocess import efields, poynting, unfold

__all__ = ["wgmodes", "wgmodes_yee", "svmodes", "waveguidemesh", "waveguidemeshfull", "fibermesh", "stretchmesh", "padmesh", "trimmesh", "efields", "poynting", "unfold"]