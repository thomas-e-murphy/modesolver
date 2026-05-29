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

__version__ = "3.0.0"
__author__ = "Thomas E. Murphy"

from .wgmodes import wgmodes
from .wgmodes_yee import wgmodes_yee
from .svmodes import svmodes
from .geometry import (
    waveguidemesh, waveguidemeshfull, fibermesh, stretchmesh, padmesh, trimmesh,
    Layer, LayerStack, Region, ClippedShape, GeometryModel,
    rectangle, polygon, disk, from_shapely,
    gds_import, gds_export,
    layoutmesh,
)
from .postprocess import collocate, poynting, unfold, gsm_step, group_index, field_to_bitmap, field_to_contours

__all__ = [
    "wgmodes", "wgmodes_yee", "svmodes",
    "waveguidemesh", "waveguidemeshfull", "fibermesh",
    "stretchmesh", "padmesh", "trimmesh",
    "Layer", "LayerStack", "Region", "ClippedShape", "GeometryModel",
    "rectangle", "polygon", "disk", "from_shapely",
    "gds_import", "gds_export",
    "layoutmesh",
    "collocate", "poynting", "unfold", "gsm_step",
    "group_index", "field_to_bitmap", "field_to_contours",
]