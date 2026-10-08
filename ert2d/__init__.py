"""ert2d - 2D DC resistivity inversion with Voronoi parameterization.

Python port of the MATLAB ``main_vor_v00`` code family.
"""

from ert2d.io.dat_reader import Survey, ArrayType, read_dat

__all__ = ["Survey", "ArrayType", "read_dat"]
__version__ = "0.0.1"
