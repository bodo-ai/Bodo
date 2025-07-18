""" For certain environments (pip) we vendor MPI4py, this modules detects whether we
are in the vendored case and routes to the correct package.
"""

try:
    from ._vendored_mpi4py import MPI as _MPI
except ImportError:
    from mpi4py import MPI as _MPI

MPI = _MPI
