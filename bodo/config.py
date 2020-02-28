# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Checks for available configurations and sets config flags.
"""

try:
    from .io import _hdf5
    import h5py

    # TODO: make sure h5py/hdf5 supports parallel
except ImportError:
    _has_h5py = False
else:
    _has_h5py = True


try:
    import pyarrow
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True
