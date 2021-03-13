# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Checks for available configurations and sets config flags.
"""

try:
    import h5py  # noqa

    from .io import _hdf5  # noqa

    # TODO: make sure h5py/hdf5 supports parallel
except ImportError:
    _has_h5py = False
else:
    _has_h5py = True


try:
    import pyarrow  # noqa
except ImportError:
    _has_pyarrow = False
else:
    _has_pyarrow = True

try:
    import scipy  # noqa
except ImportError:
    _has_scipy = False
else:
    _has_scipy = True
