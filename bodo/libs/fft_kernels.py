# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements kernels for FFT functions.
"""

import numpy as np
from numba.extending import overload
from scipy.fftpack import fftshift

import bodo
from bodo.utils.typing import raise_bodo_error


@overload(fftshift, inline="always")
def overload_fftshift(A):
    """
    Performs the fft shift operation on the input data. This rolls each
    axis by 50%. For a 1D array, this switches the two halves of the
    array. For a 2D array, this switches quadrant 1 with quadrant 3,
    and quadrant 2 with quadrant 4. This kernel currently only works
    sequentially.

    Args:
        A (np.array): array of data to be shifted. Currently only
        2D arrays supported.

    Returns:
        (np.array) The input array shifted as desired.
    """
    if bodo.utils.utils.is_array_typ(A, False) and A.ndim == 2:  # pragma: no cover
        dtype = A.dtype

        def impl(A):  # pragma: no cover
            rows, cols = A.shape
            r_off = rows // 2
            c_off = cols // 2
            res = np.empty((rows, cols), dtype)
            for r in range(rows):
                r_write = (r + r_off) % rows
                for c in range(cols):
                    c_write = (c + c_off) % cols
                    res[r_write, c_write] = A[r, c]
            return res

        return impl
    raise_bodo_error("fftshift only currently supported on 2d arrays")
