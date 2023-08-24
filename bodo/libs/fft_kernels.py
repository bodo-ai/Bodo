# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements kernels for FFT functions.
"""

import numba
import numpy as np
from numba.core import types
from numba.extending import overload
from scipy.fftpack import fft2, fftshift

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


@overload(fft2)
def overload_fft2(A):
    """
    Calculates the 2D Fast Fourier Transform. Currently only
    supported sequentially and on complex128 data. Object mode
    is used as a stopgap until we can integrate a full implementation.

    Args:
        A (np.array): A 2D array of complex128 data.

    Returns:
        (np.array) The 2D FFT of the input.
    """
    if bodo.utils.utils.is_array_typ(A, False) and A.ndim == 2:
        out_dtype = types.Array(types.complex128, 2, "C")

        def impl(A):  # pragma: no cover
            with numba.objmode(res=out_dtype):
                res = np.ascontiguousarray(fft2(A)).astype("complex128", copy=False)
            return res

        return impl
    raise_bodo_error(f"fft2 currently unsupported on input of type {A}")
