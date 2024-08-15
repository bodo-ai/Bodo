# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements kernels for FFT functions. Note that this file will only be imported
if the user has scipy installed.
"""

import numpy as np
import scipy.fft
import scipy.fftpack
from llvmlite import binding as ll
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, overload

import bodo
from bodo.ext import fft_cpp
from bodo.libs.array import array_to_info, delete_info, info_to_array
from bodo.utils.typing import raise_bodo_error

ll.add_symbol(
    "fft2_py_entry",
    fft_cpp.fft2_py_entry,
)
ll.add_symbol(
    "fftshift_py_entry",
    fft_cpp.fftshift_py_entry,
)


@intrinsic
def _fftshift(typing_context, A, shape, parallel):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.ArrayType(lir.IntType(64), 2).as_pointer(),
                lir.IntType(1),
            ],
        )
        shape_ptr = cgutils.alloca_once(builder, lir.ArrayType(lir.IntType(64), 2))
        builder.store(args[1], shape_ptr)

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="fftshift_py_entry"
        )
        ret = builder.call(fn_tp, [args[0], shape_ptr, args[2]])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        new_shape = builder.load(shape_ptr)
        return context.make_tuple(builder, sig.return_type, [ret, new_shape])

    sig = types.Tuple([bodo.libs.array.array_info_type, shape])(A, shape, parallel)
    return sig, codegen


@overload(scipy.fftpack.fftshift)
@overload(scipy.fft.fftshift)
def overload_fftshift(A, parallel=False):
    """
    Performs the fft shift operation on the input data. This rolls each
    axis by 50%. For a 1D array, this switches the two halves of the
    array. For a 2D array, this switches quadrant 1 with quadrant 3,
    and quadrant 2 with quadrant 4.

    Args:
        A (np.array): array of data to be shifted. Currently only
        2D arrays supported.

    Returns:
        (np.array) The input array shifted as desired.
    """
    if (
        bodo.utils.utils.is_array_typ(A, False)
        and A.ndim == 2
        and A.dtype in (types.complex128, types.complex64)
    ):

        def impl(A, parallel=False):  # pragma: no cover
            # If we don't copy we'll overwrite A with the output if there's no later copy
            A = A.copy()
            # array_to_info only supports 1D arrays, so we flatten the array
            flattened = np.ascontiguousarray(A).reshape(-1)
            arr_info = array_to_info(flattened)
            loaded_arr_info, new_shape = _fftshift(arr_info, A.shape, parallel)
            # info_to_array only supports 1D arrays, so we reshape the array
            # the c++ kernel may have changed the shape of the array so we need to reshape it to the new shape
            ret_arr = info_to_array(loaded_arr_info, flattened).reshape(new_shape)
            delete_info(loaded_arr_info)
            return ret_arr

        return impl
    raise_bodo_error(
        f"fftshift currently unsupported on input of type {A}, try casting to complex128"
    )


@intrinsic
def _fft2(typing_context, A, shape, parallel):
    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.ArrayType(lir.IntType(64), 2).as_pointer(),
                lir.IntType(1),
            ],
        )
        shape_ptr = cgutils.alloca_once(builder, lir.ArrayType(lir.IntType(64), 2))
        builder.store(args[1], shape_ptr)

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="fft2_py_entry"
        )
        ret = builder.call(fn_tp, [args[0], shape_ptr, args[2]])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        new_shape = builder.load(shape_ptr)
        return context.make_tuple(builder, sig.return_type, [ret, new_shape])

    sig = types.Tuple([bodo.libs.array.array_info_type, shape])(A, shape, parallel)
    return sig, codegen


@overload(scipy.fftpack.fft2)
@overload(scipy.fft.fft2)
def overload_fft2(A, parallel=False):
    """
    Calculates the 2D Fast Fourier Transform. Currently only
    supports complex64 and complex128 data.

    Args:
        A (np.array): A 2D array of complex data.
        parallel (bool): Whether to run the FFT in parallel, set by distributed pass,
        not inteded to be set by user.

    Returns:
        (np.array) The 2D FFT of the input.
    """

    # To support ints and floats we need to cast all ints and float64s to complex128
    # and float32s and float16s to complex64
    if (
        bodo.utils.utils.is_array_typ(A, False)
        and A.ndim == 2
        and A.dtype in (types.complex128, types.complex64)
    ):

        def impl(A, parallel=False):  # pragma: no cover
            # array_to_info only supports 1D arrays, so we flatten the array
            flattened = np.ascontiguousarray(A).reshape((-1,))
            arr_info = array_to_info(flattened)
            loaded_arr_info, new_shape = _fft2(arr_info, A.shape, parallel)
            # info_to_array only supports 1D arrays, so we reshape the array
            # the c++ kernel may have changed the shape of the array so we need to reshape it to the new shape
            ret_arr = info_to_array(loaded_arr_info, flattened).reshape(new_shape)
            delete_info(loaded_arr_info)
            return ret_arr

        return impl
    raise_bodo_error(
        f"fft2 currently unsupported on input of type {A}, try casting to complex64 or complex128"
    )
