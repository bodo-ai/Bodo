# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to ARRAY utilities
"""

import bodo
import numba
import numpy as np
import pandas as pd
import types

from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import unwrap_typeref


@numba.generated_jit(nopython=True)
def to_array(arr, dtype):
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_array", ["arr"], 0
        )

    def impl(arr, dtype):  # pragma: no cover
        return to_array_util(arr, dtype)

    return impl


@numba.generated_jit(nopython=True)
def to_array_util(arr, dtype):
    arg_names = ["arr", "dtype"]
    arg_types = [arr, dtype]
    propagate_null = [True, False]
    arr_dtype = unwrap_typeref(dtype)
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(arr_dtype)
    scalar_text = "res[i] = bodo.utils.conversion.coerce_scalar_to_array(arg0, 1, arg1, False)"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def array_to_string(arr, separator):
    """
    Handles cases where ARRAY_TO_STRING receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, separator]
    for i in range(len(args)):
        if isinstance(args[i], types.optional): # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_to_string",
                ["arr", "separator"],
                i,
            )

    def impl(arr, separator):  # pragma: no cover
        return array_to_string_util(arr, separator)

    return impl


@numba.generated_jit(nopython=True)
def array_to_string_util(arr, separator):
    """
    A dedicated kernel for the SQL function ARRAY_TO_STRING which takes in an
           array, (or arrray column) and a separator string (or string column), then
           casts the array to string and add separators
    Args:
        arr (array scalar/array item array): the arrray(s) to be cast to string
        separator (string array/series/scalar): the separator to add to the string
    Returns:
        A string scalar/array: the result string(s)
    """
    arg_names = ["arr", "separator"]
    arg_types = [arr, separator]
    propagate_null = [True, True]
    out_dtype = bodo.string_array_type
    scalar_text = "arr_str = ''\n"
    scalar_text += "if len(arg0) > 0:\n"
    scalar_text += "    arr_str = '' if bodo.libs.array_kernels.isna(arg0, 0) else str(arg0[0])\n"
    scalar_text += "    for j in bodo.prange(len(arg0) - 1):\n"
    scalar_text += "        arr_str += arg1 + ('' if bodo.libs.array_kernels.isna(arg0, j + 1) else str(arg0[j + 1]))\n"
    scalar_text += "res[i] = arr_str"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        array_is_scalar=True,
    )
