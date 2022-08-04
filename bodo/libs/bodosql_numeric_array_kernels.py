# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements numerical array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    raise_bodo_error,
)


@numba.generated_jit(nopython=True)
def conv(arr, old_base, new_base):
    """Handles cases where CONV receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, old_base, new_base]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.conv",
                ["arr", "old_base", "new_base"],
                i,
            )

    def impl(arr, old_base, new_base):  # pragma: no cover
        return conv_util(arr, old_base, new_base)

    return impl


@numba.generated_jit(nopython=True)
def div0(arr, divisor):
    """
    Handles cases where DIV0 receives optional arguments and forwards
    to the appropriate version of the real implementaiton.
    """
    args = [arr, divisor]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.div0", ["arr", "divisor"], i
            )

    def impl(arr, divisor):  # pragma: no cover
        return div0_util(arr, divisor)

    return impl


@numba.generated_jit(nopython=True)
def log(arr, base):
    """Handles cases where LOG receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, base]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.log",
                ["arr", "base"],
                i,
            )

    def impl(arr, base):  # pragma: no cover
        return log_util(arr, base)

    return impl


@numba.generated_jit(nopython=True)
def negate(arr):
    """Handles cases where -X receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.negate_util",
            ["arr"],
            0,
        )

    def impl(arr):  # pragma: no cover
        return negate_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def conv_util(arr, old_base, new_base):
    """A dedicated kernel for the CONV function REVERSE which takes in three
    integers (or integer columns) and converts the first column from the base
    indicated in the first second column to the base indicated by the third
    column.


    Args:
        arr (string array/series/scalar): the number(s) to be re-based
        old_base (int array/series/scalar): the original numerical base(s).
        Currently only supports numbers between 2 and 36 (inclusive).
        new_base (int array/series/scalar): the new numerical base(s). Currently
        only supports 2, 8, 10 and 16.

    Returns:
        string series/scalar: the converted numbers
    """

    verify_string_arg(arr, "CONV", "arr")
    verify_int_arg(old_base, "CONV", "old_base")
    verify_int_arg(new_base, "CONV", "new_base")

    arg_names = ["arr", "old_base", "new_base"]
    arg_types = [arr, old_base, new_base]
    propagate_null = [True] * 3
    scalar_text = "old_val = int(arg0, arg1)\n"
    scalar_text += "if arg2 == 2:\n"
    scalar_text += "   res[i] = format(old_val, 'b')\n"
    scalar_text += "elif arg2 == 8:\n"
    scalar_text += "   res[i] = format(old_val, 'o')\n"
    scalar_text += "elif arg2 == 10:\n"
    scalar_text += "   res[i] = format(old_val, 'd')\n"
    scalar_text += "elif arg2 == 16:\n"
    scalar_text += "   res[i] = format(old_val, 'x')\n"
    scalar_text += "else:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def div0_util(arr, divisor):
    """
    Kernel for div0.
    """
    verify_int_float_arg(arr, "DIV0", "arr")
    verify_int_float_arg(divisor, "DIV0", "divisor")

    arg_names = ["arr", "divisor"]
    arg_types = [arr, divisor]
    propogate_null = [True] * 2
    scalar_text = "res[i] = arg0 / arg1 if arg1 else 0\n"
    out_dtype = types.float64

    return gen_vectorized(arg_names, arg_types, propogate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def log_util(arr, base):
    """A dedicated kernel for the SQL function LOG which takes in two numbers
    (or columns) and takes the log of the first one with the second as the base.


    Args:
        arr (float array/series/scalar): the number(s) whose logarithm is being taken
        target (float array/series/scalar): the base(s) of the logarithm

    Returns:
        float series/scalar: the output of the logarithm
    """

    verify_int_float_arg(arr, "log", "arr")
    verify_int_float_arg(base, "log", "base")

    arg_names = ["arr", "base"]
    arg_types = [arr, base]
    propagate_null = [True] * 2
    scalar_text = "res[i] = np.log(arg0) / np.log(arg1)"

    out_dtype = types.Array(bodo.float64, 1, "C")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def negate_util(arr):
    """A dedicated kernel for unary negation in SQL


    Args:
        arr (numeric array/series/scalar): the number(s) whose sign is being flipped

    Returns:
        numeric series/scalar: the opposite of the input array
    """

    verify_int_float_arg(arr, "negate", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]

    # Extract the underly scalar dtype, default int32
    if arr == bodo.none:
        scalar_type = types.int32
    elif bodo.utils.utils.is_array_typ(arr, False):
        scalar_type = arr.dtype
    elif bodo.utils.utils.is_array_typ(arr, True):
        scalar_type = arr.data.dtype
    else:
        scalar_type = arr

    # If the dtype is unsigned, manually upcast then make it signed before negating
    scalar_text = {
        types.uint8: "res[i] = -np.int16(arg0)",
        types.uint16: "res[i] = -np.int32(arg0)",
        types.uint32: "res[i] = -np.int64(arg0)",
    }.get(scalar_type, "res[i] = -arg0")

    # If the dtype is unsigned, make the output dtype signed
    scalar_type = {
        types.uint8: types.int16,
        types.uint16: types.int32,
        types.uint32: types.int64,
        types.uint64: types.int64,
    }.get(scalar_type, scalar_type)

    out_dtype = bodo.utils.typing.to_nullable_type(
        bodo.utils.typing.dtype_to_array_type(scalar_type)
    )
    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


def rank_sql(arr_tup, method="average", pct=False):  # pragma: no cover
    return


@overload(rank_sql, no_unliteral=True)
def overload_rank_sql(arr_tup, method="average", pct=False):  # pragma: no cover
    """
    Series.rank modified for SQL to take a tuple of arrays.
    Assumes that the arr_tup passed in is sorted as desired, thus arguments 'na_option' and 'ascending' are unnecessary.
    """
    if not is_overload_constant_str(method):
        raise_bodo_error("Series.rank(): 'method' argument must be a constant string")
    method = get_overload_const_str(method)
    if not is_overload_constant_bool(pct):
        raise_bodo_error("Series.rank(): 'pct' argument must be a constant boolean")
    pct = get_overload_const_bool(pct)
    func_text = """def impl(arr_tup, method="average", pct=False):\n"""
    if method == "first":
        func_text += "  ret = np.arange(1, n + 1, 1, np.float64)\n"
    else:
        func_text += "  obs = bodo.libs.array_kernels._rank_detect_ties(arr_tup[0])\n"
        func_text += "  for arr in arr_tup:\n"
        func_text += "    next_obs = bodo.libs.array_kernels._rank_detect_ties(arr)\n"
        # Say the sorted_arr is ['a', 'a', 'b', 'b', 'b' 'c'], then obs is [True, False, True, False, False, True]
        # i.e. True in each index if it's the first time we are seeing the element, because of this we use | rather than &
        func_text += "    obs = obs | next_obs \n"
        func_text += "  dense = obs.cumsum()\n"
        if method == "dense":
            func_text += "  ret = bodo.utils.conversion.fix_arr_dtype(\n"
            func_text += "    dense,\n"
            func_text += "    new_dtype=np.float64,\n"
            func_text += "    copy=True,\n"
            func_text += "    nan_to_str=False,\n"
            func_text += "    from_series=True,\n"
            func_text += "  )\n"
        else:
            # cumulative counts of each unique value
            func_text += (
                "  count = np.concatenate((np.nonzero(obs)[0], np.array([len(obs)])))\n"
            )
            func_text += "  count_float = bodo.utils.conversion.fix_arr_dtype(count, new_dtype=np.float64, copy=True, nan_to_str=False, from_series=True)\n"
            if method == "max":
                func_text += "  ret = count_float[dense]\n"
            elif method == "min":
                func_text += "  ret = count_float[dense - 1] + 1\n"
            else:
                # average
                func_text += (
                    "  ret = 0.5 * (count_float[dense] + count_float[dense - 1] + 1)\n"
                )
    if pct:
        if method == "dense":
            func_text += "  div_val = np.max(ret)\n"
        else:
            func_text += "  div_val = arr.size\n"
        # NOTE: numba bug in dividing related to parfors, requires manual division
        # TODO: replace with simple division when numba bug fixed
        # [Numba Issue #8147]: https://github.com/numba/numba/pull/8147
        func_text += "  for i in range(len(ret)):\n"
        func_text += "    ret[i] = ret[i] / div_val\n"
    func_text += "  return ret\n"

    loc_vars = {}
    exec(func_text, {"np": np, "pd": pd, "bodo": bodo}, loc_vars)
    return loc_vars["impl"]
