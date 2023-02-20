# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements window/aggregation array kernels that are specific to BodoSQL.
Specifically, window/aggregation array kernels that do not concern window
frames.
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
        # Say the sorted_arr is ['a', 'a', 'b', 'b', 'b' 'c'], then obs is [True, False, True, False, False, True]
        # i.e. True in each index if it's the first time we are seeing the element, because of this we use | rather than &
        func_text += "  obs = bodo.libs.array_kernels._rank_detect_ties(arr_tup[0])\n"
        for i in range(1, len(arr_tup)):
            func_text += f"  obs = obs | bodo.libs.array_kernels._rank_detect_ties(arr_tup[{i}]) \n"
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
            func_text += "  div_val = len(arr_tup[0])\n"
        # NOTE: numba bug in dividing related to parfors, requires manual division
        # TODO: replace with simple division when numba bug fixed
        # [Numba Issue #8147]: https://github.com/numba/numba/pull/8147
        func_text += "  for i in range(len(ret)):\n"
        func_text += "    ret[i] = ret[i] / div_val\n"
    func_text += "  return ret\n"

    loc_vars = {}
    exec(func_text, {"np": np, "pd": pd, "bodo": bodo}, loc_vars)
    return loc_vars["impl"]


@numba.generated_jit(nopython=True)
def change_event(S):
    """Takes in a Series and outputs a counter that starts at zero and increases
    by one each time the input data changes (nulls do not count as new or
    changed values)

    Args:
        S (any Series): the values whose changes are being noted

    Returns
        integer Series: a counter that starts at zero and increases by 1 each
        time the values of the input change
    """

    def impl(S):  # pragma: no cover
        data = bodo.hiframes.pd_series_ext.get_series_data(S)
        n = len(data)
        result = bodo.utils.utils.alloc_type(n, types.uint64, -1)
        # Find the first non-null location
        starting_index = -1
        for i in range(n):
            result[i] = 0
            if not bodo.libs.array_kernels.isna(data, i):
                starting_index = i
                break
        # Loop over the remaining values and add 1 to the rolling sum each time
        # the array's value does not equal the most recent non-null value
        if starting_index != -1:
            most_recent = data[starting_index]
            for i in range(starting_index + 1, n):
                if bodo.libs.array_kernels.isna(data, i) or data[i] == most_recent:
                    result[i] = result[i - 1]
                else:
                    most_recent = data[i]
                    result[i] = result[i - 1] + 1
        return bodo.hiframes.pd_series_ext.init_series(
            result, bodo.hiframes.pd_index_ext.init_range_index(0, n, 1), None
        )

    return impl


@numba.generated_jit(nopython=True)
def windowed_sum(S, lower_bound, upper_bound):
    verify_int_float_arg(S, "windowed_sum", S)
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = total"

    constant_block = "constant_value = S.sum()"

    setup_block = "total = 0"

    enter_block = "total += elem0"

    exit_block = "total -= elem0"

    if isinstance(S.dtype, types.Integer):
        out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)
        propagate_nan = False
    else:
        out_dtype = bodo.utils.typing.get_float_arr_type(bodo.float64)
        propagate_nan = True

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        propagate_nan=propagate_nan,
    )


@numba.generated_jit(nopython=True)
def windowed_count(S, lower_bound, upper_bound):
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = in_window"

    constant_block = "constant_value = S.count()"

    empty_block = "res[i] = 0"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        empty_block=empty_block,
        propagate_nan=False,
    )


@numba.generated_jit(nopython=True)
def windowed_avg(S, lower_bound, upper_bound):
    verify_int_float_arg(S, "windowed_avg", S)
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = total / in_window"

    constant_block = "constant_value = S.mean()"

    setup_block = "total = 0"

    enter_block = "total += elem0"

    exit_block = "total -= elem0"

    out_dtype = bodo.utils.typing.get_float_arr_type(bodo.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
    )


@numba.generated_jit(nopython=True)
def windowed_median(S, lower_bound, upper_bound):
    verify_int_float_arg(S, "windowed_median", S)
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = np.median(vals)"

    constant_block = "constant_value = S.median()"

    setup_block = "vals = np.zeros(0, dtype=np.float64)"

    enter_block = "vals = np.append(vals, elem0)"

    exit_block = "vals = np.delete(vals, np.argwhere(vals == elem0)[0])"

    out_dtype = bodo.utils.typing.get_float_arr_type(types.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
    )


@numba.generated_jit(nopython=True)
def windowed_mode(S, lower_bound, upper_bound):
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")
    if isinstance(S, bodo.SeriesType):  # pragma: no cover
        out_dtype = S.data
    else:
        out_dtype = S

    # For float values, handle NaN as a special case:
    if is_valid_float_arg(S):
        out_dtype = bodo.utils.typing.get_float_arr_type(types.float64)

        calculate_block = "best_val, best_count = np.nan, nan_count\n"
        calculate_block += "for key in counts:\n"
        calculate_block += "   if counts[key] > best_count:\n"
        calculate_block += "      best_val, best_count = key, counts[key]\n"
        calculate_block += "res[i] = best_val"

        constant_block = "counts = {arr0[0]: 0}\n"
        constant_block += "nan_count = 0\n"
        constant_block += "for i in range(len(arr0)):\n"
        constant_block += "   if np.isnan(arr0[i]):\n"
        constant_block += "      nan_count += 1\n"
        constant_block += "   elif not bodo.libs.array_kernels.isna(arr0, i):\n"
        constant_block += "      counts[arr0[i]] = counts.get(arr0[i], 0) + 1\n"
        constant_block += calculate_block.replace("res[i]", "constant_value")

        setup_block = "counts = {0.0: 0}\n"
        setup_block += "nan_count = 0"

        enter_block = "if np.isnan(elem0):\n"
        enter_block += "   nan_count += 1\n"
        enter_block += "else:\n"
        enter_block += "   counts[elem0] = counts.get(elem0, 0) + 1"

        exit_block = "if np.isnan(elem0):\n"
        exit_block += "   nan_count -= 1\n"
        exit_block += "else:\n"
        exit_block += "   counts[elem0] = counts.get(elem0, 0) - 1"

    else:
        calculate_block = "best_val, best_count = None, 0\n"
        calculate_block += "for key in counts:\n"
        calculate_block += "   if counts[key] > best_count:\n"
        calculate_block += "      best_val, best_count = key, counts[key]\n"
        calculate_block += "res[i] = best_val"

        constant_block = "counts = {arr0[0]: 0}\n"
        constant_block += "for i in range(len(arr0)):\n"
        constant_block += "   if not bodo.libs.array_kernels.isna(arr0, i):\n"
        constant_block += "      counts[arr0[i]] = counts.get(arr0[i], 0) + 1\n"
        constant_block += calculate_block.replace("res[i]", "constant_value")

        setup_block = "counts = {arr0[0]: 0}"

        enter_block = "counts[elem0] = counts.get(elem0, 0) + 1"

        exit_block = "counts[elem0] = counts.get(elem0, 0) - 1"

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        propagate_nan=False,
    )


@numba.generated_jit(nopython=True)
def windowed_ratio_to_report(S, lower_bound, upper_bound):
    verify_int_float_arg(S, "ratio_to_report", S)
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "if total == 0 or bodo.libs.array_kernels.isna(arr0, i):\n"
    calculate_block += "   bodo.libs.array_kernels.setna(res, i)\n"
    calculate_block += "else:\n"
    calculate_block += "   res[i] = elem0 / total"

    setup_block = "total = 0"

    enter_block = "total += elem0"

    exit_block = "total -= elem0"

    out_dtype = bodo.utils.typing.get_float_arr_type(types.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
    )


@numba.generated_jit(nopython=True)
def windowed_covar_pop(Y, X, lower_bound, upper_bound):
    verify_int_float_arg(Y, "covar_pop", "Y")
    verify_int_float_arg(X, "covar_pop", "X")
    if not (
        bodo.utils.utils.is_array_typ(Y, True)
        and bodo.utils.utils.is_array_typ(X, True)
    ):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "if in_window == 0:\n"
    calculate_block += "   bodo.libs.array_kernels.setna(res, i)\n"
    calculate_block += "else:\n"
    calculate_block += (
        "   res[i] = (total_xy - (total_x * total_y) / in_window) / in_window"
    )

    setup_block = "total_x = 0\n"
    setup_block += "total_y = 0\n"
    setup_block += "total_xy = 0"

    enter_block = "total_y += elem0\n"
    enter_block += "total_x += elem1\n"
    enter_block += "total_xy += elem0 * elem1\n"

    exit_block = "total_y -= elem0\n"
    exit_block += "total_x -= elem1\n"
    exit_block += "total_xy -= elem0 * elem1\n"

    out_dtype = bodo.utils.typing.get_float_arr_type(types.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        num_args=2,
    )


@numba.generated_jit(nopython=True)
def windowed_covar_samp(Y, X, lower_bound, upper_bound):
    verify_int_float_arg(Y, "covar_samp", "Y")
    verify_int_float_arg(X, "covar_samp", "X")
    if not (
        bodo.utils.utils.is_array_typ(Y, True)
        and bodo.utils.utils.is_array_typ(X, True)
    ):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "if in_window <= 1:\n"
    calculate_block += "   bodo.libs.array_kernels.setna(res, i)\n"
    calculate_block += "else:\n"
    calculate_block += (
        "   res[i] = (total_xy - (total_x * total_y) / in_window) / (in_window - 1)\n"
    )

    setup_block = "total_x = 0\n"
    setup_block += "total_y = 0\n"
    setup_block += "total_xy = 0"

    enter_block = "total_y += elem0\n"
    enter_block += "total_x += elem1\n"
    enter_block += "total_xy += elem0 * elem1\n"

    exit_block = "total_y -= elem0\n"
    exit_block += "total_x -= elem1\n"
    exit_block += "total_xy -= elem0 * elem1\n"

    out_dtype = bodo.utils.typing.get_float_arr_type(types.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        num_args=2,
    )
