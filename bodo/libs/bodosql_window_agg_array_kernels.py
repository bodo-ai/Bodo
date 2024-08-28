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


@numba.generated_jit(nopython=True)
def null_ignoring_shift(input_arr, shift_amount, default_value):
    """Performs the the equivalent of the following operation but skipping over
    any rows that are NULL:

    input_arr.shift(shift_amount, fill_value=default_value)

     Args:
         input_arr (any array): the values that are being shifted
         shift_amount (int): the number of rows to shift by (ignoring nulls)
         default_value (optional any): the value to use by default when the
         shift amount goes out of the array

     Returns:
         input_arr (any array): an array identical to input_arr but where
         all the values are shifted by shift_amount (ignoring nulls)

     For example, consider the following input array:
         [10, NA, NA, 20, 30, 40, NA, 50, NA]

     Using .shift() on this array with a shift of 1 and a default of 0
     would return the following:
         [NA, NA, 20, 30, 40, NA, 50, NA, 0]

     Using null_ignoring_shift with the same arguments would return
     the following:
         [20, 20, 20, 30, 40, 50, 50, 0, 0]

     See the design for IGNORE NULLS here:
     https://bodo.atlassian.net/wiki/spaces/~62c43badfa577c57c3b685b2/pages/1322745956/Ignore+Nulls+in+LEAD+LAG+design
    """
    if not bodo.utils.utils.is_array_typ(input_arr, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")
    if not isinstance(shift_amount, types.Integer):  # pragma: no cover
        raise_bodo_error("Shift amount must be an integer type")

    no_default = default_value == bodo.none

    func_text = "def impl(input_arr, shift_amount, default_value):\n"
    if isinstance(input_arr, bodo.SeriesType):
        func_text += (
            "    input_arr = bodo.utils.conversion.coerce_to_array(input_arr)\n"
        )
    func_text += "    input_length = len(input_arr)\n"

    # Edge case: shifting by zero means that the output is identical to the input
    func_text += "    if (shift_amount == 0):\n"
    func_text += "        return input_arr\n"
    func_text += "    else:\n"
    func_text += "        start_index = 0\n"
    func_text += "        value_count = 0\n"
    func_text += (
        "        arr = bodo.utils.utils.alloc_type(input_length, input_arr, (-1,))\n"
    )

    # Shifting values from later in the array backward
    func_text += "        if (shift_amount < 0):\n"
    func_text += "            shift_amount = -(shift_amount)\n"
    func_text += "            end_index = 0\n"
    # Find K Valid
    func_text += "            while ((end_index < input_length) and (value_count < shift_amount)):\n"
    func_text += (
        "                if not(bodo.libs.array_kernels.isna(input_arr, end_index)):\n"
    )
    func_text += "                    value_count += 1\n"
    func_text += "                if (value_count < shift_amount):\n"
    func_text += "                    end_index += 1\n"
    # Iterate Forward
    func_text += "            while (end_index < input_length):\n"
    func_text += "                if not(bodo.libs.array_kernels.isna(input_arr, start_index)):\n"
    # If the next shifted value is required, hunt for the next non-null value after end_index
    func_text += "                    end_index += 1\n"
    func_text += "                    while (bodo.libs.array_kernels.isna(input_arr, end_index) and (end_index < input_length)):\n"
    func_text += "                        end_index += 1\n"
    func_text += "                    if (end_index >= input_length):\n"
    func_text += "                        break\n"
    func_text += "                arr[start_index] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(input_arr[end_index])\n"
    func_text += "                start_index += 1\n"
    # Fill NAs/Defaults
    func_text += "            for idx_var in range(start_index, input_length):\n"
    if no_default:
        func_text += "                bodo.libs.array_kernels.setna(arr, idx_var)\n"
    else:
        func_text += "                arr[idx_var] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(default_value)\n"

    # Shifting values from earlier in the array forward
    func_text += "        else:\n"
    func_text += "            end_index = -1\n"
    # Find K Valid & Fill NAs/Defaults
    func_text += "            while ((start_index < input_length) and (value_count < shift_amount)):\n"
    func_text += "                if not(bodo.libs.array_kernels.isna(input_arr, start_index)):\n"
    func_text += "                    value_count += 1\n"
    func_text += "                    if (end_index == -1):\n"
    func_text += "                        end_index = start_index\n"
    if no_default:
        func_text += "                bodo.libs.array_kernels.setna(arr, start_index)\n"
    else:
        func_text += "                arr[start_index] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(default_value)\n"
    func_text += "                start_index += 1\n"
    # Iterate Forward
    func_text += "            for idx_var in range(start_index, input_length):\n"
    func_text += "                arr[idx_var] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(input_arr[end_index])\n"
    func_text += (
        "                if not(bodo.libs.array_kernels.isna(input_arr, idx_var)):\n"
    )
    # If the next shifted value is required, hunt for the next non-null value after end_index
    func_text += "                    end_index += 1\n"
    func_text += "                    while bodo.libs.array_kernels.isna(input_arr, end_index):\n"
    func_text += "                        end_index += 1\n"
    func_text += "        return arr\n"

    loc_vars = {}
    exec(func_text, {"np": np, "pd": pd, "bodo": bodo}, loc_vars)
    return loc_vars["impl"]


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
    """Takes in a Series (or array) and outputs a counter that starts at zero
    and increases by one each time the input data changes (nulls do not count
    as new or changed values)

    Args:
        S (any Series/array): the values whose changes are being noted

    Returns:
        integer array: a counter that starts at zero and increases by 1 each
        time the values of the input change
    """

    def impl(S):  # pragma: no cover
        data = bodo.utils.conversion.coerce_to_array(S)
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
        return result

    return impl


@numba.generated_jit(nopython=True)
def windowed_sum(S, lower_bound, upper_bound):
    verify_numeric_arg(S, "windowed_sum", S)
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = total"

    if isinstance(S.dtype, bodo.Decimal128Type):
        prec = bodo.libs.decimal_arr_ext.DECIMAL128_MAX_PRECISION
        scale = S.dtype.scale
        out_dtype = bodo.DecimalArrayType(prec, scale)
        propagate_nan = False

        constant_block = (
            "constant_value = bodo.libs.decimal_arr_ext.sum_decimal_array(arr0)"
        )

        setup_block = "total = zero_decimal_val"

        enter_block = "total = bodo.libs.decimal_arr_ext.add_or_subtract_decimal_scalars(total, elem0, True)"

        exit_block = "total = bodo.libs.decimal_arr_ext.add_or_subtract_decimal_scalars(total, elem0, False)"

        extra_globals = {"zero_decimal_val": pa.scalar(0, pa.decimal128(38, scale))}

    else:
        if isinstance(S.dtype, types.Integer):
            out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int64)
            propagate_nan = False
        else:
            out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)
            propagate_nan = True

        constant_block = "constant_value = S.sum()"

        setup_block = "total = 0"

        enter_block = "total += elem0"

        exit_block = "total -= elem0"

        extra_globals = {}

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        propagate_nan=propagate_nan,
        extra_globals=extra_globals,
    )


def windowed_boolor(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def windowed_booland(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def windowed_boolxor(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def make_windowed_bool_aggfunc(func, cond):
    def overload_fn(S, lower_bound, upper_bound):
        verify_int_float_arg(S, func, S)
        if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
            raise_bodo_error("Input must be an array type")

        calculate_block = f"res[i] = {cond}"

        constant_block = "in_window = 0\n"
        constant_block += "true_count = 0\n"
        constant_block += "for i in range(len(arr0)):\n"
        constant_block += "  if not bodo.libs.array_kernels.isna(arr0, i):\n"
        constant_block += "    in_window += 1\n"
        constant_block += "    true_count += int(bool(arr0[i]))\n"
        constant_block += f"constant_value = {cond}"

        setup_block = "true_count = 0"

        enter_block = "true_count += int(bool(elem0))"

        exit_block = "true_count -= int(bool(elem0))"

        out_dtype = bodo.boolean_array_type

        return gen_windowed(
            calculate_block,
            out_dtype,
            constant_block=constant_block,
            setup_block=setup_block,
            enter_block=enter_block,
            exit_block=exit_block,
            propagate_nan=False,
        )

    return overload_fn


def _install_windowed_bool_aggfuncs():
    overload(windowed_boolor)(
        make_windowed_bool_aggfunc("boolor_agg", "true_count > 0")
    )
    overload(windowed_booland)(
        make_windowed_bool_aggfunc("booland_agg", "true_count == in_window")
    )
    overload(windowed_boolxor)(
        make_windowed_bool_aggfunc("boolor_agg", "true_count == 1")
    )


_install_windowed_bool_aggfuncs()


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


def windowed_count_star(n, lower_bound, upper_bound):  # pragma: no cover
    pass


@overload(windowed_count_star)
def overload_windowed_count_star(n, lower_bound, upper_bound):
    # This method cannot use gen_vectorized because gen_vectorized
    # behavior causes null values to be ignored. count(*) will count
    # records with null values.
    # That also simplifies the implementation since we don't need to
    # look at a specific column. In contrast to most of the window
    # functions, this one just takes the length of the input.
    def impl(n, lower_bound, upper_bound):  # pragma: no cover
        result = bodo.libs.int_arr_ext.alloc_int_array(n, bodo.uint32)
        if upper_bound < lower_bound:
            result[:] = np.uint32(0)
            return result
        elif lower_bound <= -n + 1 and n - 1 <= upper_bound:
            result[:] = np.uint32(n)
            return result

        for i in range(n):
            current_lower_bound = min(max(0, i + lower_bound), n)
            current_upper_bound = min(max(0, i + upper_bound + 1), n)
            result[i] = current_upper_bound - current_lower_bound
        return result

    return impl


@numba.generated_jit(nopython=True)
def windowed_count_if(S, lower_bound, upper_bound):
    """Optimized implemention for the window function version of `count_if`. For every ith row in the
    input array, define a window frame, formed by the inclusive range [i+lower_bound, i+upper_bound].
    In this window frame, count the number of True values. This count is the ith element of the result.

    Args:
        S (any Series): the values whose changes are being noted
        lower_bound (int): The lower bound of the window, where 0 is the current row,
            negative values are preceding rows, and positive values are following rows.
        upper_bound (int): The upper bound of the window, with the same logic as above.

    Returns:
        array[uint32]: Array of counts for each range--ith element is the ith window frame's True count as described above.
    """
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = true_count"

    constant_block = "constant_value = S.sum()"

    empty_block = "res[i] = 0"

    # How many true values are currently in the window frame
    setup_block = "true_count = 0"

    # Cast bools to 1 or 0, adding to the count as they enter and subtracting when they exit
    enter_block = "true_count += int(elem0)"

    exit_block = "true_count -= int(elem0)"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.uint32)

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        empty_block=empty_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
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

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        constant_block=constant_block,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
    )


def windowed_var_pop(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def windowed_var_samp(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def windowed_stddev_pop(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def windowed_stddev_samp(S, lower_bound, upper_bound):  # pragma: no cover
    pass


# Algorithm based on: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data
def make_windowed_variance_stddev_function(name, method, ddof):
    def impl(S, lower_bound, upper_bound):
        verify_int_float_arg(S, name, S)
        if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
            raise_bodo_error("Input must be an array type")

        constant_block = ""
        if ddof > 0:
            constant_block += f"if S.count() <= {ddof}:\n"
            constant_block += "   constant_value = None\n"
            constant_block += "else:\n"
        constant_block += f"   constant_value = S.{method}(ddof={ddof})\n"

        # Choose a value of k that will (ideally) minimize the magnitudes of the
        # differences and squares to maximize the precision. The mean of
        # non-NaN elements is chosen as this value.
        setup_block = "k = S[~np.isnan(S)].mean()\n"
        setup_block += "e1 = 0\n"
        setup_block += "e2 = 0"

        enter_block = "e1 += elem0 - k\n"
        enter_block += "e2 += (elem0 - k) ** 2"

        exit_block = "e1 -= elem0 - k\n"
        exit_block += "e2 -= (elem0 - k) ** 2"

        if ddof == 0:
            calculate_block = f"if in_window == {ddof + 1}:\n"
            calculate_block += f"   res[i] = 0.0\n"
        else:
            calculate_block = f"if in_window <= {ddof}:\n"
            calculate_block += f"   res[i] = None\n"
        calculation = f"((e2 - (e1 ** 2) / in_window) / (in_window - {ddof}))"
        if method == "std":
            calculation += " ** 0.5"
        calculate_block += "else:\n"
        calculate_block += f"   res[i] = {calculation}"

        out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

        return gen_windowed(
            calculate_block,
            out_dtype,
            constant_block=constant_block,
            setup_block=setup_block,
            enter_block=enter_block,
            exit_block=exit_block,
        )

    return impl


def _instal_windowed_variance_stddev_fns():
    overloads = [
        (windowed_var_pop, "windowed_var_pop", "var", 0),
        (windowed_var_samp, "windowed_var_samp", "var", 1),
        (windowed_stddev_pop, "windowed_stddev_pop", "std", 0),
        (windowed_stddev_samp, "windowed_stddev_samp", "std", 1),
    ]
    for func, name, method, ddof in overloads:
        overload(func)(make_windowed_variance_stddev_function(name, method, ddof))


_instal_windowed_variance_stddev_fns()


@numba.generated_jit(nopython=True)
def windowed_median(S, lower_bound, upper_bound):
    verify_int_float_arg(S, "windowed_median", S)
    if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    calculate_block = "res[i] = np.median(vals)"

    constant_block = "constant_value = None if S.count() == 0 else S.dropna().median()"

    setup_block = "vals = np.zeros(0, dtype=np.float64)"

    enter_block = "vals = np.append(vals, elem0)"

    exit_block = "vals = np.delete(vals, np.argwhere(vals == elem0)[0])"

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

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
        out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

        calculate_block = "best_val, best_count = np.nan, nan_count\n"
        calculate_block += "for key in counts:\n"
        calculate_block += "   if counts[key] > best_count:\n"
        calculate_block += "      best_val, best_count = key, counts[key]\n"
        calculate_block += "res[i] = best_val"

        constant_block = "counts = {arr0[0]: 0}\n"
        constant_block += "nan_count = 0\n"
        constant_block += "for i in range(len(arr0)):\n"
        constant_block += "   if bodo.libs.array_kernels.isna(arr0, i):\n"
        constant_block += "      continue\n"
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

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

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

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

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

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(types.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        num_args=2,
    )


@numba.generated_jit(nopython=True)
def windowed_corr(Y, X, lower_bound, upper_bound):
    verify_int_float_arg(Y, "corr", "Y")
    verify_int_float_arg(X, "corr", "X")
    if not (
        bodo.utils.utils.is_array_typ(Y, True)
        and bodo.utils.utils.is_array_typ(X, True)
    ):  # pragma: no cover
        raise_bodo_error("Input must be an array type")

    # Uses Pearson correlation with the formula:
    # CORR(Y, X) = COVAR_POP(Y, X) / (STD_POP(Y) * STD_POP(X)), where
    # COVAR_POP(Y, X), STD_POP(Y) and STD_POP(X) are obtained using the same
    # sliding window logic as their corresponding kernels.
    # Defaults to null in invalid cases (only 1 element in the window,
    # one of the deviations is 0, etc.)
    calculate_block = "std_y = ((e2_y - (e1_y ** 2) / in_window)) ** 0.5\n"
    calculate_block += "std_x = ((e2_x - (e1_x ** 2) / in_window)) ** 0.5\n"
    # Checks to see if either deviation is zero by checking if either is very
    # close to zero while the numerator expression is also zero (which will be
    # the case if the standard deviation for either is zero), or if either
    # standard deviation is NaN.
    calculate_block += "invalid_corr = np.isnan(std_y) or np.isnan(std_x) or (min(std_y, std_x) <= 1e-6)\n"
    calculate_block += "if in_window <= 1 or invalid_corr:\n"
    calculate_block += "   bodo.libs.array_kernels.setna(res, i)\n"
    calculate_block += "else:\n"
    # Note: the numerator and denominator are both missing a "/ in_window" term
    # because they cancel out
    calculate_block += "   numerator = (total_xy - (total_x * total_y) / in_window)\n"
    calculate_block += "   res[i] = numerator / (std_y * std_x)\n"

    setup_block = "total_y = 0\n"
    setup_block += "total_x = 0\n"
    setup_block += "total_xy = 0\n"
    # The variables used for calculating STD_POP(Y) and STD_POP(X) are shifted
    # by constants to improve numerical stability
    setup_block += "k_y = round(Y[~np.isnan(Y)].mean())\n"
    setup_block += "e1_y = 0\n"
    setup_block += "e2_y = 0\n"
    setup_block += "k_x = round(X[~np.isnan(X)].mean())\n"
    setup_block += "e1_x = 0\n"
    setup_block += "e2_x = 0\n"

    enter_block = "total_y += elem0 - k_y\n"
    enter_block += "total_x += elem1 - k_x\n"
    enter_block += "total_xy += (elem0 - k_y) * (elem1 - k_x)\n"
    enter_block += "e1_y += elem0 - k_y\n"
    enter_block += "e2_y += (elem0 - k_y) ** 2\n"
    enter_block += "e1_x += elem1 - k_x\n"
    enter_block += "e2_x += (elem1 - k_x) ** 2\n"

    exit_block = "total_y -= elem0 - k_y\n"
    exit_block += "total_x -= elem1 - k_x\n"
    exit_block += "total_xy -= (elem0 - k_y) * (elem1 - k_x)\n"
    exit_block += "e1_y -= elem0 - k_y\n"
    exit_block += "e2_y -= (elem0 - k_y) ** 2\n"
    exit_block += "e1_x -= elem1 - k_x\n"
    exit_block += "e2_x -= (elem1 - k_x) ** 2\n"

    out_dtype = bodo.libs.float_arr_ext.FloatingArrayType(bodo.float64)

    return gen_windowed(
        calculate_block,
        out_dtype,
        setup_block=setup_block,
        enter_block=enter_block,
        exit_block=exit_block,
        num_args=2,
    )


def windowed_approx_percentile(data, q):  # pragma: no cover
    pass


@overload(windowed_approx_percentile)
def overload_windowed_approx_percentile(data, q):
    """
    Returns the result of calling APPROX_PERCENTILE as a window
    function on the partition represented by data with percentile q.

    Only allowed without an order/frame:
    https://docs.snowflake.com/en/sql-reference/functions-analytic

    Args:
        data (numeric series/array) the data to get the approximate
        percentile of.
        q (float) the percentile to seek.

    Returns
        (float array) The approximate value of the qth percentile
        of the data (or null if data is all null/empty) used to fill
        an array the same size as data.
    """
    arr_type = bodo.FloatingArrayType(types.float64)

    def impl(data, q):  # pragma: no cover
        data = bodo.utils.conversion.coerce_to_array(data)
        n = len(data)
        approx = bodo.libs.array_kernels.approx_percentile(data, q)
        return bodo.utils.conversion.coerce_scalar_to_array(approx, n, arr_type)

    return impl


def str_arr_max(arr):  # pragma: no cover
    return ""


def str_arr_min(arr):  # pragma: no cover
    return ""


def make_str_arr_min_max_overload(func):
    """Takes in a function name (either min or max) and returns an overload
    for taking the min/max of an array of strings (see full docstring below)"""
    cmp = "<" if func == "min" else ">"

    def overload_fn(arr):
        """Returns the minimum/maximum value in an array of strings (only designed
        to work in sequential contexts for window functions). This is necessary
        because the infrastructure for Series.min/Series.max does not support
        string or binary data.

        Args:
            arr (string/binary array): the array whose minimum / maximum is
            being sought

        Returns:
            string/binary scalar: the minimum / maximum value in the array
            (or None if there are no non-null entries)
        """
        # Parametrize the starting value and comparison operation based on
        # the dtype and whether the function is min or max
        if arr == bodo.string_array_type:
            starting_value = '""'
        elif arr == bodo.binary_array_type:
            starting_value = 'b""'
        else:
            return None
        func_text = "def impl(arr):\n"
        func_text += f"   best_str = {starting_value}\n"
        func_text += "   has_non_na = False\n"
        func_text += "   for i in range(len(arr)):\n"
        func_text += "       if not bodo.libs.array_kernels.isna(arr, i):\n"
        func_text += "          cur_str = arr[i]\n"
        func_text += "          if has_non_na:\n"
        func_text += f"             if cur_str {cmp} best_str:\n"
        func_text += "                best_str = cur_str\n"
        func_text += "          else:\n"
        func_text += "             has_non_na = True\n"
        func_text += "             best_str = cur_str\n"
        func_text += "   if has_non_na:\n"
        func_text += "      return best_str\n"
        func_text += "   return None\n"

        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "numba": numba,
                "np": np,
                "pd": pd,
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    return overload_fn


def windowed_min(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def windowed_max(S, lower_bound, upper_bound):  # pragma: no cover
    pass


def make_windowed_min_max_function(func, cmp):
    """Takes in a function name (either min or max) and a corresponding comparison
    string (either "<" or ">") and generates an overload for the windowed_min
    / windowed_max kernel."""

    def impl(S, lower_bound, upper_bound):
        """Returns the sliding minimum/maximum of an array

        Args:
            S (any array/series): vector of data whose
            lower_bound (int): how many indices before the current row does each
            window frame start
            upper_bound (int): how many indices after the current row does each
            window frame start

        Returns:
            any array: the array that contains the minimum/maximum value of
            the original array for a slice related to each index in the array,
            streaching forward/backward based on lower_bobund and upper_bound.
        """
        if not bodo.utils.utils.is_array_typ(S, True):  # pragma: no cover
            raise_bodo_error("Input must be an array type")

        setup_block = "lo = max(0, lower_bound)\n"
        setup_block += "hi = max(0, upper_bound + 1)\n"

        # Dictionary encoded arrays have a special procedure to find the
        # min/max string within each slice
        if S == bodo.dict_str_arr_type or (
            isinstance(S, bodo.SeriesType) and S.data == bodo.dict_str_arr_type
        ):
            setup_block += "dictionary = arr0._data\n"
            setup_block += "indices = arr0._indices\n"
            # asort[i] = k means that the kth index in the original dictionary
            # is the ith largest string from the dictionary
            setup_block += "asort = bodo.hiframes.series_impl.argsort(dictionary)\n"
            # rsort[i] = k means that the ith index in the original dictionary
            # is the kth largest string from the dictionary
            setup_block += "rsort = bodo.hiframes.series_impl.argsort(asort)\n"
            # remapped[i] = k means that the string at index i is the kth largest
            # string from the original dictionary
            setup_block += (
                "remapped = bodo.utils.utils.alloc_type(n, dict_index_dtype, (-1,))\n"
            )
            setup_block += "for j in range(n):\n"
            setup_block += "  if bodo.libs.array_kernels.isna(indices, j):\n"
            setup_block += "    bodo.libs.array_kernels.setna(remapped, j)\n"
            setup_block += "  else:\n"
            setup_block += "    remapped[j] = rsort[indices[j]]\n"
            # Extract the slice of remapped and find the minimum/maximum integer
            # from within that section. If when the calculation is done and
            # ordinal = k, that means the the min/max string in the slice is
            # the kth-largest string from the original dictionary, which
            # is located at index asort[k] of the dictionary
            calculate_block = "section = remapped[lo:hi]\n"
            calculate_block += "ordinal = -1\n"
            calculate_block += "for j in range(len(section)):\n"
            calculate_block += "  if not bodo.libs.array_kernels.isna(section, j):\n"
            calculate_block += f"    if ordinal == -1 or section[j] {cmp} ordinal:\n"
            calculate_block += "      ordinal = section[j]\n"
            calculate_block += "if ordinal == -1:\n"
            calculate_block += "   bodo.libs.array_kernels.setna(res, j)\n"
            calculate_block += "else:\n"
            calculate_block += "   bodo.libs.str_arr_ext.get_str_arr_item_copy(res, i, dictionary, asort[ordinal])\n"

            constant_block = f"constant_value = bodo.libs.bodosql_array_kernels.str_arr_{func}(arr0._data)"

        elif is_valid_string_arg(S) or is_valid_binary_arg(S):
            calculate_block = (
                f"res[i] = bodo.libs.bodosql_array_kernels.str_arr_{func}(arr0[lo:hi])"
            )
            constant_block = (
                f"constant_value = bodo.libs.bodosql_array_kernels.str_arr_{func}(arr0)"
            )
        else:
            calculate_block = f"res[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(S.iloc[lo:hi].{func}())"
            constant_block = f"constant_value = bodo.utils.conversion.unbox_if_tz_naive_timestamp(S.{func}())"

        enter_block = "hi = entering + 1"

        exit_block = "lo = exiting + 1"

        extra_globals = {
            "dict_index_dtype": bodo.libs.dict_arr_ext.dict_indices_arr_type
        }

        if isinstance(S, bodo.SeriesType):  # pragma: no cover
            out_dtype = S.data
        else:
            out_dtype = S

        propagate_nan = is_valid_float_arg(S)

        return gen_windowed(
            calculate_block,
            out_dtype,
            constant_block=constant_block,
            setup_block=setup_block,
            enter_block=enter_block,
            exit_block=exit_block,
            propagate_nan=propagate_nan,
            extra_globals=extra_globals,
        )

    return impl


def _install_windowed_min_max_fns():
    overload(windowed_min)(make_windowed_min_max_function("min", "<"))
    overload(windowed_max)(make_windowed_min_max_function("max", ">"))
    overload(str_arr_min)(make_str_arr_min_max_overload("min"))
    overload(str_arr_max)(make_str_arr_min_max_overload("max"))


_install_windowed_min_max_fns()


def windowed_skew(S, lower_bound, upper_bound):  # pragma: no cover
    pass


overload(windowed_skew)(
    make_slice_window_agg(
        out_dtype_fn=lambda _: bodo.float64,
        agg_func=lambda S: f"{S}.skew()",
        min_elements=3,
    )
)


def windowed_kurtosis(S, lower_bound, upper_bound):  # pragma: no cover
    pass


overload(windowed_kurtosis)(
    make_slice_window_agg(
        out_dtype_fn=lambda _: bodo.float64,
        agg_func=lambda S: f"{S}.kurtosis()",
        min_elements=4,
    )
)


def windowed_bitor_agg(S, lower_bound, upper_bound):  # pragma: no cover
    pass


overload(windowed_bitor_agg)(
    make_slice_window_agg(
        out_dtype_fn=lambda S: bit_agg_type_inference(S),
        agg_func=lambda S: f"bodo.libs.array_kernels.bitor_agg({S})",
        propagate_nan=False,
    )
)


def windowed_bitand_agg(S, lower_bound, upper_bound):  # pragma: no cover
    pass


overload(windowed_bitand_agg)(
    make_slice_window_agg(
        out_dtype_fn=lambda S: bit_agg_type_inference(S),
        agg_func=lambda S: f"bodo.libs.array_kernels.bitand_agg({S})",
        propagate_nan=False,
    )
)


def windowed_bitxor_agg(S, lower_bound, upper_bound):  # pragma: no cover
    pass


overload(windowed_bitxor_agg)(
    make_slice_window_agg(
        out_dtype_fn=lambda S: bit_agg_type_inference(S),
        agg_func=lambda S: f"bodo.libs.array_kernels.bitxor_agg({S})",
        propagate_nan=False,
    )
)


def windowed_object_agg(K, V):  # pragma: no cover
    pass


@overload(windowed_object_agg)
def overload_windowed_object_agg(K, V):
    """
    Returns the result of calling OBJECT_AGG as a window
    function on the partition represented by arrays K and V.

    Only allowed without an order/frame:
    https://docs.snowflake.com/en/sql-reference/functions-analytic

    Args:
        K (string series/array) the data that becomes the keys.
        V (any series/array) the data that becomes the values.

    Returns
        (map array) Same number of rows as K & V where each row is
        an identical map containing each pair of values from K and V,
        excluding pairs where the current row of either K or V is null.
    """
    key_type = K.data if bodo.hiframes.pd_series_ext.is_series_type else K
    val_type = V.data if bodo.hiframes.pd_series_ext.is_series_type else V
    struct_typ_tuple = (key_type, val_type)
    map_struct_names = bodo.utils.typing.ColNamesMetaType(("key", "value"))
    map_arr = bodo.MapArrayType(key_type, val_type)

    def impl(K, V):  # pragma: no cover
        # Convert series to arrays
        key_arr = bodo.utils.conversion.coerce_to_array(K)
        val_arr = bodo.utils.conversion.coerce_to_array(V)
        n = len(key_arr)

        # Figure out which key-value pairs are both non-null so a
        # struct array can be allocated with that many rows.
        pairs_to_keep = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            pairs_to_keep[i] = not (
                bodo.libs.array_kernels.isna(key_arr, i)
                or bodo.libs.array_kernels.isna(val_arr, i)
            )
        n_keep = pairs_to_keep.sum()
        struct_arr = bodo.libs.struct_arr_ext.pre_alloc_struct_array(
            n_keep, (-1,), struct_typ_tuple, ("key", "value"), None
        )

        # Copy over the elements of K and V into the struct array, skipping
        # over rows where either K or V is null.
        write_idx = 0
        for i in range(n):
            if pairs_to_keep[i]:
                struct_arr[write_idx] = bodo.libs.struct_arr_ext.init_struct_with_nulls(
                    (key_arr[i], val_arr[i]), (False, False), map_struct_names
                )
                write_idx += 1

        # Convert the struct array into a map scalar, then replicate to create the entire array
        key_data, value_data = bodo.libs.struct_arr_ext.get_data(struct_arr)
        nulls = bodo.libs.struct_arr_ext.get_null_bitmap(struct_arr)
        map_val = bodo.libs.map_arr_ext.init_map_value(key_data, value_data, nulls)
        return bodo.utils.conversion.coerce_scalar_to_array(map_val, n, map_arr)

    return impl
