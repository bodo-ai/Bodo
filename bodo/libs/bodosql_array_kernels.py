# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
import pandas as pd
from numba.core import types
from numba.extending import overload

import bodo
from bodo.utils.typing import (
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_bool,
    is_overload_constant_bool,
    is_overload_constant_number,
    is_overload_constant_str,
    is_overload_int,
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


broadcasted_fixed_arg_functions = {
    "cond",
    "lpad",
    "rpad",
    "last_day",
    "dayname",
    "monthname",
    "weekday",
    "yearofweekiso",
    "makedate",
    "format",
    "left",
    "right",
    "ord_ascii",
    "char",
    "repeat",
    "reverse",
    "replace",
    "space",
    "int_to_days",
    "second_timestamp",
    "day_timestamp",
    "year_timestamp",
    "month_diff",
    "conv",
    "substring",
    "substring_index",
    "nullif",
    "negate",
    "log",
    "strcmp",
    "instr",
}


def gen_vectorized(
    arg_names,
    arg_types,
    propagate_null,
    scalar_text,
    out_dtype,
    arg_string=None,
    arg_sources=None,
    array_override=None,
):
    """Creates an impl for a function that has several inputs that could all be
       scalars, nulls, or arrays by broadcasting appropriately.

    Args:
        arg_names (string list): the names of each argument
        arg_types (dtype list): the types of each argument
        propagate_null (bool list): a mask indicating which arguments produce
        an output of null when the input is null
        scalar_text (string): the func_text of the core operation done on each
        set of scalar values after all broadcasting is handled. The string should
        refer to the scalar values as arg0, arg1, arg2, etc. (where arg0
        corresponds to the current value from arg_names[0]), and store the final
        answer of the calculation in res[i]
        out_dtype (dtype): the dtype of the output array
        arg_string (optional string): the string that goes in the def line to
        describe the parameters. If not provided, is inferred from arg_names
        arg_sources (optional dict): key-value pairings describing how to
        obtain the arg_names from the arguments described in arg_string
        array_override (optional string): a string representing how to obtain
        the length of the final array. If not provided, inferred from arg_types.
        If provided, ensures that the returned answer is always an array,
        even if all of the arg_types are scalars.

    Returns:
        function: a broadcasted version of the calculation described by
        scalar_text, which can be used as an overload.

    Internal Doc explaining more about this utility:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1080492110/BodoSQL+Array+Kernel+Parametrization

    Below is an example that would vectorize the sum operation, where if either
    element is NULL the output is NULL. In this case, it is being constructed
    for when both arguments are arrays.

    arg_names = ['left', 'right']
    arg_types = [series(int64, ...), series(int64, ...)]
    propagate_null = [True, True]
    out_dtype = types.int64
    scalar_text = "res[i] = arg0 + arg1"

    This would result in an impl constructed from the following func_text:

    def impl(left, right):
        n = len(left)
        res = bodo.utils.utils.alloc_type(n, out_dtype, -1)
        left = bodo.utils.conversion.coerce_to_array(left)
        right = bodo.utils.conversion.coerce_to_array(right)
        numba.parfors.parfor.init_prange()
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(left, i):
                bodo.libs.array_kernels.setna(res, i)
                continue
            if bodo.libs.array_kernels.isna(left, i):
                bodo.libs.array_kernels.setna(res, i)
                continue
            arg0 = left[i]
            arg1 = right[i]
            res[i] = arg0 + arg1
        return return bodo.hiframes.pd_series_ext.init_series(res, bodo.hiframes.pd_index_ext.init_range_index(0, n, 1), None)

    (Where out_dtype is mapped to types.int64)
    """
    are_arrays = [bodo.utils.utils.is_array_typ(typ, True) for typ in arg_types]
    all_scalar = not any(are_arrays)
    out_null = any(
        [propagate_null[i] for i in range(len(arg_types)) if arg_types[i] == bodo.none]
    )

    # Calculate the indentation of the scalar_text so that it can be removed
    first_line = scalar_text.splitlines()[0]
    base_indentation = len(first_line) - len(first_line.lstrip())

    if arg_string is None:
        arg_string = ", ".join(arg_names)

    func_text = f"def impl({arg_string}):\n"

    # Extract each argument name from the arg_string. Currently this is used for
    # a tuple input for variadic functions, but this use case may expand in the
    # future, at which point this comment will be updated
    if arg_sources is not None:
        for argument, source in arg_sources.items():
            func_text += f"   {argument} = {source}\n"

    # If all the inputs are scalar, either output None immediately or
    # compute a single scalar computation without the loop
    if all_scalar and array_override == None:
        if out_null:
            func_text += "   return None"
        else:
            for i in range(len(arg_names)):
                func_text += f"   arg{i} = {arg_names[i]}\n"
            for line in scalar_text.splitlines():
                func_text += (
                    " " * 3
                    + line[base_indentation:]
                    # res[i] is now stored as answer, since there is no res array
                    .replace("res[i] =", "answer =")
                    # Calls to setna mean that the answer is NULL, so they are
                    # replaced with "return None".
                    .replace("bodo.libs.array_kernels.setna(res, i)", "return None")
                    # NOTE: scalar_text should not contain any isna calls in
                    # the case where all of the inputs are scalar.
                    + "\n"
                )
            func_text += "   return answer"

    else:
        # Determine the size of the final output array and convert Series to arrays
        if array_override != None:
            found_size = True
            size_text = f"len({array_override})"
        found_size = False
        for i in range(len(arg_names)):
            if are_arrays[i]:
                if not found_size:
                    size_text = f"len({arg_names[i]})"
                    found_size = True
                if not bodo.utils.utils.is_array_typ(arg_types[i], False):
                    func_text += f"   {arg_names[i]} = bodo.hiframes.pd_series_ext.get_series_data({arg_names[i]})\n"

        func_text += f"   n = {size_text}\n"
        func_text += f"   res = bodo.utils.utils.alloc_type(n, out_dtype, (-1,))\n"
        func_text += "   numba.parfors.parfor.init_prange()\n"
        func_text += "   for i in numba.parfors.parfor.internal_prange(n):\n"

        # If the argument types imply that every row is null, then just set each
        # row of the output array to null
        if out_null:
            func_text += f"      bodo.libs.array_kernels.setna(res, i)\n"

        else:
            # For each column that propagates nulls, add an isna check (and
            # also convert Series to arrays)
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    if propagate_null[i]:
                        func_text += f"      if bodo.libs.array_kernels.isna({arg_names[i]}, i):\n"
                        func_text += "         bodo.libs.array_kernels.setna(res, i)\n"
                        func_text += "         continue\n"

            # Add the local variables that the scalar computation will use
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    func_text += f"      arg{i} = {arg_names[i]}[i]\n"
                else:
                    func_text += f"      arg{i} = {arg_names[i]}\n"

            # Add the scalar computation. The text must use the argument variables
            # in the form arg0, arg1, etc. and store its final answer in res[i].
            for line in scalar_text.splitlines():
                func_text += " " * 6 + line[base_indentation:] + "\n"

        func_text += "   return bodo.hiframes.pd_series_ext.init_series(res, bodo.hiframes.pd_index_ext.init_range_index(0, n, 1), None)"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "np": np,
            "out_dtype": out_dtype,
            "pd": pd,
        },
        loc_vars,
    )

    impl = loc_vars["impl"]

    return impl


def unopt_argument(func_name, arg_names, i, container_length=None):
    """Creates an impl that cases on whether or not a certain argument to a function
       is None in order to un-optionalize that argument

    Args:
        func_name (string): the name of the function with the optional arguments
        arg_names (string list): the name of each argument to the function
        i (integer): the index of the argument from arg_names being unoptionalized
        container_length (optional int): if provided, treat the single arg_name as
        a container of this many arguments. Used so we can pass in arbitrary sized
        containers or arguments to handle SQL functions with variadic arguments,
        such as coalesce

    Returns:
        function: the impl that re-calls func_name with arg_names[i] no longer optional
    """
    if container_length != None:
        args1 = [
            f"{arg_names[0]}{[j]}" if j != i else "None"
            for j in range(container_length)
        ]
        args2 = [
            f"{arg_names[0]}{[j]}"
            if j != i
            else f"bodo.utils.indexing.unoptional({arg_names[0]}[{j}])"
            for j in range(container_length)
        ]
        func_text = f"def impl({', '.join(arg_names)}):\n"
        func_text += f"   if {arg_names[0]}[{i}] is None:\n"
        func_text += f"      return {func_name}(({', '.join(args1)}))\n"
        func_text += f"   else:\n"
        func_text += f"      return {func_name}(({', '.join(args2)}))"
    else:
        args1 = [arg_names[j] if j != i else "None" for j in range(len(arg_names))]
        args2 = [
            arg_names[j]
            if j != i
            else f"bodo.utils.indexing.unoptional({arg_names[j]})"
            for j in range(len(arg_names))
        ]
        func_text = f"def impl({', '.join(arg_names)}):\n"
        func_text += f"   if {arg_names[i]} is None:\n"
        func_text += f"      return {func_name}({', '.join(args1)})\n"
        func_text += f"   else:\n"
        func_text += f"      return {func_name}({', '.join(args2)})"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
        },
        loc_vars,
    )

    impl = loc_vars["impl"]

    return impl


def verify_int_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not an integer, integer column, or NULL
    """
    if (
        arg != types.none
        and not isinstance(arg, types.Integer)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, types.Integer)
        )
        and not is_overload_int(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be an integer, integer column, or null"
        )


def verify_int_float_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer or float
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not an integer/float, integer/float
    column, or NULL
    """
    if (
        arg != types.none
        and not isinstance(arg, (types.Integer, types.Float))
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, (types.Integer, types.Float))
        )
        and not is_overload_constant_number(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a numeric, numeric column, or null"
        )


def verify_string_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an string
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not an string, string column, or NULL
    """
    if (
        arg not in (types.none, types.unicode_type)
        and not isinstance(arg, types.StringLiteral)
        and not (
            bodo.utils.utils.is_array_typ(arg, True) and arg.dtype == types.unicode_type
        )
        and not is_overload_constant_str(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a string, string column, or null"
        )


def verify_boolean_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a boolean
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not an boolean, boolean column, or NULL
    """
    if (
        arg not in (types.none, types.boolean)
        and not (
            bodo.utils.utils.is_array_typ(arg, True) and arg.dtype == types.boolean
        )
        and not is_overload_bool(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a boolean, boolean column, or null"
        )


def verify_datetime_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a datetime
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not a datetime, datetime column, or NULL
    """
    if arg not in (
        types.none,
        bodo.datetime64ns,
        bodo.pd_timestamp_type,
        bodo.hiframes.datetime_date_ext.DatetimeDateType(),
    ) and not (
        bodo.utils.utils.is_array_typ(arg, True)
        and arg.dtype
        in (bodo.datetime64ns, bodo.hiframes.datetime_date_ext.DatetimeDateType())
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a datetime, datetime column, or null"
        )


def get_common_broadcasted_type(arg_types, func_name):
    """Takes in a list of types from arrays/Series/scalars, verifies that they
    have a common underlying scalar type, and if so returns the corresponding
    array type (+ ensures that it is nullable). Assumes scalar Nones coerce to any
    type.  In all other cases, throws an error.

    Args:
        arg_types (dtype list/tuple): the types of the arrays/Series/scalars being checked
        func_name (string): the name of the function being compiled

    Returns:
        dtype: the common underlying dtype of the inputted types. If all inputs are
            Nonetype, returns nonetype, as all inputs are scalar, and there is no need
            to find a common array type.

    raises: BodoError if the underlying types are not compatible
    """
    # Extract the underlying type of each scalar/vector
    elem_types = []
    for i in range(len(arg_types)):
        # Array
        if bodo.utils.utils.is_array_typ(arg_types[i], False):
            elem_types.append(arg_types[i])
        # Series
        elif bodo.utils.utils.is_array_typ(arg_types[i], True):
            elem_types.append(arg_types[i].data)
        # Scalar
        else:
            elem_types.append(arg_types[i])
    if len(elem_types) == 0:
        return bodo.none
    elif len(elem_types) == 1:
        if bodo.utils.utils.is_array_typ(elem_types[0]):
            return bodo.utils.typing.to_nullable_type(elem_types[0])
        elif elem_types[0] == bodo.none:
            return bodo.none
        else:
            return bodo.utils.typing.to_nullable_type(
                bodo.utils.typing.dtype_to_array_type(elem_types[0])
            )
    else:
        # Verify that the underlying scalar types are common before extracting
        # the corresponding output_dtype
        scalar_dtypes = []
        for i in range(len(arg_types)):
            if bodo.utils.utils.is_array_typ(arg_types[i]):
                scalar_dtypes.append(elem_types[i].dtype)
            # Avoid appending nonetypes to elem_types,
            # as scalar NULL coerces to any type.
            elif elem_types[i] == bodo.none:
                pass
            else:
                scalar_dtypes.append(elem_types[i])

        # All arguments were None scalars, return none
        if len(scalar_dtypes) == 0:
            return bodo.none

        common_dtype, success = bodo.utils.typing.get_common_scalar_dtype(scalar_dtypes)
        if not success:
            raise_bodo_error(
                f"Cannot call {func_name} on columns with different dtypes"
            )
        return bodo.utils.typing.to_nullable_type(
            bodo.utils.typing.dtype_to_array_type(common_dtype)
        )


@numba.generated_jit(nopython=True)
def last_day(arr):
    """Handles cases where LAST_DAY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.last_day_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return last_day_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def last_day_util(arr):
    """A dedicated kernel for the SQL function LAST_DAY which takes in a datetime
    and returns the last day from that month


    Args:
        arr (datetime array/series/scalar): the timestamp whose last day is being
        searched for

    Returns:
        datetime series/scalar: the last day(s) from the month(s)
    """

    verify_datetime_arg(arr, "LAST_DAY", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(arg0) + pd.tseries.offsets.MonthEnd(n=0, normalize=True))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def dayname(arr):
    """Handles cases where DAYNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.dayname_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return dayname_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def dayname_util(arr):
    """A dedicated kernel for the SQL function DAYNAME which takes in a datetime
    and returns the day of the week as a string


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose dayname is being
        searched for

    Returns:
        string series/scalar: the day of the week from the input timestamp(s)
    """

    verify_datetime_arg(arr, "DAYNAME", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = pd.Timestamp(arg0).day_name()"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def monthname(arr):
    """Handles cases where MONTHNAME receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.monthname_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return monthname_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def monthname_util(arr):
    """A dedicated kernel for the SQL function MONTHNAME which takes in a datetime
    and returns the name of the month


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose month name is being
        searched for

    Returns:
        string series/scalar: the month name from the input timestamp(s)
    """

    verify_datetime_arg(arr, "MONTHNAME", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = pd.Timestamp(arg0).month_name()"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def weekday(arr):
    """Handles cases where WEEKDAY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.weekday_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return weekday_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def weekday_util(arr):
    """A dedicated kernel for the SQL function WEEKDAY which takes in a datetime
    and returns the day of the week (enumerated 0-6)


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose day of the
        week is being searched for

    Returns:
        int series/scalar: the day of the week from the input timestamp(s)
    """

    verify_datetime_arg(arr, "WEEKDAY", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "dt = pd.Timestamp(arg0)\n"
    scalar_text += "res[i] = bodo.hiframes.pd_timestamp_ext.get_day_of_week(dt.year, dt.month, dt.day)"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def yearofweekiso(arr):
    """Handles cases where YEAROFWEEKISO receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.yearofweekiso_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return yearofweekiso_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def yearofweekiso_util(arr):
    """A dedicated kernel for the SQL function YEAROFWEEKISO which takes in a datetime
    (or column) and returns the year of the input date(s)


    Args:
        arr (datetime array/series/scalar): the timestamp(s) whose year is being
        searched for

    Returns:
        int series/scalar: the year from the input timestamp(s)
    """

    verify_datetime_arg(arr, "YEAROFWEEKISO", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "dt = pd.Timestamp(arg0)\n"
    scalar_text += "res[i] = dt.isocalendar()[0]"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def makedate(year, day):
    """Handles cases where MAKEDATE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [year, day]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.makedate", ["year", "day"], i
            )

    def impl(year, day):  # pragma: no cover
        return makedate_util(year, day)

    return impl


@numba.generated_jit(nopython=True)
def makedate_util(year, day):
    """A dedicated kernel for the SQL function MAKEDATE which takes in two integers
    (or columns) and uses them to construct a specific date


    Args:
        year (int array/series/scalar): the year(s) of the timestamp
        day (int array/series/scalar): the day(s) of the year of the timestamp

    Returns:
        datetime series/scalar: the constructed date(s)
    """
    verify_int_arg(year, "MAKEDATE", "year")
    verify_int_arg(day, "MAKEDATE", "day")

    arg_names = ["year", "day"]
    arg_types = [year, day]
    propagate_null = [True] * 2
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(year=arg0, month=1, day=1) + pd.Timedelta(days=arg1-1))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


def lpad(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


def rpad(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


def lpad_util(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


def rpad_util(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(lpad)
def overload_lpad(arr, length, padstr):
    """Handles cases where LPAD recieves optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, length, padstr]
    for i in range(3):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.lpad", ["arr", "length", "padstr"], i
            )

    def impl(arr, length, padstr):  # pragma: no cover
        return lpad_util(arr, length, padstr)

    return impl


@overload(rpad)
def overload_rpad(arr, length, padstr):
    """Handles cases where RPAD recieves optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, length, padstr]
    for i in range(3):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.rpad", ["arr", "length", "padstr"], i
            )

    def impl(arr, length, padstr):  # pragma: no cover
        return rpad_util(arr, length, padstr)

    return impl


def create_lpad_rpad_util_overload(func_name):  # pragma: no cover
    """Creates an overload function to support the LPAD and RPAD functions on
       a string array representing a column of a SQL table

    Args:
        func_name: whether to create LPAD or RPAD

    Returns:
        (function): a utility that takes in 3 arguments (arr, length, pad_string)
        and returns LPAD/RPAD of all of the three arguments, where any of the
        arguments could be arrays/scalars/nulls.
    """

    def overload_lpad_rpad_util(arr, length, pad_string):
        verify_string_arg(arr, func_name, "arr")
        verify_int_arg(length, func_name, "length")
        verify_string_arg(pad_string, func_name, f"{func_name.lower()}_string")

        if func_name == "LPAD":
            pad_line = f"(arg2 * quotient) + arg2[:remainder] + arg0"
        elif func_name == "RPAD":
            pad_line = f"arg0 + (arg2 * quotient) + arg2[:remainder]"

        arg_names = ["arr", "length", "pad_string"]
        arg_types = [arr, length, pad_string]
        propagate_null = [True] * 3
        scalar_text = f"""\
            if arg1 <= 0:
                res[i] =  ''
            elif len(arg2) == 0:
                res[i] = arg0
            elif len(arg0) >= arg1:
                res[i] = arg0[:arg1]
            else:
                quotient = (arg1 - len(arg0)) // len(arg2)
                remainder = (arg1 - len(arg0)) % len(arg2)
                res[i] = {pad_line}"""
        out_dtype = bodo.string_array_type

        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return overload_lpad_rpad_util


def _install_lpad_rpad_overload():
    """Creates and installs the overloads for lpad_util and rpad_util"""
    for func, func_name in zip((lpad_util, rpad_util), ("LPAD", "RPAD")):
        overload_impl = create_lpad_rpad_util_overload(func_name)
        overload(func)(overload_impl)


_install_lpad_rpad_overload()


@numba.generated_jit(nopython=True)
def cond(arr, ifbranch, elsebranch):
    """Handles cases where IF receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, ifbranch, elsebranch]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.cond",
                ["arr", "ifbranch", "elsebranch"],
                i,
            )

    def impl(arr, ifbranch, elsebranch):  # pragma: no cover
        return cond_util(arr, ifbranch, elsebranch)

    return impl


@numba.generated_jit(nopython=True)
def cond_util(arr, ifbranch, elsebranch):
    """A dedicated kernel for the SQL function IF which takes in 3 values:
    a boolean (or boolean column) and two values (or columns) with the same
    type and returns the first or second value depending on whether the boolean
    is true or false


    Args:
        arr (boolean array/series/scalar): the T/F values
        ifbranch (any array/series/scalar): the value(s) to return when true
        elsebranch (any array/series/scalar): the value(s) to return when false

    Returns:
        int series/scalar: the difference in months between the two dates
    """

    verify_boolean_arg(arr, "cond", "arr")

    # Both branches cannot be scalar nulls if the output is an array
    # (causes a typing ambiguity)
    if (
        bodo.utils.utils.is_array_typ(arr, True)
        and ifbranch == bodo.none
        and elsebranch == bodo.none
    ):
        raise_bodo_error("Both branches of IF() cannot be scalar NULL")

    arg_names = ["arr", "ifbranch", "elsebranch"]
    arg_types = [arr, ifbranch, elsebranch]
    propagate_null = [False] * 3
    # If the conditional is an array, add a null check (null = False)
    if bodo.utils.utils.is_array_typ(arr, True):
        scalar_text = "if (not bodo.libs.array_kernels.isna(arr, i)) and arg0:\n"
    # If the conditional is a non-null scalar, case on its truthiness
    elif arr != bodo.none:
        scalar_text = "if arg0:\n"
    # Skip the ifbranch if the conditional is a scalar None (since we know that
    # the condition is always false)
    else:
        scalar_text = ""
    if arr != bodo.none:
        # If the ifbranch is an array, add a null check
        if bodo.utils.utils.is_array_typ(ifbranch, True):
            scalar_text += "   if bodo.libs.array_kernels.isna(ifbranch, i):\n"
            scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "   else:\n"
            scalar_text += "      res[i] = arg1\n"
        # If the ifbranch is a scalar null, just set to null
        elif ifbranch == bodo.none:
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
        # If the ifbranch is a non-null scalar, then no null check is required
        else:
            scalar_text += "   res[i] = arg1\n"
        scalar_text += "else:\n"
    # If the elsebranch is an array, add a null check
    if bodo.utils.utils.is_array_typ(elsebranch, True):
        scalar_text += "   if bodo.libs.array_kernels.isna(elsebranch, i):\n"
        scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
        scalar_text += "   else:\n"
        scalar_text += "      res[i] = arg2\n"
    # If the elsebranch is a scalar null, just set to null
    elif elsebranch == bodo.none:
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
    # If the elsebranch is a non-null scalar, then no null check is required
    else:
        scalar_text += "   res[i] = arg2\n"

    # Get the common dtype from the two branches
    out_dtype = get_common_broadcasted_type([ifbranch, elsebranch], "IF")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def substring(arr, start, length):
    """Handles cases where SUBSTRING recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, start, length]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.substring",
                ["arr", "start", "length"],
                i,
            )

    def impl(arr, start, length):  # pragma: no cover
        return substring_util(arr, start, length)

    return impl


@numba.generated_jit(nopython=True)
def substring_util(arr, start, length):
    """A dedicated kernel for the SQL function SUBSTRING which takes in a string,
       (or string column), and two integers (or integer columns) and returns
       the string starting from the index of the first integer, with a length
       corresponding to the second integer.


    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        start (integer array/series/scalar): the starting location(s) of the substring(s)
        length (integer array/series/scalar): the length(s) of the substring(s)

    Returns:
        string series/scalar: the string/column of extracted substrings
    """

    verify_string_arg(arr, "SUBSTRING", "arr")
    verify_int_arg(start, "SUBSTRING", "start")
    verify_int_arg(length, "SUBSTRING", "length")

    arg_names = ["arr", "start", "length"]
    arg_types = [arr, start, length]
    propagate_null = [True] * 3
    scalar_text = "if arg2 <= 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "elif arg1 < 0 and arg1 + arg2 >= 0:\n"
    scalar_text += "   res[i] = arg0[arg1:]\n"
    scalar_text += "else:\n"
    scalar_text += "   if arg1 > 0: arg1 -= 1\n"
    scalar_text += "   res[i] = arg0[arg1:arg1+arg2]\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def substring_index(arr, delimiter, occurrences):
    """Handles cases where SUBSTRING_INDEX recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, delimiter, occurrences]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.substring_index",
                ["arr", "delimiter", "occurrences"],
                i,
            )

    def impl(arr, delimiter, occurrences):  # pragma: no cover
        return substring_index_util(arr, delimiter, occurrences)

    return impl


@numba.generated_jit(nopython=True)
def substring_index_util(arr, delimiter, occurrences):
    """A dedicated kernel for the SQL function SUBSTRING_INDEX which takes in a
       string, (or string column), a delimiter string (or string column) and an
       occurrences integer (or integer column) and returns the prefix of the
       first string before that number of occurences of the delimiter


    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        delimiter (string array/series/scalar): the delimiter(s) to look for
        occurences (integer array/series/scalar): how many of the delimiter to look for

    Returns:
        string series/scalar: the string/column of prefixes before ocurrences
        many of the delimiter string occur
    """

    verify_string_arg(arr, "SUBSTRING_INDEX", "arr")
    verify_string_arg(delimiter, "SUBSTRING_INDEX", "delimiter")
    verify_int_arg(occurrences, "SUBSTRING_INDEX", "occurrences")

    arg_names = ["arr", "delimiter", "occurrences"]
    arg_types = [arr, delimiter, occurrences]
    propagate_null = [True] * 3
    scalar_text = "if arg1 == '' or arg2 == 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "elif arg2 >= 0:\n"
    scalar_text += "   res[i] = arg1.join(arg0.split(arg1, arg2+1)[:arg2])\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg1.join(arg0.split(arg1)[arg2:])\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


def coalesce(A):  # pragma: no cover
    # Dummy function used for overload
    return


def coalesce_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(coalesce)
def overload_coalesce(A):
    """Handles cases where COALESCE recieves optional arguments and forwards
    to the appropriate version of the real implementation"""
    if not isinstance(A, (types.Tuple, types.UniTuple)):
        raise_bodo_error("Coalesce argument must be a tuple")
    for i in range(len(A)):
        if isinstance(A[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.coalesce",
                ["A"],
                i,
                container_length=len(A),
            )

    def impl(A):  # pragma: no cover
        return coalesce_util(A)

    return impl


@overload(coalesce_util, no_unliteral=True)
def overload_coalesce_util(A):
    """A dedicated kernel for the SQL function COALESCE which takes in array of
       1+ columns/scalars and returns the first value from each row that is
       not NULL.

    Args:
        A (any array/scalar tuple): the array of values that are coalesced
        into a single column by choosing the first non-NULL value

    Raises:
        BodoError: if there are 0 columns, or the types don't match

    Returns:
        pd.Series: a Series containing the coalesce values of the input array
    """
    if len(A) == 0:
        raise_bodo_error("Cannot coalesce 0 columns")

    # Figure out which columns can be ignored (NULLS or after a scalar)
    array_override = None
    dead_cols = []
    for i in range(len(A)):
        if A[i] == bodo.none:
            dead_cols.append(i)
        elif not bodo.utils.utils.is_array_typ(A[i]):
            for j in range(i + 1, len(A)):
                dead_cols.append(j)
                if bodo.utils.utils.is_array_typ(A[j]):
                    array_override = f"A[{j}]"
            break

    arg_names = [f"A{i}" for i in range(len(A)) if i not in dead_cols]
    arg_types = [A[i] for i in range(len(A)) if i not in dead_cols]
    propagate_null = [False] * (len(A) - len(dead_cols))
    scalar_text = ""
    first = True
    found_scalar = False
    dead_offset = 0

    for i in range(len(A)):

        # If A[i] is NULL or comes after a scalar, it can be skipped
        if i in dead_cols:
            dead_offset += 1
            continue

        # If A[i] is an array, its value is the answer if it is not NULL
        elif bodo.utils.utils.is_array_typ(A[i]):
            cond = "if" if first else "elif"
            scalar_text += f"{cond} not bodo.libs.array_kernels.isna(A{i}, i):\n"
            scalar_text += f"   res[i] = arg{i-dead_offset}\n"
            first = False

        # If A[i] is a non-NULL scalar, then it is the answer and stop searching
        else:
            assert (
                not found_scalar
            ), "should not encounter more than one scalar due to dead column pruning"
            if first:
                scalar_text += f"res[i] = arg{i-dead_offset}\n"
            else:
                scalar_text += "else:\n"
                scalar_text += f"   res[i] = arg{i-dead_offset}\n"
            found_scalar = True
            break

    # If no other conditions were entered, and we did not encounter a scalar,
    # set to NULL
    if not found_scalar:
        if not first:
            scalar_text += "else:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)"
        else:
            scalar_text += "bodo.libs.array_kernels.setna(res, i)"

    # Create the mapping from each local variable to the corresponding element in the array
    # of columns/scalars
    arg_string = "A"
    arg_sources = {f"A{i}": f"A[{i}]" for i in range(len(A)) if i not in dead_cols}

    out_dtype = get_common_broadcasted_type(arg_types, "COALESCE")

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        arg_string,
        arg_sources,
        array_override,
    )


def left(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


def right(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


def left_util(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


def right_util(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(left)
def overload_left(arr, n_chars):
    """Handles cases where LEFT recieves optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, n_chars]
    for i in range(2):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.left", ["arr", "n_chars"], i
            )

    def impl(arr, n_chars):  # pragma: no cover
        return left_util(arr, n_chars)

    return impl


@overload(right)
def overload_right(arr, n_chars):
    """Handles cases where RIGHT recieves optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, n_chars]
    for i in range(2):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.right", ["arr", "n_chars"], i
            )

    def impl(arr, n_chars):  # pragma: no cover
        return right_util(arr, n_chars)

    return impl


def create_left_right_util_overload(func_name):  # pragma: no cover
    """Creates an overload function to support the LEFT and RIGHT functions on
       a string array representing a column of a SQL table

    Args:
        func_name: whether to create LEFT or RIGHT

    Returns:
        (function): a utility that takes in 2 arguments (arr, n-chars)
        and returns LEFT/RIGHT of all of the two arguments, where either of the
        arguments could be arrays/scalars/nulls.
    """

    def overload_left_right_util(arr, n_chars):
        verify_string_arg(arr, func_name, "arr")
        verify_int_arg(n_chars, func_name, "n_chars")

        arg_names = ["arr", "n_chars"]
        arg_types = [arr, n_chars]
        propagate_null = [True] * 2
        scalar_text = "if arg1 <= 0:\n"
        scalar_text += "   res[i] = ''\n"
        scalar_text += "else:\n"
        if func_name == "LEFT":
            scalar_text += "   res[i] = arg0[:arg1]"
        elif func_name == "RIGHT":
            scalar_text += "   res[i] = arg0[-arg1:]"

        out_dtype = bodo.string_array_type

        return gen_vectorized(
            arg_names, arg_types, propagate_null, scalar_text, out_dtype
        )

    return overload_left_right_util


def _install_left_right_overload():
    """Creates and installs the overloads for left_util and right_util"""
    for func, func_name in zip((left_util, right_util), ("LEFT", "RIGHT")):
        overload_impl = create_left_right_util_overload(func_name)
        overload(func)(overload_impl)


_install_left_right_overload()


@numba.generated_jit(nopython=True)
def repeat(arr, repeats):
    """Handles cases where REPEEAT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, repeats]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.repeat", ["arr", "repeats"], i
            )

    def impl(arr, repeats):  # pragma: no cover
        return repeat_util(arr, repeats)

    return impl


@numba.generated_jit(nopython=True)
def repeat_util(arr, repeats):
    """A dedicated kernel for the SQL function REPEAT which takes in a string
       and integer (either of which can be a scalar or vector) and
       concatenates the string to itself repeatedly according to the integer


    Args:
        arr (string array/series/scalar): the string(s) being repeated
        repeats (integer array/series/scalar): the number(s) of repeats

    Returns:
        string series/scalar: the repeated string(s)
    """
    verify_string_arg(arr, "REPEAT", "arr")
    verify_int_arg(repeats, "REPEAT", "repeats")

    arg_names = ["arr", "repeats"]
    arg_types = [arr, repeats]
    propagate_null = [True] * 2
    scalar_text = "if arg1 <= 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg0 * arg1"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def space(n_chars):
    """Handles cases where SPACE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(n_chars, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.space_util", ["n_chars"], 0
        )

    def impl(n_chars):  # pragma: no cover
        return space_util(n_chars)

    return impl


@numba.generated_jit(nopython=True)
def space_util(n_chars):
    """A dedicated kernel for the SQL function SPACE which takes in an integer
       (or integer column) and returns that many spaces


    Args:
        n_chars (integer array/series/scalar): the number(s) of spaces

    Returns:
        string series/scalar: the string/column of spaces
    """

    verify_int_arg(n_chars, "SPACE", "n_chars")

    arg_names = ["n_chars"]
    arg_types = [n_chars]
    propagate_null = [True]
    scalar_text = "if arg0 <= 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = ' ' * arg0"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def reverse(arr):
    """Handles cases where REVERSE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.reverse_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return reverse_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def reverse_util(arr):
    """A dedicated kernel for the SQL function REVERSE which takes in a string
       (or string column) and reverses it


    Args:
        arr (string array/series/scalar): the strings(s) to be reversed

    Returns:
        string series/scalar: the string/column that has been reversed
    """

    verify_string_arg(arr, "REVERSE", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = arg0[::-1]"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def replace(arr, to_replace, replace_with):
    """Handles cases where REPLACE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, to_replace, replace_with]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.replace",
                ["arr", "to_replace", "replace_with"],
                i,
            )

    def impl(arr, to_replace, replace_with):  # pragma: no cover
        return replace_util(arr, to_replace, replace_with)

    return impl


@numba.generated_jit(nopython=True)
def replace_util(arr, to_replace, replace_with):
    """A dedicated kernel for the SQL function REVERSE which takes in a base string
       (or string column), a second string to locate in the base string, and a
       third string with which to replace it.


    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        to_replace (string array/series/scalar): the substring(s) to replace
        replace_with (string array/series/scalar): the string(s) that replace to_replace

    Returns:
        string series/scalar: the string/column where each ocurrence of
        to_replace has been replaced by replace_with
    """

    verify_string_arg(arr, "REPLACE", "arr")
    verify_string_arg(to_replace, "REPLACE", "to_replace")
    verify_string_arg(replace_with, "REPLACE", "replace_with")

    arg_names = ["arr", "to_replace", "replace_with"]
    arg_types = [arr, to_replace, replace_with]
    propagate_null = [True] * 3
    scalar_text = "if arg1 == '':\n"
    scalar_text += "   res[i] = arg0\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg0.replace(arg1, arg2)"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def int_to_days(arr):
    """Handles cases where int_to_days receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.int_to_days_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return int_to_days_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def second_timestamp(arr):
    """Handles cases where second_timestamp receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.second_timestamp_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return second_timestamp_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def day_timestamp(arr):
    """Handles cases where day_timestamp receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.day_timestamp_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return day_timestamp_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def month_diff(arr0, arr1):
    """Handles cases where month_diff receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.month_diff",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return month_diff_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def int_to_days_util(arr):
    """A dedicated kernel for converting an integer (or integer column) to
    interval days.


    Args:
        arr (int array/series/scalar): the number(s) to be converted to timedelta(s)

    Returns:
        timedelta series/scalar: the number/column of days
    """

    verify_int_arg(arr, "int_to_days", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = (
        "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timedelta(days=arg0))"
    )

    out_dtype = np.dtype("timedelta64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def second_timestamp_util(arr):
    """A dedicated kernel for converting an integer (or integer column) to
    a timestamp in seconds.


    Args:
        arr (int array/series/scalar): the number(s) to be converted to datetime(s)

    Returns:
        datetime series/scalar: the number/column in seconds
    """

    verify_int_arg(arr, "second_timestamp", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(arg0, unit='s'))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def day_timestamp_util(arr):
    """A dedicated kernel for converting an integer (or integer column) to
    a timestamp in days.


    Args:
        arr (int array/series/scalar): the number(s) to be converted to datetime(s)

    Returns:
        datetime series/scalar: the number/column in days
    """

    verify_int_arg(arr, "day_timestamp", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = bodo.utils.conversion.unbox_if_timestamp(pd.Timestamp(arg0, unit='D'))"

    out_dtype = np.dtype("datetime64[ns]")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def month_diff_util(arr0, arr1):
    """A dedicated kernel for obtaining the floor of the difference in months
    between two Datetimes (or columns)


    Args:
        arr0 (datetime array/series/scalar): the date(s) being subtraced from
        arr1 (datetime array/series/scalar): the date(s) being subtraced

    Returns:
        int series/scalar: the difference in months between the two dates
    """

    verify_datetime_arg(arr0, "month_diff", "arr0")
    verify_datetime_arg(arr1, "month_diff", "arr1")

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    scalar_text = "A0 = bodo.utils.conversion.box_if_dt64(arg0)\n"
    scalar_text += "A1 = bodo.utils.conversion.box_if_dt64(arg1)\n"
    scalar_text += "delta = 12 * (A0.year - A1.year) + (A0.month - A1.month)\n"
    scalar_text += "remainder = ((A0 - pd.DateOffset(months=delta)) - A1).value\n"
    scalar_text += "if delta > 0 and remainder < 0:\n"
    scalar_text += "   res[i] = -(delta - 1)\n"
    scalar_text += "elif delta < 0 and remainder > 0:\n"
    scalar_text += "   res[i] = -(delta + 1)\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = -delta"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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
def ord_ascii(arr):
    """Handles cases where ORD/ASCII receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.ord_ascii_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return ord_ascii_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def ord_ascii_util(arr):
    """A dedicated kernel for the SQL function ORD/ASCII which takes in a string
       (or string column) and returns the ord value of the first character


    Args:
        arr (string array/series/scalar): the string(s) whose ord value(s) are
        being calculated

    Returns:
        integer series/scalar: the ord value of the first character(s)
    """

    verify_string_arg(arr, "ORD", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "if len(arg0) == 0:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = ord(arg0[0])"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def char(arr):
    """Handles cases where CHAR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodo.libs.bodosql_array_kernels.char_util", ["arr"], 0)

    def impl(arr):  # pragma: no cover
        return char_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def char_util(arr):
    """A dedicated kernel for the SQL function CHAR which takes in an integer
       (or integer column) and returns the character corresponding to the
       number(s)


    Args:
        arr (integer array/series/scalar): the integers(s) whose corresponding
        string(s) are being calculated

    Returns:
        string series/scalar: the character(s) corresponding to the integer(s)
    """

    verify_int_arg(arr, "CHAR", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "if 0 <= arg0 <= 127:\n"
    scalar_text += "   res[i] = chr(arg0)\n"
    scalar_text += "else:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def format(arr, places):
    """Handles cases where FORMAT recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, places]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.format", ["arr", "places"], i
            )

    def impl(arr, places):  # pragma: no cover
        return format_util(arr, places)

    return impl


@numba.generated_jit(nopython=True)
def format_util(arr, places):
    """A dedicated kernel for the SQL function FORMAT which takes in two
    integers (or columns) and formats the former with commas at every
    thousands place, with decimal precision specified by the latter column


    Args:
        arr (integer array/series/scalar): the integer(s) to be modified formatted
        places (integer array/series/scalar): the precision of the decimal place(s)

    Returns:
        string series/scalar: the string/column of formatted numbers
    """

    verify_int_float_arg(arr, "FORMAT", "arr")
    verify_int_arg(places, "FORMAT", "places")

    arg_names = ["arr", "places"]
    arg_types = [arr, places]
    propagate_null = [True] * 2
    scalar_text = "prec = max(arg1, 0)\n"
    scalar_text += "res[i] = format(arg0, f',.{prec}f')"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def strcmp(arr0, arr1):
    """Handles cases where STRCMP receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.strcmp",
                ["arr0", "arr1"],
                i,
            )

    def impl(arr0, arr1):  # pragma: no cover
        return strcmp_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def strcmp_util(arr0, arr1):
    """A dedicated kernel for the SQL function STRCMP which takes in 2 strings
    (or string columns) and returns 1 if the first is greater than the second,
    -1 if it is less, and 0 if they are equal


    Args:
        arr0 (string array/series/scalar): the first string(s) being compared
        arr1 (string array/series/scalar): the second string(s) being compared

    Returns:
        int series/scalar: -1, 0 or 1, depending on which string is bigger
    """

    verify_string_arg(arr0, "strcmp", "arr0")
    verify_string_arg(arr1, "strcmp", "arr1")

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    propagate_null = [True] * 2
    scalar_text = "if arg0 < arg1:\n"
    scalar_text += "   res[i] = -1\n"
    scalar_text += "elif arg0 > arg1:\n"
    scalar_text += "   res[i] = 1\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = 0\n"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def instr(arr, target):
    """Handles cases where INSTR receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, target]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.instr",
                ["arr", "target"],
                i,
            )

    def impl(arr, target):  # pragma: no cover
        return instr_util(arr, target)

    return impl


@numba.generated_jit(nopython=True)
def instr_util(arr, target):
    """A dedicated kernel for the SQL function INSTR which takes in 2 strings
    (or string columns) and returns the location where the second string
    first occurs inside the first (with 1-indexing), default zero if it is
    not there.


    Args:
        arr (string array/series/scalar): the first string(s) being searched in
        target (string array/series/scalar): the second string(s) being searched for

    Returns:
        int series/scalar: the location of the first occurrence of target in arr,
        or zero if it does not occur in arr.
    """

    verify_string_arg(arr, "instr", "arr")
    verify_string_arg(target, "instr", "target")

    arg_names = ["arr", "target"]
    arg_types = [arr, target]
    propagate_null = [True] * 2
    scalar_text = "res[i] = arg0.find(arg1) + 1"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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


@numba.generated_jit(nopython=True)
def nullif(arr0, arr1):
    """Handles cases where NULLIF recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.nullif", ["arr0", "arr1"], i
            )

    def impl(arr0, arr1):  # pragma: no cover
        return nullif_util(arr0, arr1)

    return impl


@numba.generated_jit(nopython=True)
def nullif_util(arr0, arr1):
    """A dedicated kernel for the SQL function NULLIF which takes in two
    scalars (or columns), which returns NULL if the two values are equal, and
    arg0 otherwise.


    Args:
        arg0 (array/series/scalar): The 0-th argument. This value is returned if
            the two arguments are equal.
        arg1 (array/series/scalar): The 1st argument.

    Returns:
        string series/scalar: the string/column of formatted numbers
    """

    arg_names = ["arr0", "arr1"]
    arg_types = [arr0, arr1]
    # If the first argument is NULL, the output is always NULL
    propagate_null = [True, False]
    # NA check needs to come first here, otherwise the equalify check misbehaves

    if arr1 == bodo.none:
        scalar_text = "res[i] = arg0\n"
    elif bodo.utils.utils.is_array_typ(arr1, True):
        scalar_text = "if bodo.libs.array_kernels.isna(arr1, i) or arg0 != arg1:\n"
        scalar_text += "   res[i] = arg0\n"
        scalar_text += "else:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)"
    else:
        scalar_text = "if arg0 != arg1:\n"
        scalar_text += "   res[i] = arg0\n"
        scalar_text += "else:\n"
        scalar_text += "   bodo.libs.array_kernels.setna(res, i)"

    out_dtype = get_common_broadcasted_type([arr0, arr1], "NULLIF")

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)
