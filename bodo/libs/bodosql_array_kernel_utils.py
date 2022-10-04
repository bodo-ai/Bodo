# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Common utilities for all BodoSQL array kernels
"""

import math
import re

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from numba.core import types

import bodo
from bodo.hiframes.pd_series_ext import (
    is_datetime_date_series_typ,
    pd_timestamp_type,
)
from bodo.utils.typing import (
    is_overload_bool,
    is_overload_constant_bytes,
    is_overload_constant_number,
    is_overload_constant_str,
    is_overload_int,
    raise_bodo_error,
)


def gen_vectorized(
    arg_names,
    arg_types,
    propagate_null,
    scalar_text,
    out_dtype,
    arg_string=None,
    arg_sources=None,
    array_override=None,
    support_dict_encoding=True,
    may_cause_duplicate_dict_array_values=False,
    prefix_code=None,
    extra_globals=None,
):
    """Creates an impl for a column compute function that has several inputs
       that could all be scalars, nulls, or arrays by broadcasting appropriately.

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
        support_dict_encoding (optional boolean): if true, allows dictionary
            encoded outputs under certain conditions
        may_cause_duplicate_dict_array_values (optional boolean): Indicates that the
            given operation may cause duplicate values in the ._data field of a dictionary
            encoded output (slicing, for example). Only has effect if support_dict_encoding
            is also true.
        prefix_code (optional string): if provided, embedes the code string
            right before the loop begins.

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
        return res

    (Where out_dtype is mapped to types.int64)

    NOTE: dictionary encoded outputs operate under the following assumptions:
    - The output will only be dictionary encoded if exactly one of the inputs
        is dicitonary encoded, and the rest are all scalars, and support_dict_encoding
        is True.
    - The indices do not change, except for some of them becoming null if the
        string they refer to is also transformed into null
    - The size of the dictionary will never change, even if several of its
        values become unused, duplicates, or nulls.
    - This function cannot be inlined as inlining the dictionary allocation
        is unsafe.
    - All functions invoked in scalar_text must be deterministic (no randomness
        involved).
    - Nulls are never converted to non-null values.
    """
    are_arrays = [bodo.utils.utils.is_array_typ(typ, True) for typ in arg_types]
    all_scalar = not any(are_arrays)
    out_null = any(
        [propagate_null[i] for i in range(len(arg_types)) if arg_types[i] == bodo.none]
    )

    # The output is dictionary-encoded if exactly one of the inputs is
    # dictionary encoded, the rest are all scalars, and the output dtype
    # is a string array
    vector_args = 0
    dict_encoded_arg = -1
    for i in range(len(arg_types)):
        if bodo.utils.utils.is_array_typ(arg_types[i], False):
            vector_args += 1
            if arg_types[i] == bodo.dict_str_arr_type:
                dict_encoded_arg = i
        elif bodo.utils.utils.is_array_typ(arg_types[i], True):
            vector_args += 1
            if arg_types[i].data == bodo.dict_str_arr_type:
                dict_encoded_arg = i
    use_dict_encoding = (
        support_dict_encoding and vector_args == 1 and dict_encoded_arg >= 0
    )
    # Flushing nulls from the new dictionary array back to the new index array
    # is only required if the scalar_text contains a setna call, or if one of
    # the arguments with null propagation is a scalar NULL
    use_null_flushing = (
        use_dict_encoding
        and out_dtype == bodo.string_array_type
        and (
            any(
                arg_types[i] == bodo.none and propagate_null[i]
                for i in range(len(arg_types))
            )
            or "bodo.libs.array_kernels.setna" in scalar_text
        )
    )

    # Calculate the indentation of the scalar_text so that it can be removed
    if prefix_code is not None:
        prefix_line = prefix_code.splitlines()[0]
        prefix_indentation = len(prefix_line) - len(prefix_line.lstrip())

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

    # If prefix_text was provided, embed it before the loop (unless the output
    # is all-null)
    if prefix_code is not None and not out_null:
        for line in prefix_code.splitlines():
            func_text += " " * 3 + line[prefix_indentation:] + "\n"

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
        # Convert all Series to arrays
        for i in range(len(arg_names)):
            if bodo.hiframes.pd_series_ext.is_series_type(arg_types[i]):
                func_text += f"   {arg_names[i]} = bodo.hiframes.pd_series_ext.get_series_data({arg_names[i]})\n"

        # If an array override is provided, use it to obtain the length
        if array_override != None:
            size_text = f"len({array_override})"

        # Otherwise, determine the size of the final output array from the
        # first argument that is an array
        else:
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    size_text = f"len({arg_names[i]})"
                    break

        # If using dictionary encoding, ensure that the argument name refers
        # to the dictionary, and also extract the indices
        if use_dict_encoding:
            # In this path the output is still dictionary encoded, so the indices
            # and other attributes need to be copied
            if out_dtype == bodo.string_array_type:
                func_text += (
                    f"   indices = {arg_names[dict_encoded_arg]}._indices.copy()\n"
                )

                # In Bodo, if has _has_global_dictionary is True, we assume no duplicate values in the
                # dictionary. Therefore, if we're performing an operation that may create duplicate values,
                # we need to set the values appropriatly.
                if may_cause_duplicate_dict_array_values:
                    func_text += f"   has_global = False\n"
                else:
                    func_text += f"   has_global = {arg_names[dict_encoded_arg]}._has_global_dictionary\n"

                func_text += (
                    f"   {arg_names[i]} = {arg_names[dict_encoded_arg]}._data\n"
                )
            # In this path the output is not dictionary encoded, so the data
            # and indices are needed but no copies are required
            else:
                func_text += f"   indices = {arg_names[dict_encoded_arg]}._indices\n"
                func_text += (
                    f"   {arg_names[i]} = {arg_names[dict_encoded_arg]}._data\n"
                )

        # If dictionary encoded outputs are not being used, then the output is
        # still bodo.string_array_type, the number of loop iterations is still the
        # length of the indices, and scalar_text/propagate_null should work the
        # same because isna checks the data & indices, and scalar_text uses the
        # arguments extracted by getitem.
        func_text += f"   n = {size_text}\n"
        if use_dict_encoding:
            # add a null value at the end of the dictionary and compute the scalar
            # kernel output for null values if we are not just propagating input nulls
            # to output array.
            # See test_bool.py::test_equal_null[vector_scalar_string]
            dict_len = "n" if propagate_null[dict_encoded_arg] else "(n + 1)"
            if not propagate_null[dict_encoded_arg]:
                arr_name = arg_names[dict_encoded_arg]
                func_text += f"   {arr_name} = bodo.libs.array_kernels.concat([{arr_name}, bodo.libs.array_kernels.gen_na_array(1, {arr_name})])\n"
            # adding one extra element in dictionary for null output if necessary
            if out_dtype == bodo.string_array_type:
                func_text += f"   res = bodo.libs.str_arr_ext.pre_alloc_string_array({dict_len}, -1)\n"
            else:
                func_text += f"   res = bodo.utils.utils.alloc_type({dict_len}, out_dtype, (-1,))\n"
            func_text += f"   for i in range({dict_len}):\n"
        else:
            func_text += "   res = bodo.utils.utils.alloc_type(n, out_dtype, (-1,))\n"
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

        # If using dictionary encoding, construct the output from the
        # new dictionary + the indices
        if use_dict_encoding:
            # Flush the nulls back to the index array, if necessary
            if use_null_flushing:
                func_text += "   numba.parfors.parfor.init_prange()\n"
                func_text += (
                    "   for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
                )
                func_text += "      if not bodo.libs.array_kernels.isna(indices, i):\n"
                func_text += "         loc = indices[i]\n"
                func_text += "         if bodo.libs.array_kernels.isna(res, loc):\n"
                func_text += "            bodo.libs.array_kernels.setna(indices, i)\n"
            # If the output dtype is a string array, create the new dictionary encoded array
            if out_dtype == bodo.string_array_type:
                func_text += "   res = bodo.libs.dict_arr_ext.init_dict_arr(res, indices, has_global)\n"
            # Otherwise, use the indices to copy the values from the smaller array
            # into a larger one (flushing nulls along the way)
            else:
                func_text += "   res2 = bodo.utils.utils.alloc_type(len(indices), out_dtype, (-1,))\n"
                func_text += "   numba.parfors.parfor.init_prange()\n"
                func_text += (
                    "   for i in numba.parfors.parfor.internal_prange(len(indices)):\n"
                )
                # Copy nulls from the old index array to the output array
                if propagate_null[dict_encoded_arg]:
                    func_text += "      if bodo.libs.array_kernels.isna(indices, i):\n"
                    func_text += "         bodo.libs.array_kernels.setna(res2, i)\n"
                    func_text += "         continue\n"
                    func_text += "      loc = indices[i]\n"
                else:
                    # last dictionary value in res is null's output for the kernel
                    func_text += "      loc = n if bodo.libs.array_kernels.isna(indices, i) else indices[i]\n"
                # Copy nulls from the smaller array to the output array
                func_text += "      if bodo.libs.array_kernels.isna(res, loc):\n"
                func_text += "         bodo.libs.array_kernels.setna(res2, i)\n"
                # Copy values from the smaller array to the larger array
                func_text += "      else:\n"
                func_text += "         res2[i] = res[loc]\n"
                func_text += "   res = res2\n"

        func_text += "   return res"
    loc_vars = {}

    exec_globals = {
        "bodo": bodo,
        "math": math,
        "numba": numba,
        "re": re,
        "np": np,
        "out_dtype": out_dtype,
        "pd": pd,
    }

    if not (extra_globals is None):
        exec_globals.update(extra_globals)

    exec(
        func_text,
        exec_globals,
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


def is_valid_int_arg(arg):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked

    returns: True if the argument is an integer, False otherwise
    """
    return not (
        arg != types.none
        and not isinstance(arg, types.Integer)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, types.Integer)
        )
        and not is_overload_int(arg)
    )


def verify_int_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer
       (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not an integer, integer column, or NULL
    """
    if not is_valid_int_arg(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be an integer, integer column, or null"
        )


def verify_int_float_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is an integer float
       or boolean (scalar or vector)

    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced

    raises: BodoError if the argument is not an integer/float/bool scalar/column, or NULL
    """
    if (
        arg != types.none
        and not isinstance(arg, (types.Integer, types.Float, types.Boolean))
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and isinstance(arg.dtype, (types.Integer, types.Float, types.Boolean))
        )
        and not is_overload_constant_number(arg)
    ):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a numeric, numeric column, or null"
        )


def is_valid_string_arg(arg):  # pragma: no cover
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not a string or string column
    """
    arg = types.unliteral(arg)
    return not (
        arg not in (types.none, types.unicode_type)
        and not (
            bodo.utils.utils.is_array_typ(arg, True)
            and (arg.dtype == types.unicode_type)
        )
        and not is_overload_constant_str(arg)
    )


def is_valid_binary_arg(arg):  # pragma: no cover
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not binary data
    """
    return not (
        arg != bodo.bytes_type
        and not (
            bodo.utils.utils.is_array_typ(arg, True) and arg.dtype == bodo.bytes_type
        )
        and not is_overload_constant_bytes(arg)
        and not isinstance(arg, types.Bytes)
    )


def is_valid_datetime_or_date_arg(arg):
    """
    Args:
        arg (dtype): the dtype of the argument being checked
    returns: False if the argument is not datetime or date data

    Note: In BodoSQL, scalar date/datetime types are both timestamp,
    and the columnar date/datetime types are both .
    """

    return arg == pd_timestamp_type or (
        bodo.utils.utils.is_array_typ(arg, True)
        and (
            is_datetime_date_series_typ(arg)
            or isinstance(arg, bodo.DatetimeArrayType)
            or arg.dtype == bodo.datetime64ns
        )
    )


def verify_string_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a string
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced
    raises: BodoError if the argument is not a string, string column, or null
    """
    if not is_valid_string_arg(arg):
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a string, string column, or null"
        )


def verify_scalar_string_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is a scalar string
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced
    raises: BodoError if the argument is not a scalar string
    """
    if arg not in (types.unicode_type, bodo.none) and not isinstance(
        arg, types.StringLiteral
    ):
        raise_bodo_error(f"{f_name} {a_name} argument must be a scalar string")


def verify_binary_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is binary data
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced
    raises: BodoError if the argument is not binary data or null
    """
    if not is_valid_binary_arg(arg):
        raise_bodo_error(f"{f_name} {a_name} argument must be binary data or null")


def verify_string_binary_arg(arg, f_name, a_name):  # pragma: no cover
    """Verifies that one of the arguments to a SQL function is binary data, string, or string column
       (scalar or vector)
    Args:
        arg (dtype): the dtype of the argument being checked
        f_name (string): the name of the function being checked
        a_name (string): the name of the argument being chekced
    raises: BodoError if the argument is not binary data, string, string column or null
    returns: True if the argument is a string, False if the argument is binary data
    """
    is_string = is_valid_string_arg(arg)
    is_binary = is_valid_binary_arg(arg)

    if is_string or is_binary:
        return is_string
    else:
        raise_bodo_error(
            f"{f_name} {a_name} argument must be a binary data, string, string column, or null"
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


def vectorized_sol(args, scalar_fn, dtype, manual_coercion=False):
    """Creates a py_output for a vectorized function using its arguments and the
       a function that is applied to the scalar values

    Args:
        args (any list): a list of arguments, each of which is either a scalar
        or vector (vectors must be the same size)
        scalar_fn (function): the function that is applied to scalar values
        corresponding to each row
        dtype (dtype): the dtype of the final output array
        manual_coercion (boolean, optional): whether to manually coerce the
        non-null elements of the output array to the dtype

    Returns:
        scalar or Series: the result of applying scalar_fn to each row of the
        vectors with scalar args broadcasted (or just the scalar output if
        all of the arguments are scalar)
    """
    length = -1
    for arg in args:
        if isinstance(
            arg, (pd.core.arrays.base.ExtensionArray, pd.Series, np.ndarray, pa.Array)
        ):
            length = len(arg)
            break
    if length == -1:
        return dtype(scalar_fn(*args)) if manual_coercion else scalar_fn(*args)
    arglist = []
    for arg in args:
        if isinstance(
            arg, (pd.core.arrays.base.ExtensionArray, pd.Series, np.ndarray, pa.Array)
        ):
            arglist.append(arg)
        else:
            arglist.append([arg] * length)
    if manual_coercion:
        return pd.Series([dtype(scalar_fn(*params)) for params in zip(*arglist)])
    else:
        return pd.Series([scalar_fn(*params) for params in zip(*arglist)], dtype=dtype)


def gen_windowed(
    calculate_block,
    constant_block,
    out_dtype,
    setup_block=None,
    enter_block=None,
    exit_block=None,
    empty_block=None,
):
    """Creates an impl for a window frame function that accumulates some value
    as elements enter and exit a window that starts/ends some number of indices
    before/after each element of the array. Unbounded preceding/following
    can be implemented by providing lower/upper bounds that are larger than
    the size of the input.

    Note: the implementations are only designed to work sequentially, BodoSQL
    window functions currently only work on partitioned data.

    Args:
        constant_block (string): what should happen if the output value will
        always be the same due to the window size being so big.
        calculate_block (string): how should the current array value be
        calculated in terms of the up-to-date accumulators
        out_dtype (dtype): what is the dtype of the output data.
        setup_block (string, optional): what should happen to initialize the
        accumulators
        enter_block (string, optional): what should happen when a non-null
        element enters the window
        exit_block (string, optional):  what should happen when a non-null
        element exits the window
        empty_block (string, optional): what should happen if the entire window
        frame is null. If None, calls setna. Defaults to None.

    Returns:
        function: a window function that takes in a Series, lower bound,
        and upper bound, and outputs an array where each value corresponds
        to the desired aggregation of the values from the specified bounds
        starting from each index.

    Usage notes:
        - When writing enter_block/exit_block, the variable "elem" is used to
          denote the current value entering/exiting the array. You may assume
          that this value is not null.
        - The variable "in_window "is used to denote the length of the window
          frame that is not out of bounds, excluding nulls.
        - When writing calculate_block, store the final answer as "res[i] = ..."
        - When writing constant_block, store the value answer as "constant_value = ..."

    The generated code will look as follows:

    def impl(S, lower_bound, upper_bound):
        n = len(S)
        arr = bodo.utils.conversion.coerce_to_array(S)
        res = bodo.utils.utils.alloc_type(n, out_dtype, -1)
        # If the slice is empty, output all nulls
        if upper_bound < lower_bound:
            for i in range(n):
                bodo.libs.array_kernels.setna(res, i)
        elif lower_bound <= -n+1 and n-1 <= upper_bound:
            # <CONSTANT_BLOCK>
            for i in range(n):
                res[i] = constant_value
        else:
            # Keep track of the first/last index of the current window,
            # the number of non-null entries in the window, and the current
            # values for the crucial variables
            exiting = lower_bound
            entering = upper_bound
            in_window = 0
            << INSERT SETUP_BLOCK HERE >>
            # Calculate the starting values
            for i in range(min(max(0, exiting), n), min(max(0, entering + 1), n)):
                if not bodo.libs.array_kernels.isna(arr, i):
                    in_window += 1
                    elem = arr[i]
                    << INSERT ENTER_BLOCK HERE >>
            # Loop over each entry and update the number of elements
            # in the window & the current sum
            for i in range(n):
                # The current element is null if the window has no
                # non-null elements, otherwise it is the current total
                if in_window == 0:
                    << INSERT EMPTY_BLOCK HERE >>
                else:
                    << INSERT CALCULATE_BLOCK HERE >>
                # If the old end of the window is in bounds and non-null,
                # remove it
                if 0 <= exiting < n:
                    if not bodo.libs.array_kernels.isna(arr, exiting):
                    in_window -= 1
                    elem = arr[exiting]
                    << INSERT EXIT_BLOCK HERE >>
                # Move the start/end of the window forward 1 step
                exiting += 1
                entering += 1
                # If the new start of the window is in bounds and non-null,
                # add it
                if 0 <= entering < n:
                    if not bodo.libs.array_kernels.isna(arr, entering):
                    in_window += 1
                    elem = arr[entering]
                    << INSERT ENTER_BLOCK HERE >>
        return res
    """
    # Calculate the indentation of the calculate_block so that it can be removed
    calculate_lines = calculate_block.splitlines()
    calculate_indentation = len(calculate_lines[0]) - len(calculate_lines[0].lstrip())

    # Calculate the indentation of the constant_block so that it can be removed
    constant_lines = constant_block.splitlines()
    constant_indentation = len(constant_lines[0]) - len(constant_lines[0].lstrip())

    if setup_block != None:
        # Calculate the indentation of the setup_block so that it can be removed
        setup_lines = setup_block.splitlines()
        setup_indentation = len(setup_lines[0]) - len(setup_lines[0].lstrip())

    if enter_block != None:
        # Calculate the indentation of the enter_block so that it can be removed
        enter_lines = enter_block.splitlines()
        enter_indentation = len(enter_lines[0]) - len(enter_lines[0].lstrip())

    if exit_block != None:
        # Calculate the indentation of the exit_block so that it can be removed
        exit_lines = exit_block.splitlines()
        exit_indentation = len(exit_lines[0]) - len(exit_lines[0].lstrip())

    # Calculate the indentation of the empty_block so that it can be removed
    if empty_block == None:
        empty_block = "bodo.libs.array_kernels.setna(res, i)"
    empty_lines = empty_block.splitlines()
    empty_indentation = len(empty_lines[0]) - len(empty_lines[0].lstrip())

    func_text = "def impl(S, lower_bound, upper_bound):\n"
    func_text += "   n = len(S)\n"
    func_text += "   arr = bodo.utils.conversion.coerce_to_array(S)\n"
    func_text += "   res = bodo.utils.utils.alloc_type(n, out_dtype, -1)\n"
    func_text += "   if upper_bound < lower_bound:\n"
    func_text += "      for i in range(n):\n"
    func_text += "         bodo.libs.array_kernels.setna(res, i)\n"
    func_text += "   elif lower_bound <= -n+1 and n-1 <= upper_bound:\n"
    func_text += (
        "\n".join([" " * 6 + line[constant_indentation:] for line in constant_lines])
        + "\n"
    )
    func_text += "      for i in range(n):\n"
    func_text += "         res[i] = constant_value\n"
    func_text += "   else:\n"
    func_text += "      exiting = lower_bound\n"
    func_text += "      entering = upper_bound\n"
    func_text += "      in_window = 0\n"
    if setup_block != None:
        func_text += (
            "\n".join([" " * 6 + line[setup_indentation:] for line in setup_lines])
            + "\n"
        )
    func_text += (
        "      for i in range(min(max(0, exiting), n), min(max(0, entering + 1), n)):\n"
    )
    func_text += "         if not bodo.libs.array_kernels.isna(arr, i):\n"
    func_text += "            in_window += 1\n"
    if enter_block != None:
        if "elem" in enter_block:
            func_text += "            elem = arr[i]\n"
        func_text += (
            "\n".join([" " * 12 + line[enter_indentation:] for line in enter_lines])
            + "\n"
        )
    func_text += "      for i in range(n):\n"
    func_text += "         if in_window == 0:\n"
    func_text += (
        "\n".join([" " * 12 + line[empty_indentation:] for line in empty_lines]) + "\n"
    )
    func_text += "         else:\n"
    func_text += (
        "\n".join([" " * 12 + line[calculate_indentation:] for line in calculate_lines])
        + "\n"
    )
    func_text += "         if 0 <= exiting < n:\n"
    func_text += "            if not bodo.libs.array_kernels.isna(arr, exiting):\n"
    func_text += "               in_window -= 1\n"
    if exit_block != None:
        if "elem" in exit_block:
            func_text += "               elem = arr[exiting]\n"
        func_text += (
            "\n".join([" " * 15 + line[exit_indentation:] for line in exit_lines])
            + "\n"
        )
    func_text += "         exiting += 1\n"
    func_text += "         entering += 1\n"
    func_text += "         if 0 <= entering < n:\n"
    func_text += "            if not bodo.libs.array_kernels.isna(arr, entering):\n"
    func_text += "               in_window += 1\n"
    if enter_block != None:
        if "elem" in enter_block:
            func_text += "               elem = arr[entering]\n"
        func_text += (
            "\n".join([" " * 15 + line[enter_indentation:] for line in enter_lines])
            + "\n"
        )
    func_text += "   return res"

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
