# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Common utilities for all BodoSQL array kernels
"""

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from numba.core import types

import bodo
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
        support_dict_encoding (optional boolean) if true, allows dictionary
        encoded outputs under certain conditions

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

    NOTE: dictionary encoded outputs operate under the following assumptions:
    - The output will only be dictionary encoded if exactly one of the inputs
        is dicitonary encoded, and hte rest are all scalars, and support_dict_encoding
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
            if arg_types[i].dtype == bodo.dict_str_arr_type:
                dict_encoded_arg = i
    use_dict_encoding = (
        support_dict_encoding
        and vector_args == 1
        and dict_encoded_arg >= 0
        and out_dtype == bodo.string_array_type
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
            func_text += f"   indices = {arg_names[dict_encoded_arg]}._indices.copy()\n"
            func_text += f"   has_global = {arg_names[dict_encoded_arg]}._has_global_dictionary\n"
            func_text += f"   {arg_names[i]} = {arg_names[dict_encoded_arg]}._data\n"

        # If dictionary encoded outputs are not being used, then the output is
        # still bodo.string_array_type, the number of loop iterations is still the
        # length of the indices, and scalar_text/propagate_null should work the
        # same because isna checks the data & indices, and scalar_text uses the
        # arguments extracted by getitem.
        func_text += f"   n = {size_text}\n"
        if use_dict_encoding:
            func_text += (
                "   res = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)\n"
            )
            func_text += "   for i in range(n):\n"
        else:
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

        # If using dictionary encoding, construct the output from the
        # new dictionary + the indices, and also make sure that all nulls in
        # the dicitonary are propageted back to the indices
        if use_dict_encoding:
            func_text += "   for i in range(n):\n"
            func_text += "      if not bodo.libs.array_kernels.isna(indices, i):\n"
            func_text += "         loc = indices[i]\n"
            func_text += "         if bodo.libs.array_kernels.isna(res, loc):\n"
            func_text += "            bodo.libs.array_kernels.setna(indices, i)\n"
            func_text += "   res = bodo.libs.dict_arr_ext.init_dict_arr(res, indices, has_global)\n"
            func_text += "   return bodo.hiframes.pd_series_ext.init_series(res, bodo.hiframes.pd_index_ext.init_range_index(0, len(indices), 1), None)"
        else:
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
    return not (
        arg not in (types.none, types.unicode_type)
        and not isinstance(arg, types.StringLiteral)
        and not (
            bodo.utils.utils.is_array_typ(arg, True) and arg.dtype == types.unicode_type
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
