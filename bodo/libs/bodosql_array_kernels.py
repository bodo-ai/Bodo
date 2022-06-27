# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements array kernels that are specific to BodoSQL
"""
import numba
import numpy as np
from numba.core import types
from numba.extending import overload

import bodo
from bodo.utils.typing import raise_bodo_error


def gen_vectorized(
    arg_names, arg_types, propogate_null, scalar_text, constructor_text, arg_string=None
):
    """Creates an impl for a function that has several inputs that could all be
       scalars, nulls, or arrays by broadcasting appropriately.

    Args:
        arg_names (string list): the names of each argument
        arg_types (dtype list): the types of each argument
        propogate_null (bool list): a mask indicating which arguments produce
        an output of null when the input is null
        scalar_text (string): the func_text of the core operation done on each
        set of scalar values after all broadcasting is handled. The string should
        refer to the scalar values as arg0, arg1, arg2, etc. (where arg0
        corresponds to the current value from arg_names[0]), and store the final
        answer of the calculation in res[i]
        constructor_text (string): the expression that generates the final
        result array. Assume that the size of the array is in a variable n.
        arg_string (optional string): string representing the parameters of the
        function as the appear in the def line. If not provided, they are
        inferred from arg_names.

    Returns:
        function: a broadcasted version of the calculation described by
        scalar_text, which can be used as an overload.

    Internal Doc explaining more about this utility:
    https://bodo.atlassian.net/wiki/spaces/B/pages/1080492110/BodoSQL+Array+Kernel+Parametrization

    Below is an example that would vectorize the sum operation, where if either
    element is NULL the output is NULL. In this case, it is being constructed
    for when both arguments are arrays.

    arg_names = ['left', 'right']
    arg_types = [Int64Array, Int64Array]
    propogate_null = [True, True]
    scalar_text = "res[i] = arg0 + arg1"
    constructor_text = "bodo.libs.int_arr_ext.alloc_int_array(n, np.int64))"

    This would result in an impl constructed from the following func_text:

    def impl(left, right):
        n = len(left)
        res = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64))
        left = bodo.utils.conversion.coerce_to_array(left)
        right = bodo.utils.conversion.coerce_to_array(right)
        for i in range(n):
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
    """
    are_arrays = [bodo.utils.utils.is_array_typ(typ, True) for typ in arg_types]
    all_scalar = not any(are_arrays)
    out_null = any(
        [propogate_null[i] for i in range(len(arg_types)) if arg_types[i] == bodo.none]
    )

    # Calculate the indentation of the scalar_text so that it can be removed
    first_line = scalar_text.splitlines()[0]
    base_indentation = len(first_line) - len(first_line.lstrip())

    if arg_string == None:
        arg_string = ", ".join(arg_names)

    func_text = f"def impl({arg_string}):\n"

    # If all the inputs are scalar, either output None immediately or
    # compute a single scalar computation without the loop
    if all_scalar:
        if out_null:
            func_text += "   return None"
        else:
            for i in range(len(arg_names)):
                func_text += f"   arg{i} = {arg_names[i]}\n"
            for line in scalar_text.splitlines():
                func_text += (
                    " " * 3
                    + line[base_indentation:].replace("res[i] =", "answer =")
                    + "\n"
                )
            func_text += "   return answer"

    else:
        # Determine the size of the final output array and convert Series to arrays
        found_size = False
        for i in range(len(arg_names)):
            if are_arrays[i]:
                if not found_size:
                    size_text = f"len({arg_names[i]})"
                    found_size = True
                if not bodo.utils.utils.is_array_typ(arg_types[i], False):
                    func_text += f"   {arg_names[i]} = bodo.hiframes.pd_series_ext.get_series_data({arg_names[i]})\n"

        func_text += f"   n = {size_text}\n"
        func_text += f"   res = {constructor_text}\n"
        func_text += "   for i in range(n):\n"

        # If the argument types imply that every row is null, then just set each
        # row of the output array to null
        if out_null:
            func_text += f"      bodo.libs.array_kernels.setna(res, i)\n"

        else:
            # For each column that propogates nulls, add an isna check (and
            # also convert Series to arrays)
            for i in range(len(arg_names)):
                if are_arrays[i]:
                    if propogate_null[i]:
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
        },
        loc_vars,
    )

    impl = loc_vars["impl"]

    return impl


def unopt_argument(func_name, arg_names, i):
    """Creates an impl that cases on whether or not a certain argument to a function
       is None in order to un-optionalize that argument

    Args:
        func_name (string): the name of hte function with the optional arguments
        arg_names (string list): the name of each argument to the function
        i (integer): the index of the argument from arg_names being unoptionalized

    Returns:
        function: the impl that re-calls func_name with arg_names[i] no longer optional
    """
    args1 = [arg_names[j] if j != i else "None" for j in range(len(arg_names))]
    args2 = [
        arg_names[j] if j != i else f"bodo.utils.indexing.unoptional({arg_names[j]})"
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
    to the apropriate version of the real implementaiton"""
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
    to the apropriate version of the real implementaiton"""
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
        if arr not in (types.none, types.unicode_type) and not (
            bodo.utils.utils.is_array_typ(arr, True) and arr.dtype == types.unicode_type
        ):
            raise_bodo_error(
                f"{func_name} can only be applied to strings, string columns, or null"
            )

        if length not in (types.none, *types.integer_domain) and not (
            bodo.utils.utils.is_array_typ(length, True)
            and length.dtype in types.integer_domain
        ):
            raise_bodo_error(
                f"{func_name} length argument must be an integer, integer column, or null"
            )

        if pad_string not in (types.none, types.unicode_type) and not (
            bodo.utils.utils.is_array_typ(pad_string, True)
            and pad_string.dtype == types.unicode_type
        ):
            raise_bodo_error(
                f"{func_name} {func_name.lower()}_string argument must be a string, string column, or null"
            )

        if func_name == "LPAD":
            pad_line = f"(arg2 * quotient) + arg2[:remainder] + arg0"
        elif func_name == "RPAD":
            pad_line = f"arg0 + (arg2 * quotient) + arg2[:remainder]"

        arg_names = ["arr", "length", "pad_string"]
        arg_types = [arr, length, pad_string]
        propogate_null = [True] * 3
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
        constructor_text = "bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)"

        return gen_vectorized(
            arg_names, arg_types, propogate_null, scalar_text, constructor_text
        )

    return overload_lpad_rpad_util


def _install_lpad_rpad_overload():
    """Creates and installs the overloads for lpad_util and rpad_util"""
    for func, func_name in zip((lpad_util, rpad_util), ("LPAD", "RPAD")):
        overload_impl = create_lpad_rpad_util_overload(func_name)
        overload(func)(overload_impl)


_install_lpad_rpad_overload()
