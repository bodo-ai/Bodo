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
    arg_names,
    arg_types,
    propogate_null,
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
        propogate_null (bool list): a mask indicating which arguments produce
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
    propogate_null = [True, True]
    out_dtype = np.int64
    scalar_text = "res[i] = arg0 + arg1"

    This would result in an impl constructed from the following func_text:

    def impl(left, right):
        n = len(left)
        res = bodo.utils.utils.alloc_type(n, out_dtype, -1)
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
        return return bodo.hiframes.pd_series_ext.init_series(res, bodo.hiframes.pd_index_ext.init_range_index(0, n, 1), None)

    (Where out_dtype is mapped to np.int64)
    """
    are_arrays = [bodo.utils.utils.is_array_typ(typ, True) for typ in arg_types]
    all_scalar = not any(are_arrays)
    out_null = any(
        [propogate_null[i] for i in range(len(arg_types)) if arg_types[i] == bodo.none]
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
        {"bodo": bodo, "numba": numba, "np": np, "out_dtype": out_dtype},
        loc_vars,
    )

    impl = loc_vars["impl"]

    return impl


def unopt_argument(func_name, arg_names, i, container_length=None):
    """Creates an impl that cases on whether or not a certain argument to a function
       is None in order to un-optionalize that argument

    Args:
        func_name (string): the name of hte function with the optional arguments
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
        out_dtype = bodo.string_array_type

        return gen_vectorized(
            arg_names, arg_types, propogate_null, scalar_text, out_dtype
        )

    return overload_lpad_rpad_util


def _install_lpad_rpad_overload():
    """Creates and installs the overloads for lpad_util and rpad_util"""
    for func, func_name in zip((lpad_util, rpad_util), ("LPAD", "RPAD")):
        overload_impl = create_lpad_rpad_util_overload(func_name)
        overload(func)(overload_impl)


_install_lpad_rpad_overload()


def coalesce(A):  # pragma: no cover
    # Dummy function used for overload
    return


def coalesce_util(A):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(coalesce)
def overload_coalesce(A):
    """Handles cases where COALESCE recieves optional arguments and forwards
    to the apropriate version of the real implementation"""
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
    propogate_null = [False] * (len(A) - len(dead_cols))
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
        out_dtype = bodo.none
    elif len(elem_types) == 1:
        if bodo.utils.utils.is_array_typ(elem_types[0]):
            out_dtype = elem_types[0]
        else:
            out_dtype = bodo.utils.typing.dtype_to_array_type(elem_types[0])
    else:
        # Verify that the underlying scalar types are common before extracting
        # the corresponding output_dtype
        scalar_dtypes = []
        for i in range(len(arg_types)):
            if bodo.utils.utils.is_array_typ(arg_types[i]):
                scalar_dtypes.append(elem_types[i].dtype)
            else:
                scalar_dtypes.append(elem_types[i])
        common_dtype, success = bodo.utils.typing.get_common_scalar_dtype(scalar_dtypes)
        if not success:
            raise_bodo_error("Cannot coalesce columns with different dtypes")
        out_dtype = bodo.utils.typing.to_nullable_type(
            bodo.utils.typing.dtype_to_array_type(common_dtype)
        )

    return gen_vectorized(
        arg_names,
        arg_types,
        propogate_null,
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
    to the apropriate version of the real implementaiton"""
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
    to the apropriate version of the real implementaiton"""
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
        if arr not in (types.none, types.unicode_type) and not (
            bodo.utils.utils.is_array_typ(arr, True) and arr.dtype == types.unicode_type
        ):
            raise_bodo_error(
                f"{func_name} can only be applied to strings, string columns, or null"
            )

        if (
            n_chars != types.none
            and not isinstance(n_chars, types.Integer)
            and not (
                bodo.utils.utils.is_array_typ(n_chars, True)
                and n_chars.dtype in types.integer_domain
            )
        ):
            raise_bodo_error(
                f"{func_name} n_chars argument must be an integer, integer column, or null"
            )

        arg_names = ["arr", "n_chars"]
        arg_types = [arr, n_chars]
        propogate_null = [True] * 2
        scalar_text = "if arg1 <= 0:\n"
        scalar_text += "   res[i] = ''\n"
        scalar_text += "else:\n"
        if func_name == "LEFT":
            scalar_text += "   res[i] = arg0[:arg1]"
        elif func_name == "RIGHT":
            scalar_text += "   res[i] = arg0[-arg1:]"

        out_dtype = bodo.string_array_type

        return gen_vectorized(
            arg_names, arg_types, propogate_null, scalar_text, out_dtype
        )

    return overload_left_right_util


def _install_left_right_overload():
    """Creates and installs the overloads for left_util and right_util"""
    for func, func_name in zip((left_util, right_util), ("LEFT", "RIGHT")):
        overload_impl = create_left_right_util_overload(func_name)
        overload(func)(overload_impl)


_install_left_right_overload()
