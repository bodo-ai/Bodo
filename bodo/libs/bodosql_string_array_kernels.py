# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements string array kernels that are specific to BodoSQL
"""


import numba
import numpy as np
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


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
def editdistance_no_max(s, t):
    """Handles cases where EDITDISTANCE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [s, t]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.editdistance_no_max",
                ["s", "t"],
                i,
            )

    def impl(s, t):  # pragma: no cover
        return editdistance_no_max_util(s, t)

    return impl


@numba.generated_jit(nopython=True)
def editdistance_with_max(s, t, maxDistance):
    """Handles cases where EDITDISTANCE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [s, t, maxDistance]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.editdistance_with_max",
                ["s", "t", "maxDistance"],
                i,
            )

    def impl(s, t, maxDistance):  # pragma: no cover
        return editdistance_with_max_util(s, t, maxDistance)

    return impl


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
def initcap(arr, delim):
    """Handles cases where INITCAP recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, delim]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.initcap",
                ["arr", "delim"],
                i,
            )

    def impl(arr, delim):  # pragma: no cover
        return initcap_util(arr, delim)

    return impl


@numba.generated_jit(nopython=True)
def instr(arr, target):
    """Handles cases where INSTR recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, target]
    for i in range(2):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.instr", ["arr", "target"], i
            )

    def impl(arr, target):  # pragma: no cover
        return instr_util(arr, target)

    return impl


def left(arr, n_chars):  # pragma: no cover
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


def lpad(arr, length, padstr):  # pragma: no cover
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


def right(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


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


def rpad(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


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
def split_part(source, delim, part):
    """Handles cases where SPLIT_PART receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, delim, part]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.split_part",
                ["source", "delim", "part"],
                i,
            )

    def impl(source, delim, part):  # pragma: no cover
        return split_part_util(source, delim, part)

    return impl


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
def strtok(source, delim, part):
    """Handles cases where STRTOK receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, delim, part]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.strtok", ["source", "delim", "part"], i
            )

    def impl(source, delim, part):  # pragma: no cover
        return strtok_util(source, delim, part)

    return impl


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
def translate(arr, source, target):
    """Handles cases where TRANSLATE recieves optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, source, target]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.translate",
                ["arr", "source", "target"],
                i,
            )

    def impl(arr, source, target):  # pragma: no cover
        return translate_util(arr, source, target)

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
        string array/scalar: the character(s) corresponding to the integer(s)
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
def initcap_util(arr, delim):
    """A dedicated kernel for the SQL function INITCAP which takes in a source
    string (or column) and a delimeter string (or column) capitalizes the first
    character and every character after the characters in the delimeter string.


    Args:
        arr (string array/series/scalar): the string(s) being capitalized
        delim (string array/series/scalar): the delimeter string(s) to capitalize
        after

    Returns:
        string array/scalar: the capitalized string(s)
    """

    verify_string_arg(arr, "INITCAP", "arr")
    verify_string_arg(delim, "INITCAP", "delim")

    arg_names = ["arr", "delim"]
    arg_types = [arr, delim]
    propagate_null = [True] * 2
    scalar_text = "capitalized = arg0[:1].upper()\n"
    scalar_text += "for j in range(1, len(arg0)):\n"
    scalar_text += "   if arg0[j-1] in arg1:\n"
    scalar_text += "      capitalized += arg0[j].upper()\n"
    scalar_text += "   else:\n"
    scalar_text += "      capitalized += arg0[j].lower()\n"
    scalar_text += "res[i] = capitalized"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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


@register_jitable
def min_edit_distance(s, t):  # pragma: no cover
    """Utility for finding the minimum edit distance between two scalar strings.
    Algorithm derived from the following:
    https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows

    Args:
        s (string): the first string being compared
        t (string): the second string being compared

    Returns:
        int: the minimum edit distance between s and t.
    """
    # Ensure that s is the shorter of the two strings
    if len(s) > len(t):
        s, t = t, s
    m, n = len(s), len(t)

    # Use a 2 x (m + 1) array to represent an n x (m + 1) array since you only
    # need to consider the previous row to generate the next row, therefore the
    # same two rows can be recycled
    row, otherRow = 1, 0
    arr = np.zeros((2, m + 1), dtype=np.uint32)

    # MED(X, "") = len(X)
    arr[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        # MED("", X) = len(X)
        arr[row, 0] = i

        # Loop over the rest of s to see if it matches with the corresponding letter of t
        for j in range(1, m + 1):

            # If these two characters match, then the diagonal entry above them is the MED
            if s[j - 1] == t[i - 1]:
                arr[row, j] = arr[otherRow, j - 1]

            # Otherwise, it is the min of the diagonal entry and the one above / to the left
            else:
                arr[row, j] = 1 + min(
                    arr[row, j - 1], arr[otherRow, j], arr[otherRow, j - 1]
                )

        row, otherRow = otherRow, row

    return arr[n % 2, m]


@register_jitable
def min_edit_distance_with_max(s, t, maxDistance):  # pragma: no cover
    """Utility for finding the minimum edit distance between two scalar strings
    when provided with a maximum distance. This is seperate from
    min_edit_distance_without_max because it has extra checks inside of the
    loops. Algorithm derived from the following:
    https://en.wikipedia.org/wiki/Levenshtein_distance#Iterative_with_two_matrix_rows


    Args:
        s (string): the first string being compared.
        t (string): the second string being compared.
        maxDistance (int): the maximum distance to search (ignored if None).

    Returns:
        int: the minimum edit distance between s and t. If maxDistance is less than
          the MED, then it is returned instead (or 0 if negative)
    """
    if maxDistance < 0:
        return 0

    # Ensure that s is the shorter of the two strings
    if len(s) > len(t):
        s, t = t, s
    m, n = len(s), len(t)

    # If the max distance is irrelevant, use the other implementation
    if m <= maxDistance and n <= maxDistance:
        return min_edit_distance(s, t)

    # Use a 2 x (m + 1) array to represent an n x (m + 1) array since you only
    # need to consider the previous row to generate the next row, therefore the
    # same two rows can be recycled
    row, otherRow = 1, 0
    arr = np.zeros((2, m + 1), dtype=np.uint32)

    # MED(X, "") = len(X)
    arr[0, :] = np.arange(m + 1)

    for i in range(1, n + 1):
        # MED("", X) = len(X)
        arr[row, 0] = i

        # Loop over the rest of s to see if it matches with the corresponding letter of t
        for j in range(1, m + 1):

            # If these two characters match, then the diagonal entry above them is the MED
            if s[j - 1] == t[i - 1]:
                arr[row, j] = arr[otherRow, j - 1]

            # Otherwise, it is the min of the diagonal entry and the one above / to the left
            else:
                arr[row, j] = 1 + min(
                    arr[row, j - 1], arr[otherRow, j], arr[otherRow, j - 1]
                )

        # If the entire row is above the max depth, halt early
        if (arr[row] >= maxDistance).all():
            return maxDistance

        row, otherRow = otherRow, row

    return min(arr[n % 2, m], maxDistance)


@numba.generated_jit(nopython=True)
def editdistance_no_max_util(s, t):
    """A dedicated kernel for the SQL function EDITDISTANCE which two strings
    (or columns) and returns the minimum edit distance between them (i.e. the
    smallest number of insertions/deletions/replacements required to make the
    two strings identical)


    Args:
        s (string array/series/scalar): the first string(s) being compared
        t (string array/series/scalar): the second string(s) being compared

    Returns:
        int series/scalar: the minimum edit distnace between the two strings
    """

    verify_string_arg(s, "editdistance_no_max", "s")
    verify_string_arg(t, "editdistance_no_max", "t")

    arg_names = ["s", "t"]
    arg_types = [s, t]
    propagate_null = [True] * 2
    scalar_text = (
        "res[i] = bodo.libs.bodosql_array_kernels.min_edit_distance(arg0, arg1)"
    )

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def editdistance_with_max_util(s, t, maxDistance):
    """Same as editdistance_no_max_util, except it supports the version with
    the third argument for the maximum distance to search before giving up.


    Args:
        s (string array/series/scalar): the first string(s) being compared
        t (string array/series/scalar): the second string(s) being compared
        maxDistance (int array/series/scalar): the distance(s) to search before giving up

    Returns:
        int series/scalar: the minimum edit distnace between the two strings.
        if it is greater than maxDistance, then maxDistance is returned. If
        maxDistance is negative, 0 is returned.
    """

    verify_string_arg(s, "editdistance_no_max", "s")
    verify_string_arg(t, "editdistance_no_max", "t")
    verify_int_arg(maxDistance, "editdistance_no_max", "t")

    arg_names = ["s", "t", "maxDistance"]
    arg_types = [s, t, maxDistance]
    propagate_null = [True] * 3
    scalar_text = "res[i] = bodo.libs.bodosql_array_kernels.min_edit_distance_with_max(arg0, arg1, arg2)"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def format_util(arr, places):
    """A dedicated kernel for the SQL function FORMAT which takes in two
    integers (or columns) and formats the former with commas at every
    thousands place, with decimal precision specified by the latter column


    Args:
        arr (integer array/series/scalar): the integer(s) to be modified formatted
        places (integer array/series/scalar): the precision of the decimal place(s)

    Returns:
        string array/scalar: the string/column of formatted numbers
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


def left_util(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


def right_util(arr, n_chars):  # pragma: no cover
    # Dummy function used for overload
    return


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
        arr_is_string = verify_string_binary_arg(arr, func_name, "arr")
        verify_int_arg(n_chars, func_name, "n_chars")

        empty_char = "''" if arr_is_string else "b''"

        arg_names = ["arr", "n_chars"]
        arg_types = [arr, n_chars]
        propagate_null = [True] * 2
        scalar_text = "if arg1 <= 0:\n"
        scalar_text += f"   res[i] = {empty_char}\n"
        scalar_text += "else:\n"
        if func_name == "LEFT":
            scalar_text += "   res[i] = arg0[:arg1]"
        elif func_name == "RIGHT":
            scalar_text += "   res[i] = arg0[-arg1:]"

        out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

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


def lpad_util(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


def rpad_util(arr, length, padstr):  # pragma: no cover
    # Dummy function used for overload
    return


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
        pad_is_string = verify_string_binary_arg(pad_string, func_name, "pad_string")
        arr_is_string = verify_string_binary_arg(arr, func_name, "arr")
        if arr_is_string != pad_is_string:
            raise bodo.utils.typing.BodoError(
                "Pad string and arr must be the same type!"
            )

        out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

        verify_int_arg(length, func_name, "length")
        verify_string_binary_arg(pad_string, func_name, f"{func_name.lower()}_string")

        if func_name == "LPAD":
            pad_line = f"(arg2 * quotient) + arg2[:remainder] + arg0"
        elif func_name == "RPAD":
            pad_line = f"arg0 + (arg2 * quotient) + arg2[:remainder]"

        arg_names = ["arr", "length", "pad_string"]
        arg_types = [arr, length, pad_string]
        propagate_null = [True] * 3

        empty_char = "''" if arr_is_string else "b''"

        scalar_text = f"""\
                if arg1 <= 0:
                    res[i] = {empty_char}
                elif len(arg2) == 0:
                    res[i] = arg0
                elif len(arg0) >= arg1:
                    res[i] = arg0[:arg1]
                else:
                    quotient = (arg1 - len(arg0)) // len(arg2)
                    remainder = (arg1 - len(arg0)) % len(arg2)
                    res[i] = {pad_line}"""

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
def repeat_util(arr, repeats):
    """A dedicated kernel for the SQL function REPEAT which takes in a string
       and integer (either of which can be a scalar or vector) and
       concatenates the string to itself repeatedly according to the integer


    Args:
        arr (string array/series/scalar): the string(s) being repeated
        repeats (integer array/series/scalar): the number(s) of repeats

    Returns:
        string array/scalar: the repeated string(s)
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
def replace_util(arr, to_replace, replace_with):
    """A dedicated kernel for the SQL function REVERSE which takes in a base string
       (or string column), a second string to locate in the base string, and a
       third string with which to replace it.


    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        to_replace (string array/series/scalar): the substring(s) to replace
        replace_with (string array/series/scalar): the string(s) that replace to_replace

    Returns:
        string array/scalar: the string/column where each ocurrence of
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
def reverse_util(arr):
    """A dedicated kernel for the SQL function REVERSE which takes in a string
       (or string column) and reverses it


    Args:
        arr (string array/series/scalar): the strings(s) to be reversed

    Returns:
        string array/scalar: the string/column that has been reversed
    """

    arr_is_string = verify_string_binary_arg(arr, "REVERSE", "arr")

    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    scalar_text = "res[i] = arg0[::-1]"

    out_dtype = bodo.string_array_type
    out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


@numba.generated_jit(nopython=True)
def space_util(n_chars):
    """A dedicated kernel for the SQL function SPACE which takes in an integer
       (or integer column) and returns that many spaces


    Args:
        n_chars (integer array/series/scalar): the number(s) of spaces

    Returns:
        string array/scalar: the string/column of spaces
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
def split_part_util(source, delim, part):
    """A dedicated kernel for the SQL function SPLIT_PART which takes in a
    source string (or column), a delimeter string (or column), and a part
    integer (or column), then splits the source string by occurrences of
    the entire delimeter string and outputs the value specified by the part.
    Part is allowed to be negative.

    Has the following edge cases:
    - Outputs NULL if source and delim are empty
    - If delim is otherwise empty, the source string is not split
    - Outputs "" if part is too small or too big


    Args:
        source (string array/series/scalar): the string(s) to be parsed
        delim (string array/series/scalar): the string(s) to split on
        part (integer array/series/scalar): the occurrence to return

    Returns:
        string array/scalar: the extracted part of the string
    """

    verify_string_arg(source, "SPLIT_PART", "source")
    verify_string_arg(delim, "SPLIT_PART", "delim")
    verify_int_arg(part, "SPLIT_PART", "part")

    arg_names = ["source", "delim", "part"]
    arg_types = [source, delim, part]
    propagate_null = [True] * 3
    # Splitting by '' is valid in SQL, but not in Python
    scalar_text = "tokens = arg0.split(arg1) if arg1 != '' else [arg0]\n"
    scalar_text += "if abs(arg2) > len(tokens):\n"
    scalar_text += "    res[i] = ''\n"
    scalar_text += "else:\n"
    scalar_text += "    res[i] = tokens[arg2 if arg2 <= 0 else arg2-1]\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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
def strtok_util(source, delim, part):
    """A dedicated kernel for the SQL function STRTOK which works the same
    as SPLIT_PART with the following differences:

    - Splits by occurrences of any character in delim instead of occurrences
      of the entire delim string
    - If part is 0, negative, or too big, returns NULL instead of ""
    - If source and delim are both empty, outputs NULL
    - Does not count the empty string as a token under any circumstances

    Args:
        source (string array/series/scalar): the string(s) to be parsed
        delim (string array/series/scalar): the string(s) to split on
        part (integer array/series/scalar): the occurrence to return

    Returns:
        string array/scalar: the extracted part of the string
    """

    verify_string_arg(source, "STRTOK", "source")
    verify_string_arg(delim, "STRTOK", "delim")
    verify_int_arg(part, "STRTOK", "part")

    arg_names = ["source", "delim", "part"]
    arg_types = [source, delim, part]
    propagate_null = [True] * 3
    scalar_text = "if (arg0 == '' and arg1 == '') or arg2 <= 0:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
    scalar_text += "else:\n"
    scalar_text += "   tokens = []\n"
    scalar_text += "   buffer = ''\n"
    scalar_text += "   for j in range(len(arg0)):\n"
    scalar_text += "      if arg0[j] in arg1:\n"
    scalar_text += "         if buffer != '':"
    scalar_text += "            tokens.append(buffer)\n"
    scalar_text += "         buffer = ''\n"
    scalar_text += "      else:\n"
    scalar_text += "         buffer += arg0[j]\n"
    scalar_text += "   if buffer != '':\n"
    scalar_text += "      tokens.append(buffer)\n"
    scalar_text += "   if arg2 > len(tokens):\n"
    scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
    scalar_text += "   else:\n"
    scalar_text += "      res[i] = tokens[arg2-1]\n"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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
        string array/scalar: the string/column of extracted substrings
    """

    arr_is_string = verify_string_binary_arg(arr, "SUBSTRING", "arr")
    verify_int_arg(start, "SUBSTRING", "start")
    verify_int_arg(length, "SUBSTRING", "length")

    out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

    arg_names = ["arr", "start", "length"]
    arg_types = [arr, start, length]
    propagate_null = [True] * 3
    scalar_text = "if arg2 <= 0:\n"
    scalar_text += "   res[i] = ''\n" if arr_is_string else "   res[i] = b''\n"
    scalar_text += "elif arg1 < 0 and arg1 + arg2 >= 0:\n"
    scalar_text += "   res[i] = arg0[arg1:]\n"
    scalar_text += "else:\n"
    scalar_text += "   if arg1 > 0: arg1 -= 1\n"
    scalar_text += "   res[i] = arg0[arg1:arg1+arg2]\n"

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)


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
        string array/scalar: the string/column of prefixes before ocurrences
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


@numba.generated_jit(nopython=True)
def translate_util(arr, source, target):
    """A dedicated kernel for the SQL function TRANSLATE which takes in a string
       (or string column) and two alphabet strings (or columns) and replaces
       each character in the source string from the first alphabet with the
       corresponding character from the second character (or deletes the
       character if the second alphabet is shorter)


    Args:
        arr (string array/series/scalar): the string(s) being translated
        source (string array/series/scalar): the characters being converted
        target (string array/series/scalar): the characters replacing the source
        alphabet

    Returns:
        string array/scalar: the translated string(s)
    """

    verify_string_arg(arr, "translate", "arr")
    verify_string_arg(source, "translate", "source")
    verify_string_arg(target, "translate", "target")

    arg_names = ["arr", "source", "target"]
    arg_types = [arr, source, target]
    propagate_null = [True] * 3
    scalar_text = "translated = ''\n"
    scalar_text += "for char in arg0:\n"
    scalar_text += "   index = arg1.find(char)\n"
    scalar_text += "   if index == -1:\n"
    scalar_text += "      translated += char\n"
    scalar_text += "   elif index < len(arg2):\n"
    scalar_text += "      translated += arg2[index]\n"
    scalar_text += "res[i] = translated"

    out_dtype = bodo.string_array_type

    return gen_vectorized(arg_names, arg_types, propagate_null, scalar_text, out_dtype)
