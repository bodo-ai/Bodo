"""
Implements string array kernels that are specific to BodoSQL
"""

import numba
import numpy as np
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
import bodo.libs.uuid
import bodosql
from bodo.utils.typing import (
    BodoError,
    get_overload_const_bool,
    get_overload_const_int,
    get_overload_const_str,
    is_overload_constant_int,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.kernels.array_kernel_utils import (
    gen_vectorized,
    is_overload_constant_str,
    is_valid_binary_arg,
    unopt_argument,
    verify_int_arg,
    verify_int_float_arg,
    verify_string_arg,
    verify_string_binary_arg,
)


@numba.generated_jit(nopython=True)
def char(arr):
    """Handles cases where CHAR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.string_array_kernels.char_util", ["arr"], 0
        )

    def impl(arr):  # pragma: no cover
        return char_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def contains(arr, pattern, dict_encoding_state=None, func_id=-1):
    """Handles cases where CONTAINS receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.contains",
                ["arr", "pattern", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, pattern, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return contains_util(arr, pattern, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def contains_util(arr, pattern, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function CONTAINS which takes in two strings/string columns
    and returns a Boolean in regards to whether or not the second string exists in the first

    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        pattern (string array): string(s) to be matched

    Returns:
        Boolean array/scalar: the scalar/column of Boolean results
    """

    verify_string_binary_arg(arr, "CONTAINS", "arr")
    verify_string_binary_arg(pattern, "CONTAINS", "pattern")

    out_dtype = bodo.boolean_array_type
    arg_names = ["arr", "pattern", "dict_encoding_state", "func_id"]
    arg_types = [arr, pattern, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "res[i] = arg1 in arg0\n"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def editdistance_no_max(s, t, dict_encoding_state=None, func_id=-1):
    """Handles cases where EDITDISTANCE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [s, t]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.editdistance_no_max",
                ["s", "t", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(s, t, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return editdistance_no_max_util(s, t, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def jarowinkler_similarity(s, t, dict_encoding_state=None, func_id=-1):
    """Handles cases where JAROWINKLER_SIMILARITY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [s, t]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.jarowinkler_similarity",
                ["s", "t", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(s, t, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return jarowinkler_similarity_util(s, t, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def editdistance_with_max(s, t, maxDistance, dict_encoding_state=None, func_id=-1):
    """Handles cases where EDITDISTANCE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [s, t, maxDistance]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.editdistance_with_max",
                ["s", "t", "maxDistance", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        s, t, maxDistance, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return editdistance_with_max_util(
            s, t, maxDistance, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True)
def endswith(source, suffix, dict_encoding_state=None, func_id=-1):
    """Handles cases where ENDSWITH receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, suffix]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.endswith",
                ["source", "suffix", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(source, suffix, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return endswith_util(source, suffix, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def format(arr, places):
    """Handles cases where FORMAT receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, places]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument("bodosql.kernels.format", ["arr", "places"], i)

    def impl(arr, places):  # pragma: no cover
        return format_util(arr, places)

    return impl


@numba.generated_jit(nopython=True)
def initcap(arr, delim, dict_encoding_state=None, func_id=-1):
    """Handles cases where INITCAP receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, delim]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.initcap",
                ["arr", "delim", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, delim, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return initcap_util(arr, delim, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def insert(source, pos, length, inject, dict_encoding_state=None, func_id=-1):
    """Handles cases where INSERT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, pos, length, inject]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.insert",
                ["source", "pos", "length", "inject", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        source, pos, length, inject, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return insert_util(source, pos, length, inject, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def instr(arr, target, dict_encoding_state=None, func_id=-1):
    """Handles cases where INSTR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, target]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.instr",
                ["arr", "target", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, target, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return instr_util(arr, target, dict_encoding_state, func_id)

    return impl


def left(arr, n_chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(left)
def overload_left(arr, n_chars, dict_encoding_state=None, func_id=-1):
    """Handles cases where LEFT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, n_chars]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.left",
                ["arr", "n_chars", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, n_chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return left_util(arr, n_chars, dict_encoding_state, func_id)

    return impl


def lpad(arr, length, padstr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(lpad)
def overload_lpad(arr, length, padstr, dict_encoding_state=None, func_id=-1):
    """Handles cases where LPAD receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, length, padstr]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.lpad",
                ["arr", "length", "padstr", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, length, padstr, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return lpad_util(arr, length, padstr, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def ord_ascii(arr, dict_encoding_state=None, func_id=-1):
    """Handles cases where ORD/ASCII receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.string_array_kernels.ord_ascii_util",
            ["arr", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return ord_ascii_util(arr, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def position(substr, source, start, dict_encoding_state=None, func_id=-1):
    """Handles cases where POSITION receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [substr, source, start]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.position",
                ["substr", "source", "start", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        substr, source, start, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return position_util(substr, source, start, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def repeat(arr, repeats, dict_encoding_state=None, func_id=-1):
    """Handles cases where REPEAT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, repeats]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.repeat",
                ["arr", "repeats", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, repeats, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return repeat_util(arr, repeats, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def replace(arr, to_replace, replace_with, dict_encoding_state=None, func_id=-1):
    """Handles cases where REPLACE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, to_replace, replace_with]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.replace",
                ["arr", "to_replace", "replace_with", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, to_replace, replace_with, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return replace_util(arr, to_replace, replace_with, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def reverse(arr, dict_encoding_state=None, func_id=-1):
    """Handles cases where REVERSE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.string_array_kernels.reverse_util",
            ["arr", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return reverse_util(arr, dict_encoding_state, func_id)

    return impl


def right(arr, n_chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(right)
def overload_right(arr, n_chars, dict_encoding_state=None, func_id=-1):
    """Handles cases where RIGHT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, n_chars]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.right",
                ["arr", "n_chars", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, n_chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return right_util(arr, n_chars, dict_encoding_state, func_id)

    return impl


def rpad(arr, length, padstr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(rpad)
def overload_rpad(arr, length, padstr, dict_encoding_state=None, func_id=-1):
    """Handles cases where RPAD receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr, length, padstr]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.rpad",
                ["arr", "length", "padstr", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, length, padstr, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return rpad_util(arr, length, padstr, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def rtrimmed_length(arr, dict_encoding_state=None, func_id=-1):
    """Handles cases where RTRIMED_LENGTH receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.string_array_kernels.rtrimmed_length_util",
            ["arr", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return rtrimmed_length_util(arr, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def space(n_chars, dict_encoding_state=None, func_id=-1):
    """Handles cases where SPACE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(n_chars, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.string_array_kernels.space_util",
            ["n_chars", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(n_chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return space_util(n_chars, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def split_part(source, delim, part, dict_encoding_state=None, func_id=-1):
    """Handles cases where SPLIT_PART receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, delim, part]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.split_part",
                ["source", "delim", "part", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        source, delim, part, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return split_part_util(source, delim, part, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def startswith(source, prefix, dict_encoding_state=None, func_id=-1):
    """Handles cases where STARTSWITH receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, prefix]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.startswith",
                ["source", "prefix", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(source, prefix, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return startswith_util(source, prefix, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def strcmp(arr0, arr1, dict_encoding_state=None, func_id=-1):
    """Handles cases where STRCMP receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [arr0, arr1]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.strcmp",
                ["arr0", "arr1", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr0, arr1, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return strcmp_util(arr0, arr1, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def strtok(source, delim, part, dict_encoding_state=None, func_id=-1):
    """Handles cases where STRTOK receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [source, delim, part]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.strtok",
                ["source", "delim", "part", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        source, delim, part, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return strtok_util(source, delim, part, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def substring(arr, start, length, dict_encoding_state=None, func_id=-1):
    """Handles cases where SUBSTRING receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, start, length]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.substring",
                ["arr", "start", "length", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, start, length, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return substring_util(arr, start, length, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def substring_suffix(
    arr, start, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    """Handles cases where SUBSTR/SUBSTRING receives two [optional] arguments only and forwards
    to args appropriate version of the real implementation

    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        start (integer array/series/scalar): the starting location(s) of the substring(s)

    Returns:
        string array/scalar: the string/column of extracted substrings
    """
    args = [arr, start]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.substring_suffix",
                ["arr", "start", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(arr, start, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return substring_suffix_util(arr, start, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def substring_index(arr, delimiter, occurrences, dict_encoding_state=None, func_id=-1):
    """Handles cases where SUBSTRING_INDEX receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, delimiter, occurrences]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.substring_index",
                ["arr", "delimiter", "occurrences", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, delimiter, occurrences, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return substring_index_util(
            arr, delimiter, occurrences, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True)
def translate(arr, source, target, dict_encoding_state=None, func_id=-1):
    """Handles cases where TRANSLATE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, source, target]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.translate",
                ["arr", "source", "target", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, source, target, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return translate_util(arr, source, target, dict_encoding_state, func_id)

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
def initcap_util(arr, delim, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function INITCAP which takes in a source
    string (or column) and a delimiter string (or column) capitalizes the first
    character and every character after the characters in the delimiter string.


    Args:
        arr (string array/series/scalar): the string(s) being capitalized
        delim (string array/series/scalar): the delimiter string(s) to capitalize
        after

    Returns:
        string array/scalar: the capitalized string(s)
    """

    verify_string_arg(arr, "INITCAP", "arr")
    verify_string_arg(delim, "INITCAP", "delim")

    arg_names = ["arr", "delim", "dict_encoding_state", "func_id"]
    arg_types = [arr, delim, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "capitalized = arg0[:1].upper()\n"
    scalar_text += "for j in range(1, len(arg0)):\n"
    scalar_text += "   if arg0[j-1] in arg1:\n"
    scalar_text += "      capitalized += arg0[j].upper()\n"
    scalar_text += "   else:\n"
    scalar_text += "      capitalized += arg0[j].lower()\n"
    scalar_text += "res[i] = capitalized"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def instr_util(arr, target, dict_encoding_state, func_id):
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

    arg_names = ["arr", "target", "dict_encoding_state", "func_id"]
    arg_types = [arr, target, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "res[i] = arg0.find(arg1) + 1"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


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
    when provided with a maximum distance. This is separate from
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
def editdistance_no_max_util(s, t, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function EDITDISTANCE which two strings
    (or columns) and returns the minimum edit distance between them (i.e. the
    smallest number of insertions/deletions/replacements required to make the
    two strings identical)


    Args:
        s (string array/series/scalar): the first string(s) being compared
        t (string array/series/scalar): the second string(s) being compared

    Returns:
        int series/scalar: the minimum edit distance between the two strings
    """

    verify_string_arg(s, "editdistance_no_max", "s")
    verify_string_arg(t, "editdistance_no_max", "t")

    arg_names = ["s", "t", "dict_encoding_state", "func_id"]
    arg_types = [s, t, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = (
        "res[i] = bodosql.kernels.string_array_kernels.min_edit_distance(arg0, arg1)"
    )

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def editdistance_with_max_util(s, t, maxDistance, dict_encoding_state, func_id):
    """Same as editdistance_no_max_util, except it supports the version with
    the third argument for the maximum distance to search before giving up.


    Args:
        s (string array/series/scalar): the first string(s) being compared
        t (string array/series/scalar): the second string(s) being compared
        maxDistance (int array/series/scalar): the distance(s) to search before giving up

    Returns:
        int series/scalar: the minimum edit distance between the two strings.
        if it is greater than maxDistance, then maxDistance is returned. If
        maxDistance is negative, 0 is returned.
    """

    verify_string_arg(s, "editdistance_no_max", "s")
    verify_string_arg(t, "editdistance_no_max", "t")
    verify_int_arg(maxDistance, "editdistance_no_max", "t")

    arg_names = ["s", "t", "maxDistance", "dict_encoding_state", "func_id"]
    arg_types = [s, t, maxDistance, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    scalar_text = "res[i] = bodosql.kernels.string_array_kernels.min_edit_distance_with_max(arg0, arg1, arg2)"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@register_jitable
def jarowinkler_similarity_fn(s, t):  # pragma: no cover
    """
    Implementation of Jaro-Winkler similarity defined by the following formula:

        len = max(|s|, |t|)
        d = len // 2 - 1
        m = number of matches between s & t (same character, at most d apart)
        t = number of transpositions (matches out of order)
        jaro_similarity = (m / |s| + m / |t| + (m - t) / m) / 3
        l = length of the longest common prefix of s and t
        jarowinkler_similarity = jaro_similarity + 0.1 * l * (1 - jarowinkler_similarity)

    Using the algorithm as DuckDB:
    https://github.com/duckdb/duckdb/blob/main/third_party/jaro_winkler/details/jaro_impl.hpp

    Structured more closely to this Geeks for Geeks implementation:
    https://www.geeksforgeeks.org/jaro-and-jaro-winkler-similarity/
    """
    # If either string is empty, the answer is automatically zero
    if min(len(s), len(t)) == 0:
        return np.int8(0)

    # For simplicity, make all characters the same case, and have s be the longer string.
    s, t = s.lower(), t.lower()
    if len(s) < len(t):
        s, t = t, s

    # Calculate the maximum matching distance
    s_len, t_len = len(s), len(t)
    d = max(s_len, t_len) // 2 - 1
    matches = 0

    # Find all of the indices in s and t that have a match in the other
    s_matches = np.full(s_len, False, dtype=np.bool_)
    t_matches = np.full(t_len, False, dtype=np.bool_)
    for s_idx in range(s_len):
        for t_idx in range(max(0, s_idx - d), min(t_len, s_idx + d + 1)):
            if s[s_idx] == t[t_idx] and not t_matches[t_idx]:
                s_matches[s_idx] = True
                t_matches[t_idx] = True
                matches += 1
                break

    # If we found no matches, stop now because we already know the answer will be zero
    if matches == 0:
        return np.int8(0)

    # Calculate how many of the matches are also transpositions
    transpositions = 0
    finger = 0
    for s_idx in range(s_len):
        if s_matches[s_idx]:
            while not t_matches[finger]:
                finger += 1
            transpositions += s[s_idx] != t[finger]
            finger += 1
    transpositions //= 2

    # Calculate the Jaro similarity
    j_similarity = (
        (matches / s_len) + (matches / t_len) + (matches - transpositions) / matches
    ) / 3.0

    # Don't use the Winkler boost if Jaro similarity is below 0.7
    if j_similarity < 0.7:
        return np.int8(100 * j_similarity)

    # Find the longest common prefix (at most 4 chars)
    common_prefix = 0
    for i in range(min(4, s_len, t_len)):
        if s[i] != t[i]:
            break
        common_prefix += 1

    # Calculate the Jaro-Winkler similarity and convert to an integer between 0 and 100
    scaling_factor = 0.1
    jw_similarity = j_similarity + scaling_factor * common_prefix * (1 - j_similarity)
    return np.int8(min(jw_similarity * 100, 100))


@numba.generated_jit(nopython=True)
def jarowinkler_similarity_util(s, t, dict_encoding_state, func_id):
    """A dedicated kernel for the Snowflake SQL function JAROWINKLER_SIMILARITY
    which takes in two strings (or columns of strings) and returns the Jaro-Winkler
    similarity between them.

    Args:
        s (string array/series/scalar): the first string(s) being compared
        t (string array/series/scalar): the second string(s) being compared

    Returns:
        int series/scalar: the jarowinkler similarity of the two strings as an integer
        between 0 and 100 (rounded down).
    """

    verify_string_arg(s, "jarowinkler_similarity", "s")
    verify_string_arg(t, "jarowinkler_similarity", "t")

    arg_names = ["s", "t", "dict_encoding_state", "func_id"]
    arg_types = [s, t, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "res[i] = bodosql.kernels.string_array_kernels.jarowinkler_similarity_fn(arg0, arg1)"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int8)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def endswith_util(source, suffix, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function ENDSWITH which takes in 2 strings
    (or string columns) and whether or not the first string ends with the second


    Args:
        source (string array/series/scalar): the string(s) being searched in
        suffix (string array/series/scalar): the string(s) being searched for

    Returns:
        boolean series/scalar: whether or not the source contains the suffix
    """

    arr_is_string = verify_string_binary_arg(source, "endswith", "source")
    if arr_is_string != verify_string_binary_arg(suffix, "endswith", "suffix") and not (
        is_overload_none(source) or is_overload_none(suffix)
    ):  # pragma: no cover
        raise BodoError("String and suffix must both be strings or both binary")

    arg_names = ["source", "suffix", "dict_encoding_state", "func_id"]
    arg_types = [source, suffix, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "res[i] = arg0.endswith(arg1)"

    out_dtype = bodo.boolean_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


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


# TODO: alter to handle negatives the same way Snowflake does ([BE-3719])
@numba.generated_jit(nopython=True)
def insert_util(arr, pos, length, inject, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function INSERT which takes in two strings
       and two integers (either of which can be a column) and inserts the second
       string inside the first, starting at the index of the first integer
       and replacing the number of characters from the second integer.

       Currently diverges from the Snowflake API when pos/length are negative.
       The Snowflake behavior for these cases seems hard to define, so for
       now this kernel treats negatives like zero.


    Args:
        arr (string array/series/scalar): the string(s) being inserted into
        pos (integer array/series/scalar): where the insert(s) begin (1 indexed)
        length (integer array/series/scalar): how many characters from arr get deleted
        inject (string array/series/scalar): the string(s) being inserted

    Returns:
        string array/scalar: the injected string(s)
    """
    arr_is_string = verify_string_binary_arg(arr, "INSERT", "arr")
    verify_int_arg(pos, "INSERT", "pos")
    verify_int_arg(length, "INSERT", "length")
    if arr_is_string != verify_string_binary_arg(inject, "INSERT", "inject") and not (
        is_overload_none(arr) or is_overload_none(inject)
    ):  # pragma: no cover
        raise BodoError("String and injected value must both be strings or both binary")

    arg_names = ["arr", "pos", "length", "inject", "dict_encoding_state", "func_id"]
    arg_types = [arr, pos, length, inject, dict_encoding_state, func_id]
    propagate_null = [True] * 4 + [False] * 2

    # Assertions create control flow with raise nodes, so we
    # raise runtime errors in a helper function.
    scalar_text = "bodosql.kernels.array_kernel_utils.check_insert_args(arg1, arg2)\n"
    scalar_text += "prefixIndex = max(arg1-1, 0)\n"
    scalar_text += "suffixIndex = prefixIndex + max(arg2, 0)\n"
    scalar_text += "res[i] = arg0[:prefixIndex] + arg3 + arg0[suffixIndex:]"

    out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


def left_util(arr, n_chars, dict_encoding_state, func_id):  # pragma: no cover
    # Dummy function used for overload
    return


def right_util(arr, n_chars, dict_encoding_state, func_id):  # pragma: no cover
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

    def overload_left_right_util(arr, n_chars, dict_encoding_state, func_id):
        arr_is_string = verify_string_binary_arg(arr, func_name, "arr")
        verify_int_arg(n_chars, func_name, "n_chars")

        empty_char = "''" if arr_is_string else "b''"

        arg_names = ["arr", "n_chars", "dict_encoding_state", "func_id"]
        arg_types = [arr, n_chars, dict_encoding_state, func_id]
        propagate_null = [True] * 2 + [False] * 2
        scalar_text = "if arg1 <= 0:\n"
        scalar_text += f"   res[i] = {empty_char}\n"
        scalar_text += "else:\n"
        if func_name == "LEFT":
            scalar_text += "   res[i] = arg0[:arg1]\n"
        elif func_name == "RIGHT":
            scalar_text += "   res[i] = arg0[-arg1:]\n"

        out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            may_cause_duplicate_dict_array_values=True,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_left_right_util


def _install_left_right_overload():
    """Creates and installs the overloads for left_util and right_util"""
    for func, func_name in zip((left_util, right_util), ("LEFT", "RIGHT")):
        overload_impl = create_left_right_util_overload(func_name)
        overload(func)(overload_impl)


_install_left_right_overload()


def lpad_util(arr, length, padstr, dict_encoding_state, func_id):  # pragma: no cover
    # Dummy function used for overload
    return


def rpad_util(arr, length, padstr, dict_encoding_state, func_id):  # pragma: no cover
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

    def overload_lpad_rpad_util(arr, length, pad_string, dict_encoding_state, func_id):
        pad_is_string = verify_string_binary_arg(pad_string, func_name, "pad_string")
        arr_is_string = verify_string_binary_arg(arr, func_name, "arr")
        if arr_is_string != pad_is_string and not (
            is_overload_none(pad_string) or is_overload_none(arr)
        ):
            raise BodoError("Pad string and arr must be the same type!")

        out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

        verify_int_arg(length, func_name, "length")
        verify_string_binary_arg(pad_string, func_name, f"{func_name.lower()}_string")

        if func_name == "LPAD":
            pad_line = "(arg2 * quotient) + arg2[:remainder] + arg0"
        elif func_name == "RPAD":
            pad_line = "arg0 + (arg2 * quotient) + arg2[:remainder]"

        arg_names = ["arr", "length", "pad_string", "dict_encoding_state", "func_id"]
        arg_types = [arr, length, pad_string, dict_encoding_state, func_id]
        propagate_null = [True] * 3 + [False] * 2

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

        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            may_cause_duplicate_dict_array_values=True,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_lpad_rpad_util


def _install_lpad_rpad_overload():
    """Creates and installs the overloads for lpad_util and rpad_util"""
    for func, func_name in zip((lpad_util, rpad_util), ("LPAD", "RPAD")):
        overload_impl = create_lpad_rpad_util_overload(func_name)
        overload(func)(overload_impl)


_install_lpad_rpad_overload()


@numba.generated_jit(nopython=True)
def ord_ascii_util(arr, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function ORD/ASCII which takes in a string
       (or string column) and returns the ord value of the first character


    Args:
        arr (string array/series/scalar): the string(s) whose ord value(s) are
        being calculated

    Returns:
        integer series/scalar: the ord value of the first character(s)
    """

    verify_string_arg(arr, "ORD", "arr")

    arg_names = ["arr", "dict_encoding_state", "func_id"]
    arg_types = [arr, dict_encoding_state, func_id]
    propagate_null = [True] + [False] * 2
    scalar_text = "if len(arg0) == 0:\n"
    scalar_text += "   res[i] = 0\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = ord(arg0[0])"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def position_util(substr, source, start, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function POSITION which takes in two strings
    (or columns) and returns the location of the first string within the second
    (1-indexed), with an optional starting location. If no match is found,
    zero is returned. The actual function can take in 2 or 3 arguments, but
    this kernel assumes that the 3rd argument is always provided.


    Args:
        substr (string array/series/scalar): the string(s) that are being searched for
        source (string array/series/scalar): the string(s) that are being searched in
        start (int array/series/scalar): the start location(s) for the search

    Returns:
        integer series/scalar: the ord value of the first character(s)
    """

    is_str = verify_string_binary_arg(substr, "POSITION", "substr")
    if is_str != verify_string_binary_arg(source, "POSITION", "source") and not (
        is_overload_none(substr) or is_overload_none(source)
    ):  # pragma: no cover
        raise BodoError("Substring and source must be both strings or both binary")
    verify_int_arg(start, "POSITION", "start")

    arg_names = ["substr", "source", "start", "dict_encoding_state", "func_id"]
    arg_types = [substr, source, start, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2

    if is_str:
        scalar_text = "res[i] = arg1.find(arg0, arg2 - 1) + 1"
    else:
        scalar_text = "res[i] = arg1._to_str().find(arg0._to_str(), arg2 - 1) + 1"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def repeat_util(arr, repeats, dict_encoding_state, func_id):
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

    arg_names = ["arr", "repeats", "dict_encoding_state", "func_id"]
    arg_types = [arr, repeats, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "if arg1 <= 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg0 * arg1"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    # NOTE: we can cause duplicate values in the case that repeats == 0 (everything goes to empty str)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def replace_util(arr, to_replace, replace_with, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function REVERSE which takes in a base string
       (or string column), a second string to locate in the base string, and a
       third string with which to replace it.


    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        to_replace (string array/series/scalar): the substring(s) to replace
        replace_with (string array/series/scalar): the string(s) that replace to_replace

    Returns:
        string array/scalar: the string/column where each occurrence of
        to_replace has been replaced by replace_with
    """

    verify_string_arg(arr, "REPLACE", "arr")
    verify_string_arg(to_replace, "REPLACE", "to_replace")
    verify_string_arg(replace_with, "REPLACE", "replace_with")

    arg_names = ["arr", "to_replace", "replace_with", "dict_encoding_state", "func_id"]
    arg_types = [arr, to_replace, replace_with, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    scalar_text = "if arg1 == '':\n"
    scalar_text += "   res[i] = arg0\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg0.replace(arg1, arg2)"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def reverse_util(arr, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function REVERSE which takes in a string
       (or string column) and reverses it

    Args:
        arr (string array/series/scalar): the strings(s) to be reversed

    Returns:
        string array/scalar: the string/column that has been reversed
    """

    arr_is_string = verify_string_binary_arg(arr, "REVERSE", "arr")

    arg_names = ["arr", "dict_encoding_state", "func_id"]
    arg_types = [arr, dict_encoding_state, func_id]
    propagate_null = [True] + [False] * 2
    scalar_text = "res[i] = arg0[::-1]"

    out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def rtrimmed_length_util(arr, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function RTRIMMED_LENGTH which takes in a
       string (or string column) and returns the number of characters after
       trailing whitespace has been removed


    Args:
        arr (string array/series/scalar): the strings(s) to have their
        post-trimming lengths calculated

    Returns:
        integer array/scalar: the number of characters in the string(s) after
        trailing whitespaces are removed
    """

    verify_string_arg(arr, "RTRIMMED_LENGTH", "arr")

    arg_names = ["arr", "dict_encoding_state", "func_id"]
    arg_types = [arr, dict_encoding_state, func_id]
    propagate_null = [True] + [False] * 2
    scalar_text = "res[i] = len(arg0.rstrip(' '))"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def space_util(n_chars, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function SPACE which takes in an integer
       (or integer column) and returns that many spaces


    Args:
        n_chars (integer array/series/scalar): the number(s) of spaces

    Returns:
        string array/scalar: the string/column of spaces
    """

    verify_int_arg(n_chars, "SPACE", "n_chars")

    arg_names = ["n_chars", "dict_encoding_state", "func_id"]
    arg_types = [n_chars, dict_encoding_state, func_id]
    propagate_null = [True] + [False] * 2
    scalar_text = "if arg0 <= 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = ' ' * arg0"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def split_part_util(source, delim, part, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function SPLIT_PART which takes in a
    source string (or column), a delimiter string (or column), and a part
    integer (or column), then splits the source string by occurrences of
    the entire delimiter string and outputs the value specified by the part.
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

    arg_names = ["source", "delim", "part", "dict_encoding_state", "func_id"]
    arg_types = [source, delim, part, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    # Splitting by '' is valid in SQL, but not in Python
    scalar_text = "tokens = arg0.split(arg1) if arg1 != '' else [arg0]\n"
    scalar_text += "if abs(arg2) > len(tokens):\n"
    scalar_text += "    res[i] = ''\n"
    scalar_text += "else:\n"
    scalar_text += "    res[i] = tokens[arg2 if arg2 <= 0 else arg2-1]\n"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def startswith_util(source, prefix, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function STARTSWITH which takes in 2 strings
    (or string columns) and whether or not the first string starts with the second


    Args:
        source (string array/series/scalar): the string(s) being searched in
        prefix (string array/series/scalar): the string(s) being searched for

    Returns:
        boolean series/scalar: whether or not the source contains the prefix
    """

    arr_is_string = verify_string_binary_arg(source, "startswith", "source")
    if arr_is_string != verify_string_binary_arg(
        prefix, "startswith", "prefix"
    ) and not (
        is_overload_none(source) or is_overload_none(prefix)
    ):  # pragma: no cover
        raise BodoError("String and prefix must both be strings or both binary")

    arg_names = ["source", "prefix", "dict_encoding_state", "func_id"]
    arg_types = [source, prefix, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "res[i] = arg0.startswith(arg1)"

    out_dtype = bodo.boolean_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def strcmp_util(arr0, arr1, dict_encoding_state, func_id):
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

    arg_names = ["arr0", "arr1", "dict_encoding_state", "func_id"]
    arg_types = [arr0, arr1, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "if arg0 < arg1:\n"
    scalar_text += "   res[i] = -1\n"
    scalar_text += "elif arg0 > arg1:\n"
    scalar_text += "   res[i] = 1\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = 0\n"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def strtok_util(source, delim, part, dict_encoding_state, func_id):
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

    arg_names = ["source", "delim", "part", "dict_encoding_state", "func_id"]
    arg_types = [source, delim, part, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
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

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def strtok_to_array(
    source, delim, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    """Handles cases where STRTOK_TO_ARRAY receives optional arguments and
    forwards to the appropriate version of the real implementation"""
    args = [source, delim]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):
            return unopt_argument(
                "bodosql.kernels.strtok_to_array",
                ["source", "delim", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(source, delim, dict_encoding_state=None, func_id=-1):
        return strtok_to_array_util(source, delim, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def strtok_to_array_util(
    source, delim, dict_encoding_state, func_id
):  # pragma: no cover
    """A dedicated kernel for the SQL function STRTOK_TO_ARRAY which tokenizes the given string using the given set of delimiters and returns the tokens as an array.

    - If delim is an empty string, return source.
    - If source is empty, return an empty array.

    Args:
        source (string array/series/scalar): the string(s) to be parsed
        delim (string array/series/scalar): the string(s) to split on

    Returns:
        An ArrayItemArray of string array: the tokenized string arrays
    """

    verify_string_arg(source, "STRTOK", "source")
    verify_string_arg(delim, "STRTOK", "delim")
    arg_names = ["source", "delim", "dict_encoding_state", "func_id"]
    arg_types = [source, delim, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    out_dtype = bodo.ArrayItemArrayType(bodo.string_array_type)

    scalar_text = "tokens = []\n"
    scalar_text += "buffer = ''\n"
    scalar_text += "for j in range(len(arg0)):\n"
    scalar_text += "   if arg0[j] in arg1:\n"
    scalar_text += "      if buffer != '':"
    scalar_text += "         tokens.append(buffer)\n"
    scalar_text += "      buffer = ''\n"
    scalar_text += "   else:\n"
    scalar_text += "      buffer += arg0[j]\n"
    scalar_text += "if buffer != '':\n"
    scalar_text += "   tokens.append(buffer)\n"
    scalar_text += "res[i] = bodo.libs.str_arr_ext.str_list_to_array(tokens)"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def substring_util(arr, start, length, dict_encoding_state, func_id):
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

    arg_names = ["arr", "start", "length", "dict_encoding_state", "func_id"]
    arg_types = [arr, start, length, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    scalar_text = "if arg2 <= 0:\n"
    scalar_text += "   res[i] = ''\n" if arr_is_string else "   res[i] = b''\n"
    scalar_text += "elif arg1 < 0 and arg1 + arg2 >= 0:\n"
    scalar_text += "   res[i] = arg0[arg1:]\n"
    scalar_text += "else:\n"
    scalar_text += "   if arg1 > 0: arg1 -= 1\n"
    scalar_text += "   res[i] = arg0[arg1:arg1+arg2]\n"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def substring_suffix_util(arr, start, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function SUBSTR/SUBSTRING which takes in a string,
       (or string column), and one integer (or integer columns) and returns
       the string starting from the index of the first integer.

    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        start (integer array/series/scalar): the starting location(s) of the substring(s)

    Returns:
        string array/scalar: the string/column of extracted substrings
    """

    arr_is_string = verify_string_binary_arg(arr, "SUBSTRING", "arr")
    verify_int_arg(start, "SUBSTRING", "start")

    out_dtype = bodo.string_array_type if arr_is_string else bodo.binary_array_type

    arg_names = ["arr", "start", "dict_encoding_state", "func_id"]
    arg_types = [arr, start, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "  if arg1 > 0: arg1 -= 1\n"
    scalar_text += "  res[i] = arg0[arg1:]\n"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def substring_index_util(arr, delimiter, occurrences, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function SUBSTRING_INDEX which takes in a
       string, (or string column), a delimiter string (or string column) and an
       occurrences integer (or integer column) and returns the prefix of the
       first string before that number of occurrences of the delimiter


    Args:
        arr (string array/series/scalar): the strings(s) to be modified
        delimiter (string array/series/scalar): the delimiter(s) to look for
        occurrences (integer array/series/scalar): how many of the delimiter to look for

    Returns:
        string array/scalar: the string/column of prefixes before occurrences
        many of the delimiter string occur
    """

    verify_string_arg(arr, "SUBSTRING_INDEX", "arr")
    verify_string_arg(delimiter, "SUBSTRING_INDEX", "delimiter")
    verify_int_arg(occurrences, "SUBSTRING_INDEX", "occurrences")

    arg_names = ["arr", "delimiter", "occurrences", "dict_encoding_state", "func_id"]
    arg_types = [arr, delimiter, occurrences, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    scalar_text = "if arg1 == '' or arg2 == 0:\n"
    scalar_text += "   res[i] = ''\n"
    scalar_text += "elif arg2 >= 0:\n"
    scalar_text += "   res[i] = arg1.join(arg0.split(arg1, arg2+1)[:arg2])\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg1.join(arg0.split(arg1)[arg2:])\n"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def translate_util(arr, source, target, dict_encoding_state, func_id):
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

    arg_names = ["arr", "source", "target", "dict_encoding_state", "func_id"]
    arg_types = [arr, source, target, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    scalar_text = "translated = ''\n"
    scalar_text += "for char in arg0:\n"
    scalar_text += "   index = arg1.find(char)\n"
    scalar_text += "   if index == -1:\n"
    scalar_text += "      translated += char\n"
    scalar_text += "   elif index < len(arg2):\n"
    scalar_text += "      translated += arg2[index]\n"
    scalar_text += "res[i] = translated"

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        may_cause_duplicate_dict_array_values=True,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


def length(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def lower(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def upper(arr, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def trim(source, chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def ltrim(source, chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def rtrim(source, chars, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    pass


def length_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    pass


def lower_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    pass


def upper_util(arr, dict_encoding_state, func_id):  # pragma: no cover
    pass


def trim_util(source, chars, dict_encoding_state, func_id):  # pragma: no cover
    pass


def ltrim_util(source, chars, dict_encoding_state, func_id):  # pragma: no cover
    pass


def rtrim_util(source, chars, dict_encoding_state, func_id):  # pragma: no cover
    pass


def create_trim_fn_overload(fn_name):
    def overload_func(source, chars, dict_encoding_state=None, func_id=-1):
        """Handles cases where this one argument string function receives optional
        arguments and forwards to the appropriate version of the real implementation"""
        args = [source, chars]
        for i, arg in enumerate(args):
            if isinstance(arg, types.optional):  # pragma: no cover
                return unopt_argument(
                    f"bodosql.kernels.{fn_name}",
                    ["source", "chars", "dict_encoding_state", "func_id"],
                    i,
                    default_map={"dict_encoding_state": None, "func_id": -1},
                )

        func_text = "def impl(source, chars, dict_encoding_state=None, func_id=-1):\n"
        func_text += f"  return bodosql.kernels.string_array_kernels.{fn_name}_util(source, chars, dict_encoding_state, func_id)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_trim_fn_util_overload(fn_name):
    """Creates an overload function to support one argument string
    functions.

    Args:
        fn_name: the function being implemented

    Returns:
        (function): a utility that takes in a string (either can be scalars
        or vectors) and returns the corresponding component based on the desired
        function.
    """

    def overload_trim_fn(
        source, chars, dict_encoding_state, func_id
    ):  # pragma: no cover
        verify_string_arg(source, fn_name, "source")
        verify_string_arg(chars, fn_name, "chars")
        arg_names = ["source", "chars", "dict_encoding_state", "func_id"]
        arg_types = [source, chars, dict_encoding_state, func_id]
        propagate_null = [True] * 2 + [False] * 2
        if fn_name == "ltrim":
            scalar_text = "res[i] = arg0.lstrip(arg1)\n"
        elif fn_name == "rtrim":
            scalar_text = "res[i] = arg0.rstrip(arg1)\n"
        else:
            scalar_text = "res[i] = arg0.strip(arg1)\n"

        out_dtype = bodo.string_array_type

        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_trim_fn


def _install_trim_fn_overloads():
    """Creates and installs the overloads for one argument string functions"""
    funcs_utils_names = [
        ("trim", trim, trim_util),
        ("ltrim", ltrim, ltrim_util),
        ("rtrim", rtrim, rtrim_util),
    ]
    for fn_name, func, util in funcs_utils_names:
        func_overload_impl = create_trim_fn_overload(fn_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_trim_fn_util_overload(fn_name)
        overload(util)(util_overload_impl)


_install_trim_fn_overloads()


def create_one_arg_str_fn_overload(fn_name):
    def overload_func(arr, dict_encoding_state=None, func_id=-1):
        """Handles cases where this one argument string function receives optional
        arguments and forwards to the appropriate version of the real implementation"""
        if isinstance(arr, types.optional):  # pragma: no cover
            return unopt_argument(
                f"bodosql.kernels.string_array_kernels.{fn_name}_util",
                ["arr", "dict_encoding_state", "func_id"],
                0,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

        func_text = "def impl(arr, dict_encoding_state=None, func_id=-1):\n"
        func_text += f"  return bodosql.kernels.string_array_kernels.{fn_name}_util(arr, dict_encoding_state, func_id)"
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "bodosql": bodosql}, loc_vars)

        return loc_vars["impl"]

    return overload_func


def create_one_arg_str_fn_util_overload(fn_name):
    """Creates an overload function to support one argument string
    functions.

    Args:
        fn_name: the function being implemented

    Returns:
        (function): a utility that takes in a string (either can be scalars
        or vectors) and returns the corresponding component based on the desired
        function.
    """

    def overload_one_arg_str_fn(arr, dict_encoding_state, func_id):  # pragma: no cover
        if fn_name == "length":
            # Length also supports binary data.
            verify_string_binary_arg(arr, fn_name, "arr")
            out_dtype = bodo.IntegerArrayType(types.int64)
            may_cause_duplicate_dict_array_values = False
            fn_call = "len(arg0)"
        else:
            verify_string_arg(arr, fn_name, "arr")
            out_dtype = bodo.string_array_type
            may_cause_duplicate_dict_array_values = True
            fn_call = f"arg0.{fn_name}()"

        arg_names = ["arr", "dict_encoding_state", "func_id"]
        arg_types = [arr, dict_encoding_state, func_id]
        propagate_null = [True] + [False] * 2
        scalar_text = f"res[i] = {fn_call}"

        use_dict_caching = not is_overload_none(dict_encoding_state)
        return gen_vectorized(
            arg_names,
            arg_types,
            propagate_null,
            scalar_text,
            out_dtype,
            may_cause_duplicate_dict_array_values=may_cause_duplicate_dict_array_values,
            # Add support for dict encoding caching with streaming.
            dict_encoding_state_name="dict_encoding_state"
            if use_dict_caching
            else None,
            func_id_name="func_id" if use_dict_caching else None,
        )

    return overload_one_arg_str_fn


def _install_one_arg_str_fn_overloads():
    """Creates and installs the overloads for one argument string functions"""
    funcs_utils_names = [
        ("length", length, length_util),
        ("lower", lower, lower_util),
        ("upper", upper, upper_util),
    ]
    for fn_name, func, util in funcs_utils_names:
        func_overload_impl = create_one_arg_str_fn_overload(fn_name)
        overload(func)(func_overload_impl)
        util_overload_impl = create_one_arg_str_fn_util_overload(fn_name)
        overload(util)(util_overload_impl)


_install_one_arg_str_fn_overloads()


@numba.generated_jit(nopython=True)
def split(string, separator, dict_encoding_state=None, func_id=-1):  # pragma: no cover
    """Handles cases where SPLIT receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [string, separator]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.split",
                ["string", "separator", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        string, separator, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return split_util(string, separator, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def split_util(string, separator, dict_encoding_state, func_id):  # pragma: no cover
    """A dedicated kernel for the SQL function SPLIT which takes in a
           string, (or string column) and a separator string (or string column) ans
           returns the result strings in arrays

    Args:
        string (string array/series/scalar): the strings(s) to be split
        separator (string array/series/scalar): the separator to split the string

    Returns:
        An ArrayItemArray of string array: the result string arrays
    """

    verify_string_arg(string, "SPLIT", "string")
    verify_string_arg(separator, "SPLIT", "separator")
    arg_names = ["string", "separator", "dict_encoding_state", "func_id"]
    arg_types = [string, separator, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    out_dtype = bodo.ArrayItemArrayType(bodo.string_array_type)
    scalar_text = "if arg1 == '':\n"
    scalar_text += "    str_list = [arg0]\n"
    scalar_text += "else:\n"
    scalar_text += "    str_list = arg0.split(arg1)\n"
    scalar_text += "res[i] = bodo.libs.str_arr_ext.str_list_to_array(str_list)"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def sha2(msg, digest_size, dict_encoding_state=None, func_id=-1):
    """Handles cases where sha2 receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [msg, digest_size]
    for i, arg in enumerate(args):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.sha2",
                ["msg", "digest_size", "dict_encoding_state", "func_id"],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        msg, digest_size, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return sha2_util(msg, digest_size, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def sha2_util(msg, digest_size, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function SPLIT which takes in a
               string, (or string column) and a separator string (or string column) and
               returns the result strings in arrays

    Args:
        msg (string scalar/column): The strings(s) to be encrypted
        digest_size (int): Size (in bits) of the output, corresponding to the specific
            SHA-2 function used to encrypt the string

    Returns:
        String scalar/column: hex-encoded string containing the N-bit SHA-2 message digest
    """
    verify_string_binary_arg(msg, "SHA2", "msg")
    verify_int_arg(digest_size, "SHA2", "digest_size")

    arg_names = ["msg", "digest_size", "dict_encoding_state", "func_id"]
    arg_types = [msg, digest_size, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    out_dtype = bodo.string_array_type
    # TODO: support bytes for SHA2
    if is_valid_binary_arg(msg):
        scalar_text = "msg_str = arg0._to_str()\n"
    else:
        scalar_text = "msg_str = arg0\n"
    scalar_text += (
        "res[i] = bodosql.kernels.crypto_funcs.sha2_algorithms(msg_str, arg1)"
    )

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def md5(msg, dict_encoding_state=None, func_id=-1):
    """Handles cases where md5 receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(msg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.md5",
            ["msg", "dict_encoding_state", "func_id"],
            0,
            default_map={"dict_encoding_state": None, "func_id": -1},
        )

    def impl(msg, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return md5_util(msg, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def md5_util(msg, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function MD5 which takes in a string,
               binary, (or string/binary column) and returns the MD5 encrypted strings

    Args:
        msg (string scalar/column): The strings(s) to be encrypted

    Returns:
        String scalar/column: hex-encoded string encrypted by MD5
    """
    verify_string_binary_arg(msg, "MD5", "msg")

    arg_names = ["msg", "dict_encoding_state", "func_id"]
    arg_types = [msg, dict_encoding_state, func_id]
    propagate_null = [True] + [False] * 2
    out_dtype = bodo.string_array_type
    if is_valid_binary_arg(msg):
        scalar_text = "msg_str = arg0._to_str()\n"
    else:
        scalar_text = "msg_str = arg0\n"
    scalar_text += "res[i] = bodosql.kernels.crypto_funcs.md5_algorithm(msg_str)"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def hex_encode(msg, case, dict_encoding_state=None, func_id=-1):
    """Handles cases where HEX_ENCODE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [msg, case]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.hex_encode",
                [
                    "msg",
                    "case",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(msg, case, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return hex_encode_util(msg, case, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def hex_encode_util(msg, case, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function HEX_ENCODE which takes in a string,
               binary, (or string/binary column) and returns the string encoded using
               the hex encoding scheme, with a second argument to specify how the
               letters a-f should be capitalized.

    Args:
        msg (string scalar/column): The strings(s) to be encrypted.
        case (integer): how to capitalize 1-f: 0 for lowercase, 1 for uppercase


    Returns:
        String scalar/column: hex-encoded string
    """
    verify_string_binary_arg(msg, "BASE64_ENCODE", "msg")

    # Verify that the case argument is either zero or 1
    case_int = get_overload_const_int(case)
    if case_int not in (0, 1):
        raise_bodo_error(f"hex_encode: invalid case integer: '{case_int}'")

    arg_names = ["msg", "case", "dict_encoding_state", "func_id"]
    arg_types = [msg, case, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    out_dtype = bodo.string_array_type
    if is_valid_binary_arg(msg):
        scalar_text = "msg_str = arg0._to_str()\n"
    else:
        scalar_text = "msg_str = arg0\n"
    scalar_text += (
        "hex_encoded = bodosql.kernels.crypto_funcs.hex_encode_algorithm(msg_str)\n"
    )
    scalar_text += (
        f"res[i] = hex_encoded{'.lower()' if case_int == 0 else '.upper()'}\n"
    )

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def hex_decode_string(msg, _try=False, dict_encoding_state=None, func_id=-1):
    """Handles cases where HEX_DECODE_STRING receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(msg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.hex_decode_string",
            [
                "msg",
                "_try",
                "dict_encoding_state",
                "func_id",
            ],
            0,
            default_map={"_try": False, "dict_encoding_state": None, "func_id": -1},
        )

    def impl(msg, _try=False, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return hex_decode_util(msg, _try, True, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def hex_decode_binary(msg, _try=False, dict_encoding_state=None, func_id=-1):
    """Handles cases where HEX_DECODE_BINARY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    if isinstance(msg, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodosql.kernels.hex_decode_binary",
            [
                "msg",
                "_try",
                "dict_encoding_state",
                "func_id",
            ],
            0,
            default_map={"_try": False, "dict_encoding_state": None, "func_id": -1},
        )

    def impl(msg, _try=False, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return hex_decode_util(msg, _try, False, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def hex_decode_util(msg, _try, _is_str, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function HEX_DECODE family of functions
       which takes in a string, produced by calling HEX_ENCODE on a string/binary value,
       and reverses the process to return the original string/binary value.

    Args:
        msg (string scalar/column): The strings(s) to be decoded.
        _try (boolean): if True, returns null on an error instead of raising an exception.
        _is_str (boolean): if True, returns the result as a string instead of binary.

    Returns:
        String scalar/column: the original string such that calling HEX_ENCODE
        with the output with the same arguments would produce the input to this function.
    """
    _try_bool = get_overload_const_bool(_try, "hex_decode", "_try")
    _is_str_bool = get_overload_const_bool(_is_str, "hex_decode", "_is_str")
    func_name = "HEX_DECODE"
    if _try_bool:
        func_name = "TRY_" + func_name
    if _is_str_bool:
        func_name += "_STRING"
    else:
        func_name += "_BINARY"

    verify_string_arg(msg, func_name, "msg")

    arg_names = ["msg", "_try", "_is_str", "dict_encoding_state", "func_id"]
    arg_types = [msg, _try, _is_str, dict_encoding_state, func_id]
    propagate_null = [True] + [False] * 4
    out_dtype = bodo.string_array_type if _is_str_bool else bodo.binary_array_type
    if _is_str_bool:
        scalar_text = "ans, success = bodosql.kernels.crypto_funcs.hex_decode_string_algorithm(arg0)\n"
    else:
        scalar_text = "ans, success = bodosql.kernels.crypto_funcs.hex_decode_binary_algorithm(arg0)\n"
    scalar_text += "if success:\n"
    scalar_text += "  res[i] = ans\n"
    scalar_text += "else:\n"
    if _try_bool:
        scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
    else:
        scalar_text += (
            f"  raise ValueError('{func_name} failed due to malformed string input')\n"
        )

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def base64_encode(msg, max_line_length, alphabet, dict_encoding_state=None, func_id=-1):
    """Handles cases where BASE64_ENCODE receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [msg, max_line_length, alphabet]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.base64_encode",
                [
                    "msg",
                    "max_line_length",
                    "alphabet",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        msg, max_line_length, alphabet, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return base64_encode_util(
            msg, max_line_length, alphabet, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def base64_encode_util(msg, max_line_length, alphabet, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function BASE64_ENCODE which takes in a string,
               binary, (or string/binary column) and returns the string encoded using
               the base64 encoding scheme, with optional extra arguments to specify
               the max line length and the final/padding characters.

    Args:
        msg (string scalar/column): The strings(s) to be encrypted.
        max_line_length (constant string): The 3 letter string where the first character is
        the character to be used as index 62, the second character is the character to be used
        as index 63, and the third character is the padding character (normally '+/=').
        alphabet (constant string): The 3 letter string where the first character is
        the character to be used as index 62, the second character is the character to be used
        as index 63, and the third character is the padding character (normally '+/=').


    Returns:
        String scalar/column: base64-encoded string
    """
    verify_string_binary_arg(msg, "BASE64_ENCODE", "msg")

    # Verify that the max line length is a constant non-negative integer
    verify_int_arg(max_line_length, "BASE64_ENCODE", "max_line_length")
    if not is_overload_constant_int(max_line_length):
        raise_bodo_error(
            "base64_encode: non-constant integer max_line_length argument not currently supported"
        )
    max_line_length_int = get_overload_const_int(max_line_length)
    if max_line_length_int < 0:
        raise_bodo_error(
            f"base64_encode: invalid max_line_length integer: '{max_line_length_int}'"
        )

    # Verify that the alphabet argument is a 3-character constant string that does not have any
    # repeats with itself or with the regular domain of base64 encoding chars (a-z, A-Z and 0-9)
    if not is_overload_constant_str(alphabet):
        raise_bodo_error(
            "base64_encode: non-constant string alphabet argument not currently supported"
        )
    alphabet_str = get_overload_const_str(alphabet)
    if len(alphabet_str) < 1:
        alphabet_str += "+"
    if len(alphabet_str) < 2:
        alphabet_str += "/"
    if len(alphabet_str) < 3:
        alphabet_str += "="
    if (
        len(alphabet_str) != 3
        or any(c.isalnum() or c == "\n" for c in alphabet_str)
        or len(set(alphabet_str)) != 3
    ):
        raise_bodo_error(f"base64_encode: invalid alphabet string: '{alphabet_str}'")
    extra_globals = {
        "char_62": alphabet_str[0],
        "char_63": alphabet_str[1],
        "char_pad": alphabet_str[2],
        "line_length": max_line_length_int,
    }

    arg_names = ["msg", "max_line_length", "alphabet", "dict_encoding_state", "func_id"]
    arg_types = [msg, max_line_length, alphabet, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    out_dtype = bodo.string_array_type
    if is_valid_binary_arg(msg):
        scalar_text = "msg_str = arg0._to_str()\n"
    else:
        scalar_text = "msg_str = arg0\n"
    scalar_text += "res[i] = bodosql.kernels.crypto_funcs.base64_encode_algorithm(msg_str, line_length, char_62, char_63, char_pad)"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def base64_decode_string(
    msg, alphabet, _try=False, dict_encoding_state=None, func_id=-1
):
    """Handles cases where BASE64_DECODE_STRING receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [msg, alphabet]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.base64_decode_string",
                [
                    "msg",
                    "alphabet",
                    "try",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"_try": False, "dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        msg, alphabet, _try=False, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return base64_decode_util(
            msg, alphabet, _try, True, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def base64_decode_binary(
    msg, alphabet, _try=False, dict_encoding_state=None, func_id=-1
):
    """Handles cases where BASE64_DECODE_BINARY receives optional arguments and forwards
    to the appropriate version of the real implementation"""
    args = [msg, alphabet]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.base64_decode_binary",
                [
                    "msg",
                    "alphabet",
                    "try",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"_try": False, "dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        msg, alphabet, _try=False, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return base64_decode_util(
            msg, alphabet, _try, False, dict_encoding_state, func_id
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def base64_decode_util(msg, alphabet, _try, _is_str, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function BASE64_DECODE family of functions
       which takes in a string, produced by calling BASE64_ENCODE on a string/binary value,
       and reverses the process to return the original string/binary value.

    Args:
        msg (string scalar/column): The strings(s) to be decoded.
        alphabet (constant string): The 3 letter string where the first character is
        the character to be used as index 62, the second character is the character to be used
        as index 63, and the third character is the padding character (normally '+/=').
        _try (boolean): if True, returns null on an error instead of raising an exception.
        _is_str (boolean): if True, returns the result as a string instead of binary.

    Returns:
        String scalar/column: the original string such that calling BASE64_ENCODE
        with the output with the same arguments would produce the input to this function.
    """
    _try_bool = get_overload_const_bool(_try, "base64_decode", "_try")
    _is_str_bool = get_overload_const_bool(_is_str, "base64_decode", "_is_str")
    func_name = "BASE64_DECODE"
    if _try_bool:
        func_name = "TRY_" + func_name
    if _is_str_bool:
        func_name += "_STRING"
    else:
        func_name += "_BINARY"

    verify_string_arg(msg, func_name, "msg")

    # Verify that the alphabet argument is a 3-character constant string that does not have any
    # repeats with itself or with the regular domain of base64 encoding chars (a-z, A-Z and 0-9)
    if not is_overload_constant_str(alphabet):
        raise_bodo_error(
            "base64_encode: non-constant string alphabet argument not currently supported"
        )
    alphabet_str = get_overload_const_str(alphabet)
    if len(alphabet_str) < 1:
        alphabet_str += "+"
    if len(alphabet_str) < 2:
        alphabet_str += "/"
    if len(alphabet_str) < 3:
        alphabet_str += "="
    if (
        len(alphabet_str) != 3
        or any(c.isalnum() or c == "\n" for c in alphabet_str)
        or len(set(alphabet_str)) != 3
    ):
        raise_bodo_error(
            f"{func_name.lower()}: invalid alphabet string: '{alphabet_str}'"
        )

    extra_globals = {
        "char_62": alphabet_str[0],
        "char_63": alphabet_str[1],
        "char_pad": alphabet_str[2],
    }

    arg_names = ["msg", "alphabet", "_try", "_is_str", "dict_encoding_state", "func_id"]
    arg_types = [msg, alphabet, _try, _is_str, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 4
    out_dtype = bodo.string_array_type if _is_str_bool else bodo.binary_array_type
    scalar_text = f"ans, success = bodosql.kernels.crypto_funcs.base64_decode_algorithm(arg0, char_62, char_63, char_pad, {_is_str_bool})\n"
    scalar_text += "if success:\n"
    scalar_text += "  res[i] = ans\n"
    scalar_text += "else:\n"
    if _try_bool:
        scalar_text += "  bodo.libs.array_kernels.setna(res, i)\n"
    else:
        scalar_text += (
            f"  raise ValueError('{func_name} failed due to malformed string input')\n"
        )

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True)
def uuid4(A):
    if A == bodo.none:

        def impl(A):  # pragma: no cover
            return bodo.libs.uuid.uuidV4()

    else:

        def impl(A):  # pragma: no cover
            n = len(A)
            res = bodo.libs.str_arr_ext.pre_alloc_string_array(n, 36)
            numba.parfors.parfor.init_prange()
            for i in numba.parfors.parfor.internal_prange(n):
                res[i] = bodo.libs.uuid.uuidV4()
            return res

    return impl


@numba.njit
def uuid5(namespace, name, dict_encoding_state=None, func_id=-1):
    return uuid5_util(namespace, name, dict_encoding_state, func_id)


@numba.generated_jit(nopython=True)
def uuid5_util(namespace, name, dict_encoding_state, func_id):
    verify_string_arg(namespace, "UUID_STRING", "namespace")
    verify_string_arg(name, "UUID_STRING", "name")

    out_dtype = bodo.string_array_type
    arg_names = ["namespace", "name", "dict_encoding_state", "func_id"]
    arg_types = [namespace, name, dict_encoding_state, func_id]
    propagate_null = [True] * 2 + [False] * 2
    scalar_text = "res[i] = bodo.libs.uuid.uuidV5(arg0, arg1)\n"

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )
