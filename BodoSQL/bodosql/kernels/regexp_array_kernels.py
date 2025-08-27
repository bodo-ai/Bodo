"""
Implements regexp array kernels that are specific to BodoSQL
"""

import re

import numba
from numba.core import types
from numba.extending import register_jitable

import bodo
from bodo.libs.array import get_replace_regex, get_replace_regex_dict_state
from bodo.libs.re_ext import init_const_pattern
from bodo.utils.typing import (
    get_overload_const_int,
    is_overload_constant_int,
    is_overload_none,
)
from bodosql.kernels.array_kernel_utils import (
    gen_vectorized,
    unopt_argument,
    verify_int_arg,
    verify_scalar_string_arg,
    verify_string_arg,
)


@register_jitable
def posix_to_re(pattern):
    """Transforms POSIX regexp syntax to the variety that Python's re module uses
    by mapping character classes to the corresponding set of Python characters.
    Mappings found here: https://github.com/micromatch/posix-character-classes

    Currently, errors are caused when a null terminator is inside of the
    embedded string literals, so [:ascii:] and [:word:] start at character 1
    instead of character 0.

    Args:
        pattern (string): the pattern in POSIX regexp syntax
        match_entire_string (boolean, optional): whether or not to add anchors
        to the pattern (default False)

    Returns:
        string: the transformed pattern in Python regexp syntax
    """
    posix_classes = {
        "[:alnum:]": "A-Za-z0-9",
        "[:alpha:]": "A-Za-z",
        "[:ascii:]": "\x01-\x7f",
        "[:blank:]": " \t",
        "[:cntrl:]": "\x01-\x1f\x7f",
        "[:digit:]": "0-9",
        "[:graph:]": "\x21-\x7e",
        "[:lower:]": "a-z",
        "[:print:]": "\x20-\x7e",
        "[:punct:]": "\\]\\[!\"#$%&'()*+,./:;<=>?@\\^_`{|}~-",
        "[:space:]": " \t\r\n\v\f",
        "[:upper:]": "A-Z",
        "[:word:]": "A-Za-z0-9_",
        "[:xdigit:]": "A-Fa-f0-9",
    }
    for key in posix_classes:
        pattern = pattern.replace(key, posix_classes[key])
    return pattern


def make_flag_bit_vector(flags):
    """Transforms Snowflake a REGEXP flag string into the corresponding Python
    regexp bit vector by or-ing together the correct flags. The important ones
    in this case are i, m and s, which correspond to regexp flags of the
    same name. If i and c are both in the string, ignore the i unless it
    comes after c.

    Args:
        flags (string): a string whose characters determine which regexp
        flags need to be used.

    Returns:
        RegexFlagsType: the corresponding flags from the input string
        or-ed together
    """
    result = 0
    # Regular expressions are case sensitive unless the I flag is used
    if "i" in flags:
        if "c" not in flags or flags.rindex("i") > flags.rindex("c"):
            result = result | re.I
    # Regular expressions only allow anchor chars ^ and $ to interact with
    # the start/end of a string, unless the M flag is used
    if "m" in flags:
        result = result | re.M
    # Regular expressions do not allow the . character to capture a newline
    # char, unless the S flag is used
    if "s" in flags:
        result = result | re.S
    return result


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_count(arr, pattern, position, flags, dict_encoding_state=None, func_id=-1):
    """Handles cases where REGEXP_COUNT receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, position, flags]
    for i in range(4):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.regexp_count",
                [
                    "arr",
                    "pattern",
                    "position",
                    "flags",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, pattern, position, flags, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return regexp_count_util(
            arr,
            pattern,
            position,
            numba.literally(flags),
            dict_encoding_state,
            func_id,
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_instr(
    arr,
    pattern,
    position,
    occurrence,
    option,
    flags,
    group,
    dict_encoding_state=None,
    func_id=-1,
):
    """Handles cases where REGEXP_INSTR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, position, occurrence, option, flags, group]
    for i in range(7):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.regexp_instr",
                [
                    "arr",
                    "pattern",
                    "position",
                    "occurrence",
                    "option",
                    "flags",
                    "group",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr,
        pattern,
        position,
        occurrence,
        option,
        flags,
        group,
        dict_encoding_state=None,
        func_id=-1,
    ):  # pragma: no cover
        return regexp_instr_util(
            arr,
            numba.literally(pattern),
            position,
            occurrence,
            option,
            numba.literally(flags),
            group,
            dict_encoding_state,
            func_id,
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_like(arr, pattern, flags, dict_encoding_state=None, func_id=-1):
    """Handles cases where REGEXP_LIKE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, flags]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.regexp_like",
                [
                    "arr",
                    "pattern",
                    "flags",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr, pattern, flags, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return regexp_like_util(
            arr,
            pattern,
            numba.literally(flags),
            dict_encoding_state,
            func_id,
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_replace(
    arr,
    pattern,
    replacement,
    position,
    occurrence,
    flags,
    dict_encoding_state=None,
    func_id=-1,
):
    """Handles cases where REGEXP_REPLACE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, replacement, position, occurrence, flags]
    for i in range(6):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.regexp_replace",
                [
                    "arr",
                    "pattern",
                    "replacement",
                    "position",
                    "occurrence",
                    "flags",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr,
        pattern,
        replacement,
        position,
        occurrence,
        flags,
        dict_encoding_state=None,
        func_id=-1,
    ):  # pragma: no cover
        return regexp_replace_util(
            arr,
            pattern,
            replacement,
            position,
            occurrence,
            numba.literally(flags),
            dict_encoding_state,
            func_id,
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_substr(
    arr,
    pattern,
    position,
    occurrence,
    flags,
    group,
    dict_encoding_state=None,
    func_id=-1,
):
    """Handles cases where REGEXP_SUBSTR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, position, occurrence, flags, group]
    for i in range(6):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodosql.kernels.regexp_substr",
                [
                    "arr",
                    "pattern",
                    "position",
                    "occurrence",
                    "flags",
                    "group",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    def impl(
        arr,
        pattern,
        position,
        occurrence,
        flags,
        group,
        dict_encoding_state=None,
        func_id=-1,
    ):  # pragma: no cover
        return regexp_substr_util(
            arr,
            numba.literally(pattern),
            position,
            occurrence,
            numba.literally(flags),
            group,
            dict_encoding_state,
            func_id,
        )

    return impl


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_count_util(arr, pattern, position, flags, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function REGEXP_COUNT which takes in a string
       (or column), a pattern, a position, and regexp control flags and returns
       the number of occurrences of the pattern in the string starting at the
       position.


    Args:
        arr (string array/series/scalar): the string(s) being searched.
        pattern (string): the regexp being searched for.
        position (integer array/series/scalar): the starting position(s) (1-indexed).
        Throws an error if negative.
        flags (string): the regexp control flags.

    Returns:
        int series/scalar: the number of matches
    """
    verify_string_arg(arr, "REGEXP_COUNT", "arr")
    verify_string_arg(pattern, "REGEXP_COUNT", "pattern")
    verify_int_arg(position, "REGEXP_COUNT", "position")
    verify_scalar_string_arg(flags, "REGEXP_COUNT", "flags")

    arg_names = [
        "arr",
        "pattern",
        "position",
        "flags",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [arr, pattern, position, flags, dict_encoding_state, func_id]
    propagate_null = [True] * 4 + [False] * 2

    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bit_vector = make_flag_bit_vector(flag_str)

    prefix_code = "\n"
    scalar_text = ""
    if bodo.utils.utils.is_array_typ(position, True):
        scalar_text += "if arg2 <= 0: raise ValueError('REGEXP_COUNT requires a positive position')\n"
    else:
        prefix_code += "if position <= 0: raise ValueError('REGEXP_COUNT requires a positive position')\n"

    extra_globals = None

    if bodo.utils.typing.is_overload_constant_str(pattern):
        pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
        converted_pattern = posix_to_re(pattern_str)

        if converted_pattern == "":
            scalar_text += "res[i] = 0"
        else:
            # Generate the compile at compile time to avoid an extra objmode
            # step at runtime.
            extra_globals = {
                "r": re.compile(converted_pattern, flag_bit_vector),
            }
            scalar_text += "res[i] = bodo.libs.re_ext.re_count(r, arg0[arg2-1:])"
    else:
        extra_globals = {"flag_bit_vector": flag_bit_vector, "posix_to_re": posix_to_re}
        scalar_text += "r = re.compile(posix_to_re(arg1), flag_bit_vector)\n"
        scalar_text += "res[i] = bodo.libs.re_ext.re_count(r, arg0[arg2-1:])"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_instr_util(
    arr,
    pattern,
    position,
    occurrence,
    option,
    flags,
    group,
    dict_encoding_state,
    func_id,
):
    """A dedicated kernel for the SQL function REGEXP_INSTR which takes in a string
       (or column), a pattern, a number of occurrences, an option flag, a position,
       regexp control flags, and a group number, and returns the location of an
       occurrence of the pattern in the string starting at the position (or of
       one of its subgroups).

       Note: this function is expected to have 'e' in the flag string if
       a group is provided, and if 'e' is provided but a group is not then the
       default is 1. Both of these behaviors are covered by StringFnCodeGen.java.


    Args:
        arr (string array/series/scalar): the string(s) being searched.
        pattern (string): the regexp being searched for.
        position (integer array/series/scalar): the starting position(s) (1-indexed).
        Throws an error if negative.
        occurrence (integer array/series/scalar): which matches to locate (1-indexed).
        Throws an error if negative.
        option (integer array/series/scalar): if zero, returns the start of the
        match. If 1, returns the end of the match. Otherwise, throws an error.
        flags (string): the regexp control flags
        group (integer array/series/scalar): which subgroup to return (only used
        if the flag strings contains 'e').

    Returns:
        int series/scalar: the location of the matches
    """
    verify_string_arg(arr, "REGEXP_INSTR", "arr")
    verify_scalar_string_arg(pattern, "REGEXP_INSTR", "pattern")
    verify_int_arg(position, "REGEXP_INSTR", "position")
    verify_int_arg(occurrence, "REGEXP_INSTR", "occurrence")
    verify_int_arg(option, "REGEXP_INSTR", "option")
    verify_scalar_string_arg(flags, "REGEXP_INSTR", "flags")
    verify_int_arg(group, "REGEXP_INSTR", "group")

    arg_names = [
        "arr",
        "pattern",
        "position",
        "occurrence",
        "option",
        "flags",
        "group",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [
        arr,
        pattern,
        position,
        occurrence,
        option,
        flags,
        group,
        dict_encoding_state,
        func_id,
    ]
    propagate_null = [True] * 7 + [False] * 2

    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    n_groups = re.compile(pattern_str).groups
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bit_vector = make_flag_bit_vector(flag_str)

    prefix_code = "\n"
    scalar_text = ""

    if bodo.utils.utils.is_array_typ(position, True):
        scalar_text += "if arg2 <= 0: raise ValueError('REGEXP_INSTR requires a positive position')\n"
    else:
        prefix_code += "if position <= 0: raise ValueError('REGEXP_INSTR requires a positive position')\n"
    if bodo.utils.utils.is_array_typ(occurrence, True):
        scalar_text += "if arg3 <= 0: raise ValueError('REGEXP_INSTR requires a positive occurrence')\n"
    else:
        prefix_code += "if occurrence <= 0: raise ValueError('REGEXP_INSTR requires a positive occurrence')\n"
    if bodo.utils.utils.is_array_typ(option, True):
        scalar_text += "if arg4 != 0 and arg4 != 1: raise ValueError('REGEXP_INSTR requires option to be 0 or 1')\n"
    else:
        prefix_code += "if option != 0 and option != 1: raise ValueError('REGEXP_INSTR requires option to be 0 or 1')\n"

    if "e" in flag_str:
        if bodo.utils.utils.is_array_typ(group, True):
            scalar_text += f"if not (1 <= arg6 <= {n_groups}): raise ValueError('REGEXP_INSTR requires a valid group number')\n"
        else:
            prefix_code += f"if not (1 <= group <= {n_groups}): raise ValueError('REGEXP_INSTR requires a valid group number')\n"

    extra_globals = None
    if converted_pattern == "":
        scalar_text += "res[i] = 0"
    else:
        # Generate the compile at compile time to avoid an extra objmode
        # step at runtime.
        extra_globals = {"r": re.compile(converted_pattern, flag_bit_vector)}
        scalar_text += "arg0 = arg0[arg2-1:]\n"
        scalar_text += "res[i] = 0\n"
        scalar_text += "offset = arg2\n"
        scalar_text += "for j in range(arg3):\n"
        scalar_text += "   match = r.search(arg0)\n"
        scalar_text += "   if match is None:\n"
        scalar_text += "      res[i] = 0\n"
        scalar_text += "      break\n"
        scalar_text += "   start, end = match.span()\n"
        scalar_text += "   if j == arg3 - 1:\n"
        if "e" in flag_str:
            scalar_text += "      res[i] = offset + match.span(arg6)[arg4]\n"
        else:
            scalar_text += "      res[i] = offset + match.span()[arg4]\n"
        scalar_text += "   else:\n"
        scalar_text += "      offset += end\n"
        scalar_text += "      arg0 = arg0[end:]\n"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_like_util(arr, pattern, flags, dict_encoding_state, func_id):
    """A dedicated kernel for the SQL function REGEXP_LIKE which takes in a string
       (or column), a pattern, and regexp control flags and returns
       whether or not the pattern matches the entire string.


    Args:
        arr (string array/series/scalar): the string(s) being searched.
        pattern (string): the regexp being searched for.
        flags (string): the regexp control flags.

    Returns:
        boolean series/scalar: whether or not the string(s) match
    """
    verify_string_arg(arr, "REGEXP_LIKE", "arr")
    verify_string_arg(pattern, "REGEXP_LIKE", "pattern")
    verify_scalar_string_arg(flags, "REGEXP_LIKE", "flags")

    arg_names = ["arr", "pattern", "flags", "dict_encoding_state", "func_id"]
    arg_types = [arr, pattern, flags, dict_encoding_state, func_id]
    propagate_null = [True] * 3 + [False] * 2
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bit_vector = make_flag_bit_vector(flag_str)
    extra_globals = None

    if bodo.utils.typing.is_overload_constant_str(pattern):
        pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
        converted_pattern = posix_to_re(pattern_str)

        if converted_pattern == "":
            scalar_text = "res[i] = len(arg0) == 0"
        else:
            # Generate the compile at compile time to avoid an extra objmode
            # step at runtime.
            extra_globals = {"r": re.compile(converted_pattern, flag_bit_vector)}
            scalar_text = "if r.fullmatch(arg0) is None:\n"
            scalar_text += "   res[i] = False\n"
            scalar_text += "else:\n"
            scalar_text += "   res[i] = True\n"
    else:
        extra_globals = {"flag_bit_vector": flag_bit_vector, "posix_to_re": posix_to_re}
        scalar_text = "converted_pattern = posix_to_re(arg1)\n"
        scalar_text += "r = re.compile(converted_pattern, flag_bit_vector)\n"
        scalar_text += "if r.fullmatch(arg0) is None:\n"
        scalar_text += "   res[i] = False\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = True\n"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array_type

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


def _gen_regex_replace_body():
    """generate scalar computation for regexp_replace kernel"""
    scalar_text = "result = arg0[:arg3-1]\n"
    scalar_text += "arg0 = arg0[arg3-1:]\n"
    # If replacing everything, just use re.sub()
    scalar_text += "if arg4 == 0:\n"
    scalar_text += "   res[i] = result + r.sub(arg2, arg0)\n"
    # Otherwise, repeatedly find matches and truncate, then replace the
    # first match in the remaining suffix
    scalar_text += "else:\n"
    scalar_text += "   nomatch = False\n"
    scalar_text += "   for j in range(arg4 - 1):\n"
    scalar_text += "      match = r.search(arg0)\n"
    scalar_text += "      if match is None:\n"
    scalar_text += "         res[i] = result + arg0\n"
    scalar_text += "         nomatch = True\n"
    scalar_text += "         break\n"
    scalar_text += "      _, end = match.span()\n"
    scalar_text += "      result += arg0[:end]\n"
    scalar_text += "      arg0 = arg0[end:]\n"
    scalar_text += "   if nomatch == False:\n"
    scalar_text += "      result += r.sub(arg2, arg0, count=1)\n"
    scalar_text += "      res[i] = result"
    return scalar_text


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_replace_util(
    arr, pattern, replacement, position, occurrence, flags, dict_encoding_state, func_id
):
    """A dedicated kernel for the SQL function REGEXP_REPLACE which takes in a string
       (or column), a pattern, a replacement string, an occurrence number, a position,
       and regexp control flags and returns the string(s) with the specified
       match occurrence replaced with the string provided, starting the search
       at the position specified.


    Args:
        arr (string array/series/scalar): the string(s) being searched.
        pattern (string): the regexp being searched for.
        replacement (string array/series/scalar): the string to replace matches with.
        position (integer array/series/scalar): the starting position(s) (1-indexed).
        occurrence (integer array/series/scalar): which matches to replace (1-indexed).
        Throws an error if negative.
        If 0, replaces all the matches.
        flags (string): the regexp control flags
        group (integer array/series/scalar): which subgroup to return (only used
        if the flag strings contains 'e').

    Returns:
        int series/scalar: the location of the matches
    """
    verify_string_arg(arr, "REGEXP_REPLACE", "arr")
    verify_string_arg(pattern, "REGEXP_REPLACE", "pattern")
    verify_string_arg(replacement, "REGEXP_REPLACE", "replacement")
    verify_int_arg(position, "REGEXP_REPLACE", "position")
    verify_int_arg(occurrence, "REGEXP_REPLACE", "occurrence")
    verify_scalar_string_arg(flags, "REGEXP_REPLACE", "flags")

    arg_names = [
        "arr",
        "pattern",
        "replacement",
        "position",
        "occurrence",
        "flags",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [
        arr,
        pattern,
        replacement,
        position,
        occurrence,
        flags,
        dict_encoding_state,
        func_id,
    ]
    propagate_null = [True] * 6 + [False, False]
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bit_vector = make_flag_bit_vector(flag_str)

    prefix_code = "\n"
    scalar_text = ""
    if bodo.utils.utils.is_array_typ(position, True):
        scalar_text += "if arg3 <= 0: raise ValueError('REGEXP_REPLACE requires a positive position')\n"
    else:
        prefix_code += "if position <= 0: raise ValueError('REGEXP_REPLACE requires a positive position')\n"
    if bodo.utils.utils.is_array_typ(occurrence, True):
        scalar_text += "if arg4 < 0: raise ValueError('REGEXP_REPLACE requires a non-negative occurrence')\n"
    else:
        prefix_code += "if occurrence < 0: raise ValueError('REGEXP_REPLACE requires a non-negative occurrence')\n"
    extra_globals = None

    if bodo.utils.typing.is_overload_constant_str(pattern):
        pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
        converted_pattern = posix_to_re(pattern_str)
        # Take an optimized path if the user has only provided the array, pattern,
        # and replacement and its possible to handle the regex entirely in C++.
        # TODO: We should be able to support additional arguments inside position
        # and occurrence without being constant.
        if (
            bodo.utils.utils.is_array_typ(arr, True)
            and not bodo.hiframes.series_str_impl.is_regex_unsupported(
                converted_pattern
            )
            and (
                not bodo.utils.utils.is_array_typ(replacement, True)
                and not is_overload_none(replacement)
            )
            and is_overload_constant_int(position)
            and get_overload_const_int(position) == 1
            and is_overload_constant_int(occurrence)
            and get_overload_const_int(occurrence) == 0
            and flag_bit_vector == 0
        ):
            # Optimized implementation just calls into C++ and has it handle the entire array.
            use_dict_caching = arr == bodo.dict_str_arr_type and not is_overload_none(
                dict_encoding_state
            )
            if use_dict_caching:

                def impl_state(
                    arr,
                    pattern,
                    replacement,
                    position,
                    occurrence,
                    flags,
                    dict_encoding_state,
                    func_id,
                ):  # pragma: no cover
                    utf8_pattern = bodo.libs.str_ext.unicode_to_utf8(pattern)
                    utf8_replacement = bodo.libs.str_ext.unicode_to_utf8(replacement)
                    out_arr = get_replace_regex_dict_state(
                        arr,
                        utf8_pattern,
                        utf8_replacement,
                        dict_encoding_state,
                        func_id,
                    )
                    return out_arr

                return impl_state

            else:

                def impl(
                    arr,
                    pattern,
                    replacement,
                    position,
                    occurrence,
                    flags,
                    dict_encoding_state,
                    func_id,
                ):  # pragma: no cover
                    utf8_pattern = bodo.libs.str_ext.unicode_to_utf8(pattern)
                    utf8_replacement = bodo.libs.str_ext.unicode_to_utf8(replacement)
                    out_arr = get_replace_regex(arr, utf8_pattern, utf8_replacement)
                    return out_arr

                return impl

        if converted_pattern == "":
            scalar_text += "res[i] = arg0"
        else:
            # Generate the compile at compile time to avoid an extra objmode
            # step at runtime.
            extra_globals = {"r": re.compile(converted_pattern, flag_bit_vector)}
            scalar_text += _gen_regex_replace_body()
    else:
        # Non-constant pattern
        extra_globals = {"flag_bit_vector": flag_bit_vector, "posix_to_re": posix_to_re}
        scalar_text += "r = re.compile(posix_to_re(arg1), flag_bit_vector)\n"
        scalar_text += _gen_regex_replace_body()

    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
        # Several different values could be replaced with the same output
        # pattern, so there could be duplicates.
        may_cause_duplicate_dict_array_values=True,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )


@numba.generated_jit(nopython=True, no_unliteral=True)
def regexp_substr_util(
    arr, pattern, position, occurrence, flags, group, dict_encoding_state, func_id
):
    """A dedicated kernel for the SQL function REGEXP_SUBSTR which takes in a string
       (or column), a pattern, a number of occurrences, a position, regexp control
       flags, and a group number, and returns the substring of the original
       string corresponding to an occurrence of the pattern (or of one of its
       subgroups).

       Note: this function is expected to have 'e' in the flag string if
       a group is provided, and if 'e' is provided but a group is not then the
       default is 1. Both of these behaviors are covered by StringFnCodeGen.java.


    Args:
        arr (string array/series/scalar): the string(s) being searched
        pattern (string): the regexp being searched for.
        position (integer array/series/scalar): the starting position(s) (1-indexed).
        Throws an error if negative.
        occurrence (integer array/series/scalar): which matches to return (1-indexed).
        Throws an error if negative.
        flags (string): the regexp control flags
        group (integer array/series/scalar): which subgroup of the match to return
        (only used if the flag strings contains 'e').

    Returns:
        string series/scalar: the substring(s) that caused the match
    """
    verify_string_arg(arr, "REGEXP_SUBSTR", "arr")
    verify_scalar_string_arg(pattern, "REGEXP_SUBSTR", "pattern")
    verify_int_arg(position, "REGEXP_SUBSTR", "position")
    verify_int_arg(occurrence, "REGEXP_SUBSTR", "occurrence")
    verify_scalar_string_arg(flags, "REGEXP_SUBSTR", "flags")
    verify_int_arg(group, "REGEXP_SUBSTR", "group")

    arg_names = [
        "arr",
        "pattern",
        "position",
        "occurrence",
        "flags",
        "group",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [
        arr,
        pattern,
        position,
        occurrence,
        flags,
        group,
        dict_encoding_state,
        func_id,
    ]
    propagate_null = [True] * 6 + [False] * 2

    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    n_groups = re.compile(pattern_str).groups
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bit_vector = make_flag_bit_vector(flag_str)

    prefix_code = "\n"
    scalar_text = ""

    if bodo.utils.utils.is_array_typ(position, True):
        scalar_text += "if arg2 <= 0: raise ValueError('REGEXP_SUBSTR requires a positive position')\n"
    else:
        prefix_code += "if position <= 0: raise ValueError('REGEXP_SUBSTR requires a positive position')\n"
    if bodo.utils.utils.is_array_typ(occurrence, True):
        scalar_text += "if arg3 <= 0: raise ValueError('REGEXP_SUBSTR requires a positive occurrence')\n"
    else:
        prefix_code += "if occurrence <= 0: raise ValueError('REGEXP_SUBSTR requires a positive occurrence')\n"
    if "e" in flag_str:
        if bodo.utils.utils.is_array_typ(group, True):
            scalar_text += f"if not (1 <= arg5 <= {n_groups}): raise ValueError('REGEXP_SUBSTR requires a valid group number')\n"
        else:
            prefix_code += f"if not (1 <= group <= {n_groups}): raise ValueError('REGEXP_SUBSTR requires a valid group number')\n"

    extra_globals = None
    if converted_pattern == "":
        scalar_text += "bodo.libs.array_kernels.setna(res, i)"
    else:
        # Generate the compile at compile time to avoid an extra objmode
        # step at runtime.
        extra_globals = {
            "non_const_r": re.compile(converted_pattern, flag_bit_vector),
            "init_const_pattern": init_const_pattern,
        }
        prefix_code += (
            f"r = init_const_pattern(non_const_r, {repr(converted_pattern)})\n"
        )

        # FROM https://docs.snowflake.com/en/sql-reference/functions/regexp_substr:
        #  If a group_num is specified, Snowflake allows extraction even if
        #  the 'e' option was not also specified. The 'e' is implied.
        if "e" in flag_str and not is_overload_none(group):
            scalar_text += "matches = r.findall(arg0[arg2-1:])\n"
            scalar_text += "if len(matches) < arg3:\n"
            scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "else:\n"
            if n_groups == 1:
                scalar_text += "   res[i] = matches[arg3-1]\n"
            else:
                scalar_text += "   res[i] = matches[arg3-1][arg5-1]\n"
        else:
            scalar_text += "arg0 = str(arg0)[arg2-1:]\n"
            scalar_text += "for j in range(arg3):\n"
            scalar_text += "   match = r.search(arg0)\n"
            scalar_text += "   if match is None:\n"
            scalar_text += "      bodo.libs.array_kernels.setna(res, i)\n"
            scalar_text += "      break\n"
            scalar_text += "   start, end = match.span()\n"
            scalar_text += "   if j == arg3 - 1:\n"
            scalar_text += "      res[i] = arg0[start:end]\n"
            scalar_text += "   else:\n"
            scalar_text += "      arg0 = arg0[end:]\n"
    out_dtype = bodo.string_array_type

    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        # Substrings of unique strings can lead to collisions
        may_cause_duplicate_dict_array_values=True,
        prefix_code=prefix_code,
        extra_globals=extra_globals,
        # Add support for dict encoding caching with streaming.
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
    )
