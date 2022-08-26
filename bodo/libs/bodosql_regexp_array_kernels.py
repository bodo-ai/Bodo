# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implements regexp array kernels that are specific to BodoSQL
"""

import re

import numba
from numba.core import types

import bodo
from bodo.libs.bodosql_array_kernel_utils import *


def posix_to_re(pattern):
    """Transforms POSIX regexp syntax to the variety that Python's re module uses
    by mapping character classes to the corresponding set of Python characters.
    Mappings found here: https://github.com/micromatch/posix-character-classes

    Currently, errors are caused when a null terminator is inside of the
    embedded stirng literals, so [:ascii:] and [:word:] start at character 1
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
        "[:punct:]": "\]\[!\"#$%&'()*+,./:;<=>?@\^_`{|}~-",
        "[:space:]": " \t\r\n\v\f",
        "[:upper:]": "A-Z",
        "[:word:]": "A-Za-z0-9_",
        "[:xdigit:]": "A-Fa-f0-9",
    }
    for key in posix_classes:
        pattern = pattern.replace(key, posix_classes[key])
    return pattern


def make_flag_bitvector(flags):
    """Transforms Snowflake a REGEXP flag string into the corresponding Python
    regexp bitvector by or-ing together the correct flags. The important ones
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


@numba.generated_jit(nopython=True)
def regexp_count(arr, pattern, position, flags):
    """Handles cases where REGEXP_COUNT receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, position, flags]
    for i in range(4):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regexp_count",
                ["arr", "pattern", "position", "flags"],
                i,
            )

    def impl(arr, pattern, position, flags):  # pragma: no cover
        return regexp_count_util(
            arr, numba.literally(pattern), position, numba.literally(flags)
        )

    return impl


@numba.generated_jit(nopython=True)
def regexp_instr(arr, pattern, position, occurrence, option, flags, group):
    """Handles cases where REGEXP_INSTR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, position, occurrence, option, flags, group]
    for i in range(7):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regexp_instr",
                [
                    "arr",
                    "pattern",
                    "position",
                    "occurrence",
                    "option",
                    "flags",
                    "group",
                ],
                i,
            )

    def impl(
        arr, pattern, position, occurrence, option, flags, group
    ):  # pragma: no cover
        return regexp_instr_util(
            arr,
            numba.literally(pattern),
            position,
            occurrence,
            option,
            numba.literally(flags),
            group,
        )

    return impl


@numba.generated_jit(nopython=True)
def regexp_like(arr, pattern, flags):
    """Handles cases where REGEXP_LIKE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, flags]
    for i in range(3):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regexp_like",
                ["arr", "pattern", "flags"],
                i,
            )

    def impl(arr, pattern, flags):  # pragma: no cover
        return regexp_like_util(arr, numba.literally(pattern), numba.literally(flags))

    return impl


@numba.generated_jit(nopython=True)
def regexp_replace(arr, pattern, replacement, position, occurrence, flags):
    """Handles cases where REGEXP_REPLACE receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, replacement, position, occurrence, flags]
    for i in range(6):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regexp_replace",
                ["arr", "pattern", "replacement", "position", "occurrence", "flags"],
                i,
            )

    def impl(
        arr, pattern, replacement, position, occurrence, flags
    ):  # pragma: no cover
        return regexp_replace_util(
            arr,
            numba.literally(pattern),
            replacement,
            position,
            occurrence,
            numba.literally(flags),
        )

    return impl


@numba.generated_jit(nopython=True)
def regexp_substr(arr, pattern, position, occurrence, flags, group):
    """Handles cases where REGEXP_SUBSTR receives optional arguments and forwards
    to args appropriate version of the real implementation"""
    args = [arr, pattern, position, occurrence, flags, group]
    for i in range(6):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.regexp_substr",
                ["arr", "pattern", "position", "occurrence", "flags", "group"],
                i,
            )

    def impl(arr, pattern, position, occurrence, flags, group):  # pragma: no cover
        return regexp_substr_util(
            arr,
            numba.literally(pattern),
            position,
            occurrence,
            numba.literally(flags),
            group,
        )

    return impl


@numba.generated_jit(nopython=True)
def regexp_count_util(arr, pattern, position, flags):
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
    verify_scalar_string_arg(pattern, "REGEXP_COUNT", "pattern")
    verify_int_arg(position, "REGEXP_COUNT", "position")
    verify_scalar_string_arg(flags, "REGEXP_COUNT", "flags")

    arg_names = ["arr", "pattern", "position", "flags"]
    arg_types = [arr, pattern, position, flags]
    propagate_null = [True] * 4

    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bitvector = make_flag_bitvector(flag_str)

    prefix_code = "\n"
    scalar_text = ""
    if bodo.utils.utils.is_array_typ(position, True):
        scalar_text += "if arg2 <= 0: raise ValueError('REGEXP_COUNT requires a positive position')\n"
    else:
        prefix_code += "if position <= 0: raise ValueError('REGEXP_COUNT requires a positive position')\n"

    if converted_pattern == "":
        scalar_text += "res[i] = 0"
    else:
        prefix_code += f"r = re.compile({repr(converted_pattern)}, {flag_bitvector})"
        scalar_text += "res[i] = len(r.findall(arg0[arg2-1:]))"

    out_dtype = bodo.libs.int_arr_ext.IntegerArrayType(types.int32)

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


@numba.generated_jit(nopython=True)
def regexp_instr_util(arr, pattern, position, occurrence, option, flags, group):
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

    arg_names = ["arr", "pattern", "position", "occurrence", "option", "flags", "group"]
    arg_types = [arr, pattern, position, occurrence, option, flags, group]
    propagate_null = [True] * 7

    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    n_groups = re.compile(pattern_str).groups
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bitvector = make_flag_bitvector(flag_str)

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

    if converted_pattern == "":
        scalar_text += "res[i] = 0"
    else:
        prefix_code += f"r = re.compile({repr(converted_pattern)}, {flag_bitvector})"
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

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


@numba.generated_jit(nopython=True)
def regexp_like_util(arr, pattern, flags):
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
    verify_scalar_string_arg(pattern, "REGEXP_LIKE", "pattern")
    verify_scalar_string_arg(flags, "REGEXP_LIKE", "flags")

    arg_names = ["arr", "pattern", "flags"]
    arg_types = [arr, pattern, flags]
    propagate_null = [True] * 3
    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bitvector = make_flag_bitvector(flag_str)
    if converted_pattern == "":
        prefix_code = None
        scalar_text = "res[i] = len(arg0) == 0"
    else:
        prefix_code = f"r = re.compile({repr(converted_pattern)}, {flag_bitvector})"
        scalar_text = "if r.fullmatch(arg0) is None:\n"
        scalar_text += "   res[i] = False\n"
        scalar_text += "else:\n"
        scalar_text += "   res[i] = True\n"

    out_dtype = bodo.libs.bool_arr_ext.boolean_array

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


@numba.generated_jit(nopython=True)
def regexp_replace_util(arr, pattern, replacement, position, occurrence, flags):
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
        Throws an error if negative.
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
    verify_scalar_string_arg(pattern, "REGEXP_REPLACE", "pattern")
    verify_string_arg(replacement, "REGEXP_REPLACE", "replacement")
    verify_int_arg(position, "REGEXP_REPLACE", "position")
    verify_int_arg(occurrence, "REGEXP_REPLACE", "occurrence")
    verify_scalar_string_arg(flags, "REGEXP_REPLACE", "flags")

    arg_names = ["arr", "pattern", "replacement", "position", "occurrence", "flags"]
    arg_types = [arr, pattern, replacement, position, occurrence, flags]
    propagate_null = [True] * 6

    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bitvector = make_flag_bitvector(flag_str)

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

    if converted_pattern == "":
        scalar_text += "res[i] = arg0"
    else:
        prefix_code += f"r = re.compile({repr(converted_pattern)}, {flag_bitvector})"
        scalar_text += "result = arg0[:arg3-1]\n"
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

    out_dtype = bodo.string_array_type

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )


@numba.generated_jit(nopython=True)
def regexp_substr_util(arr, pattern, position, occurrence, flags, group):
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

    arg_names = ["arr", "pattern", "position", "occurrence", "flags", "group"]
    arg_types = [arr, pattern, position, occurrence, flags, group]
    propagate_null = [True] * 6

    pattern_str = bodo.utils.typing.get_overload_const_str(pattern)
    converted_pattern = posix_to_re(pattern_str)
    n_groups = re.compile(pattern_str).groups
    flag_str = bodo.utils.typing.get_overload_const_str(flags)
    flag_bitvector = make_flag_bitvector(flag_str)

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

    if converted_pattern == "":
        scalar_text += "bodo.libs.array_kernels.setna(res, i)"
    else:
        prefix_code += f"r = re.compile({repr(converted_pattern)}, {flag_bitvector})"
        if "e" in flag_str:
            scalar_text += "matches = r.findall(arg0[arg2-1:])\n"
            scalar_text += f"if len(matches) < arg3:\n"
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

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
    )
