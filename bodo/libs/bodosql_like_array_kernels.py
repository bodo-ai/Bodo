# Copyright (C) 2023 Bodo Inc. All rights reserved.
"""
Implements like array kernels that are specific to BodoSQL
"""

import re
from typing import Tuple

import numpy as np
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
from bodo.libs.bodosql_array_kernel_utils import (
    gen_vectorized,
    unopt_argument,
    verify_string_arg,
)
from bodo.utils.typing import (
    BodoError,
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)


def convert_sql_pattern_to_python_compile_time(
    pattern: str,
    escape: str,
    make_output_lowercase: bool,
) -> Tuple[str, bool, bool, bool, bool]:
    """
    Converts a SQL pattern to its Python equivalent. This is used for like/ilike
    paths where the pattern is a constant string literal. One challenge that arises
    is that Python has additional special characters that SQL doesn't have. As a result,
    we have to be careful and escape patterns that may arise in Python.

    In some cases it may also be possible to avoid regex entirely. In those cases
    it may be possible to replace a regular expression check with `==`, `startswith`,
    `endswith` or `in`. In those cases we pass back a few extra boolean values:

        - requires_regex: Can we avoid regex entirely
        - must_match_start: Must the pattern be found at the start of the string?
        - must_match_end: Must the pattern be found at the end of the string?
        - match_anything: Will the pattern match any non-null string?

    If requires_regex == False then the string returned will be returned without escaping
    the contents.

    Note there may be additional optimizations possible based on the _ escape
    character, but at this time we only consider % for avoiding regex.

    Here are a couple example calls to explain this more clearly

    ("%s", "", False) -> ("s", False, False, True, False)

        This first pattern matches any string that ends with "s". This does not
        require a regex so we keep the pattern as a regular string and set
        requires_regex=False. Then we set must_match_end=True so we know we must do endswith.

    ("Ha^%", "^", False) -> ("Ha%", False, True, True, False)

        This second pattern matches the literal string "Ha%". The escape value of ^
        tells us that "^%" matches a literal "%" rather than a wild card. As a result
        we do not need a regex (there are no wildcard), but we must match the string exactly.

    ("Ha_d", "", True) -> ("^ha.d$", True, True, True, False)

        This third pattern requires an actual regex. Since both the start and end must match
        we append the required "^" and "$" to the Python regex. In addition, since we specified
        "make_output_lowercase" the output pattern will be converted to all lowercase characters.
        This is used for case insensitive comparison.

    Args:
        pattern (str): A SQL pattern passed to like. This pattern can contain values
        that are interpreted literally and SQL wildcards (_ and %).

        escape (str): Character used to escape SQL wildcards. If this character is followed
        by either a _ or a % then that matches the literal character _ or %. For example,
        if escape == ^, then ^% -> %. If there is no escape character this will be the empty
        string, but we do not have an optimized path.

        make_output_lowercase(bool): Should the output pattern be converted to lowercase.
        For case insensitive matching we convert everything to lowercase. However, we cannot
        just convert the whole pattern because the escape character must remain case sensitive.


    Returns:
        Tuple[str, bool, bool, bool, bool]: The modified string and a series of variables
        used for deciding which operator needs to be used to compute the result.
    """
    # At high level we iterate through the string one character
    # at a time checking its value. In this process we create several
    # "groups", which are just batches of characters that we process all
    # at once (escape or convert to lowercase). When iterating can have 1 of 3 cases.
    #
    # Case 1: Escape character match
    #
    #   If we match the escape character then we check if the next character
    #   is a valid SQL wildcard. If so we append the current group
    #   and append the literal wildcard character.
    #
    # Case 2: Wildcard match
    #   If the character is a wildcard then we append the current group.
    #   In addition we add the Python regex equivalent for the wildcard. % maps to ".*"
    #   and _ maps to "."
    #
    # Case 3: Regular Character match
    #   Do nothing except advance the state/update metadata.
    #
    #
    # Once we have iterated through the string we have the groups necessary to construct a
    # pattern. We combine the groups together and depending on metadata either output a regular
    # string or a Python regex (as well as other booleans used for optimizations). Since we don't
    # know if a string is a regex until we finish, we keep two copies of the groups, one for if
    # we create a regex and one for if we keep it a regular string. Here is an example of splitting
    # a string into groups:
    #
    # pattern="St.ar_er", escape=""
    #   This has one wildcard, so the string effectively has 3 groups: "St.ar", "_", "er". Our two
    #   lists looks like:
    #
    #   escaped_lst (regex): ["St\\.ar", ".", "er"] - Note we must escape the "." to a literal.
    #
    #   unescaped_lst (regular string): ["St.ar", "er"] - We omit the wildcard because this
    #           list can't be chosen in this case.

    def append_group(unescaped_lst, escaped_lst, pattern, group_start, group_end):
        """Append a group of characters from the pattern to the two lists.
        For escaped_lst the pattern should be escaped to avoid conflicts with
        Python regex special characters.

        Args:
            unescaped_lst (List[str]): List of unescaped groups. Used by the regular
                String path.
            escaped_lst (List[str]): List of unescaped groups. Used by the regular
                String path.
            pattern (str): The whole pattern
            group_start (int): Index for the start of the group.
            group_end (int): Index for the end of the group (non-inclusive)
        """
        if group_start != group_end:
            group = pattern[group_start:group_end]
            # Make the group lowercase before escape so escape characters remain
            # accurate.
            if make_output_lowercase:
                group = group.lower()
            unescaped_lst.append(group)
            # Escape any possible Python wildcards
            escaped_lst.append(re.escape(group))

    # List of groups that have been escaped.
    escaped_lst = []
    # List of groups that haven't been escaped
    unescaped_lst = []
    # Wildcards to check for SQL
    sql_wildcards = ("%", "_")

    # Metadata we store to enable optimizations. These can tell us
    # if a string requires a regex for example.

    # Track the first non percent location
    # for tracking if we have all % or leading %
    first_non_percent = -1
    # Track info about the size of a percent group
    # for determining trailing %
    group_is_percent = False
    percent_group_size = 0
    # Get information about the first non starting percent
    # index. This is used to determine if we need a regex
    # (we can skip regex if the only % groups are at start
    # and end and there is no _ character).
    first_non_start_percent = -1

    # End of metadata

    # Index starting the current group
    group_start = 0
    # Should we output a regex or regular String
    requires_regex = False
    pattern_size = len(pattern)
    i = 0
    while i < pattern_size:
        current_char = pattern[i]
        # Case 1: Escape character match
        # If we match escape followed by a wildcard then this is the literal wildcard.
        if (
            current_char == escape
            and (i < (pattern_size - 1))
            and pattern[i + 1] in sql_wildcards
        ):
            # Add the previous group if it exists
            append_group(unescaped_lst, escaped_lst, pattern, group_start, i)
            # Update our metadata to indicate that the current group
            # is not a %. In addition, we indicate the first non-percent
            # character has been reached.
            group_is_percent = False
            if first_non_percent == -1:
                first_non_percent = i
            # Append the wildcard. To future proof against new
            # re special character changes in later Python versions
            # we escape even though its not necessary now.
            wildcard = pattern[i + 1]
            unescaped_lst.append(wildcard)
            escaped_lst.append(re.escape(wildcard))
            # Skip the character and the wildcard for the next group
            group_start = i + 2
            i += 2
        else:
            # Case 2: Wildcard Match
            if current_char in sql_wildcards:
                # Add the previous group if it exists
                append_group(unescaped_lst, escaped_lst, pattern, group_start, i)
                # Next group will start after this section
                group_start = i + 1
                if current_char == "%":
                    if not group_is_percent:
                        percent_group_size = 0
                    group_is_percent = True
                    percent_group_size += 1
                    # We can omit any leading % as an
                    # optimization.
                    if first_non_percent != -1:
                        # Replace the wildcards. We can avoid
                        # unescaped_lst because it can't be chosen
                        # if there any kept wild cards.
                        escaped_lst.append(".*")
                        # If first_non_percent != -1 and first_non_start_percent == -1
                        # then we know we have reached our first % that isn't at the start
                        # of the string. As a result we update our metadata.
                        if first_non_start_percent == -1:
                            first_non_start_percent = i
                else:
                    # We are not optimized for _ yet, so
                    # we always require a regex.
                    requires_regex = True
                    # Update our metadata to indicate that the current group
                    # is not a %. In addition, we indicate the first non-percent
                    # character has been reached.
                    group_is_percent = False
                    if first_non_percent == -1:
                        first_non_percent = i

                    # Replace the wildcards. We can avoid
                    # unescaped_lst because it can't be chosen
                    # if there any kept wild cards.
                    escaped_lst.append(".")
            # Case 3: Regular Character match
            else:
                # Update our metadata to indicate that the current group
                # is not a %. In addition, we indicate the first non-percent
                # character has been reached.
                group_is_percent = False
                if first_non_percent == -1:
                    first_non_percent = i
            i += 1

    # If we didn't trail with a wildcard append the final group.
    append_group(unescaped_lst, escaped_lst, pattern, group_start, len(pattern))

    # Set the metadata for the output flags. We are basically checking for the following
    # information:
    #
    # requires_regex - Did the regex contain a % in the middle or a _ anywhere (not escaped).
    #   For example "b%t" or "_t" -> True, but "%s" -> False.
    #
    # must_match_start: Was the first character a non % character (or do we have an empty string)
    #
    # must_match_end: Was the last character a % that's not escaped.
    #
    # match_anything: Is the string entirely %

    # Determine if we have an internal %. This mean
    # we have a percent after the start and it is
    # not the trailing group.
    has_internal_percent = first_non_start_percent != -1 and (
        (not group_is_percent)
        or ((first_non_start_percent + percent_group_size) != len(pattern))
    )

    # Determine if we need a regex.
    requires_regex = requires_regex or has_internal_percent

    # If we have the empty string for pattern or don't
    # start with %s then we must match the start.
    must_match_start = (first_non_percent == 0) or (len(pattern) == 0)

    # If we don't end in a percent we must match the end
    must_match_end = not group_is_percent

    # If we have all % we are always True
    match_anything = first_non_percent == -1 and len(pattern) > 0

    # Create the final pattern depending on if we need a regex.
    if requires_regex:
        # Update the regex to include ^ and $ if necessary.

        # We can remove any trailing percents if we don't match the end
        if must_match_end:
            # If we must match the end append a $
            escaped_lst.append("$")
        else:
            # We can omit any trailing % as an optimization
            escaped_lst = escaped_lst[:-percent_group_size]

        if must_match_start:
            # If we must match the start insert a ^
            escaped_lst = ["^"] + escaped_lst

        # Regex uses the escaped list
        target_list = escaped_lst
    else:
        # Non-regex must use the unescaped list
        target_list = unescaped_lst

    final_pattern = "".join(target_list)

    return (
        final_pattern,
        requires_regex,
        must_match_start,
        must_match_end,
        match_anything,
    )


# Compute the patterns at runtime by compiling the compile time code with numba.
convert_sql_pattern_to_python_runtime = register_jitable(
    convert_sql_pattern_to_python_compile_time
)


def like_kernel(
    arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


@overload(like_kernel, no_unliteral=True)
def overload_like_kernel(
    arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
):
    """BodoSQL array kernel to implement the SQL like and ilike
    operations.

    Args:
        arr (types.Type): A string scalar or column.
        pattern (types.Type): A scalar or column for the pattern.
        escape (types.StringLiteral): A scalar or column for the escape character. If
            there is no escape this will be the empty string and a string literal.
        case_insensitive (types.BooleanLiteral): Is the operation case insensitive.
            ilike=True, like=False

    Returns:
        types.Type: A boolean array or scalar for if the arr matches.
    """
    if not is_overload_constant_bool(case_insensitive):  # pragma: no cover
        raise_bodo_error("like_kernel(): 'case_insensitive' must be a constant boolean")

    # Note: We don't check case_insensitive because it must be a boolean literal.
    for i, arg in enumerate((arr, pattern, escape)):
        if isinstance(arg, types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.like_kernel",
                [
                    "arr",
                    "pattern",
                    "escape",
                    "case_insensitive",
                    "dict_encoding_state",
                    "func_id",
                ],
                i,
                default_map={"dict_encoding_state": None, "func_id": -1},
            )

    if is_overload_constant_str(pattern) and is_overload_constant_str(escape):
        # Take an optimized path if the pattern and escape are both literals.
        # If either one is not a literal then we cannot compute the pattern
        # at compile time.
        def impl(
            arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
        ):  # pragma: no cover
            return like_kernel_const_pattern_util(
                arr, pattern, escape, case_insensitive, dict_encoding_state, func_id
            )

    elif bodo.utils.utils.is_array_typ(pattern, True) or bodo.utils.utils.is_array_typ(
        escape, True
    ):

        def impl(
            arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
        ):  # pragma: no cover
            # Note: We don't include dict_encoding_state or func_id because we don't have a way
            # to cache the data yet.
            return like_kernel_arr_pattern_util(arr, pattern, escape, case_insensitive)

    else:

        def impl(
            arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
        ):  # pragma: no cover
            return like_kernel_scalar_pattern_util(
                arr, pattern, escape, case_insensitive, dict_encoding_state, func_id
            )

    return impl


def like_kernel_const_pattern_util(
    arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


@overload(like_kernel_const_pattern_util, no_unliteral=True)
def overload_like_kernel_const_pattern_util(
    arr, pattern, escape, case_insensitive, dict_encoding_state, func_id
):
    """Implementation of the SQL like and ilike kernels with both the pattern and escape
    are constant strings. In this case the pattern is computed at compile time and
    the generated code is optimized based on the pattern

    Args:
        arr (types.Type): A string scalar or column.
        pattern (types.StringLiteral): A string literal for the pattern.
        escape (types.StringLiteral): A string literal for the escape character. If
            there is no escape this will be the empty string.
        case_insensitive (types.BooleanLiteral): Is the operation case insensitive.
            ilike=True, like=False

    Raises:
        BodoError: One of the inputs doesn't match the expected format.

    Returns:
        types.Type: A boolean array or scalar for if the arr matches.
    """
    verify_string_arg(arr, "LIKE_KERNEL", "arr")
    if not is_overload_constant_str(pattern):  # pragma: no cover
        raise_bodo_error("like_kernel(): 'pattern' must be a constant string")
    const_pattern = get_overload_const_str(pattern)
    if not is_overload_constant_str(pattern):  # pragma: no cover
        raise_bodo_error("like_kernel(): 'escape' must be a constant string")
    const_escape = get_overload_const_str(escape)
    if len(const_escape) > 1:  # pragma: no cover
        raise BodoError(
            "like_kernel(): 'escape' must be a single character if provided."
        )
    if not is_overload_constant_bool(case_insensitive):  # pragma: no cover
        raise_bodo_error("like_kernel(): 'case_insensitive' must be a constant boolean")
    is_case_insensitive = get_overload_const_bool(case_insensitive)

    arg_names = [
        "arr",
        "pattern",
        "escape",
        "case_insensitive",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [arr, pattern, escape, case_insensitive, dict_encoding_state, func_id]
    # By definition only the array can contain nulls.
    propagate_null = [True, False, False, False, False, False]
    out_dtype = bodo.boolean_array_type
    # Some paths have prefix and/or need extra globals
    prefix_code = None
    extra_globals = {}

    # Convert the SQL pattern to a Python pattern
    (
        python_pattern,
        requires_regex,
        must_match_start,
        must_match_end,
        match_anything,
    ) = convert_sql_pattern_to_python_compile_time(
        const_pattern, const_escape, is_case_insensitive
    )
    if match_anything:
        scalar_text = "res[i] = True\n"
    else:
        if is_case_insensitive:
            # To match non-wildcards make everything lower case
            scalar_text = "arg0 = arg0.lower()\n"
        else:
            scalar_text = ""
        if requires_regex:
            extra_globals["matcher"] = re.compile(python_pattern)
            scalar_text += "res[i] = bool(matcher.search(arg0))\n"
        else:
            extra_globals["python_pattern"] = python_pattern
            if must_match_start and must_match_end:
                scalar_text += "res[i] = arg0 == python_pattern\n"
            elif must_match_start:
                scalar_text += "res[i] = arg0.startswith(python_pattern)\n"
            elif must_match_end:
                scalar_text += "res[i] = arg0.endswith(python_pattern)\n"
            else:
                scalar_text += "res[i] = python_pattern in arg0\n"

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


def like_kernel_arr_pattern_util(
    arr, pattern, escape, case_insensitive
):  # pragma: no cover
    pass


@overload(like_kernel_arr_pattern_util, no_unliteral=True)
def overload_like_kernel_arr_pattern_util(arr, pattern, escape, case_insensitive):
    """Implementation of the SQL like and ilike kernels where either the pattern or the escape
    are an array. In this case the pattern must be computed on each iteration of the loop
    at runtime.

    Args:
        arr (types.Type): A string scalar or column.
        pattern (types.Type): A scalar scalar or column for the pattern.
        escape (types.StringLiteral): A scalar scalar or column for the escape character.
        case_insensitive (types.BooleanLiteral): Is the operation case insensitive.
            ilike=True, like=False

    Returns:
        types.Type: A boolean array or scalar for if the arr matches.
    """
    verify_string_arg(arr, "LIKE_KERNEL", "arr")
    verify_string_arg(pattern, "LIKE_KERNEL", "arr")
    verify_string_arg(escape, "LIKE_KERNEL", "arr")
    if not is_overload_constant_bool(case_insensitive):  # pragma: no cover
        raise_bodo_error("like_kernel(): 'case_insensitive' must be a constant boolean")
    is_case_insensitive = get_overload_const_bool(case_insensitive)

    arg_names = ["arr", "pattern", "escape", "case_insensitive"]
    arg_types = [arr, pattern, escape, case_insensitive]
    # By definition case_insensitive cannot be null.
    propagate_null = [True, True, True, False]
    out_dtype = bodo.boolean_array_type
    extra_globals = {
        "convert_sql_pattern": convert_sql_pattern_to_python_runtime,
        # Dictionary encoding optimized function.
        "convert_sql_pattern_dict_encoding": convert_sql_pattern_to_python_runtime_dict_encoding,
        # Cache functions for the 'use_multiple_dict_encoding_path' case
        "alloc_like_kernel_cache": bodo.libs.array._alloc_like_kernel_cache,
        "add_to_like_kernel_cache": bodo.libs.array._add_to_like_kernel_cache,
        "check_like_kernel_cache": bodo.libs.array._check_like_kernel_cache,
        "dealloc_like_kernel_cache": bodo.libs.array._dealloc_like_kernel_cache,
    }
    prefix_code = None
    scalar_text = ""
    pattern_conversion_use_dict_encoding = not is_overload_none(arr) and (
        (
            pattern == bodo.dict_str_arr_type
            and types.unliteral(escape) == types.unicode_type
        )
        or (
            types.unliteral(pattern) == types.unicode_type
            and escape == bodo.dict_str_arr_type
        )
    )
    # If we have two dictionary encoded arrays we can use the indices array to
    # cache results.
    use_multiple_dict_encoding_path = (
        arr == bodo.dict_str_arr_type and pattern_conversion_use_dict_encoding
    )
    prefix_code = ""
    suffix_code = None
    if is_case_insensitive and not is_overload_none(arr):
        # Lower the input once at the onset to enable dict encoding.
        # TODO: Move as a requirement for this kernel in the codegen step. This will
        # allow removing redundant lower calls.
        prefix_code += "arr = bodo.libs.bodosql_array_kernels.lower(arr)\n"
    if pattern_conversion_use_dict_encoding:
        # If we are using dictionary encoding we convert the patterns before the for loop
        prefix_code += "(python_pattern_arr, requires_regex_arr, must_match_start_arr, must_match_end_arr, match_anything_arr) = convert_sql_pattern_dict_encoding(pattern, escape, case_insensitive)\n"
        # We will access these outputs using the indices array.
        if pattern == bodo.dict_str_arr_type:
            prefix_code += "conversion_indices_arr = pattern._indices\n"
        else:
            prefix_code += "conversion_indices_arr = escape._indices\n"
    if use_multiple_dict_encoding_path:
        # If both the pattern and arr are dictionary encoded we can cache results
        # since we can't compute directly on dictionaries.
        prefix_code += "arr_indices = arr._indices\n"
        # We use a cache defined in C++ for better performance due to Numba dictionaries not being the most performant.
        # Set the reserved size of the hashmap to min(dict_size(arr) * size(python_pattern_arr), len(arr))
        # to reduce re-allocations as much as possible.
        prefix_code += "cache_reserve_size = min(len(arr._data) * len(python_pattern_arr), len(arr_indices))\n"
        prefix_code += "idx_cache = alloc_like_kernel_cache(cache_reserve_size)\n"
        # Add corresponding suffix code to deallocate the cache at the end.
        if suffix_code is None:
            suffix_code = ""
        suffix_code += "dealloc_like_kernel_cache(idx_cache)\n"
    # Convert the pattern on each iteration since it may change.
    # XXX Consider moving to a helper function to keep the IR smaller?
    if use_multiple_dict_encoding_path:
        scalar_text += "a_idx = arr_indices[i]\n"
        # use_multiple_dict_encoding_path = True implies pattern_conversion_use_dict_encoding = True
        scalar_text += "c_idx = conversion_indices_arr[i]\n"
        scalar_text += "cache_out = check_like_kernel_cache(idx_cache, a_idx, c_idx)\n"
        scalar_text += "if cache_out != -1:\n"  # i.e. in the cache
        scalar_text += "  res[i] = bool(cache_out)\n"
        scalar_text += "  continue\n"
    if pattern_conversion_use_dict_encoding:
        if not use_multiple_dict_encoding_path:
            scalar_text += "c_idx = conversion_indices_arr[i]\n"
        scalar_text += "(python_pattern, requires_regex, must_match_start, must_match_end, match_anything) = python_pattern_arr[c_idx], requires_regex_arr[c_idx], must_match_start_arr[c_idx], must_match_end_arr[c_idx], match_anything_arr[c_idx]\n"
    else:
        # Manually allocate arg1
        if bodo.utils.utils.is_array_typ(pattern, False):
            scalar_text += "arg1 = pattern[i]\n"
        else:
            scalar_text += "arg1 = pattern\n"
        # Manually allocate arg2
        if bodo.utils.utils.is_array_typ(escape, False):
            scalar_text += "arg2 = escape[i]\n"
        else:
            scalar_text += "arg2 = escape\n"
        scalar_text += "(python_pattern, requires_regex, must_match_start, must_match_end, match_anything) = convert_sql_pattern(arg1, arg2, case_insensitive)\n"
    # Manually allocate arg0 if we don't have a cache hit.
    if bodo.utils.utils.is_array_typ(arr, False):
        scalar_text += "arg0 = arr[i]\n"
    else:
        scalar_text += "arg0 = arr\n"
    scalar_text += "if match_anything:\n"
    scalar_text += "  result = True\n"
    scalar_text += "elif requires_regex:\n"
    scalar_text += "  matcher = re.compile(python_pattern)\n"
    scalar_text += "  result = bool(matcher.search(arg0))\n"
    scalar_text += "elif must_match_start and must_match_end:\n"
    scalar_text += "  result = arg0 == python_pattern\n"
    scalar_text += "elif must_match_start:\n"
    scalar_text += "  result = arg0.startswith(python_pattern)\n"
    scalar_text += "elif must_match_end:\n"
    scalar_text += "  result = arg0.endswith(python_pattern)\n"
    scalar_text += "else:\n"
    scalar_text += "  result = python_pattern in arg0\n"
    if use_multiple_dict_encoding_path:
        # Store the result in the cache.
        scalar_text += "add_to_like_kernel_cache(idx_cache, a_idx, c_idx, result)\n"
    scalar_text += "res[i] = result\n"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        prefix_code=prefix_code,
        extra_globals=extra_globals,
        # Manually support dictionary encoding in this implementation
        # to ensure we are more flexible.
        support_dict_encoding=False,
        suffix_code=suffix_code,
        # Allocating scalars can have significant overhead for the dictionary case
        # so we manually check if after checking the dictionary.
        alloc_array_scalars=False,
    )


def like_kernel_scalar_pattern_util(
    arr, pattern, escape, case_insensitive, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


@overload(like_kernel_scalar_pattern_util, no_unliteral=True)
def overload_like_kernel_scalar_pattern_util(
    arr, pattern, escape, case_insensitive, dict_encoding_state, func_id
):
    """Implementation of the SQL like and ilike kernels where both the pattern and the escape
    are scalars. In this case the pattern must be computed on each iteration of the loop
    at runtime.

    Args:
        arr (types.Type): A string scalar or column.
        pattern (types.Type): A scalar scalar or column for the pattern.
        escape (types.StringLiteral): A scalar scalar or column for the escape character.
        case_insensitive (types.BooleanLiteral): Is the operation case insensitive.
            ilike=True, like=False

    Returns:
        types.Type: A boolean array or scalar for if the arr matches.
    """
    verify_string_arg(arr, "LIKE_KERNEL", "arr")
    verify_string_arg(pattern, "LIKE_KERNEL", "arr")
    verify_string_arg(escape, "LIKE_KERNEL", "arr")
    if not is_overload_constant_bool(case_insensitive):  # pragma: no cover
        raise_bodo_error("like_kernel(): 'case_insensitive' must be a constant boolean")
    is_case_insensitive = get_overload_const_bool(case_insensitive)

    arg_names = [
        "arr",
        "pattern",
        "escape",
        "case_insensitive",
        "dict_encoding_state",
        "func_id",
    ]
    arg_types = [arr, pattern, escape, case_insensitive, dict_encoding_state, func_id]
    # By definition case_insensitive cannot be null.
    propagate_null = [True, True, True, False, False, False]
    out_dtype = bodo.boolean_array_type
    extra_globals = {
        "convert_sql_pattern": convert_sql_pattern_to_python_runtime,
        # A dummy matcher used to ensure type stability if we don't
        # need a regex so we don't always have to call compile.
        "global_matcher": re.compile(""),
    }
    if is_overload_none(pattern) or is_overload_none(escape):
        # Avoid typing issues due to the prefix code if pattern or
        # escape is null and arr is an array.
        prefix_code = None
        scalar_text = "pass"
    else:
        # Convert the pattern as prefix code so we don't compute the pattern multiple times on each iteration.
        prefix_code = "(python_pattern, requires_regex, must_match_start, must_match_end, match_anything) = convert_sql_pattern(pattern, escape, case_insensitive)\n"
        # If we need to compile the code generate a global matcher. Otherwise match a lowered
        # matcher for type stability
        prefix_code += "if requires_regex:\n"
        prefix_code += "    matcher = re.compile(python_pattern)\n"
        prefix_code += "else:\n"
        prefix_code += "    matcher = global_matcher\n"
        # Generate the scalar code.
        if is_case_insensitive:
            # If we are case insensitive we converted the pattern and arg
            # to lower case to all safe comparisons.
            scalar_text = "arg0 = arg0.lower()\n"
        else:
            scalar_text = ""
        # XXX Consider moving to a helper function to keep the IR smaller?
        scalar_text += "if match_anything:\n"
        scalar_text += "  res[i] = True\n"
        scalar_text += "elif requires_regex:\n"
        scalar_text += "  res[i] = bool(matcher.search(arg0))\n"
        scalar_text += "elif must_match_start and must_match_end:\n"
        scalar_text += "  res[i] = arg0 == python_pattern\n"
        scalar_text += "elif must_match_start:\n"
        scalar_text += "  res[i] = arg0.startswith(python_pattern)\n"
        scalar_text += "elif must_match_end:\n"
        scalar_text += "  res[i] = arg0.endswith(python_pattern)\n"
        scalar_text += "else:\n"
        scalar_text += "  res[i] = python_pattern in arg0\n"

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


def convert_sql_pattern_to_python_runtime_dict_encoding(
    pattern, escape, case_insensitive
):  # pragma: no cover
    pass


@overload(convert_sql_pattern_to_python_runtime_dict_encoding)
def overload_convert_sql_pattern_to_python_runtime_dict_encoding(
    pattern, escape, case_insensitive
):  # pragma: no cover
    """Implementation for convert_sql_pattern_to_python_runtime that enables leveraging
    dictionary encoding to both reduce memory usage and improve performance.
    convert_sql_pattern_to_python_runtime seems to be particularly slow in the array pattern
    case based on experimentation in the query.

    Args:
        pattern (Union[types.unicode_type | bodo.dict_str_arr_type]): The pattern to convert. This is either
            a scalar or a dictionary encoded array. If this is an array escape must be a scalar.
        escape (Union[types.unicode_type | bodo.dict_str_arr_type]): The escape character. This is either
            a scalar or a dictionary encoded array. If this is an array pattern must be a scalar.
        case_insensitive (types.boolean): Is the conversion case insensitive.

    Returns: Tuple[bodo.string_array, bodo.boolean_array_type, bodo.boolean_array_type, bodo.boolean_array_type, bodo.boolean_array_type]
    """
    if pattern == bodo.dict_str_arr_type:
        assert (
            types.unliteral(escape) == types.unicode_type
        ), "escape must be a scalar if pattern is a dictionary encoded array"
        dict_input = "pattern"
        call_inputs = "dict_arr[i], escape, case_insensitive"
    else:
        assert (
            escape == bodo.dict_str_arr_type
        ), "At least one of pattern or escape must be a dictionary encoded array"
        assert (
            types.unliteral(pattern) == types.unicode_type
        ), "pattern must be a scalar if escape is a dictionary encoded array"
        dict_input = "escape"
        call_inputs = "pattern, dict_arr[i], case_insensitive"
    func_text = f"""def impl(pattern, escape, case_insensitive):
        # Fetch the actual dictionary.
        dict_arr = {dict_input}._data
        n = len(dict_arr)
        # Allocate the output arrays.
        python_pattern_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
        requires_regex = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        must_match_start = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        must_match_end = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        match_anything = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in range(n):
            if not bodo.libs.array_kernels.isna(dict_arr, i):
                # Convert the pattern.
                python_pattern_arr[i], requires_regex[i], must_match_start[i], must_match_end[i], match_anything[i] = convert_sql_pattern_to_python_runtime(
                    {call_inputs}
                )
            else:
                # Ensure we don't have any issues due to skipping NA values for strings. All others
                # won't matter.
                python_pattern_arr[i] = ""
        return (python_pattern_arr, requires_regex, must_match_start, must_match_end, match_anything)"""
    local_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "np": np,
            "convert_sql_pattern_to_python_runtime": convert_sql_pattern_to_python_runtime,
        },
        local_vars,
    )
    return local_vars["impl"]
