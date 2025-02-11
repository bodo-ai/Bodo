"""
Implements like array kernels that are specific to BodoSQL
"""

import re

import numpy as np
from numba.core import types
from numba.extending import overload, register_jitable

import bodo
import bodosql
from bodo.ir.filter import convert_sql_pattern_to_python_compile_time
from bodo.utils.typing import (
    BodoError,
    get_overload_const_bool,
    get_overload_const_str,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_none,
    raise_bodo_error,
)
from bodosql.kernels.array_kernel_utils import (
    gen_vectorized,
    unopt_argument,
    verify_string_arg,
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
                "bodosql.kernels.like_kernel",
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
        prefix_code += "arr = bodosql.kernels.lower(arr)\n"
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
        assert types.unliteral(escape) == types.unicode_type, (
            "escape must be a scalar if pattern is a dictionary encoded array"
        )
        dict_input = "pattern"
        call_inputs = "dict_arr[i], escape, case_insensitive"
    else:
        assert escape == bodo.dict_str_arr_type, (
            "At least one of pattern or escape must be a dictionary encoded array"
        )
        assert types.unliteral(pattern) == types.unicode_type, (
            "pattern must be a scalar if escape is a dictionary encoded array"
        )
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
            "bodosql": bodosql,
            "np": np,
            "convert_sql_pattern_to_python_runtime": convert_sql_pattern_to_python_runtime,
        },
        local_vars,
    )
    return local_vars["impl"]
