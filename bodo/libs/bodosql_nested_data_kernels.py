# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to ARRAY utilities
"""

import types

import numba

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    get_overload_const_bool,
    is_overload_constant_bool,
    is_overload_none,
    raise_bodo_error,
    unwrap_typeref,
)


@numba.generated_jit(nopython=True)
def object_keys(arr):
    """
    Handles cases where OBJECT_KEYS receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.object_keys_util",
            ["arr"],
            0,
        )

    def impl(arr):  # pragma: no cover
        return object_keys_util(arr)

    return impl


@numba.generated_jit(nopython=True)
def object_keys_util(arr):
    """
    A dedicated kernel for the SQL function OBJECT which takes in an
    a JSON value (either a scalar, or a map/struct array) and returns
    an array of all of its keys.

    Args:
        arr (array scalar/array item array): the JSON value(s)

    Returns:
        string array / string array array: the keys of the JSON value(s)
    """
    arg_names = ["arr"]
    arg_types = [arr]
    propagate_null = [True]
    # TODO: see if we can optimize this for dictionary encoding, at least for the struct cases?
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(bodo.string_array_type)
    typ = arr
    if bodo.hiframes.pd_series_ext.is_series_type(typ):
        typ = typ.data
    if bodo.utils.utils.is_array_typ(typ) and isinstance(
        typ.dtype, bodo.libs.struct_arr_ext.StructType
    ):
        scalar_text = (
            f"res[i] = bodo.libs.str_arr_ext.str_list_to_array({list(typ.dtype.names)})"
        )
    elif isinstance(typ, bodo.libs.struct_arr_ext.StructType):
        scalar_text = (
            f"res[i] = bodo.libs.str_arr_ext.str_list_to_array({list(typ.names)})"
        )
    elif isinstance(typ, bodo.libs.map_arr_ext.MapArrayType) or (
        isinstance(typ, types.DictType) and typ.key_type == types.unicode_type
    ):
        scalar_text = "res[i] = bodo.libs.str_arr_ext.str_list_to_array(list(arg0))\n"
    elif typ == bodo.none:
        scalar_text = "res[i] = None"
    else:
        raise_bodo_error(f"object_keys: unsupported type {arr}")
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
    )


@numba.generated_jit(nopython=True)
def to_array(arr, dtype, dict_encoding_state=None, func_id=-1):
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument("bodo.libs.bodosql_array_kernels.to_array", ["arr"], 0)

    def impl(arr, dtype, dict_encoding_state=None, func_id=-1):  # pragma: no cover
        return to_array_util(arr, dtype, dict_encoding_state, func_id)

    return impl


@numba.generated_jit(nopython=True)
def to_array_util(arr, dtype, dict_encoding_state, func_id):
    arg_names = ["arr", "dtype", "dict_encoding_state", "func_id"]
    arg_types = [arr, dtype, dict_encoding_state, func_id]
    propagate_null = [True, False, False, False]
    arr_dtype = unwrap_typeref(dtype)
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(arr_dtype)
    scalar_text = (
        "res[i] = bodo.utils.conversion.coerce_scalar_to_array(arg0, 1, arg1, False)"
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
def arrays_overlap(array_0, array_1, is_scalar_0=False, is_scalar_1=False):
    """
    Handles cases where ARRAYS_OVERLAP receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [array_0, array_1]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.arrays_overlap",
                ["array_0", "array_1", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(
        array_0, array_1, is_scalar_0=False, is_scalar_1=False
    ):  # pragma: no cover
        return arrays_overlap_util(array_0, array_1, is_scalar_0, is_scalar_1)

    return impl


@numba.generated_jit(nopython=True)
def arrays_overlap_util(array_0, array_1, is_scalar_0, is_scalar_1):
    """
    A dedicated kernel for the SQL function ARRAYS_OVERLAP which takes in two
    arrays (or columns of arrays) and returns whether they have overlap

    Args:
        arr (array scalar/array item array): the first array(s) to compare
        arr (array scalar/array item array): the second array(s) to compare
        is_single_row (boolean): if true, treats the inputs as scalar arrays

    Returns:
        boolean scalar/vector: whether the arrays have any common elements
    """
    is_scalar_0_bool = get_overload_const_bool(is_scalar_0)
    is_scalar_1_bool = get_overload_const_bool(is_scalar_1)
    arg_names = ["array_0", "array_1", "is_scalar_0", "is_scalar_1"]
    arg_types = [array_0, array_1, is_scalar_0, is_scalar_1]
    propagate_null = [True, True, False, False]
    out_dtype = bodo.boolean_array_type
    scalar_text = "has_overlap = False\n"
    scalar_text += "for idx0 in range(len(arg0)):\n"
    scalar_text += "   null0 = bodo.libs.array_kernels.isna(arg0, idx0)\n"
    scalar_text += "   for idx1 in range(len(arg1)):\n"
    scalar_text += "      null1 = bodo.libs.array_kernels.isna(arg1, idx1)\n"
    scalar_text += "      if (null0 and null1) or ((not null0) and (not null1) and bodo.libs.bodosql_array_kernels.semi_safe_equals(arg0[idx0], arg1[idx1])):\n"
    scalar_text += "         has_overlap = True\n"
    scalar_text += "         break\n"
    scalar_text += "   if has_overlap: break\n"
    scalar_text += "res[i] = has_overlap"
    are_arrays = [not is_scalar_0_bool, not is_scalar_1_bool, False, False]
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


@numba.generated_jit(nopython=True)
def array_position(elem, container, is_scalar_0=False, is_scalar_1=False):
    """
    Handles cases where ARRAYS_OVERLAP receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [elem, container, is_scalar_0, is_scalar_1]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_position",
                ["elem", "container", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(elem, container, is_scalar_0=False, is_scalar_1=False):  # pragma: no cover
        return array_position_util(elem, container, is_scalar_0, is_scalar_1)

    return impl


@numba.generated_jit(nopython=True)
def array_position_util(elem, container, elem_is_scalar, container_is_scalar):
    """
    A dedicated kernel for the SQL function ARRAY_POSITION which takes in an
    element and an array (or column of arrays) and returns the zero-indexed
    position of the first occurrence of the element in the array (including nulls).

    Args:
        elem (array scalar/array item array): the element(s) to look for
        container (array scalar/array item array): the array(s) to search through
        elem_is_scalar (boolean): if true, treats the first argument as a scalar even if it is an array.
        container_is_scalar (boolean): if true, treats the second argument as a scalar even if it is an array.

    Returns:
        integer scalar/vector: the index of the first match to elem in container
        (zero-indexed), or null if there is no match.
    """
    elem_is_scalar_bool = get_overload_const_bool(elem_is_scalar)
    container_is_scalar_bool = get_overload_const_bool(container_is_scalar)
    arg_names = ["elem", "container", "elem_is_scalar", "container_is_scalar"]
    arg_types = [elem, container, elem_is_scalar, container_is_scalar]
    propagate_null = [False, True, False]
    out_dtype = bodo.IntegerArrayType(types.int32)
    are_arrays = [not elem_is_scalar_bool, not container_is_scalar_bool, False, False]
    scalar_text = "match = -1\n"
    if elem == bodo.none:
        scalar_text += "null0 = True\n"
    elif are_arrays[0]:
        scalar_text += "null0 = bodo.libs.array_kernels.isna(elem, i)\n"
    else:
        scalar_text += "null0 = False\n"
    scalar_text += "for idx1 in range(len(arg1)):\n"
    scalar_text += "   null1 = bodo.libs.array_kernels.isna(arg1, idx1)\n"
    scalar_text += "   if (null0 and null1) or ((not null0) and (not null1) and bodo.libs.bodosql_array_kernels.semi_safe_equals(arg0, arg1[idx1])):\n"
    scalar_text += "         match = idx1\n"
    scalar_text += "         break\n"
    scalar_text += "if match == -1:\n"
    scalar_text += "   bodo.libs.array_kernels.setna(res, i)\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = match"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


@numba.generated_jit(nopython=True)
def array_to_string(arr, separator):
    """
    Handles cases where ARRAY_TO_STRING receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, separator]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_to_string",
                ["arr", "separator"],
                i,
            )

    def impl(arr, separator):  # pragma: no cover
        return array_to_string_util(arr, separator)

    return impl


@numba.generated_jit(nopython=True)
def array_to_string_util(arr, separator):
    """
    A dedicated kernel for the SQL function ARRAY_TO_STRING which takes in an
           array, (or arrray column) and a separator string (or string column), then
           casts the array to string and add separators
    Args:
        arr (array scalar/array item array): the arrray(s) to be cast to string
        separator (string array/series/scalar): the separator to add to the string
    Returns:
        A string scalar/array: the result string(s)
    """
    arg_names = ["arr", "separator"]
    arg_types = [arr, separator]
    propagate_null = [True, True]
    out_dtype = bodo.string_array_type
    scalar_text = "arr_str = ''\n"
    scalar_text += "if len(arg0) > 0:\n"
    scalar_text += (
        "    arr_str = '' if bodo.libs.array_kernels.isna(arg0, 0) else str(arg0[0])\n"
    )
    scalar_text += "    for j in bodo.prange(len(arg0) - 1):\n"
    scalar_text += "        arr_str += arg1 + ('' if bodo.libs.array_kernels.isna(arg0, j + 1) else str(arg0[j + 1]))\n"
    scalar_text += "res[i] = arr_str"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        array_is_scalar=True,
        # Protect against the input being a "scalar" string/dict array.
        support_dict_encoding=False,
    )


@numba.generated_jit(nopython=True)
def array_size(arr, is_single_row):
    """
    Handles cases where ARRAY_SIZE receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.array_size",
            [
                "arr",
                "is_single_row",
            ],
            0,
        )

    def impl(arr, is_single_row):  # pragma: no cover
        return array_size_util(arr, is_single_row)

    return impl


@numba.generated_jit(nopython=True)
def array_size_util(arr, is_single_row):
    """
    A dedicated kernel for the SQL function ARRAY_SIZE which takes in an
           array, (or array column). If it is an array it returns the size, if it is a column
           it returns the size of each array in the column.
    Args:
        arr (array scalar/array item array): the array(s) to get the size of
        is_single_row (bool literal): Whether this is called in a single row context, necessary
        to determine whether to return the length of a nested array or an array of the lengths
        of it's children in a case statment
    Returns:
        An integer scalar/array: the result lengths
    """
    if is_overload_none(arr):
        return lambda arr, is_single_row: None

    if not is_overload_constant_bool(is_single_row):  # pragma: no cover
        raise_bodo_error("array_size(): 'is_single_row' must be a constant boolean")

    if not is_array_item_array(arr) and not (
        bodo.utils.utils.is_array_typ(arr) and is_single_row
    ):  # pragma: no cover
        # When not is_single_row only array item ararys are supported
        # When is_single_row then all arrays are supported
        raise_bodo_error(
            f"array_size(): unsupported for type {arr} when is_single_row={is_single_row}"
        )

    # Whether to call len on each element or on arr itself
    arr_is_array = is_array_item_array(arr) and not get_overload_const_bool(
        is_single_row
    )

    scalar_text = "res[i] = len(arg0)"
    arg_names = ["arr", "is_single_row"]
    arg_types = [
        bodo.utils.conversion.coerce_to_array(arr),
        is_single_row,
    ]
    propagate_null = [True, False, False, False]
    out_dtype = bodo.IntegerArrayType(types.int32)
    are_arrays = [arr_is_array] + [False] * 3
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )
