# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to ARRAY utilities
"""

import types

import numba

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import (
    dtype_to_array_type,
    get_overload_const_bool,
    is_bin_arr_type,
    is_overload_constant_bool,
    is_overload_none,
    is_str_arr_type,
    raise_bodo_error,
)
from bodo.utils.utils import is_array_typ


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
    elif (
        isinstance(typ, bodo.libs.map_arr_ext.MapArrayType)
        or (isinstance(typ, types.DictType) and typ.key_type == types.unicode_type)
        or (
            isinstance(typ, bodo.libs.map_arr_ext.MapScalarType)
            and typ.key_arr_type == bodo.string_array_type
        )
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


def array_except(
    arr, to_remove, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    pass


@overload(array_except, no_unliteral=True)
def overload_array_except(arr, to_remove, is_scalar_0=False, is_scalar_1=False):
    """
    Handles cases where ARRAY_EXCEPT receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, to_remove]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_except",
                ["arr", "to_remove", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(arr, to_remove, is_scalar_0=False, is_scalar_1=False):  # pragma: no cover
        return array_except_util(arr, to_remove, is_scalar_0, is_scalar_1)

    return impl


def array_except_util(arr, to_remove, is_scalar_0, is_scalar_1):  # pragma: no cover
    pass


@overload(array_except_util, no_unliteral=True)
def overload_array_except_util(arr, to_remove, is_scalar_0, is_scalar_1):
    """
    A dedicated kernel for the SQL function ARRAY_EXCEPT which takes in
    two arrays (or columns of arrays) and returns the elements of the first
    that do not appear in the second.

    Args:
        arr (array scalar/array item array): the starting values
        to_remove (array scalar/array item array): the array whose
        elements are dropped from arr

    Returns:
        array: arr with the elements of to_remove dropped
    """
    are_arrays = [
        not get_overload_const_bool(is_scalar_0, "array_except", "is_scalar_0"),
        not get_overload_const_bool(is_scalar_1, "array_except", "is_scalar_0"),
        False,
        False,
    ]
    verify_array_arg(arr, not are_arrays[0], "ARRAY_EXCEPT", "arr")
    verify_array_arg(to_remove, not are_arrays[1], "ARRAY_EXCEPT", "to_remove")
    arg_names = ["arr", "to_remove", "is_scalar_0", "is_scalar_1"]
    arg_types = [arr, to_remove, is_scalar_0, is_scalar_1]
    propagate_null = [True] * 2 + [False] * 2
    if are_arrays[0]:
        out_dtype = arr
    else:
        out_dtype = to_remove

    # A boolean array to keep track of which indices from the original
    # array should be copied over, and use it it index into the original
    # at the end in order to produce the final answer.
    scalar_text = "elems_to_keep = np.ones(len(arg0), dtype=np.bool_)\n"

    # A boolean array to keep track of which indices in the second array have
    # found a match in the first array, that way no single index in the second
    # array gets counted more than once.
    scalar_text += "already_matched = np.zeros(len(arg1), dtype=np.bool_)\n"

    # Loop over each element in the original array and update it's index in
    # elems_to_keep to False if any element in the second array matches it,
    # skipping any indices that already had a match.
    scalar_text += "for idx0 in range(len(arg0)):\n"
    scalar_text += "   null0 = bodo.libs.array_kernels.isna(arg0, idx0)\n"
    scalar_text += "   for idx1 in range(len(arg1)):\n"
    scalar_text += "      if already_matched[idx1]: continue\n"
    scalar_text += "      null1 = bodo.libs.array_kernels.isna(arg1, idx1)\n"
    scalar_text += "      if (null0 and null1) or ((not null0) and (not null1) and bodo.libs.bodosql_array_kernels.semi_safe_equals(arg0[idx0], arg1[idx1])):\n"
    scalar_text += "         already_matched[idx1] = True\n"
    scalar_text += "         elems_to_keep[idx0] = False\n"
    scalar_text += "         break\n"
    scalar_text += "res[i] = arg0[elems_to_keep]"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


def array_intersection(
    arr_0, arr_1, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    pass


@overload(array_intersection, no_unliteral=True)
def overload_array_intersection(arr_0, arr_1, is_scalar_0=False, is_scalar_1=False):
    """
    Handles cases where ARRAY_INTERSECTION receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr_0, arr_1]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_intersection",
                ["arr_0", "arr_1", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(arr_0, arr_1, is_scalar_0=False, is_scalar_1=False):  # pragma: no cover
        return array_intersection_util(arr_0, arr_1, is_scalar_0, is_scalar_1)

    return impl


def array_intersection_util(arr_0, arr_1, is_scalar_0, is_scalar_1):  # pragma: no cover
    pass


@overload(array_intersection_util, no_unliteral=True)
def overload_array_intersection_util(arr_0, arr_1, is_scalar_0, is_scalar_1):
    """
    A dedicated kernel for the SQL function ARRAY_INTERSECTION which takes in
    two arrays (or columns of arrays) and returns the intersection of the two
    arrays.

    Args:
        arr_0 (array scalar/array item array): the first array
        arr_1 (array scalar/array item array): the second array
        is_scalar_0 (boolean): whether arr_0 is a scalar array
        is_scalar_1 (boolean): whether arr_1 is a scalar array

    Returns:
        array: the elements that appear in both arr_0 and arr_1. If an element
        appears more than once in either arr_0 or arr_1, it is kept the smaller
        number of times that it appears.
    """
    are_arrays = [
        not get_overload_const_bool(is_scalar_0, "array_intersection", "is_scalar_0"),
        not get_overload_const_bool(is_scalar_1, "array_intersection", "is_scalar_1"),
        False,
        False,
    ]
    verify_array_arg(arr_0, not are_arrays[0], "ARRAY_INTERSECTION", "arr_0")
    verify_array_arg(arr_1, not are_arrays[1], "ARRAY_INTERSECTION", "arr_1")
    arg_names = ["arr_0", "arr_1", "is_scalar_0", "is_scalar_1"]
    arg_types = [arr_0, arr_1, is_scalar_0, is_scalar_1]
    propagate_null = [True] * 2 + [False] * 2
    if are_arrays[0]:
        out_dtype = arr_0
    else:
        out_dtype = arr_1

    # A boolean array to keep track of which indices from the original
    # array should be copied over, and use it it index into the original
    # at the end in order to produce the final answer.
    scalar_text = "elems_to_keep = np.zeros(len(arg0), dtype=np.bool_)\n"

    # A boolean array to keep track of which indices in the second array have
    # found a match in the first array, that way no single index in the second
    # array gets counted more than once.
    scalar_text += "already_matched = np.zeros(len(arg1), dtype=np.bool_)\n"

    # Loop over each element in the original array and update it's index in
    # elems_to_keep to True if any element in the second array matches it,
    # skipping any indices that already had a match.
    scalar_text += "for idx0 in range(len(arg0)):\n"
    scalar_text += "   has_match = False\n"
    scalar_text += "   null0 = bodo.libs.array_kernels.isna(arg0, idx0)\n"
    scalar_text += "   for idx1 in range(len(arg1)):\n"
    scalar_text += "      if already_matched[idx1]: continue\n"
    scalar_text += "      null1 = bodo.libs.array_kernels.isna(arg1, idx1)\n"
    scalar_text += "      if (null0 and null1) or ((not null0) and (not null1) and bodo.libs.bodosql_array_kernels.semi_safe_equals(arg0[idx0], arg1[idx1])):\n"
    scalar_text += "         elems_to_keep[idx0] = True\n"
    scalar_text += "         already_matched[idx1] = True\n"
    scalar_text += "         break\n"
    scalar_text += "res[i] = arg0[elems_to_keep]"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


def array_cat(arr_0, arr_1, is_scalar_0=False, is_scalar_1=False):  # pragma: no cover
    pass


@overload(array_cat, no_unliteral=True)
def overload_array_cat(arr_0, arr_1, is_scalar_0=False, is_scalar_1=False):
    """
    Handles cases where ARRAY_CAT receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr_0, arr_1]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):  # pragma: no cover
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_cat",
                ["arr_0", "arr_1", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(arr_0, arr_1, is_scalar_0=False, is_scalar_1=False):  # pragma: no cover
        return array_cat_util(arr_0, arr_1, is_scalar_0, is_scalar_1)

    return impl


def array_cat_util(arr_0, arr_1, is_scalar_0, is_scalar_1):  # pragma: no cover
    pass


@overload(array_cat_util, no_unliteral=True)
def overload_array_cat_util(arr_0, arr_1, is_scalar_0, is_scalar_1):
    """
    A dedicated kernel for the SQL function ARRAY_CAT which takes in
    two arrays (or columns of arrays) and returns an array containing the
    elements of the first array followed by the elements of the second array.

    Args:
        arr_0 (array scalar/array item array): the first array
        arr_1 (array scalar/array item array): the second array
        is_scalar_0 (boolean): whether arr_0 is a scalar array
        is_scalar_1 (boolean): whether arr_1 is a scalar array

    Returns:
        array: an array containing the elements of the first array
        followed by the elements of the second array.
    """
    is_scalar_0 = get_overload_const_bool(is_scalar_0, "array_cat", "is_scalar_0")
    is_scalar_1 = get_overload_const_bool(is_scalar_1, "array_cat", "is_scalar_1")
    are_arrays = [
        not is_scalar_0,
        not is_scalar_1,
        False,
        False,
    ]
    verify_array_arg(arr_0, is_scalar_0, "ARRAY_CAT", "arr_0")
    verify_array_arg(arr_1, is_scalar_1, "ARRAY_CAT", "arr_1")
    arg_names = ["arr_0", "arr_1", "is_scalar_0", "is_scalar_1"]
    arg_types = [arr_0, arr_1, is_scalar_0, is_scalar_1]
    propagate_null = [True] * 2 + [False] * 2
    if are_arrays[0]:
        out_dtype = arr_0
    else:
        out_dtype = arr_1

    # Infer what type of array needs to be allocated for each row
    inner_dtype = out_dtype if is_scalar_0 and is_scalar_1 else out_dtype.dtype
    extra_globals = {"inner_dtype": inner_dtype}
    scalar_text = "length_0 = len(arg0)\n"
    scalar_text += "length_1 = len(arg1)\n"
    if is_str_arr_type(inner_dtype):
        scalar_text += "inner_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(length_0 + length_1, -1)\n"
    elif is_bin_arr_type(inner_dtype):
        scalar_text += "inner_arr = bodo.libs.binary_arr_ext.pre_alloc_binary_array(length_0 + length_1, -1)\n"
    else:
        scalar_text += "inner_arr = bodo.utils.utils.alloc_type(length_0 + length_1, inner_dtype, (-1,))\n"

    # Note: setitem with a slice is used to copy over elements from the original arrays
    # to the answer to avoid flaws in getitem with map arrays.

    # Loop over each index in the first array and copy it (or a null) into the
    # same index of the result array
    scalar_text += "for idx0 in range(length_0):\n"
    scalar_text += "   if bodo.libs.array_kernels.isna(arg0, idx0):\n"
    scalar_text += "      bodo.libs.array_kernels.setna(inner_arr, idx0)\n"
    scalar_text += "   else:\n"
    scalar_text += "      inner_arr[idx0:idx0+1] = arg0[idx0:idx0+1]\n"

    # Loop over each index in the second array and copy it (or a null) into the
    # same index of the result array plus an offset to account for all of the
    # elements from the first array.
    scalar_text += "for idx1 in range(length_1):\n"
    scalar_text += "   if bodo.libs.array_kernels.isna(arg1, idx1):\n"
    scalar_text += "      bodo.libs.array_kernels.setna(inner_arr, length_0 + idx1)\n"
    scalar_text += "   else:\n"
    scalar_text += "      write_idx = length_0 + idx1\n"
    scalar_text += "      inner_arr[write_idx:write_idx+1] = arg1[idx1:idx1+1]\n"
    scalar_text += "res[i] = inner_arr"

    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
        extra_globals=extra_globals,
    )


def to_array(
    arr, is_scalar=False, dict_encoding_state=None, func_id=-1
):  # pragma: no cover
    pass


@overload(to_array, no_unliteral=True)
def overload_to_array(arr, is_scalar=False, dict_encoding_state=None, func_id=-1):
    """
    Handles cases where TO_ARRAY receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    if isinstance(arr, types.optional):  # pragma: no cover
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.to_array",
            ["arr", "is_scalar", "dict_encoding_state", "func_id"],
            0,
            default_map={
                "is_scalar": False,
                "dict_encoding_state": None,
                "func_id": -1,
            },
        )

    def impl(
        arr, is_scalar=False, dict_encoding_state=None, func_id=-1
    ):  # pragma: no cover
        return to_array_util(arr, is_scalar, dict_encoding_state, func_id)

    return impl


def to_array_util(arr, is_scalar, dict_encoding_state, func_id):  # pragma: no cover
    pass


@overload(to_array_util, no_unliteral=True)
def overload_to_array_util(arr, is_scalar, dict_encoding_state, func_id):
    is_scalar_bool = get_overload_const_bool(is_scalar, "to_array", "is_scalar")
    arg_names = ["arr", "is_scalar", "dict_encoding_state", "func_id"]
    arg_types = [arr, is_scalar, dict_encoding_state, func_id]
    propagate_null = [True, False, False, False]
    inner_dtype = arr if is_scalar_bool else arr.dtype
    inner_dtype = (
        inner_dtype
        if is_array_typ(inner_dtype)
        else dtype_to_array_type(inner_dtype, True)
    )
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(inner_dtype)
    scalar_text = "res[i] = bodo.utils.conversion.coerce_scalar_to_array(arg0, 1, inner_dtype, False)"
    use_dict_caching = not is_overload_none(dict_encoding_state)
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        extra_globals={"inner_dtype": inner_dtype},
        dict_encoding_state_name="dict_encoding_state" if use_dict_caching else None,
        func_id_name="func_id" if use_dict_caching else None,
        are_arrays=[not is_scalar_bool, False, False, False],
    )


def arrays_overlap(
    array_0, array_1, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(arrays_overlap, no_unliteral=True)
def overload_arrays_overlap(array_0, array_1, is_scalar_0=False, is_scalar_1=False):
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


def arrays_overlap_util(array_0, array_1, is_scalar_0, is_scalar_1):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(arrays_overlap_util, no_unliteral=True)
def overload_arrays_overlap_util(array_0, array_1, is_scalar_0, is_scalar_1):
    """
    A dedicated kernel for the SQL function ARRAYS_OVERLAP which takes in two
    arrays (or columns of arrays) and returns whether they have overlap

    Args:
        array_0 (array scalar/array item array): the first array(s) to compare
        array_1 (array scalar/array item array): the second array(s) to compare
        is_scalar (boolean): if true, treats the inputs as scalar arrays

    Returns:
        boolean scalar/vector: whether the arrays have any common elements
    """
    is_scalar_0_bool = get_overload_const_bool(
        is_scalar_0, "arrays_overlap", "is_scalar_0"
    )
    is_scalar_1_bool = get_overload_const_bool(
        is_scalar_1, "arrays_overlap", "is_scalar_1"
    )
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


def array_position(
    elem, container, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_position, no_unliteral=True)
def overload_array_position(elem, container, is_scalar_0=False, is_scalar_1=False):
    """
    Handles cases where ARRAY_POSITION receives optional arguments and
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


def array_position_util(
    elem, container, elem_is_scalar, container_is_scalar
):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_position_util, no_unliteral=True)
def overload_array_position_util(elem, container, elem_is_scalar, container_is_scalar):
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
    elem_is_scalar_bool = get_overload_const_bool(
        elem_is_scalar, "array_position", "elem_is_scalar"
    )
    container_is_scalar_bool = get_overload_const_bool(
        container_is_scalar, "array_position", "container_is_scalar"
    )
    verify_array_arg(container, container_is_scalar_bool, "ARRAY_POSITION", "container")
    arg_names = ["elem", "container", "elem_is_scalar", "container_is_scalar"]
    arg_types = [elem, container, elem_is_scalar, container_is_scalar]
    propagate_null = [False, True, False, False]
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


def array_contains(
    elem, container, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_contains, no_unliteral=True)
def overload_array_contains(
    elem, container, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    """
    Handles cases where ARRAY_CONTAINS receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [elem, container]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_position",
                ["elem", "container", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(elem, container, is_scalar_0=False, is_scalar_1=False):
        return array_contains_util(elem, container, is_scalar_0, is_scalar_1)

    return impl


def array_contains_util(
    elem, container, elem_is_scalar, container_is_scalar
):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_contains_util, no_unliteral=True)
def overload_array_contains_util(
    elem, container, elem_is_scalar, container_is_scalar
):  # pragma: no cover
    """
    A dedicated kernel for the SQL function ARRAY_CONTAINS which takes in an
    element and an array (or column of arrays) and returns a boolean value
    indicating if elem is contained in container

    Args:
        elem (array scalar/array item array): the element(s) to look for
        container (array scalar/array item array): the array(s) to search through
        elem_is_scalar (boolean): if true, treats the first argument as a scalar even if it is an array.
        container_is_scalar (boolean): if true, treats the second argument as a scalar even if it is an array.

    Returns:
        boolean scalar/vector: if elem is contained in container, and null if container is null.
    """
    elem_is_scalar_bool = get_overload_const_bool(
        elem_is_scalar, "array_contains", "elem_is_scalar"
    )
    container_is_scalar_bool = get_overload_const_bool(
        container_is_scalar, "array_contains", "container_is_scalar"
    )
    verify_array_arg(container, container_is_scalar_bool, "ARRAY_CONTAINS", "container")
    arg_names = ["elem", "container", "elem_is_scalar", "container_is_scalar"]
    arg_types = [elem, container, elem_is_scalar, container_is_scalar]
    propagate_null = [False, True, False, False]
    out_dtype = bodo.boolean_array_type
    are_arrays = [not elem_is_scalar_bool, not container_is_scalar_bool, False, False]
    scalar_text = "found_match = False\n"
    if elem == bodo.none:
        scalar_text += "null0 = True\n"
    elif are_arrays[0]:
        scalar_text += "null0 = bodo.libs.array_kernels.isna(elem, i)\n"
    else:
        scalar_text += "null0 = False\n"
    scalar_text += "for idx1 in range(len(arg1)):\n"
    scalar_text += "   null1 = bodo.libs.array_kernels.isna(arg1, idx1)\n"
    scalar_text += "   if (null0 and null1) or (not null0 and not null1 and bodo.libs.bodosql_array_kernels.semi_safe_equals(arg0, arg1[idx1])):\n"
    scalar_text += "      found_match = True\n"
    scalar_text += "      break\n"
    scalar_text += "res[i] = found_match"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


def array_to_string(arr, separator, is_scalar=False):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_to_string, no_unliteral=True)
def overload_array_to_string(arr, separator, is_scalar=False):  # pragma: no cover
    """
    Handles cases where ARRAY_TO_STRING receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, separator]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_to_string",
                ["arr", "separator", "is_scalar"],
                i,
                default_map={"is_scalar": False},
            )

    def impl(arr, separator, is_scalar=False):
        return array_to_string_util(arr, separator, is_scalar)

    return impl


def array_to_string_util(arr, separator, is_scalar):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_to_string_util, no_unliteral=True)
def overload_array_to_string_util(arr, separator, is_scalar):  # pragma: no cover
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
    is_scalar_bool = get_overload_const_bool(is_scalar, "array_to_string", "is_scalar")
    verify_array_arg(arr, is_scalar_bool, "ARRAY_TO_STRING", "arr")
    verify_string_arg(separator, "ARRAY_TO_STRING", "separator")
    arg_names = ["arr", "separator", "is_scalar"]
    arg_types = [arr, separator, is_scalar]
    propagate_null = [True, True, False]
    out_dtype = bodo.string_array_type
    scalar_text = "arr_str = ''\n"
    scalar_text += "for idx0 in range(len(arg0)):\n"
    scalar_text += "   arr_str += arg1 + ('' if bodo.libs.array_kernels.isna(arg0, idx0) else bodo.libs.bodosql_array_kernels.to_char(arg0[idx0], None, True))\n"
    scalar_text += "res[i] = arr_str[len(arg1):]"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=[
            not is_scalar_bool,
            bodo.utils.utils.is_array_typ(separator),
            False,
        ],
        # Protect against the input being a "scalar" string/dict array.
        support_dict_encoding=False,
    )


def array_size(arr, is_scalar=False):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_size, no_unliteral=True)
def overload_array_size(arr, is_scalar=False):  # pragma: no cover
    """
    Handles cases where ARRAY_SIZE receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.array_size",
            ["arr", "is_scalar"],
            0,
            default_map={"is_scalar": False},
        )

    def impl(arr, is_scalar=False):
        return array_size_util(arr, is_scalar)

    return impl


def array_size_util(arr, is_scalar):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(array_size_util, no_unliteral=True)
def overload_array_size_util(arr, is_scalar):  # pragma: no cover
    """
    A dedicated kernel for the SQL function ARRAY_SIZE which takes in an
           array, (or array column). If it is an array it returns the size, if it is a column
           it returns the size of each array in the column.
    Args:
        arr (array scalar/array item array): the array(s) to get the size of
        is_scalar (bool literal): Whether this is called in a single row context, necessary
        to determine whether to return the length of a nested array or an array of the lengths
        of it's children in a case statment
    Returns:
        An integer scalar/array: the result lengths
    """
    if is_overload_none(arr):
        return lambda arr, is_scalar: None
    if not is_overload_constant_bool(is_scalar):
        raise_bodo_error("array_size(): 'is_scalar' must be a constant boolean")

    is_scalar_bool = get_overload_const_bool(is_scalar, "array_size", "is_scalar")

    verify_array_arg(arr, is_scalar_bool, "ARRAY_SIZE", "arr")

    if (
        not is_overload_none(arr)
        and not is_array_item_array(arr)
        and not (bodo.utils.utils.is_array_typ(arr) and is_scalar)
    ):
        # When not is_scalar only array item ararys are supported
        # When is_scalar then all arrays are supported
        raise_bodo_error(
            f"array_size(): unsupported for type {arr} when is_scalar={is_scalar}"
        )

    # Whether to call len on each element or on arr itself
    arr_is_array = is_array_item_array(arr) and not is_scalar_bool

    scalar_text = "res[i] = len(arg0)"
    arg_names = ["arr", "is_scalar"]
    arg_types = [
        bodo.utils.conversion.coerce_to_array(arr),
        is_scalar,
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


def array_compact(arr, is_scalar=False):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(array_compact, no_unliteral=True)  # pragma: no cover
def overload_array_compact(arr, is_scalar=False):
    """
    Handles cases where ARRAY_COMPACT receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    if isinstance(arr, types.optional):
        return unopt_argument(
            "bodo.libs.bodosql_array_kernels.array_compact",
            ["arr", "is_scalar"],
            0,
            default_map={"is_scalar": False},
        )

    def impl(arr, is_scalar=False):
        return array_compact_util(arr, is_scalar)

    return impl


def array_compact_util(arr, is_scalar):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(array_compact_util, no_unliteral=True)  # pragma: no cover
def overload_array_compact_util(arr, is_scalar):
    is_scalar_bool = get_overload_const_bool(is_scalar, "array_compact", "is_scalar")
    verify_array_arg(arr, is_scalar_bool, "ARRAY_COMPACT", "arr")
    arg_names = ["arr", "is_scalar"]
    arg_types = [arr, is_scalar]
    propagate_null = [True, False]
    if is_scalar_bool:
        out_dtype = bodo.null_array_type
    else:
        out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(arr.dtype)
    scalar_text = "elems_to_keep = np.ones(len(arg0), dtype=np.bool_)\n"
    scalar_text += "for idx0 in range(len(arg0)):\n"
    scalar_text += "   if bodo.libs.array_kernels.isna(arg0, idx0):\n"
    scalar_text += "      elems_to_keep[idx0] = False\n"
    scalar_text += "res[i] = arg0[elems_to_keep]"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=[not is_scalar_bool, False],
    )


def array_remove(
    arr, to_remove, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    pass


@overload(array_remove, no_unliteral=True)
def overload_array_remove(
    arr, to_remove, is_scalar_0=False, is_scalar_1=False
):  # pragma: no cover
    """
    Handles cases where ARRAY_REMOVE receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, to_remove]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_remove",
                ["arr", "to_remove", "is_scalar_0", "is_scalar_1"],
                i,
                default_map={"is_scalar_0": False, "is_scalar_1": False},
            )

    def impl(arr, to_remove, is_scalar_0=False, is_scalar_1=False):
        return array_remove_util(arr, to_remove, is_scalar_0, is_scalar_1)

    return impl


def array_remove_util(arr, to_remove, is_scalar_0, is_scalar_1):  # pragma: no cover
    pass


@overload(array_remove_util, no_unliteral=True)
def overload_array_remove_util(
    arr, to_remove, is_scalar_0, is_scalar_1
):  # pragma: no cover
    """
    A dedicated kernel for the SQL function ARRAY_REMOVE which takes
    in an array and an element to remove, and remove all elements
    that equal to the provided element from the given array.

    Args:
        arr (array scalar/array item array): the original array
        to_remove (scalar/vector): the element to remove

    Returns:
        array: arr with all elements equal to to_remove dropped
    """
    are_arrays = [
        not get_overload_const_bool(is_scalar_0, "array_remove", "is_scalar_0"),
        not get_overload_const_bool(is_scalar_1, "array_remove", "is_scalar_1"),
        False,
        False,
    ]
    verify_array_arg(arr, not are_arrays[0], "ARRAY_REMOVE", "arr")
    arg_names = ["arr", "to_remove", "is_scalar_0", "is_scalar_1"]
    arg_types = [arr, to_remove, is_scalar_0, is_scalar_1]
    propagate_null = [True, True, False, False]
    if is_overload_none(arr):
        out_dtype = bodo.null_array_type
    else:
        out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(
            arr.dtype if are_arrays[0] else arr
        )
    scalar_text = "elems_to_keep = np.empty(len(arg0), np.bool_)\n"
    scalar_text += "for idx0 in range(len(arg0)):\n"
    scalar_text += "   elems_to_keep[idx0] = bodo.libs.array_kernels.isna(arg0, idx0) or not bodo.libs.bodosql_array_kernels.semi_safe_equals(arg0[idx0], arg1)\n"
    scalar_text += "res[i] = arg0[elems_to_keep]"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


def array_remove_at(arr, pos, is_scalar=False):  # pragma: no cover
    pass


@overload(array_remove_at, no_unliteral=True)
def overload_array_remove_at(arr, pos, is_scalar=False):  # pragma: no cover
    """
    Handles cases where ARRAY_REMOVE receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, pos]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_remove_at",
                ["arr", "pos", "is_scalar"],
                i,
                default_map={"is_scalar": False},
            )

    def impl(arr, pos, is_scalar=False):
        return array_remove_at_util(arr, pos, is_scalar)

    return impl


def array_remove_at_util(arr, pos, is_scalar):  # pragma: no cover
    pass


@overload(array_remove_at_util, no_unliteral=True)
def overload_array_remove_at_util(arr, pos, is_scalar):  # pragma: no cover
    """
    A dedicated kernel for the SQL function ARRAY_REMOVE_AT which takes
    in an array and an index, and remove the element at that index from the array.
    Args:
        arr (array scalar/array item array): the original array
        pos (scalar/vector): the index of the element to remove
    Returns:
        array: arr with element of index pos removed
    """
    are_arrays = [
        not get_overload_const_bool(is_scalar, "array_remove_at", "is_scalar"),
        is_array_typ(pos),
        False,
    ]
    verify_array_arg(arr, are_arrays[0], "ARRAY_REMOVE_AT", "arr")
    verify_int_arg(pos, "ARRAY_REMOVE_AT", "pos")
    arg_names = ["arr", "pos", "is_scalar"]
    arg_types = [arr, pos, is_scalar]
    propagate_null = [True, True, False]
    out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(
        arr.dtype if are_arrays[0] else arr
    )
    scalar_text = "if -len(arg0) <= arg1 and arg1 < len(arg0):\n"
    scalar_text += "   elems_to_keep = np.ones(len(arg0), np.bool_)\n"
    scalar_text += "   elems_to_keep[arg1] = False\n"
    scalar_text += "   res[i] = arg0[elems_to_keep]\n"
    scalar_text += "else:\n"
    scalar_text += "   res[i] = arg0"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=are_arrays,
    )


def array_slice(arr, from_, to, is_scalar=False):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(array_slice, no_unliteral=True)
def overload_array_slice(arr, from_, to, is_scalar=False):  # pragma: no cover
    """
    Handles cases where ARRAY_SLICE receives optional arguments and
    forwards to the appropriate version of the real implementation
    """
    args = [arr, from_, to]
    for i in range(len(args)):
        if isinstance(args[i], types.optional):
            return unopt_argument(
                "bodo.libs.bodosql_array_kernels.array_slice",
                ["arr", "from_", "to", "is_scalar"],
                i,
                default_map={"is_scalar": False},
            )

    def impl(arr, from_, to, is_scalar=False):
        return array_slice_util(arr, from_, to, is_scalar)

    return impl


def array_slice_util(arr, from_, to, is_scalar):  # pragma: no cover
    # Dummy function used for overload
    return


@overload(array_slice_util, no_unliteral=True)
def overload_array_slice_util(arr, from_, to, is_scalar):  # pragma: no cover
    is_scalar_bool = get_overload_const_bool(is_scalar, "array_slice", "is_scalar")
    verify_array_arg(arr, is_scalar_bool, "ARRAY_SLICE", "arr")
    arg_names = ["arr", "from_", "to", "is_scalar"]
    arg_types = [arr, from_, to, is_scalar]
    propagate_null = [True, True, True, False]
    if is_overload_none(arr):
        out_dtype = bodo.null_array_type
    else:
        inner_arr_type = arr if is_scalar_bool else arr.dtype
        out_dtype = bodo.libs.array_item_arr_ext.ArrayItemArrayType(inner_arr_type)
    scalar_text = "res[i] = arg0[arg1:arg2]"
    return gen_vectorized(
        arg_names,
        arg_types,
        propagate_null,
        scalar_text,
        out_dtype,
        are_arrays=[
            not is_scalar_bool,
            bodo.utils.utils.is_array_typ(from_),
            bodo.utils.utils.is_array_typ(to),
            False,
        ],
    )


def to_object(data):  # pragma: no cover
    pass


@overload(to_object, inline="always")
def overload_to_object(data):  # pragma: no cover
    """
    A dedicated kernel for the SQL function TO_OBJECT which takes
    in a value of any type and returns it if it is a valid object,
    and throws an error otherwise. Because of its simple nature,
    this kernel can always be inlined, so it does not need to be
    included in the distributed analysis sets.

    A valid object is one of the following:
        - Null array
        - Null scalar
        - Struct array
        - Struct scalar
        - Map array
        - Map scalar
        - Optional type where the underlying type is a null/struct/map scalar
    """
    if (
        isinstance(
            data,
            (
                bodo.StructArrayType,
                bodo.StructType,
                bodo.MapArrayType,
                bodo.libs.map_arr_ext.MapScalarType,
                types.DictType,
            ),
        )
        or (data in (bodo.none, bodo.null_array_type))
        or (
            isinstance(data, types.optional)
            and isinstance(
                data.type,
                (bodo.StructType, bodo.libs.map_arr_ext.MapScalarType, types.DictType),
            )
        )
    ):
        return lambda data: data  # pragma: no cover
    raise_bodo_error(f"Called TO_OBJECT on non-object data: {data}")
