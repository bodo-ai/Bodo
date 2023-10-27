# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Implements BodoSQL array kernels related to ARRAY utilities
"""

import types

import numba

import bodo
from bodo.libs.bodosql_array_kernel_utils import *
from bodo.utils.typing import is_overload_none, unwrap_typeref


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
