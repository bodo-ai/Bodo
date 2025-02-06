"""Tools for handling bodo arrays, e.g. passing to C/C++ code"""

import warnings
from collections import defaultdict

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_cast
from numba.core.typing.templates import AbstractTemplate, infer_global, signature
from numba.cpython.listobj import ListInstance
from numba.extending import (
    NativeValue,
    intrinsic,
    lower_builtin,
    models,
    overload,
    register_model,
    typeof_impl,
)
from numba.np.arrayobj import _getitem_array_single_int

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    get_categories_int_type,
)
from bodo.hiframes.time_ext import TimeArrayType, TimeType
from bodo.libs import array_ext
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayPayloadType,
    ArrayItemArrayType,
    _get_array_item_arr_payload,
    define_array_item_dtor,
    offset_type,
)
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.decimal_arr_ext import DecimalArrayType, int128_type
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.map_arr_ext import MapArrayType, _get_map_arr_data_type
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.libs.str_arr_ext import (
    _get_str_binary_arr_payload,
    char_arr_type,
    null_bitmap_arr_type,
    string_array_type,
)
from bodo.libs.struct_arr_ext import (
    StructArrayPayloadType,
    StructArrayType,
    _get_struct_arr_payload,
    define_struct_arr_dtor,
)
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.typing import (
    BodoError,
    MetaType,
    decode_if_dict_array,
    get_overload_const_int,
    is_overload_constant_int,
    is_overload_none,
    is_str_arr_type,
    raise_bodo_error,
    type_has_unknown_cats,
    unwrap_typeref,
)
from bodo.utils.utils import (
    bodo_exec,
    cached_call_internal,
    check_and_propagate_cpp_exception,
    numba_to_c_type,
)

ll.add_symbol("array_item_array_to_info", array_ext.array_item_array_to_info)
ll.add_symbol("struct_array_to_info", array_ext.struct_array_to_info)
ll.add_symbol("map_array_to_info", array_ext.map_array_to_info)
ll.add_symbol("string_array_to_info", array_ext.string_array_to_info)
ll.add_symbol("dict_str_array_to_info", array_ext.dict_str_array_to_info)
ll.add_symbol("get_has_global_dictionary", array_ext.get_has_global_dictionary)
ll.add_symbol(
    "get_has_unique_local_dictionary", array_ext.get_has_unique_local_dictionary
)
ll.add_symbol("get_dict_id", array_ext.get_dict_id)
ll.add_symbol("numpy_array_to_info", array_ext.numpy_array_to_info)
ll.add_symbol("categorical_array_to_info", array_ext.categorical_array_to_info)
ll.add_symbol("null_array_to_info", array_ext.null_array_to_info)
ll.add_symbol("nullable_array_to_info", array_ext.nullable_array_to_info)
ll.add_symbol("interval_array_to_info", array_ext.interval_array_to_info)
ll.add_symbol("decimal_array_to_info", array_ext.decimal_array_to_info)
ll.add_symbol("time_array_to_info", array_ext.time_array_to_info)
ll.add_symbol("timestamp_tz_array_to_info", array_ext.timestamp_tz_array_to_info)
ll.add_symbol("info_to_array_item_array", array_ext.info_to_array_item_array)
ll.add_symbol("info_to_struct_array", array_ext.info_to_struct_array)
ll.add_symbol("get_child_info", array_ext.get_child_info)
ll.add_symbol("info_to_string_array", array_ext.info_to_string_array)
ll.add_symbol("info_to_numpy_array", array_ext.info_to_numpy_array)
ll.add_symbol("info_to_null_array", array_ext.info_to_null_array)
ll.add_symbol("info_to_nullable_array", array_ext.info_to_nullable_array)
ll.add_symbol("info_to_interval_array", array_ext.info_to_interval_array)
ll.add_symbol("info_to_timestamptz_array", array_ext.info_to_timestamptz_array)
ll.add_symbol("arr_info_list_to_table", array_ext.arr_info_list_to_table)
ll.add_symbol("info_from_table", array_ext.info_from_table)
ll.add_symbol("delete_info", array_ext.delete_info)
ll.add_symbol(
    "bodo_array_from_pyarrow_py_entry", array_ext.bodo_array_from_pyarrow_py_entry
)
ll.add_symbol(
    "pd_pyarrow_array_from_bodo_array_py_entry",
    array_ext.pd_pyarrow_array_from_bodo_array_py_entry,
)
ll.add_symbol("array_info_unpin", array_ext.array_info_unpin)
ll.add_symbol("delete_table", array_ext.delete_table)
ll.add_symbol("cpp_table_map_to_list", array_ext.cpp_table_map_to_list)
ll.add_symbol("shuffle_table_py_entrypt", array_ext.shuffle_table_py_entrypt)
ll.add_symbol("get_shuffle_info", array_ext.get_shuffle_info)
ll.add_symbol("delete_shuffle_info", array_ext.delete_shuffle_info)
ll.add_symbol("reverse_shuffle_table", array_ext.reverse_shuffle_table)
ll.add_symbol("hash_join_table", array_ext.hash_join_table)
ll.add_symbol("nested_loop_join_table", array_ext.nested_loop_join_table)
ll.add_symbol("interval_join_table", array_ext.interval_join_table)
ll.add_symbol(
    "drop_duplicates_table_py_entry", array_ext.drop_duplicates_table_py_entry
)
ll.add_symbol("sort_values_table_py_entry", array_ext.sort_values_table_py_entry)
ll.add_symbol(
    "sort_table_for_interval_join_py_entrypoint",
    array_ext.sort_table_for_interval_join_py_entrypoint,
)
ll.add_symbol("sample_table_py_entry", array_ext.sample_table_py_entry)
ll.add_symbol(
    "shuffle_renormalization_py_entrypt", array_ext.shuffle_renormalization_py_entrypt
)
ll.add_symbol(
    "shuffle_renormalization_group_py_entrypt",
    array_ext.shuffle_renormalization_group_py_entrypt,
)
ll.add_symbol("groupby_and_aggregate", array_ext.groupby_and_aggregate_py_entry)
ll.add_symbol(
    "drop_duplicates_local_dictionary_py_entry",
    array_ext.drop_duplicates_local_dictionary_py_entry,
)
ll.add_symbol("get_groupby_labels_py_entry", array_ext.get_groupby_labels_py_entry)
ll.add_symbol("array_isin_py_entry", array_ext.array_isin_py_entry)
ll.add_symbol("get_search_regex_py_entry", array_ext.get_search_regex_py_entry)
ll.add_symbol("get_replace_regex_py_entry", array_ext.get_replace_regex_py_entry)
ll.add_symbol(
    "get_replace_regex_dict_state_py_entry",
    array_ext.get_replace_regex_dict_state_py_entry,
)
ll.add_symbol("array_info_getitem", array_ext.array_info_getitem)
ll.add_symbol("array_info_getdata1", array_ext.array_info_getdata1)
ll.add_symbol("union_tables", array_ext.union_tables)
ll.add_symbol("concat_tables_py_entry", array_ext.concat_tables_py_entry)
ll.add_symbol("alloc_like_kernel_cache", array_ext.alloc_like_kernel_cache)
ll.add_symbol("add_to_like_kernel_cache", array_ext.add_to_like_kernel_cache)
ll.add_symbol("check_like_kernel_cache", array_ext.check_like_kernel_cache)
ll.add_symbol("dealloc_like_kernel_cache", array_ext.dealloc_like_kernel_cache)
ll.add_symbol(
    "BODO_NRT_MemInfo_alloc_safe_aligned", array_ext.NRT_MemInfo_alloc_safe_aligned
)
ll.add_symbol("retrieve_table_py_entry", array_ext.retrieve_table_py_entry)


# Sentinal for field names when converting tuple arrays to struct arrays (workaround
# since Arrow doesn't support tuple arrays)
TUPLE_ARRAY_SENTINEL = "_bodo_tuple_array_field"


class LikeKernelCache(types.Opaque):
    """
    Cache for SQL Like Kernel where both the array and the pattern
    array are dictionary encoded.
    We map the pair of indices from the two dictionary encoded arrays
    to a boolean computation output value.
    We implement the cache in C++ for better performance.
    """

    def __init__(self):
        super().__init__(name="LikeKernelCache")


like_kernel_cache_type = LikeKernelCache()
types.like_kernel_cache_type = like_kernel_cache_type  # type: ignore
register_model(LikeKernelCache)(models.OpaqueModel)


_alloc_like_kernel_cache = types.ExternalFunction(
    "alloc_like_kernel_cache", types.like_kernel_cache_type(types.uint64)
)
_add_to_like_kernel_cache = types.ExternalFunction(
    "add_to_like_kernel_cache",
    types.void(types.like_kernel_cache_type, types.uint32, types.uint32, types.bool_),
)
_check_like_kernel_cache = types.ExternalFunction(
    "check_like_kernel_cache",
    types.int8(types.like_kernel_cache_type, types.uint32, types.uint32),
)
_dealloc_like_kernel_cache = types.ExternalFunction(
    "dealloc_like_kernel_cache", types.void(types.like_kernel_cache_type)
)


class ArrayInfoType(types.Type):
    def __init__(self):
        super().__init__(name="ArrayInfoType()")


array_info_type = ArrayInfoType()
register_model(ArrayInfoType)(models.OpaqueModel)


class TableTypeCPP(types.Type):
    def __init__(self):
        super().__init__(name="TableTypeCPP()")


table_type = TableTypeCPP()
register_model(TableTypeCPP)(models.OpaqueModel)


@lower_cast(table_type, types.voidptr)
def lower_table_type(context, builder, fromty, toty, val):  # pragma: no cover
    return val


@lower_cast(array_info_type, types.voidptr)
def lower_array_type(context, builder, fromty, toty, val):  # pragma: no cover
    return val


@intrinsic
def array_to_info(typingctx, arr_type_t):
    """convert array to array info wrapper to pass to C++"""
    return array_info_type(arr_type_t), array_to_info_codegen


def array_to_info_codegen(context, builder, sig, args):
    """
    Codegen for array_to_info. This isn't a closure because
    this function is called directly to call array_to_info
    from an intrinsic.
    """
    (in_arr,) = args
    arr_type = sig.args[0]

    # NOTE: arr_info struct keeps a reference

    if isinstance(arr_type, TupleArrayType):
        # TupleArray uses same model as StructArray so we just use a
        # StructArrayType to generate LLVM
        tuple_array = context.make_helper(builder, arr_type, in_arr)
        in_arr = tuple_array.data
        arr_type = StructArrayType(
            arr_type.data,
            tuple(f"{TUPLE_ARRAY_SENTINEL}{i}" for i in range(len(arr_type.data))),
        )

    if isinstance(arr_type, MapArrayType):
        map_array = context.make_helper(builder, arr_type, in_arr)
        inner_arr = map_array.data
        inner_arr_type = _get_map_arr_data_type(arr_type)
        inner_arr_info = array_to_info_codegen(
            context, builder, array_info_type(inner_arr_type), (inner_arr,)
        )

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="map_array_to_info"
        )
        return builder.call(
            fn_tp,
            [
                inner_arr_info,
            ],
        )

    if isinstance(arr_type, ArrayItemArrayType):
        payload = _get_array_item_arr_payload(context, builder, arr_type, in_arr)
        inner_arr = payload.data
        inner_arr_info = array_to_info_codegen(
            context, builder, array_info_type(arr_type.dtype), (inner_arr,)
        )

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="array_item_array_to_info"
        )
        return builder.call(
            fn_tp,
            [
                payload.n_arrays,
                inner_arr_info,
                payload.offsets,
                payload.null_bitmap,
            ],
        )

    if isinstance(arr_type, StructArrayType):
        payload = _get_struct_arr_payload(context, builder, arr_type, in_arr)

        # get array_infos of child arrays
        inner_arr_infos = []
        for i, field_type in enumerate(arr_type.data):
            inner_arr = builder.extract_value(payload.data, i)
            inner_arr_infos.append(
                array_to_info_codegen(
                    context, builder, array_info_type(field_type), (inner_arr,)
                )
            )
        # NOTE: passing type to pack_array() is necessary in case value list is empty
        inner_arr_infos_ptr = cgutils.alloca_once_value(
            builder,
            cgutils.pack_array(
                builder, inner_arr_infos, context.get_data_type(array_info_type)
            ),
        )
        # get field names
        field_names = cgutils.pack_array(
            builder,
            [context.insert_const_string(builder.module, a) for a in arr_type.names],
            context.get_data_type(types.voidptr),
        )
        field_names_ptr = cgutils.alloca_once_value(builder, field_names)

        null_bitmap = context.make_helper(
            builder, null_bitmap_arr_type, payload.null_bitmap
        )

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="struct_array_to_info"
        )
        return builder.call(
            fn_tp,
            [
                context.get_constant(types.int64, len(arr_type.data)),
                payload.n_structs,
                builder.bitcast(inner_arr_infos_ptr, lir.IntType(8).as_pointer()),
                builder.bitcast(field_names_ptr, lir.IntType(8).as_pointer()),
                null_bitmap.meminfo,
            ],
        )

    # StringArray
    if arr_type in (string_array_type, binary_array_type):
        payload = _get_str_binary_arr_payload(context, builder, in_arr, arr_type)
        char_arr = context.make_helper(builder, char_arr_type, payload.data)
        # TODO: make sure char_arr.meminfo_offset is zero since not supported in C++

        is_bytes = context.get_constant(types.int32, int(arr_type == binary_array_type))
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="string_array_to_info"
        )
        return builder.call(
            fn_tp,
            [
                payload.n_arrays,
                char_arr.meminfo,
                payload.offsets,
                payload.null_bitmap,
                is_bytes,
            ],
        )

    # dictionary-encoded string array
    if arr_type == bodo.dict_str_arr_type:
        # pass string array and indices array as array_info to C++
        arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
        str_arr = arr.data
        indices_arr = arr.indices
        sig = array_info_type(arr_type.data)
        str_arr_info = array_to_info_codegen(context, builder, sig, (str_arr,))

        sig = array_info_type(bodo.libs.dict_arr_ext.dict_indices_arr_type)
        indices_arr_info = array_to_info_codegen(context, builder, sig, (indices_arr,))

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # string_array_info
                lir.IntType(8).as_pointer(),  # indices_arr_info
                lir.IntType(32),  # has_global_dictionary flag
                lir.IntType(32),  # has_unique_local_dictionary flag
                lir.IntType(64),  # dict_id
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="dict_str_array_to_info"
        )

        # cast boolean to int32 to avoid potential bool data model mismatch
        has_global_dictionary = builder.zext(arr.has_global_dictionary, lir.IntType(32))
        has_unique_local_dictionary = builder.zext(
            arr.has_unique_local_dictionary, lir.IntType(32)
        )
        return builder.call(
            fn_tp,
            [
                str_arr_info,
                indices_arr_info,
                has_global_dictionary,
                has_unique_local_dictionary,
                arr.dict_id,
            ],
        )

    if arr_type == bodo.hiframes.timestamptz_ext.timestamptz_array_type:
        arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
        data_ts_arr = context.make_array(
            bodo.hiframes.timestamptz_ext.timestamptz_array_type.ts_arr_type()
        )(context, builder, arr.data_ts)
        data_offset_arr = context.make_array(
            bodo.hiframes.timestamptz_ext.timestamptz_array_type.offset_array_type()
        )(context, builder, arr.data_offset)
        length = builder.extract_value(data_ts_arr.shape, 0)
        null_bitmap = context.make_helper(
            builder, null_bitmap_arr_type, arr.null_bitmap
        )

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),  # n_items
                lir.IntType(8).as_pointer(),  # data_ts_info
                lir.IntType(8).as_pointer(),  # data_offset_info
                lir.IntType(8).as_pointer(),  # null_bitmap_info
                lir.IntType(8).as_pointer(),  # data_ts_meminfo
                lir.IntType(8).as_pointer(),  # data_offset_meminfo
                lir.IntType(8).as_pointer(),  # null_bitmap_meminfo
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="timestamp_tz_array_to_info"
        )

        return builder.call(
            fn_tp,
            [
                length,
                builder.bitcast(data_ts_arr.data, lir.IntType(8).as_pointer()),
                builder.bitcast(data_offset_arr.data, lir.IntType(8).as_pointer()),
                builder.bitcast(null_bitmap.data, lir.IntType(8).as_pointer()),
                data_ts_arr.meminfo,
                data_offset_arr.meminfo,
                null_bitmap.meminfo,
            ],
        )

    # get codes array from CategoricalArrayType to be handled similar to other Numpy
    # arrays.
    is_categorical = False
    if isinstance(arr_type, CategoricalArrayType):
        num_categories = context.compile_internal(
            builder,
            lambda a: len(a.dtype.categories),
            types.intp(arr_type),
            [in_arr],
        )
        in_arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr).codes
        int_dtype = get_categories_int_type(arr_type.dtype)
        arr_type = types.Array(int_dtype, 1, "C")
        is_categorical = True

    # Numpy
    if isinstance(arr_type, types.Array):
        arr = context.make_array(arr_type)(context, builder, in_arr)
        assert arr_type.ndim == 1, "only 1D array shuffle supported"
        length = builder.extract_value(arr.shape, 0)
        dtype = arr_type.dtype
        typ_enum = numba_to_c_type(dtype)
        typ_arg = cgutils.alloca_once_value(
            builder, lir.Constant(lir.IntType(32), typ_enum)
        )

        if is_categorical:
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, name="categorical_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    length,
                    builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                    builder.load(typ_arg),
                    num_categories,
                    arr.meminfo,
                ],
            )
        else:
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, name="numpy_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    length,
                    builder.bitcast(arr.data, lir.IntType(8).as_pointer()),
                    builder.load(typ_arg),
                    arr.meminfo,
                ],
            )

    # null array
    if arr_type == bodo.null_array_type:
        arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
        # TODO: Add a null dtype in C++. Since adding C++ support enables
        # passing the null array anywhere, including places where null arrays
        # may not be supported, we initially allocate an int8 array that will
        # be unused.
        length = arr.length
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="null_array_to_info"
        )
        return builder.call(
            fn_tp,
            [
                length,
            ],
        )

    # nullable integer/bool array
    if isinstance(
        arr_type,
        (
            IntegerArrayType,
            FloatingArrayType,
            DecimalArrayType,
            TimeArrayType,
            DatetimeArrayType,
        ),
    ) or arr_type in (
        boolean_array_type,
        datetime_date_array_type,
    ):
        arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
        dtype = arr_type.dtype
        np_dtype = dtype
        if isinstance(arr_type, DecimalArrayType):
            np_dtype = int128_type
        elif isinstance(arr_type, DatetimeArrayType):
            np_dtype = bodo.datetime64ns
        elif arr_type == datetime_date_array_type:
            np_dtype = types.int32
        elif arr_type == boolean_array_type:
            np_dtype = types.int8
        data_arr = context.make_array(types.Array(np_dtype, 1, "C"))(
            context, builder, arr.data
        )
        if arr_type == boolean_array_type:
            length = arr.length
        else:
            length = builder.extract_value(data_arr.shape, 0)
        bitmap_arr = context.make_array(types.Array(types.uint8, 1, "C"))(
            context, builder, arr.null_bitmap
        )

        typ_enum = numba_to_c_type(dtype)
        typ_arg = cgutils.alloca_once_value(
            builder, lir.Constant(lir.IntType(32), typ_enum)
        )

        if isinstance(arr_type, DecimalArrayType):
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(32),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, name="decimal_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    length,
                    builder.bitcast(data_arr.data, lir.IntType(8).as_pointer()),
                    builder.load(typ_arg),
                    builder.bitcast(bitmap_arr.data, lir.IntType(8).as_pointer()),
                    data_arr.meminfo,
                    bitmap_arr.meminfo,
                    context.get_constant(types.int32, arr_type.precision),
                    context.get_constant(types.int32, arr_type.scale),
                ],
            )
        elif isinstance(arr_type, TimeArrayType):
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, name="time_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    length,
                    builder.bitcast(data_arr.data, lir.IntType(8).as_pointer()),
                    builder.load(typ_arg),
                    builder.bitcast(bitmap_arr.data, lir.IntType(8).as_pointer()),
                    data_arr.meminfo,
                    bitmap_arr.meminfo,
                    lir.Constant(lir.IntType(32), arr_type.precision),
                ],
            )
        else:
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = cgutils.get_or_insert_function(
                builder.module, fnty, name="nullable_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    length,
                    builder.bitcast(data_arr.data, lir.IntType(8).as_pointer()),
                    builder.load(typ_arg),
                    builder.bitcast(bitmap_arr.data, lir.IntType(8).as_pointer()),
                    data_arr.meminfo,
                    bitmap_arr.meminfo,
                ],
            )

    # interval array
    if isinstance(arr_type, IntervalArrayType):
        assert isinstance(arr_type.arr_type, types.Array), (
            "array_to_info(): only IntervalArrayType with Numpy arrays supported"
        )
        arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
        left_arr = context.make_array(arr_type.arr_type)(context, builder, arr.left)
        right_arr = context.make_array(arr_type.arr_type)(context, builder, arr.right)
        length = builder.extract_value(left_arr.shape, 0)

        typ_enum = numba_to_c_type(arr_type.arr_type.dtype)
        typ_arg = cgutils.alloca_once_value(
            builder, lir.Constant(lir.IntType(32), typ_enum)
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="interval_array_to_info"
        )
        return builder.call(
            fn_tp,
            [
                length,
                builder.bitcast(left_arr.data, lir.IntType(8).as_pointer()),
                builder.bitcast(right_arr.data, lir.IntType(8).as_pointer()),
                builder.load(typ_arg),
                left_arr.meminfo,
                right_arr.meminfo,
            ],
        )

    # Dummy handling for PrimitiveArrayType used in string array since bodo.gatherv()
    # calls itself on string array data which generates an unnecessary CPython wrapper
    # for a nested array. Boxing of nested array uses info_to_array().
    # See test_scatterv_gatherv_allgatherv_df_jit"[df_value2]"
    if isinstance(arr_type, bodo.PrimitiveArrayType):
        return context.get_constant_null(array_info_type)

    raise_bodo_error(f"array_to_info(): array type {arr_type} is not supported")


def _lower_info_to_array_numpy(arr_type, context, builder, in_info, raise_py_err=True):
    assert arr_type.ndim == 1, "only 1D array supported"
    arr = context.make_array(arr_type)(context, builder)

    length_ptr = cgutils.alloca_once(builder, lir.IntType(64))
    data_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
    meminfo_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),  # info
            lir.IntType(64).as_pointer(),  # num_items
            lir.IntType(8).as_pointer().as_pointer(),  # data
            lir.IntType(8).as_pointer().as_pointer(),
        ],
    )  # meminfo
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="info_to_numpy_array"
    )
    builder.call(fn_tp, [in_info, length_ptr, data_ptr, meminfo_ptr])
    if raise_py_err:
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, [builder.load(length_ptr)], ty=intp_t)
    itemsize = context.get_constant(
        types.intp,
        context.get_abi_sizeof(context.get_data_type(arr_type.dtype)),
    )
    strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

    data = builder.bitcast(
        builder.load(data_ptr),
        context.get_data_type(arr_type.dtype).as_pointer(),
    )

    numba.np.arrayobj.populate_array(
        arr,
        data=data,
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=builder.load(meminfo_ptr),
    )
    return arr._getvalue()


def _lower_info_to_array_item_array(
    context, builder, arr_type, in_info, raise_py_err=True
):
    """Convert C++ array_info* to array(array(string)).
    Allocates an array(array(item)) payload and uses C++ to fill n_arrays/offsets/
    null_bitmap fields.
    Converts the internal string array array_info* to set the data array.
    """
    inner_arr_type = arr_type.dtype
    array_item_data_type = ArrayItemArrayType(inner_arr_type)

    # create payload type
    payload_type = ArrayItemArrayPayloadType(array_item_data_type)
    alloc_type = context.get_value_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_array_item_dtor(
        context, builder, array_item_data_type, payload_type
    )

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)

    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(),
        [
            lir.IntType(8).as_pointer(),  # info
            lir.IntType(64).as_pointer(),  # n_arrays
            context.get_value_type(types.MemInfoPointer(offset_type)).as_pointer(),
            context.get_value_type(types.MemInfoPointer(types.uint8)).as_pointer(),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="info_to_array_item_array"
    )
    str_arr_info = builder.call(
        fn_tp,
        [
            in_info,
            payload._get_ptr_by_name("n_arrays"),
            payload._get_ptr_by_name("offsets"),
            payload._get_ptr_by_name("null_bitmap"),
        ],
    )
    if raise_py_err:
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    # The data array of array(array(string)) is a string array
    payload.data = info_to_array_codegen(
        context,
        builder,
        inner_arr_type(array_info_type, inner_arr_type),
        (str_arr_info, context.get_constant_null(arr_type)),
        raise_py_err,
    )
    builder.store(payload._getvalue(), meminfo_data_ptr)

    array_item_array = context.make_helper(builder, array_item_data_type)
    array_item_array.meminfo = meminfo
    return array_item_array._getvalue()


def _lower_info_to_struct_array(context, builder, arr_type, in_info, raise_py_err=True):
    """Convert C++ array_info* to struct array.
    Allocates a struct array payload and uses C++ to fill null_bitmap field.
    Calls info_to_array_codegen() recursively to convert child arrays.
    """

    # create payload type
    payload_type = StructArrayPayloadType(arr_type.data)
    alloc_type = context.get_value_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_struct_arr_dtor(context, builder, arr_type, payload_type)

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)

    # get nested infos and null bitmap from C++
    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer().as_pointer(),
        [
            lir.IntType(8).as_pointer(),  # info
            lir.IntType(64).as_pointer(),
            context.get_value_type(null_bitmap_arr_type).as_pointer(),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="info_to_struct_array"
    )
    builder.call(
        fn_tp,
        [
            in_info,
            payload._get_ptr_by_name("n_structs"),
            payload._get_ptr_by_name("null_bitmap"),
        ],
    )
    if raise_py_err:
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(),
        [
            lir.IntType(8).as_pointer(),  # info
            lir.IntType(64),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="get_child_info")

    # convert inner array infos
    data_arrs = []
    for i, inner_arr_type in enumerate(arr_type.data):
        inner_info = builder.call(
            fn_tp, [in_info, context.get_constant(types.int64, i)]
        )
        data_arrs.append(
            info_to_array_codegen(
                context,
                builder,
                inner_arr_type(array_info_type, inner_arr_type),
                (inner_info, context.get_constant_null(arr_type)),
                raise_py_err,
            )
        )

    payload.data = (
        cgutils.pack_array(builder, data_arrs)
        if types.is_homogeneous(*arr_type.data)
        else cgutils.pack_struct(builder, data_arrs)
    )
    builder.store(payload._getvalue(), meminfo_data_ptr)

    struct_array = context.make_helper(builder, arr_type)
    struct_array.meminfo = meminfo
    return struct_array._getvalue()


def info_to_array_codegen(context, builder, sig, args, raise_py_err=True):
    array_type = sig.args[1]
    arr_type = (
        array_type.instance_type
        if isinstance(array_type, types.TypeRef)
        else array_type
    )
    in_info, _ = args

    if isinstance(arr_type, TupleArrayType):
        # TupleArray is just a StructArray in C++
        tuple_array = context.make_helper(builder, arr_type)
        struct_arr_type = StructArrayType(arr_type.data)
        inner_sig = struct_arr_type(array_info_type, struct_arr_type)
        tuple_array.data = info_to_array_codegen(
            context, builder, inner_sig, args, raise_py_err
        )
        return tuple_array._getvalue()

    if isinstance(arr_type, MapArrayType):
        # Extract data array info from input array info
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_child_info"
        )
        data_arr_info = builder.call(
            fn_tp,
            [
                in_info,
                lir.Constant(lir.IntType(64), 0),
            ],
        )

        map_data_arr_type = _get_map_arr_data_type(arr_type)
        inner_sig = map_data_arr_type(array_info_type, map_data_arr_type)
        map_array = context.make_helper(builder, arr_type)
        map_array.data = info_to_array_codegen(
            context, builder, inner_sig, [data_arr_info, args[1]], raise_py_err
        )
        return map_array._getvalue()

    if isinstance(arr_type, ArrayItemArrayType):
        return _lower_info_to_array_item_array(
            context, builder, arr_type, in_info, raise_py_err
        )

    if isinstance(arr_type, StructArrayType):
        return _lower_info_to_struct_array(
            context, builder, arr_type, in_info, raise_py_err
        )

    # StringArray
    if arr_type in (string_array_type, binary_array_type):
        return _gen_info_to_string_array(
            context, builder, arr_type, in_info, raise_py_err
        )

    # dictionary-encoded string array
    if arr_type == bodo.dict_str_arr_type:
        # extract nested array infos from input array info
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),  # info
                # info number (0 for getting the string array or 1 for indices array)
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_child_info"
        )
        str_arr_info = builder.call(
            fn_tp,
            [
                in_info,
                lir.Constant(lir.IntType(64), 0),
            ],
        )
        indices_arr_info = builder.call(
            fn_tp,
            [
                in_info,
                lir.Constant(lir.IntType(64), 1),
            ],
        )

        dict_array = context.make_helper(builder, arr_type)
        sig = arr_type.data(array_info_type, arr_type.data)
        dict_array.data = info_to_array_codegen(
            context,
            builder,
            sig,
            (str_arr_info, context.get_constant_null(arr_type.data)),
            raise_py_err,
        )

        indices_arr_t = bodo.libs.dict_arr_ext.dict_indices_arr_type
        sig = indices_arr_t(array_info_type, indices_arr_t)
        dict_array.indices = info_to_array_codegen(
            context,
            builder,
            sig,
            (indices_arr_info, context.get_constant_null(indices_arr_t)),
            raise_py_err,
        )

        fnty = lir.FunctionType(
            lir.IntType(32),
            [
                lir.IntType(8).as_pointer(),  # info
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_has_global_dictionary"
        )
        has_global_dictionary = builder.call(
            fn_tp,
            [
                in_info,
            ],
        )

        # cast int32 to bool
        dict_array.has_global_dictionary = builder.trunc(
            has_global_dictionary, cgutils.bool_t
        )

        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_has_unique_local_dictionary"
        )
        has_unique_local_dictionary = builder.call(
            fn_tp,
            [
                in_info,
            ],
        )

        # cast int32 to bool
        dict_array.has_unique_local_dictionary = builder.trunc(
            has_unique_local_dictionary, cgutils.bool_t
        )

        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(8).as_pointer(),  # info
            ],
        )
        fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="get_dict_id")
        dict_array.dict_id = builder.call(
            fn_tp,
            [
                in_info,
            ],
        )

        return dict_array._getvalue()

    # categorical array
    if isinstance(arr_type, CategoricalArrayType):
        out_arr = cgutils.create_struct_proxy(arr_type)(context, builder)
        int_dtype = get_categories_int_type(arr_type.dtype)
        int_arr_type = types.Array(int_dtype, 1, "C")
        out_arr.codes = _lower_info_to_array_numpy(
            int_arr_type, context, builder, in_info, raise_py_err
        )
        # set categorical dtype of output array to be same as input array
        if isinstance(array_type, types.TypeRef):
            assert arr_type.dtype.categories is not None, (
                "info_to_array: unknown categories"
            )
            # create the new categorical dtype inside the function instead of passing as
            # constant. This avoids constant lowered Index inside the dtype, which can
            # be slow since it cannot have a dictionary.
            # see https://github.com/bodo-ai/Bodo/pull/3563
            is_ordered = arr_type.dtype.ordered
            new_cats_arr = bodo.utils.utils.create_categorical_type(
                arr_type.dtype.categories, arr_type.dtype.data.data, is_ordered
            )
            new_cats_tup = MetaType(arr_type.dtype.categories)
            int_type = arr_type.dtype.int_type
            cats_arr_type = arr_type.dtype.data.data
            cats_arr = context.get_constant_generic(
                builder, cats_arr_type, new_cats_arr
            )
            dtype = context.compile_internal(
                builder,
                lambda c_arr: bodo.hiframes.pd_categorical_ext.init_cat_dtype(
                    bodo.utils.conversion.index_from_array(c_arr),
                    is_ordered,
                    int_type,
                    new_cats_tup,
                ),
                arr_type.dtype(cats_arr_type),
                [cats_arr],
            )  # pragma: no cover
        else:
            dtype = cgutils.create_struct_proxy(arr_type)(
                context, builder, args[1]
            ).dtype
            context.nrt.incref(builder, arr_type.dtype, dtype)
        out_arr.dtype = dtype
        return out_arr._getvalue()

    # Numpy
    if isinstance(arr_type, types.Array):
        return _lower_info_to_array_numpy(
            arr_type, context, builder, in_info, raise_py_err
        )

    # Dummy handling for PrimitiveArrayType used in string array since bodo.gatherv()
    # calls itself on string array data which generates an unnecessary CPython wrapper
    # for a nested array. Unboxing of nested array uses info_to_array().
    # See test_scatterv_gatherv_allgatherv_df_jit"[df_value2]"
    if isinstance(arr_type, bodo.PrimitiveArrayType):
        return context.get_constant_null(arr_type)

    # null array
    if arr_type == bodo.null_array_type:
        arr = cgutils.create_struct_proxy(arr_type)(context, builder)
        # Set the array as not empty
        arr.not_empty = lir.Constant(lir.IntType(1), 1)
        length_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # info
                lir.IntType(64).as_pointer(),  # num_items
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="info_to_null_array"
        )
        builder.call(
            fn_tp,
            [
                in_info,
                length_ptr,
            ],
        )
        if raise_py_err:
            bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        arr.length = builder.load(length_ptr)
        return arr._getvalue()

    # Timestamp TZ array
    if arr_type == bodo.hiframes.timestamptz_ext.timestamptz_array_type:
        arr = cgutils.create_struct_proxy(arr_type)(context, builder)
        data_ts_arr_type = arr_type.ts_arr_type()
        data_ts_arr = context.make_array(data_ts_arr_type)(context, builder)
        data_offset_arr_type = arr_type.offset_array_type()
        data_offset_arr = context.make_array(data_offset_arr_type)(context, builder)
        nulls_arr_type = null_bitmap_arr_type
        nulls_arr = context.make_array(nulls_arr_type)(context, builder)

        length_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        n_bytes_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        data_ts_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        data_offset_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        nulls_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        meminfo_data_ts_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        meminfo_data_offsets_ptr = cgutils.alloca_once(
            builder, lir.IntType(8).as_pointer()
        )
        meminfo_nulls_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # info
                lir.IntType(64).as_pointer(),  # num_items
                lir.IntType(64).as_pointer(),  # num_bytes
                lir.IntType(8).as_pointer().as_pointer(),  # data ts
                lir.IntType(8).as_pointer().as_pointer(),  # data offset
                lir.IntType(8).as_pointer().as_pointer(),  # nulls
                lir.IntType(8).as_pointer().as_pointer(),  # meminfo ts
                lir.IntType(8).as_pointer().as_pointer(),  # meminfo offset
                lir.IntType(8).as_pointer().as_pointer(),  # meminfo nulls
            ],
        )  # meminfo_nulls
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="info_to_timestamptz_array"
        )
        builder.call(
            fn_tp,
            [
                in_info,
                length_ptr,
                n_bytes_ptr,
                data_ts_ptr,
                data_offset_ptr,
                nulls_ptr,
                meminfo_data_ts_ptr,
                meminfo_data_offsets_ptr,
                meminfo_nulls_ptr,
            ],
        )
        if raise_py_err:
            bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

        intp_t = context.get_value_type(types.intp)

        # Load the array components
        arrs = []
        for size_ptr, data_ptr, dest_arr, scalar_type, meminfo_ptr in [
            (length_ptr, data_ts_ptr, data_ts_arr, types.int64, meminfo_data_ts_ptr),
            (
                length_ptr,
                data_offset_ptr,
                data_offset_arr,
                types.int16,
                meminfo_data_offsets_ptr,
            ),
            (n_bytes_ptr, nulls_ptr, nulls_arr, types.uint8, meminfo_nulls_ptr),
        ]:
            shape_array = cgutils.pack_array(
                builder, [builder.load(size_ptr)], ty=intp_t
            )
            itemsize = context.get_constant(
                types.intp,
                context.get_abi_sizeof(context.get_data_type(scalar_type)),
            )
            strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

            data = builder.bitcast(
                builder.load(data_ptr),
                context.get_data_type(scalar_type).as_pointer(),
            )

            numba.np.arrayobj.populate_array(
                dest_arr,
                data=data,
                shape=shape_array,
                strides=strides_array,
                itemsize=itemsize,
                meminfo=builder.load(meminfo_ptr),
            )
            arrs.append(dest_arr._getvalue())

        # Update the timestamp tz array
        arr.data_ts = arrs[0]
        arr.data_offset = arrs[1]
        arr.null_bitmap = arrs[2]

        return arr._getvalue()

    # nullable integer/bool array
    if isinstance(
        arr_type,
        (
            IntegerArrayType,
            FloatingArrayType,
            DecimalArrayType,
            TimeArrayType,
            DatetimeArrayType,
        ),
    ) or arr_type in (
        boolean_array_type,
        datetime_date_array_type,
    ):
        arr = cgutils.create_struct_proxy(arr_type)(context, builder)
        np_dtype = arr_type.dtype
        if isinstance(arr_type, DecimalArrayType):
            np_dtype = int128_type
        elif isinstance(arr_type, DatetimeArrayType):
            np_dtype = bodo.datetime64ns
        elif arr_type == datetime_date_array_type:
            np_dtype = types.int32
        elif arr_type == boolean_array_type:
            # Boolean array stores bits so we can't use boolean.
            np_dtype = types.uint8
        data_arr_type = types.Array(np_dtype, 1, "C")
        data_arr = context.make_array(data_arr_type)(context, builder)
        nulls_arr_type = types.Array(types.uint8, 1, "C")
        nulls_arr = context.make_array(nulls_arr_type)(context, builder)

        length_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        n_bytes_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        data_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        nulls_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        meminfo_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        meminfo_nulls_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # info
                lir.IntType(64).as_pointer(),  # num_items
                lir.IntType(64).as_pointer(),  # num_bytes
                lir.IntType(8).as_pointer().as_pointer(),  # data
                lir.IntType(8).as_pointer().as_pointer(),  # nulls
                lir.IntType(8).as_pointer().as_pointer(),  # meminfo
                lir.IntType(8).as_pointer().as_pointer(),
            ],
        )  # meminfo_nulls
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="info_to_nullable_array"
        )
        builder.call(
            fn_tp,
            [
                in_info,
                length_ptr,
                n_bytes_ptr,
                data_ptr,
                nulls_ptr,
                meminfo_ptr,
                meminfo_nulls_ptr,
            ],
        )
        if raise_py_err:
            bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

        intp_t = context.get_value_type(types.intp)

        # data array
        if arr_type == boolean_array_type:
            data_length_ptr = n_bytes_ptr
        else:
            data_length_ptr = length_ptr
        shape_array = cgutils.pack_array(
            builder, [builder.load(data_length_ptr)], ty=intp_t
        )
        itemsize = context.get_constant(
            types.intp,
            context.get_abi_sizeof(context.get_data_type(np_dtype)),
        )
        strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

        data = builder.bitcast(
            builder.load(data_ptr),
            context.get_data_type(np_dtype).as_pointer(),
        )

        numba.np.arrayobj.populate_array(
            data_arr,
            data=data,
            shape=shape_array,
            strides=strides_array,
            itemsize=itemsize,
            meminfo=builder.load(meminfo_ptr),
        )
        arr.data = data_arr._getvalue()

        # nulls array
        shape_array = cgutils.pack_array(
            builder, [builder.load(n_bytes_ptr)], ty=intp_t
        )
        itemsize = context.get_constant(
            types.intp, context.get_abi_sizeof(context.get_data_type(types.uint8))
        )
        strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

        data = builder.bitcast(
            builder.load(nulls_ptr), context.get_data_type(types.uint8).as_pointer()
        )

        numba.np.arrayobj.populate_array(
            nulls_arr,
            data=data,
            shape=shape_array,
            strides=strides_array,
            itemsize=itemsize,
            meminfo=builder.load(meminfo_nulls_ptr),
        )
        arr.null_bitmap = nulls_arr._getvalue()
        if arr_type == boolean_array_type:
            # Boolean array needs the total array length.
            arr.length = builder.load(length_ptr)
        return arr._getvalue()

    # interval array
    if isinstance(arr_type, IntervalArrayType):
        arr = cgutils.create_struct_proxy(arr_type)(context, builder)
        left_arr = context.make_array(arr_type.arr_type)(context, builder)
        right_arr = context.make_array(arr_type.arr_type)(context, builder)

        length_ptr = cgutils.alloca_once(builder, lir.IntType(64))
        left_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        right_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        meminfo_left_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
        meminfo_right_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # info
                lir.IntType(64).as_pointer(),  # num_items
                lir.IntType(8).as_pointer().as_pointer(),  # left_ptr
                lir.IntType(8).as_pointer().as_pointer(),  # right_ptr
                lir.IntType(8).as_pointer().as_pointer(),  # left meminfo
                lir.IntType(8).as_pointer().as_pointer(),  # right meminfo
            ],
        )  # meminfo_nulls
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="info_to_interval_array"
        )
        builder.call(
            fn_tp,
            [
                in_info,
                length_ptr,
                left_ptr,
                right_ptr,
                meminfo_left_ptr,
                meminfo_right_ptr,
            ],
        )
        if raise_py_err:
            bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

        intp_t = context.get_value_type(types.intp)

        # left array
        shape_array = cgutils.pack_array(builder, [builder.load(length_ptr)], ty=intp_t)
        itemsize = context.get_constant(
            types.intp,
            context.get_abi_sizeof(context.get_data_type(arr_type.arr_type.dtype)),
        )
        strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

        left_data = builder.bitcast(
            builder.load(left_ptr),
            context.get_data_type(arr_type.arr_type.dtype).as_pointer(),
        )

        numba.np.arrayobj.populate_array(
            left_arr,
            data=left_data,
            shape=shape_array,
            strides=strides_array,
            itemsize=itemsize,
            meminfo=builder.load(meminfo_left_ptr),
        )
        arr.left = left_arr._getvalue()

        # right array
        right_data = builder.bitcast(
            builder.load(right_ptr),
            context.get_data_type(arr_type.arr_type.dtype).as_pointer(),
        )

        numba.np.arrayobj.populate_array(
            right_arr,
            data=right_data,
            shape=shape_array,
            strides=strides_array,
            itemsize=itemsize,
            meminfo=builder.load(meminfo_right_ptr),
        )
        arr.right = right_arr._getvalue()

        return arr._getvalue()

    raise_bodo_error(f"info_to_array(): array type {arr_type} is not supported")


def _gen_info_to_string_array(context, builder, arr_type, info_ptr, raise_py_err=True):
    """Generate LLVM code for converting array_info to string/binary array.
    Creates the necessary structs, allocates the array(item) payload meminfo, and
    calls info_to_string_array().
    """

    string_array = context.make_helper(builder, arr_type)
    array_item_data_type = ArrayItemArrayType(char_arr_type)
    array_item_array = context.make_helper(builder, array_item_data_type)

    # create payload type
    payload_type = ArrayItemArrayPayloadType(array_item_data_type)
    alloc_type = context.get_value_type(payload_type)
    alloc_size = context.get_abi_sizeof(alloc_type)

    # define dtor
    dtor_fn = define_array_item_dtor(
        context, builder, array_item_data_type, payload_type
    )

    # create meminfo
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, alloc_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, alloc_type.as_pointer())

    # alloc values in payload
    payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    char_arr = cgutils.create_struct_proxy(char_arr_type)(context, builder)

    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),  # info
            lir.IntType(64).as_pointer(),  # n_arrays
            lir.IntType(64).as_pointer(),  # n_chars
            context.get_value_type(types.MemInfoPointer(types.uint8)).as_pointer(),
            context.get_value_type(types.MemInfoPointer(offset_type)).as_pointer(),
            context.get_value_type(types.MemInfoPointer(types.uint8)).as_pointer(),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="info_to_string_array"
    )
    builder.call(
        fn_tp,
        [
            info_ptr,
            payload._get_ptr_by_name("n_arrays"),
            char_arr._get_ptr_by_name("length"),
            char_arr._get_ptr_by_name("meminfo"),
            payload._get_ptr_by_name("offsets"),
            payload._get_ptr_by_name("null_bitmap"),
        ],
    )
    if raise_py_err:
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)

    # C++ string array doesn't support offsets
    char_arr.meminfo_offset = context.get_constant(types.int64, 0)
    payload.data = char_arr._getvalue()
    builder.store(payload._getvalue(), meminfo_data_ptr)
    array_item_array.meminfo = meminfo
    string_array.data = array_item_array._getvalue()
    return string_array._getvalue()


@intrinsic
def info_to_array(typingctx, info_type, array_type):
    """convert array info wrapper from C++ to regular array object"""
    arr_type = unwrap_typeref(array_type)
    assert info_type == array_info_type, "info_to_array: expected info type"
    return arr_type(info_type, array_type), info_to_array_codegen


@typeof_impl.register(pd.arrays.ArrowExtensionArray)
def _typeof_pd_arrow_arr(val, c):
    return bodo.io.helpers.pyarrow_type_to_numba(val._pa_array.type)


def convert_arrow_arr_to_dict(arr, arrow_type):
    """Convert PyArrow array's type to a type with dictionary-encoding as specified by
    arrow_type. Not using Arrow's cast() since not working for nested arrays
    (as of Arrow 13).

    Args:
        arr (pa.Array): input PyArrow array
        arrow_type (DataType): target Arrow array type with dictionary encoding

    Returns:
        pa.Array: converted PyArrow array
    """
    if (
        pa.types.is_large_list(arrow_type)
        or pa.types.is_list(arrow_type)
        or pa.types.is_fixed_size_list(arrow_type)
    ):
        new_arr = arr.from_arrays(
            arr.offsets, convert_arrow_arr_to_dict(arr.values, arrow_type.value_type)
        )
        # Arrow's from_arrays ignores nulls (bug as of Arrow 14) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:2], children=[new_arr.values]
        )

    if pa.types.is_struct(arrow_type):
        new_arrs = [
            convert_arrow_arr_to_dict(arr.field(i), arrow_type.field(i).type)
            for i in range(arr.type.num_fields)
        ]
        names = [arr.type.field(i).name for i in range(arr.type.num_fields)]
        new_arr = arr.from_arrays(new_arrs, names)
        # Arrow's from_arrays ignores nulls (bug as of Arrow 14) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:1], children=new_arrs
        )

    if pa.types.is_map(arrow_type):
        new_keys = convert_arrow_arr_to_dict(arr.keys, arrow_type.key_type)
        new_items = convert_arrow_arr_to_dict(arr.items, arrow_type.item_type)
        new_arr = arr.from_arrays(arr.offsets, new_keys, new_items)
        # Arrow's from_arrays ignores nulls (bug as of Arrow 14) so we add them back manually
        buffs = new_arr.buffers()
        buffs[0] = pa.compute.invert(arr.is_null()).buffers()[1]
        return new_arr.from_buffers(
            new_arr.type, len(new_arr), buffs[:2], children=[new_arr.values]
        )

    if (
        pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)
    ) and pa.types.is_dictionary(arrow_type):
        return arr.dictionary_encode()

    return arr


def to_pa_arr(A, arrow_type, arrow_type_no_dict):
    """Convert input to PyArrow array with specified type"""
    if isinstance(A, pa.Array):
        return A

    if isinstance(A, pd.arrays.ArrowExtensionArray):
        return A._pa_array.combine_chunks()

    # Handle 2D string arrays unboxed as array(array(str))
    # See test_one_hot_encoder
    if isinstance(A, np.ndarray) and A.ndim == 2:
        A = [A[i] for i in range(len(A))]

    arr = pa.array(A, arrow_type_no_dict)

    if arrow_type != arrow_type_no_dict:
        arr = convert_arrow_arr_to_dict(arr, arrow_type)

    return arr


def unbox_array_using_arrow(typ, val, c):
    """Unboxing method for arrays using Arrow arrays (used by all nested array
    types and nullable array types)

    Args:
        typ (types.Type): Numba array type to unbox
        val (PyObject): array object to unbox
        c (_UnboxContext): Unboxing context

    Returns:
        NativeValue: Unboxed array
    """
    arrow_type, _ = bodo.io.helpers._numba_to_pyarrow_type(typ, use_dict_arr=True)
    arrow_type_no_dict, _ = bodo.io.helpers._numba_to_pyarrow_type(
        typ, use_dict_arr=False
    )

    to_pa_arr_obj = c.pyapi.unserialize(c.pyapi.serialize_object(to_pa_arr))
    arrow_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(arrow_type))
    arrow_type_no_dict_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(arrow_type_no_dict)
    )
    val = c.pyapi.call_function_objargs(
        to_pa_arr_obj, [val, arrow_type_obj, arrow_type_no_dict_obj]
    )
    c.pyapi.decref(to_pa_arr_obj)
    c.pyapi.decref(arrow_type_obj)
    c.pyapi.decref(arrow_type_no_dict_obj)

    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(),
        [lir.IntType(8).as_pointer()],
    )
    fn_tp = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="bodo_array_from_pyarrow_py_entry"
    )
    arr_info = c.builder.call(fn_tp, [val])

    bodo_array = bodo.libs.array.info_to_array_codegen(
        c.context,
        c.builder,
        typ(bodo.libs.array.array_info_type, typ),
        (arr_info, c.context.get_constant_null(typ)),
        # Avoid raising error in unboxing context to avoid calling convention issues
        raise_py_err=False,
    )

    # delete output array_info
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
        ],
    )
    fn_tp = cgutils.get_or_insert_function(c.builder.module, fnty, name="delete_info")
    c.builder.call(fn_tp, [arr_info])

    # decref since val is output of to_pa_list_arr() and not coming from user context
    c.pyapi.decref(val)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(bodo_array, is_error=is_error)


def _convert_to_pa_map_arr(arr, arrow_type):
    """Convert array(struct) returned from C++ to proper map array.
    NOTE: there could be nested maps inside.
    Also converts boolean array to null array since C++ doesn't support null array yet
    and uses boolean.

    Args:
        arr (pd.arrays.ArrowExtensionArray): input array(struct) array
        arrow_type (DataType): target Arrow map array type

    Returns:
        ArrowExtensionArray: equivalent map array
    """
    if arr.type == arrow_type:
        return arr

    # Convert list(struct) to map
    if (
        (
            pa.types.is_large_list(arr.type)
            or pa.types.is_list(arr.type)
            or pa.types.is_fixed_size_list(arr.type)
        )
        and pa.types.is_struct(arr.type.value_type)
        and pa.types.is_map(arrow_type)
    ):
        new_arr = pa.MapArray.from_arrays(
            arr.offsets,
            arr.values.field(0),
            _convert_to_pa_map_arr(arr.values.field(1), arrow_type.item_type),
        )
        # Arrow's from_arrays ignores nulls (bug as of Arrow 13) so we add them back manually
        buffs = new_arr.buffers()
        buffs[0] = pa.compute.invert(arr.is_null()).buffers()[1]
        return new_arr.from_buffers(
            new_arr.type, len(new_arr), buffs[:2], children=[new_arr.values]
        )

    # Handle struct recursively
    if pa.types.is_struct(arr.type):
        new_arrs = [
            _convert_to_pa_map_arr(arr.field(i), arrow_type.field(i).type)
            for i in range(arr.type.num_fields)
        ]
        names = [arr.type.field(i).name for i in range(arr.type.num_fields)]
        new_arr = pa.StructArray.from_arrays(new_arrs, names)
        # Arrow's from_arrays ignores nulls (bug as of Arrow 13) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:1], children=new_arrs
        )

    # Handle list recursively
    if (
        pa.types.is_large_list(arr.type)
        or pa.types.is_list(arr.type)
        or pa.types.is_fixed_size_list(arr.type)
    ):
        new_arr = pa.LargeListArray.from_arrays(
            arr.offsets, _convert_to_pa_map_arr(arr.values, arrow_type.value_type)
        )
        # Arrow's from_arrays ignores nulls (bug as of Arrow 13) so we add them back manually
        return pa.Array.from_buffers(
            new_arr.type, len(new_arr), arr.buffers()[:2], children=[new_arr.values]
        )

    # Convert bool array to null array (since C++ doesn't support null array yet and
    # uses bool array)
    if arrow_type == pa.null() and pa.types.is_boolean(arr.type):
        return pa.NullArray.from_buffers(arrow_type, len(arr), arr.buffers()[:1])

    return arr


def fix_boxed_nested_array(arr, arrow_type):
    """Convert array returned from C++ to proper array type since C++ doesn't have
    map array and timezone information.
    NOTE: there could be differences inside nested arrays

    Args:
        arr (pd.arrays.ArrowExtensionArray): input array
        arrow_type (DataType): target Arrow array type

    Returns:
        ArrowExtensionArray: equivalent array with proper type
    """
    arr = arr._pa_array.combine_chunks()
    new_arr = _convert_to_pa_map_arr(arr, arrow_type)

    # Bodo C++ doesn't have details like timezone which need fixed
    if new_arr.type != arrow_type:
        new_arr = pa.compute.cast(arr, arrow_type)

    # Convert struct array workaround for tuple array back to object array
    if (
        pa.types.is_struct(new_arr.type)
        and (new_arr.type.num_fields > 0)
        and all(
            new_arr.type.field(i).name.startswith(TUPLE_ARRAY_SENTINEL)
            for i in range(new_arr.type.num_fields)
        )
    ):
        warnings.warn(
            "Returning object arrays during boxing since tuple arrays are not supported by Arrow. Use struct arrays for better performance."
        )
        out = pd.array(new_arr, pd.ArrowDtype(new_arr.type))
        return pd.Series(
            [None if pd.isna(a) else tuple(a.values()) for a in out]
        ).values

    return pd.arrays.ArrowExtensionArray(new_arr)


def box_array_using_arrow(typ, val, c):
    """Boxing method for arrays using Arrow arrays (used by all nested array
    types and nullable arrays)

    Args:
        typ (types.Type): Numba array type to box
        val (PyObject): array object to box
        c (_BoxContext): boxing context

    Returns:
        PyObject: boxed Pandas PyArrow array
    """

    arr_info = array_to_info_codegen(c.context, c.builder, array_info_type(typ), (val,))
    fnty = lir.FunctionType(
        c.pyapi.pyobj,
        [
            lir.IntType(8).as_pointer(),
        ],
    )
    box_fname = "pd_pyarrow_array_from_bodo_array_py_entry"
    fn_get = cgutils.get_or_insert_function(c.builder.module, fnty, name=box_fname)
    arr = c.builder.call(
        fn_get,
        [
            arr_info,
        ],
    )

    # Convert array(struct) returned from C++ to proper array type (map array, timezone)
    # NOTE: needs to handle nested cases
    to_pa_map_arr_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(fix_boxed_nested_array)
    )
    arrow_type, _ = bodo.io.helpers._numba_to_pyarrow_type(typ, use_dict_arr=True)
    arrow_type_obj = c.pyapi.unserialize(c.pyapi.serialize_object(arrow_type))
    map_arr = c.pyapi.call_function_objargs(to_pa_map_arr_obj, [arr, arrow_type_obj])
    c.pyapi.decref(to_pa_map_arr_obj)
    c.pyapi.decref(arr)
    c.pyapi.decref(arrow_type_obj)
    arr = map_arr

    c.context.nrt.decref(c.builder, typ, val)
    return arr


@intrinsic
def arr_info_list_to_table(typingctx, list_arr_info_typ=None):
    assert list_arr_info_typ == types.List(array_info_type)
    return table_type(list_arr_info_typ), arr_info_list_to_table_codegen


def arr_info_list_to_table_codegen(context, builder, sig, args):
    """
    Codegen for arr_info_list_to_table. This isn't a closure so it can be
    called from other intrinsics.
    """
    (info_list,) = args
    inst = numba.cpython.listobj.ListInstance(context, builder, sig.args[0], info_list)
    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(),
        [lir.IntType(8).as_pointer().as_pointer(), lir.IntType(64)],
    )
    fn_tp = cgutils.get_or_insert_function(
        builder.module, fnty, name="arr_info_list_to_table"
    )
    return builder.call(fn_tp, [inst.data, inst.size])


def array_from_cpp_table_codegen(context, builder, sig, args):
    """codegen for array_from_cpp_table() below"""
    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(), lir.IntType(64)]
    )
    fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="info_from_table")
    info_ptr = builder.call(fn_tp, args[:2])
    out_arr = info_to_array_codegen(
        context,
        builder,
        sig.return_type(array_info_type, sig.args[2]),
        (info_ptr, args[2]),
    )

    # delete array_info pointer returned from info_from_table()
    fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
    fn_tp = cgutils.get_or_insert_function(builder.module, fnty, name="delete_info")
    builder.call(fn_tp, [info_ptr])

    return out_arr


@intrinsic
def array_from_cpp_table(typingctx, table_t, ind_t, array_type_t):
    """Return a Python array from a column of a C++ table pointer

    Args:
        typingctx (TypingContext): part of intrinsic interface but unused
        table_t (table_type): input C++ table
        ind_t (int): column index
        array_type_t (array|TypeRef): data type of output array

    Returns:
        array: Python array extracted from C++ table
    """
    arr_type = unwrap_typeref(array_type_t)
    return arr_type(table_t, ind_t, array_type_t), array_from_cpp_table_codegen


@intrinsic
def cpp_table_to_py_table(
    typingctx, cpp_table_t, table_idx_arr_t, py_table_type_t, default_length_t
):
    """Extract columns of a C++ table and create a Python table.
    table_index_arr specifies which columns to extract

    default_length_t: Holds the length to set if all columns are dead. If an API doesn't support
    this then the caller should pass 0.
    """
    assert cpp_table_t == table_type, "invalid cpp table type"
    assert (
        isinstance(table_idx_arr_t, types.Array)
        and table_idx_arr_t.dtype == types.int64
    ), "invalid table index array"
    assert isinstance(py_table_type_t, types.TypeRef), "invalid py table ref"
    assert isinstance(default_length_t, types.Integer), "invalid length type"
    py_table_type = py_table_type_t.instance_type

    def codegen(context, builder, sig, args):
        cpp_table, table_idx_arr, _, default_length = args

        # create python table
        table = cgutils.create_struct_proxy(py_table_type)(context, builder)
        table.parent = cgutils.get_null_value(table.parent.type)
        cpp_table_idx_struct = context.make_array(table_idx_arr_t)(
            context, builder, table_idx_arr
        )
        neg_one = context.get_constant(types.int64, -1)
        len_ptr = cgutils.alloca_once_value(builder, default_length)

        # generate code for each block
        for t, blk in py_table_type.type_to_blk.items():
            n_arrs = context.get_constant(
                types.int64, len(py_table_type.block_to_arr_ind[blk])
            )
            # not using allocate() since its exception causes calling convention error
            _, out_arr_list = ListInstance.allocate_ex(
                context, builder, types.List(t), n_arrs
            )
            out_arr_list.size = n_arrs
            # lower array of array indices for block to use within the loop
            # using array since list doesn't have constant lowering
            arr_inds = context.make_constant_array(
                builder,
                types.Array(types.int64, 1, "C"),
                np.array(py_table_type.block_to_arr_ind[blk], dtype=np.int64),
            )
            arr_inds_struct = context.make_array(types.Array(types.int64, 1, "C"))(
                context, builder, arr_inds
            )
            with cgutils.for_range(builder, n_arrs) as loop:
                i = loop.index
                # get array index in Python table
                arr_ind = _getitem_array_single_int(
                    context,
                    builder,
                    types.int64,
                    types.Array(types.int64, 1, "C"),
                    arr_inds_struct,
                    i,
                )
                # get array info index in C++ table
                cpp_table_ind = _getitem_array_single_int(
                    context,
                    builder,
                    types.int64,
                    table_idx_arr_t,
                    cpp_table_idx_struct,
                    arr_ind,
                )
                is_loaded = builder.icmp_unsigned("!=", cpp_table_ind, neg_one)
                with builder.if_else(is_loaded) as (then, orelse):
                    with then:
                        # extract info and convert to array
                        arr = array_from_cpp_table_codegen(
                            context,
                            builder,
                            t(table_type, types.int64, types.TypeRef(t)),
                            [
                                cpp_table,
                                cpp_table_ind,
                                context.get_constant_null(types.TypeRef(t)),
                            ],
                        )

                        out_arr_list.inititem(i, arr, incref=False)
                        length = context.compile_internal(
                            builder,
                            lambda arr: len(arr),
                            types.int64(t),
                            [arr],
                        )
                        builder.store(length, len_ptr)
                    with orelse:
                        # Initialize the list value to null otherwise
                        null_ptr = context.get_constant_null(t)
                        out_arr_list.inititem(i, null_ptr, incref=False)

            setattr(table, f"block_{blk}", out_arr_list.value)

        table.len = builder.load(len_ptr)
        return table._getvalue()

    return (
        py_table_type(cpp_table_t, table_idx_arr_t, py_table_type_t, types.int64),
        codegen,
    )


@numba.generated_jit(
    nopython=True, no_cpython_wrapper=True, no_unliteral=True, cache=True
)
def cpp_table_to_py_data(
    cpp_table,
    out_col_inds_t,
    out_types_t,
    n_rows_t,
    n_table_cols_t,
    unknown_cat_arrs_t=None,
    cat_inds_t=None,
):
    """convert C++ table to Python data with types described in out_types_t. The first
    output is a TableType or array or none, and there rest are arrays.

    Args:
        cpp_table (C++ table_type): C++ table type to convert
        out_col_inds_t (MetaType(list[int])): list of logical column numbers for each
            C++ table column.
        out_types_t (Tuple(TypeRef[TableType|array|NoneType], TypeRef[array], ...)):
            data types of output Python data that form logical columns. If the first
            type is a table type, it has logical columns 0..n_py_table_arrs-1.
            The first type can be a regular array or none as well.
            The rest are regular arrays that have the rest of data (n_py_table_arrs,
            n_py_table_arrs+1, ...)
        n_rows_t (int): Number of rows in output table. Necessary since all table
            columns may be dead, but the length of the table may be used.
        n_table_cols_t (int): number of logical columns in the table structure (vs.
            extra arrays). Necessary since the table may be dead and out_types_t[0] may
            be types.none (table type not always available).
        unknown_cat_arrs_t (Tuple(array) | none): Reference arrays for output
            categorical arrays with unknown categories (one for each). Necessary for
            creating proper output array from cpp array info.
            If not passed, the corresponding out_types_t should have the reference
            array (easy to pass reference array in out_types_t in case of Sort since
            each output corresponds to an input in same position).
        cat_inds_t (MetaType(Tuple(int))): Logical output indices of reference arrays
            passed in unknown_cat_arrs_t.

    Returns:
        Tuple(Table, array, ...): python data corresponding to input C++ table
    """
    out_col_inds = out_col_inds_t.instance_type.meta
    py_table_type = unwrap_typeref(out_types_t.types[0])
    # arrays to return after filling the Python table, includes dead output for easier
    # logical index handling
    extra_arr_types = [
        unwrap_typeref(out_types_t.types[i]) for i in range(1, len(out_types_t.types))
    ]

    glbls = {}
    n_py_table_arrs = get_overload_const_int(n_table_cols_t)
    py_to_cpp_inds = {k: i for i, k in enumerate(out_col_inds)}

    # map output column number to index in unknown_cat_arrs_t
    if not is_overload_none(unknown_cat_arrs_t):
        cat_arr_inds = {c: i for i, c in enumerate(cat_inds_t.instance_type.meta)}

    # basic structure:
    # for each block in table:
    #   for each array_ind in block:
    #     if array_ind in output_inds:
    #       block[array_ind] = info_to_array(cpp_table[out_ind])

    out_vars = []

    func_text = "def bodo_cpp_table_to_py_data(cpp_table, out_col_inds_t, out_types_t, n_rows_t, n_table_cols_t, unknown_cat_arrs_t=None, cat_inds_t=None):\n"

    if isinstance(py_table_type, bodo.TableType):
        func_text += "  py_table = init_table(py_table_type, False)\n"
        func_text += "  py_table = set_table_len(py_table, n_rows_t)\n"

        for typ, blk in py_table_type.type_to_blk.items():
            out_inds = [
                py_to_cpp_inds.get(i, -1) for i in py_table_type.block_to_arr_ind[blk]
            ]
            glbls[f"out_inds_{blk}"] = np.array(out_inds, np.int64)
            glbls[f"out_type_{blk}"] = typ
            glbls[f"typ_list_{blk}"] = types.List(typ)
            out_type = f"out_type_{blk}"
            if type_has_unknown_cats(typ):
                # use unknown_cat_arrs_t if provided (Aggregate case)
                # use input data (assuming corresponds to output) if unknown_cat_arrs_t
                # is not provided (Sort case)
                if is_overload_none(unknown_cat_arrs_t):
                    func_text += f"  in_arr_list_{blk} = get_table_block(out_types_t[0], {blk})\n"
                    out_type = f"in_arr_list_{blk}[i]"
                else:
                    glbls[f"cat_arr_inds_{blk}"] = np.array(
                        [
                            cat_arr_inds.get(i, -1)
                            for i in py_table_type.block_to_arr_ind[blk]
                        ],
                        np.int64,
                    )
                    out_type = f"unknown_cat_arrs_t[cat_arr_inds_{blk}[i]]"
            n_arrs = len(py_table_type.block_to_arr_ind[blk])
            func_text += (
                f"  arr_list_{blk} = alloc_list_like(typ_list_{blk}, {n_arrs}, False)\n"
            )
            func_text += f"  for i in range(len(arr_list_{blk})):\n"
            func_text += f"    cpp_ind_{blk} = out_inds_{blk}[i]\n"
            func_text += f"    if cpp_ind_{blk} == -1:\n"
            func_text += "      continue\n"
            func_text += f"    arr_{blk} = array_from_cpp_table(cpp_table, cpp_ind_{blk}, {out_type})\n"
            func_text += f"    arr_list_{blk}[i] = arr_{blk}\n"
            func_text += (
                f"  py_table = set_table_block(py_table, arr_list_{blk}, {blk})\n"
            )
        out_vars.append("py_table")
    elif py_table_type != types.none:
        # regular array case
        out_ind = py_to_cpp_inds.get(0, -1)
        if out_ind != -1:
            glbls["arr_typ_arg0"] = py_table_type
            out_type = "arr_typ_arg0"
            if type_has_unknown_cats(py_table_type):
                if is_overload_none(unknown_cat_arrs_t):
                    out_type = "out_types_t[0]"
                else:
                    out_type = f"unknown_cat_arrs_t[{cat_arr_inds[0]}]"
            func_text += (
                f"  out_arg0 = array_from_cpp_table(cpp_table, {out_ind}, {out_type})\n"
            )
            out_vars.append("out_arg0")

    for i, t in enumerate(extra_arr_types):
        out_ind = py_to_cpp_inds.get(n_py_table_arrs + i, -1)
        if out_ind != -1:
            glbls[f"extra_arr_type_{i}"] = t
            out_type = f"extra_arr_type_{i}"
            if type_has_unknown_cats(t):
                if is_overload_none(unknown_cat_arrs_t):
                    out_type = f"out_types_t[{i + 1}]"
                else:
                    out_type = (
                        f"unknown_cat_arrs_t[{cat_arr_inds[n_py_table_arrs + i]}]"
                    )
            func_text += (
                f"  out_{i} = array_from_cpp_table(cpp_table, {out_ind}, {out_type})\n"
            )
            out_vars.append(f"out_{i}")

    comma = "," if len(out_vars) == 1 else ""
    func_text += f"  return ({', '.join(out_vars)}{comma})\n"

    glbls.update(
        {
            "init_table": bodo.hiframes.table.init_table,
            "alloc_list_like": bodo.hiframes.table.alloc_list_like,
            "set_table_block": bodo.hiframes.table.set_table_block,
            "set_table_len": bodo.hiframes.table.set_table_len,
            "get_table_block": bodo.hiframes.table.get_table_block,
            "array_from_cpp_table": array_from_cpp_table,
            "out_col_inds": list(out_col_inds),
            "py_table_type": py_table_type,
        }
    )

    return bodo_exec(func_text, glbls, {}, __name__)


@intrinsic
def py_table_to_cpp_table(typingctx, py_table_t, py_table_type_t):
    """Extract columns of a Python table and creates a C++ table."""
    assert isinstance(py_table_t, bodo.hiframes.table.TableType), (
        "invalid py table type"
    )
    assert isinstance(py_table_type_t, types.TypeRef), "invalid py table ref"
    py_table_type = py_table_type_t.instance_type

    def codegen(context, builder, sig, args):
        py_table, _ = args
        # Table info.
        table_struct = cgutils.create_struct_proxy(py_table_type)(
            context, builder, py_table
        )

        if py_table_type.has_runtime_cols:
            # For runtime columns compute the length of each list
            n_total_arrs = lir.Constant(lir.IntType(64), 0)
            for blk, t in enumerate(py_table_type.arr_types):
                arr_list = getattr(table_struct, f"block_{blk}")
                arr_list_inst = ListInstance(context, builder, types.List(t), arr_list)
                n_total_arrs = builder.add(n_total_arrs, arr_list_inst.size)
        else:
            n_total_arrs = lir.Constant(lir.IntType(64), len(py_table_type.arr_types))
        # Allocate a list for arrays.
        # Note: This function assumes we don't have any dead
        # columns and needs to be updated if this changes
        _, table_arr_list = ListInstance.allocate_ex(
            context, builder, types.List(array_info_type), n_total_arrs
        )
        table_arr_list.size = n_total_arrs
        # generate code for each block. This should call array_to_info
        # and insert into the list.
        if py_table_type.has_runtime_cols:
            # Runtime columns are assumed to have their actual and logical
            # order match. This means that block_0 contains the first k columns,
            # with array 0 being column 0, array 1, being column 1, ...,
            # array k being column k. Then block_1 has column 0 as block k + 1,
            # and so on.
            #
            # This works because we rely on operations that output runtime columns
            # (such as pivot) to either define the output column order
            # or operations that access a particular column number (i.e. getitem)
            # should be forbidden.
            cpp_table_idx = lir.Constant(lir.IntType(64), 0)
            for blk, t in enumerate(py_table_type.arr_types):
                arr_list = getattr(table_struct, f"block_{blk}")
                arr_list_inst = ListInstance(context, builder, types.List(t), arr_list)
                n_arrs = arr_list_inst.size
                with cgutils.for_range(builder, n_arrs) as loop:
                    i = loop.index
                    # Note: We don't unbox here because runtime columns should never
                    # require unboxing.
                    # Get the array
                    arr = arr_list_inst.getitem(i)
                    # Call array to info
                    array_to_info_sig = signature(array_info_type, t)
                    array_to_info_args = (arr,)
                    array_info_val = array_to_info_codegen(
                        context, builder, array_to_info_sig, array_to_info_args
                    )
                    # We simply assign to the next location in the table. We simply
                    # increment by 1 each time). The actually increment happens
                    # at the end because there appear to be control flow issues
                    # when reassigning inside the loop.
                    table_arr_list.inititem(
                        builder.add(cpp_table_idx, i), array_info_val, incref=False
                    )
                # Increment the cpp_table_idx
                cpp_table_idx = builder.add(cpp_table_idx, n_arrs)
        else:
            for t, blk in py_table_type.type_to_blk.items():
                n_arrs = context.get_constant(
                    types.int64, len(py_table_type.block_to_arr_ind[blk])
                )
                arr_list = getattr(table_struct, f"block_{blk}")
                arr_list_inst = ListInstance(context, builder, types.List(t), arr_list)
                # lower array of array indices for block to use within the loop
                # using array since list doesn't have constant lowering
                arr_inds = context.make_constant_array(
                    builder,
                    types.Array(types.int64, 1, "C"),
                    np.array(py_table_type.block_to_arr_ind[blk], dtype=np.int64),
                )
                arr_inds_struct = context.make_array(types.Array(types.int64, 1, "C"))(
                    context, builder, arr_inds
                )
                with cgutils.for_range(builder, n_arrs) as loop:
                    i = loop.index
                    # get array index in total list
                    arr_ind = _getitem_array_single_int(
                        context,
                        builder,
                        types.int64,
                        types.Array(types.int64, 1, "C"),
                        arr_inds_struct,
                        i,
                    )
                    # Ensure we don't need to unbox
                    ensure_column_unboxed_sig = signature(
                        types.none,
                        py_table_type,
                        types.List(t),
                        types.int64,
                        types.int64,
                    )
                    ensure_column_unboxed_args = (py_table, arr_list, i, arr_ind)
                    bodo.hiframes.table.ensure_column_unboxed_codegen(
                        context,
                        builder,
                        ensure_column_unboxed_sig,
                        ensure_column_unboxed_args,
                    )
                    # Get the array
                    arr = arr_list_inst.getitem(i)
                    # Call array to info
                    array_to_info_sig = signature(array_info_type, t)
                    array_to_info_args = (arr,)
                    array_info_val = array_to_info_codegen(
                        context, builder, array_to_info_sig, array_to_info_args
                    )
                    table_arr_list.inititem(arr_ind, array_info_val, incref=False)

        list_val = table_arr_list.value
        arr_info_list_to_table_sig = signature(table_type, types.List(array_info_type))
        arr_info_list_to_table_args = (list_val,)
        cpp_table = arr_info_list_to_table_codegen(
            context, builder, arr_info_list_to_table_sig, arr_info_list_to_table_args
        )
        # Decref the intermediate list.
        context.nrt.decref(builder, types.List(array_info_type), list_val)
        return cpp_table

    return table_type(py_table_type, py_table_type_t), codegen


def py_data_to_cpp_table(py_table, extra_arrs_tup, in_col_inds, n_table_cols):
    pass


@infer_global(py_data_to_cpp_table)
class PyDataToCppTableInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 4
        py_table, extra_arrs_tup, _, n_table_cols_t = args

        assert py_table == types.none or isinstance(py_table, bodo.TableType)
        assert isinstance(extra_arrs_tup, types.BaseTuple)
        assert all(
            isinstance(t, types.ArrayCompatible) or t == types.none
            for t in extra_arrs_tup
        ), f"extra_arrs_tup must be a tuple of arrays or None, is {extra_arrs_tup}"

        if not is_overload_constant_int(n_table_cols_t):
            raise_bodo_error(
                "py_data_to_cpp_table:: n_table_cols must be a constant integer"
            )

        return signature(table_type, *args)


PyDataToCppTableInfer._no_unliteral = True


@lower_builtin(py_data_to_cpp_table, types.VarArg(types.Any))
def lower_py_data_to_cpp_table(context, builder, sig, args):
    """lower table_filter() using gen_table_filter_impl above"""
    impl = gen_py_data_to_cpp_table_impl(*sig.args)
    return cached_call_internal(context, builder, impl, sig, args)


def gen_py_data_to_cpp_table_impl(
    py_table, extra_arrs_tup, in_col_inds_t, n_table_cols_t
):
    """Convert Python data (table and arrays) to a C++ table.
    Args:
        py_table (TableType): Python table to convert
        extra_arrs_tup (tuple(array)): extra arrays to convert, includes dead columns
            for easier logical index handling.
        in_col_inds_t (MetaType(list[int])): logical indices in input for each C++
            output table column. Actual input data is a table that has logical
            columns 0..n_py_table_arrs-1, and regular arrays that have the rest of data
            (n_py_table_arrs, n_py_table_arrs+1, ...).

    Returns:
        C++ table: converted C++ table
    """
    in_col_inds = in_col_inds_t.instance_type.meta

    glbls = {}
    n_py_table_arrs = get_overload_const_int(n_table_cols_t)

    # Technically a single Python array can hold multiple spots
    # in the C++ table. As a result we look for any duplicates
    # to process with regular array format.
    duplicates = defaultdict(list)
    py_to_cpp_inds = {}
    for i, k in enumerate(in_col_inds):
        if k in py_to_cpp_inds:
            duplicates[k].append(i)
        else:
            py_to_cpp_inds[k] = i

    # basic structure:
    # for each block in py_table:
    #   for each array_ind in block:
    #     if array_ind in output_inds:
    #       output_list[out_ind] = array_to_info(block[array_ind])

    func_text = "def bodo_impl_py_data_to_cpp_table(py_table, extra_arrs_tup, in_col_inds_t, n_table_cols_t):\n"
    func_text += (
        f"  cpp_arr_list = alloc_empty_list_type({len(in_col_inds)}, array_info_type)\n"
    )

    if py_table != types.none:
        for blk in py_table.type_to_blk.values():
            out_inds = [
                py_to_cpp_inds.get(i, -1) for i in py_table.block_to_arr_ind[blk]
            ]
            glbls[f"out_inds_{blk}"] = np.array(out_inds, np.int64)
            glbls[f"arr_inds_{blk}"] = np.array(
                py_table.block_to_arr_ind[blk], np.int64
            )
            func_text += f"  arr_list_{blk} = get_table_block(py_table, {blk})\n"
            func_text += f"  for i in range(len(arr_list_{blk})):\n"
            func_text += f"    out_arr_ind_{blk} = out_inds_{blk}[i]\n"
            func_text += f"    if out_arr_ind_{blk} == -1:\n"
            func_text += "      continue\n"
            func_text += f"    arr_ind_{blk} = arr_inds_{blk}[i]\n"
            func_text += f"    ensure_column_unboxed(py_table, arr_list_{blk}, i, arr_ind_{blk})\n"
            func_text += f"    cpp_arr_list[out_arr_ind_{blk}] = array_to_info(arr_list_{blk}[i])\n"

        # Handle any table duplicates as individual arrays.
        for arr_num, ind_list in duplicates.items():
            if arr_num < n_py_table_arrs:
                blk = py_table.block_nums[arr_num]
                in_ind = py_table.block_offsets[arr_num]
                for out_ind in ind_list:
                    # Since this is a duplicate the array must already be loaded.
                    func_text += f"  cpp_arr_list[{out_ind}] = array_to_info(arr_list_{blk}[{in_ind}])\n"

    for i in range(len(extra_arrs_tup)):
        first_ind = py_to_cpp_inds.get(n_py_table_arrs + i, -1)
        if first_ind != -1:
            total_out_inds = [first_ind] + duplicates.get(n_py_table_arrs + i, [])
            for out_ind in total_out_inds:
                func_text += (
                    f"  cpp_arr_list[{out_ind}] = array_to_info(extra_arrs_tup[{i}])\n"
                )

    func_text += "  return arr_info_list_to_table(cpp_arr_list)\n"

    glbls.update(
        {
            "array_info_type": array_info_type,
            "alloc_empty_list_type": bodo.hiframes.table.alloc_empty_list_type,
            "get_table_block": bodo.hiframes.table.get_table_block,
            "ensure_column_unboxed": bodo.hiframes.table.ensure_column_unboxed,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
        }
    )

    return bodo_exec(func_text, glbls, {}, __name__)


delete_info = types.ExternalFunction(
    "delete_info",
    types.void(array_info_type),
)


delete_table = types.ExternalFunction(
    "delete_table",
    types.void(table_type),
)


cpp_table_map_to_list = types.ExternalFunction(
    "cpp_table_map_to_list",
    table_type(table_type),
)


# TODO add a test for this
@intrinsic
def concat_tables_cpp(typing_ctx, table_info_list_t):  # pragma: no cover
    assert table_info_list_t == types.List(table_type), (
        "table_info_list_t must be a list of table_type"
    )

    def codegen(context, builder, sig, args):
        (info_list,) = args
        list_inst = numba.cpython.listobj.ListInstance(
            context, builder, sig.args[0], info_list
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="concat_tables_py_entry"
        )
        args = [list_inst.data, list_inst.size]
        res = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return res

    return table_type(table_info_list_t), codegen


@intrinsic
def union_tables_cpp(typing_ctx, table_info_list_t, drop_duplicates_t, is_parallel_t):
    assert table_info_list_t == types.List(table_type), (
        "table_info_list_t must be a list of table_type"
    )
    assert types.unliteral(drop_duplicates_t) == types.bool_, (
        "drop_duplicates_t must be an boolean"
    )
    assert types.unliteral(is_parallel_t) == types.bool_, (
        "is_parallel_t must be an boolean"
    )

    def codegen(context, builder, sig, args):
        info_list, drop_duplicates, is_parallel = args
        list_inst = numba.cpython.listobj.ListInstance(
            context, builder, sig.args[0], info_list
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="union_tables"
        )
        args = [list_inst.data, list_inst.size, drop_duplicates, is_parallel]
        res = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return res

    return table_type(table_info_list_t, types.bool_, types.bool_), codegen


def union_tables(tables_tup, drop_duplicates, is_parallel=False):
    pass


@overload(union_tables)
def overload_union_tables(table_tup, drop_duplicates, out_table_typ, is_parallel=False):
    """
    Wrapper around union_tables that allows setting is_parallel in distributed pass.
    """
    table_typs = table_tup.types
    # All inputs have the same number of columns, so generate info from the first input.
    base_typ = table_typs[0]
    if isinstance(base_typ, bodo.TableType):
        n_cols = len(base_typ.arr_types)
    else:
        # Input must be a tuple of arrays.
        n_cols = len(base_typ.types)

    glbls = {
        "bodo": bodo,
        "in_col_inds": MetaType(tuple(range(n_cols))),
        "out_col_inds": np.array(range(n_cols), dtype=np.int64),
    }

    func_text = (
        "def impl(table_tup, drop_duplicates, out_table_typ, is_parallel=False):\n"
    )
    # Step 1: Convert each of the inputs to a C++ table.
    for i, table_typ in enumerate(table_typs):
        if isinstance(table_typ, bodo.TableType):
            func_text += f"  table{i} = table_tup[{i}]\n"
            func_text += f"  arrs{i} = ()\n"
            table_cols = n_cols
        else:
            func_text += f"  table{i} = None\n"
            func_text += f"  arrs{i} = table_tup[{i}]\n"
            table_cols = 0
        func_text += f"  cpp_table{i} = bodo.libs.array.py_data_to_cpp_table(table{i}, arrs{i}, in_col_inds, {table_cols})\n"
    # Step 2 generate code to union the C++ tables.
    tables = [f"cpp_table{i}" for i in range(len(table_typs))]
    tuple_inputs = ", ".join(tables)
    func_text += f"  out_cpp_table = bodo.libs.array.union_tables_cpp([{tuple_inputs}], drop_duplicates, is_parallel)\n"
    # Step 3 convert the C++ table to a Python table.
    func_text += "  out_py_table = bodo.libs.array.cpp_table_to_py_table(out_cpp_table, out_col_inds, out_table_typ, 0)\n"
    # Step 4 free the output C++ table without modifying the refcounts.
    func_text += "  bodo.libs.array.delete_table(out_cpp_table)\n"
    func_text += "  return out_py_table\n"
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]


# TODO Add a test for this
@intrinsic
def shuffle_table(
    typingctx, table_t, n_keys_t, _is_parallel, keep_comm_info_t
):  # pragma: no cover
    """shuffle input table so that rows with same key are on the same process.
    Steals a reference from the input table.
    'keep_comm_info' parameter specifies if shuffle information should be kept in
    output table, to be used for reverse shuffle later (e.g. in groupby apply).
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):  # pragma: no cover
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(32),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="shuffle_table_py_entrypt"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return table_type(table_t, types.int64, types.boolean, types.int32), codegen


class ShuffleInfoType(types.Type):
    def __init__(self):
        super().__init__(name="ShuffleInfoType()")


shuffle_info_type = ShuffleInfoType()
register_model(ShuffleInfoType)(models.OpaqueModel)


get_shuffle_info = types.ExternalFunction(
    "get_shuffle_info",
    shuffle_info_type(table_type),
)


@intrinsic
def delete_shuffle_info(typingctx, shuffle_info_t=None):
    """delete shuffle info data if not none"""

    def codegen(context, builder, sig, args):
        if sig.args[0] == types.none:
            return

        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="delete_shuffle_info"
        )
        builder.call(fn_tp, args)

    return types.none(shuffle_info_t), codegen


@intrinsic
def reverse_shuffle_table(typingctx, table_t, shuffle_info_t):
    """call reverse shuffle if shuffle info not none"""

    def codegen(context, builder, sig, args):
        if sig.args[-1] == types.none:
            return context.get_constant_null(table_type)

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer()],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="reverse_shuffle_table"
        )
        return builder.call(fn_tp, args)

    return table_type(table_type, shuffle_info_t), codegen


@intrinsic
def get_null_shuffle_info(typingctx):
    """return a null shuffle info object"""

    def codegen(context, builder, sig, args):
        return context.get_constant_null(sig.return_type)

    return shuffle_info_type(), codegen


@intrinsic
def hash_join_table(
    typingctx,
    left_table_t,
    right_table_t,
    left_parallel_t,
    right_parallel_t,
    n_keys_t,
    n_data_left_t,
    n_data_right_t,
    same_vect_t,
    key_in_out_t,
    same_need_typechange_t,
    is_left_t,
    is_right_t,
    is_join_t,
    extra_data_col_t,
    indicator,
    _bodo_na_equal,
    _bodo_rebalance_output_if_skewed,
    cond_func,
    left_col_nums,
    left_col_nums_len,
    right_col_nums,
    right_col_nums_len,
    num_rows_ptr_t,
):
    """
    Interface to the hash join of two tables.
    """
    assert left_table_t == table_type
    assert right_table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="hash_join_table"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            left_table_t,
            right_table_t,
            types.boolean,
            types.boolean,
            types.int64,
            types.int64,
            types.int64,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.int64,
            types.voidptr,
            types.int64,
            types.voidptr,
        ),
        codegen,
    )


@intrinsic
def nested_loop_join_table(
    typingctx,
    left_table_t,
    right_table_t,
    left_parallel_t,
    right_parallel_t,
    is_left_t,
    is_right_t,
    key_in_output_t,
    need_typechange_t,
    _bodo_rebalance_output_if_skewed,
    cond_func,
    left_col_nums,
    left_col_nums_len,
    right_col_nums,
    right_col_nums_len,
    num_rows_ptr_t,
):
    """
    Call cpp function for cross join of two tables.
    """
    assert left_table_t == table_type, "nested_loop_join_table: cpp table type expected"
    assert right_table_t == table_type, (
        "nested_loop_join_table: cpp table type expected"
    )

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="nested_loop_join_table"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            left_table_t,
            right_table_t,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.int64,
            types.voidptr,
            types.int64,
            types.voidptr,
        ),
        codegen,
    )


@intrinsic
def interval_join_table(
    typingctx,
    left_table_t,
    right_table_t,
    left_parallel_t,
    right_parallel_t,
    is_left_t,
    is_right_t,
    is_left_point_t,
    is_strict_contains_t,
    is_strict_left_t,
    point_col_id_t,
    interval_start_col_id_t,
    interval_end_col_id_t,
    key_in_output_t,
    need_typechange_t,
    _bodo_rebalance_output_if_skewed,
    num_rows_ptr_t,
):
    """
    Call cpp function for optimized interval join of two tables.
    Point in interval and interval overlap joins are supported.
    """
    assert left_table_t == table_type, "interval_join_table: cpp table type expected"
    assert right_table_t == table_type, "interval_join_table: cpp table type expected"

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="interval_join_table"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            left_table_t,
            right_table_t,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.uint64,
            types.uint64,
            types.uint64,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.voidptr,
        ),
        codegen,
    )


@intrinsic
def sort_values_table_py_entry(
    typingctx,
    table_t,
    n_keys_t,
    vect_ascending_t,
    na_position_b_t,
    dead_keys_t,
    n_rows_t,
    bounds_t,
    parallel_t,
):
    """
    Interface to the sorting of tables.
    """
    assert table_t == table_type, "C++ table type expected"

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="sort_values_table_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t,
            types.int64,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.boolean,
        ),
        codegen,
    )


@intrinsic
def sort_table_for_interval_join(
    typingctx,
    table_t,
    bounds_arr_t,
    is_table_point_side_t,
    parallel_t,
):
    """
    Interface to the sorting of a table for interval join.
    Bounds must be provided.
    """
    assert table_t == table_type, "C++ table type expected"
    assert bounds_arr_t == array_info_type, "C++ Array Info type expected"

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="sort_table_for_interval_join_py_entrypoint"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t,
            bounds_arr_t,
            types.boolean,
            types.boolean,
        ),
        codegen,
    )


@intrinsic
def sample_table(
    typingctx, table_t, n_keys_t, frac_t, replace_t, random_state_t, parallel_t
):
    """
    Interface to the sampling of tables.
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.DoubleType(),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="sample_table_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t,
            types.int64,
            types.float64,
            types.boolean,
            types.int64,
            types.boolean,
        ),
        codegen,
    )


@intrinsic
def shuffle_renormalization(typingctx, table_t, random_t, random_seed_t, is_parallel_t):
    """
    Interface to the rebalancing of the table
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="shuffle_renormalization_py_entrypt"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(table_t, types.int32, types.int64, types.boolean),
        codegen,
    )


@intrinsic
def shuffle_renormalization_group(
    typingctx, table_t, random_t, random_seed_t, is_parallel_t, num_ranks_t, ranks_t
):
    """
    Interface to the rebalancing of the table
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="shuffle_renormalization_group_py_entrypt"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t, types.int32, types.int64, types.boolean, types.int64, types.voidptr
        ),
        codegen,
    )


@intrinsic
def drop_duplicates_cpp_table(
    typingctx, table_t, parallel_t, nkey_t, keep_t, dropna, drop_local_first
):
    """
    Interface to dropping duplicate entry in tables
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="drop_duplicates_table_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t,
            types.boolean,
            types.int64,
            types.int64,
            types.boolean,
            types.boolean,
        ),
        codegen,
    )


@intrinsic
def groupby_and_aggregate(
    typingctx,
    table_t,
    n_keys_t,
    cols_per_func_t,
    nwindows_calls_per_func_t,
    n_funcs_t,
    input_has_index,
    ftypes,
    func_offsets,
    udf_n_redvars,
    is_parallel,
    skip_na_data_t,
    shift_periods_t,
    transform_func,
    head_n,
    return_keys,
    return_index,
    dropna,
    update_cb,
    combine_cb,
    eval_cb,
    general_udfs_cb,
    udf_table_dummy_t,
    n_out_rows_t,
    window_ascending_t,
    window_na_position_t,
    window_args_t,
    n_window_args_per_func_t,
    n_input_cols_per_func_t,
    maintain_input_size_t,
    n_shuffle_keys_t,
    use_sql_rules_t,
):
    """
    Interface to groupby_and_aggregate function in C++ library for groupby
    offloading.
    """
    assert table_t == table_type
    assert udf_table_dummy_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # table_info*
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),  # shift_periods_t
                lir.IntType(8).as_pointer(),  # transform_func
                lir.IntType(64),  # head_n
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),  # groupby key dropna
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),  # window_ascending
                lir.IntType(8).as_pointer(),  # window_na_position
                lir.IntType(8).as_pointer(),  # window_args
                lir.IntType(8).as_pointer(),  # n_window_args_per_func
                lir.IntType(8).as_pointer(),  # n_input_cols_per_func
                lir.IntType(1),  # maintain_input_size
                lir.IntType(64),  # n_shuffle_keys_t
                lir.IntType(1),  # use_sql_rules_t
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="groupby_and_aggregate"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        table_type(
            table_t,
            types.int64,
            types.voidptr,
            types.voidptr,
            types.int64,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.boolean,
            types.int64,  # shift_periods
            types.voidptr,  # transform_func
            types.int64,  # head_n
            types.boolean,
            types.boolean,
            types.boolean,  # dropna
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            table_t,
            types.voidptr,
            types.voidptr,  # window_ascending
            types.voidptr,  # window_na_position
            window_args_t,  # window_args
            types.voidptr,  # n_window_args_per_func
            types.voidptr,  # n_input_cols_per_func
            types.boolean,  # maintain_input_size
            types.int64,  # n_shuffle_keys_t
            types.boolean,  # use_sql_rules_t
        ),
        codegen,
    )


_drop_duplicates_local_dictionary = types.ExternalFunction(
    "drop_duplicates_local_dictionary_py_entry",
    array_info_type(array_info_type, types.bool_),
)


@numba.njit(no_cpython_wrapper=True)
def drop_duplicates_local_dictionary(dict_arr, sort_dictionary):  # pragma: no cover
    dict_arr_info = array_to_info(dict_arr)
    # The _drop_duplicates_local_dictionary operation is done by modifying the
    # existing pointer, which is why we call info_to_array on the original array info
    out_dict_arr_info = _drop_duplicates_local_dictionary(
        dict_arr_info, sort_dictionary
    )
    check_and_propagate_cpp_exception()
    out_arr = info_to_array(out_dict_arr_info, bodo.dict_str_arr_type)
    delete_info(out_dict_arr_info)
    return out_arr


get_groupby_labels = types.ExternalFunction(
    "get_groupby_labels_py_entry",
    types.int64(table_type, types.voidptr, types.voidptr, types.boolean, types.bool_),
)


_array_isin = types.ExternalFunction(
    "array_isin_py_entry",
    types.void(array_info_type, array_info_type, array_info_type, types.bool_),
)


@numba.njit(no_cpython_wrapper=True)
def array_isin(out_arr, in_arr, in_values, is_parallel):  # pragma: no cover
    in_arr = decode_if_dict_array(in_arr)
    in_values = decode_if_dict_array(in_values)

    in_arr_info = array_to_info(in_arr)
    in_values_info = array_to_info(in_values)
    out_arr_info = array_to_info(out_arr)

    _array_isin(
        out_arr_info,
        in_arr_info,
        in_values_info,
        is_parallel,
    )
    check_and_propagate_cpp_exception()


_get_search_regex = types.ExternalFunction(
    "get_search_regex_py_entry",
    # params: in array, case-sensitive flag, pattern, output boolean array
    types.void(
        array_info_type,
        types.bool_,
        types.bool_,
        types.voidptr,
        array_info_type,
        types.bool_,
    ),
)

_get_replace_regex = types.ExternalFunction(
    "get_replace_regex_py_entry",
    # params: in array, pattern, replacement,
    # Output: out array
    array_info_type(array_info_type, types.voidptr, types.voidptr),
)


@intrinsic
def _get_replace_regex_dict_state(
    typingctx, arr_info_t, pattern_t, replace_t, dict_encoding_state_t, func_id_t
):
    assert arr_info_t == array_info_type
    assert isinstance(func_id_t, types.Integer), "func_id must be an integer"

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="get_replace_regex_dict_state_py_entry"
        )
        ret = builder.call(fn_tp, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    sig = array_info_type(
        arr_info_t, types.voidptr, types.voidptr, dict_encoding_state_t, types.int64
    )
    return sig, codegen


@numba.njit(no_cpython_wrapper=True)
def get_search_regex(
    in_arr, case, match, pat, out_arr, do_full_match=False
):  # pragma: no cover
    in_arr_info = array_to_info(in_arr)
    out_arr_info = array_to_info(out_arr)
    _get_search_regex(in_arr_info, case, match, pat, out_arr_info, do_full_match)
    check_and_propagate_cpp_exception()


@numba.njit(no_cpython_wrapper=True)
def get_replace_regex(in_arr, pattern_typ, replace_typ):  # pragma: no cover
    in_arr_info = array_to_info(in_arr)
    out_arr_info = _get_replace_regex(in_arr_info, pattern_typ, replace_typ)
    check_and_propagate_cpp_exception()
    out = info_to_array(out_arr_info, in_arr)
    delete_info(out_arr_info)
    return out


@numba.njit(no_cpython_wrapper=True)
def get_replace_regex_dict_state(
    in_arr, pattern_typ, replace_typ, dict_encoding_state, func_id
):  # pragma: no cover
    in_arr_info = array_to_info(in_arr)
    out_arr_info = _get_replace_regex_dict_state(
        in_arr_info, pattern_typ, replace_typ, dict_encoding_state, func_id
    )
    out = info_to_array(out_arr_info, in_arr)
    delete_info(out_arr_info)
    return out


def _gen_row_access_intrinsic(col_array_typ, c_ind):
    """Generate an intrinsic for loading a value from a table column with
    'col_array_typ' array type. 'c_ind' is the index of the column within the table.
    The intrinsic's input is an array of pointers for the table's data either array
    info or data depending on the type and a row index.

    For example, col_dtype=int64, c_ind=1, table=[A_data_ptr, B_data_ptr, C_data_ptr],
    row_ind=2 will return 6 for the table below.
    A  B  C
    1  4  7
    2  5  8
    3  6  9

    NOTE: This function may execute even if the data is NA, so the implementation must
    not segfault when accessing NA data.
    """
    from llvmlite import ir as lir

    col_dtype = col_array_typ.dtype

    if isinstance(
        col_dtype,
        (types.Number, TimeType, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype),
    ) or col_dtype in [
        bodo.datetime_date_type,
        bodo.datetime64ns,
        bodo.timedelta64ns,
        types.bool_,
    ]:
        # Note: PandasDatetimeTZDtype is not the return type for scalar data.
        # In C++ the data is just a datetime64ns
        if isinstance(col_dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype):
            col_dtype = bodo.datetime64ns

        # This code path just returns the data.
        @intrinsic
        def getitem_func(typingctx, table_t, ind_t):
            def codegen(context, builder, sig, args):
                table, row_ind = args
                # cast void* to void**
                table = builder.bitcast(table, lir.IntType(8).as_pointer().as_pointer())
                # get data pointer for input column and cast to proper data type
                col_ind = lir.Constant(lir.IntType(64), c_ind)
                col_ptr = builder.load(builder.gep(table, [col_ind]))

                if col_array_typ == boolean_array_type:
                    # Boolean arrays store 1 bit per value, so we need a custom path to load the bit.
                    col_ptr = builder.bitcast(
                        col_ptr, context.get_data_type(types.uint8).as_pointer()
                    )
                    data_val = bodo.utils.cg_helpers.get_bitmap_bit(
                        builder, col_ptr, row_ind
                    )
                    # Case the loaded bit to bool
                    return context.cast(
                        builder,
                        data_val,
                        types.uint8,
                        col_dtype,
                    )
                else:
                    col_ptr = builder.bitcast(
                        col_ptr, context.get_data_type(col_dtype).as_pointer()
                    )
                    data_val = builder.gep(col_ptr, [row_ind])
                    # Similar to Numpy array getitem in Numba:
                    # https://github.com/numba/numba/blob/2298ad6186d177f39c564046890263b0f1c74ecc/numba/np/arrayobj.py#L130
                    # makes sure we don't get LLVM i1 vs i8 mismatches for bool scalars
                    return context.unpack_value(builder, col_dtype, data_val)

            return col_dtype(types.voidptr, types.int64), codegen

        return getitem_func

    if col_array_typ in (bodo.string_array_type, bodo.binary_array_type):
        # If we have a unicode type we want to leave the raw
        # data pointer as a void* because we don't have a full
        # string yet.

        # This code path returns the data + length

        @intrinsic
        def getitem_func(typingctx, table_t, ind_t):
            def codegen(context, builder, sig, args):
                table, row_ind = args
                # cast void* to void**
                table = builder.bitcast(table, lir.IntType(8).as_pointer().as_pointer())
                # get data pointer for input column and cast to proper data type
                col_ind = lir.Constant(lir.IntType(64), c_ind)
                col_ptr = builder.load(builder.gep(table, [col_ind]))
                fnty = lir.FunctionType(
                    lir.IntType(8).as_pointer(),
                    [
                        lir.IntType(8).as_pointer(),
                        lir.IntType(64),
                        lir.IntType(64).as_pointer(),
                    ],
                )
                getitem_fn = cgutils.get_or_insert_function(
                    builder.module, fnty, name="array_info_getitem"
                )
                # Allocate for the output size
                size = cgutils.alloca_once(builder, lir.IntType(64))
                args = (col_ptr, row_ind, size)
                data_ptr = builder.call(getitem_fn, args)
                decode_sig = bodo.string_type(types.voidptr, types.int64)
                return context.compile_internal(
                    builder,
                    lambda data, length: bodo.libs.str_arr_ext.decode_utf8(
                        data, length
                    ),
                    decode_sig,
                    [data_ptr, builder.load(size)],
                )

            return (
                bodo.string_type(types.voidptr, types.int64),
                codegen,
            )

        return getitem_func

    if col_array_typ == bodo.libs.dict_arr_ext.dict_str_arr_type:
        # If we have a dictionary string type we want to extract the two
        # components and execute them differently. First we want to run to
        # extract the index in the dictionary in the intrinsic and get the
        # unicode data from C++.
        # This code path returns the data + length
        @intrinsic
        def getitem_func(typingctx, table_t, ind_t):
            def codegen(context, builder, sig, args):
                # Define some constants
                zero = lir.Constant(lir.IntType(64), 0)
                one = lir.Constant(lir.IntType(64), 1)

                table, row_ind = args
                # cast void* to void**
                table = builder.bitcast(table, lir.IntType(8).as_pointer().as_pointer())
                # get data pointer for the input column
                col_ind = lir.Constant(lir.IntType(64), c_ind)
                col_ptr = builder.load(builder.gep(table, [col_ind]))
                # Extract the index array from the dict array
                fnty = lir.FunctionType(
                    lir.IntType(8).as_pointer(),
                    [
                        lir.IntType(8).as_pointer(),
                        lir.IntType(64),
                    ],
                )
                get_info_func = cgutils.get_or_insert_function(
                    builder.module, fnty, name="get_child_info"
                )
                args = (col_ptr, one)
                indices_array_info = builder.call(get_info_func, args)
                # Extract the data from the array info
                fnty = lir.FunctionType(
                    lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer()]
                )
                get_data_func = cgutils.get_or_insert_function(
                    builder.module, fnty, name="array_info_getdata1"
                )
                args = (indices_array_info,)
                index_ptr = builder.call(get_data_func, args)
                index_ptr = builder.bitcast(
                    index_ptr,
                    context.get_data_type(col_array_typ.indices_dtype).as_pointer(),
                )
                dict_loc = builder.sext(
                    builder.load(builder.gep(index_ptr, [row_ind])), lir.IntType(64)
                )
                # NA gets checked after this function.
                # Extract the dictionary from the dict array
                args = (col_ptr, zero)
                dictionary_ptr = builder.call(get_info_func, args)
                fnty = lir.FunctionType(
                    lir.IntType(8).as_pointer(),
                    [
                        lir.IntType(8).as_pointer(),
                        lir.IntType(64),
                        lir.IntType(64).as_pointer(),
                    ],
                )
                getitem_fn = cgutils.get_or_insert_function(
                    builder.module, fnty, name="array_info_getitem"
                )
                # Allocate for the output size
                size = cgutils.alloca_once(builder, lir.IntType(64))
                args = (dictionary_ptr, dict_loc, size)
                data_ptr = builder.call(getitem_fn, args)
                decode_sig = bodo.string_type(types.voidptr, types.int64)
                return context.compile_internal(
                    builder,
                    lambda data, length: bodo.libs.str_arr_ext.decode_utf8(
                        data, length
                    ),
                    decode_sig,
                    [data_ptr, builder.load(size)],
                )

            return (
                bodo.string_type(types.voidptr, types.int64),
                codegen,
            )

        return getitem_func

    raise BodoError(
        f"General Join Conditions with '{col_array_typ}' column type and '{col_dtype}' data type not supported"
    )


def _gen_row_na_check_intrinsic(col_array_dtype, c_ind):
    """Generate an intrinsic for checking is a value is NA from a table column with
    array type 'col_array_dtype'. 'c_ind' is the index of the column within the table.
    The intrinsic's input is an array of data pointers or nullbit map depending on
    the type and a row index.

    For example, col_dtype=StringArray, c_ind=1, table=[A_bitmap, B_bitmap, C_bitmap],
    row_ind=2 will return True.
    A  B     C
    1  NA    7
    2  "qe"  8
    3  "ef"  9
    """
    if (
        isinstance(
            col_array_dtype, (IntegerArrayType, FloatingArrayType, bodo.TimeArrayType)
        )
        or col_array_dtype
        in (
            bodo.libs.bool_arr_ext.boolean_array_type,
            bodo.binary_array_type,
            bodo.datetime_date_array_type,
        )
        or is_str_arr_type(col_array_dtype)
    ):
        # These arrays use a null bitmap to store NA values.
        @intrinsic
        def checkna_func(typingctx, table_t, ind_t):
            def codegen(context, builder, sig, args):
                null_bitmaps, row_ind = args
                # cast void* to void**
                null_bitmaps = builder.bitcast(
                    null_bitmaps, lir.IntType(8).as_pointer().as_pointer()
                )
                # get null bitmap for input column and cast to proper data type
                col_ind = lir.Constant(lir.IntType(64), c_ind)
                col_ptr = builder.load(builder.gep(null_bitmaps, [col_ind]))
                null_bitmap = builder.bitcast(
                    col_ptr, context.get_data_type(types.bool_).as_pointer()
                )
                is_na = bodo.utils.cg_helpers.get_bitmap_bit(
                    builder, null_bitmap, row_ind
                )
                # IS NA is non-zero if null else 0.
                not_na = builder.icmp_unsigned(
                    "!=", is_na, lir.Constant(lir.IntType(8), 0)
                )
                # Since the & is bitwise we need the result to be either -1 or 0,
                # so we sign extend the result.
                return builder.sext(not_na, lir.IntType(8))

            # Return int8 because we don't want the actual bit
            return types.int8(types.voidptr, types.int64), codegen

        return checkna_func

    elif isinstance(col_array_dtype, (types.Array, bodo.DatetimeArrayType)):
        col_dtype = col_array_dtype.dtype
        if col_dtype in [
            bodo.datetime64ns,
            bodo.timedelta64ns,
        ] or isinstance(col_dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype):
            # Note: PandasDatetimeTZDtype is not the return type for scalar data.
            # In C++ the data is just a datetime64ns
            if isinstance(
                col_dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
            ):
                col_dtype = bodo.datetime64ns

            # Datetime arrays represent NULL by using pd._libs.iNaT
            @intrinsic
            def checkna_func(typingctx, table_t, ind_t):
                def codegen(context, builder, sig, args):
                    table, row_ind = args
                    # cast void* to void**
                    table = builder.bitcast(
                        table, lir.IntType(8).as_pointer().as_pointer()
                    )
                    # get data pointer for input column and cast to proper data type
                    col_ind = lir.Constant(lir.IntType(64), c_ind)
                    col_ptr = builder.load(builder.gep(table, [col_ind]))
                    col_ptr = builder.bitcast(
                        col_ptr, context.get_data_type(col_dtype).as_pointer()
                    )
                    value = builder.load(builder.gep(col_ptr, [row_ind]))
                    not_na = builder.icmp_unsigned(
                        "!=", value, lir.Constant(lir.IntType(64), pd._libs.iNaT)
                    )
                    # Since the & is bitwise we need the result to be either -1 or 0,
                    # so we sign extend the result.
                    return builder.sext(not_na, lir.IntType(8))

                # Return int8 because we don't want the actual bit
                return types.int8(types.voidptr, types.int64), codegen

            return checkna_func

        elif isinstance(col_dtype, types.Float):
            # If we have float NA values are stored as nan.
            # In this situation we need to check isnan. If we assume IEEE-754 Floating Point,
            # this check is simply checking certain bits
            @intrinsic
            def checkna_func(typingctx, table_t, ind_t):
                def codegen(context, builder, sig, args):
                    table, row_ind = args
                    # cast void* to void**
                    table = builder.bitcast(
                        table, lir.IntType(8).as_pointer().as_pointer()
                    )
                    # get data pointer for input column and cast to proper data type
                    col_ind = lir.Constant(lir.IntType(64), c_ind)
                    col_ptr = builder.load(builder.gep(table, [col_ind]))
                    col_ptr = builder.bitcast(
                        col_ptr, context.get_data_type(col_dtype).as_pointer()
                    )

                    # Get the float value
                    value = builder.load(builder.gep(col_ptr, [row_ind]))

                    isnan_sig = signature(
                        types.bool_,
                        col_dtype,
                    )

                    # This function is a lowering function so we can call it directly
                    is_na = numba.np.npyfuncs.np_real_isnan_impl(
                        context, builder, isnan_sig, (value,)
                    )

                    # Since the & is bitwise we need the result to be either -1 or 0, so
                    # sign extend and flip all of the bits
                    return builder.not_(builder.sext(is_na, lir.IntType(8)))

                # Return int8 because we don't want the actual bit
                return types.int8(types.voidptr, types.int64), codegen

            return checkna_func

    raise BodoError(
        f"General Join Conditions with '{col_array_dtype}' column type not supported"
    )


@intrinsic
def _retrieve_table(typingctx, cpp_table, row_bitmask):
    """This function takes in a cpp table and a bitmask over it's rows
    and returns a new table after applying the bitmask. If the bitmask is
    all True copying is skipped.
    """

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),  # output is a table
            [
                lir.IntType(8).as_pointer(),  # in_table
                lir.IntType(8).as_pointer(),  # row_bitmask
            ],
        )
        fn_tp = cgutils.get_or_insert_function(
            builder.module, fnty, name="retrieve_table_py_entry"
        )
        func_args = [args[0], args[1]]
        table_ret = builder.call(fn_tp, func_args)  # change this to bitmask
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return table_ret

    sig = cpp_table(
        cpp_table,
        array_info_type,
    )

    return sig, codegen


# ----------- Export Array and Table Info Unpin for Testing -----------------

_array_info_unpin = types.ExternalFunction(
    "array_info_unpin", types.void(array_info_type)
)
