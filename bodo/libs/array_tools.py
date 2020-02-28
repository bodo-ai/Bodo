# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Tools for handling bodo arrays, e.g. passing to C/C++ code
"""
import numpy as np
import numba
from numba import types, cgutils
from numba.extending import (
    typeof_impl,
    type_callable,
    models,
    register_model,
    NativeValue,
    make_attribute_wrapper,
    lower_builtin,
    box,
    unbox,
    lower_getattr,
    intrinsic,
    overload_method,
    overload,
    overload_attribute,
)
import bodo
from bodo.libs.str_arr_ext import string_array_type
from bodo.utils.utils import numba_to_c_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType, int128_type
from bodo.hiframes.pd_categorical_ext import CategoricalArray, get_categories_int_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.datetime_date_ext import datetime_date_array_type

from bodo.libs import array_tools_ext
from llvmlite import ir as lir
import llvmlite.binding as ll

ll.add_symbol("string_array_to_info", array_tools_ext.string_array_to_info)
ll.add_symbol("numpy_array_to_info", array_tools_ext.numpy_array_to_info)
ll.add_symbol("nullable_array_to_info", array_tools_ext.nullable_array_to_info)
ll.add_symbol("decimal_array_to_info", array_tools_ext.decimal_array_to_info)
ll.add_symbol("info_to_string_array", array_tools_ext.info_to_string_array)
ll.add_symbol("info_to_numpy_array", array_tools_ext.info_to_numpy_array)
ll.add_symbol("info_to_nullable_array", array_tools_ext.info_to_nullable_array)
ll.add_symbol("alloc_numpy", array_tools_ext.alloc_numpy)
ll.add_symbol("alloc_string_array", array_tools_ext.alloc_string_array)
ll.add_symbol("arr_info_list_to_table", array_tools_ext.arr_info_list_to_table)
ll.add_symbol("info_from_table", array_tools_ext.info_from_table)
ll.add_symbol("delete_table", array_tools_ext.delete_table)
ll.add_symbol("shuffle_table", array_tools_ext.shuffle_table)
ll.add_symbol("hash_join_table", array_tools_ext.hash_join_table)
ll.add_symbol("drop_duplicates_table", array_tools_ext.drop_duplicates_table)
ll.add_symbol("sort_values_table", array_tools_ext.sort_values_table)
ll.add_symbol("groupby_and_aggregate", array_tools_ext.groupby_and_aggregate)
ll.add_symbol("array_isin", array_tools_ext.array_isin)
ll.add_symbol(
    "compute_node_partition_by_hash", array_tools_ext.compute_node_partition_by_hash
)


class ArrayInfoType(types.Type):
    def __init__(self):
        super(ArrayInfoType, self).__init__(name="ArrayInfoType()")


array_info_type = ArrayInfoType()
register_model(ArrayInfoType)(models.OpaqueModel)


class TableType(types.Type):
    def __init__(self):
        super(TableType, self).__init__(name="TableType()")


table_type = TableType()
register_model(TableType)(models.OpaqueModel)


@intrinsic
def array_to_info(typingctx, arr_type_t):
    def codegen(context, builder, sig, args):
        (in_arr,) = args
        arr_type = arr_type_t
        # arr_info struct keeps a reference
        if context.enable_nrt:
            context.nrt.incref(builder, arr_type, in_arr)

        # StringArray
        if arr_type == string_array_type:
            string_array = context.make_helper(builder, string_array_type, in_arr)
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="string_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    string_array.num_items,
                    string_array.num_total_chars,
                    string_array.data,
                    string_array.offsets,
                    string_array.null_bitmap,
                    string_array.meminfo,
                ],
            )

        # get codes array from CategoricalArray to be handled similar to other Numpy
        # arrays.
        # TODO: create CategoricalArray on C++ side to handle NAs (-1) properly
        if isinstance(arr_type, CategoricalArray):
            in_arr = cgutils.create_struct_proxy(arr_type)(
                context, builder, in_arr
            ).codes
            int_dtype = get_categories_int_type(arr_type.dtype)
            arr_type = types.Array(int_dtype, 1, "C")

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

            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="numpy_array_to_info"
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

        # nullable integer/bool array
        if (
            isinstance(arr_type, (IntegerArrayType, DecimalArrayType))
            or arr_type in (boolean_array, datetime_date_array_type)
        ):
            arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
            dtype = arr_type.dtype
            np_dtype = dtype
            if isinstance(arr_type, DecimalArrayType):
                np_dtype = int128_type
            if arr_type == datetime_date_array_type:
                np_dtype = types.int64
            data_arr = context.make_array(types.Array(np_dtype, 1, "C"))(
                context, builder, arr.data
            )
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
                fn_tp = builder.module.get_or_insert_function(
                    fnty, name="decimal_array_to_info"
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
                fn_tp = builder.module.get_or_insert_function(
                    fnty, name="nullable_array_to_info"
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

    return array_info_type(arr_type_t), codegen


def _lower_info_to_array_numpy(arr_type, context, builder, in_info):
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
    fn_tp = builder.module.get_or_insert_function(fnty, name="info_to_numpy_array")
    builder.call(fn_tp, [in_info, length_ptr, data_ptr, meminfo_ptr])

    intp_t = context.get_value_type(types.intp)
    shape_array = cgutils.pack_array(builder, [builder.load(length_ptr)], ty=intp_t)
    itemsize = context.get_constant(
        types.intp, context.get_abi_sizeof(context.get_data_type(arr_type.dtype)),
    )
    strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

    data = builder.bitcast(
        builder.load(data_ptr), context.get_data_type(arr_type.dtype).as_pointer(),
    )

    numba.targets.arrayobj.populate_array(
        arr,
        data=data,
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=builder.load(meminfo_ptr),
    )
    return arr._getvalue()


@intrinsic
def info_to_array(typingctx, info_type, arr_type):
    assert info_type == array_info_type

    def codegen(context, builder, sig, args):
        in_info, _ = args
        # TODO: update meminfo?

        # StringArray
        if arr_type == string_array_type:
            string_array = context.make_helper(builder, string_array_type)
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),  # info
                    lir.IntType(64).as_pointer(),  # num_items
                    lir.IntType(64).as_pointer(),  # num_total_chars
                    lir.IntType(8).as_pointer().as_pointer(),  # data
                    lir.IntType(32).as_pointer().as_pointer(),  # offsets
                    lir.IntType(8).as_pointer().as_pointer(),  # null_bitmap
                    lir.IntType(8).as_pointer().as_pointer(),
                ],
            )  # meminfo
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_string_array"
            )
            builder.call(
                fn_tp,
                [
                    in_info,
                    string_array._get_ptr_by_name("num_items"),
                    string_array._get_ptr_by_name("num_total_chars"),
                    string_array._get_ptr_by_name("data"),
                    string_array._get_ptr_by_name("offsets"),
                    string_array._get_ptr_by_name("null_bitmap"),
                    string_array._get_ptr_by_name("meminfo"),
                ],
            )
            return string_array._getvalue()

        if isinstance(arr_type, CategoricalArray):
            out_arr = cgutils.create_struct_proxy(arr_type)(context, builder)
            int_dtype = get_categories_int_type(arr_type.dtype)
            int_arr_type = types.Array(int_dtype, 1, "C")
            out_arr.codes = _lower_info_to_array_numpy(
                int_arr_type, context, builder, in_info
            )
            return out_arr._getvalue()

        # Numpy
        if isinstance(arr_type, types.Array):
            return _lower_info_to_array_numpy(arr_type, context, builder, in_info)

        # nullable integer/bool array
        if (
            isinstance(arr_type, (IntegerArrayType, DecimalArrayType))
            or arr_type in (boolean_array, datetime_date_array_type)
        ):
            arr = cgutils.create_struct_proxy(arr_type)(context, builder)
            np_dtype = arr_type.dtype
            if isinstance(arr_type, DecimalArrayType):
                np_dtype = int128_type
            if arr_type == datetime_date_array_type:
                np_dtype = types.int64
            data_arr_type = types.Array(np_dtype, 1, "C")
            data_arr = context.make_array(data_arr_type)(context, builder)
            nulls_arr_type = types.Array(types.uint8, 1, "C")
            nulls_arr = context.make_array(nulls_arr_type)(context, builder)

            length_ptr = cgutils.alloca_once(builder, lir.IntType(64))
            n_bytes_ptr = cgutils.alloca_once(builder, lir.IntType(64))
            data_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
            nulls_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
            meminfo_ptr = cgutils.alloca_once(builder, lir.IntType(8).as_pointer())
            meminfo_nulls_ptr = cgutils.alloca_once(
                builder, lir.IntType(8).as_pointer()
            )

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
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_nullable_array"
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

            intp_t = context.get_value_type(types.intp)

            # data array
            shape_array = cgutils.pack_array(
                builder, [builder.load(length_ptr)], ty=intp_t
            )
            itemsize = context.get_constant(
                types.intp, context.get_abi_sizeof(context.get_data_type(np_dtype)),
            )
            strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

            data = builder.bitcast(
                builder.load(data_ptr), context.get_data_type(np_dtype).as_pointer(),
            )

            numba.targets.arrayobj.populate_array(
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

            numba.targets.arrayobj.populate_array(
                nulls_arr,
                data=data,
                shape=shape_array,
                strides=strides_array,
                itemsize=itemsize,
                meminfo=builder.load(meminfo_nulls_ptr),
            )
            arr.null_bitmap = nulls_arr._getvalue()
            return arr._getvalue()

    return arr_type(info_type, arr_type), codegen


@intrinsic
def test_alloc_np(typingctx, len_typ, arr_type):
    def codegen(context, builder, sig, args):
        length, _ = args
        typ_enum = numba_to_c_type(arr_type.dtype)
        typ_arg = cgutils.alloca_once_value(
            builder, lir.Constant(lir.IntType(32), typ_enum)
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(), [lir.IntType(64), lir.IntType(32)]  # num_items
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="alloc_numpy")
        return builder.call(fn_tp, [length, builder.load(typ_arg)])

    return array_info_type(len_typ, arr_type), codegen


@intrinsic
def test_alloc_string(typingctx, len_typ, n_chars_typ):
    def codegen(context, builder, sig, args):
        length, n_chars = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(), [lir.IntType(64), lir.IntType(64)]  # num_items
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="alloc_string_array")
        return builder.call(fn_tp, [length, n_chars])

    return array_info_type(len_typ, n_chars_typ), codegen


@intrinsic
def arr_info_list_to_table(typingctx, list_arr_info_typ):
    assert list_arr_info_typ == types.List(array_info_type)

    def codegen(context, builder, sig, args):
        (info_list,) = args
        inst = numba.targets.listobj.ListInstance(
            context, builder, sig.args[0], info_list
        )
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer().as_pointer(), lir.IntType(64)],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="arr_info_list_to_table"
        )
        return builder.call(fn_tp, [inst.data, inst.size])

    return table_type(list_arr_info_typ), codegen


@intrinsic
def info_from_table(typingctx, table_t, ind_t):
    # disabled assert because there are cfuncs that need to call info_from_table
    # on void ptrs received from C++
    # assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(), lir.IntType(64)]
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="info_from_table")
        return builder.call(fn_tp, args)

    return array_info_type(table_t, ind_t), codegen


@intrinsic
def delete_table(typingctx, table_t):
    """Deletes table and its array_info objects. Doesn't delete array data.
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_tp = builder.module.get_or_insert_function(fnty, name="delete_table")
        builder.call(fn_tp, args)

    return types.void(table_t), codegen


@intrinsic
def shuffle_table(typingctx, table_t, n_keys_t):
    """
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(), lir.IntType(64)]
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="shuffle_table")
        return builder.call(fn_tp, args)

    return table_type(table_t, types.int64), codegen


@intrinsic
def hash_join_table(
    typingctx,
    table_t,
    n_keys_t,
    n_data_left_t,
    n_data_right_t,
    same_vect_t,
    is_left_t,
    is_right_t,
    optional_col_t,
):
    """
    Interface to the hash join of two tables.
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="hash_join_table")
        return builder.call(fn_tp, args)

    return (
        table_type(
            table_t,
            types.int64,
            types.int64,
            types.int64,
            types.voidptr,
            types.boolean,
            types.boolean,
            types.boolean,
        ),
        codegen,
    )


@intrinsic
def compute_node_partition_by_hash(typingctx, table_t, n_keys_t, n_pes_t):
    """
    Interface to the computation of the hash node partition from C++
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(64)],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="compute_node_partition_by_hash"
        )
        return builder.call(fn_tp, args)

    return table_type(table_t, types.int64, types.int64), codegen


@intrinsic
def sort_values_table(typingctx, table_t, n_keys_t, vect_ascending_t, na_position_b_t):
    """
    Interface to the sorting of tables.
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="sort_values_table")
        return builder.call(fn_tp, args)

    return table_type(table_t, types.int64, types.voidptr, types.boolean), codegen


@intrinsic
def drop_duplicates_table(typingctx, table_t, parallel_t, nkey_t, keep_t):
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
            ],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="drop_duplicates_table"
        )
        return builder.call(fn_tp, args)

    return table_type(table_t, types.boolean, types.int64, types.int64), codegen


@intrinsic
def groupby_and_aggregate(
    typingctx,
    table_t,
    n_keys_t,
    ftypes,
    func_offsets,
    udf_n_redvars,
    is_parallel,
    skipdropna_t,
    return_keys,
    return_index,
    update_cb,
    combine_cb,
    eval_cb,
    udf_table_dummy_t,
):
    """
    Interface to groupby_and_aggregate function in C++ library for groupby
    offloading.
    """
    assert table_t == table_type
    assert udf_table_dummy_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="groupby_and_aggregate"
        )
        return builder.call(fn_tp, args)

    return (
        table_type(
            table_t,
            types.int64,
            types.intp,
            types.intp,
            types.intp,
            types.boolean,
            types.boolean,
            types.boolean,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            table_t,
        ),
        codegen,
    )


_array_isin = types.ExternalFunction(
    "array_isin",
    types.void(array_info_type, array_info_type, array_info_type, types.bool_),
)


@numba.njit
def array_isin(out_arr, in_arr, in_values, is_parallel):
    _array_isin(
        array_to_info(out_arr),
        array_to_info(in_arr),
        array_to_info(in_values),
        is_parallel,
    )
