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
from bodo.utils.utils import _numba_to_c_type_map
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array

from bodo.libs import array_tools_ext
from llvmlite import ir as lir
import llvmlite.binding as ll

ll.add_symbol("string_array_to_info", array_tools_ext.string_array_to_info)
ll.add_symbol("numpy_array_to_info", array_tools_ext.numpy_array_to_info)
ll.add_symbol("nullable_array_to_info", array_tools_ext.nullable_array_to_info)
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
ll.add_symbol(
    "drop_duplicates_table_outplace", array_tools_ext.drop_duplicates_table_outplace
)
ll.add_symbol("sort_values_table", array_tools_ext.sort_values_table)
ll.add_symbol("groupby_and_aggregate", array_tools_ext.groupby_and_aggregate)
ll.add_symbol("groupby_and_aggregate_nunique", array_tools_ext.groupby_and_aggregate_nunique)


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
def array_to_info(typingctx, arr_type):
    def codegen(context, builder, sig, args):
        in_arr, = args
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

        # Numpy
        if isinstance(arr_type, types.Array):
            arr = context.make_array(arr_type)(context, builder, in_arr)
            assert arr_type.ndim == 1, "only 1D array shuffle supported"
            length = builder.extract_value(arr.shape, 0)
            dtype = arr_type.dtype
            # handle Categorical type
            if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
                dtype = bodo.hiframes.pd_categorical_ext.get_categories_int_type(dtype)
            typ_enum = _numba_to_c_type_map[dtype]
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
        if isinstance(arr_type, IntegerArrayType) or arr_type == boolean_array:
            arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
            dtype = arr_type.dtype
            data_arr = context.make_array(types.Array(dtype, 1, "C"))(
                context, builder, arr.data
            )
            length = builder.extract_value(data_arr.shape, 0)
            bitmap_arr = context.make_array(types.Array(types.uint8, 1, "C"))(
                context, builder, arr.null_bitmap
            )

            typ_enum = _numba_to_c_type_map[dtype]
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

    return array_info_type(arr_type), codegen


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

        # Numpy
        if isinstance(arr_type, types.Array):
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
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_numpy_array"
            )
            builder.call(fn_tp, [in_info, length_ptr, data_ptr, meminfo_ptr])

            intp_t = context.get_value_type(types.intp)
            shape_array = cgutils.pack_array(
                builder, [builder.load(length_ptr)], ty=intp_t
            )
            itemsize = context.get_constant(
                types.intp,
                context.get_abi_sizeof(context.get_data_type(arr_type.dtype)),
            )
            strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

            data = builder.bitcast(
                builder.load(data_ptr),
                context.get_data_type(arr_type.dtype).as_pointer(),
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

        # nullable integer/bool array
        if isinstance(arr_type, IntegerArrayType) or arr_type == boolean_array:
            arr = cgutils.create_struct_proxy(arr_type)(context, builder)
            data_arr_type = types.Array(arr_type.dtype, 1, "C")
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
                types.intp,
                context.get_abi_sizeof(context.get_data_type(arr_type.dtype)),
            )
            strides_array = cgutils.pack_array(builder, [itemsize], ty=intp_t)

            data = builder.bitcast(
                builder.load(data_ptr),
                context.get_data_type(arr_type.dtype).as_pointer(),
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
        typ_enum = _numba_to_c_type_map[arr_type.dtype]
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
        info_list, = args
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
):
    """
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
        ),
        codegen,
    )


@intrinsic
def sort_values_table(typingctx, table_t, n_keys_t, vect_ascending_t, na_position_b_t):
    """
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.IntType(8).as_pointer(),
                                [lir.IntType(8).as_pointer(),
                                 lir.IntType(64),
                                 lir.IntType(8).as_pointer(),
                                 lir.IntType(1)])
        fn_tp = builder.module.get_or_insert_function(fnty, name="sort_values_table")
        return builder.call(fn_tp, args)

    return table_type(table_t, types.int64, types.voidptr, types.boolean), codegen


@intrinsic
def drop_duplicates_table_outplace(typingctx, table_t, subset_vect_t, keep_t):
    """
    Interface to dropping duplicate entry in tables
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(8).as_pointer(), lir.IntType(64)],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="drop_duplicates_table_outplace"
        )
        return builder.call(fn_tp, args)

    return table_type(table_t, types.voidptr, types.int64), codegen


@intrinsic
def groupby_and_aggregate(
    typingctx,
    table_t,
    n_keys_t,
    ftype,
    num_funcs,
    is_parallel,
    update_cb,
    combine_cb,
    eval_cb,
    out_table_dummy_t,
):
    """
    Interface to groupby_and_aggregate function in C++ library for groupby
    offloading.
    """
    assert table_t == table_type
    assert out_table_dummy_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(32),
                lir.IntType(32),
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
            types.int32,
            types.int32,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            table_t,
        ),
        codegen,
    )


@intrinsic
def groupby_and_aggregate_nunique(typingctx, table_t, n_keys_t, is_parallel):
    """
    Interface to groupby_and_aggregate_nunique function in C++ library for groupby
    offloading.
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(), [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(1)]
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="groupby_and_aggregate_nunique")
        return builder.call(fn_tp, args)

    return table_type(table_t, types.int64, types.boolean), codegen

