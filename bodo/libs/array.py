# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Tools for handling bodo arrays, e.g. passing to C/C++ code
"""
import numba
from numba.core import types, cgutils
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
from bodo.libs.str_arr_ext import string_array_type, _get_string_arr_payload
from bodo.libs.list_str_arr_ext import list_string_array_type
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayType,
    _get_array_item_arr_payload,
    offset_typ,
    ArrayItemArrayPayloadType,
    define_array_item_dtor,
)
from bodo.utils.utils import numba_to_c_type, CTypeEnum
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType, int128_type
from bodo.hiframes.pd_categorical_ext import CategoricalArray, get_categories_int_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.utils.transform import get_type_alloc_counts

from bodo.libs import array_ext
from llvmlite import ir as lir
import llvmlite.binding as ll

ll.add_symbol("list_string_array_to_info", array_ext.list_string_array_to_info)
ll.add_symbol("nested_array_to_info", array_ext.nested_array_to_info)
ll.add_symbol("string_array_to_info", array_ext.string_array_to_info)
ll.add_symbol("numpy_array_to_info", array_ext.numpy_array_to_info)
ll.add_symbol("nullable_array_to_info", array_ext.nullable_array_to_info)
ll.add_symbol("decimal_array_to_info", array_ext.decimal_array_to_info)
ll.add_symbol("info_to_nested_array", array_ext.info_to_nested_array)
ll.add_symbol("info_to_string_array", array_ext.info_to_string_array)
ll.add_symbol("info_to_numpy_array", array_ext.info_to_numpy_array)
ll.add_symbol("info_to_nullable_array", array_ext.info_to_nullable_array)
ll.add_symbol("info_to_list_string_array", array_ext.info_to_list_string_array)
ll.add_symbol("alloc_numpy", array_ext.alloc_numpy)
ll.add_symbol("alloc_string_array", array_ext.alloc_string_array)
ll.add_symbol("arr_info_list_to_table", array_ext.arr_info_list_to_table)
ll.add_symbol("info_from_table", array_ext.info_from_table)
ll.add_symbol("delete_table", array_ext.delete_table)
ll.add_symbol("shuffle_table", array_ext.shuffle_table)
ll.add_symbol("hash_join_table", array_ext.hash_join_table)
ll.add_symbol("drop_duplicates_table", array_ext.drop_duplicates_table)
ll.add_symbol("sort_values_table", array_ext.sort_values_table)
ll.add_symbol("groupby_and_aggregate", array_ext.groupby_and_aggregate)
ll.add_symbol("array_isin", array_ext.array_isin)
ll.add_symbol(
    "compute_node_partition_by_hash", array_ext.compute_node_partition_by_hash
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
        context.nrt.incref(builder, arr_type, in_arr)

        # ListStringArray
        if arr_type == list_string_array_type:
            list_string_array = context.make_helper(
                builder, list_string_array_type, in_arr
            )
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(64),
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32).as_pointer(),
                    lir.IntType(32).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="list_string_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    list_string_array.num_items,
                    list_string_array.num_total_strings,
                    list_string_array.num_total_chars,
                    list_string_array.data,
                    list_string_array.data_offsets,
                    list_string_array.index_offsets,
                    list_string_array.null_bitmap,
                    list_string_array.meminfo,
                ],
            )

        # nested arrays path. TODO: add StructArrayType
        if isinstance(arr_type, ArrayItemArrayType):

            def get_types(arr_typ):
                """ Get list of all types (in Bodo_CTypes enum format) in the
                    nested structure rooted at arr_typ """
                if isinstance(
                    arr_typ, (types.Array, ArrayItemArrayType, IntegerArrayType)
                ):  # elements in the nested structure that are arrays are a List in Arrow
                    return [CTypeEnum.LIST.value] + get_types(arr_typ.dtype)
                # TODO: add Struct, Categorical, String
                else:
                    return [numba_to_c_type(arr_typ)]

            def get_lengths(arr_typ, arr):
                """ Get array of lengths of all arrays in nested structure """
                if isinstance(arr_typ, ArrayItemArrayType):
                    payload = _get_array_item_arr_payload(
                        context, builder, arr_typ, arr
                    )
                    lengths = get_lengths(arr_typ.dtype, payload.data)
                    lengths = cgutils.pack_array(
                        builder,
                        [payload.n_arrays]
                        + [
                            builder.extract_value(lengths, i)
                            for i in range(lengths.type.count)
                        ],
                    )
                # TODO: add Struct, Categorical, String
                elif isinstance(
                    arr_typ, (IntegerArrayType, DecimalArrayType)
                ) or arr_typ in (boolean_array, datetime_date_array_type,):
                    len_sig = numba.core.typing.signature(types.intp, arr_typ)
                    length = context.compile_internal(
                        builder, lambda a: len(a), len_sig, [arr]
                    )
                    lengths = cgutils.pack_array(builder, [length])
                else:
                    raise RuntimeError("array_to_info: unsupported type for subarray")
                return lengths

            def get_buffers(arr_typ, arr):
                """ Get array of buffers (offsets, nulls, data) of all arrays in nested structure """
                if isinstance(arr_typ, ArrayItemArrayType):
                    payload = _get_array_item_arr_payload(
                        context, builder, arr_typ, arr
                    )
                    buffs_data = get_buffers(arr_typ.dtype, payload.data)
                    offsets_arr = context.make_array(types.Array(offset_typ, 1, "C"))(
                        context, builder, payload.offsets
                    )
                    offsets_ptr = builder.bitcast(
                        offsets_arr.data, lir.IntType(8).as_pointer()
                    )
                    null_bitmap_arr = context.make_array(
                        types.Array(types.uint8, 1, "C")
                    )(context, builder, payload.null_bitmap)
                    null_bitmap_ptr = builder.bitcast(
                        null_bitmap_arr.data, lir.IntType(8).as_pointer()
                    )
                    buffers = cgutils.pack_array(
                        builder,
                        [offsets_ptr, null_bitmap_ptr]
                        + [
                            builder.extract_value(buffs_data, i)
                            for i in range(buffs_data.type.count)
                        ],
                    )
                elif isinstance(
                    arr_typ, (IntegerArrayType, DecimalArrayType)
                ) or arr_typ in (boolean_array, datetime_date_array_type,):
                    np_dtype = arr_typ.dtype
                    if isinstance(arr_typ, DecimalArrayType):
                        np_dtype = int128_type
                    elif arr_typ == datetime_date_array_type:
                        np_dtype = types.int64
                    arr = cgutils.create_struct_proxy(arr_typ)(context, builder, arr)
                    data_arr = context.make_array(types.Array(np_dtype, 1, "C"))(
                        context, builder, arr.data
                    )
                    null_bitmap_arr = context.make_array(
                        types.Array(types.uint8, 1, "C")
                    )(context, builder, arr.null_bitmap)
                    data_ptr = builder.bitcast(
                        data_arr.data, lir.IntType(8).as_pointer()
                    )
                    null_bitmap_ptr = builder.bitcast(
                        null_bitmap_arr.data, lir.IntType(8).as_pointer()
                    )
                    buffers = cgutils.pack_array(builder, [null_bitmap_ptr, data_ptr])
                else:
                    raise RuntimeError("array_to_info: unsupported type for subarray")
                return buffers

            # get list of all types in the nested datastructure (to pass to C++)
            types_list = get_types(arr_type.dtype)
            types_array = cgutils.pack_array(
                builder, [context.get_constant(types.int32, t) for t in types_list]
            )
            types_array_ptr = cgutils.alloca_once_value(builder, types_array)

            # get lengths of all arrays in the nested datastructure (to pass to C++)
            lengths = get_lengths(arr_type, in_arr)
            lengths_ptr = cgutils.alloca_once_value(builder, lengths)

            # get pointers to every individual buffer in the nested datastructure
            # (to pass to C++): offsets, nulls, data
            buffers = get_buffers(arr_type, in_arr)
            buffers_ptr = cgutils.alloca_once_value(builder, buffers)

            array_item_array = context.make_helper(builder, arr_type, in_arr)

            # pass all the data to C++ so that an array_info using nested Arrow
            # Array is constructed
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(32).as_pointer(),
                    lir.IntType(8).as_pointer().as_pointer(),
                    lir.IntType(64).as_pointer(),
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="nested_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    builder.bitcast(types_array_ptr, lir.IntType(32).as_pointer()),
                    builder.bitcast(
                        buffers_ptr, lir.IntType(8).as_pointer().as_pointer()
                    ),
                    builder.bitcast(lengths_ptr, lir.IntType(64).as_pointer()),
                    array_item_array.meminfo,
                ],
            )

        # StringArray
        if arr_type == string_array_type:
            string_array = context.make_helper(builder, string_array_type, in_arr)
            payload = _get_string_arr_payload(context, builder, in_arr)
            num_total_chars = builder.zext(
                builder.load(builder.gep(payload.offsets, [payload.num_strings])),
                lir.IntType(64),
            )
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
                    payload.num_strings,
                    num_total_chars,
                    payload.data,
                    payload.offsets,
                    payload.null_bitmap,
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
        if isinstance(arr_type, (IntegerArrayType, DecimalArrayType)) or arr_type in (
            boolean_array,
            datetime_date_array_type,
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

    numba.np.arrayobj.populate_array(
        arr,
        data=data,
        shape=shape_array,
        strides=strides_array,
        itemsize=itemsize,
        meminfo=builder.load(meminfo_ptr),
    )
    return arr._getvalue()


def nested_to_array(
    context, builder, arr_typ, lengths_ptr, array_infos_ptr, lengths_pos, infos_pos
):
    """ LLVM codegen for info_to_array for nested types. Is called recursively """

    ll_array_info_type = context.get_data_type(array_info_type)

    if isinstance(arr_typ, ArrayItemArrayType):
        # construct ArrayItemArray given the array of lengths and array_infos received from C++

        # call codegen for child array
        sub_arr = nested_to_array(
            context,
            builder,
            arr_typ.dtype,
            lengths_ptr,
            array_infos_ptr,
            lengths_pos + 1,
            infos_pos + 2,
        )

        payload_type = ArrayItemArrayPayloadType(arr_typ)
        payload_alloc_type = context.get_data_type(payload_type)
        payload_alloc_size = context.get_abi_sizeof(payload_alloc_type)

        # define dtor
        dtor_fn = define_array_item_dtor(context, builder, arr_typ, payload_type)

        # create meminfo
        meminfo = context.nrt.meminfo_alloc_dtor(
            builder, context.get_constant(types.uintp, payload_alloc_size), dtor_fn
        )
        meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
        meminfo_data_ptr = builder.bitcast(
            meminfo_void_ptr, payload_alloc_type.as_pointer()
        )

        # convert my array_infos to numpy arrays and set payload attributes
        payload = cgutils.create_struct_proxy(payload_type)(context, builder)
        payload.n_arrays = builder.extract_value(builder.load(lengths_ptr), lengths_pos)
        payload.data = (
            sub_arr  # data is the child array which was constructed before this
        )

        infos = builder.load(array_infos_ptr)

        # offsets array from array_info
        offsets_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos), ll_array_info_type
        )
        payload.offsets = _lower_info_to_array_numpy(
            types.Array(types.uint32, 1, "C"), context, builder, offsets_info_ptr,
        )

        # nulls array from array_info
        nulls_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos + 1), ll_array_info_type
        )
        payload.null_bitmap = _lower_info_to_array_numpy(
            types.Array(types.uint8, 1, "C"), context, builder, nulls_info_ptr
        )

        builder.store(payload._getvalue(), meminfo_data_ptr)
        array_item_array = context.make_helper(builder, arr_typ)
        array_item_array.meminfo = meminfo
        return array_item_array._getvalue()
    # TODO StructArrayType
    # TODO Categorical

    # StringArray
    elif arr_typ == string_array_type:
        infos = builder.load(array_infos_ptr)
        info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos), ll_array_info_type
        )

        string_array = context.make_helper(builder, string_array_type)
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # info
                lir.IntType(8).as_pointer().as_pointer(),  # meminfo
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="info_to_string_array")
        builder.call(
            fn_tp, [info_ptr, string_array._get_ptr_by_name("meminfo"),],
        )
        return string_array._getvalue()

    # Numpy
    elif isinstance(arr_typ, types.Array):
        # since the array comes from Arrow, it has a null buffer which should
        # be all 1s. we ignore it because this type is not nullable

        # data array from array_info
        infos = builder.load(array_infos_ptr)
        data_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos + 1), ll_array_info_type
        )
        return _lower_info_to_array_numpy(arr_typ, context, builder, data_info_ptr,)

    elif isinstance(arr_typ, (IntegerArrayType, DecimalArrayType)) or arr_typ in (
        boolean_array,
        datetime_date_array_type,
    ):
        # construct "primitive" array given the array_infos received from C++
        arr = cgutils.create_struct_proxy(arr_typ)(context, builder)
        np_dtype = arr_typ.dtype
        if isinstance(arr_typ, DecimalArrayType):
            np_dtype = int128_type
        elif arr_typ == datetime_date_array_type:
            np_dtype = types.int64

        infos = builder.load(array_infos_ptr)

        # nulls array from array_info
        nulls_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos), ll_array_info_type
        )
        arr.null_bitmap = _lower_info_to_array_numpy(
            types.Array(types.uint8, 1, "C"), context, builder, nulls_info_ptr
        )

        # data array from array_info
        data_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos + 1), ll_array_info_type
        )
        arr.data = _lower_info_to_array_numpy(
            types.Array(np_dtype, 1, "C"), context, builder, data_info_ptr,
        )

        return arr._getvalue()


@intrinsic
def info_to_array(typingctx, info_type, arr_type):
    assert info_type == array_info_type

    def codegen(context, builder, sig, args):
        in_info, _ = args
        # TODO: update meminfo?

        # ListStringArray
        if arr_type == list_string_array_type:
            list_string_array = context.make_helper(builder, list_string_array_type)
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),  # info
                    lir.IntType(64).as_pointer(),  # num_items
                    lir.IntType(64).as_pointer(),  # num_strings
                    lir.IntType(64).as_pointer(),  # num_tot_chars
                    lir.IntType(8).as_pointer().as_pointer(),  # data
                    lir.IntType(32).as_pointer().as_pointer(),  # data_offsets
                    lir.IntType(32).as_pointer().as_pointer(),  # index_offsets
                    lir.IntType(8).as_pointer().as_pointer(),  # null_bitmap
                    lir.IntType(8).as_pointer().as_pointer(),
                ],
            )  # meminfo
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_list_string_array"
            )
            builder.call(
                fn_tp,
                [
                    in_info,
                    list_string_array._get_ptr_by_name("num_items"),
                    list_string_array._get_ptr_by_name("num_total_strings"),
                    list_string_array._get_ptr_by_name("num_total_chars"),
                    list_string_array._get_ptr_by_name("data"),
                    list_string_array._get_ptr_by_name("data_offsets"),
                    list_string_array._get_ptr_by_name("index_offsets"),
                    list_string_array._get_ptr_by_name("null_bitmap"),
                    list_string_array._get_ptr_by_name("meminfo"),
                ],
            )
            return list_string_array._getvalue()

        elif isinstance(arr_type, ArrayItemArrayType):
            # TODO add StructArrayType here

            def get_num_infos(arr_typ):
                """ get number of array_infos that need to be returned from
                    C++ to reconstruct this array """
                if isinstance(arr_typ, ArrayItemArrayType):
                    # 1 buffer for offsets, 1 buffer for nulls
                    return 2 + get_num_infos(arr_typ.dtype)
                # TODO StructArrayType, Numpy arrays and others
                elif arr_typ == string_array_type:
                    # C++ will just use one array_info
                    return 1
                else:
                    # for primitive types: nulls and data
                    # NOTE for non-nullable arrays C++ will still return two
                    # buffers since it doesn't know that the Arrow array of
                    # primitive values is going to be converted to a Numpy array
                    # (all Arrow arrays are nullable)
                    return 2

            n = get_type_alloc_counts(arr_type)
            # allocate zero-initialized array of lengths for each array in
            # nested datastructure (to be filled out by C++)
            lengths = cgutils.pack_array(
                builder, [lir.Constant(lir.IntType(64), 0) for _ in range(n)]
            )
            lengths_ptr = cgutils.alloca_once_value(builder, lengths)
            # allocate array of null pointers for each buffer in the
            # nested datastructure (to be filled out by C++ as pointers to array_info)
            nullptr = lir.Constant(lir.IntType(8).as_pointer(), None)
            array_infos = cgutils.pack_array(
                builder, [nullptr for _ in range(get_num_infos(arr_type))]
            )
            array_infos_ptr = cgutils.alloca_once_value(builder, array_infos)

            # call C++ info_to_nested_array to fill lengths and array_info arrays
            # each array_info corresponds to one individual buffer (can be
            # offsets, null or data buffer)
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),  # info
                    lir.IntType(64).as_pointer(),  # lengths array
                    lir.IntType(8).as_pointer().as_pointer(),  # array of array_info*
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_nested_array"
            )
            builder.call(
                fn_tp,
                [
                    in_info,
                    builder.bitcast(lengths_ptr, lir.IntType(64).as_pointer()),
                    builder.bitcast(
                        array_infos_ptr, lir.IntType(8).as_pointer().as_pointer()
                    ),
                ],
            )

            # generate code recursively to construct nested arrays from buffers
            # returned from C++
            return nested_to_array(
                context, builder, arr_type, lengths_ptr, array_infos_ptr, 0, 0
            )

        # StringArray
        if arr_type == string_array_type:
            string_array = context.make_helper(builder, string_array_type)
            fnty = lir.FunctionType(
                lir.VoidType(),
                [
                    lir.IntType(8).as_pointer(),  # info
                    lir.IntType(8).as_pointer().as_pointer(),  # meminfo
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_string_array"
            )
            builder.call(
                fn_tp, [in_info, string_array._get_ptr_by_name("meminfo"),],
            )
            return string_array._getvalue()

        if isinstance(arr_type, CategoricalArray):
            out_arr = cgutils.create_struct_proxy(arr_type)(context, builder)
            int_dtype = get_categories_int_type(arr_type.dtype)
            int_arr_type = types.Array(int_dtype, 1, "C")
            out_arr.codes = _lower_info_to_array_numpy(
                int_arr_type, context, builder, in_info
            )
            # set categorical dtype of output array to be same as input array
            dtype = cgutils.create_struct_proxy(arr_type)(
                context, builder, args[1]
            ).dtype
            out_arr.dtype = dtype
            context.nrt.incref(builder, arr_type.dtype, dtype)
            return out_arr._getvalue()

        # Numpy
        if isinstance(arr_type, types.Array):
            return _lower_info_to_array_numpy(arr_type, context, builder, in_info)

        # nullable integer/bool array
        if isinstance(arr_type, (IntegerArrayType, DecimalArrayType)) or arr_type in (
            boolean_array,
            datetime_date_array_type,
        ):
            arr = cgutils.create_struct_proxy(arr_type)(context, builder)
            np_dtype = arr_type.dtype
            if isinstance(arr_type, DecimalArrayType):
                np_dtype = int128_type
            elif arr_type == datetime_date_array_type:
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
        inst = numba.cpython.listobj.ListInstance(
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
    same_need_typechange_t,
    is_left_t,
    is_right_t,
    is_join_t,
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
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
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
            types.voidptr,
            types.boolean,
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
def sort_values_table(
    typingctx, table_t, n_keys_t, vect_ascending_t, na_position_b_t, parallel_t
):
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
                lir.IntType(1),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="sort_values_table")
        return builder.call(fn_tp, args)

    return (
        table_type(table_t, types.int64, types.voidptr, types.boolean, types.boolean),
        codegen,
    )


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
