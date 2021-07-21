# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Tools for handling bodo arrays, e.g. passing to C/C++ code
"""
import llvmlite.binding as ll
import numba
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.extending import intrinsic, models, register_model

from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import (
    CategoricalArrayType,
    get_categories_int_type,
)
from bodo.libs import array_ext
from bodo.libs.array_item_arr_ext import (
    ArrayItemArrayPayloadType,
    ArrayItemArrayType,
    _get_array_item_arr_payload,
    define_array_item_dtor,
    offset_type,
)
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType, int128_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.interval_arr_ext import IntervalArrayType
from bodo.libs.str_arr_ext import (
    _get_string_arr_payload,
    char_arr_type,
    null_bitmap_arr_type,
    offset_arr_type,
    string_array_type,
)
from bodo.libs.struct_arr_ext import (
    StructArrayPayloadType,
    StructArrayType,
    StructType,
    _get_struct_arr_payload,
    define_struct_arr_dtor,
)
from bodo.libs.tuple_arr_ext import TupleArrayType
from bodo.utils.typing import BodoError
from bodo.utils.utils import (
    CTypeEnum,
    check_and_propagate_cpp_exception,
    numba_to_c_type,
)

ll.add_symbol("list_string_array_to_info", array_ext.list_string_array_to_info)
ll.add_symbol("nested_array_to_info", array_ext.nested_array_to_info)
ll.add_symbol("string_array_to_info", array_ext.string_array_to_info)
ll.add_symbol("numpy_array_to_info", array_ext.numpy_array_to_info)
ll.add_symbol("categorical_array_to_info", array_ext.categorical_array_to_info)
ll.add_symbol("nullable_array_to_info", array_ext.nullable_array_to_info)
ll.add_symbol("interval_array_to_info", array_ext.interval_array_to_info)
ll.add_symbol("decimal_array_to_info", array_ext.decimal_array_to_info)
ll.add_symbol("info_to_nested_array", array_ext.info_to_nested_array)
ll.add_symbol("info_to_list_string_array", array_ext.info_to_list_string_array)
ll.add_symbol("info_to_string_array", array_ext.info_to_string_array)
ll.add_symbol("info_to_numpy_array", array_ext.info_to_numpy_array)
ll.add_symbol("info_to_nullable_array", array_ext.info_to_nullable_array)
ll.add_symbol("info_to_interval_array", array_ext.info_to_interval_array)
ll.add_symbol("alloc_numpy", array_ext.alloc_numpy)
ll.add_symbol("alloc_string_array", array_ext.alloc_string_array)
ll.add_symbol("arr_info_list_to_table", array_ext.arr_info_list_to_table)
ll.add_symbol("info_from_table", array_ext.info_from_table)
ll.add_symbol("delete_info_decref_array", array_ext.delete_info_decref_array)
ll.add_symbol("delete_table_decref_arrays", array_ext.delete_table_decref_arrays)
ll.add_symbol("delete_table", array_ext.delete_table)
ll.add_symbol("shuffle_table", array_ext.shuffle_table)
ll.add_symbol("get_shuffle_info", array_ext.get_shuffle_info)
ll.add_symbol("delete_shuffle_info", array_ext.delete_shuffle_info)
ll.add_symbol("reverse_shuffle_table", array_ext.reverse_shuffle_table)
ll.add_symbol("hash_join_table", array_ext.hash_join_table)
ll.add_symbol("drop_duplicates_table", array_ext.drop_duplicates_table)
ll.add_symbol("sort_values_table", array_ext.sort_values_table)
ll.add_symbol("sample_table", array_ext.sample_table)
ll.add_symbol("shuffle_renormalization", array_ext.shuffle_renormalization)
ll.add_symbol("shuffle_renormalization_group", array_ext.shuffle_renormalization_group)
ll.add_symbol("groupby_and_aggregate", array_ext.groupby_and_aggregate)
ll.add_symbol("pivot_groupby_and_aggregate", array_ext.pivot_groupby_and_aggregate)
ll.add_symbol("get_groupby_labels", array_ext.get_groupby_labels)
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
def array_to_info(typingctx, arr_type_t=None):
    """convert array to array info wrapper to pass to C++"""

    def codegen(context, builder, sig, args):
        (in_arr,) = args
        arr_type = arr_type_t
        if isinstance(arr_type, TupleArrayType):
            # TupleArray uses same model as StructArray so we just use a
            # StructArrayType to generate LLVM
            tuple_array = context.make_helper(builder, arr_type, in_arr)
            in_arr = tuple_array.data
            arr_type = StructArrayType(arr_type.data, ("dummy",) * len(arr_type.data))
        # arr_info struct keeps a reference
        context.nrt.incref(builder, arr_type, in_arr)

        if (
            isinstance(arr_type, ArrayItemArrayType)
            and arr_type.dtype == string_array_type
        ):
            # map ArrayItemArrayType(StringArrayType()) to array_info of type LIST_STRING
            array_item_array = context.make_helper(builder, arr_type, in_arr)
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(8).as_pointer(),
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="list_string_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    array_item_array.meminfo,
                ],
            )

        if isinstance(arr_type, (ArrayItemArrayType, StructArrayType)):

            def get_types(arr_typ):
                """Get list of all types (in Bodo_CTypes enum format) in the
                nested structure rooted at arr_typ"""
                if isinstance(arr_typ, ArrayItemArrayType):
                    return [CTypeEnum.LIST.value] + get_types(arr_typ.dtype)
                # TODO: add Categorical, String
                elif isinstance(arr_typ, (StructType, StructArrayType)):
                    # for struct include the type ID and number of fields
                    types_list = [CTypeEnum.STRUCT.value, len(arr_typ.names)]
                    for field_typ in arr_typ.data:
                        types_list += get_types(field_typ)
                    return types_list
                elif (
                    isinstance(arr_typ, (types.Array, IntegerArrayType))
                    or arr_typ == boolean_array
                ):
                    return get_types(arr_typ.dtype)
                elif arr_typ == string_array_type:
                    return [CTypeEnum.STRING.value]
                elif arr_typ == binary_array_type:
                    return [CTypeEnum.BINARY.value]
                elif isinstance(arr_typ, DecimalArrayType):
                    return [CTypeEnum.Decimal.value, arr_typ.precision, arr_typ.scale]
                else:
                    return [numba_to_c_type(arr_typ)]

            def get_lengths(arr_typ, arr):
                """ Get array of lengths of all arrays in nested structure """
                length = context.compile_internal(
                    builder, lambda a: len(a), types.intp(arr_typ), [arr]
                )
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
                elif isinstance(arr_typ, StructArrayType):
                    payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
                    lengths = []
                    for i, field_typ in enumerate(arr_typ.data):
                        sub_lens = get_lengths(
                            field_typ, builder.extract_value(payload.data, i)
                        )
                        lengths += [
                            builder.extract_value(sub_lens, j)
                            for j in range(sub_lens.type.count)
                        ]
                    lengths = cgutils.pack_array(
                        builder,
                        # lengths array needs to have same number of items as types array.
                        # for struct, second item is num fields, so we set -1 as dummy value
                        [length, context.get_constant(types.int64, -1)] + lengths,
                    )
                # TODO: add Struct, Categorical, String
                elif isinstance(
                    arr_typ, (IntegerArrayType, DecimalArrayType, types.Array)
                ) or arr_typ in (
                    boolean_array,
                    datetime_date_array_type,
                    string_array_type,
                    binary_array_type,
                ):
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
                    offsets_arr = context.make_array(types.Array(offset_type, 1, "C"))(
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
                elif isinstance(arr_typ, StructArrayType):
                    payload = _get_struct_arr_payload(context, builder, arr_typ, arr)
                    buffs_data = []
                    for i, field_typ in enumerate(arr_typ.data):
                        field_bufs = get_buffers(
                            field_typ, builder.extract_value(payload.data, i)
                        )
                        buffs_data += [
                            builder.extract_value(field_bufs, j)
                            for j in range(field_bufs.type.count)
                        ]
                    null_bitmap_arr = context.make_array(
                        types.Array(types.uint8, 1, "C")
                    )(context, builder, payload.null_bitmap)
                    null_bitmap_ptr = builder.bitcast(
                        null_bitmap_arr.data, lir.IntType(8).as_pointer()
                    )
                    buffers = cgutils.pack_array(
                        builder,
                        [null_bitmap_ptr] + buffs_data,
                    )
                elif isinstance(
                    arr_typ, (IntegerArrayType, DecimalArrayType)
                ) or arr_typ in (
                    boolean_array,
                    datetime_date_array_type,
                ):
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
                elif arr_typ in (string_array_type, binary_array_type):
                    payload = _get_string_arr_payload(context, builder, arr)
                    offsets = context.make_helper(
                        builder, offset_arr_type, payload.offsets
                    ).data
                    data = context.make_helper(
                        builder, char_arr_type, payload.data
                    ).data
                    null_bitmap = context.make_helper(
                        builder, null_bitmap_arr_type, payload.null_bitmap
                    ).data
                    buffers = cgutils.pack_array(
                        builder,
                        [
                            builder.bitcast(offsets, lir.IntType(8).as_pointer()),
                            builder.bitcast(null_bitmap, lir.IntType(8).as_pointer()),
                            builder.bitcast(data, lir.IntType(8).as_pointer()),
                        ],
                    )
                elif isinstance(arr_typ, types.Array):
                    arr = context.make_array(arr_typ)(context, builder, arr)
                    data_ptr = builder.bitcast(arr.data, lir.IntType(8).as_pointer())
                    nullptr = lir.Constant(lir.IntType(8).as_pointer(), None)
                    # Numpy arrays don't have null_bitmap. pass nullptr to indicate
                    # no nulls
                    buffers = cgutils.pack_array(builder, [nullptr, data_ptr])
                else:
                    raise RuntimeError(
                        "array_to_info: unsupported type for subarray " + str(arr_typ)
                    )
                return buffers

            def get_field_names(arr_typ):
                l_field_names = []
                if isinstance(arr_typ, StructArrayType):
                    for e_name, e_arr in zip(arr_typ.dtype.names, arr_typ.data):
                        l_field_names.append(e_name)
                        l_field_names += get_field_names(e_arr)
                elif isinstance(arr_typ, ArrayItemArrayType):
                    l_field_names += get_field_names(arr_typ.dtype)
                return l_field_names

            # get list of all types in the nested datastructure (to pass to C++)
            types_list = get_types(arr_type)
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

            l_field_names = get_field_names(arr_type)
            # Field names is used for struct arrays. For nested data structures that do not
            # contain structs we may have an empty l_field_names which causes typing problems.
            # Thus in that case we assign to some non-empty array that will not be used.
            if len(l_field_names) == 0:
                l_field_names = ["irrelevant"]
            field_names = cgutils.pack_array(
                builder,
                [context.insert_const_string(builder.module, a) for a in l_field_names],
            )
            field_names_ptr = cgutils.alloca_once_value(builder, field_names)

            nested_array = context.make_helper(builder, arr_type, in_arr)

            # pass all the data to C++ so that an array_info using nested Arrow
            # Array is constructed
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(32).as_pointer(),
                    lir.IntType(8).as_pointer().as_pointer(),
                    lir.IntType(64).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(
                        8
                    ).as_pointer(),  # Maybe it should be lit.IntType(8).as_pointer().as_pointer()
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="nested_array_to_info"
            )
            ret = builder.call(
                fn_tp,
                [
                    builder.bitcast(types_array_ptr, lir.IntType(32).as_pointer()),
                    builder.bitcast(
                        buffers_ptr, lir.IntType(8).as_pointer().as_pointer()
                    ),
                    builder.bitcast(lengths_ptr, lir.IntType(64).as_pointer()),
                    builder.bitcast(field_names_ptr, lir.IntType(8).as_pointer()),
                    nested_array.meminfo,
                ],
            )
            context.compile_internal(
                builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
            )  # pragma: no cover
            return ret

        # StringArray
        if arr_type in (string_array_type, binary_array_type):
            string_array = context.make_helper(builder, arr_type, in_arr)
            array_item_data_type = ArrayItemArrayType(char_arr_type)
            array_item_array = context.make_helper(
                builder, array_item_data_type, string_array.data
            )
            payload = _get_string_arr_payload(context, builder, in_arr)
            offsets = context.make_helper(
                builder, offset_arr_type, payload.offsets
            ).data
            data = context.make_helper(builder, char_arr_type, payload.data).data
            null_bitmap = context.make_helper(
                builder, null_bitmap_arr_type, payload.null_bitmap
            ).data
            num_total_chars = builder.zext(
                builder.load(builder.gep(offsets, [payload.n_arrays])),
                lir.IntType(64),
            )
            is_bytes = context.get_constant(
                types.int32, int(arr_type == binary_array_type)
            )
            fnty = lir.FunctionType(
                lir.IntType(8).as_pointer(),
                [
                    lir.IntType(64),
                    lir.IntType(64),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(offset_type.bitwidth).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(8).as_pointer(),
                    lir.IntType(32),
                ],
            )
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="string_array_to_info"
            )
            return builder.call(
                fn_tp,
                [
                    payload.n_arrays,
                    num_total_chars,
                    data,
                    offsets,
                    null_bitmap,
                    array_item_array.meminfo,
                    is_bytes,
                ],
            )

        # get codes array from CategoricalArrayType to be handled similar to other Numpy
        # arrays.
        is_categorical = False
        if isinstance(arr_type, CategoricalArrayType):
            # undo the initial incref since the original array is not fully passed to
            # C++ (e.g. dtype value is not passed)
            context.nrt.decref(builder, arr_type, in_arr)
            num_categories = context.compile_internal(
                builder,
                lambda a: len(a.dtype.categories),
                types.intp(arr_type),
                [in_arr],
            )
            in_arr = cgutils.create_struct_proxy(arr_type)(
                context, builder, in_arr
            ).codes
            int_dtype = get_categories_int_type(arr_type.dtype)
            arr_type = types.Array(int_dtype, 1, "C")
            is_categorical = True
            # incref the actual array passed to C++
            context.nrt.incref(builder, arr_type, in_arr)

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
                fn_tp = builder.module.get_or_insert_function(
                    fnty, name="categorical_array_to_info"
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

        # interval array
        if isinstance(arr_type, IntervalArrayType):
            assert isinstance(
                arr_type.arr_type, types.Array
            ), "array_to_info(): only IntervalArrayType with Numpy arrays supported"
            arr = cgutils.create_struct_proxy(arr_type)(context, builder, in_arr)
            left_arr = context.make_array(arr_type.arr_type)(context, builder, arr.left)
            right_arr = context.make_array(arr_type.arr_type)(
                context, builder, arr.right
            )
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
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="interval_array_to_info"
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

        raise BodoError(f"array_to_info(): array type {arr_type} is not supported")

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
    context.compile_internal(
        builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
    )  # pragma: no cover

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


def _lower_info_to_array_list_string_array(arr_type, context, builder, in_info):
    array_item_array_from_cpp = context.make_helper(builder, arr_type)
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),  # info
            lir.IntType(8).as_pointer().as_pointer(),  # meminfo
        ],
    )
    fn_tp = builder.module.get_or_insert_function(
        fnty, name="info_to_list_string_array"
    )
    builder.call(
        fn_tp,
        [
            in_info,
            array_item_array_from_cpp._get_ptr_by_name("meminfo"),
        ],
    )
    context.compile_internal(
        builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
    )  # pragma: no cover

    return array_item_array_from_cpp._getvalue()


def nested_to_array(
    context, builder, arr_typ, lengths_ptr, array_infos_ptr, lengths_pos, infos_pos
):
    """ LLVM codegen for info_to_array for nested types. Is called recursively """

    ll_array_info_type = context.get_data_type(array_info_type)

    if isinstance(arr_typ, ArrayItemArrayType):
        # construct ArrayItemArray given the array of lengths and array_infos received from C++

        lengths_offset = lengths_pos
        infos_offset = infos_pos

        # call codegen for child array
        sub_arr, lengths_pos, infos_pos = nested_to_array(
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
        payload.n_arrays = builder.extract_value(
            builder.load(lengths_ptr), lengths_offset
        )
        payload.data = (
            sub_arr  # data is the child array which was constructed before this
        )

        infos = builder.load(array_infos_ptr)

        # offsets array from array_info
        offsets_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_offset), ll_array_info_type
        )
        payload.offsets = _lower_info_to_array_numpy(
            types.Array(offset_type, 1, "C"),
            context,
            builder,
            offsets_info_ptr,
        )

        # nulls array from array_info
        nulls_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_offset + 1), ll_array_info_type
        )
        payload.null_bitmap = _lower_info_to_array_numpy(
            types.Array(types.uint8, 1, "C"), context, builder, nulls_info_ptr
        )

        builder.store(payload._getvalue(), meminfo_data_ptr)
        array_item_array = context.make_helper(builder, arr_typ)
        array_item_array.meminfo = meminfo
        return array_item_array._getvalue(), lengths_pos, infos_pos

    elif isinstance(arr_typ, StructArrayType):
        # construct StructArrayType given the array of lengths and array_infos received from C++

        # call codegen for children arrays
        sub_arrs = []
        infos_offset = infos_pos
        lengths_pos += 1
        infos_pos += 1
        for d in arr_typ.data:
            sub_arr, lengths_pos, infos_pos = nested_to_array(
                context,
                builder,
                d,
                lengths_ptr,
                array_infos_ptr,
                lengths_pos,
                infos_pos,
            )
            sub_arrs.append(sub_arr)

        # create payload type
        payload_type = StructArrayPayloadType(arr_typ.data)
        payload_alloc_type = context.get_value_type(payload_type)
        payload_alloc_size = context.get_abi_sizeof(payload_alloc_type)

        # define dtor
        dtor_fn = define_struct_arr_dtor(context, builder, arr_typ, payload_type)

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

        payload.data = (
            cgutils.pack_array(builder, sub_arrs)
            if types.is_homogeneous(*arr_typ.data)
            else cgutils.pack_struct(builder, sub_arrs)
        )

        infos = builder.load(array_infos_ptr)

        # nulls array from array_info
        nulls_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_offset), ll_array_info_type
        )
        payload.null_bitmap = _lower_info_to_array_numpy(
            types.Array(types.uint8, 1, "C"), context, builder, nulls_info_ptr
        )

        builder.store(payload._getvalue(), meminfo_data_ptr)
        struct_array = context.make_helper(builder, arr_typ)
        struct_array.meminfo = meminfo
        return struct_array._getvalue(), lengths_pos, infos_pos

    # TODO Categorical

    # StringArray
    elif arr_typ in (string_array_type, binary_array_type):
        infos = builder.load(array_infos_ptr)
        info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos), ll_array_info_type
        )

        string_array = context.make_helper(builder, arr_typ)
        array_item_data_type = ArrayItemArrayType(char_arr_type)
        array_item_array = context.make_helper(builder, array_item_data_type)
        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(8).as_pointer(),  # info
                lir.IntType(8).as_pointer().as_pointer(),  # meminfo
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="info_to_string_array")
        builder.call(
            fn_tp,
            [
                info_ptr,
                array_item_array._get_ptr_by_name("meminfo"),
            ],
        )
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        string_array.data = array_item_array._getvalue()
        return string_array._getvalue(), lengths_pos + 1, infos_pos + 2

    # Numpy
    elif isinstance(arr_typ, types.Array):
        # since the array comes from Arrow, it has a null buffer which should
        # be all 1s. we ignore it because this type is not nullable

        # data array from array_info
        infos = builder.load(array_infos_ptr)
        data_info_ptr = builder.bitcast(
            builder.extract_value(infos, infos_pos + 1), ll_array_info_type
        )
        return (
            _lower_info_to_array_numpy(
                arr_typ,
                context,
                builder,
                data_info_ptr,
            ),
            lengths_pos + 1,
            infos_pos + 2,
        )

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
            types.Array(np_dtype, 1, "C"),
            context,
            builder,
            data_info_ptr,
        )

        return arr._getvalue(), lengths_pos + 1, infos_pos + 2


@intrinsic
def info_to_array(typingctx, info_type, array_type):
    """convert array info wrapper from C++ to regular array object"""
    arr_type = (
        array_type.instance_type
        if isinstance(array_type, types.TypeRef)
        else array_type
    )
    assert info_type == array_info_type

    def codegen(context, builder, sig, args):
        in_info, _ = args
        # TODO: update meminfo?

        if (
            isinstance(arr_type, ArrayItemArrayType)
            and arr_type.dtype == string_array_type
        ):
            return _lower_info_to_array_list_string_array(
                arr_type, context, builder, in_info
            )

        if isinstance(arr_type, (ArrayItemArrayType, StructArrayType, TupleArrayType)):

            def get_num_arrays(arr_typ):
                """ get total number of arrays in nested array """
                if isinstance(arr_typ, ArrayItemArrayType):
                    return 1 + get_num_arrays(arr_typ.dtype)
                elif isinstance(arr_typ, StructArrayType):
                    return 1 + sum([get_num_arrays(d) for d in arr_typ.data])
                else:
                    return 1

            def get_num_infos(arr_typ):
                """get number of array_infos that need to be returned from
                C++ to reconstruct this array"""
                if isinstance(arr_typ, ArrayItemArrayType):
                    # 1 buffer for offsets, 1 buffer for nulls + children buffer count
                    return 2 + get_num_infos(arr_typ.dtype)
                elif isinstance(arr_typ, StructArrayType):
                    # 1 for nulls + children buffer count
                    return 1 + sum([get_num_infos(d) for d in arr_typ.data])
                elif arr_typ in (string_array_type, binary_array_type):
                    # C++ will just use one array_info
                    return 1
                else:
                    # for primitive types: nulls and data
                    # NOTE for non-nullable arrays C++ will still return two
                    # buffers since it doesn't know that the Arrow array of
                    # primitive values is going to be converted to a Numpy array
                    # (all Arrow arrays are nullable)
                    return 2

            if isinstance(arr_type, TupleArrayType):
                # TupleArray uses same model as StructArray so we just use a
                # StructArrayType to generate LLVM
                cpp_arr_type = StructArrayType(
                    arr_type.data, ("dummy",) * len(arr_type.data)
                )
            else:
                cpp_arr_type = arr_type

            n = get_num_arrays(cpp_arr_type)
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
                builder, [nullptr for _ in range(get_num_infos(cpp_arr_type))]
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
            context.compile_internal(
                builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
            )  # pragma: no cover

            # generate code recursively to construct nested arrays from buffers
            # returned from C++
            arr, _, _ = nested_to_array(
                context, builder, cpp_arr_type, lengths_ptr, array_infos_ptr, 0, 0
            )
            if isinstance(arr_type, TupleArrayType):
                # nested_to_array returns StructArray, not TupleArray so we
                # have to return one here
                tuple_array = context.make_helper(builder, arr_type)
                tuple_array.data = arr
                context.nrt.incref(builder, cpp_arr_type, arr)
                arr = tuple_array._getvalue()
            return arr

        # StringArray
        if arr_type in (string_array_type, binary_array_type):
            string_array = context.make_helper(builder, arr_type)
            array_item_data_type = ArrayItemArrayType(char_arr_type)
            array_item_array = context.make_helper(builder, array_item_data_type)
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
                fn_tp,
                [
                    in_info,
                    array_item_array._get_ptr_by_name("meminfo"),
                ],
            )
            context.compile_internal(
                builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
            )  # pragma: no cover
            string_array.data = array_item_array._getvalue()
            return string_array._getvalue()

        if isinstance(arr_type, CategoricalArrayType):
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
            context.compile_internal(
                builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
            )  # pragma: no cover

            intp_t = context.get_value_type(types.intp)

            # data array
            shape_array = cgutils.pack_array(
                builder, [builder.load(length_ptr)], ty=intp_t
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
            meminfo_right_ptr = cgutils.alloca_once(
                builder, lir.IntType(8).as_pointer()
            )

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
            fn_tp = builder.module.get_or_insert_function(
                fnty, name="info_to_interval_array"
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
            context.compile_internal(
                builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
            )  # pragma: no cover

            intp_t = context.get_value_type(types.intp)

            # left array
            shape_array = cgutils.pack_array(
                builder, [builder.load(length_ptr)], ty=intp_t
            )
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

        raise BodoError(f"info_to_array(): array type {arr_type} is not supported")

    return arr_type(info_type, array_type), codegen


@intrinsic
def test_alloc_np(typingctx, len_typ, arr_type):
    array_type = (
        arr_type.instance_type if isinstance(arr_type, types.TypeRef) else arr_type
    )

    def codegen(context, builder, sig, args):
        length, _ = args
        typ_enum = numba_to_c_type(array_type.dtype)
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
def arr_info_list_to_table(typingctx, list_arr_info_typ=None):
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


delete_info_decref_array = types.ExternalFunction(
    "delete_info_decref_array",
    types.void(array_info_type),
)


delete_table_decref_arrays = types.ExternalFunction(
    "delete_table_decref_arrays",
    types.void(table_type),
)


@intrinsic
def delete_table(typingctx, table_t=None):
    """Deletes table and its array_info objects. Doesn't delete array data."""
    assert table_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(lir.VoidType(), [lir.IntType(8).as_pointer()])
        fn_tp = builder.module.get_or_insert_function(fnty, name="delete_table")
        builder.call(fn_tp, args)

    return types.void(table_t), codegen


# TODO Add a test for this
@intrinsic
def shuffle_table(typingctx, table_t, n_keys_t, keep_comm_info_t):  # pragma: no cover
    """shuffle input table so that rows with same key are on the same process.
    Steals a reference from the input table.
    'keep_comm_info' parameter specifies if shuffle information should be kept in
    output table, to be used for reverse shuffle later (e.g. in groupby apply).
    """
    assert table_t == table_type

    def codegen(context, builder, sig, args):  # pragma: no cover
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [lir.IntType(8).as_pointer(), lir.IntType(64), lir.IntType(32)],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="shuffle_table")
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return table_type(table_t, types.int64, types.int32), codegen


class ShuffleInfoType(types.Type):
    def __init__(self):
        super(ShuffleInfoType, self).__init__(name="ShuffleInfoType()")


shuffle_info_type = ShuffleInfoType()
register_model(ShuffleInfoType)(models.OpaqueModel)


get_shuffle_info = types.ExternalFunction(
    "get_shuffle_info",
    shuffle_info_type(table_type),
)


delete_shuffle_info = types.ExternalFunction(
    "delete_shuffle_info",
    types.void(shuffle_info_type),
)


reverse_shuffle_table = types.ExternalFunction(
    "reverse_shuffle_table",
    table_type(table_type, shuffle_info_type),
)


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
    same_need_typechange_t,
    is_left_t,
    is_right_t,
    is_join_t,
    optional_col_t,
    indicator,
    _bodo_na_equal,
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
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="hash_join_table")
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
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
            types.boolean,
            types.boolean,
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
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

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
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return (
        table_type(table_t, types.int64, types.voidptr, types.boolean, types.boolean),
        codegen,
    )


@intrinsic
def sample_table(typingctx, table_t, n_keys_t, frac_t, replace_t, parallel_t):
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
                lir.DoubleType(),
                lir.IntType(1),
                lir.IntType(1),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(fnty, name="sample_table")
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return (
        table_type(table_t, types.int64, types.float64, types.boolean, types.boolean),
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
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="shuffle_renormalization"
        )
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
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
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="shuffle_renormalization_group"
        )
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return (
        table_type(
            table_t, types.int32, types.int64, types.boolean, types.int64, types.voidptr
        ),
        codegen,
    )


@intrinsic
def drop_duplicates_table(
    typingctx, table_t, parallel_t, nkey_t, keep_t, ncols, dropna
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
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="drop_duplicates_table"
        )
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return (
        table_type(
            table_t, types.boolean, types.int64, types.int64, types.int64, types.boolean
        ),
        codegen,
    )


@intrinsic
def pivot_groupby_and_aggregate(
    typingctx,
    table_t,
    n_keys_t,
    dispatch_table_t,
    dispatch_info_t,
    input_has_index,
    ftypes,
    func_offsets,
    udf_n_redvars,
    is_parallel,
    is_crosstab,
    skipdropna_t,
    return_keys,
    return_index,
    update_cb,
    combine_cb,
    eval_cb,
    udf_table_dummy_t,
):
    """
    Interface to pivot_groupby_and_aggregate function in C++ library for groupby
    offloading.
    """
    assert table_t == table_type
    assert dispatch_table_t == table_type
    assert dispatch_info_t == table_type
    assert udf_table_dummy_t == table_type

    def codegen(context, builder, sig, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
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
            fnty, name="pivot_groupby_and_aggregate"
        )
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return (
        table_type(
            table_t,
            types.int64,
            table_t,
            table_t,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.boolean,
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


@intrinsic
def groupby_and_aggregate(
    typingctx,
    table_t,
    n_keys_t,
    input_has_index,
    ftypes,
    func_offsets,
    udf_n_redvars,
    is_parallel,
    skipdropna_t,
    shift_periods_t,
    transform_func,
    return_keys,
    return_index,
    dropna,
    update_cb,
    combine_cb,
    eval_cb,
    general_udfs_cb,
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
                lir.IntType(1),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),  # shift_periods_t
                lir.IntType(64),  # transform_func
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1),  # groupby key dropna
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
            ],
        )
        fn_tp = builder.module.get_or_insert_function(
            fnty, name="groupby_and_aggregate"
        )
        ret = builder.call(fn_tp, args)
        context.compile_internal(
            builder, lambda: check_and_propagate_cpp_exception(), types.none(), []
        )  # pragma: no cover
        return ret

    return (
        table_type(
            table_t,
            types.int64,
            types.boolean,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.boolean,
            types.boolean,
            types.int64,  # shift_periods
            types.int64,  # transform_func
            types.boolean,
            types.boolean,
            types.boolean,  # dropna
            types.voidptr,
            types.voidptr,
            types.voidptr,
            types.voidptr,
            table_t,
        ),
        codegen,
    )


get_groupby_labels = types.ExternalFunction(
    "get_groupby_labels",
    types.int64(table_type, types.voidptr, types.boolean),
)


_array_isin = types.ExternalFunction(
    "array_isin",
    types.void(array_info_type, array_info_type, array_info_type, types.bool_),
)


@numba.njit
def array_isin(out_arr, in_arr, in_values, is_parallel):  # pragma: no cover

    in_arr_info = array_to_info(in_arr)
    in_values_info = array_to_info(in_values)
    out_arr_info = array_to_info(out_arr)
    # NOTE: creating a dummy table to avoid Numba's bug in refcount pruning
    dummy_table = arr_info_list_to_table([in_arr_info, in_values_info, out_arr_info])

    _array_isin(
        out_arr_info,
        in_arr_info,
        in_values_info,
        is_parallel,
    )
    check_and_propagate_cpp_exception()
    # no need to decref since array_isin decrefs input/output
    delete_table(dummy_table)
