"""helper functions for code generation with llvmlite
"""
from numba.core import types, cgutils
import bodo
from llvmlite import ir as lir
import llvmlite.binding as ll

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hdist
from bodo.libs import array_ext

ll.add_symbol("array_getitem", array_ext.array_getitem)
ll.add_symbol("seq_getitem", array_ext.seq_getitem)
ll.add_symbol("list_check", array_ext.list_check)
ll.add_symbol("is_na_value", array_ext.is_na_value)


def set_bitmap_bit(builder, null_bitmap_ptr, ind, val):
    """set bit number 'ind' of bitmap array 'null_bitmap_ptr' to val
    """
    byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
    bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
    byte_ptr = builder.gep(null_bitmap_ptr, [byte_ind], inbounds=True)
    byte = builder.load(byte_ptr)
    ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
    mask_tup = cgutils.alloca_once_value(
        builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128))
    )
    mask = builder.load(
        builder.gep(
            mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True
        )
    )
    if val:
        # set masked bit
        builder.store(builder.or_(byte, mask), byte_ptr)
    else:
        # flip all bits of mask e.g. 11111101
        mask = builder.xor(mask, lir.Constant(lir.IntType(8), -1))
        # unset masked bit
        builder.store(builder.and_(byte, mask), byte_ptr)


def get_bitmap_bit(builder, null_bitmap_ptr, ind):
    """get bit number 'ind' of bitmap array 'null_bitmap_ptr'
    """
    # (null_bitmap[i / 8] & kBitmask[i % 8])
    byte_ind = builder.lshr(ind, lir.Constant(lir.IntType(64), 3))
    bit_ind = builder.urem(ind, lir.Constant(lir.IntType(64), 8))
    byte = builder.load(builder.gep(null_bitmap_ptr, [byte_ind], inbounds=True))
    ll_typ_mask = lir.ArrayType(lir.IntType(8), 8)
    mask_tup = cgutils.alloca_once_value(
        builder, lir.Constant(ll_typ_mask, (1, 2, 4, 8, 16, 32, 64, 128))
    )
    mask = builder.load(
        builder.gep(
            mask_tup, [lir.Constant(lir.IntType(64), 0), bit_ind], inbounds=True
        )
    )
    return builder.and_(byte, mask)


def pyarray_getitem(builder, context, arr_obj, ind):
    """getitem of 1D Numpy array
    """
    pyobj = context.get_argument_type(types.pyobject)
    py_ssize_t = context.get_value_type(types.intp)
    arr_get_fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [pyobj, py_ssize_t])
    arr_get_fn = builder.module.get_or_insert_function(
        arr_get_fnty, name="array_getptr1"
    )
    arr_getitem_fnty = lir.FunctionType(pyobj, [pyobj, lir.IntType(8).as_pointer()])
    arr_getitem_fn = builder.module.get_or_insert_function(
        arr_getitem_fnty, name="array_getitem"
    )
    arr_ptr = builder.call(arr_get_fn, [arr_obj, ind])
    return builder.call(arr_getitem_fn, [arr_obj, arr_ptr])


def pyarray_setitem(builder, context, arr_obj, ind, val_obj):
    """setitem of 1D Numpy array
    """
    pyobj = context.get_argument_type(types.pyobject)
    py_ssize_t = context.get_value_type(types.intp)
    arr_get_fnty = lir.FunctionType(lir.IntType(8).as_pointer(), [pyobj, py_ssize_t])
    arr_get_fn = builder.module.get_or_insert_function(
        arr_get_fnty, name="array_getptr1"
    )
    arr_setitem_fnty = lir.FunctionType(
        lir.VoidType(), [pyobj, lir.IntType(8).as_pointer(), pyobj]
    )
    arr_setitem_fn = builder.module.get_or_insert_function(
        arr_setitem_fnty, name="array_setitem"
    )
    arr_ptr = builder.call(arr_get_fn, [arr_obj, ind])
    builder.call(arr_setitem_fn, [arr_obj, arr_ptr, val_obj])


def seq_getitem(builder, context, obj, ind):
    """getitem for a sequence object (e.g. list/array)
    """
    pyobj = context.get_argument_type(types.pyobject)
    py_ssize_t = context.get_value_type(types.intp)
    getitem_fnty = lir.FunctionType(pyobj, [pyobj, py_ssize_t])
    getitem_fn = builder.module.get_or_insert_function(getitem_fnty, name="seq_getitem")
    return builder.call(getitem_fn, [obj, ind])


def is_na_value(builder, context, val, C_NA):
    """check if Python object 'val' is an NA value (None, or np.nan or pd.NA).
    passing pd.NA in as C_NA to avoid getattr overheads inside loops.
    """
    pyobj = context.get_argument_type(types.pyobject)
    arr_isna_fnty = lir.FunctionType(lir.IntType(32), [pyobj, pyobj])
    arr_isna_fn = builder.module.get_or_insert_function(
        arr_isna_fnty, name="is_na_value"
    )
    return builder.call(arr_isna_fn, [val, C_NA])


def list_check(builder, context, obj):
    """check if Python object 'obj' is a list
    """
    pyobj = context.get_argument_type(types.pyobject)
    int32_type = context.get_value_type(types.int32)
    fnty = lir.FunctionType(int32_type, [pyobj])
    fn = builder.module.get_or_insert_function(fnty, name="list_check")
    return builder.call(fn, [obj])


def to_arr_obj_if_list_obj(c, context, builder, val, typ):
    """convert object 'val' to array if it is a list to enable proper unboxing
    """
    if not (isinstance(typ, types.List) or bodo.utils.utils.is_array_typ(typ, False)):
        return val

    # convert val to array if it is a list (list_check is needed since converting
    # pd arrays like IntergerArray can cause errors)
    # if isinstance(val, list):
    #   val = np.array(val)
    val_ptr = cgutils.alloca_once_value(builder, val)
    is_list = list_check(builder, context, val)
    is_list_cond = builder.icmp_unsigned("!=", is_list, lir.Constant(is_list.type, 0))
    with builder.if_then(is_list_cond):

        # get np.array to convert list items to array
        mod_name = context.insert_const_string(builder.module, "numpy")
        np_mod_obj = c.pyapi.import_module_noblock(mod_name)
        np_array_method = c.pyapi.object_getattr_string(np_mod_obj, "asarray")
        dtype_str = "object_"
        # float lists become float arrays, but others are object arrays
        # (see _value_to_array in boxing.py)
        if isinstance(typ.dtype, types.Float):
            dtype_str = str(typ.dtype)
        dtype_obj = c.pyapi.object_getattr_string(np_mod_obj, dtype_str)

        old_obj = builder.load(val_ptr)
        new_obj = c.pyapi.call(
            np_array_method, c.pyapi.tuple_pack([old_obj, dtype_obj])
        )
        # TODO: this decref causes crashes for some reason in test_array_item_array.py
        # needs to be investigated to avoid object leaks
        # c.pyapi.decref(old_obj)
        builder.store(new_obj, val_ptr)
        c.pyapi.decref(np_mod_obj)
        c.pyapi.decref(np_array_method)
        c.pyapi.decref(dtype_obj)

    val = builder.load(val_ptr)
    return val


def get_array_elem_counts(c, builder, context, arr_obj, typ):
    """count elements of array for upfront allocation (called recursively).
    For example, [[1, None, 3], [3], None, [4, None, 2]] return (4, 7).
    A recursive example: [[[1, None, 3], [2]], [[3], [4, 5]], None, [[4, None, 2]]]
    returns (4, 5, 10).
    """
    from bodo.libs.array_item_arr_ext import ArrayItemArrayType
    from bodo.libs.struct_arr_ext import StructArrayType, StructType
    from bodo.libs.str_arr_ext import string_array_type, get_utf8_size

    # get utf8 character count for strings
    if typ == bodo.string_type:
        str_item = c.pyapi.to_native_value(bodo.string_type, arr_obj).value
        _is_error, n_chars = c.pyapi.call_jit_code(
            lambda a: get_utf8_size(a), types.int64(bodo.string_type), [str_item]
        )
        context.nrt.decref(builder, typ, str_item)
        return cgutils.pack_array(builder, [n_chars])

    # get counts for struct value (e.g. counts for strings or array items in struct)
    if isinstance(typ, StructType):

        # get pd.NA object to check for new NA kind
        mod_name = context.insert_const_string(builder.module, "pandas")
        pd_mod_obj = c.pyapi.import_module_noblock(mod_name)
        C_NA = c.pyapi.object_getattr_string(pd_mod_obj, "NA")

        n_nested_count = bodo.utils.transform.get_type_alloc_counts(typ)
        counts_val = context.make_tuple(
            builder,
            types.Tuple(n_nested_count * [types.int64]),
            n_nested_count * [context.get_constant(types.int64, 0)],
        )
        counts = cgutils.alloca_once_value(builder, counts_val)
        curr_count_ind = 0
        for i, t in enumerate(typ.data):
            n_nested_count_t = bodo.utils.transform.get_type_alloc_counts(t)
            if n_nested_count_t == 0:
                continue
            val_obj = c.pyapi.dict_getitem_string(arr_obj, typ.names[i])
            # check for NA
            is_na = is_na_value(builder, context, val_obj, C_NA)
            not_na_cond = builder.icmp_unsigned(
                "!=", is_na, lir.Constant(is_na.type, 1)
            )
            with builder.if_then(not_na_cond):
                counts_val = builder.load(counts)
                n_vals_inner = get_array_elem_counts(c, builder, context, val_obj, t)
                for i in range(n_nested_count_t):
                    total_count = builder.extract_value(counts_val, curr_count_ind + i)
                    curr_count = builder.extract_value(n_vals_inner, i)
                    counts_val = builder.insert_value(
                        counts_val,
                        builder.add(total_count, curr_count),
                        curr_count_ind + i,
                    )
                builder.store(counts_val, counts)
            # no need to decref val_obj, dict_getitem_string returns borrowed ref
            curr_count_ind += n_nested_count_t

        c.pyapi.decref(pd_mod_obj)
        c.pyapi.decref(C_NA)
        return builder.load(counts)

    # return empty tuple for non-arrays
    if not bodo.utils.utils.is_array_typ(typ, False):
        return cgutils.pack_array(builder, [], lir.IntType(64))

    n = bodo.utils.utils.object_length(c, arr_obj)

    # return (n,) for non-nested arrays
    if not (
        isinstance(typ, (ArrayItemArrayType, StructArrayType))
        or typ == string_array_type
    ):
        return cgutils.pack_array(builder, [n])

    # get pd.NA object to check for new NA kind
    mod_name = context.insert_const_string(builder.module, "pandas")
    pd_mod_obj = c.pyapi.import_module_noblock(mod_name)
    C_NA = c.pyapi.object_getattr_string(pd_mod_obj, "NA")

    # create a tuple for nested counts
    n_nested_count = bodo.utils.transform.get_type_alloc_counts(typ)
    counts_val = context.make_tuple(
        builder,
        types.Tuple(n_nested_count * [types.int64]),
        [n] + (n_nested_count - 1) * [context.get_constant(types.int64, 0)],
    )
    counts = cgutils.alloca_once_value(builder, counts_val)

    # pseudocode for code generation:
    # counts = (n, 0, 0, ...)  # tuple of nested counts
    # for i in range(len(A)):
    #   arr_item_obj = A[i]
    #   if not isna(arr_item_obj):
    #     counts[1:] += get_array_elem_counts(arr_item_obj)

    # for each array
    with cgutils.for_range(builder, n) as loop:
        array_ind = loop.index
        # arr_item_obj = A[i]
        arr_item_obj = seq_getitem(builder, context, arr_obj, array_ind)
        # check for NA
        is_na = is_na_value(builder, context, arr_item_obj, C_NA)
        not_na_cond = builder.icmp_unsigned("!=", is_na, lir.Constant(is_na.type, 1))
        with builder.if_then(not_na_cond):
            if isinstance(typ, ArrayItemArrayType) or typ == string_array_type:
                counts_val = builder.load(counts)
                # if nested, call recursively and add values
                n_vals_inner = get_array_elem_counts(
                    c, builder, context, arr_item_obj, typ.dtype
                )
                # counts[1:] += get_array_elem_counts(arr_item_obj)
                for i in range(n_nested_count - 1):
                    total_count = builder.extract_value(counts_val, i + 1)
                    curr_count = builder.extract_value(n_vals_inner, i)
                    counts_val = builder.insert_value(
                        counts_val, builder.add(total_count, curr_count), i + 1
                    )
                builder.store(counts_val, counts)
            else:
                assert isinstance(typ, StructArrayType), typ
                curr_count_ind = 1  # skip total array length
                for i, t in enumerate(typ.data):
                    n_nested_count_t = bodo.utils.transform.get_type_alloc_counts(
                        t.dtype
                    )
                    if n_nested_count_t == 0:
                        continue
                    val_obj = c.pyapi.dict_getitem_string(arr_item_obj, typ.names[i])
                    # check for NA
                    is_na = is_na_value(builder, context, val_obj, C_NA)
                    not_na_cond = builder.icmp_unsigned(
                        "!=", is_na, lir.Constant(is_na.type, 1)
                    )
                    with builder.if_then(not_na_cond):
                        counts_val = builder.load(counts)
                        n_vals_inner = get_array_elem_counts(
                            c, builder, context, val_obj, t.dtype
                        )
                        for i in range(n_nested_count_t):
                            total_count = builder.extract_value(
                                counts_val, curr_count_ind + i
                            )
                            curr_count = builder.extract_value(n_vals_inner, i)
                            counts_val = builder.insert_value(
                                counts_val,
                                builder.add(total_count, curr_count),
                                curr_count_ind + i,
                            )
                        builder.store(counts_val, counts)
                    # no need to decref val_obj, dict_getitem_string returns borrowed ref
                    curr_count_ind += n_nested_count_t
        c.pyapi.decref(arr_item_obj)

    c.pyapi.decref(pd_mod_obj)
    c.pyapi.decref(C_NA)
    return builder.load(counts)


def gen_allocate_array(context, builder, arr_type, n_elems, c=None):
    """gen array allocation for type 'arr_type'.
    'n_elems' is a tuple of all counts needed for allocation, e.g. (3, 5) for array item
    that has 3 arrays and 5 primitive elements.
    'c' is boxing/unboxing context and can be None if not in boxing/unboxing steps.
    When in boxing/unboxing, 'call_jit_code' is used to handle Python error model.
    """
    n_counts = n_elems.type.count
    assert n_counts >= 1
    length = builder.extract_value(n_elems, 0)
    # if nested counts are provided, pack a new tuple for them
    if n_counts != 1:
        n_nested_elems = cgutils.pack_array(
            builder, [builder.extract_value(n_elems, i) for i in range(1, n_counts)]
        )
        nested_counts_typ = types.Tuple([types.int64] * (n_counts - 1))
    else:
        n_nested_elems = context.get_dummy_value()
        nested_counts_typ = types.none
    # call alloc_type
    t_ref = types.TypeRef(arr_type)
    sig = arr_type(types.int64, t_ref, nested_counts_typ)
    args = [length, context.get_dummy_value(), n_nested_elems]
    impl = lambda n, t, s: bodo.utils.utils.alloc_type(n, t, s)
    if c:
        _is_error, out_arr = c.pyapi.call_jit_code(impl, sig, args)
    else:
        out_arr = context.compile_internal(builder, impl, sig, args)
    return out_arr
