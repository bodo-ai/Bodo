"""helper functions for code generation with llvmlite
"""
from numba.core import types, cgutils
from llvmlite import ir as lir
import llvmlite.binding as ll

# NOTE: importing hdist is necessary for MPI initialization before array_ext
from bodo.libs import hdist
from bodo.libs import array_ext

ll.add_symbol("array_getitem", array_ext.array_getitem)
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
