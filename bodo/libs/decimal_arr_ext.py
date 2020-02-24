# Copyright (C) 2020 Bodo Inc. All rights reserved.
"""Decimal array corresponding to Arrow Decimal128Array type.
It is similar to Spark's DecimalType. From Spark's docs:
'The DecimalType must have fixed precision (the maximum total number of digits) and
scale (the number of digits on the right of dot). For example, (5, 2) can support the
value from [-999.99 to 999.99].
The precision can be up to 38, the scale must be less or equal to precision.'
'When infer schema from decimal.Decimal objects, it will be DecimalType(38, 18).'
"""
import operator
import pandas as pd
import numpy as np
import numba
import bodo
from numba import types
from numba import cgutils
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
from numba.array_analysis import ArrayAnalysis

from llvmlite import ir as lir
import llvmlite.binding as ll
from bodo.libs import decimal_ext

ll.add_symbol("box_decimal_array", decimal_ext.box_decimal_array)

from bodo.utils.typing import (
    get_overload_const_int,
    is_overload_constant_int,
    int128_type,
    decimal_type,
)


class DecimalArrayType(types.ArrayCompatible):
    def __init__(self, precision, scale):
        self.precision = precision
        self.scale = scale
        super(DecimalArrayType, self).__init__(
            name="DecimalArrayType({}, {})".format(precision, scale)
        )

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return DecimalArrayType(self.precision, self.scale)

    @property
    def dtype(self):
        return decimal_type


# store data and nulls as regular numpy arrays without payload machineray
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
# NOTE: storing data as int128 elements. struct of 8 bytes could be better depending on
# the operations needed
@register_model(DecimalArrayType)
class DecimalArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", types.Array(int128_type, 1, "C")),
            ("null_bitmap", types.Array(types.uint8, 1, "C")),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DecimalArrayType, "data", "_data")
make_attribute_wrapper(DecimalArrayType, "null_bitmap", "_null_bitmap")


@intrinsic
def init_decimal_array(typingctx, data, null_bitmap, precision_tp, scale_tp=None):
    """Create a DecimalArray with provided data and null bitmap values.
    """
    assert data == types.Array(int128_type, 1, "C")
    assert null_bitmap == types.Array(types.uint8, 1, "C")
    assert is_overload_constant_int(precision_tp)
    assert is_overload_constant_int(scale_tp)

    def codegen(context, builder, signature, args):
        data_val, bitmap_val, _, _ = args
        # create decimal_arr struct and store values
        decimal_arr = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        decimal_arr.data = data_val
        decimal_arr.null_bitmap = bitmap_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], bitmap_val)

        return decimal_arr._getvalue()

    precision = get_overload_const_int(precision_tp)
    scale = get_overload_const_int(scale_tp)
    ret_typ = DecimalArrayType(precision, scale)
    sig = ret_typ(data, null_bitmap, precision_tp, scale_tp)
    return sig, codegen


# high-level allocation function for decimal arrays
@numba.njit(no_cpython_wrapper=True)
def alloc_decimal_array(n, precision, scale):
    data_arr = np.empty(n, dtype=int128_type)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    return init_decimal_array(data_arr, nulls, precision, scale)


def alloc_decimal_array_equiv(self, scope, equiv_set, args, kws):
    """Array analysis function for alloc_decimal_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 1 and not kws
    return args[0], []


ArrayAnalysis._analyze_op_call_bodo_libs_decimal_arr_ext_alloc_decimal_array = (
    alloc_decimal_array_equiv
)


@box(DecimalArrayType)
def box_decimal_arr(typ, val, c):
    """
    Box decimal array into ndarray with decimal.Decimal values.
    Represents null as None to match Pandas.
    """
    in_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    data_arr = c.context.make_array(types.Array(int128_type, 1, "C"))(
        c.context, c.builder, in_arr.data
    )
    bitmap_arr_data = c.context.make_array(types.Array(types.uint8, 1, "C"))(
        c.context, c.builder, in_arr.null_bitmap
    ).data

    n = c.builder.extract_value(data_arr.shape, 0)
    scale = c.context.get_constant(types.int32, typ.scale)

    fnty = lir.FunctionType(
        c.pyapi.pyobj,
        [
            lir.IntType(64),
            lir.IntType(128).as_pointer(),
            lir.IntType(8).as_pointer(),
            lir.IntType(32),
        ],
    )
    fn_get = c.builder.module.get_or_insert_function(fnty, name="box_decimal_array")
    obj_arr = c.builder.call(fn_get, [n, data_arr.data, bitmap_arr_data, scale,],)

    return obj_arr


@unbox(DecimalArrayType)
def unbox_decimal_arr(typ, val, c):
    """
    Unbox a numpy array with Decimal objects into native DecimalArray
    """
    decimal_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # allocate data and null_bitmap arrays
    n = c.pyapi.long_as_longlong(c.pyapi.call_method(val, "__len__", ()))
    n_bytes = c.builder.udiv(
        c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    data_arr_struct = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(int128_type, 1, "C"), [n]
    )
    bitmap_arr_struct = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(types.uint8, 1, "C"), [n_bytes]
    )

    # function signature of unbox_decimal_array
    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(64),
            lir.IntType(128).as_pointer(),
            lir.IntType(8).as_pointer(),
        ],
    )
    fn = c.builder.module.get_or_insert_function(fnty, name="unbox_decimal_array")
    c.builder.call(
        fn, [val, n, data_arr_struct.data, bitmap_arr_struct.data,],
    )

    decimal_arr.null_bitmap = bitmap_arr_struct._getvalue()
    decimal_arr.data = data_arr_struct._getvalue()

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(decimal_arr._getvalue(), is_error=is_error)


@overload(len)
def overload_decimal_arr_len(A):
    if isinstance(A, DecimalArrayType):
        return lambda A: len(A._data)


@overload_attribute(DecimalArrayType, "shape")
def overload_decimal_arr_shape(A):
    return lambda A: (len(A._data),)


@overload_attribute(DecimalArrayType, "ndim")
def overload_decimal_arr_ndim(A):
    return lambda A: 1
