# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""Decimal array corresponding to Arrow Decimal128Array type.
It is similar to Spark's DecimalType. From Spark's docs:
'The DecimalType must have fixed precision (the maximum total number of digits) and
scale (the number of digits on the right of dot). For example, (5, 2) can support the
value from [-999.99 to 999.99].
The precision can be up to 38, the scale must be less or equal to precision.'
'When infer schema from decimal.Decimal objects, it will be DecimalType(38, 18).'
"""
import operator
from decimal import Decimal
from enum import Enum

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
import pyarrow as pa
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import lower_constant
from numba.extending import (
    NativeValue,
    box,
    intrinsic,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.libs import decimal_ext

ll.add_symbol("box_decimal_array", decimal_ext.box_decimal_array)
ll.add_symbol("unbox_decimal", decimal_ext.unbox_decimal)
ll.add_symbol("box_decimal", decimal_ext.box_decimal)
ll.add_symbol("unbox_decimal_array", decimal_ext.unbox_decimal_array)
ll.add_symbol("decimal_to_str", decimal_ext.decimal_to_str)
ll.add_symbol("str_to_decimal", decimal_ext.str_to_decimal)
ll.add_symbol("decimal_to_double", decimal_ext.decimal_to_double_py_entry)
ll.add_symbol("decimal_to_int64", decimal_ext.decimal_to_int64_py_entry)
ll.add_symbol("int64_to_decimal", decimal_ext.int64_to_decimal)

ll.add_symbol("arrow_compute_cmp_py_entry", decimal_ext.arrow_compute_cmp_py_entry)
ll.add_symbol(
    "arrow_compute_cmp_decimal_int_py_entry",
    decimal_ext.arrow_compute_cmp_decimal_int_py_entry,
)
ll.add_symbol(
    "arrow_compute_cmp_decimal_float_py_entry",
    decimal_ext.arrow_compute_cmp_decimal_float_py_entry,
)
ll.add_symbol(
    "arrow_compute_cmp_decimal_decimal_py_entry",
    decimal_ext.arrow_compute_cmp_decimal_decimal_py_entry,
)


from bodo.utils.indexing import (
    array_getitem_bool_index,
    array_getitem_int_index,
    array_getitem_slice_index,
    array_setitem_bool_index,
    array_setitem_int_index,
    array_setitem_slice_index,
)
from bodo.utils.typing import (
    BodoError,
    get_overload_const_int,
    is_iterable_type,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_none,
    is_scalar_type,
)

int128_type = types.Integer("int128", 128)

int_to_decimal_precision = {
    types.int8: 2,
    types.int16: 4,
    types.int32: 9,
    types.int64: 18,
    types.uint8: 2,
    types.uint16: 4,
    types.uint32: 9,
    types.uint64: 19,
}


class Decimal128Type(types.Type):
    """data type for Decimal128 values similar to Arrow's Decimal128"""

    def __init__(self, precision, scale):
        assert isinstance(precision, int)
        assert isinstance(scale, int)
        super(Decimal128Type, self).__init__(
            name="Decimal128Type({}, {})".format(precision, scale)
        )
        self.precision = precision
        self.scale = scale
        self.bitwidth = 128  # needed for using IntegerModel

    def unify(self, typingctx, other):
        """Allow casting int/decimal if scale is 0"""
        if isinstance(other, types.Integer) and self.scale == 0:
            other = types.unliteral(other)
            # return integer if it's wider
            if int_to_decimal_precision[other] >= self.precision:
                return other
            return self


# For the processing of the data we have to put a precision and scale.
# As it turn out when reading boxed data we may certainly have precision not 38
# and scale not 18.
# But we choose to arbitrarily assign precision=38 and scale=18 and it turns
# out that it works.
@typeof_impl.register(Decimal)
def typeof_decimal_value(val, c):
    return Decimal128Type(38, 18)


@typeof_impl.register(pa.Decimal128Scalar)
def typeof_decimal_value(val, c):
    t = val.type
    return Decimal128Type(t.precision, t.scale)


register_model(Decimal128Type)(models.IntegerModel)


@intrinsic(prefer_literal=True)
def int128_to_decimal128type(typingctx, val, precision_tp, scale_tp=None):
    """cast int128 to decimal128"""
    assert val == int128_type
    assert is_overload_constant_int(precision_tp)
    assert is_overload_constant_int(scale_tp)

    def codegen(context, builder, signature, args):
        return args[0]

    precision = get_overload_const_int(precision_tp)
    scale = get_overload_const_int(scale_tp)
    return (
        Decimal128Type(precision, scale)(int128_type, precision_tp, scale_tp),
        codegen,
    )


@intrinsic(prefer_literal=True)
def decimal128type_to_int128(typingctx, val):
    """cast int128 to decimal128"""
    assert isinstance(val, Decimal128Type)

    def codegen(context, builder, signature, args):
        return args[0]

    return int128_type(val), codegen


def decimal_to_str_codegen(context, builder, signature, args, scale):
    (val,) = args
    scale = context.get_constant(types.int32, scale)

    uni_str = cgutils.create_struct_proxy(types.unicode_type)(context, builder)

    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(128),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(64).as_pointer(),
            lir.IntType(32),
        ],
    )
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="decimal_to_str")
    builder.call(
        fn,
        [
            val,
            uni_str._get_ptr_by_name("meminfo"),
            uni_str._get_ptr_by_name("length"),
            scale,
        ],
    )

    # output is always ASCII
    uni_str.kind = context.get_constant(
        types.int32, numba.cpython.unicode.PY_UNICODE_1BYTE_KIND
    )
    uni_str.is_ascii = context.get_constant(types.int32, 1)
    # set hash value -1 to indicate "need to compute hash"
    uni_str.hash = context.get_constant(numba.cpython.unicode._Py_hash_t, -1)
    uni_str.data = context.nrt.meminfo_data(builder, uni_str.meminfo)
    # Set parent to NULL
    uni_str.parent = cgutils.get_null_value(uni_str.parent.type)
    return uni_str._getvalue()


@intrinsic(prefer_literal=True)
def decimal_to_str(typingctx, val_t=None):
    """convert decimal128 to string"""
    assert isinstance(val_t, Decimal128Type)

    def codegen(context, builder, signature, args):
        return decimal_to_str_codegen(context, builder, signature, args, val_t.scale)

    return bodo.string_type(val_t), codegen


def str_to_decimal_codegen(context, builder, signature, args):
    val, _, _ = args
    val = bodo.libs.str_ext.gen_unicode_to_std_str(context, builder, val)
    fnty = lir.FunctionType(
        lir.IntType(128),
        [
            lir.IntType(8).as_pointer(),
        ],
    )
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="str_to_decimal")
    decimal_val = builder.call(
        fn,
        [
            val,
        ],
    )
    return decimal_val


@intrinsic(prefer_literal=True)
def str_to_decimal(typingctx, val, precision_tp, scale_tp=None):
    """convert string ot decimal128"""
    assert val == bodo.string_type or is_overload_constant_str(val)
    assert is_overload_constant_int(precision_tp)
    assert is_overload_constant_int(scale_tp)

    def codegen(context, builder, signature, args):
        return str_to_decimal_codegen(context, builder, signature, args)

    precision = get_overload_const_int(precision_tp)
    scale = get_overload_const_int(scale_tp)
    return (
        Decimal128Type(precision, scale)(val, precision_tp, scale_tp),
        codegen,
    )


# We cannot have exact matching between Python and Bodo
# regarding the strings between decimal.
# If you write Decimal("4.0"), Decimal("4.00"), or Decimal("4")
# their python output is "4.0", "4.00", and "4"
# but for Bodo the output is always "4"
@overload_method(Decimal128Type, "__str__")
def overload_str_decimal(val):
    def impl(val):  # pragma: no cover
        return decimal_to_str(val)

    return impl


@intrinsic(prefer_literal=True)
def decimal128type_to_int64_tuple(typingctx, val):
    """convert decimal128type to a 2-tuple of int64 values"""
    assert isinstance(val, Decimal128Type)

    def codegen(context, builder, signature, args):
        # allocate a lir.ArrayType and store value using pointer bitcast
        res = cgutils.alloca_once(builder, lir.ArrayType(lir.IntType(64), 2))
        builder.store(args[0], builder.bitcast(res, lir.IntType(128).as_pointer()))
        return builder.load(res)

    return types.UniTuple(types.int64, 2)(val), codegen


_arrow_compute_cmp_decimal_int = types.ExternalFunction(
    "arrow_compute_cmp_decimal_int_py_entry",
    types.bool_(
        types.int32,
        int128_type,
        types.int32,
        types.int32,
        types.int64,
    ),
)


_arrow_compute_cmp_decimal_float = types.ExternalFunction(
    "arrow_compute_cmp_decimal_float_py_entry",
    types.bool_(
        types.int32,
        int128_type,
        types.int32,
        types.int32,
        types.float64,
    ),
)


_arrow_compute_cmp_decimal_decimal = types.ExternalFunction(
    "arrow_compute_cmp_decimal_decimal_py_entry",
    types.bool_(
        types.int32,
        int128_type,
        types.int32,
        types.int32,
        types.int32,
        types.int32,
        int128_type,
    ),
)


def decimal_create_cmp_op_overload(op):
    """create overload function for comparison operators with datetime_date_array"""

    def overload_cmp(lhs, rhs):
        if isinstance(lhs, Decimal128Type) and isinstance(rhs, Decimal128Type):
            op_enum = cmp_op_to_enum[op].value
            precision1 = lhs.precision
            scale1 = lhs.scale
            precision2 = rhs.precision
            scale2 = rhs.scale

            def impl(lhs, rhs):  # pragma: no cover
                out = _arrow_compute_cmp_decimal_decimal(
                    op_enum,
                    decimal128type_to_int128(lhs),
                    precision1,
                    scale1,
                    precision2,
                    scale2,
                    decimal128type_to_int128(rhs),
                )
                bodo.utils.utils.check_and_propagate_cpp_exception()
                return out

            return impl

        elif isinstance(lhs, Decimal128Type) and isinstance(rhs, types.Integer):
            op_enum = cmp_op_to_enum[op].value
            precision = lhs.precision
            scale = lhs.scale

            def impl(lhs, rhs):  # pragma: no cover
                out = _arrow_compute_cmp_decimal_int(
                    op_enum, decimal128type_to_int128(lhs), precision, scale, rhs
                )
                bodo.utils.utils.check_and_propagate_cpp_exception()
                return out

            return impl

        elif isinstance(lhs, types.Integer) and isinstance(rhs, Decimal128Type):
            op_enum = cmp_op_to_enum[op].value
            precision = rhs.precision
            scale = rhs.scale
            op_enum = _reverse_cmp_op[op_enum]

            def impl(lhs, rhs):  # pragma: no cover
                out = _arrow_compute_cmp_decimal_int(
                    op_enum, decimal128type_to_int128(rhs), precision, scale, lhs
                )
                bodo.utils.utils.check_and_propagate_cpp_exception()
                return out

            return impl

        elif isinstance(lhs, Decimal128Type) and isinstance(rhs, types.Float):
            op_enum = cmp_op_to_enum[op].value
            precision = lhs.precision
            scale = lhs.scale

            def impl(lhs, rhs):  # pragma: no cover
                out = _arrow_compute_cmp_decimal_float(
                    op_enum, decimal128type_to_int128(lhs), precision, scale, rhs
                )
                bodo.utils.utils.check_and_propagate_cpp_exception()
                return out

            return impl

        elif isinstance(lhs, types.Float) and isinstance(rhs, Decimal128Type):
            op_enum = cmp_op_to_enum[op].value
            precision = rhs.precision
            scale = rhs.scale
            op_enum = _reverse_cmp_op[op_enum]

            def impl(lhs, rhs):  # pragma: no cover
                out = _arrow_compute_cmp_decimal_float(
                    op_enum, decimal128type_to_int128(rhs), precision, scale, lhs
                )
                bodo.utils.utils.check_and_propagate_cpp_exception()
                return out

            return impl

    return overload_cmp


@lower_constant(Decimal128Type)
def lower_constant_decimal(context, builder, ty, pyval):
    # call a Numba function to unbox and convert to a constant 2-tuple of int64 values
    int64_tuple = numba.njit(lambda v: decimal128type_to_int64_tuple(v))(pyval)
    # pack int64 tuple in LLVM constant
    consts = [
        context.get_constant_generic(builder, types.int64, v) for v in int64_tuple
    ]
    t = cgutils.pack_array(builder, consts)
    # convert int64 tuple to int128 using pointer bitcast
    res = cgutils.alloca_once(builder, lir.IntType(128))
    builder.store(
        t, builder.bitcast(res, lir.ArrayType(lir.IntType(64), 2).as_pointer())
    )
    return builder.load(res)


@overload(Decimal, no_unliteral=True)
def decimal_constructor_overload(value="0", context=None):
    if not is_overload_none(context):  # pragma: no cover
        raise BodoError("decimal.Decimal() context argument not supported yet")

    # [BE-4399]: implement a more efficient constructor for integers/floats that
    # does not need to convert to strings first, and has more precise float support
    if (
        isinstance(value, (types.Number,))
        or is_overload_constant_str(value)
        or value == bodo.string_type
    ):

        def impl(value="0", context=None):  # pragma: no cover
            return str_to_decimal(str(value), 38, 18)

        return impl
    # TODO: Add support for the tuple, and Decimal arguments
    else:
        raise BodoError(
            "decimal.Decimal() value type must be an integer, float or string"
        )


@overload(bool, no_unliteral=True)
def decimal_to_bool(dec):
    """
    Check if the underlying integer value is 0
    """
    if not isinstance(dec, Decimal128Type):  # pragma: no cover
        return

    def impl(dec):  # pragma: no cover
        return bool(decimal128type_to_int128(dec))

    return impl


def decimal_to_float64_codegen(context, builder, signature, args, scale):
    (val,) = args
    scale = context.get_constant(types.int8, scale)

    fnty = lir.FunctionType(
        lir.DoubleType(),
        [
            lir.IntType(128),
            lir.IntType(8),
        ],
    )
    fn = cgutils.get_or_insert_function(builder.module, fnty, name="decimal_to_double")
    ret = builder.call(fn, [val, scale])
    bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
    return ret


@intrinsic(prefer_literal=True)
def decimal_to_float64(typingctx, val_t):
    """convert decimal128 to float"""
    assert isinstance(val_t, Decimal128Type)

    def codegen(context, builder, signature, args):
        return decimal_to_float64_codegen(
            context, builder, signature, args, val_t.scale
        )

    return types.float64(val_t), codegen


@overload(float, no_unliteral=True)
def overload_float_ctor_from_dec(dec):
    """
    Convert a decimal value to a float value
    TODO: Make Numba native for compiler benefits
    """
    if not isinstance(dec, Decimal128Type):  # pragma: no cover
        return

    def impl(dec):  # pragma: no cover
        return decimal_to_float64(dec)

    return impl


@intrinsic
def decimal_to_int64(typingctx, val_t):
    """convert decimal128 to int"""
    assert isinstance(val_t, Decimal128Type), "Decimal128Type expected"

    def codegen(context, builder, sig, args):
        (val,) = args
        precision = context.get_constant(types.int8, sig.args[0].precision)
        scale = context.get_constant(types.int8, sig.args[0].scale)

        fnty = lir.FunctionType(
            lir.IntType(64),
            [
                lir.IntType(128),
                lir.IntType(8),
                lir.IntType(8),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="decimal_to_int64"
        )
        ret = builder.call(fn, [val, precision, scale])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return types.int64(val_t), codegen


@overload(int)
def overload_int_ctor_from_dec(dec):
    """
    Convert a decimal value to an int value
    """
    if not isinstance(dec, Decimal128Type):  # pragma: no cover
        return

    def impl(dec):  # pragma: no cover
        return decimal_to_int64(dec)

    return impl


def to_pa_decimal_scalar(a):
    """convert scalar 'a' to a PyArrow Decimal128Scalar if not already."""
    if isinstance(a, pa.Decimal128Scalar):
        return a

    assert isinstance(a, Decimal), "to_pa_decimal_scalar: Decimal value expected"
    return pa.scalar(a, pa.decimal128(38, 18))


@unbox(Decimal128Type)
def unbox_decimal(typ, val, c):
    """
    Unbox a PyArrow Decimal128Scalar or a decimal.Decimal object into native
    Decimal128Type
    """

    # val = to_pa_decimal_scalar(val)
    to_pa_decimal_scalar_obj = c.pyapi.unserialize(
        c.pyapi.serialize_object(to_pa_decimal_scalar)
    )
    val = c.pyapi.call_function_objargs(to_pa_decimal_scalar_obj, [val])
    c.pyapi.decref(to_pa_decimal_scalar_obj)

    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(8).as_pointer(),
            lir.IntType(128).as_pointer(),
        ],
    )
    fn = cgutils.get_or_insert_function(c.builder.module, fnty, name="unbox_decimal")
    res = cgutils.alloca_once(c.builder, c.context.get_value_type(int128_type))
    c.builder.call(
        fn,
        [val, res],
    )
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    res_ret = c.builder.load(res)

    # decref since val is output of to_pa_decimal_scalar() and not coming from user
    # context
    c.pyapi.decref(val)

    return NativeValue(res_ret, is_error=is_error)


@box(Decimal128Type)
def box_decimal(typ, val, c):
    """Box Decimal128Type to PyArrow Decimal128Scalar"""

    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(),
        [
            lir.IntType(128),
            lir.IntType(8),
            lir.IntType(8),
        ],
    )
    fn = cgutils.get_or_insert_function(c.builder.module, fnty, name="box_decimal")

    precision = c.context.get_constant(types.int8, typ.precision)
    scale = c.context.get_constant(types.int8, typ.scale)

    return c.builder.call(
        fn,
        [val, precision, scale],
    )


@lower_cast(types.Integer, Decimal128Type)
def cast_int_to_decimal(context, builder, fromty, toty, val):
    assert toty.scale == 0, "cast_int_to_decimal: scale 0 expected"
    # Convert int value to int128 using sign extend
    return builder.sext(val, lir.IntType(128))


@lower_cast(Decimal128Type, types.Integer)
def cast_decimal_to_int(context, builder, fromty, toty, val):
    assert fromty.scale == 0, "cast_decimal_to_int: scale 0 expected"
    # Truncate int128 to target integer
    return builder.trunc(val, lir.IntType(types.unliteral(toty).bitwidth))


@overload_method(Decimal128Type, "__hash__", no_unliteral=True)
def decimal_hash(val):  # pragma: no cover
    def impl(val):
        return hash(decimal_to_str(val))

    return impl


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
        return Decimal128Type(self.precision, self.scale)


# store data and nulls as regular numpy arrays without payload machinery
# since this struct is immutable (data and null_bitmap are not assigned new
# arrays after initialization)
# NOTE: storing data as int128 elements. struct of 8 bytes could be better depending on
# the operations needed

data_type = types.Array(int128_type, 1, "C")
nulls_type = types.Array(types.uint8, 1, "C")


@register_model(DecimalArrayType)
class DecimalArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", data_type),
            ("null_bitmap", nulls_type),
        ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DecimalArrayType, "data", "_data")
make_attribute_wrapper(DecimalArrayType, "null_bitmap", "_null_bitmap")


@intrinsic(prefer_literal=True)
def init_decimal_array(typingctx, data, null_bitmap, precision_tp, scale_tp=None):
    """Create a DecimalArray with provided data and null bitmap values."""
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
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], bitmap_val)

        return decimal_arr._getvalue()

    precision = get_overload_const_int(precision_tp)
    scale = get_overload_const_int(scale_tp)
    ret_typ = DecimalArrayType(precision, scale)
    sig = ret_typ(data, null_bitmap, precision_tp, scale_tp)
    return sig, codegen


@lower_constant(DecimalArrayType)
def lower_constant_decimal_arr(context, builder, typ, pyval):
    n = len(pyval)
    n_const = context.get_constant(types.int64, n)
    data_arr_struct = bodo.utils.utils._empty_nd_impl(
        context, builder, types.Array(int128_type, 1, "C"), [n_const]
    )
    nulls_arr = np.empty((n + 7) >> 3, np.uint8)

    def f(arr, idx, val):
        arr[idx] = decimal128type_to_int128(val)

    # TODO: Replace with an implementation that doesn't produce IR for every element of a constant array
    for i, s in enumerate(pyval):
        is_na = pd.isna(s)
        bodo.libs.int_arr_ext.set_bit_to_arr(nulls_arr, i, int(not is_na))
        if not is_na:
            context.compile_internal(
                builder,
                f,
                types.void(
                    types.Array(int128_type, 1, "C"),
                    types.int64,
                    Decimal128Type(typ.precision, typ.scale),
                ),
                [
                    data_arr_struct._getvalue(),
                    context.get_constant(types.int64, i),
                    context.get_constant_generic(
                        builder, Decimal128Type(typ.precision, typ.scale), s
                    ),
                ],
            )

    nulls_const_arr = context.get_constant_generic(builder, nulls_type, nulls_arr)

    decimal_arr = context.make_helper(builder, typ)
    decimal_arr.data = data_arr_struct._getvalue()
    decimal_arr.null_bitmap = nulls_const_arr
    return decimal_arr._getvalue()


# high-level allocation function for decimal arrays
@numba.njit(no_cpython_wrapper=True)
def alloc_decimal_array(n, precision, scale):  # pragma: no cover
    data_arr = np.empty(n, dtype=int128_type)
    nulls = np.empty((n + 7) >> 3, dtype=np.uint8)
    return init_decimal_array(data_arr, nulls, precision, scale)


def alloc_decimal_array_equiv(self, scope, equiv_set, loc, args, kws):
    """Array analysis function for alloc_decimal_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 3 and not kws
    return ArrayAnalysis.AnalyzeResult(shape=args[0], pre=[])


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
    fn_get = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="box_decimal_array"
    )
    obj_arr = c.builder.call(
        fn_get,
        [
            n,
            data_arr.data,
            bitmap_arr_data,
            scale,
        ],
    )

    c.context.nrt.decref(c.builder, typ, val)
    return obj_arr


@unbox(DecimalArrayType)
def unbox_decimal_arr(typ, val, c):
    """
    Unbox a numpy array with Decimal objects into native DecimalArray
    """
    decimal_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    # allocate data and null_bitmap arrays
    n_obj = c.pyapi.call_method(val, "__len__", ())
    n = c.pyapi.long_as_longlong(n_obj)
    c.pyapi.decref(n_obj)

    n_bitmask_bytes = c.builder.udiv(
        c.builder.add(n, lir.Constant(lir.IntType(64), 7)),
        lir.Constant(lir.IntType(64), 8),
    )
    data_arr_struct = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(int128_type, 1, "C"), [n]
    )
    bitmap_arr_struct = bodo.utils.utils._empty_nd_impl(
        c.context, c.builder, types.Array(types.uint8, 1, "C"), [n_bitmask_bytes]
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
    fn = cgutils.get_or_insert_function(
        c.builder.module, fnty, name="unbox_decimal_array"
    )
    c.builder.call(
        fn,
        [
            val,
            n,
            data_arr_struct.data,
            bitmap_arr_struct.data,
        ],
    )

    decimal_arr.null_bitmap = bitmap_arr_struct._getvalue()
    decimal_arr.data = data_arr_struct._getvalue()

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(decimal_arr._getvalue(), is_error=is_error)


@overload_method(DecimalArrayType, "copy", no_unliteral=True)
def overload_decimal_arr_copy(A):
    precision = A.precision
    scale = A.scale
    return lambda A: bodo.libs.decimal_arr_ext.init_decimal_array(
        A._data.copy(),
        A._null_bitmap.copy(),
        precision,
        scale,
    )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_decimal_arr_len(A):
    if isinstance(A, DecimalArrayType):
        return lambda A: len(A._data)  # pragma: no cover


@overload_attribute(DecimalArrayType, "shape")
def overload_decimal_arr_shape(A):
    return lambda A: (len(A._data),)  # pragma: no cover


@overload_attribute(DecimalArrayType, "dtype")
def overload_decimal_arr_dtype(A):
    return lambda A: np.object_  # pragma: no cover


@overload_attribute(DecimalArrayType, "ndim")
def overload_decimal_arr_ndim(A):
    return lambda A: 1  # pragma: no cover


@overload_attribute(DecimalArrayType, "nbytes")
def decimal_arr_nbytes_overload(A):
    return lambda A: A._data.nbytes + A._null_bitmap.nbytes  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def decimal_arr_setitem(A, idx, val):
    if not isinstance(A, DecimalArrayType):
        return

    if val == types.none or isinstance(val, types.optional):  # pragma: no cover
        # None/Optional goes through a separate step.
        return

    typ_err_msg = f"setitem for DecimalArray with indexing type {idx} received an incorrect 'value' type {val}."

    # scalar case
    if isinstance(idx, types.Integer):
        # This is the existing type check
        if isinstance(val, Decimal128Type):

            def impl_scalar(A, idx, val):  # pragma: no cover
                A._data[idx] = decimal128type_to_int128(val)
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            # Covered by test_series_iat_setitem , test_series_iloc_setitem_int , test_series_setitem_int
            return impl_scalar
        else:
            # [BE-4399] make a more efficient way to insert integers/floats into
            # a Decimal array
            def impl_scalar(A, idx, val):  # pragma: no cover
                A._data[idx] = decimal128type_to_int128(
                    str_to_decimal(str(val), 38, 18)
                )
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

        return impl_scalar

    if not (
        (is_iterable_type(val) and isinstance(val.dtype, bodo.Decimal128Type))
        or isinstance(val, Decimal128Type)
    ):
        raise BodoError(typ_err_msg)

    # index is integer array/list
    if is_list_like_index_type(idx) and isinstance(idx.dtype, types.Integer):
        if isinstance(val, Decimal128Type):
            return lambda A, idx, val: array_setitem_int_index(
                A, idx, decimal128type_to_int128(val)
            )  # pragma: no cover

        def impl_arr_ind_mask(A, idx, val):  # pragma: no cover
            array_setitem_int_index(A, idx, val)

        # covered by test_series_iloc_setitem_list_int
        return impl_arr_ind_mask

    # bool array
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        if isinstance(val, Decimal128Type):
            return lambda A, idx, val: array_setitem_bool_index(
                A, idx, decimal128type_to_int128(val)
            )  # pragma: no cover

        def impl_bool_ind_mask(A, idx, val):  # pragma: no cover
            array_setitem_bool_index(A, idx, val)

        return impl_bool_ind_mask

    # slice case
    if isinstance(idx, types.SliceType):
        if isinstance(val, Decimal128Type):
            return lambda A, idx, val: array_setitem_slice_index(
                A, idx, decimal128type_to_int128(val)
            )  # pragma: no cover

        def impl_slice_mask(A, idx, val):  # pragma: no cover
            array_setitem_slice_index(A, idx, val)

        # covered by test_series_setitem_slice
        return impl_slice_mask

    # This should be the only DecimalArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"setitem for DecimalArray with indexing type {idx} not supported."
    )  # pragma: no cover


@overload(operator.getitem, no_unliteral=True)
def decimal_arr_getitem(A, ind):
    if not isinstance(A, DecimalArrayType):
        return

    # covered by test_series_iat_getitem , test_series_iloc_getitem_int
    if isinstance(ind, types.Integer):
        precision = A.precision
        scale = A.scale
        # XXX: cannot handle NA for scalar getitem since not type stable
        return lambda A, ind: int128_to_decimal128type(A._data[ind], precision, scale)

    # bool arr indexing.
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        precision = A.precision
        scale = A.scale

        def impl(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_bool_index(A, ind)
            return init_decimal_array(new_data, new_mask, precision, scale)

        return impl

    # int arr indexing
    if is_list_like_index_type(ind) and isinstance(ind.dtype, types.Integer):
        precision = A.precision
        scale = A.scale

        def impl(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_int_index(A, ind)
            return init_decimal_array(new_data, new_mask, precision, scale)

        return impl

    # slice case
    if isinstance(ind, types.SliceType):
        precision = A.precision
        scale = A.scale

        def impl_slice(A, ind):  # pragma: no cover
            new_data, new_mask = array_getitem_slice_index(A, ind)
            return init_decimal_array(new_data, new_mask, precision, scale)

        return impl_slice

    # This should be the only DecimalArray implementation.
    # We only expect to reach this case if more idx options are added.
    raise BodoError(
        f"getitem for DecimalArray with indexing type {ind} not supported."
    )  # pragma: no cover


@overload(operator.setitem, no_unliteral=True)
def np_arr_setitem_decimal(A, idx, val):
    """Make sure decimal scalar can be stored in int/float arrays by casting to
    int/float
    """
    if not (
        isinstance(A, types.Array)
        and isinstance(idx, types.Integer)
        and isinstance(val, Decimal128Type)
    ):
        return

    if isinstance(A.dtype, types.Float):

        def impl_decimal_setitem_float(A, idx, val):  # pragma: no cover
            A[idx] = float(val)

        return impl_decimal_setitem_float

    if isinstance(A.dtype, types.Integer):

        def impl_decimal_setitem_int(A, idx, val):  # pragma: no cover
            A[idx] = int(val)

        return impl_decimal_setitem_int

    raise BodoError(
        f"setitem for array type {A} with indexing type {idx} and scalar type {val} not supported."
    )  # pragma: no cover


####################### cmp operators ###############################


# int values designating cmp operators to pass to C++
# XXX: these are defined in _decimal_ext.cpp and must match here
class CmpOpEnum(Enum):
    LT = 0
    LE = 1
    EQ = 2
    NE = 3
    GT = 4
    GE = 5


# Reverse of cmp operators to use when we switch arguments (e.g. a <= b to b >= a)
_reverse_cmp_op = {
    CmpOpEnum.LT.value: CmpOpEnum.GT.value,
    CmpOpEnum.LE.value: CmpOpEnum.GE.value,
    CmpOpEnum.EQ.value: CmpOpEnum.EQ.value,
    CmpOpEnum.NE.value: CmpOpEnum.NE.value,
    CmpOpEnum.GT.value: CmpOpEnum.LT.value,
    CmpOpEnum.GE.value: CmpOpEnum.LE.value,
}


cmp_op_to_enum = {
    operator.lt: CmpOpEnum.LT,
    operator.le: CmpOpEnum.LE,
    operator.eq: CmpOpEnum.EQ,
    operator.ne: CmpOpEnum.NE,
    operator.gt: CmpOpEnum.GT,
    operator.ge: CmpOpEnum.GE,
}


def array_or_scalar_to_info(a):  # pragma: no cover
    pass


@overload(array_or_scalar_to_info)
def overload_array_or_scalar_to_info(a):
    """Returns array_info for array or scalar (converted to array) input, and a flag
    indicating that input was scalar.
    """
    from bodo.libs.array import array_to_info

    if bodo.utils.utils.is_array_typ(a, False):
        return lambda a: (array_to_info(a), False)  # pragma: no cover

    assert is_scalar_type(
        a
    ), f"array_or_scalar_to_info: scalar type expected but input is {a}"

    return lambda a: (
        array_to_info(bodo.utils.conversion.coerce_to_array(a, True, True, 1, False)),
        True,
    )  # pragma: no cover


def call_arrow_compute_cmp(op, lhs, rhs):
    """Create an implementation that calls Arrow compute for comparison
    operator op with input types lhs and rhs
    """
    from bodo.libs.array import (
        array_info_type,
        delete_info,
        info_to_array,
    )

    _arrow_compute_cmp = types.ExternalFunction(
        "arrow_compute_cmp_py_entry",
        array_info_type(
            types.int32,
            array_info_type,
            array_info_type,
            types.bool_,
            types.bool_,
        ),
    )

    op_enum = cmp_op_to_enum[op].value
    out_array_type = bodo.boolean_array_type

    def impl_pc_binop(lhs, rhs):  # pragma: no cover
        # For simplicity, convert scalar inputs to arrays and pass a flag to C++ to
        # convert back to scalars
        lhs, is_scalar_lhs = array_or_scalar_to_info(lhs)
        rhs, is_scalar_rhs = array_or_scalar_to_info(rhs)
        out_info = _arrow_compute_cmp(op_enum, lhs, rhs, is_scalar_lhs, is_scalar_rhs)
        bodo.utils.utils.check_and_propagate_cpp_exception()
        out_arr = info_to_array(out_info, out_array_type)
        delete_info(out_info)
        return out_arr

    return impl_pc_binop


def create_cmp_op_overload(op):
    """Creates an overload function (not implementation) for comparison operator op
    that handles decimal array input(s).
    """

    def overload_decimal_op(lhs, rhs):
        if isinstance(lhs, DecimalArrayType) or isinstance(rhs, DecimalArrayType):
            allowed_types = (
                DecimalArrayType,
                bodo.IntegerArrayType,
                bodo.FloatingArrayType,
                types.Array,
                types.Integer,
                types.Float,
                Decimal128Type,
            )
            # TODO[BSE-2502]: support other types
            if not isinstance(lhs, allowed_types) or not isinstance(rhs, allowed_types):
                raise BodoError(f"Invalid decimal comparison with {lhs} and {rhs}")
            return call_arrow_compute_cmp(op, lhs, rhs)

    return overload_decimal_op


cmp_ops = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
]


def _install_cmp_ops():
    """Install overloads for comparison operators"""
    for op in cmp_ops:
        overload_impl = create_cmp_op_overload(op)
        overload(op)(overload_impl)


_install_cmp_ops()
