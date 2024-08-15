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
from bodo.utils.typing import (
    assert_bodo_error,
    get_overload_const_bool,
    is_overload_constant_bool,
    raise_bodo_error,
    unwrap_typeref,
)

ll.add_symbol("unbox_decimal", decimal_ext.unbox_decimal)
ll.add_symbol("box_decimal", decimal_ext.box_decimal)
ll.add_symbol("decimal_to_str", decimal_ext.decimal_to_str)
ll.add_symbol(
    "str_to_decimal_scalar_py_entry", decimal_ext.str_to_decimal_scalar_py_entry
)
ll.add_symbol(
    "str_to_decimal_array_py_entry", decimal_ext.str_to_decimal_array_py_entry
)
ll.add_symbol("decimal_to_double", decimal_ext.decimal_to_double_py_entry)
ll.add_symbol("decimal_arr_to_double", decimal_ext.decimal_arr_to_double_py_entry)
ll.add_symbol("decimal_to_int64", decimal_ext.decimal_to_int64_py_entry)
ll.add_symbol("int_to_decimal_array", decimal_ext.int_to_decimal_array)
ll.add_symbol(
    "cast_float_to_decimal_scalar", decimal_ext.cast_float_to_decimal_scalar_py_entry
)
ll.add_symbol(
    "cast_float_to_decimal_array", decimal_ext.cast_float_to_decimal_array_py_entry
)

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
ll.add_symbol(
    "cast_decimal_to_decimal_scalar_safe",
    decimal_ext.cast_decimal_to_decimal_scalar_safe_py_entry,
)
ll.add_symbol(
    "cast_decimal_to_decimal_scalar_unsafe",
    decimal_ext.cast_decimal_to_decimal_scalar_unsafe_py_entry,
)
ll.add_symbol(
    "cast_decimal_to_decimal_array_safe",
    decimal_ext.cast_decimal_to_decimal_array_safe_py_entry,
)
ll.add_symbol(
    "cast_decimal_to_decimal_array_unsafe",
    decimal_ext.cast_decimal_to_decimal_array_unsafe_py_entry,
)
ll.add_symbol(
    "decimal_scalar_sign",
    decimal_ext.decimal_scalar_sign_py_entry,
)
ll.add_symbol(
    "decimal_array_sign",
    decimal_ext.decimal_array_sign_py_entry,
)
ll.add_symbol(
    "sum_decimal_array",
    decimal_ext.sum_decimal_array_py_entry,
)
ll.add_symbol(
    "add_or_subtract_decimal_scalars",
    decimal_ext.add_or_subtract_decimal_scalars_py_entry,
)
ll.add_symbol(
    "add_or_subtract_decimal_arrays",
    decimal_ext.add_or_subtract_decimal_arrays_py_entry,
)
ll.add_symbol(
    "multiply_decimal_scalars",
    decimal_ext.multiply_decimal_scalars_py_entry,
)
ll.add_symbol(
    "multiply_decimal_arrays",
    decimal_ext.multiply_decimal_arrays_py_entry,
)
ll.add_symbol(
    "modulo_decimal_scalars",
    decimal_ext.modulo_decimal_scalars_py_entry,
)
ll.add_symbol(
    "modulo_decimal_arrays",
    decimal_ext.modulo_decimal_arrays_py_entry,
)
ll.add_symbol(
    "divide_decimal_scalars",
    decimal_ext.divide_decimal_scalars_py_entry,
)
ll.add_symbol(
    "divide_decimal_arrays",
    decimal_ext.divide_decimal_arrays_py_entry,
)
ll.add_symbol(
    "round_decimal_array",
    decimal_ext.round_decimal_array_py_entry,
)
ll.add_symbol(
    "round_decimal_scalar",
    decimal_ext.round_decimal_scalar_py_entry,
)
ll.add_symbol(
    "ceil_floor_decimal_array",
    decimal_ext.ceil_floor_decimal_array_py_entry,
)
ll.add_symbol(
    "ceil_floor_decimal_scalar",
    decimal_ext.ceil_floor_decimal_scalar_py_entry,
)
ll.add_symbol(
    "trunc_decimal_array",
    decimal_ext.trunc_decimal_array_py_entry,
)
ll.add_symbol(
    "trunc_decimal_scalar",
    decimal_ext.trunc_decimal_scalar_py_entry,
)
ll.add_symbol(
    "abs_decimal_scalar",
    decimal_ext.abs_decimal_scalar_py_entry,
)
ll.add_symbol(
    "abs_decimal_array",
    decimal_ext.abs_decimal_array_py_entry,
)
ll.add_symbol(
    "factorial_decimal_scalar",
    decimal_ext.factorial_decimal_scalar_py_entry,
)
ll.add_symbol(
    "factorial_decimal_array",
    decimal_ext.factorial_decimal_array_py_entry,
)


ll.add_symbol(
    "decimal_array_to_str_array",
    decimal_ext.decimal_array_to_str_array_py_entry,
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
    types.int8: 3,
    types.int16: 5,
    types.int32: 10,
    types.int64: 19,
    types.uint8: 3,
    types.uint16: 5,
    types.uint32: 10,
    types.uint64: 20,
}

DECIMAL128_MAX_PRECISION = 38


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
            if int_to_decimal_precision[other] > self.precision:
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
def int128_to_decimal128type(typingctx, val, precision_tp, scale_tp):
    """cast int128 to decimal128"""
    assert val == int128_type
    assert_bodo_error(is_overload_constant_int(precision_tp))
    assert_bodo_error(is_overload_constant_int(scale_tp))

    def codegen(context, builder, signature, args):
        return args[0]

    precision = get_overload_const_int(precision_tp)
    scale = get_overload_const_int(scale_tp)
    return (
        Decimal128Type(precision, scale)(int128_type, precision_tp, scale_tp),
        codegen,
    )


@intrinsic
def decimal128type_to_int128(typingctx, val):
    """cast int128 to decimal128"""
    assert isinstance(val, Decimal128Type)

    def codegen(context, builder, signature, args):
        return args[0]

    return int128_type(val), codegen


def decimal_to_str_codegen(context, builder, signature, args, scale):
    (val,) = args
    scale = context.get_constant(types.int32, scale)
    remove_trailing_zeros = context.get_constant(types.bool_, 1)
    uni_str = cgutils.create_struct_proxy(types.unicode_type)(context, builder)

    fnty = lir.FunctionType(
        lir.VoidType(),
        [
            lir.IntType(128),
            lir.IntType(8).as_pointer().as_pointer(),
            lir.IntType(64).as_pointer(),
            lir.IntType(32),
            lir.IntType(1),
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
            remove_trailing_zeros,
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


@intrinsic
def decimal_to_str(typingctx, val_t=None):
    """convert decimal128 to string"""
    assert isinstance(val_t, Decimal128Type)

    def codegen(context, builder, signature, args):
        return decimal_to_str_codegen(context, builder, signature, args, val_t.scale)

    return bodo.string_type(val_t), codegen


@intrinsic(prefer_literal=True)
def _str_to_decimal_scalar(typingctx, val, precision_tp, scale_tp):
    """convert string to decimal128. This returns a tuple of
    (Decimal128Type, bool) where the bool indicates if the value
    errored in parsing or fitting in the final decimal value."""
    assert val == bodo.string_type or is_overload_constant_str(val)
    assert_bodo_error(is_overload_constant_int(precision_tp))
    assert_bodo_error(is_overload_constant_int(scale_tp))

    def codegen(context, builder, signature, args):
        val, precision, scale = args
        val = bodo.libs.str_ext.gen_unicode_to_std_str(context, builder, val)
        error_ptr = cgutils.alloca_once(builder, lir.IntType(1))
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="str_to_decimal_scalar_py_entry"
        )
        decimal_val = builder.call(
            fn,
            [
                val,
                precision,
                scale,
                error_ptr,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        errors = builder.load(error_ptr)
        return context.make_tuple(builder, signature.return_type, [decimal_val, errors])

    precision = get_overload_const_int(precision_tp)
    scale = get_overload_const_int(scale_tp)
    decimal_type = Decimal128Type(precision, scale)
    return_type = types.Tuple((decimal_type, types.bool_))
    return return_type(val, precision_tp, scale_tp), codegen


def str_to_decimal_scalar(val, precision, scale, null_on_error):
    pass


@overload(str_to_decimal_scalar, prefer_literal=True)
def overload_str_to_decimal_scalar(val, precision, scale, null_on_error):
    if (
        not is_overload_constant_int(precision)
        or not is_overload_constant_int(scale)
        or not is_overload_constant_bool(null_on_error)
    ):
        raise_bodo_error(
            "str_to_decimal_scalar: constant precision, scale, and null_on_error expected."
        )
    raise_exception = not get_overload_const_bool(null_on_error)
    if raise_exception:

        def impl(val, precision, scale, null_on_error):  # pragma: no cover
            val, overflow = _str_to_decimal_scalar(str(val), precision, scale)
            if overflow:
                raise RuntimeError(
                    "String value is out of range for decimal or doesn't parse properly"
                )
            return val

        return impl

    else:

        def impl(val, precision, scale, null_on_error):  # pragma: no cover
            val, overflow = _str_to_decimal_scalar(str(val), precision, scale)
            if overflow:
                return None
            return val

        return impl


@intrinsic
def _str_to_decimal_array(typingctx, val_tp, precision_tp, scale_tp, null_on_error_tp):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="str_to_decimal_array_py_entry"
        )
        ret = builder.call(fn, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return array_info_type(val_tp, precision_tp, scale_tp, null_on_error_tp), codegen


def str_to_decimal_array(arr, precision, scale, null_on_error):
    pass


@overload(str_to_decimal_array, prefer_literal=True)
def overload_str_to_decimal_array(arr, precision, scale, null_on_error):
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    if not is_overload_constant_int(precision) or not is_overload_constant_int(scale):
        raise_bodo_error("str_to_decimal_array: constant precision and scale expected.")

    _precision = get_overload_const_int(precision)
    _scale = get_overload_const_int(scale)
    _output_type = DecimalArrayType(_precision, _scale)

    def impl(arr, precision, scale, null_on_error):  # pragma: no cover
        input_info = array_to_info(arr)
        out_info = _str_to_decimal_array(input_info, precision, scale, null_on_error)
        out_arr = info_to_array(out_info, _output_type)
        delete_info(out_info)
        return out_arr

    return impl


def decimal_array_to_str_array(arr):
    pass


@overload(decimal_array_to_str_array)
def overload_decimal_array_to_str_array(arr):
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    def impl(arr):  # pragma: no cover
        input_info = array_to_info(arr)
        out_info = _decimal_array_to_str_array(input_info)
        out_arr = info_to_array(out_info, bodo.string_array_type)
        delete_info(out_info)
        return out_arr

    return impl


@intrinsic
def _decimal_array_to_str_array(typingctx, arr_t):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="decimal_array_to_str_array"
        )
        ret = builder.call(fn, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (array_info_type(arr_t), codegen)


def decimal_scalar_to_str(arr):
    """
    Converts a decimal scalar to a SNOWFLAKE-style string,
    preserving trailing zeros to fit the scale.
    """
    pass


@overload(decimal_scalar_to_str)
def overload_decimal_scalar_to_str(arr):
    def impl(arr):  # pragma: no cover
        out = _decimal_scalar_to_str(arr)
        return out

    return impl


@intrinsic
def _decimal_scalar_to_str(typingctx, arr_t):
    def codegen(context, builder, signature, args):
        (val,) = args
        scale = context.get_constant(types.int32, arr_t.scale)
        remove_trailing_zeros = context.get_constant(types.bool_, 0)

        uni_str = cgutils.create_struct_proxy(types.unicode_type)(context, builder)

        fnty = lir.FunctionType(
            lir.VoidType(),
            [
                lir.IntType(128),
                lir.IntType(8).as_pointer().as_pointer(),
                lir.IntType(64).as_pointer(),
                lir.IntType(32),
                lir.IntType(1),
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
                remove_trailing_zeros,
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

    return bodo.string_type(arr_t), codegen


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


@intrinsic
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

    if is_overload_constant_str(value) or value == bodo.string_type:

        def impl(value="0", context=None):  # pragma: no cover
            return str_to_decimal_scalar(value, 38, 18, False)

        return impl
    elif isinstance(value, types.Integer):

        def impl(value="0", context=None):  # pragma: no cover
            # TODO: Assign a scale + precision based on the integer type.
            decimal_int = int_to_decimal_scalar(value)
            return _cast_decimal_to_decimal_scalar_unsafe(decimal_int, 38, 18)

        return impl
    elif isinstance(value, types.Float):

        def impl(value="0", context=None):  # pragma: no cover
            return float_to_decimal_scalar(value, 38, 18, False)

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


@intrinsic
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


def decimal_arr_to_float64(arr):
    pass


@overload(decimal_arr_to_float64)
def overload_decimal_arr_to_float64(arr):
    """
    Convert a decimal array to a float array
    """
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    assert isinstance(
        arr, DecimalArrayType
    ), "decimal_arr_to_float64: decimal array expected"

    output_arr_type = bodo.FloatingArrayType(types.float64)

    def impl(arr):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_info = _decimal_arr_to_float64(arr_info)
        out_arr = info_to_array(out_info, output_arr_type)
        delete_info(out_info)
        return out_arr

    return impl


@intrinsic
def _decimal_arr_to_float64(typingctx, val_t):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="decimal_arr_to_double"
        )
        ret = builder.call(fn, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return array_info_type(val_t), codegen


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


def validate_decimal_arguments(fname, precision_type, scale_type):
    if not is_overload_constant_int(precision_type):
        raise_bodo_error(f"{fname}: constant new_precision expected")
    if not is_overload_constant_int(scale_type):
        raise_bodo_error(f"{fname}: constant new_scale expected")
    precision = get_overload_const_int(precision_type)
    scale = get_overload_const_int(scale_type)
    assert precision <= 38, f"{fname}: precision <= 38 expected, {precision} provided "
    assert scale <= 37, f"{fname}: scale <= 37 expected, {scale} provided"
    return precision, scale


def cast_decimal_to_decimal_array(
    val, new_precision, new_scale, null_on_error
):  # pragma: no cover
    pass


@overload(cast_decimal_to_decimal_array, prefer_literal=True)
def overload_cast_decimal_to_decimal_array(
    val, new_precision, new_scale, null_on_error
):
    """
    Converts a decimal whose leading digits are assumed to fully fit inside
    the new precision and scale and returns a new decimal with the new precision
    and scale after rescaling the decimal.
    """
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    precision, scale = validate_decimal_arguments(
        "cast_decimal_to_decimal_array", new_precision, new_scale
    )

    old_leading_digits = val.precision - val.scale
    new_leading_digits = precision - scale
    check_leading_digits = old_leading_digits > new_leading_digits

    input_type = val
    output_type = DecimalArrayType(precision, scale)
    if check_leading_digits:

        def impl(val, new_precision, new_scale, null_on_error):  # pragma: no cover
            input_info = array_to_info(val)
            out_info = _cast_decimal_to_decimal_array_safe(
                input_info, input_type, output_type, null_on_error
            )
            out_arr = info_to_array(out_info, output_type)
            delete_info(out_info)
            return out_arr

        return impl
    else:
        # Avoid checking leading digits if we know they fit
        def impl(val, new_precision, new_scale, null_on_error):  # pragma: no cover
            input_info = array_to_info(val)
            out_info = _cast_decimal_to_decimal_array_unsafe(
                input_info, input_type, output_type
            )
            out_arr = info_to_array(out_info, output_type)
            delete_info(out_info)
            return out_arr

        return impl


@intrinsic
def _cast_decimal_to_decimal_array_unsafe(
    typingctx, val_t, input_type_t, output_type_t
):
    from bodo.libs.array import array_info_type

    input_type = unwrap_typeref(input_type_t)
    output_type = unwrap_typeref(output_type_t)
    shift_amount = output_type.scale - input_type.scale

    def codegen(context, builder, signature, args):
        val, _, _ = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
            ],
        )
        scale_amount_const = context.get_constant(types.int64, shift_amount)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="cast_decimal_to_decimal_array_unsafe"
        )
        ret = builder.call(fn, [val, scale_amount_const])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return array_info_type(val_t, input_type_t, output_type_t), codegen


@intrinsic
def _cast_decimal_to_decimal_array_safe(
    typingctx, val_t, input_type_t, output_type_t, null_on_error_t
):
    from bodo.libs.array import array_info_type

    input_type = unwrap_typeref(input_type_t)
    output_type = unwrap_typeref(output_type_t)
    shift_amount = output_type.scale - input_type.scale
    new_leading_digits = output_type.precision - output_type.scale
    # In the safe path we need to check that we aren't truncating the
    # leading digits of the input decimal. This can occur if we increase
    # the scale of the decimal or decrease the precision.
    # For example, Decimal(38, 2) -> Decimal(38, 4) goes from 36 to 34 leading
    # digits. To do this we can check that the actual decimal value would fit
    # inside the new location by confirming its less than 10^(old_scale + new_leading_digits)
    max_allowed_input_precision = input_type.scale + new_leading_digits

    def codegen(context, builder, signature, args):
        val, _, _, null_on_error = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        scale_amount_const = context.get_constant(types.int64, shift_amount)
        max_allowed_input_precision_const = context.get_constant(
            types.int64, max_allowed_input_precision
        )

        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="cast_decimal_to_decimal_array_safe"
        )
        ret = builder.call(
            fn,
            [val, scale_amount_const, max_allowed_input_precision_const, null_on_error],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return array_info_type(val_t, input_type_t, output_type_t, null_on_error_t), codegen


def cast_decimal_to_decimal_scalar(
    val, new_precision, new_scale, null_on_error
):  # pragma: no cover
    pass


@overload(cast_decimal_to_decimal_scalar, prefer_literal=True)
def overload_cast_decimal_to_decimal_scalar(
    val, new_precision, new_scale, null_on_error
):
    """
    Converts a decimal whose leading digits are assumed to fully fit inside
    the new precision and scale and returns a new decimal with the new precision
    and scale after rescaling the decimal.
    """
    precision, scale = validate_decimal_arguments(
        "cast_decimal_to_decimal_scalar", new_precision, new_scale
    )

    old_leading_digits = val.precision - val.scale
    new_leading_digits = precision - scale
    check_leading_digits = old_leading_digits > new_leading_digits

    if check_leading_digits:

        def impl(val, new_precision, new_scale, null_on_error):  # pragma: no cover
            value, safe = _cast_decimal_to_decimal_scalar_safe(
                val, new_precision, new_scale
            )
            # Note: We evaluate this in Python since there isn't a great way to have the intrinsic
            # sometimes return NULL.
            if not safe:
                if null_on_error:
                    return None
                else:
                    raise ValueError("Number out of representable range")
            else:
                return value

        return impl
    else:
        # Avoid checking leading digits if we know they fit
        def impl(val, new_precision, new_scale, null_on_error):  # pragma: no cover
            return _cast_decimal_to_decimal_scalar_unsafe(val, new_precision, new_scale)

        return impl


@intrinsic(prefer_literal=True)
def _cast_decimal_to_decimal_scalar_unsafe(typingctx, val_t, precision_t, scale_t):
    """
    Cast a Decimal128 value to a new Decimal 128 value with a new precision and scale.
    """

    assert isinstance(val_t, Decimal128Type)
    assert_bodo_error(is_overload_constant_int(precision_t))
    assert_bodo_error(is_overload_constant_int(scale_t))
    precision = get_overload_const_int(precision_t)
    scale = get_overload_const_int(scale_t)
    shift_amount = scale - val_t.scale

    def codegen(context, builder, signature, args):
        val, _, _ = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
            ],
        )
        scale_amount_const = context.get_constant(types.int64, shift_amount)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="cast_decimal_to_decimal_scalar_unsafe"
        )
        ret = builder.call(fn, [val, scale_amount_const])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    decimal_type = Decimal128Type(precision, scale)
    return decimal_type(val_t, precision_t, scale_t), codegen


@intrinsic(prefer_literal=True)
def _cast_decimal_to_decimal_scalar_safe(typingctx, val_t, precision_t, scale_t):
    """
    Cast a Decimal128 value to a new Decimal 128 value with a new precision and scale.
    """

    assert isinstance(val_t, Decimal128Type)
    assert_bodo_error(is_overload_constant_int(precision_t))
    assert_bodo_error(is_overload_constant_int(scale_t))
    scale = get_overload_const_int(scale_t)
    precision = get_overload_const_int(precision_t)
    shift_amount = scale - val_t.scale
    new_leading_digits = precision - scale
    n = val_t.scale + new_leading_digits

    def codegen(context, builder, signature, args):
        val, _, _ = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1).as_pointer(),
            ],
        )
        scale_amount_const = context.get_constant(types.int64, shift_amount)
        n_const = context.get_constant(types.int64, n)
        safe_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="cast_decimal_to_decimal_scalar_safe"
        )
        ret = builder.call(fn, [val, scale_amount_const, n_const, safe_pointer])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        safe = builder.load(safe_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, safe])

    decimal_type = Decimal128Type(precision, scale)
    ret_type = types.Tuple([decimal_type, types.bool_])
    return ret_type(val_t, precision_t, scale_t), codegen


def float_to_decimal_array(val, precision, scale, null_on_error):  # pragma: no cover
    pass


@overload(float_to_decimal_array, prefer_literal=True)
def overload_float_to_decimal_array(val, precision, scale, null_on_error):
    """
    Converts a decimal whose leading digits are assumed to fully fit inside
    the new precision and scale and returns a new decimal with the new precision
    and scale after rescaling the decimal.
    """
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    precision, scale = validate_decimal_arguments(
        "cast_float_to_decimal_array", precision, scale
    )

    output_type = DecimalArrayType(precision, scale)

    def impl(val, precision, scale, null_on_error):  # pragma: no cover
        input_info = array_to_info(val)
        out_info = _cast_float_to_decimal_array(
            input_info, np.int32(precision), np.int32(scale), null_on_error
        )
        out_arr = info_to_array(out_info, output_type)
        delete_info(out_info)
        return out_arr

    return impl


@intrinsic
def _cast_float_to_decimal_array(
    typingctx, val_t, scale_t, precision_t, null_on_error_t
):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(1),
            ],
        )

        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="cast_float_to_decimal_array"
        )
        ret = builder.call(fn, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return array_info_type(val_t, scale_t, precision_t, null_on_error_t), codegen


def float_to_decimal_scalar(val, precision, scale, null_on_error):  # pragma: no cover
    pass


@overload(float_to_decimal_scalar, prefer_literal=True)
def overload_float_to_decimal_scalar(val, precision, scale, null_on_error):
    """
    Converts a float to a decimal with the specified precision/scale. If this is impossible
    due to the magnitude of the value versus the precision/scale, either returns null
    or throws an error (depending on null_on_error).
    """
    validate_decimal_arguments("float_to_decimal", precision, scale)

    def impl(val, precision, scale, null_on_error):  # pragma: no cover
        value, safe = _cast_float_to_decimal_scalar(np.float64(val), precision, scale)
        # Note: We evaluate this in Python since there isn't a great way to have the intrinsic
        # sometimes return NULL.
        if not safe:
            if null_on_error:
                return None
            else:
                raise ValueError("Number out of representable range")
        else:
            return value

    return impl


@intrinsic(prefer_literal=True)
def _cast_float_to_decimal_scalar(typingctx, val_t, precision_t, scale_t):
    """
    Cast a float64 value to a new Decimal 128 value with a new precision and scale.
    """

    assert val_t == types.float64
    assert_bodo_error(is_overload_constant_int(precision_t))
    assert_bodo_error(is_overload_constant_int(scale_t))
    scale = get_overload_const_int(scale_t)
    precision = get_overload_const_int(precision_t)

    def codegen(context, builder, signature, args):
        val, prec, scale = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.DoubleType(),
                lir.IntType(32),
                lir.IntType(32),
                lir.IntType(1).as_pointer(),
            ],
        )
        safe_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="cast_float_to_decimal_scalar"
        )
        ret = builder.call(fn, [val, prec, scale, safe_pointer])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        safe = builder.load(safe_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, safe])

    decimal_type = Decimal128Type(precision, scale)
    ret_type = types.Tuple([decimal_type, types.bool_])
    return ret_type(val_t, types.int32, types.int32), codegen


def decimal_scalar_sign(val):  # pragma: no cover
    pass


@overload(decimal_scalar_sign)
def overload_decimal_scalar_sign(val):
    """
    Returns the sign of the decimal scalar. 0 for 0, 1 for positive, -1 for negative.
    """
    assert isinstance(
        val, Decimal128Type
    ), "decimal_scalar_sign: Decimal128Type expected"

    def impl(val):  # pragma: no cover
        return _decimal_scalar_sign(val)

    return impl


@intrinsic
def _decimal_scalar_sign(typingctx, val_t):
    """
    Returns the sign of the decimal scalar. 0 for 0, 1 for positive, -1 for negative.
    """
    assert isinstance(val_t, Decimal128Type), "Decimal128Type expected"

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8),
            [
                lir.IntType(128),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="decimal_scalar_sign"
        )
        ret = builder.call(fn, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return types.int8(val_t), codegen


def decimal_array_sign(arr):  # pragma: no cover
    pass


@overload(decimal_array_sign)
def overload_decimal_array_sign(arr):
    """
    Returns the element-wise signs of the decimal array.
    0 for 0, 1 for positive, -1 for negative.
    """
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    assert isinstance(
        arr, DecimalArrayType
    ), "decimal_array_sign: DecimalArrayType expected"

    output_arr_type = bodo.IntegerArrayType(types.int8)

    def impl(arr):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_info = _decimal_array_sign(arr_info)
        out_arr = info_to_array(out_info, output_arr_type)
        delete_info(out_info)
        return out_arr

    return impl


@intrinsic
def _decimal_array_sign(typingctx, val_t):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="decimal_array_sign"
        )
        ret = builder.call(fn, args)
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return array_info_type(val_t), codegen


@intrinsic(prefer_literal=True)
def _sum_decimal_array(typingctx, arr_t, in_scale_t, parallel_t):
    def codegen(context, builder, signature, args):
        (arr, _, parallel) = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(1).as_pointer(),
                lir.IntType(1),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="sum_decimal_array"
        )
        is_null_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(fn, [arr, is_null_pointer, parallel])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        is_null = builder.load(is_null_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, is_null])

    in_scale = get_overload_const_int(in_scale_t)
    output_decimal_type = Decimal128Type(DECIMAL128_MAX_PRECISION, in_scale)
    ret_type = types.Tuple([output_decimal_type, types.bool_])
    return (ret_type(arr_t, in_scale_t, parallel_t), codegen)


def sum_decimal_array(arr, parallel=False):  # pragma: no cover
    pass


@overload(sum_decimal_array)
def overload_sum_decimal_array(arr, parallel=False):
    """
    Compute the sum of a decimal array.
    """
    from bodo.libs.array import array_to_info

    in_scale = arr.dtype.scale

    def impl(arr, parallel=False):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_scalar, is_null = _sum_decimal_array(arr_info, in_scale, parallel)
        if is_null:
            return None
        return out_scalar

    return impl


def decimal_addition_subtraction_output_precision_scale(p1, s1, p2, s2):
    """
    Calculate the output precision and scale for a addition/subtraction of two decimals.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#addition-and-subtraction
    """
    l1 = p1 - s1
    l2 = p2 - s2
    l = max(l1, l2) + 1
    s = max(s1, s2)
    p = min(l + s, 38)
    return p, s


def add_or_subtract_decimal_scalars(d1, d2, do_addition):  # pragma: no cover
    pass


@overload(add_or_subtract_decimal_scalars)
def overload_add_or_subtract_decimal_scalars(d1, d2, do_addition):
    """
    Add or subtract two decimal scalars together. If overflow occurs
    this raises an exception.

    do_addition should be set to True for addition, False for subtraction.
    """
    if not isinstance(d1, Decimal128Type) or not isinstance(
        d2, Decimal128Type
    ):  # pragma: no cover
        raise BodoError(
            "add_or_subtract_decimal_scalars: Decimal128Type expected for both inputs"
        )

    p, s = decimal_addition_subtraction_output_precision_scale(
        d1.precision, d1.scale, d2.precision, d2.scale
    )

    def impl(d1, d2, do_addition):  # pragma: no cover
        output, overflow = _add_or_subtract_decimal_scalars(d1, d2, p, s, do_addition)
        if overflow:
            raise ValueError("Number out of representable range")
        else:
            return output

    return impl


@intrinsic(prefer_literal=True)
def _add_or_subtract_decimal_scalars(
    typingctx, d1_t, d2_t, precision_t, scale_t, do_addition_t
):
    assert isinstance(d1_t, Decimal128Type)
    assert isinstance(d2_t, Decimal128Type)
    assert_bodo_error(is_overload_constant_int(precision_t))
    assert_bodo_error(is_overload_constant_int(scale_t))
    assert_bodo_error(is_overload_constant_bool(do_addition_t))
    output_precision = get_overload_const_int(precision_t)
    output_scale = get_overload_const_int(scale_t)
    d1_precision = d1_t.precision
    d1_scale = d1_t.scale
    d2_precision = d2_t.precision
    d2_scale = d2_t.scale

    def codegen(context, builder, signature, args):
        d1, d2, output_precision, output_scale, do_addition = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
            ],
        )
        d1_precision_const = context.get_constant(types.int64, d1_precision)
        d1_scale_const = context.get_constant(types.int64, d1_scale)
        d2_precision_const = context.get_constant(types.int64, d2_precision)
        d2_scale_const = context.get_constant(types.int64, d2_scale)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="add_or_subtract_decimal_scalars"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                d1,
                d1_precision_const,
                d1_scale_const,
                d2,
                d2_precision_const,
                d2_scale_const,
                output_precision,
                output_scale,
                do_addition,
                overflow_pointer,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    output_decimal_type = Decimal128Type(output_precision, output_scale)
    ret_type = types.Tuple([output_decimal_type, types.bool_])
    return ret_type(d1_t, d2_t, precision_t, scale_t, do_addition_t), codegen


def add_or_subtract_decimal_arrays(d1, d2, do_addition):  # pragma: no cover
    pass


@overload(add_or_subtract_decimal_arrays)
def overload_add_or_subtract_decimal_arrays(d1, d2, do_addition):
    """
    Add or subtract two decimal arrays together. If overflow occurs,
    this raises an exception.

    do_addition should be set to True for addition, False for subtraction.
    """
    from bodo.libs.array import delete_info, info_to_array

    if not isinstance(d1, DecimalArrayType) and not isinstance(
        d2, DecimalArrayType
    ):  # pragma: no cover
        raise BodoError(
            "add_or_subtract_decimal_arrays: DecimalArrayType expected at least one inputs"
        )

    if not isinstance(d1, (DecimalArrayType, Decimal128Type)) or not isinstance(
        d2, (DecimalArrayType, Decimal128Type)
    ):  # pragma: no cover
        raise BodoError(
            "add_or_subtract_decimal_arrays: both arguments must be either a decimal array or a decimal scalar"
        )

    p, s = decimal_addition_subtraction_output_precision_scale(
        d1.precision, d1.scale, d2.precision, d2.scale
    )
    output_decimal_arr_type = DecimalArrayType(p, s)

    def impl(d1, d2, do_addition):  # pragma: no cover
        d1_info, is_scalar_d1 = array_or_scalar_to_info(d1)
        d2_info, is_scalar_d2 = array_or_scalar_to_info(d2)
        out_arr_info, overflow = _add_or_subtract_decimal_arrays(
            d1_info, d2_info, is_scalar_d1, is_scalar_d2, p, s, do_addition
        )
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        if overflow:
            raise ValueError("Number out of representable range")
        return out_arr

    return impl


@intrinsic
def _add_or_subtract_decimal_arrays(
    typingctx,
    d1_t,
    d2_t,
    is_scalar_d1_t,
    is_scalar_d2_t,
    out_precision_t,
    out_scale_t,
    do_addition_t,
):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        (
            d1,
            d2,
            is_scalar_d1,
            is_scalar_d2,
            output_precision,
            output_scale,
            do_addition,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="add_or_subtract_decimal_arrays"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                d1,
                d2,
                is_scalar_d1,
                is_scalar_d2,
                output_precision,
                output_scale,
                do_addition,
                overflow_pointer,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    ret_type = types.Tuple([array_info_type, types.bool_])
    return (
        ret_type(
            d1_t,
            d2_t,
            is_scalar_d1_t,
            is_scalar_d2_t,
            out_precision_t,
            out_scale_t,
            do_addition_t,
        ),
        codegen,
    )


def decimal_multiplication_output_precision_scale(p1, s1, p2, s2):
    """
    Calculate the output precision and scale for a multiplication of two decimals.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#multiplication
    """
    l1 = p1 - s1
    l2 = p2 - s2
    l = l1 + l2
    s = min(s1 + s2, max(s1, s2, 12))
    p = min(l + s, 38)
    return p, s


def multiply_decimal_scalars(d1, d2):  # pragma: no cover
    pass


@overload(multiply_decimal_scalars)
def overload_multiply_decimal_scalars(d1, d2):
    """
    Multiply two decimal scalars together. If overflow occurs
    this raises an exception.
    """
    if not isinstance(d1, Decimal128Type) or not isinstance(d2, Decimal128Type):
        raise BodoError(
            "multiply_decimal_scalars: Decimal128Type expected for both inputs"
        )

    p, s = decimal_multiplication_output_precision_scale(
        d1.precision, d1.scale, d2.precision, d2.scale
    )

    def impl(d1, d2):  # pragma: no cover
        output, overflow = _multiply_decimal_scalars(d1, d2, p, s)
        if overflow:
            raise ValueError("Number out of representable range")
        else:
            return output

    return impl


@intrinsic(prefer_literal=True)
def _multiply_decimal_scalars(typingctx, d1_t, d2_t, precision_t, scale_t):
    assert isinstance(d1_t, Decimal128Type)
    assert isinstance(d2_t, Decimal128Type)
    assert_bodo_error(is_overload_constant_int(precision_t))
    assert_bodo_error(is_overload_constant_int(scale_t))
    output_precision = get_overload_const_int(precision_t)
    output_scale = get_overload_const_int(scale_t)
    d1_precision = d1_t.precision
    d1_scale = d1_t.scale
    d2_precision = d2_t.precision
    d2_scale = d2_t.scale

    def codegen(context, builder, signature, args):
        d1, d2, output_precision, output_scale = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1).as_pointer(),
            ],
        )
        d1_precision_const = context.get_constant(types.int64, d1_precision)
        d1_scale_const = context.get_constant(types.int64, d1_scale)
        d2_precision_const = context.get_constant(types.int64, d2_precision)
        d2_scale_const = context.get_constant(types.int64, d2_scale)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="multiply_decimal_scalars"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                d1,
                d1_precision_const,
                d1_scale_const,
                d2,
                d2_precision_const,
                d2_scale_const,
                output_precision,
                output_scale,
                overflow_pointer,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    output_decimal_type = Decimal128Type(output_precision, output_scale)
    ret_type = types.Tuple([output_decimal_type, types.bool_])
    return ret_type(d1_t, d2_t, precision_t, scale_t), codegen


def multiply_decimal_arrays(d1, d2):  # pragma: no cover
    pass


@overload(multiply_decimal_arrays)
def overload_multiply_decimal_arrays(d1, d2):
    """
    Multiply two decimal arrays together. If overflow occurs,
    this raises an exception.
    """
    from bodo.libs.array import delete_info, info_to_array

    assert isinstance(
        d1, (DecimalArrayType, Decimal128Type)
    ), "multiply_decimal_arrays: decimal input1 expected"
    assert isinstance(
        d2, (DecimalArrayType, Decimal128Type)
    ), "multiply_decimal_arrays: decimal input2 expected"
    assert isinstance(d1, DecimalArrayType) or isinstance(
        d2, DecimalArrayType
    ), "multiply_decimal_arrays: decimal array expected"

    p, s = decimal_multiplication_output_precision_scale(
        d1.precision, d1.scale, d2.precision, d2.scale
    )
    output_decimal_arr_type = DecimalArrayType(p, s)

    def impl(d1, d2):  # pragma: no cover
        # For simplicity, convert scalar inputs to arrays and pass a flag to C++ to
        # convert back to scalars
        d1_info, is_scalar_d1 = array_or_scalar_to_info(d1)
        d2_info, is_scalar_d2 = array_or_scalar_to_info(d2)
        out_arr_info, overflow = _multiply_decimal_arrays(
            d1_info, d2_info, p, s, is_scalar_d1, is_scalar_d2
        )
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        if overflow:
            raise ValueError("Number out of representable range")
        return out_arr

    return impl


@intrinsic
def _multiply_decimal_arrays(
    typingctx, d1_t, d2_t, out_precision_t, out_scale_t, is_scalar_d1_t, is_scalar_d2_t
):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        d1, d2, output_precision, output_scale, is_scalar_d1, is_scalar_d2 = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="multiply_decimal_arrays"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                d1,
                d2,
                output_precision,
                output_scale,
                is_scalar_d1,
                is_scalar_d2,
                overflow_pointer,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    ret_type = types.Tuple([array_info_type, types.bool_])
    return (
        ret_type(
            d1_t, d2_t, out_precision_t, out_scale_t, is_scalar_d1_t, is_scalar_d2_t
        ),
        codegen,
    )


def decimal_misc_nary_output_precision_scale(precisions, scales):
    """
    Calculate the output precision and scale for a miscellaneous n-ary operator.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#other-n-ary-operations
    """
    leading_digits = [p - s for p, s in zip(precisions, scales)]
    l = max(leading_digits)
    s = max(scales)
    p = min(l + s, 38)
    return p, s


def modulo_decimal_scalars(d1, d2):  # pragma: no cover
    pass


@overload(modulo_decimal_scalars)
def overload_modulo_decimal_scalars(d1, d2):
    """
    Perform the modulo operation on two decimal scalars.
    """
    if not isinstance(d1, Decimal128Type) or not isinstance(
        d2, Decimal128Type
    ):  # pragma: no cover
        raise BodoError(
            "modulo_decimal_scalars: Decimal128Type expected for both inputs"
        )

    p, s = decimal_misc_nary_output_precision_scale(
        [d1.precision, d2.precision], [d1.scale, d2.scale]
    )

    def impl(d1, d2):  # pragma: no cover
        return _modulo_decimal_scalars(d1, d2, p, s)

    return impl


@intrinsic(prefer_literal=True)
def _modulo_decimal_scalars(typingctx, d1_t, d2_t, out_precision_t, out_scale_t):
    assert isinstance(d1_t, Decimal128Type), "_modulo_decimal_scalars: decimal expected"
    assert isinstance(d2_t, Decimal128Type), "_modulo_decimal_scalars: decimal expected"
    assert_bodo_error(
        is_overload_constant_int(out_precision_t),
        "_modulo_decimal_scalars: constant precision expected",
    )
    assert_bodo_error(
        is_overload_constant_int(out_scale_t),
        "_modulo_decimal_scalars: constant scale expected",
    )
    output_precision = get_overload_const_int(out_precision_t)
    output_scale = get_overload_const_int(out_scale_t)
    d1_precision = d1_t.precision
    d1_scale = d1_t.scale
    d2_precision = d2_t.precision
    d2_scale = d2_t.scale

    def codegen(context, builder, signature, args):
        d1, d2, output_precision, output_scale = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        d1_precision_const = context.get_constant(types.int64, d1_precision)
        d1_scale_const = context.get_constant(types.int64, d1_scale)
        d2_precision_const = context.get_constant(types.int64, d2_precision)
        d2_scale_const = context.get_constant(types.int64, d2_scale)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="modulo_decimal_scalars"
        )
        ret = builder.call(
            fn,
            [
                d1,
                d1_precision_const,
                d1_scale_const,
                d2,
                d2_precision_const,
                d2_scale_const,
                output_precision,
                output_scale,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    output_decimal_type = Decimal128Type(output_precision, output_scale)
    return output_decimal_type(d1_t, d2_t, out_precision_t, out_scale_t), codegen


def modulo_decimal_arrays(d1, d2):  # pragma: no cover
    pass


@overload(modulo_decimal_arrays)
def overload_modulo_decimal_arrays(d1, d2):
    """
    Mod two decimal arrays or a decimal array and a decimal scalar.
    """
    from bodo.libs.array import delete_info, info_to_array

    assert isinstance(
        d1, (DecimalArrayType, Decimal128Type)
    ), "modulo_decimal_arrays: decimal input1 expected"
    assert isinstance(
        d2, (DecimalArrayType, Decimal128Type)
    ), "modulo_decimal_arrays: decimal input2 expected"
    assert isinstance(d1, DecimalArrayType) or isinstance(
        d2, DecimalArrayType
    ), "modulo_decimal_arrays: decimal array expected"

    p, s = decimal_misc_nary_output_precision_scale(
        [d1.precision, d2.precision], [d1.scale, d2.scale]
    )
    output_decimal_arr_type = DecimalArrayType(p, s)

    def impl(d1, d2):  # pragma: no cover
        # For simplicity, convert scalar inputs to arrays and pass a flag to C++ to
        # convert back to scalars
        d1_info, is_scalar_d1 = array_or_scalar_to_info(d1)
        d2_info, is_scalar_d2 = array_or_scalar_to_info(d2)
        out_arr_info = _modulo_decimal_arrays(
            d1_info, d2_info, p, s, is_scalar_d1, is_scalar_d2
        )
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        return out_arr

    return impl


@intrinsic
def _modulo_decimal_arrays(
    typingctx, d1_t, d2_t, out_precision_t, out_scale_t, is_scalar_d1_t, is_scalar_d2_t
):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        d1, d2, output_precision, output_scale, is_scalar_d1, is_scalar_d2 = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="modulo_decimal_arrays"
        )
        ret = builder.call(
            fn,
            [
                d1,
                d2,
                is_scalar_d1,
                is_scalar_d2,
                output_precision,
                output_scale,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        array_info_type(
            d1_t, d2_t, out_precision_t, out_scale_t, is_scalar_d1_t, is_scalar_d2_t
        ),
        codegen,
    )


def decimal_division_output_precision_scale(p1, s1, p2, s2):
    """
    Calculate the output precision and scale for a division of two decimals.
    See: https://docs.snowflake.com/en/sql-reference/operators-arithmetic#division
    """
    l1 = p1 - s1
    l = l1 + s2
    s = max(s1, min(s1 + 6, 12))
    p = min(l + s, 38)
    return p, s


def divide_decimal_scalars(d1, d2, do_div0=False):  # pragma: no cover
    pass


@overload(divide_decimal_scalars)
def overload_divide_decimal_scalars(d1, d2, do_div0=False):
    """
    Divide two decimal scalars. If overflow occurs this raises an exception.
    """
    if not isinstance(d1, Decimal128Type) or not isinstance(d2, Decimal128Type):
        raise BodoError(
            "divide_decimal_scalars: Decimal128Type expected for both inputs"
        )

    p, s = decimal_division_output_precision_scale(
        d1.precision, d1.scale, d2.precision, d2.scale
    )

    def impl(d1, d2, do_div0=False):  # pragma: no cover
        output, overflow = _divide_decimal_scalars(d1, d2, p, s, do_div0)
        if overflow:
            raise ValueError("Number out of representable range")
        else:
            return output

    return impl


@intrinsic(prefer_literal=True)
def _divide_decimal_scalars(typingctx, d1_t, d2_t, precision_t, scale_t, do_div0):
    assert isinstance(d1_t, Decimal128Type), "_divide_decimal_scalars: decimal expected"
    assert isinstance(d2_t, Decimal128Type), "_divide_decimal_scalars: decimal expected"
    assert_bodo_error(
        is_overload_constant_int(precision_t),
        "_divide_decimal_scalars: constant precision expected",
    )
    assert_bodo_error(
        is_overload_constant_int(scale_t),
        "_divide_decimal_scalars: constant scale expected",
    )
    output_precision = get_overload_const_int(precision_t)
    output_scale = get_overload_const_int(scale_t)
    d1_precision = d1_t.precision
    d1_scale = d1_t.scale
    d2_precision = d2_t.precision
    d2_scale = d2_t.scale

    def codegen(context, builder, signature, args):
        d1, d2, output_precision, output_scale, do_div0 = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1).as_pointer(),
                lir.IntType(1),
            ],
        )
        d1_precision_const = context.get_constant(types.int64, d1_precision)
        d1_scale_const = context.get_constant(types.int64, d1_scale)
        d2_precision_const = context.get_constant(types.int64, d2_precision)
        d2_scale_const = context.get_constant(types.int64, d2_scale)
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="divide_decimal_scalars"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                d1,
                d1_precision_const,
                d1_scale_const,
                d2,
                d2_precision_const,
                d2_scale_const,
                output_precision,
                output_scale,
                overflow_pointer,
                do_div0,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    output_decimal_type = Decimal128Type(output_precision, output_scale)
    ret_type = types.Tuple([output_decimal_type, types.bool_])
    return ret_type(d1_t, d2_t, precision_t, scale_t, do_div0), codegen


def divide_decimal_arrays(d1, d2, do_div0=False):  # pragma: no cover
    pass


@overload(divide_decimal_arrays)
def overload_divide_decimal_arrays(d1, d2, do_div0=False):
    """
    Divide two decimal arrays or a decimal array and a decimal scalar.
    Raises an exception if overflows.
    """
    from bodo.libs.array import delete_info, info_to_array

    assert isinstance(
        d1, (DecimalArrayType, Decimal128Type)
    ), "divide_decimal_arrays: decimal input1 expected"
    assert isinstance(
        d2, (DecimalArrayType, Decimal128Type)
    ), "divide_decimal_arrays: decimal input2 expected"
    assert isinstance(d1, DecimalArrayType) or isinstance(
        d2, DecimalArrayType
    ), "divide_decimal_arrays: decimal array expected"

    p, s = decimal_division_output_precision_scale(
        d1.precision, d1.scale, d2.precision, d2.scale
    )
    output_decimal_arr_type = DecimalArrayType(p, s)

    def impl(d1, d2, do_div0=False):  # pragma: no cover
        # For simplicity, convert scalar inputs to arrays and pass a flag to C++ to
        # convert back to scalars
        d1_info, is_scalar_d1 = array_or_scalar_to_info(d1)
        d2_info, is_scalar_d2 = array_or_scalar_to_info(d2)
        out_arr_info, overflow = _divide_decimal_arrays(
            d1_info, d2_info, p, s, is_scalar_d1, is_scalar_d2, do_div0
        )
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        if overflow:
            raise ValueError("Number out of representable range")
        return out_arr

    return impl


@intrinsic
def _divide_decimal_arrays(
    typingctx,
    d1_t,
    d2_t,
    out_precision_t,
    out_scale_t,
    is_scalar_d1_t,
    is_scalar_d2_t,
    do_div0,
):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        (
            d1,
            d2,
            output_precision,
            output_scale,
            is_scalar_d1,
            is_scalar_d2,
            do_div0,
        ) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
                lir.IntType(1),
                lir.IntType(1).as_pointer(),
                lir.IntType(1),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="divide_decimal_arrays"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                d1,
                d2,
                output_precision,
                output_scale,
                is_scalar_d1,
                is_scalar_d2,
                overflow_pointer,
                do_div0,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    ret_type = types.Tuple([array_info_type, types.bool_])
    return (
        ret_type(
            d1_t,
            d2_t,
            out_precision_t,
            out_scale_t,
            is_scalar_d1_t,
            is_scalar_d2_t,
            do_div0,
        ),
        codegen,
    )


def round_decimal_array(arr, round_scale, output_p, output_s):  # pragma: no cover
    pass


@overload(round_decimal_array, inline="always", prefer_literal=True)
def overload_round_decimal_array(arr, round_scale, output_p, output_s):
    """
    Rounds a decimal array to a given scale.
    Negative round_scale rounds to the left of the decimal point,
    truncating -round_scale digits and multiplying by 10^round_scale.
    If overflow occurs, this raises an exception.
    """
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    assert isinstance(
        arr, DecimalArrayType
    ), "round_decimal_array: decimal arr expected"
    assert isinstance(
        round_scale, types.Integer
    ), "round_decimal_array: integer round_scale expected"
    assert isinstance(
        output_p, types.Integer
    ), "round_decimal_array: integer output_p expected"

    assert isinstance(
        output_s, types.Integer
    ), "round_decimal_array: integer output_s expected"

    output_p_val = get_overload_const_int(output_p)
    output_s_val = get_overload_const_int(output_s)
    output_decimal_arr_type = DecimalArrayType(output_p_val, output_s_val)

    def impl(arr, round_scale, output_p, output_s):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_arr_info, overflow = _round_decimal_array(
            arr_info, round_scale, output_p, output_s
        )
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        if overflow:
            raise ValueError("Number out of representable range")
        return out_arr

    return impl


@intrinsic
def _round_decimal_array(typingctx, arr_t, round_scale_t, output_p_t, output_s_t):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        arr, round_scale, output_p, output_s = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="round_decimal_array"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [
                arr,
                round_scale,
                output_p,
                output_s,
                overflow_pointer,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    ret_type = types.Tuple([array_info_type, types.bool_])
    return (
        ret_type(arr_t, round_scale_t, output_p_t, output_s_t),
        codegen,
    )


def round_decimal_scalar(
    val, round_scale, input_p, input_s, output_p, output_s
):  # pragma: no cover
    pass


@overload(round_decimal_scalar, inline="always", prefer_literal=True)
def overload_round_decimal_scalar(
    val, round_scale, input_p, input_s, output_p, output_s
):
    """
    Rounds a decimal scalar to a given scale.
    Negative round_scale rounds to the left of the decimal point,
    truncating -round_scale digits and multiplying by 10^round_scale.
    If overflow occurs, this raises an exception.
    """

    def impl(
        val, round_scale, input_p, input_s, output_p, output_s
    ):  # pragma: no cover
        result, overflow = _round_decimal_scalar(
            val, round_scale, input_p, input_s, output_p, output_s
        )
        if overflow:
            raise ValueError("Number out of representable range")
        else:
            return result

    return impl


@intrinsic(prefer_literal=True)
def _round_decimal_scalar(
    typingctx, val_t, round_scale_t, input_p_t, input_s_t, output_p_t, output_s_t
):
    assert isinstance(val_t, Decimal128Type), "_round_decimal_scalar: decimal expected"
    assert isinstance(
        round_scale_t, types.Integer
    ), "_round_decimal_scalar: integer expected"
    assert_bodo_error(is_overload_constant_int(output_p_t))
    assert_bodo_error(is_overload_constant_int(output_s_t))
    assert_bodo_error(is_overload_constant_int(input_p_t))
    assert_bodo_error(is_overload_constant_int(input_s_t))

    def codegen(context, builder, signature, args):
        val, round_scale, input_p, input_s, output_p, output_s = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="round_decimal_scalar"
        )
        overflow_pointer = cgutils.alloca_once(builder, lir.IntType(1))
        ret = builder.call(
            fn,
            [val, round_scale, input_p, input_s, overflow_pointer],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        overflow = builder.load(overflow_pointer)
        return context.make_tuple(builder, signature.return_type, [ret, overflow])

    output_precision = get_overload_const_int(output_p_t)
    output_scale = get_overload_const_int(output_s_t)
    output_decimal_type = Decimal128Type(output_precision, output_scale)
    ret_type = types.Tuple([output_decimal_type, types.bool_])
    return (
        ret_type(val_t, round_scale_t, input_p_t, input_s_t, output_p_t, output_s_t),
        codegen,
    )


def ceil_floor_decimal_scalar(
    value, input_p, input_s, output_p, output_s, round_scale, is_ceil
):  # pragma: no cover
    pass


@overload(ceil_floor_decimal_scalar, inline="always", prefer_literal=True)
def overload_ceil_floor_decimal_scalar(
    value, input_p, input_s, output_p, output_s, round_scale, is_ceil
):
    """
    Ceil or floor a decimal scalar.
    """

    def impl(
        value, input_p, input_s, output_p, output_s, round_scale, is_ceil
    ):  # pragma: no cover
        result = _ceil_floor_decimal_scalar(
            value, input_p, input_s, output_p, output_s, round_scale, is_ceil
        )
        return result

    return impl


@intrinsic(prefer_literal=True)
def _ceil_floor_decimal_scalar(
    typingctx,
    value_t,
    input_p_t,
    input_s_t,
    output_p_t,
    output_s_t,
    round_scale_t,
    is_ceil_t,
):
    assert isinstance(
        value_t, Decimal128Type
    ), "_ceil_floor_decimal_scalar: decimal expected for value"

    def codegen(context, builder, signature, args):
        value, input_p, input_s, output_p, output_s, round_scale, is_ceil = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="ceil_floor_decimal_scalar"
        )
        ret = builder.call(
            fn,
            [value, input_p, input_s, round_scale, is_ceil],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    output_precision = get_overload_const_int(output_p_t)
    output_scale = get_overload_const_int(output_s_t)
    output_decimal_type = Decimal128Type(output_precision, output_scale)
    return (
        output_decimal_type(
            value_t,
            input_p_t,
            input_s_t,
            output_p_t,
            output_s_t,
            round_scale_t,
            is_ceil_t,
        ),
        codegen,
    )


def ceil_floor_decimal_array(
    arr, round_scale, output_p, output_s, is_ceil
):  # pragma: no cover
    pass


@overload(ceil_floor_decimal_array, inline="always", prefer_literal=True)
def overload_ceil_floor_decimal_array(arr, round_scale, output_p, output_s, is_ceil):
    """
    Ceil or floor a decimal array.
    """

    from bodo.libs.array import array_to_info, delete_info, info_to_array

    assert isinstance(
        arr, DecimalArrayType
    ), "ceil_floor_decimal_array: decimal arr expected"
    assert isinstance(
        round_scale, types.Integer
    ), "ceil_floor_decimal_array: integer round_scale expected"
    assert isinstance(
        output_p, types.Integer
    ), "ceil_floor_decimal_array: integer output_p expected"
    assert isinstance(
        output_s, types.Integer
    ), "ceil_floor_decimal_array: integer output_s expected"

    output_p_val = get_overload_const_int(output_p)
    output_s_val = get_overload_const_int(output_s)
    output_decimal_arr_type = DecimalArrayType(output_p_val, output_s_val)

    def impl(arr, round_scale, output_p, output_s, is_ceil):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_arr_info = _ceil_floor_decimal_array(
            arr_info, round_scale, output_p, output_s, is_ceil
        )
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        return out_arr

    return impl


@intrinsic
def _ceil_floor_decimal_array(
    typingctx, arr_t, round_scale_t, output_p_t, output_s_t, is_ceil_t
):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        arr, round_scale, output_p, output_s, is_ceil = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(1),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="ceil_floor_decimal_array"
        )
        ret = builder.call(
            fn,
            [
                arr,
                output_p,
                output_s,
                round_scale,
                is_ceil,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        array_info_type(arr_t, round_scale_t, output_p_t, output_s_t, is_ceil_t),
        codegen,
    )


def trunc_decimal_scalar(
    value,
    input_p,
    input_s,
    output_p,
    output_s,
    round_scale,
):  # pragma: no cover
    pass


@overload(trunc_decimal_scalar, inline="always", prefer_literal=True)
def overload_trunc_decimal_scalar(
    value,
    input_p,
    input_s,
    output_p,
    output_s,
    round_scale,
):
    """
    Truncate a decimal scalar.
    """

    def impl(
        value,
        input_p,
        input_s,
        output_p,
        output_s,
        round_scale,
    ):  # pragma: no cover
        result = _trunc_decimal_scalar(
            value,
            input_p,
            input_s,
            output_p,
            output_s,
            round_scale,
        )
        return result

    return impl


@intrinsic(prefer_literal=True)
def _trunc_decimal_scalar(
    typingctx,
    value_t,
    input_p_t,
    input_s_t,
    output_p_t,
    output_s_t,
    round_scale_t,
):
    assert isinstance(
        value_t, Decimal128Type
    ), "_trunc_decimal_scalar: decimal expected for value"

    def codegen(context, builder, signature, args):
        value, input_p, input_s, output_p, output_s, round_scale = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="trunc_decimal_scalar"
        )
        ret = builder.call(
            fn,
            [
                value,
                input_p,
                input_s,
                output_p,
                output_s,
                round_scale,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    output_precision = get_overload_const_int(output_p_t)
    output_scale = get_overload_const_int(output_s_t)
    output_decimal_type = Decimal128Type(output_precision, output_scale)
    return (
        output_decimal_type(
            value_t,
            input_p_t,
            input_s_t,
            output_p_t,
            output_s_t,
            round_scale_t,
        ),
        codegen,
    )


def trunc_decimal_array(arr, round_scale, output_p, output_s):  # pragma: no cover
    pass


@overload(trunc_decimal_array, inline="always", prefer_literal=True)
def overload_trunc_decimal_array(arr, round_scale, output_p, output_s):
    """
    Truncate a decimal array.
    """

    from bodo.libs.array import array_to_info, delete_info, info_to_array

    assert isinstance(
        arr, DecimalArrayType
    ), "trunc_decimal_array: decimal arr expected"
    assert isinstance(
        round_scale, types.Integer
    ), "trunc_decimal_array: integer round_scale expected"
    assert isinstance(
        output_p, types.Integer
    ), "trunc_decimal_array: integer output_p expected"
    assert isinstance(
        output_s, types.Integer
    ), "trunc_decimal_array: integer output_s expected"

    output_p_val = get_overload_const_int(output_p)
    output_s_val = get_overload_const_int(output_s)
    output_decimal_arr_type = DecimalArrayType(output_p_val, output_s_val)

    def impl(arr, round_scale, output_p, output_s):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_arr_info = _trunc_decimal_array(arr_info, round_scale, output_p, output_s)
        out_arr = info_to_array(out_arr_info, output_decimal_arr_type)
        delete_info(out_arr_info)
        return out_arr

    return impl


@intrinsic
def _trunc_decimal_array(typingctx, arr_t, round_scale_t, output_p_t, output_s_t):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        arr, round_scale, output_p, output_s = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
                lir.IntType(64),
                lir.IntType(64),
                lir.IntType(64),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="trunc_decimal_array"
        )
        ret = builder.call(
            fn,
            [
                arr,
                output_p,
                output_s,
                round_scale,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        array_info_type(arr_t, round_scale_t, output_p_t, output_s_t),
        codegen,
    )


def abs_decimal_array(arr):  # pragma: no cover
    pass


@overload(abs_decimal_array, inline="always")
def overload_abs_decimal_array(arr):
    """
    Return the absolute value of a decimal array.
    """
    assert isinstance(
        arr, DecimalArrayType
    ), "abs_decimal_array: decimal array expected"
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    out_precision = arr.precision
    out_scale = arr.scale
    out_type = DecimalArrayType(out_precision, out_scale)

    def impl(arr):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_arr_info = _abs_decimal_array(arr_info)
        out_arr = info_to_array(out_arr_info, out_type)
        delete_info(out_arr_info)
        return out_arr

    return impl


@intrinsic
def _abs_decimal_array(typingctx, arr):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        (arr,) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="abs_decimal_array"
        )
        ret = builder.call(fn, [arr])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (array_info_type(arr), codegen)


def abs_decimal_scalar(arr):  # pragma: no cover
    pass


@overload(abs_decimal_scalar, inline="always")
def overload_abs_decimal_scalar(arr):
    """
    Return the absolute value of a decimal scalar.
    """
    assert isinstance(arr, Decimal128Type)

    def impl(arr):
        return _abs_decimal_scalar(arr)

    return impl


@intrinsic
def _abs_decimal_scalar(typingctx, arr_t):
    assert isinstance(arr_t, Decimal128Type), "_abs_decimal_scalar: decimal expected"

    def codegen(context, builder, signature, args):
        arr = args[0]
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="abs_decimal_scalar"
        )
        ret = builder.call(
            fn,
            [
                arr,
            ],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    output_precision = arr_t.precision
    output_scale = arr_t.scale
    output_decimal_type = Decimal128Type(output_precision, output_scale)
    return (output_decimal_type(arr_t), codegen)


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


def factorial_decimal_scalar(val):  # pragma: no cover
    pass


@overload(factorial_decimal_scalar)
def overload_factorial_decimal_scalar(val):
    """
    Calculate the factorial of a decimal scalar.
    """
    assert isinstance(val, Decimal128Type), "factorial_decimal_scalar: decimal expected"
    input_s = val.scale

    def impl(val):  # pragma: no cover
        return _factorial_decimal_scalar(val, input_s)

    return impl


@intrinsic
def _factorial_decimal_scalar(typingctx, val_t, input_s):
    assert isinstance(
        val_t, Decimal128Type
    ), "_factorial_decimal_scalar: decimal expected"

    def codegen(context, builder, signature, args):
        val, input_s = args
        fnty = lir.FunctionType(
            lir.IntType(128),
            [
                lir.IntType(128),
                lir.IntType(64),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="factorial_decimal_scalar"
        )
        ret = builder.call(
            fn,
            [val, input_s],
        )
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    output_precision = 37
    output_scale = 0
    output_decimal_type = Decimal128Type(output_precision, output_scale)
    return (output_decimal_type(val_t, input_s), codegen)


def factorial_decimal_array(arr):  # pragma: no cover
    pass


@overload(factorial_decimal_array)
def overload_factorial_decimal_array(arr):
    """
    Calculate the factorial of a decimal array.
    """
    assert isinstance(
        arr, DecimalArrayType
    ), "factorial_decimal_array: decimal array expected"
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    out_precision = 37
    out_scale = 0
    out_type = DecimalArrayType(out_precision, out_scale)

    def impl(arr):  # pragma: no cover
        arr_info = array_to_info(arr)
        out_arr_info = _factorial_decimal_array(arr_info)
        out_arr = info_to_array(out_arr_info, out_type)
        delete_info(out_arr_info)
        return out_arr

    return impl


@intrinsic
def _factorial_decimal_array(typingctx, arr_t):
    from bodo.libs.array import array_info_type

    def codegen(context, builder, signature, args):
        (arr,) = args
        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="factorial_decimal_array"
        )
        ret = builder.call(fn, [arr])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (array_info_type(arr_t), codegen)


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
def init_decimal_array(typingctx, data, null_bitmap, precision_tp, scale_tp):
    """Create a DecimalArray with provided data and null bitmap values."""
    assert data == types.Array(int128_type, 1, "C")
    assert null_bitmap == types.Array(types.uint8, 1, "C")
    assert_bodo_error(is_overload_constant_int(precision_tp))
    assert_bodo_error(is_overload_constant_int(scale_tp))

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
    Box decimal array into Pandas Arrow extension array
    """
    # Reusing nested array boxing since it covers decimal arrays as well
    return bodo.libs.array.box_nested_array(typ, val, c)


@unbox(DecimalArrayType)
def unbox_decimal_arr(typ, val, c):
    """
    Unbox a numpy array with Decimal objects or Arrow decimal array into native
    DecimalArray
    """
    # Reusing nested array unboxing since it covers decimal arrays as well
    return bodo.libs.array.unbox_nested_array(typ, val, c)


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
        _precision = A.precision
        _scale = A.scale
        # This is the existing type check
        if isinstance(val, Decimal128Type):

            def impl_scalar(A, idx, val):  # pragma: no cover
                A._data[idx] = decimal128type_to_int128(val)
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            # Covered by test_series_iat_setitem , test_series_iloc_setitem_int , test_series_setitem_int
            return impl_scalar
        elif isinstance(val, types.Integer):
            max_real_precision = int_to_decimal_precision[val]
            always_safe = max_real_precision <= _precision

            def impl_scalar(A, idx, val):  # pragma: no cover
                decimal_int = int_to_decimal_scalar(val)
                if always_safe:
                    cast_decimal = _cast_decimal_to_decimal_scalar_unsafe(
                        decimal_int, _precision, _scale
                    )
                else:
                    cast_decimal, overflow = _cast_decimal_to_decimal_scalar_safe(
                        decimal_int, _precision, _scale
                    )
                    if overflow:
                        raise ValueError("Number out of representable range")
                A._data[idx] = decimal128type_to_int128(cast_decimal)
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            return impl_scalar
        elif isinstance(val, types.Float):

            def impl_scalar(A, idx, val):  # pragma: no cover
                A._data[idx] = decimal128type_to_int128(
                    float_to_decimal_scalar(val, _precision, _scale, False)
                )
                bodo.libs.int_arr_ext.set_bit_to_arr(A._null_bitmap, idx, 1)

            return impl_scalar
        else:
            raise BodoError(
                f"setitem for DecimalArray with scalar type {val} not supported."
            )  # pragma: no cover

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
    from bodo.libs.array import array_info_type, delete_info, info_to_array

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


@intrinsic
def int_to_decimal_scalar(typingctx, val_t):
    """convert integer to decimal128"""
    assert isinstance(val_t, types.Integer), "expected integer type"

    # Convert int value to int128 using sign extend
    def int_codegen(context, builder, sig, args):
        (val,) = args
        return builder.sext(val, lir.IntType(128))

    # Convert int value to int128 using zero extend
    def uint_codegen(context, builder, sig, args):
        (val,) = args
        return builder.zext(val, lir.IntType(128))

    prec = int_to_decimal_precision[val_t]

    codegen = int_codegen if val_t.signed else uint_codegen

    return Decimal128Type(prec, 0)(val_t), codegen


@intrinsic
def int_to_decimal_array(typingctx, info_t):
    """cast integer array to decimal array"""

    def codegen(context, builder, sig, args):
        (val,) = args

        fnty = lir.FunctionType(
            lir.IntType(8).as_pointer(),
            [
                lir.IntType(8).as_pointer(),
            ],
        )
        fn = cgutils.get_or_insert_function(
            builder.module, fnty, name="int_to_decimal_array"
        )
        ret = builder.call(fn, [val])
        bodo.utils.utils.inlined_check_and_propagate_cpp_exception(context, builder)
        return ret

    return (
        bodo.libs.array.array_info_type(info_t),
        codegen,
    )


def int_to_decimal(int_arg):  # pragma: no cover
    pass


@overload(int_to_decimal)
def overload_int_to_decimal(int_arg):
    """
    Takes in an integer scalar or array and converts it to a decimal scalar or array
    """
    from bodo.libs.array import array_to_info, delete_info, info_to_array

    if isinstance(int_arg, types.Integer):

        def impl(int_arg):  # pragma: no cover
            return bodo.libs.decimal_arr_ext.int_to_decimal_scalar(int_arg)

        return impl
    elif bodo.utils.utils.is_array_typ(int_arg, False) and isinstance(
        int_arg.dtype, types.Integer
    ):
        prec = int_to_decimal_precision[int_arg.dtype]
        result_type = DecimalArrayType(prec, 0)

        def impl(int_arg):  # pragma: no cover
            arr_info = array_to_info(int_arg)
            res_arr_info = bodo.libs.decimal_arr_ext.int_to_decimal_array(arr_info)
            res_arr = info_to_array(res_arr_info, result_type)
            delete_info(res_arr_info)
            return res_arr

        return impl
    else:
        raise_bodo_error(f"int_to_decimal: unsupported argument type {int_arg}")
