# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Numba extension support for datetime.date objects and their arrays.
"""
import operator
import pandas as pd
import numpy as np
import datetime
import numba
from numba import types
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed, lower_builtin
from numba.typing import signature
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
    lower_cast,
    lower_getattr,
    infer_getattr,
    overload_method,
    overload,
    intrinsic,
    overload_attribute,
    register_jitable,
)
from numba import cgutils
from numba.targets.arrayobj import (
    make_array,
    _empty_nd_impl,
    store_item,
    basic_indexing,
)
from numba.typing.templates import (
    infer_getattr,
    AttributeTemplate,
    bound_function,
    signature,
    infer_global,
    AbstractTemplate,
    ConcreteTemplate,
)
from numba.array_analysis import ArrayAnalysis
from llvmlite import ir as lir
import bodo


# datetime.date implementation that uses a single int to store year/month/day
# Does not need refcounted object wrapping since it is immutable
class DatetimeDateType(types.Type):
    def __init__(self):
        super(DatetimeDateType, self).__init__(name="DatetimeDateType()")
        self.bitwidth = 64  # needed for using IntegerModel


datetime_date_type = DatetimeDateType()


@typeof_impl.register(datetime.date)
def typeof_datetime_date(val, c):
    return datetime_date_type


register_model(DatetimeDateType)(models.IntegerModel)


# extraction of year/month/day attributes
@infer_getattr
class DatetimeAttribute(AttributeTemplate):
    key = DatetimeDateType

    def resolve_year(self, typ):
        return types.int64

    def resolve_month(self, typ):
        return types.int64

    def resolve_day(self, typ):
        return types.int64


@lower_getattr(DatetimeDateType, "year")
def datetime_get_year(context, builder, typ, val):
    return builder.lshr(val, lir.Constant(lir.IntType(64), 32))


@lower_getattr(DatetimeDateType, "month")
def datetime_get_month(context, builder, typ, val):
    return builder.and_(
        builder.lshr(val, lir.Constant(lir.IntType(64), 16)),
        lir.Constant(lir.IntType(64), 0xFFFF),
    )


@lower_getattr(DatetimeDateType, "day")
def datetime_get_day(context, builder, typ, val):
    return builder.and_(val, lir.Constant(lir.IntType(64), 0xFFFF))


@unbox(DatetimeDateType)
def unbox_datetime_date(typ, val, c):

    year_obj = c.pyapi.object_getattr_string(val, "year")
    month_obj = c.pyapi.object_getattr_string(val, "month")
    day_obj = c.pyapi.object_getattr_string(val, "day")

    yll = c.pyapi.long_as_longlong(year_obj)
    mll = c.pyapi.long_as_longlong(month_obj)
    dll = c.pyapi.long_as_longlong(day_obj)

    nopython_date = c.builder.add(
        dll,
        c.builder.add(
            c.builder.shl(yll, lir.Constant(lir.IntType(64), 32)),
            c.builder.shl(mll, lir.Constant(lir.IntType(64), 16)),
        ),
    )

    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(nopython_date, is_error=is_error)


@box(DatetimeDateType)
def box_datetime_date(typ, val, c):
    year_obj = c.pyapi.long_from_longlong(
        c.builder.lshr(val, lir.Constant(lir.IntType(64), 32))
    )
    month_obj = c.pyapi.long_from_longlong(
        c.builder.and_(
            c.builder.lshr(val, lir.Constant(lir.IntType(64), 16)),
            lir.Constant(lir.IntType(64), 0xFFFF),
        )
    )
    day_obj = c.pyapi.long_from_longlong(
        c.builder.and_(val, lir.Constant(lir.IntType(64), 0xFFFF))
    )

    dt_obj = c.pyapi.unserialize(c.pyapi.serialize_object(datetime.date))
    res = c.pyapi.call_function_objargs(dt_obj, (year_obj, month_obj, day_obj))
    c.pyapi.decref(year_obj)
    c.pyapi.decref(month_obj)
    c.pyapi.decref(day_obj)
    return res


@type_callable(datetime.date)
def type_datetime_date(context):
    def typer(year, month, day):
        # TODO: check types
        return datetime_date_type

    return typer


@lower_builtin(datetime.date, types.int64, types.int64, types.int64)
def impl_ctor_datetime_date(context, builder, sig, args):
    year, month, day = args
    nopython_date = builder.add(
        day,
        builder.add(
            builder.shl(year, lir.Constant(lir.IntType(64), 32)),
            builder.shl(month, lir.Constant(lir.IntType(64), 16)),
        ),
    )
    return nopython_date


@intrinsic
def cast_int_to_datetime_date(typingctx, val=None):
    """Cast int value to datetime.date
    """
    assert val == types.int64

    def codegen(context, builder, signature, args):
        return args[0]

    return datetime_date_type(types.int64), codegen


@intrinsic
def cast_datetime_date_to_int(typingctx, val=None):
    """Cast datetime.date value to int
    """
    assert val == datetime_date_type

    def codegen(context, builder, signature, args):
        return args[0]

    return types.int64(datetime_date_type), codegen


##################### Array of datetime.date objects ##########################


class DatetimeDateArrayType(types.ArrayCompatible):
    def __init__(self):
        super(DatetimeDateArrayType, self).__init__(name="DatetimeDateArrayType()")

    @property
    def as_array(self):
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return datetime_date_type

    def copy(self):
        return DatetimeDateArrayType()


datetime_date_array_type = DatetimeDateArrayType()


# datetime.date array has only an array integers to store data
@register_model(DatetimeDateArrayType)
class DatetimeDateArrayModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", types.Array(types.int64, 1, "C"))]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DatetimeDateArrayType, "data", "_data")


# TODO: move to utils or Numba
def object_length(c, obj):
    """
    len(obj)
    """
    pyobj_lltyp = c.context.get_argument_type(types.pyobject)
    fnty = lir.FunctionType(lir.IntType(64), [pyobj_lltyp])
    fn = c.builder.module.get_or_insert_function(fnty, name="PyObject_Length")
    return c.builder.call(fn, (obj,))


def sequence_getitem(c, obj, ind):
    """
    seq[ind]
    """
    pyobj_lltyp = c.context.get_argument_type(types.pyobject)
    fnty = lir.FunctionType(pyobj_lltyp, [pyobj_lltyp, lir.IntType(64)])
    fn = c.builder.module.get_or_insert_function(fnty, name="PySequence_GetItem")
    return c.builder.call(fn, (obj, ind))


@unbox(DatetimeDateArrayType)
def unbox_datetime_date_array(typ, val, c):
    n = object_length(c, val)
    arr_typ = types.Array(types.intp, 1, "C")
    out_arr = _empty_nd_impl(c.context, c.builder, arr_typ, [n])
    out_dt_date_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder)

    with cgutils.for_range(c.builder, n) as loop:
        dt_date = sequence_getitem(c, val, loop.index)
        int_date = unbox_datetime_date(datetime_date_type, dt_date, c).value
        dataptr, shapes, strides = basic_indexing(
            c.context, c.builder, arr_typ, out_arr, (types.intp,), (loop.index,)
        )
        store_item(c.context, c.builder, arr_typ, int_date, dataptr)

    out_dt_date_arr.data = out_arr._getvalue()
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(out_dt_date_arr._getvalue(), is_error=is_error)


def int_to_datetime_date_python(ia):
    return datetime.date(ia >> 32, (ia >> 16) & 0xFFFF, ia & 0xFFFF)


def int_array_to_datetime_date(ia):
    return np.vectorize(int_to_datetime_date_python)(ia)


@box(DatetimeDateArrayType)
def box_datetime_date_array(typ, val, c):
    dt_date_arr = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    ary = c.pyapi.from_native_value(
        types.Array(types.int64, 1, "C"), dt_date_arr.data, c.env_manager
    )
    hpat_name = c.context.insert_const_string(c.builder.module, "bodo")
    hpat_mod = c.pyapi.import_module_noblock(hpat_name)
    hi_mod = c.pyapi.object_getattr_string(hpat_mod, "hiframes")
    pte_mod = c.pyapi.object_getattr_string(hi_mod, "datetime_date_ext")
    iatdd = c.pyapi.object_getattr_string(pte_mod, "int_array_to_datetime_date")
    res = c.pyapi.call_function_objargs(iatdd, [ary])
    return res


@intrinsic
def init_datetime_date_array(typingctx, data=None):
    """Create a DatetimeDateArrayType with provided data values.
    """
    assert data == types.Array(types.int64, 1, "C")

    def codegen(context, builder, signature, args):
        (data_val,) = args
        # create arr struct and store values
        dt_date_arr = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        dt_date_arr.data = data_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)

        return dt_date_arr._getvalue()

    sig = datetime_date_array_type(data)
    return sig, codegen


@numba.njit(no_cpython_wrapper=True)
def alloc_datetime_date_array(n):
    data_arr = np.empty(n, dtype=np.int64)
    return init_datetime_date_array(data_arr)


def alloc_datetime_date_array_equiv(self, scope, equiv_set, args, kws):
    """Array analysis function for alloc_datetime_date_array() passed to Numba's array
    analysis extension. Assigns output array's size as equivalent to the input size
    variable.
    """
    assert len(args) == 1 and not kws
    return args[0], []


ArrayAnalysis._analyze_op_call_bodo_hiframes_datetime_date_ext_alloc_datetime_date_array = (
    alloc_datetime_date_array_equiv
)


@overload(operator.getitem)
def dt_date_arr_getitem(A, ind):
    if A != datetime_date_array_type:
        return

    if isinstance(ind, types.Integer):
        return lambda A, ind: cast_int_to_datetime_date(A._data[ind])


@overload(operator.setitem)
def dt_date_arr_setitem(A, ind, val):
    if A != datetime_date_array_type:
        return

    if isinstance(ind, types.Integer):

        def impl(A, ind, val):
            A._data[ind] = cast_datetime_date_to_int(val)

        return impl


@overload(len)
def overload_len_datetime_date_arr(A):
    if A == datetime_date_array_type:
        return lambda A: len(A._data)


types.datetime_date_type = datetime_date_type


@register_jitable
def today_impl():
    """Internal call to support datetime.date.today().
    Untyped pass replaces datetime.date.today() with this call since class methods are
    not supported in Numba's typing
    """
    with numba.objmode(d="datetime_date_type"):
        d = datetime.date.today()
    return d
