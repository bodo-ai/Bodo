# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Numba extension support for datetime.date objects and their arrays.
"""
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


@numba.njit
def convert_datetime_date_array_to_native(x):  # pragma: no cover
    return np.array(
        [(val.day + (val.month << 16) + (val.year << 32)) for val in x], dtype=np.int64
    )


@numba.njit
def datetime_date_ctor(y, m, d):  # pragma: no cover
    return datetime.date(y, m, d)


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


def unbox_datetime_date_array(typ, val, c):
    #
    n = object_length(c, val)
    # cgutils.printf(c.builder, "len %d\n", n)
    arr_typ = types.Array(types.intp, 1, "C")
    out_arr = _empty_nd_impl(c.context, c.builder, arr_typ, [n])

    with cgutils.for_range(c.builder, n) as loop:
        dt_date = sequence_getitem(c, val, loop.index)
        int_date = unbox_datetime_date(datetime_date_type, dt_date, c).value
        dataptr, shapes, strides = basic_indexing(
            c.context, c.builder, arr_typ, out_arr, (types.intp,), (loop.index,)
        )
        store_item(c.context, c.builder, arr_typ, int_date, dataptr)

    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(out_arr._getvalue(), is_error=is_error)


def int_to_datetime_date_python(ia):
    return datetime.date(ia >> 32, (ia >> 16) & 0xFFFF, ia & 0xFFFF)


def int_array_to_datetime_date(ia):
    return np.vectorize(int_to_datetime_date_python)(ia)


def box_datetime_date_array(typ, val, c):
    ary = c.pyapi.from_native_value(
        types.Array(types.int64, 1, "C"), val, c.env_manager
    )
    hpat_name = c.context.insert_const_string(c.builder.module, "bodo")
    hpat_mod = c.pyapi.import_module_noblock(hpat_name)
    hi_mod = c.pyapi.object_getattr_string(hpat_mod, "hiframes")
    pte_mod = c.pyapi.object_getattr_string(hi_mod, "datetime_date_ext")
    iatdd = c.pyapi.object_getattr_string(pte_mod, "int_array_to_datetime_date")
    res = c.pyapi.call_function_objargs(iatdd, [ary])
    return res


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
def datetime_date_to_int(typingctx, dt_date_tp):
    assert dt_date_tp == datetime_date_type

    def codegen(context, builder, sig, args):
        return args[0]

    return signature(types.int64, datetime_date_type), codegen


@intrinsic
def int_to_datetime_date(typingctx, dt_date_tp):
    assert dt_date_tp == types.intp

    def codegen(context, builder, sig, args):
        return args[0]

    return signature(datetime_date_type, types.int64), codegen


def set_df_datetime_date(df, cname, arr):
    df[cname] = arr


@infer_global(set_df_datetime_date)
class SetDfDTInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 3
        assert isinstance(args[1], types.Literal)
        return signature(types.none, *args)


@lower_builtin(set_df_datetime_date, types.Any, types.Literal, types.Array)
def set_df_datetime_date_lower(context, builder, sig, args):
    #
    col_name = sig.args[1].literal_value
    data_arr = make_array(sig.args[2])(context, builder, args[2])
    num_elems = builder.extract_value(data_arr.shape, 0)

    pyapi = context.get_python_api(builder)
    gil_state = pyapi.gil_ensure()  # acquire GIL

    dt_class = pyapi.unserialize(pyapi.serialize_object(datetime.date))

    fnty = lir.FunctionType(
        lir.IntType(8).as_pointer(),
        [lir.IntType(64).as_pointer(), lir.IntType(64), lir.IntType(8).as_pointer()],
    )
    fn = builder.module.get_or_insert_function(
        fnty, name="np_datetime_date_array_from_packed_ints"
    )
    py_arr = builder.call(fn, [data_arr.data, num_elems, dt_class])

    # get column as string obj
    cstr = context.insert_const_string(builder.module, col_name)
    cstr_obj = pyapi.string_from_string(cstr)

    # set column array
    pyapi.object_setitem(args[0], cstr_obj, py_arr)

    pyapi.decref(py_arr)
    pyapi.decref(cstr_obj)

    pyapi.gil_release(gil_state)  # release GIL

    return context.get_dummy_value()


_data_array_typ = types.Array(types.int64, 1, "C")

# Array of datetime date objects, same as Array but knows how to box
# TODO: defer to Array for all operations
class ArrayDatetimeDate(types.Array):
    def __init__(self):
        super(ArrayDatetimeDate, self).__init__(
            datetime_date_type, 1, "C", name="array_datetime_date"
        )


array_datetime_date = ArrayDatetimeDate()


@register_model(ArrayDatetimeDate)
class ArrayDatetimeDateModel(models.ArrayModel):
    def __init__(self, dmm, fe_type):
        super(ArrayDatetimeDateModel, self).__init__(dmm, _data_array_typ)


@box(ArrayDatetimeDate)
def box_df_dummy(typ, val, c):
    return box_datetime_date_array(typ, val, c)


# dummy function use to change type of Array(datetime_date) to
# array_datetime_date
def np_arr_to_array_datetime_date(A):  # pragma: no cover
    return A


@infer_global(np_arr_to_array_datetime_date)
class NpArrToArrDtType(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        return signature(array_datetime_date, *args)


@lower_builtin(np_arr_to_array_datetime_date, types.Array(types.int64, 1, "C"))
@lower_builtin(np_arr_to_array_datetime_date, types.Array(datetime_date_type, 1, "C"))
def lower_np_arr_to_array_datetime_date(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])
