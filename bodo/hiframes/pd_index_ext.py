# Copyright (C) 2019 Bodo Inc. All rights reserved.
import operator
import datetime
import pandas as pd
import numpy as np
import numba
from numba import types, cgutils
from numba.extending import (
    models,
    register_model,
    lower_cast,
    infer_getattr,
    type_callable,
    infer,
    overload,
    make_attribute_wrapper,
    box,
    intrinsic,
    typeof_impl,
    unbox,
    NativeValue,
    overload_attribute,
    overload_method,
)
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)

import bodo
from bodo.libs.str_ext import string_type
import bodo.hiframes
from bodo.hiframes.pd_series_ext import is_str_series_typ, string_array_type, SeriesType
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
import bodo.utils.conversion
from bodo.utils.utils import BooleanLiteral
from bodo.utils.typing import (
    is_overload_none,
    is_overload_true,
    is_overload_false,
    get_val_type_maybe_str_literal,
)
from bodo.libs.int_arr_ext import IntegerArrayType


_dt_index_data_typ = types.Array(types.NPDatetime("ns"), 1, "C")
_timedelta_index_data_typ = types.Array(types.NPTimedelta("ns"), 1, "C")
iNaT = pd._libs.tslibs.iNaT
NaT = types.NPDatetime("ns")("NaT")  # TODO: pd.NaT


@typeof_impl.register(pd.Index)
def typeof_pd_index(val, c):
    if val.inferred_type == "string" or pd._libs.lib.infer_dtype(val, True) == "string":
        # Index.inferred_type doesn't skip NAs so we call infer_dtype with
        # skipna=True
        return StringIndexType(get_val_type_maybe_str_literal(val.name))

    # XXX: assume string data type for empty Index with object dtype
    if val.equals(pd.Index([])):
        return StringIndexType(get_val_type_maybe_str_literal(val.name))

    # catch-all for non-supported Index types
    # RangeIndex is directly supported (TODO: make sure this is not called)
    raise NotImplementedError("unsupported pd.Index type")


# -------------------------  DatetimeIndex -----------------------------


class DatetimeIndexType(types.IterableType):
    """type class for DatetimeIndex objects.
    """

    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like freq/tz/dtype/yearfirst?
        self.name_typ = name_typ
        super(DatetimeIndexType, self).__init__(
            name="DatetimeIndex(name = {})".format(name_typ)
        )

    ndim = 1

    def copy(self):
        return DatetimeIndexType(self.name_typ)

    @property
    def key(self):
        # needed?
        return self.name_typ

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timestamp
        return types.iterators.ArrayIterator(_dt_index_data_typ)


@typeof_impl.register(pd.DatetimeIndex)
def typeof_datetime_index(val, c):
    # TODO: check value for freq, tz, etc. and raise error since unsupported
    return DatetimeIndexType(get_val_type_maybe_str_literal(val.name))


@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: use payload to support mutable name
        members = [("data", _dt_index_data_typ), ("name", fe_type.name_typ)]
        super(DatetimeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DatetimeIndexType, "data", "_data")
make_attribute_wrapper(DatetimeIndexType, "name", "_name")


@box(DatetimeIndexType)
def box_dt_index(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    dt_index = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    arr = c.pyapi.from_native_value(_dt_index_data_typ, dt_index.data, c.env_manager)
    name = c.pyapi.from_native_value(typ.name_typ, dt_index.name, c.env_manager)

    # call pd.DatetimeIndex(arr, name=name)
    kws = c.pyapi.dict_pack([("name", name)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "DatetimeIndex")
    res = c.pyapi.call(const_call, c.pyapi.tuple_pack([arr]), kws)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(kws)
    return res


@unbox(DatetimeIndexType)
def unbox_datetime_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    data = c.pyapi.to_native_value(
        _dt_index_data_typ, c.pyapi.object_getattr_string(val, "values")
    ).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, "name")
    ).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


@intrinsic
def init_datetime_index(typingctx, data, name=None):
    """Create a DatetimeIndex with provided data and name values.
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create dt_index struct and store values
        dt_index = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        dt_index.data = data_val
        dt_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], name_val)

        return dt_index._getvalue()

    ret_typ = DatetimeIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


# support DatetimeIndex date fields such as I.year
def gen_dti_field_impl(field):
    # TODO: NaN
    func_text = "def impl(dti):\n"
    func_text += "    numba.parfor.init_prange()\n"
    func_text += "    A = bodo.hiframes.pd_index_ext.get_index_data(dti)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(dti)\n"
    func_text += "    n = len(A)\n"
    # all datetimeindex fields return int64 same as Timestamp fields
    func_text += "    S = np.empty(n, np.int64)\n"
    func_text += "    for i in numba.parfor.internal_prange(n):\n"
    func_text += "        dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])\n"
    func_text += "        ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n"
    func_text += "        S[i] = ts." + field + "\n"
    func_text += "    return bodo.hiframes.pd_index_ext.init_numeric_index(S, name)\n"
    loc_vars = {}
    # print(func_text)
    exec(func_text, {"numba": numba, "np": np, "bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _install_dti_date_fields():
    for field in bodo.hiframes.pd_timestamp_ext.date_fields:
        impl = gen_dti_field_impl(field)
        overload_attribute(DatetimeIndexType, field)(lambda dti: impl)


_install_dti_date_fields()


@overload_attribute(DatetimeIndexType, "date")
def overload_datetime_index_date(dti):
    # TODO: NaN

    def impl(dti):  # pragma: no cover
        numba.parfor.init_prange()
        A = bodo.hiframes.pd_index_ext.get_index_data(dti)
        n = len(A)
        S = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
        for i in numba.parfor.internal_prange(n):
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            S[i] = datetime.date(ts.year, ts.month, ts.day)
        return S

    return impl


@numba.njit
def _dti_val_finalize(s, count):  # pragma: no cover
    if not count:
        s = iNaT  # TODO: NaT type boxing in timestamp
    return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)


@overload_method(DatetimeIndexType, "min")
def overload_datetime_index_min(dti, axis=None, skipna=True):
    # TODO skipna = False
    if not is_overload_none(axis) or not is_overload_true(skipna):
        raise ValueError("Index.min(): axis and skipna arguments not supported yet")

    def impl(dti, axis=None, skipna=True):  # pragma: no cover
        numba.parfor.init_prange()
        in_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
        s = numba.targets.builtins.get_type_max_value(numba.types.int64)
        count = 0
        for i in numba.parfor.internal_prange(len(in_arr)):
            if not bodo.libs.array_kernels.isna(in_arr, i):
                val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                s = min(s, val)
                count += 1
        return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

    return impl


# TODO: refactor min/max
@overload_method(DatetimeIndexType, "max")
def overload_datetime_index_max(dti, axis=None, skipna=True):
    # TODO skipna = False
    if not is_overload_none(axis) or not is_overload_true(skipna):
        raise ValueError("Index.max(): axis and skipna arguments not supported yet")

    def impl(dti, axis=None, skipna=True):  # pragma: no cover
        numba.parfor.init_prange()
        in_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
        s = numba.targets.builtins.get_type_min_value(numba.types.int64)
        count = 0
        for i in numba.parfor.internal_prange(len(in_arr)):
            if not bodo.libs.array_kernels.isna(in_arr, i):
                val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                s = max(s, val)
                count += 1
        return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

    return impl


@infer_getattr
class DatetimeIndexAttribute(AttributeTemplate):
    key = DatetimeIndexType

    def resolve_values(self, ary):
        return _dt_index_data_typ


@overload(pd.DatetimeIndex)
def pd_datetimeindex_overload(
    data=None,
    freq=None,
    start=None,
    end=None,
    periods=None,
    tz=None,
    normalize=False,
    closed=None,
    ambiguous="raise",
    dayfirst=False,
    yearfirst=False,
    dtype=None,
    copy=False,
    name=None,
    verify_integrity=True,
):
    # TODO: check/handle other input
    if is_overload_none(data):
        raise ValueError("data argument in pd.DatetimeIndex() expected")

    # check unsupported, TODO: normalize, dayfirst, yearfirst, ...
    if any(not is_overload_none(a) for a in (freq, start, end, periods, tz, closed)):
        raise ValueError("only data argument in pd.DatetimeIndex() supported")

    def f(
        data=None,
        freq=None,
        start=None,
        end=None,
        periods=None,
        tz=None,
        normalize=False,
        closed=None,
        ambiguous="raise",
        dayfirst=False,
        yearfirst=False,
        dtype=None,
        copy=False,
        name=None,
        verify_integrity=True,
    ):  # pragma: no cover
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        S = bodo.utils.conversion.convert_to_dt64ns(data_arr)
        return bodo.hiframes.pd_index_ext.init_datetime_index(S, name)

    return f


@overload(operator.sub)
def overload_datetime_index_sub(arg1, arg2):
    # DatetimeIndex - Timestamp
    if (
        isinstance(arg1, DatetimeIndexType)
        and arg2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
    ):

        def impl(arg1, arg2):  # pragma: no cover
            numba.parfor.init_prange()
            in_arr = bodo.hiframes.pd_index_ext.get_index_data(arg1)
            name = bodo.hiframes.pd_index_ext.get_index_name(arg1)
            n = len(in_arr)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            tsint = arg2.value
            for i in numba.parfor.internal_prange(n):
                S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]) - tsint
                )
            return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

        return impl

    # Timestamp - DatetimeIndex
    if (
        isinstance(arg2, DatetimeIndexType)
        and arg1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
    ):

        def impl(arg1, arg2):  # pragma: no cover
            numba.parfor.init_prange()
            in_arr = bodo.hiframes.pd_index_ext.get_index_data(arg2)
            name = bodo.hiframes.pd_index_ext.get_index_name(arg2)
            n = len(in_arr)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            tsint = arg1.value
            for i in numba.parfor.internal_prange(n):
                S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    tsint - bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                )
            return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

        return impl


# bionp of DatetimeIndex and string
def gen_dti_str_binop_impl(op, is_arg1_dti):
    # is_arg1_dti: is the first argument DatetimeIndex and second argument str
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def impl(arg1, arg2):\n"
    if is_arg1_dti:
        func_text += "  dt_index, _str = arg1, arg2\n"
        comp = "arr[i] {} other".format(op_str)
    else:
        func_text += "  dt_index, _str = arg2, arg1\n"
        comp = "other {} arr[i]".format(op_str)
    func_text += "  arr = bodo.hiframes.pd_index_ext.get_index_data(dt_index)\n"
    func_text += "  l = len(arr)\n"
    func_text += "  other = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n"
    func_text += "  S = numba.unsafe.ndarray.empty_inferred((l,))\n"
    func_text += "  for i in numba.parfor.internal_prange(l):\n"
    func_text += "    S[i] = {}\n".format(comp)
    func_text += "  return S\n"
    # print(func_text)
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def overload_binop_dti_str(op):
    def overload_impl(arg1, arg2):
        if isinstance(arg1, DatetimeIndexType) and types.unliteral(arg2) == string_type:
            return gen_dti_str_binop_impl(op, True)
        if isinstance(arg2, DatetimeIndexType) and types.unliteral(arg1) == string_type:
            return gen_dti_str_binop_impl(op, False)

    return overload_impl


def _install_dti_str_comp_ops():
    for op in (
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ):
        overload(op)(overload_binop_dti_str(op))


_install_dti_str_comp_ops()


@overload(operator.getitem)
def overload_datetime_index_getitem(dti, ind):
    # TODO: other getitem cases
    if isinstance(dti, DatetimeIndexType):
        if isinstance(ind, types.Integer):

            def impl(dti, ind):  # pragma: no cover
                dti_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
                dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dti_arr[ind])
                return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                    dt64
                )

            return impl
        else:
            # slice, boolean array, etc.
            # TODO: other Index or Series objects as index?
            def impl(dti, ind):  # pragma: no cover
                dti_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
                name = bodo.hiframes.pd_index_ext.get_index_name(dti)
                new_arr = dti_arr[ind]
                return bodo.hiframes.pd_index_ext.init_datetime_index(new_arr, name)

            return impl


# from pandas.core.arrays.datetimelike
@numba.njit
def validate_endpoints(closed):  # pragma: no cover
    """
    Check that the `closed` argument is among [None, "left", "right"]

    Parameters
    ----------
    closed : {None, "left", "right"}

    Returns
    -------
    left_closed : bool
    right_closed : bool

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")

    return left_closed, right_closed


@numba.njit
def to_offset_value(freq):  # pragma: no cover
    """Converts freq (string and integer) to offset nanoseconds.
    """
    if freq is None:
        return None

    with numba.objmode(r="int64"):
        r = pd.tseries.frequencies.to_offset(freq).nanos
    return r


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def _dummy_convert_none_to_int(val):
    """Dummy function that converts None to integer, used when branch pruning
    fails to remove None branch, causing errors. The conversion path should
    never actually execute.
    """
    if is_overload_none(val):

        def impl(val):  # pragma: no cover
            assert 0
            return 0

        return impl
    return lambda val: val


@overload(pd.date_range)
def pd_date_range_overload(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize=False,
    name=None,
    closed=None,
):
    # TODO: check/handle other input
    # check unsupported, TODO: normalize, dayfirst, yearfirst, ...
    # TODO: parallelize after Numba branch pruning issue is fixed
    if not is_overload_none(tz):
        raise ValueError("pd.date_range(): tz argument not supported yet")

    if is_overload_none(freq) and any(
        is_overload_none(t) for t in (start, end, periods)
    ):
        freq = "D"  # change just to enable check below

    # exactly three parameters should
    if sum(not is_overload_none(t) for t in (start, end, periods, freq)) != 3:
        raise ValueError(
            "Of the four parameters: start, end, periods, "
            "and freq, exactly three must be specified"
        )

    def f(
        start=None,
        end=None,
        periods=None,
        freq=None,
        tz=None,
        normalize=False,
        name=None,
        closed=None,
    ):  # pragma: no cover

        if freq is None and (start is None or end is None or periods is None):
            freq = "D"

        freq = bodo.hiframes.pd_index_ext.to_offset_value(freq)

        start_t = pd.Timestamp("2018-01-01")  # dummy value for typing
        if start is not None:
            start_t = pd.Timestamp(start)

        end_t = pd.Timestamp("2018-01-01")  # dummy value for typing
        if end is not None:
            end_t = pd.Timestamp(end)

        if start is None and end is None and closed is not None:
            raise ValueError(
                "Closed has to be None if not both of start" "and end are defined"
            )
        # TODO: check start and end for NaT
        # if start is NaT or end is NaT:
        #     raise ValueError("Neither `start` nor `end` can be NaT")

        left_closed, right_closed = bodo.hiframes.pd_index_ext.validate_endpoints(
            closed
        )

        if freq is not None:
            # pandas/core/arrays/_ranges/generate_regular_range
            # TODO: handle overflows
            stride = freq
            if periods is None:
                b = start_t.value
                e = b + (end_t.value - b) // stride * stride + stride // 2 + 1
            elif start is not None:
                b = start_t.value
                addend = np.int64(periods) * np.int64(stride)
                e = np.int64(b) + addend
            elif end is not None:
                e = end_t.value + stride
                addend = np.int64(periods) * np.int64(-stride)
                b = np.int64(e) + addend
            else:
                raise ValueError(
                    "at least 'start' or 'end' should be specified "
                    "if a 'period' is given."
                )

            # TODO: handle overflows
            arr = np.arange(b, e, stride, np.int64)
        else:
            # TODO: fix Numba's linspace to support dtype
            # arr = np.linspace(
            #     0, end_t.value - start_t.value,
            #     periods, dtype=np.int64) + start.value
            # XXX Numba's branch pruning fails to remove period=None so use
            # dummy function
            # TODO: fix Numba's branch pruning pass
            # using Numpy's linspace algorithm
            periods = _dummy_convert_none_to_int(periods)
            delta = end_t.value - start_t.value
            step = delta / (periods - 1)
            arr1 = np.arange(0, periods, 1, np.float64)
            arr1 *= step
            arr1 += start_t.value
            arr = arr1.astype(np.int64)
            arr[-1] = end_t.value

        if not left_closed and len(arr) and arr[0] == start_t.value:
            arr = arr[1:]
        if not right_closed and len(arr) and arr[-1] == end_t.value:
            arr = arr[:-1]

        S = bodo.utils.conversion.convert_to_dt64ns(arr)
        return bodo.hiframes.pd_index_ext.init_datetime_index(S, name)

    return f


# ------------------------------ Timedelta ---------------------------


# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType):
    """Temporary type class for TimedeltaIndex objects.
    """

    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like unit/freq?
        self.name_typ = name_typ
        super(TimedeltaIndexType, self).__init__(
            name="TimedeltaIndexType(named = {})".format(name_typ)
        )

    ndim = 1

    def copy(self):
        return TimedeltaIndexType(self.name_typ)

    @property
    def key(self):
        # needed?
        return self.name_typ

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timedelta
        return types.iterators.ArrayIterator(_timedelta_index_data_typ)


@register_model(TimedeltaIndexType)
class TimedeltaIndexTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", _timedelta_index_data_typ), ("name", fe_type.name_typ)]
        super(TimedeltaIndexTypeModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(pd.TimedeltaIndex)
def typeof_timedelta_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return TimedeltaIndexType(get_val_type_maybe_str_literal(val.name))


@box(TimedeltaIndexType)
def box_timedelta_index(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    timedelta_index = numba.cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    arr = c.pyapi.from_native_value(
        _timedelta_index_data_typ, timedelta_index.data, c.env_manager
    )
    name = c.pyapi.from_native_value(typ.name_typ, timedelta_index.name, c.env_manager)

    # call pd.TimedeltaIndex(arr, name=name)
    kws = c.pyapi.dict_pack([("name", name)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "TimedeltaIndex")
    res = c.pyapi.call(const_call, c.pyapi.tuple_pack([arr]), kws)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(kws)
    return res


@unbox(TimedeltaIndexType)
def unbox_timedelta_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    data = c.pyapi.to_native_value(
        _timedelta_index_data_typ, c.pyapi.object_getattr_string(val, "values")
    ).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, "name")
    ).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


@intrinsic
def init_timedelta_index(typingctx, data, name=None):
    """Create a TimedeltaIndex with provided data and name values.
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create timedelta_index struct and store values
        timedelta_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        timedelta_index.data = data_val
        timedelta_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], name_val)

        return timedelta_index._getvalue()

    ret_typ = TimedeltaIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@infer_getattr
class TimedeltaIndexAttribute(AttributeTemplate):
    key = TimedeltaIndexType

    def resolve_values(self, ary):
        return _timedelta_index_data_typ

    # TODO: support pd.Timedelta
    # @bound_function("timedelta_index.max")
    # def resolve_max(self, ary, args, kws):
    #     assert not kws
    #     return signature(pandas_timestamp_type, *args)

    # @bound_function("timedelta_index.min")
    # def resolve_min(self, ary, args, kws):
    #     assert not kws
    #     return signature(pandas_timestamp_type, *args)


make_attribute_wrapper(TimedeltaIndexType, "data", "_data")
make_attribute_wrapper(TimedeltaIndexType, "name", "_name")


# support TimedeltaIndex time fields such as T.days
def gen_tdi_field_impl(field):
    # TODO: NaN
    func_text = "def impl(tdi):\n"
    func_text += "    numba.parfor.init_prange()\n"
    func_text += "    A = bodo.hiframes.pd_index_ext.get_index_data(tdi)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(tdi)\n"
    func_text += "    n = len(A)\n"
    # all timedeltaindex fields return int64 same as Timestamp fields
    func_text += "    S = np.empty(n, np.int64)\n"
    func_text += "    for i in numba.parfor.internal_prange(n):\n"
    func_text += (
        "        td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(A[i])\n"
    )
    if field == "nanoseconds":
        func_text += "        S[i] = td64 % 1000\n"
    elif field == "microseconds":
        func_text += "        S[i] = td64 // 1000 % 100000\n"
    elif field == "seconds":
        func_text += "        S[i] = td64 // (1000 * 1000000) % (60 * 60 * 24)\n"
    elif field == "days":
        func_text += "        S[i] = td64 // (1000 * 1000000 * 60 * 60 * 24)\n"
    else:
        assert False, "invalid timedelta field"
    func_text += "    return bodo.hiframes.pd_index_ext.init_numeric_index(S, name)\n"
    loc_vars = {}
    # print(func_text)
    exec(func_text, {"numba": numba, "np": np, "bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _install_tdi_time_fields():
    for field in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
        impl = gen_tdi_field_impl(field)
        overload_attribute(TimedeltaIndexType, field)(lambda tdi: impl)


_install_tdi_time_fields()


@overload(pd.TimedeltaIndex)
def pd_timedelta_index_overload(
    data=None,
    unit=None,
    freq=None,
    start=None,
    end=None,
    periods=None,
    closed=None,
    dtype=None,
    copy=False,
    name=None,
    verify_integrity=None,
):
    # TODO handle dtype=dtype('<m8[ns]') default
    # TODO: check/handle other input
    if is_overload_none(data):
        raise ValueError("data argument in pd.TimedeltaIndex() expected")

    if any(
        not is_overload_none(a)
        for a in (unit, freq, start, end, periods, closed, dtype)
    ):
        raise ValueError("only data argument in pd.TimedeltaIndex() supported")

    def impl(
        data=None,
        unit=None,
        freq=None,
        start=None,
        end=None,
        periods=None,
        closed=None,
        dtype=None,
        copy=False,
        name=None,
        verify_integrity=None,
    ):  # pragma: no cover
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        S = bodo.utils.conversion.convert_to_td64ns(data_arr)
        return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

    return impl


# ---------------- RangeIndex -------------------


# pd.RangeIndex(): simply keep start/stop/step/name
class RangeIndexType(types.IterableType):
    """type class for pd.RangeIndex() objects.
    """

    def __init__(self, name_typ):
        self.name_typ = name_typ
        super(RangeIndexType, self).__init__(name="RangeIndexType({})".format(name_typ))

    ndim = 1

    def copy(self):
        return RangeIndexType(self.name_typ)

    @property
    def iterator_type(self):
        return types.iterators.RangeIteratorType(types.int64)

    @property
    def dtype(self):
        return types.int64


@typeof_impl.register(pd.RangeIndex)
def typeof_pd_range_index(val, c):
    return RangeIndexType(get_val_type_maybe_str_literal(val.name))


@register_model(RangeIndexType)
class RangeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("start", types.int64),
            ("stop", types.int64),
            ("step", types.int64),
            ("name", fe_type.name_typ),
        ]
        super(RangeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(RangeIndexType, "start", "_start")
make_attribute_wrapper(RangeIndexType, "stop", "_stop")
make_attribute_wrapper(RangeIndexType, "step", "_step")
make_attribute_wrapper(RangeIndexType, "name", "_name")


@box(RangeIndexType)
def box_range_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    range_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    start = c.pyapi.from_native_value(types.int64, range_val.start, c.env_manager)
    stop = c.pyapi.from_native_value(types.int64, range_val.stop, c.env_manager)
    step = c.pyapi.from_native_value(types.int64, range_val.step, c.env_manager)
    name = c.pyapi.from_native_value(typ.name_typ, range_val.name, c.env_manager)
    kws = c.pyapi.dict_pack([("name", name)])
    const_call = c.pyapi.object_getattr_string(class_obj, "RangeIndex")
    index_obj = c.pyapi.call(const_call, c.pyapi.tuple_pack([start, stop, step]), kws)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(kws)
    return index_obj


@intrinsic
def init_range_index(typingctx, start, stop, step, name=None):
    """Create RangeIndex object
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        assert len(args) == 4
        range_val = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        range_val.start = args[0]
        range_val.stop = args[1]
        range_val.step = args[2]
        range_val.name = args[3]
        return range_val._getvalue()

    return RangeIndexType(name)(start, stop, step, name), codegen


@unbox(RangeIndexType)
def unbox_range_index(typ, val, c):
    # get start/stop/step attributes
    start = c.pyapi.to_native_value(
        types.int64, c.pyapi.object_getattr_string(val, "start")
    ).value
    stop = c.pyapi.to_native_value(
        types.int64, c.pyapi.object_getattr_string(val, "stop")
    ).value
    step = c.pyapi.to_native_value(
        types.int64, c.pyapi.object_getattr_string(val, "step")
    ).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, "name")
    ).value

    # create range struct
    range_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    range_val.start = start
    range_val.stop = stop
    range_val.step = step
    range_val.name = name
    return NativeValue(range_val._getvalue())


@overload(pd.RangeIndex)
def range_index_overload(
    start=None, stop=None, step=None, dtype=None, copy=False, name=None, fastpath=None
):

    # validate the arguments
    def _ensure_int_or_none(value, field):
        msg = (
            "RangeIndex(...) must be called with integers,"
            " {value} was passed for {field}"
        )
        if (
            not is_overload_none(value)
            and not isinstance(value, types.IntegerLiteral)
            and not isinstance(value, types.Integer)
        ):
            raise TypeError(msg.format(value=value, field=field))

    _ensure_int_or_none(start, "start")
    _ensure_int_or_none(stop, "stop")
    _ensure_int_or_none(step, "step")

    # all none error case
    if is_overload_none(start) and is_overload_none(stop) and is_overload_none(step):
        msg = "RangeIndex(...) must be called with integers"
        raise TypeError(msg)

    # codegen the init function
    _start = "start"
    _stop = "stop"
    _step = "step"

    if is_overload_none(start):
        _start = "0"
    if is_overload_none(stop):
        _stop = "start"
        _start = "0"
    if is_overload_none(step):
        _step = "1"

    func_text = "def _pd_range_index_imp(start=None, stop=None, step=None, dtype=None, copy=False, name=None, fastpath=None):\n"
    func_text += "  return init_range_index({}, {}, {}, name)\n".format(
        _start, _stop, _step
    )
    loc_vars = {}
    exec(func_text, {"init_range_index": init_range_index}, loc_vars)
    # print(func_text)
    _pd_range_index_imp = loc_vars["_pd_range_index_imp"]
    return _pd_range_index_imp


@overload(operator.getitem)
def overload_range_index_getitem(I, idx):
    if isinstance(I, RangeIndexType):
        if isinstance(idx, types.Integer):
            # TODO: test
            # TODO: check valid
            return lambda I, idx: (idx * I._step) + I._start

        if isinstance(idx, types.SliceType):
            # TODO: test
            def impl(I, idx):  # pragma: no cover
                slice_idx = numba.unicode._normalize_slice(idx, len(I))
                start = I._start + I._step * slice_idx.start
                stop = I._start + I._step * slice_idx.stop
                step = I._step * slice_idx.step
                return bodo.hiframes.pd_index_ext.init_range_index(
                    start, stop, step, None
                )

            return impl

        # delegate to integer index, TODO: test
        return lambda I, idx: bodo.hiframes.pd_index_ext.init_numeric_index(
            np.arange(I._start, I._stop, I._step, np.int64)[idx]
        )


@overload(len)
def overload_range_len(r):
    if isinstance(r, RangeIndexType):
        # TODO: test
        return lambda r: max(0, -(-(r._stop - r._start) // r._step))


# ---------------- PeriodIndex -------------------


# Simple type for PeriodIndex for now, freq is saved as a constant string
class PeriodIndexType(types.IterableType):
    """type class for pd.PeriodIndex. Contains frequency as constant string
    """

    def __init__(self, freq, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.freq = freq
        self.name_typ = name_typ
        super(PeriodIndexType, self).__init__(
            name="PeriodIndexType({}, {})".format(freq, name_typ)
        )

    ndim = 1

    def copy(self):
        return PeriodIndexType(self.freq, self.name_typ)

    @property
    def iterator_type(self):
        # TODO: handle iterator
        return types.iterators.ArrayIterator(types.Array(types.int64, 1, "C"))


@typeof_impl.register(pd.PeriodIndex)
def typeof_pd_period_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return PeriodIndexType(val.freqstr, get_val_type_maybe_str_literal(val.name))


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(PeriodIndexType)
class PeriodIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: nullable integer array?
        members = [
            ("data", types.Array(types.int64, 1, "C")),
            ("name", fe_type.name_typ),
        ]
        super(PeriodIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(PeriodIndexType, "data", "_data")
make_attribute_wrapper(PeriodIndexType, "name", "_name")


@box(PeriodIndexType)
def box_period_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data = c.pyapi.from_native_value(
        types.Array(types.int64, 1, "C"), index_val.data, c.env_manager
    )
    name = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    freq = c.pyapi.string_from_constant_string(typ.freq)

    # call pd.PeriodIndex(ordinal=data, name=name, freq=freq)
    kws = c.pyapi.dict_pack([("ordinal", data), ("name", name), ("freq", freq)])
    const_call = c.pyapi.object_getattr_string(class_obj, "PeriodIndex")
    index_obj = c.pyapi.call(const_call, c.pyapi.tuple_pack([]), kws)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(kws)
    return index_obj


@unbox(PeriodIndexType)
def unbox_period_index(typ, val, c):
    # get data and name attributes
    arr_typ = types.Array(types.int64, 1, "C")
    data = c.pyapi.to_native_value(
        arr_typ, c.pyapi.object_getattr_string(val, "asi8")
    ).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, "name")
    ).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


# ---------------- NumericIndex -------------------


# represents numeric indices (excluding RangeIndex):
#   Int64Index, UInt64Index, Float64Index
class NumericIndexType(types.IterableType):
    """type class for pd.Int64Index/UInt64Index/Float64Index objects.
    """

    def __init__(self, dtype, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.dtype = dtype
        self.name_typ = name_typ
        super(NumericIndexType, self).__init__(
            name="NumericIndexType({}, {})".format(dtype, name_typ)
        )

    ndim = 1

    def copy(self):
        return NumericIndexType(self.dtype, self.name_typ)

    @property
    def iterator_type(self):
        # TODO: handle iterator
        return types.iterators.ArrayIterator(types.Array(self.dtype, 1, "C"))


@typeof_impl.register(pd.Int64Index)
def typeof_pd_int64_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return NumericIndexType(types.int64, get_val_type_maybe_str_literal(val.name))


@typeof_impl.register(pd.UInt64Index)
def typeof_pd_uint64_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return NumericIndexType(types.uint64, get_val_type_maybe_str_literal(val.name))


@typeof_impl.register(pd.Float64Index)
def typeof_pd_float64_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return NumericIndexType(types.float64, get_val_type_maybe_str_literal(val.name))


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(NumericIndexType)
class NumericIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: nullable integer array (e.g. to hold DatetimeIndex.year)
        members = [
            ("data", types.Array(fe_type.dtype, 1, "C")),
            ("name", fe_type.name_typ),
        ]
        super(NumericIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(NumericIndexType, "data", "_data")
make_attribute_wrapper(NumericIndexType, "name", "_name")


@box(NumericIndexType)
def box_numeric_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data = c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, "C"), index_val.data, c.env_manager
    )
    name = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    assert typ.dtype in (types.int64, types.uint64, types.float64)
    func_name = "Int64Index"
    if typ.dtype == types.uint64:
        func_name = "UInt64Index"
    elif typ.dtype == types.float64:
        func_name = "Float64Index"
    else:
        assert typ.dtype == types.int64

    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    index_obj = c.pyapi.call_method(class_obj, func_name, (data, dtype, copy, name))
    c.pyapi.decref(class_obj)
    return index_obj


@intrinsic
def init_numeric_index(typingctx, data, name=None):
    """Create NumericIndex object
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        assert len(args) == 2
        index_typ = signature.return_type
        index_val = cgutils.create_struct_proxy(index_typ)(context, builder)
        index_val.data = args[0]
        index_val.name = args[1]
        # increase refcount of stored values
        if context.enable_nrt:
            arr_typ = types.Array(index_typ.dtype, 1, "C")
            context.nrt.incref(builder, arr_typ, args[0])
            context.nrt.incref(builder, index_typ.name_typ, args[1])
        return index_val._getvalue()

    return NumericIndexType(data.dtype, name)(data, name), codegen


@unbox(NumericIndexType)
def unbox_numeric_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    arr_typ = types.Array(typ.dtype, 1, "C")
    data = c.pyapi.to_native_value(
        arr_typ, c.pyapi.object_getattr_string(val, "values")
    ).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, "name")
    ).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


def create_numeric_constructor(func, default_dtype):
    def overload_impl(data=None, dtype=None, copy=False, name=None, fastpath=None):
        if is_overload_false(copy):
            # if copy is False for sure, specialize to avoid branch
            def impl(
                data=None, dtype=None, copy=False, name=None, fastpath=None
            ):  # pragma: no cover
                data_arr = bodo.utils.conversion.coerce_to_ndarray(data)
                data_res = bodo.utils.conversion.fix_arr_dtype(
                    data_arr, np.dtype(default_dtype)
                )
                return bodo.hiframes.pd_index_ext.init_numeric_index(data_res, name)

        else:

            def impl(
                data=None, dtype=None, copy=False, name=None, fastpath=None
            ):  # pragma: no cover
                data_arr = bodo.utils.conversion.coerce_to_ndarray(data)
                if copy:
                    data_arr = data_arr.copy()  # TODO: np.array() with copy
                data_res = bodo.utils.conversion.fix_arr_dtype(
                    data_arr, np.dtype(default_dtype)
                )
                return bodo.hiframes.pd_index_ext.init_numeric_index(data_res, name)

        return impl

    return overload_impl


def _install_numeric_constructors():
    for func, default_dtype in (
        (pd.Int64Index, np.int64),
        (pd.UInt64Index, np.uint64),
        (pd.Float64Index, np.float64),
    ):
        overload_impl = create_numeric_constructor(func, default_dtype)
        overload(func)(overload_impl)


_install_numeric_constructors()


# ---------------- StringIndex -------------------


# represents string index, which doesn't have direct Pandas type
# pd.Index() infers string
class StringIndexType(types.IterableType):
    """type class for pd.Index() objects with 'string' as inferred_dtype.
    """

    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        super(StringIndexType, self).__init__(
            name="StringIndexType({})".format(name_typ)
        )

    ndim = 1

    def copy(self):
        return StringIndexType(self.name_typ)

    @property
    def iterator_type(self):
        # TODO: handle iterator
        return bodo.libs.str_arr_ext.StringArrayIterator()


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(StringIndexType)
class StringIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", string_array_type), ("name", fe_type.name_typ)]
        super(StringIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(StringIndexType, "data", "_data")
make_attribute_wrapper(StringIndexType, "name", "_name")


@box(StringIndexType)
def box_string_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    data = c.pyapi.from_native_value(string_array_type, index_val.data, c.env_manager)
    name = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    index_obj = c.pyapi.call_method(class_obj, "Index", (data, dtype, copy, name))
    c.pyapi.decref(class_obj)
    return index_obj


@intrinsic
def init_string_index(typingctx, data, name=None):
    """Create StringIndex object
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        assert len(args) == 2
        index_typ = signature.return_type
        index_val = cgutils.create_struct_proxy(index_typ)(context, builder)
        index_val.data = args[0]
        index_val.name = args[1]
        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, string_array_type, args[0])
            context.nrt.incref(builder, index_typ.name_typ, args[1])
        return index_val._getvalue()

    return StringIndexType(name)(data, name), codegen


@unbox(StringIndexType)
def unbox_string_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    data = c.pyapi.to_native_value(
        string_array_type, c.pyapi.object_getattr_string(val, "values")
    ).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, "name")
    ).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


@overload(operator.getitem)
def overload_index_getitem(I, ind):
    if isinstance(I, (NumericIndexType, StringIndexType)):
        return lambda I, ind: bodo.hiframes.pd_index_ext.get_index_data(I)[ind]


# similar to index_from_array()
def array_typ_to_index(arr_typ, name_typ=None):
    if arr_typ == bodo.string_array_type:
        return StringIndexType(name_typ)

    assert isinstance(arr_typ, types.Array) or isinstance(arr_typ, IntegerArrayType)
    if arr_typ.dtype == types.NPDatetime("ns"):
        return DatetimeIndexType(name_typ)

    if arr_typ.dtype == types.NPTimedelta("ns"):
        return TimedeltaIndexType(name_typ)

    if isinstance(arr_typ.dtype, types.Integer):
        if not arr_typ.dtype.signed:
            return NumericIndexType(types.uint64, name_typ)
        else:
            return NumericIndexType(types.int64, name_typ)

    if isinstance(arr_typ.dtype, types.Float):
        return NumericIndexType(types.float64, name_typ)

    raise TypeError("invalid index type {}".format(arr_typ))


def is_pd_index_type(t):
    return isinstance(
        t,
        (
            NumericIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            PeriodIndexType,
            StringIndexType,
            RangeIndexType,
        ),
    )


# TODO: test
@overload_method(RangeIndexType, "take")
@overload_method(NumericIndexType, "take")
@overload_method(StringIndexType, "take")
@overload_method(PeriodIndexType, "take")
@overload_method(DatetimeIndexType, "take")
@overload_method(TimedeltaIndexType, "take")
def overload_index_take(I, indices):
    return lambda I, indices: I[indices]


@overload_method(RangeIndexType, "isna")
@overload_method(NumericIndexType, "isna")
@overload_method(StringIndexType, "isna")
@overload_method(PeriodIndexType, "isna")
@overload_method(DatetimeIndexType, "isna")
@overload_method(TimedeltaIndexType, "isna")
@overload_method(RangeIndexType, "isnull")
@overload_method(NumericIndexType, "isnull")
@overload_method(StringIndexType, "isnull")
@overload_method(PeriodIndexType, "isnull")
@overload_method(DatetimeIndexType, "isnull")
@overload_method(TimedeltaIndexType, "isnull")
def overload_index_isna(I):
    if isinstance(I, RangeIndexType):
        # TODO: parallelize np.full in PA
        # return lambda I: np.full(len(I), False, np.bool_)
        def impl(I):  # pragma: no cover
            numba.parfor.init_prange()
            n = len(I)
            out_arr = np.empty(n, np.bool_)
            for i in numba.parfor.internal_prange(n):
                out_arr[i] = False
            return out_arr

        return impl

    def impl(I):  # pragma: no cover
        numba.parfor.init_prange()
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfor.internal_prange(n):
            out_arr[i] = bodo.libs.array_kernels.isna(arr, i)
        return out_arr

    return impl


@overload(len)
def overload_index_len(I):
    if isinstance(
        I,
        (
            NumericIndexType,
            StringIndexType,
            PeriodIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
        ),
    ):
        # TODO: test
        return lambda I: len(bodo.hiframes.pd_index_ext.get_index_data(I))


@numba.generated_jit(nopython=True)
def get_index_data(S):
    return lambda S: S._data


@numba.generated_jit(nopython=True)
def get_index_name(S):
    return lambda S: S._name


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.ir_utils.alias_func_extensions[
    ("get_index_data", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_datetime_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_timedelta_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_numeric_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_string_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
