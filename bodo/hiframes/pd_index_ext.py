import operator
import pandas as pd
import numpy as np
import numba
from numba import types, cgutils
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper, box, intrinsic,
    typeof_impl, unbox, NativeValue)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)
from numba.targets.boxing import box_array

import bodo
from bodo.libs.str_ext import string_type
import bodo.hiframes
from bodo.hiframes.pd_series_ext import (is_str_series_typ, string_array_type,
    SeriesType)
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type, datetime_date_type
from bodo.hiframes.datetime_date_ext import array_datetime_date


_dt_index_data_typ = types.Array(types.NPDatetime('ns'), 1, 'C')
_timedelta_index_data_typ = types.Array(types.NPTimedelta('ns'), 1, 'C')


# -------------------------  DatetimeIndex -----------------------------


class DatetimeIndexType(types.IterableType):
    """type class for DatetimeIndex objects.
    """
    def __init__(self, name_typ=None):
        # TODO: support other properties like freq/tz/dtype/yearfirst?
        self.name_typ = name_typ
        super(DatetimeIndexType, self).__init__(
            name="DatetimeIndex(name = {})".format(name_typ))

    def copy(self):
        # XXX is copy necessary?
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
    return DatetimeIndexType(numba.typeof(val.name))


@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: use payload to support mutable name
        members = [
            ('data', _dt_index_data_typ),
            ('name', fe_type.name_typ),
        ]
        super(DatetimeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DatetimeIndexType, 'data', '_data')
make_attribute_wrapper(DatetimeIndexType, 'name', '_name')


@box(DatetimeIndexType)
def box_dt_index(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    dt_index = numba.cgutils.create_struct_proxy(
            typ)(c.context, c.builder, val)

    arr = c.pyapi.from_native_value(
        _dt_index_data_typ, dt_index.data, c.env_manager)
    name = c.pyapi.from_native_value(typ.name_typ, dt_index.name)

    # call pd.DatetimeIndex(arr, name=name)
    kws = c.pyapi.dict_pack([('name', name)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, 'DatetimeIndex')
    res = c.pyapi.call(const_call, c.pyapi.tuple_pack([arr]), kws)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    return res


@unbox(DatetimeIndexType)
def unbox_datetime_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    data = c.pyapi.to_native_value(
        _dt_index_data_typ, c.pyapi.object_getattr_string(val, 'values')).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, 'name')).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


@infer_getattr
class DatetimeIndexAttribute(AttributeTemplate):
    key = DatetimeIndexType

    def resolve_values(self, ary):
        return _dt_index_data_typ

    def resolve_date(self, ary):
        return array_datetime_date

    @bound_function("dt_index.max")
    def resolve_max(self, ary, args, kws):
        assert not kws
        return signature(pandas_timestamp_type, *args)

    @bound_function("dt_index.min")
    def resolve_min(self, ary, args, kws):
        assert not kws
        return signature(pandas_timestamp_type, *args)


# all datetimeindex fields return int64 same as Timestamp fields
def resolve_date_field(self, ary):
    # TODO: return Int64Index
    return SeriesType(types.int64)

for field in bodo.hiframes.pd_timestamp_ext.date_fields:
    setattr(DatetimeIndexAttribute, "resolve_" + field, resolve_date_field)



@overload(pd.DatetimeIndex)
def pd_datetimeindex_overload(data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True):
    # TODO: check/handle other input
    if data is None:
        raise ValueError("data argument in pd.DatetimeIndex() expected")

    if data != string_array_type and not is_str_series_typ(data):
        return (lambda data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True: bodo.hiframes.api.init_datetime_index(
            bodo.hiframes.api.ts_series_to_arr_typ(data), name))

    def f(data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True):
        S = bodo.hiframes.api.parse_datetimes_from_strings(data)
        return bodo.hiframes.api.init_datetime_index(S, name)

    return f



# ----------- Timedelta

# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType):
    """Temporary type class for TimedeltaIndex objects.
    """
    def __init__(self, is_named=False):
        # TODO: support other properties like unit/freq?
        self.is_named = is_named
        super(TimedeltaIndexType, self).__init__(
            name="TimedeltaIndexType(is_named = {})".format(is_named))

    def copy(self):
        # XXX is copy necessary?
        return TimedeltaIndexType(self.is_named)

    @property
    def key(self):
        # needed?
        return self.is_named

    def unify(self, typingctx, other):
        # needed?
        return super(TimedeltaIndexType, self).unify(typingctx, other)

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timedelta
        return types.iterators.ArrayIterator(_timedelta_index_data_typ)


@register_model(TimedeltaIndexType)
class TimedeltaIndexTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', _timedelta_index_data_typ),
            ('name', string_type),
        ]
        super(TimedeltaIndexTypeModel, self).__init__(dmm, fe_type, members)

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


# all datetimeindex fields return int64 same as Timestamp fields
def resolve_timedelta_field(self, ary):
    # TODO: return Int64Index
    return types.Array(types.int64, 1, 'C')

for field in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
    setattr(TimedeltaIndexAttribute, "resolve_" + field, resolve_timedelta_field)


# ---------------- RangeIndex -------------------


# pd.RangeIndex(): simply keep start/stop/step
class RangeIndexType(types.IterableType):
    """type class for pd.RangeIndex() objects.
    """
    def __init__(self):
        super(RangeIndexType, self).__init__(
            name="RangeIndexType()")

    @property
    def iterator_type(self):
        return types.iterators.RangeIteratorType(types.int64)


range_index_type = RangeIndexType()


@typeof_impl.register(pd.RangeIndex)
def typeof_pd_range_index(val, c):
    return range_index_type


@register_model(RangeIndexType)
class RangeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('start', types.int64),
            ('stop', types.int64),
            ('step', types.int64),
        ]
        super(RangeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(RangeIndexType, 'start', '_start')
make_attribute_wrapper(RangeIndexType, 'stop', '_stop')
make_attribute_wrapper(RangeIndexType, 'step', '_step')


@box(RangeIndexType)
def box_range_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    range_val = cgutils.create_struct_proxy(range_index_type)(
            c.context, c.builder, val)
    start = c.pyapi.from_native_value(
        types.int64, range_val.start, c.env_manager)
    stop = c.pyapi.from_native_value(
        types.int64, range_val.stop, c.env_manager)
    step = c.pyapi.from_native_value(
        types.int64, range_val.step, c.env_manager)
    range_obj = c.pyapi.call_method(
        class_obj, "RangeIndex", (start, stop, step))
    c.pyapi.decref(class_obj)
    return range_obj


@intrinsic
def init_range_index(typingctx, start, stop, step=None):
    """Create RangeIndex object
    """

    def codegen(context, builder, signature, args):
        assert len(args) == 3
        range_val = cgutils.create_struct_proxy(range_index_type)(
            context, builder)
        range_val.start = args[0]
        range_val.stop = args[1]
        range_val.step = args[2]
        return range_val._getvalue()

    return range_index_type(start, stop, step), codegen


@unbox(RangeIndexType)
def unbox_range_index(typ, val, c):
    # get start/stop/step attributes
    start = c.pyapi.to_native_value(
        types.int64, c.pyapi.object_getattr_string(val, '_start')).value
    stop = c.pyapi.to_native_value(
        types.int64, c.pyapi.object_getattr_string(val, '_stop')).value
    step = c.pyapi.to_native_value(
        types.int64, c.pyapi.object_getattr_string(val, '_step')).value

    # create range struct
    range_val = cgutils.create_struct_proxy(range_index_type)(
            c.context, c.builder)
    range_val.start = start
    range_val.stop = stop
    range_val.step = step
    return NativeValue(range_val._getvalue())


@overload(pd.RangeIndex)
def range_index_overload(start=None, stop=None, step=None, dtype=None,
                         copy=False, name=None, fastpath=None):

    def _is_none(val):
        return val is None or val == types.none

    # validate the arguments
    def _ensure_int_or_none(value, field):
        msg = ("RangeIndex(...) must be called with integers,"
                " {value} was passed for {field}")
        if (not _is_none(value)
                and not isinstance(value, types.IntegerLiteral)
                and not value == types.int64):
            raise TypeError(msg.format(value=value,
                                        field=field))

    _ensure_int_or_none(start, 'start')
    _ensure_int_or_none(stop, 'stop')
    _ensure_int_or_none(step, 'step')

    # all none error case
    if _is_none(start) and _is_none(stop) and _is_none(step):
        msg = "RangeIndex(...) must be called with integers"
        raise TypeError(msg)

    # codegen the init function
    _start = 'start'
    _stop = 'stop'
    _step = 'step'

    if _is_none(start):
        _start = '0'
    if _is_none(stop):
        _stop = 'start'
        _start = '0'
    if _is_none(step):
        _step = '1'

    func_text = "def _pd_range_index_imp(start=None, stop=None, step=None, dtype=None, copy=False, name=None, fastpath=None):\n"
    func_text += "  return init_range_index({}, {}, {})\n".format(_start, _stop, _step)
    loc_vars = {}
    exec(func_text, {'init_range_index': init_range_index}, loc_vars)
    # print(func_text)
    _pd_range_index_imp = loc_vars['_pd_range_index_imp']
    return _pd_range_index_imp


# ---------------- NumericIndex -------------------


# represents numeric indices (excluding RangeIndex):
#   Int64Index, UInt64Index, Float64Index
class NumericIndexType(types.IterableType):
    """type class for pd.Int64Index/UInt64Index/Float64Index objects.
    """
    def __init__(self, dtype, name_typ):
        self.dtype = dtype
        self.name_typ = name_typ
        super(NumericIndexType, self).__init__(
            name="NumericIndexType({}, {})".format(dtype, name_typ))

    @property
    def iterator_type(self):
        # TODO: handle iterator
        return types.iterators.ArrayIterator(types.Array(self.dtype, 1, 'C'))


@typeof_impl.register(pd.Int64Index)
def typeof_pd_int64_index(val, c):
    return NumericIndexType(types.int64, numba.typeof(val.name))


@typeof_impl.register(pd.UInt64Index)
def typeof_pd_uint64_index(val, c):
    return NumericIndexType(types.uint64, numba.typeof(val.name))


@typeof_impl.register(pd.Float64Index)
def typeof_pd_float64_index(val, c):
    return NumericIndexType(types.float64, numba.typeof(val.name))


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(NumericIndexType)
class NumericIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: nullable integer array (e.g. to hold DatetimeIndex.year)
        members = [
            ('data', types.Array(fe_type.dtype, 1, 'C')),
            ('name', fe_type.name_typ),
        ]
        super(NumericIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(NumericIndexType, 'data', '_data')
make_attribute_wrapper(NumericIndexType, 'name', '_name')


@box(NumericIndexType)
def box_numeric_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    index_val = cgutils.create_struct_proxy(typ)(
            c.context, c.builder, val)
    data = c.pyapi.from_native_value(
        types.Array(typ.dtype, 1, 'C'), index_val.data, c.env_manager)
    name = c.pyapi.from_native_value(
        typ.name_typ, index_val.name, c.env_manager)

    assert typ.dtype in (types.int64, types.uint64, types.float64)
    func_name = 'Int64Index'
    if typ.dtype == types.uint64:
        func_name = 'UInt64Index'
    elif typ.dtype == types.float64:
        func_name = 'Float64Index'
    else:
        assert typ.dtype == types.int64

    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    index_obj = c.pyapi.call_method(
        class_obj, func_name, (data, dtype, copy, name))
    c.pyapi.decref(class_obj)
    return index_obj


@intrinsic
def init_numeric_index(typingctx, data, name=None):
    """Create NumericIndex object
    """

    def codegen(context, builder, signature, args):
        assert len(args) == 2
        index_typ = signature.return_type
        index_val = cgutils.create_struct_proxy(index_typ)(context, builder)
        index_val.data = args[0]
        index_val.name = args[1]
        # increase refcount of stored values
        if context.enable_nrt:
            arr_typ = types.Array(index_typ.dtype, 1, 'C')
            context.nrt.incref(builder, arr_typ, args[0])
            context.nrt.incref(builder, index_typ.name_typ, args[1])
        return index_val._getvalue()

    return NumericIndexType(data.dtype, name)(data, name), codegen


@unbox(NumericIndexType)
def unbox_numeric_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    arr_typ = types.Array(typ.dtype, 1, 'C')
    data = c.pyapi.to_native_value(
        arr_typ, c.pyapi.object_getattr_string(val, 'values')).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, 'name')).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


@numba.generated_jit
def _coerce_to_ndarray(data):
    # TODO: other cases handled by this function in Pandas like scalar
    """
    Coerces data to ndarray.
    """
    if isinstance(data, types.Array):
        return lambda data: data

    if isinstance(data, NumericIndexType):
        return lambda data: data._data

    if isinstance(data, (types.List, types.Tuple)):
        # TODO: check homogenous for tuple
        return lambda data: np.asarray(data)

    if isinstance(data, SeriesType):
        return lambda data: bodo.hiframes.api.get_series_data(data)

    raise TypeError("cannot coerce {} to array".format(data))


@numba.generated_jit
def _fix_arr_dtype(data, new_dtype):
    assert isinstance(data, types.Array)
    assert isinstance(new_dtype, types.DType)

    if data.dtype != new_dtype.dtype:
        return lambda data, new_dtype: data.astype(new_dtype)

    return lambda data, new_dtype: data


def create_numeric_constructor(func, default_dtype):

    def impl(data=None, dtype=None, copy=False, name=None, fastpath=None):
        data_arr = _coerce_to_ndarray(data)
        if copy:
            data_arr = data_arr.copy()  # TODO: np.array() with copy
        data_res = _fix_arr_dtype(data_arr, np.dtype(default_dtype))
        return init_numeric_index(data_res, name)

    overload(func)(
        lambda data=None, dtype=None, copy=False, name=None, fastpath=None:
        impl)


create_numeric_constructor(pd.Int64Index, np.int64)
create_numeric_constructor(pd.UInt64Index, np.uint64)
create_numeric_constructor(pd.Float64Index, np.float64)
