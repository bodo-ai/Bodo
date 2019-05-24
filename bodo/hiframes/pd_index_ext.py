import operator
import pandas as pd
import numpy as np
import numba
from numba import types, cgutils
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper, box, intrinsic,
    typeof_impl, unbox, NativeValue, overload_attribute, overload_method)
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
import bodo.utils.conversion
from bodo.utils.utils import BooleanLiteral


_dt_index_data_typ = types.Array(types.NPDatetime('ns'), 1, 'C')
_timedelta_index_data_typ = types.Array(types.NPTimedelta('ns'), 1, 'C')
iNaT = pd._libs.tslibs.iNaT
NaT = types.NPDatetime('ns')('NaT')  # TODO: pd.NaT


# -------------------------  DatetimeIndex -----------------------------


class DatetimeIndexType(types.IterableType):
    """type class for DatetimeIndex objects.
    """
    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
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
    name = c.pyapi.from_native_value(
        typ.name_typ, dt_index.name, c.env_manager)

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


# support DatetimeIndex date fields such as I.year
def gen_dti_field_impl(field):
    # TODO: NaN
    func_text = 'def impl(dti):\n'
    func_text += '    numba.parfor.init_prange()\n'
    func_text += '    A = bodo.hiframes.api.get_index_data(dti)\n'
    func_text += '    name = bodo.hiframes.api.get_index_name(dti)\n'
    func_text += '    n = len(A)\n'
    # all datetimeindex fields return int64 same as Timestamp fields
    func_text += '    S = np.empty(n, np.int64)\n'
    func_text += '    for i in numba.parfor.internal_prange(n):\n'
    func_text += '        dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])\n'
    func_text += '        ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n'
    func_text += '        S[i] = ts.' + field + '\n'
    func_text += '    return bodo.hiframes.pd_index_ext.init_numeric_index(S, name)\n'
    loc_vars = {}
    # print(func_text)
    exec(func_text, {'numba': numba, 'np': np, 'bodo': bodo}, loc_vars)
    impl = loc_vars['impl']
    return impl


for field in bodo.hiframes.pd_timestamp_ext.date_fields:
    impl = gen_dti_field_impl(field)
    overload_attribute(DatetimeIndexType, field)(lambda dti: impl)


@overload_attribute(DatetimeIndexType, 'date')
def overload_datetime_index_date(dti):
    # TODO: NaN

    def impl(dti):
        numba.parfor.init_prange()
        A = bodo.hiframes.api.get_index_data(dti)
        n = len(A)
        S = numba.unsafe.ndarray.empty_inferred((n,))
        for i in numba.parfor.internal_prange(n):
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            S[i] = bodo.hiframes.pd_timestamp_ext.datetime_date_ctor(ts.year, ts.month, ts.day)
        return bodo.hiframes.datetime_date_ext.np_arr_to_array_datetime_date(S)

    return impl


@numba.njit
def _dti_val_finalize(s, count):  # pragma: no cover
    if not count:
        s = iNaT  # TODO: NaT type boxing in timestamp
    return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)


@overload_method(DatetimeIndexType, 'min')
def overload_datetime_index_min(dti, axis=None, skipna=True):
    # TODO skipna = False
    if not _is_none(axis) or not _is_true(skipna):
        raise ValueError(
            "Index.min(): axis and skipna arguments not supported yet")

    def impl(dti, axis=None, skipna=True):
        numba.parfor.init_prange()
        in_arr = bodo.hiframes.api.get_index_data(dti)
        s = numba.targets.builtins.get_type_max_value(numba.types.int64)
        count = 0
        for i in numba.parfor.internal_prange(len(in_arr)):
            if not bodo.hiframes.api.isna(in_arr, i):
                val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                s = min(s, val)
                count += 1
        return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

    return impl


# TODO: refactor min/max
@overload_method(DatetimeIndexType, 'max')
def overload_datetime_index_max(dti, axis=None, skipna=True):
    # TODO skipna = False
    if not _is_none(axis) or not _is_true(skipna):
        raise ValueError(
            "Index.max(): axis and skipna arguments not supported yet")

    def impl(dti, axis=None, skipna=True):
        numba.parfor.init_prange()
        in_arr = bodo.hiframes.api.get_index_data(dti)
        s = numba.targets.builtins.get_type_min_value(numba.types.int64)
        count = 0
        for i in numba.parfor.internal_prange(len(in_arr)):
            if not bodo.hiframes.api.isna(in_arr, i):
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
def pd_datetimeindex_overload(data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True):
    # TODO: check/handle other input
    if _is_none(data):
        raise ValueError("data argument in pd.DatetimeIndex() expected")

    # check unsupported, TODO: normalize, dayfirst, yearfirst, ...
    if any(not _is_none(a) for a in (freq, start, end, periods, tz, closed)):
        raise ValueError("only data argument in pd.DatetimeIndex() supported")

    def f(data=None, freq=None, start=None, end=None,
        periods=None, tz=None, normalize=False, closed=None, ambiguous='raise',
        dayfirst=False, yearfirst=False, dtype=None, copy=False, name=None,
        verify_integrity=True):
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        S = bodo.utils.conversion.convert_to_dt64ns(data_arr)
        return bodo.hiframes.api.init_datetime_index(S, name)

    return f


@overload(operator.sub)
def overload_datetime_index_sub(arg1, arg2):
    # DatetimeIndex - Timestamp
    if (isinstance(arg1, DatetimeIndexType)
            and arg2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type):
        def impl(arg1, arg2):  # pragma: no cover
            numba.parfor.init_prange()
            in_arr = bodo.hiframes.api.get_index_data(arg1)
            name = bodo.hiframes.api.get_index_name(arg1)
            n = len(in_arr)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            tsint = bodo.hiframes.pd_timestamp_ext.convert_timestamp_to_datetime64(arg2)
            for i in numba.parfor.internal_prange(n):
                S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]) - tsint)
            return bodo.hiframes.api.init_timedelta_index(S, name)

        return impl

    # Timestamp - DatetimeIndex
    if (isinstance(arg2, DatetimeIndexType)
            and arg1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type):
        def impl(arg1, arg2):  # pragma: no cover
            numba.parfor.init_prange()
            in_arr = bodo.hiframes.api.get_index_data(arg2)
            name = bodo.hiframes.api.get_index_name(arg2)
            n = len(in_arr)
            S = numba.unsafe.ndarray.empty_inferred((n,))
            tsint = bodo.hiframes.pd_timestamp_ext.convert_timestamp_to_datetime64(arg1)
            for i in numba.parfor.internal_prange(n):
                S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    tsint - bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]))
            return bodo.hiframes.api.init_timedelta_index(S, name)

        return impl


# bionp of DatetimeIndex and string
def gen_dti_str_binop_impl(op, is_arg1_dti):
    # is_arg1_dti: is the first argument DatetimeIndex and second argument str
    op_str = numba.utils.OPERATORS_TO_BUILTINS[op]
    func_text = 'def impl(arg1, arg2):\n'
    if is_arg1_dti:
        func_text += '  dt_index, _str = arg1, arg2\n'
        comp = 'arr[i] {} other'.format(op_str)
    else:
        func_text += '  dt_index, _str = arg2, arg1\n'
        comp = 'other {} arr[i]'.format(op_str)
    func_text += '  arr = bodo.hiframes.api.get_index_data(dt_index)\n'
    func_text += '  l = len(arr)\n'
    func_text += '  other = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n'
    func_text += '  S = numba.unsafe.ndarray.empty_inferred((l,))\n'
    func_text += '  for i in numba.parfor.internal_prange(l):\n'
    func_text += '    S[i] = {}\n'.format(comp)
    func_text += '  return S\n'
    # print(func_text)
    loc_vars = {}
    exec(func_text, {'bodo': bodo, 'numba': numba, 'np': np}, loc_vars)
    impl = loc_vars['impl']
    return impl


def overload_binop_dti_str(op):

    def overload_impl(arg1, arg2):
        if isinstance(arg1, DatetimeIndexType) and types.unliteral(arg2) == string_type:
            return gen_dti_str_binop_impl(op, True)
        if isinstance(arg2, DatetimeIndexType) and types.unliteral(arg1) == string_type:
            return gen_dti_str_binop_impl(op, False)

    return overload_impl


for op in (operator.eq, operator.ne, operator.ge, operator.gt, operator.le,
        operator.lt):
    overload(op)(overload_binop_dti_str(op))


@overload(operator.getitem)
def overload_datetime_index_getitem(dti, ind):
    # TODO: other getitem cases
    if isinstance(dti, DatetimeIndexType):
        if isinstance(ind, types.Integer):
            def impl(dti, ind):
                dti_arr = bodo.hiframes.api.get_index_data(dti)
                dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dti_arr[ind])
                return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            return impl
        else:
            # slice, boolean array, etc.
            # TODO: other Index or Series objects as index?
            def impl(dti, ind):
                dti_arr = bodo.hiframes.api.get_index_data(dti)
                name = bodo.hiframes.api.get_index_name(dti)
                new_arr = dti_arr[ind]
                return bodo.hiframes.api.init_datetime_index(new_arr, name)
            return impl


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
            name="TimedeltaIndexType(named = {})".format(name_typ))

    def copy(self):
        # XXX is copy necessary?
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
        members = [
            ('data', _timedelta_index_data_typ),
            ('name', fe_type.name_typ),
        ]
        super(TimedeltaIndexTypeModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(pd.TimedeltaIndex)
def typeof_timedelta_index(val, c):
    return TimedeltaIndexType(numba.typeof(val.name))


@box(TimedeltaIndexType)
def box_timedelta_index(typ, val, c):
    """
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    timedelta_index = numba.cgutils.create_struct_proxy(
            typ)(c.context, c.builder, val)

    arr = c.pyapi.from_native_value(
        _timedelta_index_data_typ, timedelta_index.data, c.env_manager)
    name = c.pyapi.from_native_value(
        typ.name_typ, timedelta_index.name, c.env_manager)

    # call pd.TimedeltaIndex(arr, name=name)
    kws = c.pyapi.dict_pack([('name', name)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, 'TimedeltaIndex')
    res = c.pyapi.call(const_call, c.pyapi.tuple_pack([arr]), kws)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    return res


@unbox(TimedeltaIndexType)
def unbox_timedelta_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    data = c.pyapi.to_native_value(
        _timedelta_index_data_typ, c.pyapi.object_getattr_string(val, 'values')).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, 'name')).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


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


make_attribute_wrapper(TimedeltaIndexType, 'data', '_data')
make_attribute_wrapper(TimedeltaIndexType, 'name', '_name')


# all datetimeindex fields return int64 same as Timestamp fields
def resolve_timedelta_field(self, ary):
    # TODO: return Int64Index
    return types.Array(types.int64, 1, 'C')

for field in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
    setattr(TimedeltaIndexAttribute, "resolve_" + field, resolve_timedelta_field)


@overload(pd.TimedeltaIndex)
def pd_timedelta_index_overload(data=None, unit=None, freq=None, start=None,
        end=None, periods=None, closed=None, dtype=None, copy=False,
        name=None, verify_integrity=None):
    # TODO handle dtype=dtype('<m8[ns]') default
    # TODO: check/handle other input
    if _is_none(data):
        raise ValueError("data argument in pd.TimedeltaIndex() expected")

    if any(not _is_none(a) for a in (unit, freq, start, end, periods, closed,
                                                                       dtype)):
        raise ValueError("only data argument in pd.TimedeltaIndex() supported")

    def impl(data=None, unit=None, freq=None, start=None,
            end=None, periods=None, closed=None, dtype=None, copy=False,
            name=None, verify_integrity=None):
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        S = bodo.utils.conversion.convert_to_td64ns(data_arr)
        return bodo.hiframes.api.init_timedelta_index(S, name)

    return impl


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
    def __init__(self, dtype, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
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
    name = types.none if name is None else name

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


def create_numeric_constructor(func, default_dtype):

    def impl(data=None, dtype=None, copy=False, name=None, fastpath=None):
        data_arr = bodo.utils.conversion.coerce_to_ndarray(data)
        if copy:
            data_arr = data_arr.copy()  # TODO: np.array() with copy
        data_res = bodo.utils.conversion.fix_arr_dtype(
            data_arr, np.dtype(default_dtype))
        return init_numeric_index(data_res, name)

    overload(func)(
        lambda data=None, dtype=None, copy=False, name=None, fastpath=None:
        impl)


create_numeric_constructor(pd.Int64Index, np.int64)
create_numeric_constructor(pd.UInt64Index, np.uint64)
create_numeric_constructor(pd.Float64Index, np.float64)



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
            name="StringIndexType({})".format(name_typ))

    @property
    def iterator_type(self):
        # TODO: handle iterator
        return bodo.libs.str_arr_ext.StringArrayIterator()


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(StringIndexType)
class StringIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', string_array_type),
            ('name', fe_type.name_typ),
        ]
        super(StringIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(StringIndexType, 'data', '_data')
make_attribute_wrapper(StringIndexType, 'name', '_name')


@box(StringIndexType)
def box_string_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    index_val = cgutils.create_struct_proxy(typ)(
            c.context, c.builder, val)
    data = c.pyapi.from_native_value(
        string_array_type, index_val.data, c.env_manager)
    name = c.pyapi.from_native_value(
        typ.name_typ, index_val.name, c.env_manager)

    dtype = c.pyapi.make_none()
    copy = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    index_obj = c.pyapi.call_method(
        class_obj, 'Index', (data, dtype, copy, name))
    c.pyapi.decref(class_obj)
    return index_obj


@intrinsic
def init_string_index(typingctx, data, name=None):
    """Create StringIndex object
    """
    name = types.none if name is None else name

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
        string_array_type, c.pyapi.object_getattr_string(val, 'values')).value
    name = c.pyapi.to_native_value(
        typ.name_typ, c.pyapi.object_getattr_string(val, 'name')).value

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    return NativeValue(index_val._getvalue())


@overload(operator.getitem)
def overload_index_getitem(I, ind):
    if isinstance(I, (NumericIndexType, StringIndexType)):
        return lambda I, ind: bodo.hiframes.api.get_index_data(I)[ind]


def _is_none(val):
    return (val is None or val == types.none
            or getattr(val, 'value', False) is None)


def _is_true(val):
    return (val == True or val == BooleanLiteral(True)
            or getattr(val, 'value', False) is True)


# similar to index_from_array()
def array_typ_to_index(arr_typ):
    if arr_typ == bodo.string_array_type:
        return StringIndexType()

    assert isinstance(arr_typ, types.Array)
    if arr_typ.dtype == types.NPDatetime('ns'):
        return DatetimeIndexType()

    if isinstance(arr_typ.dtype, types.Integer):
        if not arr_typ.dtype.signed:
            return NumericIndexType(types.uint64)
        else:
            return NumericIndexType(types.int64)

    if isinstance(arr_typ.dtype, types.Float):
        return NumericIndexType(types.float64)

    raise TypeError("invalid index type {}".format(arr_typ))
