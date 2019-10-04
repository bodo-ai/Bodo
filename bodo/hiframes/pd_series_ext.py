"""
Implement pd.Series typing and data model handling.
"""
import operator
import pandas as pd
import numpy as np
import numba
from numba import types
from numba.extending import (models, register_model, lower_cast, infer_getattr,
    type_callable, infer, overload, make_attribute_wrapper)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)
from numba.typing.arraydecl import (get_array_index_type, _expand_integer,
    ArrayAttribute, SetItemBuffer)
from numba.typing.npydecl import (Numpy_rules_ufunc, NumpyRulesArrayOperator,
    NumpyRulesInplaceArrayOperator)
import bodo
from bodo.libs.str_ext import string_type, list_string_array_type
from bodo.libs.str_arr_ext import (string_array_type, offset_typ, char_typ,
    str_arr_payload_type, StringArrayType, GetItemStringArray)
from bodo.libs.int_arr_ext import IntegerArrayType, IntDtype
from bodo.libs.bool_arr_ext import boolean_array
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type, datetime_date_type
from bodo.hiframes.pd_categorical_ext import (PDCategoricalDtype,
    CategoricalArray)
from bodo.hiframes.rolling import supported_rolling_funcs
import datetime
from bodo.hiframes.split_impl import (string_array_split_view_type,
    GetItemStringArraySplitView)
from bodo.utils.typing import (is_overload_none, is_overload_true,
    is_overload_false)


class SeriesType(types.IterableType, types.ArrayCompatible):
    """Temporary type class for Series objects.
    """
    def __init__(self, dtype, data=None, index=None, name_typ=None):
        # keeping data array in type since operators can make changes such
        # as making array unaligned etc.
        # data is underlying array type and can have different dtype
        data = _get_series_array_type(dtype) if data is None else data
        # store regular dtype instead of IntDtype to avoid errors
        dtype = dtype.dtype if isinstance(dtype, IntDtype) else dtype
        # convert Record to tuple (for tuple output of map)
        # TODO: handle actual Record objects in Series?
        dtype = (types.Tuple(list(dict(dtype.members).values()))
                if isinstance(dtype, types.Record) else dtype)
        self.dtype = dtype
        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        index = types.none if index is None else index
        self.index = index  # index should be an Index type (not Array)
        self.name_typ = name_typ
        super(SeriesType, self).__init__(
            name="series({}, {}, {}, {})".format(dtype, data, index, name_typ))

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, 'C')

    def copy(self, dtype=None, index=None):
        # XXX is copy necessary?
        if index is None:
            index = types.none if self.index == types.none else self.index.copy()
        dtype = dtype if dtype is not None else self.dtype
        data = _get_series_array_type(dtype)
        return SeriesType(dtype, data, index, self.name_typ)

    @property
    def key(self):
        # needed?
        return self.dtype, self.data, self.index, self.name_typ

    def unify(self, typingctx, other):
        if isinstance(other, SeriesType):
            new_index = types.none
            if self.index != types.none and other.index != types.none:
                new_index = self.index.unify(typingctx, other.index)
            elif other.index != types.none:
                new_index = other.index
            elif self.index != types.none:
                new_index = self.index

            # If dtype matches or other.dtype is undefined (inferred)
            if other.dtype == self.dtype or not other.dtype.is_precise():
                return SeriesType(
                    self.dtype,
                    self.data.unify(typingctx, other.data),
                    new_index)

        # XXX: unify Series/Array as Array
        return super(SeriesType, self).unify(typingctx, other)

    # XXX too dangerous, is it needed?
    # def can_convert_to(self, typingctx, other):
    #     # same as types.Array
    #     if (isinstance(other, SeriesType) and other.dtype == self.dtype):
    #         # called for overload selection sometimes
    #         # TODO: fix index
    #         if self.index == types.none and other.index == types.none:
    #             return self.data.can_convert_to(typingctx, other.data)
    #         if self.index != types.none and other.index != types.none:
    #             return max(self.data.can_convert_to(typingctx, other.data),
    #                 self.index.can_convert_to(typingctx, other.index))

    def is_precise(self):
        # same as types.Array
        return self.dtype.is_precise()

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timestamp
        return types.iterators.ArrayIterator(self.data)


def _get_series_array_type(dtype):
    """get underlying default array type of series based on its dtype
    """
    # list(list(str))
    if dtype == types.List(string_type):
        # default data layout is list but split view is used if possible
        return list_string_array_type
    # string array
    elif dtype == string_type:
        return string_array_type

    # categorical
    if isinstance(dtype, PDCategoricalDtype):
        return CategoricalArray(dtype)

    if isinstance(dtype, IntDtype):
        return IntegerArrayType(dtype.dtype)

    if dtype == types.bool_:
        return boolean_array

    # use recarray data layout for series of tuples
    if isinstance(dtype, types.BaseTuple):
        if any(not isinstance(t, types.Number) for t in dtype.types):
            # TODO: support more types. what types can be in recarrays?
            raise ValueError("series tuple dtype {} includes non-numerics".format(dtype))
        np_dtype = np.dtype(
            ','.join(str(t) for t in dtype.types), align=True)
        dtype = numba.numpy_support.from_dtype(np_dtype)

    if dtype == datetime_date_type:
        return bodo.hiframes.datetime_date_ext.array_datetime_date

    # TODO: other types?
    # regular numpy array
    return types.Array(dtype, 1, 'C')


def is_bool_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.bool_


def is_str_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == string_type


def is_dt64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPDatetime('ns')


def is_timedelta64_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == types.NPTimedelta('ns')


def is_datetime_date_series_typ(t):
    return isinstance(t, SeriesType) and t.dtype == datetime_date_type


# payload type inside meminfo so that mutation are seen by all references
class SeriesPayloadType(types.Type):
    def __init__(self, series_type):
        self.series_type = series_type
        super(SeriesPayloadType, self).__init__(
            name='SeriesPayloadType({})'.format(series_type))


@register_model(SeriesPayloadType)
class SeriesPayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('data', fe_type.series_type.data),
            ('index', fe_type.series_type.index),
            ('name', fe_type.series_type.name_typ),
        ]
        super(SeriesPayloadModel, self).__init__(dmm, fe_type, members)


@register_model(SeriesType)
class SeriesModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = SeriesPayloadType(fe_type)
        #payload_type = types.Opaque('Opaque.Series')
        # TODO: does meminfo decref content when object is deallocated?
        members = [
            ('meminfo', types.MemInfoPointer(payload_type)),
            # for boxed Series, enables updating original Series object
            ('parent', types.pyobject),
        ]
        super(SeriesModel, self).__init__(dmm, fe_type, members)



def series_to_array_type(typ):
    return typ.data
    # return _get_series_array_type(typ.dtype)


def is_series_type(typ):
    return isinstance(typ, SeriesType)


def arr_to_series_type(arr):
    series_type = None
    if isinstance(arr, types.Array):
        series_type = SeriesType(arr.dtype, arr)
    elif arr == string_array_type:
        # StringArray is readonly
        series_type = SeriesType(string_type)
    elif arr == list_string_array_type:
        series_type = SeriesType(types.List(string_type))
    elif arr == string_array_split_view_type:
        series_type = SeriesType(types.List(string_type),
                        string_array_split_view_type)
    return series_type


def if_series_to_array_type(typ):
    if isinstance(typ, SeriesType):
        return series_to_array_type(typ)

    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple(
            [if_series_to_array_type(t) for t in typ.types])
    if isinstance(typ, types.List) and isinstance(typ.dtype, SeriesType):
        return types.List(if_series_to_array_type(typ.dtype))
    if isinstance(typ, types.Set):
        return types.Set(if_series_to_array_type(typ.dtype))
    # TODO: other types that can have Series inside?
    return typ


def if_arr_to_series_type(typ):
    if isinstance(typ, types.Array) or typ in (string_array_type,
            list_string_array_type, string_array_split_view_type):
        return arr_to_series_type(typ)
    if isinstance(typ, (types.Tuple, types.UniTuple)):
        return types.Tuple([if_arr_to_series_type(t) for t in typ.types])
    if isinstance(typ, types.List):
        return types.List(if_arr_to_series_type(typ.dtype))
    if isinstance(typ, types.Set):
        return types.Set(if_arr_to_series_type(typ.dtype))
    # TODO: other types that can have Arrays inside?
    return typ


@numba.njit
def convert_index_to_int64(S):
    # convert Series with none index to numeric index
    data = bodo.hiframes.api.get_series_data(S)
    index_arr = np.arange(len(data))
    name = bodo.hiframes.api.get_series_name(S)
    return bodo.hiframes.api.init_series(
        data,
        bodo.utils.conversion.convert_to_index(index_arr),
        name)


# cast Series(int8) to Series(cat) for init_series() in test_csv_cat1
# TODO: separate array type for Categorical data
@lower_cast(SeriesType, SeriesType)
def cast_series(context, builder, fromty, toty, val):
    # convert None index to Integer index if everything else is same
    if (fromty.copy(index=toty.index) == toty and fromty.index == types.none
            and toty.index == bodo.hiframes.pd_index_ext.NumericIndexType(
                types.int64, types.none)):
        cn_typ = context.typing_context.resolve_value_type(
            convert_index_to_int64)
        cn_sig = cn_typ.get_call_type(
            context.typing_context, (fromty,), {})
        cn_impl = context.get_function(cn_typ, cn_sig)
        return cn_impl(builder, (val,))

    return val


# --------------------------------------------------------------------------- #
# --- typing similar to arrays adopted from arraydecl.py, npydecl.py -------- #


@infer_getattr
class SeriesAttribute(AttributeTemplate):
    key = SeriesType

    def resolve_dt(self, ary):
        assert ary.dtype == types.NPDatetime('ns')
        return series_dt_methods_type

    @bound_function("series.rolling")
    def resolve_rolling(self, ary, args, kws):
        def rolling_stub(window, min_periods=None, center=False, win_type=None,
                on=None, axis=0, closed=None):
            pass
        pysig = numba.utils.pysignature(rolling_stub)
        return signature(SeriesRollingType(ary), *args).replace(pysig=pysig)

    def _resolve_map_func(self, ary, func, pysig):

        dtype = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype == types.NPDatetime('ns'):
            dtype = pandas_timestamp_type
        code = func.literal_value.code
        _globals = {'np': np, 'pd': pd}
        # XXX hack in series_pass to make globals available
        if hasattr(func.literal_value, 'globals'):
            # TODO: use code.co_names to find globals actually used?
            _globals = func.literal_value.globals

        f_ir = numba.ir_utils.get_ir_of_code(_globals, code)
        _, f_return_type, _ = numba.typed_passes.type_inference_stage(
                self.context, f_ir, (dtype,), None)

        return signature(
            SeriesType(f_return_type, index=ary.index, name_typ=ary.name_typ),
            (func,)).replace(pysig=pysig)

    @bound_function("series.map")
    def resolve_map(self, ary, args, kws):
        kwargs = dict(kws)
        func = args[0] if len(args) > 0 else kwargs['arg']

        def map_stub(arg, na_action=None):
            pass

        pysig = numba.utils.pysignature(map_stub)
        return self._resolve_map_func(ary, func, pysig)

    @bound_function("series.apply")
    def resolve_apply(self, ary, args, kws):
        kwargs = dict(kws)
        func = args[0] if len(args) > 0 else kwargs['func']

        def apply_stub(func, convert_dtype=True, args=()):
            pass

        pysig = numba.utils.pysignature(apply_stub)
        # TODO: handle apply differences: extra args, np ufuncs etc.
        return self._resolve_map_func(ary, func, pysig)

    def _resolve_combine_func(self, ary, args, kws):
        # handle kwargs
        kwargs = dict(kws)
        other = args[0] if len(args) > 0 else types.unliteral(kwargs['other'])
        func = args[1] if len(args) > 1 else kwargs['func']
        fill_value = args[2] if len(args) > 2 else types.unliteral(
                                          kwargs.get('fill_value', types.none))

        def combine_stub(other, func, fill_value=None):
            pass
        pysig = numba.utils.pysignature(combine_stub)

        # get return type
        dtype1 = ary.dtype
        # getitem returns Timestamp for dt_index and series(dt64)
        if dtype1 == types.NPDatetime('ns'):
            dtype1 = pandas_timestamp_type
        dtype2 = other.dtype
        if dtype2 == types.NPDatetime('ns'):
            dtype2 = pandas_timestamp_type
        code = func.literal_value.code
        f_ir = numba.ir_utils.get_ir_of_code({'np': np, 'pd': pd}, code)
        _, f_return_type, _ = numba.typed_passes.type_inference_stage(
                self.context, f_ir, (dtype1,dtype2,), None)

        # TODO: output name is always None in Pandas?
        sig = signature(SeriesType(f_return_type, index=ary.index,
            name_typ=types.none),
            (other, func, fill_value))
        return sig.replace(pysig=pysig)

    @bound_function("series.combine")
    def resolve_combine(self, ary, args, kws):
        return self._resolve_combine_func(ary, args, kws)

    # TODO: use overload when Series.aggregate is supported
    @bound_function("series.value_counts")
    def resolve_value_counts(self, ary, args, kws):
        # output is int series with original data as index
        index_typ = bodo.hiframes.pd_index_ext.array_typ_to_index(ary.data)
        out = SeriesType(
            types.int64, types.Array(types.int64, 1, 'C'), index_typ,
            ary.name_typ)
        return signature(out, *args)


# pd.Series supports all operators except << and >>
series_binary_ops = tuple(
    op for op in numba.typing.npydecl.NumpyRulesArrayOperator._op_map.keys()
    if op not in (operator.lshift, operator.rshift)
)


# TODO: support itruediv, Numpy doesn't support it, and output can have
# a different type (output of integer division is float)
series_inplace_binary_ops = tuple(
    op for op in \
        numba.typing.npydecl.NumpyRulesInplaceArrayOperator._op_map.keys()
    if op not in (operator.ilshift, operator.irshift, operator.itruediv)
)

inplace_binop_to_imm = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.ifloordiv: operator.floordiv,
    operator.imod: operator.mod,
    operator.ipow: operator.pow,
    operator.iand: operator.and_,
    operator.ior: operator.or_,
    operator.ixor: operator.xor,
}


series_unary_ops = (operator.neg, operator.invert, operator.pos)


str2str_methods = ('capitalize', 'lower', 'lstrip', 'rstrip',
            'strip', 'swapcase', 'title', 'upper')


str2bool_methods = ('isalnum', 'isalpha', 'isdigit',
    'isspace', 'islower', 'isupper', 'istitle', 'isnumeric', 'isdecimal')


class SeriesDtMethodType(types.Type):
    def __init__(self):
        name = "SeriesDtMethodType"
        super(SeriesDtMethodType, self).__init__(name)

series_dt_methods_type = SeriesDtMethodType()


@infer_getattr
class SeriesDtMethodAttribute(AttributeTemplate):
    key = SeriesDtMethodType

    def resolve_date(self, ary):
        return SeriesType(datetime_date_type)  # TODO: name, index


# all date fields return int64 same as Timestamp fields
def resolve_date_field(self, ary):
    return SeriesType(types.int64)

for field in bodo.hiframes.pd_timestamp_ext.date_fields:
    setattr(SeriesDtMethodAttribute, "resolve_" + field, resolve_date_field)


class SeriesRollingType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesRollingType({})".format(stype)
        super(SeriesRollingType, self).__init__(name)


@infer_getattr
class SeriesRollingAttribute(AttributeTemplate):
    key = SeriesRollingType

    @bound_function("rolling.apply")
    def resolve_apply(self, ary, args, kws):
        # result is always float64 (see Pandas window.pyx:roll_generic)
        return signature(SeriesType(types.float64, index=ary.stype.index,
            name_typ=ary.stype.name_typ), *args)

    @bound_function("rolling.cov")
    def resolve_cov(self, ary, args, kws):
        return signature(SeriesType(types.float64, index=ary.stype.index,
            name_typ=ary.stype.name_typ), *args)

    @bound_function("rolling.corr")
    def resolve_corr(self, ary, args, kws):
        return signature(SeriesType(types.float64, index=ary.stype.index,
            name_typ=ary.stype.name_typ), *args)


# similar to install_array_method in arraydecl.py
def install_rolling_method(name):
    def rolling_attribute_attachment(self, ary):
        def rolling_generic(self, args, kws):
            # output is always float64
            return signature(SeriesType(types.float64, index=ary.stype.index,
                name_typ=ary.stype.name_typ), *args)
        my_attr = {"key": "rolling." + name, "generic": rolling_generic}
        temp_class = type("Rolling_" + name, (AbstractTemplate,), my_attr)
        return types.BoundFunction(temp_class, ary)

    setattr(SeriesRollingAttribute, "resolve_" + name,
        rolling_attribute_attachment)


for fname in supported_rolling_funcs:
    install_rolling_method(fname)


@infer
@infer_global(operator.eq)
@infer_global(operator.ne)
@infer_global(operator.ge)
@infer_global(operator.gt)
@infer_global(operator.le)
@infer_global(operator.lt)
class SeriesCompEqual(AbstractTemplate):
    key = '=='
    def generic(self, args, kws):
        from bodo.libs.str_arr_ext import is_str_arr_typ
        assert not kws
        [va, vb] = args
        # if one of the inputs is string array
        if is_str_series_typ(va) or is_str_series_typ(vb):
            # inputs should be either string array or string
            assert is_str_arr_typ(va) or va == string_type
            assert is_str_arr_typ(vb) or vb == string_type
            return signature(SeriesType(types.boolean, boolean_array), va, vb)

        if ((is_dt64_series_typ(va) and vb in (string_type, types.NPDatetime('ns')))
                or (is_dt64_series_typ(vb) and va in (string_type, types.NPDatetime('ns')))):
            return signature(SeriesType(types.boolean, boolean_array), va, vb)

        if is_dt64_series_typ(va) and is_dt64_series_typ(vb):
            return signature(SeriesType(types.boolean, boolean_array), va, vb)


@infer
class CmpOpNEqSeries(SeriesCompEqual):
    key = '!='

@infer
class CmpOpGESeries(SeriesCompEqual):
    key = '>='

@infer
class CmpOpGTSeries(SeriesCompEqual):
    key = '>'

@infer
class CmpOpLESeries(SeriesCompEqual):
    key = '<='

@infer
class CmpOpLTSeries(SeriesCompEqual):
    key = '<'


# TODO: handle all timedelta args
def type_sub(context):
    def typer(val1, val2):
        if is_dt64_series_typ(val1) and val2 == pandas_timestamp_type:
            return SeriesType(types.NPTimedelta('ns'))

    return typer

type_callable('-')(type_sub)
type_callable(operator.sub)(type_sub)


@overload(pd.Series)
def pd_series_overload(data=None, index=None, dtype=None, name=None,
                                                   copy=False, fastpath=False):

    # TODO: support isinstance in branch pruning pass
    # cases: dict, np.ndarray, Series, Index, arraylike (list, ...)

    # TODO: None or empty data
    if is_overload_none(data):
        raise ValueError("pd.Series(): 'data' argument required.")

    # fastpath not supported
    if not is_overload_false(fastpath):
        raise ValueError("pd.Series(): 'fastpath' argument not supported.")


    def impl(data=None, index=None, dtype=None, name=None, copy=False,
                                                               fastpath=False):
        # extract name if data is has name (Series/Index) and name is None
        name_t = bodo.utils.conversion.extract_name_if_none(data, name)
        index_t = bodo.utils.conversion.extract_index_if_none(data, index)
        data_t1 = bodo.utils.conversion.coerce_to_array(data, True, True)

        # TODO: support sanitize_array() of Pandas
        # TODO: add branch pruning to inline_closure_call
        # if dtype is not None:
        #     data_t2 = data_t1.astype(dtype)
        # else:
        #     data_t2 = data_t1

        # TODO: copy if index to avoid aliasing issues
        # data_t2 = data_t1
        data_t2 = bodo.utils.conversion.fix_arr_dtype(data_t1, dtype)

        # TODO: enable when branch pruning works for this
        # if copy:
        #     data_t2 = data_t1.copy()

        return bodo.hiframes.api.init_series(
                data_t2,
                bodo.utils.conversion.convert_to_index(index_t),
                name_t)
    return impl
