# Copyright (C) 2019 Bodo Inc. All rights reserved.
import datetime
import operator

import llvmlite.llvmpy.core as lc
import numba
import numpy as np
import pandas as pd
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_new_ref, lower_constant
from numba.core.typing.templates import AttributeTemplate, signature
from numba.extending import (
    NativeValue,
    box,
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_jitable,
    register_model,
    typeof_impl,
    unbox,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
import bodo.hiframes
import bodo.utils.conversion
from bodo.hiframes.datetime_timedelta_ext import pd_timedelta_type
from bodo.hiframes.pd_series_ext import SeriesType, string_array_type
from bodo.hiframes.pd_timestamp_ext import pandas_timestamp_type
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_ext import string_type
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    create_unsupported_overload,
    get_overload_const_func,
    get_overload_const_str,
    get_udf_error_msg,
    get_udf_out_arr_type,
    get_val_type_maybe_str_literal,
    is_const_func_type,
    is_heterogeneous_tuple_type,
    is_iterable_type,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    raise_bodo_error,
)

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

    # TODO: Replace with a specific type for DateIndex, so these can be boxed
    if val.inferred_type == "date":
        return DatetimeIndexType(get_val_type_maybe_str_literal(val.name))

    # catch-all for non-supported Index types
    # RangeIndex is directly supported (TODO: make sure this is not called)
    raise NotImplementedError("unsupported pd.Index type")


# -------------------------  DatetimeIndex ------------------------------


class DatetimeIndexType(types.IterableType, types.ArrayCompatible):
    """type class for DatetimeIndex objects."""

    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like freq/tz/dtype/yearfirst?
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = types.Array(bodo.datetime64ns, 1, "C")
        super(DatetimeIndexType, self).__init__(
            name="DatetimeIndex(name = {})".format(name_typ)
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return types.NPDatetime("ns")

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


types.datetime_index = DatetimeIndexType()


@typeof_impl.register(pd.DatetimeIndex)
def typeof_datetime_index(val, c):
    # TODO: check value for freq, tz, etc. and raise error since unsupported
    return DatetimeIndexType(get_val_type_maybe_str_literal(val.name))


@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: use payload to support mutable name
        members = [
            ("data", _dt_index_data_typ),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(_dt_index_data_typ.dtype, types.int64)),
        ]
        super(DatetimeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DatetimeIndexType, "data", "_data")
make_attribute_wrapper(DatetimeIndexType, "name", "_name")
make_attribute_wrapper(DatetimeIndexType, "dict", "_dict")


@overload_method(DatetimeIndexType, "copy", no_unliteral=True)
def overload_datetime_index_copy(A):
    return lambda A: bodo.hiframes.pd_index_ext.init_datetime_index(
        A._data.copy(), A._name
    )  # pragma: no cover


@overload_attribute(DatetimeIndexType, "name")
def DatetimeIndex_get_name(di):
    def impl(di):  # pragma: no cover
        return di._name

    return impl


@box(DatetimeIndexType)
def box_dt_index(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    dt_index = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    c.context.nrt.incref(c.builder, _dt_index_data_typ, dt_index.data)
    arr_obj = c.pyapi.from_native_value(
        _dt_index_data_typ, dt_index.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, dt_index.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, dt_index.name, c.env_manager)

    # call pd.DatetimeIndex(arr, name=name)
    args = c.pyapi.tuple_pack([arr_obj])
    kws = c.pyapi.dict_pack([("name", name_obj)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "DatetimeIndex")
    res = c.pyapi.call(const_call, args, kws)

    c.pyapi.decref(arr_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(DatetimeIndexType)
def unbox_datetime_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    values_obj = c.pyapi.object_getattr_string(val, "values")
    data = c.pyapi.to_native_value(_dt_index_data_typ, values_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    dtype = _dt_index_data_typ.dtype
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@intrinsic
def init_datetime_index(typingctx, data, name=None):
    """Create a DatetimeIndex with provided data and name values."""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create dt_index struct and store values
        dt_index = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        dt_index.data = data_val
        dt_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], name_val)

        # create empty dict for get_loc hashmap
        dtype = _dt_index_data_typ.dtype
        dt_index.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(dtype, types.int64),
            types.DictType(dtype, types.int64)(),
            [],
        )  # pragma: no cover

        return dt_index._getvalue()

    ret_typ = DatetimeIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


def init_index_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) >= 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_datetime_index = (
    init_index_equiv
)


# support DatetimeIndex date fields such as I.year
def gen_dti_field_impl(field):
    # TODO: NaN
    func_text = "def impl(dti):\n"
    func_text += "    numba.parfors.parfor.init_prange()\n"
    func_text += "    A = bodo.hiframes.pd_index_ext.get_index_data(dti)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(dti)\n"
    func_text += "    n = len(A)\n"
    # all datetimeindex fields return int64 same as Timestamp fields
    func_text += "    S = np.empty(n, np.int64)\n"
    # TODO: use nullable int when supported by NumericIndex?
    # func_text += "    S = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)\n"
    func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
    # func_text += "        if bodo.libs.array_kernels.isna(A, i):\n"
    # func_text += "            bodo.libs.array_kernels.setna(S, i)\n"
    # func_text += "            continue\n"
    func_text += "        dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])\n"
    func_text += "        ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n"
    if field in [
        "weekday",
    ]:
        func_text += "        S[i] = ts." + field + "()\n"
    else:
        func_text += "        S[i] = ts." + field + "\n"
    func_text += "    return bodo.hiframes.pd_index_ext.init_numeric_index(S, name)\n"
    loc_vars = {}
    # print(func_text)
    exec(func_text, {"numba": numba, "np": np, "bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def _install_dti_date_fields():
    for field in bodo.hiframes.pd_timestamp_ext.date_fields:
        if field in [
            "is_leap_year",
        ]:
            continue
        impl = gen_dti_field_impl(field)
        overload_attribute(DatetimeIndexType, field)(lambda dti: impl)


_install_dti_date_fields()


@overload_attribute(DatetimeIndexType, "is_leap_year")
def overload_datetime_index_is_leap_year(dti):
    def impl(dti):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        A = bodo.hiframes.pd_index_ext.get_index_data(dti)
        n = len(A)
        # TODO (ritwika): use nullable bool array.
        S = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            S[i] = bodo.hiframes.pd_timestamp_ext.is_leap_year(ts.year)
        return S

    return impl


@overload_attribute(DatetimeIndexType, "date")
def overload_datetime_index_date(dti):
    # TODO: NaN

    def impl(dti):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        A = bodo.hiframes.pd_index_ext.get_index_data(dti)
        n = len(A)
        S = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
        for i in numba.parfors.parfor.internal_prange(n):
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            S[i] = datetime.date(ts.year, ts.month, ts.day)
        return S

    return impl


@numba.njit(no_cpython_wrapper=True)
def _dti_val_finalize(s, count):  # pragma: no cover
    if not count:
        s = iNaT  # TODO: NaT type boxing in timestamp
    return bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(s)


@numba.njit(no_cpython_wrapper=True)
def _tdi_val_finalize(s, count):  # pragma: no cover
    return pd.Timedelta("nan") if not count else pd.Timedelta(s)


@overload_method(DatetimeIndexType, "min", no_unliteral=True)
def overload_datetime_index_min(dti, axis=None, skipna=True):
    # TODO skipna = False
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=None, skipna=True)
    check_unsupported_args("DatetimeIndex.min", unsupported_args, arg_defaults)

    def impl(dti, axis=None, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        in_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
        s = numba.cpython.builtins.get_type_max_value(numba.core.types.int64)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(in_arr)):
            if not bodo.libs.array_kernels.isna(in_arr, i):
                val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                s = min(s, val)
                count += 1
        return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

    return impl


# TODO: refactor min/max
@overload_method(DatetimeIndexType, "max", no_unliteral=True)
def overload_datetime_index_max(dti, axis=None, skipna=True):
    # TODO skipna = False
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=None, skipna=True)
    check_unsupported_args("DatetimeIndex.max", unsupported_args, arg_defaults)

    def impl(dti, axis=None, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        in_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
        s = numba.cpython.builtins.get_type_min_value(numba.core.types.int64)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(in_arr)):
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


@overload(pd.DatetimeIndex, no_unliteral=True)
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
        raise BodoError("data argument in pd.DatetimeIndex() expected")

    # check unsupported, TODO: normalize, dayfirst, yearfirst, ...
    if any(not is_overload_none(a) for a in (freq, start, end, periods, tz, closed)):
        raise BodoError("only data argument in pd.DatetimeIndex() supported")

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


@overload(operator.sub, no_unliteral=True)
def overload_datetime_index_sub(arg1, arg2):
    # DatetimeIndex - Timestamp
    if (
        isinstance(arg1, DatetimeIndexType)
        and arg2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
    ):
        timedelta64_dtype = np.dtype("timedelta64[ns]")

        def impl(arg1, arg2):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            in_arr = bodo.hiframes.pd_index_ext.get_index_data(arg1)
            name = bodo.hiframes.pd_index_ext.get_index_name(arg1)
            n = len(in_arr)
            S = np.empty(n, timedelta64_dtype)
            tsint = arg2.value
            for i in numba.parfors.parfor.internal_prange(n):
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
        timedelta64_dtype = np.dtype("timedelta64[ns]")

        def impl(arg1, arg2):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            in_arr = bodo.hiframes.pd_index_ext.get_index_data(arg2)
            name = bodo.hiframes.pd_index_ext.get_index_name(arg2)
            n = len(in_arr)
            S = np.empty(n, timedelta64_dtype)
            tsint = arg1.value
            for i in numba.parfors.parfor.internal_prange(n):
                S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    tsint - bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                )
            return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

        return impl


# bionp of DatetimeIndex and string
def gen_dti_str_binop_impl(op, is_lhs_dti):
    # is_arg1_dti: is the first argument DatetimeIndex and second argument str
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def impl(lhs, rhs):\n"
    if is_lhs_dti:
        func_text += "  dt_index, _str = lhs, rhs\n"
        comp = "arr[i] {} other".format(op_str)
    else:
        func_text += "  dt_index, _str = rhs, lhs\n"
        comp = "other {} arr[i]".format(op_str)
    func_text += "  arr = bodo.hiframes.pd_index_ext.get_index_data(dt_index)\n"
    func_text += "  l = len(arr)\n"
    func_text += "  other = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n"
    func_text += "  S = bodo.libs.bool_arr_ext.alloc_bool_array(l)\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(l):\n"
    func_text += "    S[i] = {}\n".format(comp)
    func_text += "  return S\n"
    # print(func_text)
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, loc_vars)
    impl = loc_vars["impl"]
    return impl


def overload_binop_dti_str(op):
    def overload_impl(lhs, rhs):
        if isinstance(lhs, DatetimeIndexType) and types.unliteral(rhs) == string_type:
            return gen_dti_str_binop_impl(op, True)
        if isinstance(rhs, DatetimeIndexType) and types.unliteral(lhs) == string_type:
            return gen_dti_str_binop_impl(op, False)

    return overload_impl


@overload(pd.Index, inline="always", no_unliteral=True)
def pd_index_overload(data=None, dtype=None, copy=False, name=None, tupleize_cols=True):

    # Todo: support Categorical dtype, Interval dtype, Period dtype, MultiIndex (?)
    # Todo: Extension dtype (?)

    # unliteral e.g. Tuple(Literal[int](3), Literal[int](1)) to UniTuple(int64 x 2)
    # NOTE: unliteral of LiteralList is Poison type in Numba
    data = types.unliteral(data) if not isinstance(data, types.LiteralList) else data

    data_dtype = getattr(data, "dtype", None)
    if not is_overload_none(dtype):
        elem_type = dtype.dtype
    else:
        elem_type = data_dtype

    # Range index:
    if isinstance(data, RangeIndexType):

        def impl(
            data=None, dtype=None, copy=False, name=None, tupleize_cols=True
        ):  # pragma: no cover
            return pd.RangeIndex(data, name=name)

    # Datetime index:
    elif isinstance(data, DatetimeIndexType) or elem_type == types.NPDatetime("ns"):

        def impl(
            data=None, dtype=None, copy=False, name=None, tupleize_cols=True
        ):  # pragma: no cover
            return pd.DatetimeIndex(data, name=name)

    # Timedelta index:
    elif isinstance(data, TimedeltaIndexType) or elem_type == types.NPTimedelta("ns"):

        def impl(
            data=None, dtype=None, copy=False, name=None, tupleize_cols=True
        ):  # pragma: no cover
            return pd.TimedeltaIndex(data, name=name)

    elif is_heterogeneous_tuple_type(data):
        # TODO(ehsan): handle 'dtype' argument if possible

        def impl(
            data=None, dtype=None, copy=False, name=None, tupleize_cols=True
        ):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_heter_index(data, name)

        return impl

    # ----- Data: Array type ------
    elif bodo.utils.utils.is_array_typ(data, False) or isinstance(
        data, (SeriesType, types.List, types.UniTuple)
    ):
        # Numeric Indices:
        if elem_type in (
            types.int64,
            types.int32,
            types.float64,
            types.uint32,
            types.uint64,
        ):

            def impl(
                data=None, dtype=None, copy=False, name=None, tupleize_cols=True
            ):  # pragma: no cover
                data_arr = bodo.utils.conversion.coerce_to_ndarray(data)
                data_coerced = bodo.utils.conversion.fix_arr_dtype(data_arr, elem_type)
                return bodo.hiframes.pd_index_ext.init_numeric_index(data_coerced, name)

        # String index:
        elif elem_type == types.string:

            def impl(
                data=None, dtype=None, copy=False, name=None, tupleize_cols=True
            ):  # pragma: no cover
                return bodo.hiframes.pd_index_ext.init_string_index(
                    bodo.utils.conversion.coerce_to_array(data), name
                )

        else:
            raise BodoError("pd.Index(): provided array is of unsupported type.")

    # raise error for data being None or scalar
    elif is_overload_none(data):
        raise BodoError(
            "data argument in pd.Index() is invalid: None or scalar is not acceptable"
        )
    else:
        raise BodoError(
            f"pd.Index(): the provided argument type {data} is not supported"
        )

    return impl


@overload(operator.getitem, no_unliteral=True)
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


@overload(operator.getitem, no_unliteral=True)
def overload_timedelta_index_getitem(I, ind):
    """getitem overload for TimedeltaIndex"""
    if not isinstance(I, TimedeltaIndexType):
        return

    if isinstance(ind, types.Integer):

        def impl(I, ind):  # pragma: no cover
            tdi_arr = bodo.hiframes.pd_index_ext.get_index_data(I)
            return pd.Timedelta(tdi_arr[ind])

        return impl

    # slice, boolean array, etc.
    # TODO: other Index or Series objects as index?
    def impl(I, ind):  # pragma: no cover
        tdi_arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        name = bodo.hiframes.pd_index_ext.get_index_name(I)
        new_arr = tdi_arr[ind]
        return bodo.hiframes.pd_index_ext.init_timedelta_index(new_arr, name)

    return impl


# from pandas.core.arrays.datetimelike
@numba.njit(no_cpython_wrapper=True)
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


@numba.njit(no_cpython_wrapper=True)
def to_offset_value(freq):  # pragma: no cover
    """Converts freq (string and integer) to offset nanoseconds."""
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
            # assert 0  # fails to compile in Numba 0.49 (test_pd_date_range)
            return 0

        return impl
    return lambda val: val  # pragma: no cover


@overload(pd.date_range, no_unliteral=True)
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

    unsupported_args = dict(tz=tz, normalize=normalize)
    arg_defaults = dict(tz=None, normalize=False)
    check_unsupported_args("pd.date_range", unsupported_args, arg_defaults)

    if not is_overload_none(tz):
        raise BodoError("pd.date_range(): tz argument not supported yet")

    if is_overload_none(freq) and any(
        is_overload_none(t) for t in (start, end, periods)
    ):
        freq = "D"  # change just to enable check below

    # exactly three parameters should
    if sum(not is_overload_none(t) for t in (start, end, periods, freq)) != 3:
        raise BodoError(
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
                periods = _dummy_convert_none_to_int(periods)
                b = start_t.value
                addend = np.int64(periods) * np.int64(stride)
                e = np.int64(b) + addend
            elif end is not None:
                periods = _dummy_convert_none_to_int(periods)
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


@overload(pd.timedelta_range, no_unliteral=True)
def pd_timedelta_range_overload(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name=None,
    closed=None,
):
    if is_overload_none(freq) and any(
        is_overload_none(t) for t in (start, end, periods)
    ):
        freq = "D"  # change just to enable check below

    # exactly three parameters should
    if sum(not is_overload_none(t) for t in (start, end, periods, freq)) != 3:
        raise BodoError(
            "Of the four parameters: start, end, periods, "
            "and freq, exactly three must be specified"
        )

    def f(
        start=None,
        end=None,
        periods=None,
        freq=None,
        name=None,
        closed=None,
    ):  # pragma: no cover

        # pandas source code performs the below conditional in timedelta_range
        if freq is None and (start is None or end is None or periods is None):
            freq = "D"
        freq = bodo.hiframes.pd_index_ext.to_offset_value(freq)

        start_t = pd.Timedelta("1 day")  # dummy value for typing
        if start is not None:
            start_t = pd.Timedelta(start)

        end_t = pd.Timedelta("1 day")  # dummy value for typing
        if end is not None:
            end_t = pd.Timedelta(end)

        if start is None and end is None and closed is not None:
            raise ValueError(
                "Closed has to be None if not both of start and end are defined"
            )

        left_closed, right_closed = bodo.hiframes.pd_index_ext.validate_endpoints(
            closed
        )

        if freq is not None:
            # pandas/core/arrays/_ranges/generate_regular_range
            stride = freq
            if periods is None:
                b = start_t.value
                e = b + (end_t.value - b) // stride * stride + stride // 2 + 1
            elif start is not None:
                periods = _dummy_convert_none_to_int(periods)
                b = start_t.value
                addend = np.int64(periods) * np.int64(stride)
                e = np.int64(b) + addend
            elif end is not None:
                periods = _dummy_convert_none_to_int(periods)
                e = end_t.value + stride
                addend = np.int64(periods) * np.int64(-stride)
                b = np.int64(e) + addend
            else:
                raise ValueError(
                    "at least 'start' or 'end' should be specified "
                    "if a 'period' is given."
                )
            arr = np.arange(b, e, stride, np.int64)
        else:
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
        return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

    return f


@overload_method(DatetimeIndexType, "isocalendar", inline="always", no_unliteral=True)
def overload_pd_timestamp_isocalendar(idx):
    def impl(idx):  # pragma: no cover
        A = bodo.hiframes.pd_index_ext.get_index_data(idx)
        numba.parfors.parfor.init_prange()
        n = len(A)
        years = bodo.libs.int_arr_ext.alloc_int_array(n, np.uint32)
        weeks = bodo.libs.int_arr_ext.alloc_int_array(n, np.uint32)
        days = bodo.libs.int_arr_ext.alloc_int_array(n, np.uint32)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(A, i):
                bodo.libs.array_kernels.setna(years, i)
                bodo.libs.array_kernels.setna(weeks, i)
                bodo.libs.array_kernels.setna(days, i)
                continue
            (
                years[i],
                weeks[i],
                days[i],
            ) = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                A[i]
            ).isocalendar()
        return bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (years, weeks, days), idx, ("year", "week", "day")
        )

    return impl


# ------------------------------ Timedelta ---------------------------


# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType, types.ArrayCompatible):
    """Temporary type class for TimedeltaIndex objects."""

    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like unit/freq?
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = types.Array(bodo.timedelta64ns, 1, "C")
        super(TimedeltaIndexType, self).__init__(
            name="TimedeltaIndexType(named = {})".format(name_typ)
        )

    ndim = 1

    def copy(self):
        return TimedeltaIndexType(self.name_typ)

    @property
    def dtype(self):
        return types.NPTimedelta("ns")

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def key(self):
        # needed?
        return self.name_typ

    @property
    def iterator_type(self):
        # same as Buffer
        # TODO: fix timedelta
        return types.iterators.ArrayIterator(_timedelta_index_data_typ)


timedelta_index = TimedeltaIndexType()
types.timedelta_index = timedelta_index


@register_model(TimedeltaIndexType)
class TimedeltaIndexTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", _timedelta_index_data_typ),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(_timedelta_index_data_typ.dtype, types.int64)),
        ]
        super(TimedeltaIndexTypeModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(pd.TimedeltaIndex)
def typeof_timedelta_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return TimedeltaIndexType(get_val_type_maybe_str_literal(val.name))


@box(TimedeltaIndexType)
def box_timedelta_index(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    timedelta_index = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, val
    )

    c.context.nrt.incref(c.builder, _timedelta_index_data_typ, timedelta_index.data)
    arr_obj = c.pyapi.from_native_value(
        _timedelta_index_data_typ, timedelta_index.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, timedelta_index.name)
    name_obj = c.pyapi.from_native_value(
        typ.name_typ, timedelta_index.name, c.env_manager
    )

    # call pd.TimedeltaIndex(arr, name=name)
    args = c.pyapi.tuple_pack([arr_obj])
    kws = c.pyapi.dict_pack([("name", name_obj)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "TimedeltaIndex")
    res = c.pyapi.call(const_call, args, kws)

    c.pyapi.decref(arr_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(TimedeltaIndexType)
def unbox_timedelta_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    values_obj = c.pyapi.object_getattr_string(val, "values")
    data = c.pyapi.to_native_value(_timedelta_index_data_typ, values_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    dtype = _timedelta_index_data_typ.dtype
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@intrinsic
def init_timedelta_index(typingctx, data, name=None):
    """Create a TimedeltaIndex with provided data and name values."""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create timedelta_index struct and store values
        timedelta_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        timedelta_index.data = data_val
        timedelta_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], name_val)

        # create empty dict for get_loc hashmap
        dtype = _timedelta_index_data_typ.dtype
        timedelta_index.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(dtype, types.int64),
            types.DictType(dtype, types.int64)(),
            [],
        )  # pragma: no cover

        return timedelta_index._getvalue()

    ret_typ = TimedeltaIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_timedelta_index = (
    init_index_equiv
)


@infer_getattr
class TimedeltaIndexAttribute(AttributeTemplate):
    key = TimedeltaIndexType

    def resolve_values(self, ary):
        return _timedelta_index_data_typ

    # TODO: support pd.Timedelta
    # @bound_function("timedelta_index.max", no_unliteral=True)
    # def resolve_max(self, ary, args, kws):
    #     assert not kws
    #     return signature(pandas_timestamp_type, *args)

    # @bound_function("timedelta_index.min", no_unliteral=True)
    # def resolve_min(self, ary, args, kws):
    #     assert not kws
    #     return signature(pandas_timestamp_type, *args)


make_attribute_wrapper(TimedeltaIndexType, "data", "_data")
make_attribute_wrapper(TimedeltaIndexType, "name", "_name")
make_attribute_wrapper(TimedeltaIndexType, "dict", "_dict")


@overload_method(TimedeltaIndexType, "copy", no_unliteral=True)
def overload_timedelta_index_copy(A):
    return lambda A: bodo.hiframes.pd_index_ext.init_timedelta_index(
        A._data.copy(), A._name
    )  # pragma: no cover


@overload_method(TimedeltaIndexType, "min", inline="always", no_unliteral=True)
def overload_timedelta_index_min(tdi, axis=None, skipna=True):
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=None, skipna=True)
    check_unsupported_args("TimedeltaIndex.min", unsupported_args, arg_defaults)

    def impl(tdi, axis=None, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        data = bodo.hiframes.pd_index_ext.get_index_data(tdi)
        n = len(data)
        min_val = numba.cpython.builtins.get_type_max_value(numba.core.types.int64)
        count = 0
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(data, i):
                continue
            val = bodo.hiframes.datetime_timedelta_ext.cast_numpy_timedelta_to_int(
                data[i]
            )
            count += 1
            min_val = min(min_val, val)
        ret_val = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(min_val)

        return bodo.hiframes.pd_index_ext._tdi_val_finalize(ret_val, count)

    return impl


@overload_method(TimedeltaIndexType, "max", inline="always", no_unliteral=True)
def overload_timedelta_index_max(tdi, axis=None, skipna=True):
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=None, skipna=True)
    check_unsupported_args("TimedeltaIndex.max", unsupported_args, arg_defaults)

    if not is_overload_none(axis) or not is_overload_true(skipna):
        raise BodoError("Index.min(): axis and skipna arguments not supported yet")

    def impl(tdi, axis=None, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        data = bodo.hiframes.pd_index_ext.get_index_data(tdi)
        n = len(data)
        max_val = numba.cpython.builtins.get_type_min_value(numba.core.types.int64)
        count = 0
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(data, i):
                continue
            val = bodo.hiframes.datetime_timedelta_ext.cast_numpy_timedelta_to_int(
                data[i]
            )
            count += 1
            max_val = max(max_val, val)
        ret_val = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(max_val)
        return bodo.hiframes.pd_index_ext._tdi_val_finalize(ret_val, count)

    return impl


@overload_attribute(TimedeltaIndexType, "name")
def TimeDeltaIndex_get_name(tdi):
    def impl(tdi):  # pragma: no cover
        return tdi._name

    return impl


# support TimedeltaIndex time fields such as T.days
def gen_tdi_field_impl(field):
    # TODO: NaN
    func_text = "def impl(tdi):\n"
    func_text += "    numba.parfors.parfor.init_prange()\n"
    func_text += "    A = bodo.hiframes.pd_index_ext.get_index_data(tdi)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(tdi)\n"
    func_text += "    n = len(A)\n"
    # all timedeltaindex fields return int64 same as Timestamp fields
    func_text += "    S = np.empty(n, np.int64)\n"
    # TODO: use nullable int when supported by NumericIndex?
    # func_text += "    S = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)\n"
    func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
    # func_text += "        if bodo.libs.array_kernels.isna(A, i):\n"
    # func_text += "            bodo.libs.array_kernels.setna(S, i)\n"
    # func_text += "            continue\n"
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


@overload(pd.TimedeltaIndex, no_unliteral=True)
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
        raise BodoError("data argument in pd.TimedeltaIndex() expected")

    if any(
        not is_overload_none(a)
        for a in (unit, freq, start, end, periods, closed, dtype)
    ):
        raise BodoError("only data argument in pd.TimedeltaIndex() supported")

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
class RangeIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.RangeIndex() objects."""

    def __init__(self, name_typ):
        self.name_typ = name_typ
        super(RangeIndexType, self).__init__(name="RangeIndexType({})".format(name_typ))

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return RangeIndexType(self.name_typ)

    @property
    def iterator_type(self):
        return types.iterators.RangeIteratorType(types.int64)

    @property
    def dtype(self):
        return types.int64

    @property
    def pandas_type_name(self):
        return str(self.dtype)

    @property
    def numpy_type_name(self):
        return str(self.dtype)

    def unify(self, typingctx, other):
        """unify RangeIndexType with equivalent NumericIndexType"""
        if isinstance(other, NumericIndexType):
            name_typ = self.name_typ.unify(typingctx, other.name_typ)
            # TODO: test and support name type differences properly
            if name_typ is None:
                name_typ = types.none
            return NumericIndexType(types.int64, name_typ)


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


@overload_method(RangeIndexType, "copy", no_unliteral=True)
def overload_range_index_copy(A):
    return lambda A: bodo.hiframes.pd_index_ext.init_range_index(
        A._start, A._stop, A._step, A._name
    )  # pragma: no cover


@box(RangeIndexType)
def box_range_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    range_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    start_obj = c.pyapi.from_native_value(types.int64, range_val.start, c.env_manager)
    stop_obj = c.pyapi.from_native_value(types.int64, range_val.stop, c.env_manager)
    step_obj = c.pyapi.from_native_value(types.int64, range_val.step, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_typ, range_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, range_val.name, c.env_manager)

    # call pd.RangeIndex(start, stop, step, name=name)
    args = c.pyapi.tuple_pack([start_obj, stop_obj, step_obj])
    kws = c.pyapi.dict_pack([("name", name_obj)])
    const_call = c.pyapi.object_getattr_string(class_obj, "RangeIndex")
    index_obj = c.pyapi.call(const_call, args, kws)

    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)
    c.pyapi.decref(step_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@intrinsic
def init_range_index(typingctx, start, stop, step, name=None):
    """Create RangeIndex object"""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        assert len(args) == 4
        range_val = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        range_val.start = args[0]
        range_val.stop = args[1]
        range_val.step = args[2]
        range_val.name = args[3]
        context.nrt.incref(builder, signature.return_type.name_typ, args[3])
        return range_val._getvalue()

    return RangeIndexType(name)(start, stop, step, name), codegen


def init_range_index_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for RangeIndex. We can infer equivalence only when start=0 and
    step=1.
    """
    assert len(args) == 4 and not kws
    start, stop, step, _ = args
    # RangeIndex is equivalent to 'stop' input when start=0 and step=1
    if (
        self.typemap[start.name] == types.IntegerLiteral(0)
        and self.typemap[step.name] == types.IntegerLiteral(1)
        and equiv_set.has_shape(stop)
    ):
        return ArrayAnalysis.AnalyzeResult(shape=stop, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_range_index = (
    init_range_index_equiv
)


@unbox(RangeIndexType)
def unbox_range_index(typ, val, c):
    # get start/stop/step attributes
    start_obj = c.pyapi.object_getattr_string(val, "start")
    start = c.pyapi.to_native_value(types.int64, start_obj).value
    stop_obj = c.pyapi.object_getattr_string(val, "stop")
    stop = c.pyapi.to_native_value(types.int64, stop_obj).value
    step_obj = c.pyapi.object_getattr_string(val, "step")
    step = c.pyapi.to_native_value(types.int64, step_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(start_obj)
    c.pyapi.decref(stop_obj)
    c.pyapi.decref(step_obj)
    c.pyapi.decref(name_obj)

    # create range struct
    range_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    range_val.start = start
    range_val.stop = stop
    range_val.step = step
    range_val.name = name
    return NativeValue(range_val._getvalue())


@lower_constant(RangeIndexType)
def lower_constant_range_index(context, builder, ty, pyval):
    """embed constant RangeIndex by simply creating the data struct and assigning values"""
    start = context.get_constant(types.int64, pyval.start)
    stop = context.get_constant(types.int64, pyval.stop)
    step = context.get_constant(types.int64, pyval.step)
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    # create range struct
    range_val = cgutils.create_struct_proxy(ty)(context, builder)
    range_val.start = start
    range_val.stop = stop
    range_val.step = step
    range_val.name = name
    return range_val._getvalue()


@overload(pd.RangeIndex, no_unliteral=True)
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


@overload_attribute(RangeIndexType, "name")
def rangeIndex_get_name(ri):
    def impl(ri):  # pragma: no cover
        return ri._name

    return impl


@overload_attribute(RangeIndexType, "start")
def rangeIndex_get_start(ri):
    def impl(ri):  # pragma: no cover
        return ri._start

    return impl


@overload_attribute(RangeIndexType, "stop")
def rangeIndex_get_stop(ri):
    def impl(ri):  # pragma: no cover
        return ri._stop

    return impl


@overload_attribute(RangeIndexType, "step")
def rangeIndex_get_step(ri):
    def impl(ri):  # pragma: no cover
        return ri._step

    return impl


@overload(operator.getitem, no_unliteral=True)
def overload_range_index_getitem(I, idx):
    if isinstance(I, RangeIndexType):
        if isinstance(types.unliteral(idx), types.Integer):
            # TODO: test
            # TODO: check valid
            return lambda I, idx: (idx * I._step) + I._start  # pragma: no cover

        if isinstance(idx, types.SliceType):
            # TODO: test
            def impl(I, idx):  # pragma: no cover
                slice_idx = numba.cpython.unicode._normalize_slice(idx, len(I))
                name = bodo.hiframes.pd_index_ext.get_index_name(I)
                start = I._start + I._step * slice_idx.start
                stop = I._start + I._step * slice_idx.stop
                step = I._step * slice_idx.step
                return bodo.hiframes.pd_index_ext.init_range_index(
                    start, stop, step, name
                )

            return impl

        # delegate to integer index, TODO: test
        return lambda I, idx: bodo.hiframes.pd_index_ext.init_numeric_index(
            np.arange(I._start, I._stop, I._step, np.int64)[idx],
            bodo.hiframes.pd_index_ext.get_index_name(I),
        )  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_range_len(r):
    if isinstance(r, RangeIndexType):
        # TODO: test
        return lambda r: max(0, -(-(r._stop - r._start) // r._step))  # pragma: no cover


# ---------------- PeriodIndex -------------------


# Simple type for PeriodIndex for now, freq is saved as a constant string
class PeriodIndexType(types.IterableType):
    """type class for pd.PeriodIndex. Contains frequency as constant string"""

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
            ("dict", types.DictType(types.int64, types.int64)),
        ]
        super(PeriodIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(PeriodIndexType, "data", "_data")
make_attribute_wrapper(PeriodIndexType, "name", "_name")
make_attribute_wrapper(PeriodIndexType, "dict", "_dict")


@overload_method(PeriodIndexType, "copy", no_unliteral=True)
def overload_period_index_copy(A):
    freq = A.freq
    return lambda A: bodo.hiframes.pd_index_ext.init_period_index(
        A._data.copy(), A._name, freq
    )  # pragma: no cover


@intrinsic
def init_period_index(typingctx, data, name, freq):
    """Create a PeriodIndex with provided data, name and freq values."""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        data_val, name_val, _ = args
        index_typ = signature.return_type
        period_index = cgutils.create_struct_proxy(index_typ)(context, builder)
        period_index.data = data_val
        period_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], args[0])
        context.nrt.incref(builder, signature.args[1], args[1])

        # create empty dict for get_loc hashmap
        period_index.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(types.int64, types.int64),
            types.DictType(types.int64, types.int64)(),
            [],
        )  # pragma: no cover

        return period_index._getvalue()

    freq_val = get_overload_const_str(freq)
    ret_typ = PeriodIndexType(freq_val, name)
    sig = signature(ret_typ, data, name, freq)
    return sig, codegen


@overload_attribute(PeriodIndexType, "name")
def PeriodIndex_get_name(pi):
    def impl(pi):  # pragma: no cover
        return pi._name

    return impl


@box(PeriodIndexType)
def box_period_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    c.context.nrt.incref(c.builder, types.Array(types.int64, 1, "C"), index_val.data)
    data_obj = c.pyapi.from_native_value(
        types.Array(types.int64, 1, "C"), index_val.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)
    freq_obj = c.pyapi.string_from_constant_string(typ.freq)

    # call pd.PeriodIndex(ordinal=data, name=name, freq=freq)
    args = c.pyapi.tuple_pack([])
    kws = c.pyapi.dict_pack(
        [("ordinal", data_obj), ("name", name_obj), ("freq", freq_obj)]
    )
    const_call = c.pyapi.object_getattr_string(class_obj, "PeriodIndex")
    index_obj = c.pyapi.call(const_call, args, kws)

    c.pyapi.decref(data_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(freq_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@unbox(PeriodIndexType)
def unbox_period_index(typ, val, c):
    # get data and name attributes
    arr_typ = types.Array(types.int64, 1, "C")
    asi8_obj = c.pyapi.object_getattr_string(val, "asi8")
    data = c.pyapi.to_native_value(arr_typ, asi8_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(asi8_obj)
    c.pyapi.decref(name_obj)

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(types.int64, types.int64),
        types.DictType(types.int64, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


# ---------------- NumericIndex -------------------


# represents numeric indices (excluding RangeIndex):
#   Int64Index, UInt64Index, Float64Index
class NumericIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.Int64Index/UInt64Index/Float64Index objects."""

    def __init__(self, dtype, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        self.dtype = dtype
        self.name_typ = name_typ
        data = (
            bodo.hiframes.pd_series_ext._get_series_array_type(dtype)
            if data is None
            else data
        )
        self.data = data
        super(NumericIndexType, self).__init__(
            name=f"NumericIndexType({dtype}, {name_typ}, {data})"
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return NumericIndexType(self.dtype, self.name_typ)

    @property
    def iterator_type(self):
        # TODO: handle iterator
        return types.iterators.ArrayIterator(types.Array(self.dtype, 1, "C"))

    @property
    def pandas_type_name(self):
        return str(self.dtype)

    @property
    def numpy_type_name(self):
        return str(self.dtype)


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
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(fe_type.dtype, types.int64)),
        ]
        super(NumericIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(NumericIndexType, "data", "_data")
make_attribute_wrapper(NumericIndexType, "name", "_name")
make_attribute_wrapper(NumericIndexType, "dict", "_dict")


@overload_method(NumericIndexType, "copy", no_unliteral=True)
def overload_numeric_index_copy(A):
    return lambda A: bodo.hiframes.pd_index_ext.init_numeric_index(
        A._data.copy(), A._name
    )  # pragma: no cover


@overload_attribute(NumericIndexType, "name")
def NumericIndex_get_name(ni):
    def impl(ni):  # pragma: no cover
        return ni._name

    return impl


@box(NumericIndexType)
def box_numeric_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    c.context.nrt.incref(c.builder, typ.data, index_val.data)
    data_obj = c.pyapi.from_native_value(typ.data, index_val.data, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    assert typ.dtype in (types.int64, types.uint64, types.float64)
    func_name = "Int64Index"
    if typ.dtype == types.uint64:
        func_name = "UInt64Index"
    elif typ.dtype == types.float64:
        func_name = "Float64Index"
    else:
        assert typ.dtype == types.int64

    dtype_obj = c.pyapi.make_none()
    copy_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    index_obj = c.pyapi.call_method(
        class_obj, func_name, (data_obj, dtype_obj, copy_obj, name_obj)
    )

    c.pyapi.decref(data_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(copy_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@intrinsic
def init_numeric_index(typingctx, data, name=None):
    """Create NumericIndex object"""
    name = types.none if is_overload_none(name) else name

    def codegen(context, builder, signature, args):
        assert len(args) == 2
        index_typ = signature.return_type
        index_val = cgutils.create_struct_proxy(index_typ)(context, builder)
        index_val.data = args[0]
        index_val.name = args[1]
        # increase refcount of stored values
        context.nrt.incref(builder, index_typ.data, args[0])
        context.nrt.incref(builder, index_typ.name_typ, args[1])
        # create empty dict for get_loc hashmap
        dtype = index_typ.dtype
        index_val.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(dtype, types.int64),
            types.DictType(dtype, types.int64)(),
            [],
        )  # pragma: no cover
        return index_val._getvalue()

    return NumericIndexType(data.dtype, name, data)(data, name), codegen


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_numeric_index = (
    init_index_equiv
)


@unbox(NumericIndexType)
def unbox_numeric_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    values_obj = c.pyapi.object_getattr_string(val, "values")
    data = c.pyapi.to_native_value(typ.data, values_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    dtype = typ.dtype
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
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
        overload(func, no_unliteral=True)(overload_impl)


_install_numeric_constructors()


# ---------------- StringIndex -------------------


# represents string index, which doesn't have direct Pandas type
# pd.Index() infers string
class StringIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.Index() objects with 'string' as inferred_dtype."""

    def __init__(self, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = string_array_type
        super(StringIndexType, self).__init__(
            name="StringIndexType({})".format(name_typ)
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return StringIndexType(self.name_typ)

    @property
    def dtype(self):
        return string_type

    @property
    def pandas_type_name(self):
        return "unicode"

    @property
    def numpy_type_name(self):
        return "object"

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
            ("data", string_array_type),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(string_type, types.int64)),
        ]
        super(StringIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(StringIndexType, "data", "_data")
make_attribute_wrapper(StringIndexType, "name", "_name")
make_attribute_wrapper(StringIndexType, "dict", "_dict")


@overload_method(StringIndexType, "copy", no_unliteral=True)
def overload_string_index_copy(A):
    return lambda A: bodo.hiframes.pd_index_ext.init_string_index(
        A._data.copy(), A._name
    )  # pragma: no cover


@box(StringIndexType)
def box_string_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    c.context.nrt.incref(c.builder, string_array_type, index_val.data)
    data_obj = c.pyapi.from_native_value(
        string_array_type, index_val.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    dtype_obj = c.pyapi.make_none()
    copy_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    # call pd.Index(data, dtype, copy, name)
    index_obj = c.pyapi.call_method(
        class_obj, "Index", (data_obj, dtype_obj, copy_obj, name_obj)
    )

    c.pyapi.decref(data_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(copy_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@intrinsic
def init_string_index(typingctx, data, name=None):
    """Create StringIndex object"""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        assert len(args) == 2
        index_typ = signature.return_type
        index_val = cgutils.create_struct_proxy(index_typ)(context, builder)
        index_val.data = args[0]
        index_val.name = args[1]
        # increase refcount of stored values
        context.nrt.incref(builder, string_array_type, args[0])
        context.nrt.incref(builder, index_typ.name_typ, args[1])
        # create empty dict for get_loc hashmap
        index_val.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(string_type, types.int64),
            types.DictType(string_type, types.int64)(),
            [],
        )  # pragma: no cover
        return index_val._getvalue()

    return StringIndexType(name)(data, name), codegen


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_string_index = (
    init_index_equiv
)


@unbox(StringIndexType)
def unbox_string_index(typ, val, c):
    # get data and name attributes
    # TODO: use to_numpy()
    values_obj = c.pyapi.object_getattr_string(val, "values")
    data = c.pyapi.to_native_value(string_array_type, values_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(values_obj)
    c.pyapi.decref(name_obj)

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(string_type, types.int64),
        types.DictType(string_type, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@overload_attribute(StringIndexType, "name")
def stringIndex_get_name(si):
    def impl(si):  # pragma: no cover
        return si._name

    return impl


@overload(operator.getitem, no_unliteral=True)
def overload_index_getitem(I, ind):
    # output of integer indexing is scalar value
    if isinstance(I, (NumericIndexType, StringIndexType)) and isinstance(
        ind, types.Integer
    ):
        return lambda I, ind: bodo.hiframes.pd_index_ext.get_index_data(I)[
            ind
        ]  # pragma: no cover

    # output of slice, bool array ... indexing is pd.Index
    if isinstance(I, NumericIndexType):
        return lambda I, ind: bodo.hiframes.pd_index_ext.init_numeric_index(
            bodo.hiframes.pd_index_ext.get_index_data(I)[ind],
            bodo.hiframes.pd_index_ext.get_index_name(I),
        )  # pragma: no cover

    if isinstance(I, StringIndexType):
        return lambda I, ind: bodo.hiframes.pd_index_ext.init_string_index(
            bodo.hiframes.pd_index_ext.get_index_data(I)[ind],
            bodo.hiframes.pd_index_ext.get_index_name(I),
        )  # pragma: no cover


# similar to index_from_array()
def array_typ_to_index(arr_typ, name_typ=None):
    if arr_typ == bodo.string_array_type:
        return StringIndexType(name_typ)

    assert (
        isinstance(arr_typ, (types.Array, IntegerArrayType))
        or arr_typ == bodo.datetime_date_array_type
    ), f"Converting array type {arr_typ} to index not supported"

    # TODO: Pandas keeps datetime_date Index as a generic Index(, dtype=object)
    # Fix this implementation to match.
    if arr_typ == bodo.datetime_date_array_type or arr_typ.dtype == types.NPDatetime(
        "ns"
    ):
        return DatetimeIndexType(name_typ)

    if arr_typ.dtype == types.NPTimedelta("ns"):
        return TimedeltaIndexType(name_typ)

    if isinstance(arr_typ.dtype, types.Integer):
        if not arr_typ.dtype.signed:
            return NumericIndexType(types.uint64, name_typ, arr_typ)
        else:
            return NumericIndexType(types.int64, name_typ, arr_typ)

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
            HeterogeneousIndexType,
        ),
    )


# TODO: test
@overload_method(RangeIndexType, "take", no_unliteral=True)
@overload_method(NumericIndexType, "take", no_unliteral=True)
@overload_method(StringIndexType, "take", no_unliteral=True)
@overload_method(PeriodIndexType, "take", no_unliteral=True)
@overload_method(DatetimeIndexType, "take", no_unliteral=True)
@overload_method(TimedeltaIndexType, "take", no_unliteral=True)
def overload_index_take(I, indices):
    return lambda I, indices: I[indices]  # pragma: no cover


@numba.njit(no_cpython_wrapper=True)
def _init_engine(I):  # pragma: no cover
    """initialize the Index hashmap engine (just a simple dict for now)"""
    if len(I) > 0 and not I._dict:
        arr = bodo.utils.conversion.coerce_to_array(I)
        for i in range(len(arr)):
            val = arr[i]
            if val in I._dict:
                raise ValueError("Index.get_loc(): non-unique Index not supported yet")
            I._dict[val] = i


@overload(operator.contains, no_unliteral=True)
def index_contains(I, val):
    """support for "val in I" operator. Uses the Index hashmap for faster results."""
    if not is_index_type(I):  # pragma: no cover
        return

    if isinstance(I, RangeIndexType):
        return lambda I, val: range_contains(
            I.start, I.stop, I.step, val
        )  # pragma: no cover

    def impl(I, val):  # pragma: no cover
        # build the index dict if not initialized yet
        _init_engine(I)
        return bodo.utils.conversion.unbox_if_timestamp(val) in I._dict

    return impl


@register_jitable
def range_contains(start, stop, step, val):  # pragma: no cover
    """check 'val' to be in range(start, stop, step)"""

    # check to see if value in start/stop range (NOTE: step cannot be 0)
    if step > 0 and not (start <= val < stop):
        return False
    if step < 0 and not (stop <= val < start):
        return False

    # check stride
    return ((val - start) % step) == 0


@overload_method(RangeIndexType, "get_loc", no_unliteral=True)
@overload_method(NumericIndexType, "get_loc", no_unliteral=True)
@overload_method(StringIndexType, "get_loc", no_unliteral=True)
@overload_method(PeriodIndexType, "get_loc", no_unliteral=True)
@overload_method(DatetimeIndexType, "get_loc", no_unliteral=True)
@overload_method(TimedeltaIndexType, "get_loc", no_unliteral=True)
def overload_index_get_loc(I, key, method=None, tolerance=None):
    """simple get_loc implementation intended for cases with small Index like
    df.columns.get_loc(c). Only supports Index with unique values (scalar return).
    TODO(ehsan): use a proper hash engine like Pandas inside Index objects
    """
    unsupported_args = dict(method=method, tolerance=tolerance)
    arg_defaults = dict(method=None, tolerance=None)
    check_unsupported_args("Index.get_loc", unsupported_args, arg_defaults)

    # Timestamp/Timedelta types are handled the same as datetime64/timedelta64
    key = types.unliteral(key)
    if key == pandas_timestamp_type:
        key = bodo.datetime64ns
    if key == pd_timedelta_type:
        key = bodo.timedelta64ns

    if key != I.dtype:  # pragma: no cover
        raise_bodo_error("Index.get_loc(): invalid label type in Index.get_loc()")

    # RangeIndex doesn't need a hashmap
    if isinstance(I, RangeIndexType):
        # Pandas uses range.index() of Python, so using similar implementation
        # https://github.com/python/cpython/blob/8e1b40627551909687db8914971b0faf6cf7a079/Objects/rangeobject.c#L576
        def impl_range(I, key, method=None, tolerance=None):  # pragma: no cover
            if not range_contains(I.start, I.stop, I.step, key):
                raise KeyError("Index.get_loc(): key not found")
            return key - I.start if I.step == 1 else (key - I.start) // I.step

        return impl_range

    def impl(I, key, method=None, tolerance=None):  # pragma: no cover
        # build the index dict if not initialized yet
        _init_engine(I)

        key = bodo.utils.conversion.unbox_if_timestamp(key)
        ind = I._dict.get(key, -1)

        if ind == -1:
            raise KeyError("Index.get_loc(): key not found")
        return ind

    return impl


@overload_method(RangeIndexType, "isna", no_unliteral=True)
@overload_method(NumericIndexType, "isna", no_unliteral=True)
@overload_method(StringIndexType, "isna", no_unliteral=True)
@overload_method(PeriodIndexType, "isna", no_unliteral=True)
@overload_method(DatetimeIndexType, "isna", no_unliteral=True)
@overload_method(TimedeltaIndexType, "isna", no_unliteral=True)
@overload_method(RangeIndexType, "isnull", no_unliteral=True)
@overload_method(NumericIndexType, "isnull", no_unliteral=True)
@overload_method(StringIndexType, "isnull", no_unliteral=True)
@overload_method(PeriodIndexType, "isnull", no_unliteral=True)
@overload_method(DatetimeIndexType, "isnull", no_unliteral=True)
@overload_method(TimedeltaIndexType, "isnull", no_unliteral=True)
def overload_index_isna(I):
    if isinstance(I, RangeIndexType):
        # TODO: parallelize np.full in PA
        # return lambda I: np.full(len(I), False, np.bool_)
        def impl(I):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            n = len(I)
            out_arr = np.empty(n, np.bool_)
            for i in numba.parfors.parfor.internal_prange(n):
                out_arr[i] = False
            return out_arr

        return impl

    def impl(I):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        n = len(arr)
        out_arr = np.empty(n, np.bool_)
        for i in numba.parfors.parfor.internal_prange(n):
            out_arr[i] = bodo.libs.array_kernels.isna(arr, i)
        return out_arr

    return impl


@overload_attribute(RangeIndexType, "values")
@overload_attribute(NumericIndexType, "values")
@overload_attribute(StringIndexType, "values")
@overload_attribute(PeriodIndexType, "values")
@overload_attribute(DatetimeIndexType, "values")
@overload_attribute(TimedeltaIndexType, "values")
def overload_values(I):
    return lambda I: bodo.utils.conversion.coerce_to_array(I)  # pragma: no cover


@overload(len, no_unliteral=True)
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
        return lambda I: len(
            bodo.hiframes.pd_index_ext.get_index_data(I)
        )  # pragma: no cover


@overload_attribute(DatetimeIndexType, "shape")
@overload_attribute(NumericIndexType, "shape")
@overload_attribute(StringIndexType, "shape")
@overload_attribute(PeriodIndexType, "shape")
@overload_attribute(TimedeltaIndexType, "shape")
def overload_index_shape(s):
    return lambda s: (
        len(bodo.hiframes.pd_index_ext.get_index_data(s)),
    )  # pragma: no cover


@overload_attribute(RangeIndexType, "shape")
def overload_range_index_shape(s):
    return lambda s: (len(s),)  # pragma: no cover


@numba.generated_jit(nopython=True)
def get_index_data(S):
    return lambda S: S._data  # pragma: no cover


@numba.generated_jit(nopython=True)
def get_index_name(S):
    return lambda S: S._name  # pragma: no cover


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("get_index_data", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("init_datetime_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("init_timedelta_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("init_numeric_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("init_string_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func


# array analysis extension
def get_index_data_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_get_index_data = (
    get_index_data_equiv
)


@overload_method(RangeIndexType, "map", inline="always", no_unliteral=True)
@overload_method(NumericIndexType, "map", inline="always", no_unliteral=True)
@overload_method(StringIndexType, "map", inline="always", no_unliteral=True)
@overload_method(PeriodIndexType, "map", inline="always", no_unliteral=True)
@overload_method(DatetimeIndexType, "map", inline="always", no_unliteral=True)
@overload_method(TimedeltaIndexType, "map", inline="always", no_unliteral=True)
def overload_index_map(I, mapper, na_action=None):
    if not is_const_func_type(mapper):
        raise BodoError("Index.map(): 'mapper' should be a function")

    dtype = I.dtype
    # getitem returns Timestamp for dt_index (TODO: pd.Timedelta when available)
    if dtype == types.NPDatetime("ns"):
        dtype = pandas_timestamp_type
    if dtype == types.NPTimedelta("ns"):
        dtype = pd_timedelta_type

    # get output element type
    typing_context = numba.core.registry.cpu_target.typing_context
    try:
        f_return_type = get_const_func_output_type(mapper, (dtype,), {}, typing_context)
    except Exception as e:
        raise_bodo_error(get_udf_error_msg("Index.map()", e), e.loc)

    out_arr_type = get_udf_out_arr_type(f_return_type)

    func = get_overload_const_func(mapper)
    func_text = "def f(I, mapper, na_action=None):\n"
    func_text += "  name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    func_text += "  A = bodo.utils.conversion.coerce_to_array(I)\n"
    func_text += "  numba.parfors.parfor.init_prange()\n"
    func_text += "  n = len(A)\n"
    func_text += "  S = bodo.utils.utils.alloc_type(n, _arr_typ, (-1,))\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "    t2 = bodo.utils.conversion.box_if_dt64(A[i])\n"
    func_text += "    v = map_func(t2)\n"
    func_text += "    S[i] = bodo.utils.conversion.unbox_if_timestamp(v)\n"
    func_text += "  return bodo.utils.conversion.index_from_array(S, name)\n"

    map_func = bodo.compiler.udf_jit(func)

    loc_vars = {}
    exec(
        func_text,
        {
            "numba": numba,
            "np": np,
            "pd": pd,
            "bodo": bodo,
            "map_func": map_func,
            "_arr_typ": out_arr_type,
            "init_nested_counts": bodo.utils.indexing.init_nested_counts,
            "add_nested_counts": bodo.utils.indexing.add_nested_counts,
            "data_arr_type": out_arr_type.dtype,
        },
        loc_vars,
    )
    f = loc_vars["f"]
    return f


@lower_builtin(operator.is_, NumericIndexType, NumericIndexType)
@lower_builtin(operator.is_, StringIndexType, StringIndexType)
@lower_builtin(operator.is_, PeriodIndexType, PeriodIndexType)
@lower_builtin(operator.is_, DatetimeIndexType, DatetimeIndexType)
@lower_builtin(operator.is_, TimedeltaIndexType, TimedeltaIndexType)
def index_is(context, builder, sig, args):
    aty, bty = sig.args
    if aty != bty:  # pragma: no cover
        return cgutils.false_bit

    def index_is_impl(a, b):  # pragma: no cover
        return a._data is b._data and a._name is b._name

    return context.compile_internal(builder, index_is_impl, sig, args)


@lower_builtin(operator.is_, RangeIndexType, RangeIndexType)
def range_index_is(context, builder, sig, args):
    aty, bty = sig.args
    if aty != bty:  # pragma: no cover
        return cgutils.false_bit

    def index_is_impl(a, b):  # pragma: no cover
        return (
            a._start == b._start
            and a._stop == b._stop
            and a._step == b._step
            and a._name is b._name
        )

    return context.compile_internal(builder, index_is_impl, sig, args)


# TODO(ehsan): binary operators should be handled and tested for all Index types,
# properly (this is just to enable common cases in the short term). See #1415
####################### binary operators ###############################


def create_binary_op_overload(op):
    def overload_index_binary_op(S, other):

        # left arg is Index
        if is_index_type(S):

            def impl(S, other):  # pragma: no cover
                arr = bodo.utils.conversion.coerce_to_array(S)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(other)
                out_arr = op(arr, other_arr)
                return out_arr

            return impl

        # right arg is Index
        if is_index_type(other):

            def impl2(S, other):  # pragma: no cover
                arr = bodo.utils.conversion.coerce_to_array(other)
                other_arr = bodo.utils.conversion.get_array_if_series_or_index(S)
                out_arr = op(other_arr, arr)
                return out_arr

            return impl2

        if isinstance(S, HeterogeneousIndexType):
            # handle as regular array data if not actually heterogeneous
            if not is_heterogeneous_tuple_type(S.data):

                def impl3(S, other):  # pragma: no cover
                    data = bodo.utils.conversion.coerce_to_array(S)
                    arr = bodo.utils.conversion.coerce_to_array(data)
                    other_arr = bodo.utils.conversion.get_array_if_series_or_index(
                        other
                    )
                    out_arr = op(arr, other_arr)
                    return out_arr

                return impl3

            count = len(S.data.types)
            # TODO(ehsan): return Numpy array (fix Numba errors)
            func_text = "def f(S, other):\n"
            func_text += "  return [{}]\n".format(
                ",".join(
                    "op(S[{}], other{})".format(
                        i, f"[{i}]" if is_iterable_type(other) else ""
                    )
                    for i in range(count)
                ),
            )
            loc_vars = {}
            exec(func_text, {"op": op, "np": np}, loc_vars)
            impl = loc_vars["f"]
            return impl

        if isinstance(other, HeterogeneousIndexType):
            # handle as regular array data if not actually heterogeneous
            if not is_heterogeneous_tuple_type(other.data):

                def impl4(S, other):  # pragma: no cover
                    data = bodo.hiframes.pd_index_ext.get_index_data(other)
                    arr = bodo.utils.conversion.coerce_to_array(data)
                    other_arr = bodo.utils.conversion.get_array_if_series_or_index(S)
                    out_arr = op(other_arr, arr)
                    return out_arr

                return impl4

            count = len(other.data.types)
            # TODO(ehsan): return Numpy array (fix Numba errors)
            func_text = "def f(S, other):\n"
            func_text += "  return [{}]\n".format(
                ",".join(
                    "op(S{}, other[{}])".format(
                        f"[{i}]" if is_iterable_type(S) else "", i
                    )
                    for i in range(count)
                ),
            )
            loc_vars = {}
            exec(func_text, {"op": op, "np": np}, loc_vars)
            impl = loc_vars["f"]
            return impl

    return overload_index_binary_op


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        overload_impl = create_binary_op_overload(op)
        overload(op, inline="always", no_unliteral=True)(overload_impl)


_install_binary_ops()


# TODO(Nick): Consolidate this with is_pd_index_type?
# They only differ by HeterogeneousIndexType
def is_index_type(t):
    """return True if 't' is an Index type"""
    return isinstance(
        t,
        (
            RangeIndexType,
            NumericIndexType,
            StringIndexType,
            PeriodIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
        ),
    )


@lower_cast(RangeIndexType, NumericIndexType)
def cast_range_index_to_int_index(context, builder, fromty, toty, val):
    """cast RangeIndex to equivalent Int64Index"""
    f = lambda I: init_numeric_index(
        np.arange(I._start, I._stop, I._step),
        bodo.hiframes.pd_index_ext.get_index_name(I),
    )  # pragma: no cover
    return context.compile_internal(builder, f, toty(fromty), [val])


class HeterogeneousIndexType(types.Type):
    """
    Type class for Index objects with potentially heterogeneous but limited number of
    values (e.g. pd.Index([1, 'A']))
    """

    ndim = 1

    def __init__(self, data=None, name_type=None):

        self.data = data
        name_type = types.none if name_type is None else name_type
        self.name_type = name_type
        super(HeterogeneousIndexType, self).__init__(
            name=f"heter_index({data}, {name_type})"
        )

    def copy(self):
        return HeterogeneousIndexType(self.data, self.name_type)

    @property
    def key(self):
        return self.data, self.name_type


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(HeterogeneousIndexType)
class HeterogeneousIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", fe_type.data), ("name", fe_type.name_type)]
        super(HeterogeneousIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(HeterogeneousIndexType, "data", "_data")
make_attribute_wrapper(HeterogeneousIndexType, "name", "_name")


@overload_method(HeterogeneousIndexType, "copy", no_unliteral=True)
def overload_heter_index_copy(A):
    # NOTE: assuming data is immutable
    return lambda A: bodo.hiframes.pd_index_ext.init_heter_index(
        A._data, A._name
    )  # pragma: no cover


# TODO(ehsan): test
@box(HeterogeneousIndexType)
def box_heter_index(typ, val, c):  # pragma: no cover
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    c.context.nrt.incref(c.builder, typ.data, index_val.data)
    data_obj = c.pyapi.from_native_value(typ.data, index_val.data, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_type, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_type, index_val.name, c.env_manager)

    dtype_obj = c.pyapi.make_none()
    copy_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

    # call pd.Index(data, dtype, copy, name)
    index_obj = c.pyapi.call_method(
        class_obj, "Index", (data_obj, dtype_obj, copy_obj, name_obj)
    )

    c.pyapi.decref(data_obj)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(copy_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(class_obj)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@intrinsic
def init_heter_index(typingctx, data, name=None):
    """Create HeterogeneousIndex object"""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        assert len(args) == 2
        index_typ = signature.return_type
        index_val = cgutils.create_struct_proxy(index_typ)(context, builder)
        index_val.data = args[0]
        index_val.name = args[1]
        # increase refcount of stored values
        context.nrt.incref(builder, index_typ.data, args[0])
        context.nrt.incref(builder, index_typ.name_type, args[1])
        return index_val._getvalue()

    return HeterogeneousIndexType(data, name)(data, name), codegen


@overload_attribute(HeterogeneousIndexType, "name")
def heter_index_get_name(si):
    def impl(si):  # pragma: no cover
        return si._name

    return impl


# TODO(ehsan): test
@overload(operator.getitem, no_unliteral=True)
def overload_heter_index_getitem(I, ind):  # pragma: no cover
    if not isinstance(I, HeterogeneousIndexType):
        return

    # output of integer indexing is scalar value
    if isinstance(ind, types.Integer):
        return lambda I, ind: bodo.hiframes.pd_index_ext.get_index_data(I)[
            ind
        ]  # pragma: no cover

    # output of slice, bool array ... indexing is pd.Index
    if isinstance(I, HeterogeneousIndexType):
        return lambda I, ind: bodo.hiframes.pd_index_ext.init_heter_index(
            bodo.hiframes.pd_index_ext.get_index_data(I)[ind],
            bodo.hiframes.pd_index_ext.get_index_name(I),
        )  # pragma: no cover


@lower_constant(DatetimeIndexType)
@lower_constant(TimedeltaIndexType)
def lower_constant_time_index(context, builder, ty, pyval):
    """ Constant lowering for DatetimeIndexType and TimedeltaIndexType. """
    data = context.get_constant_generic(
        builder, types.Array(types.int64, 1, "C"), pyval.values.view(np.int64)
    )
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    dt_val = cgutils.create_struct_proxy(ty)(context, builder)
    dt_val.data = data
    dt_val.name = name

    # create empty dict for get_loc hashmap
    dtype = ty.dtype
    dt_val.dict = context.compile_internal(
        builder,
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )  # pragma: no cover

    return dt_val._getvalue()


@lower_constant(PeriodIndexType)
def lower_constant_period_index(context, builder, ty, pyval):
    """ Constant lowering for PeriodIndexType. """
    data = context.get_constant_generic(
        builder, types.Array(types.int64, 1, "C"), pyval.asi8
    )
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    index_val = cgutils.create_struct_proxy(ty)(context, builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    index_val.dict = context.compile_internal(
        builder,
        lambda: numba.typed.Dict.empty(types.int64, types.int64),
        types.DictType(types.int64, types.int64)(),
        [],
    )  # pragma: no cover

    return index_val._getvalue()


@lower_constant(NumericIndexType)
def lower_constant_numeric_index(context, builder, ty, pyval):
    """ Constant lowering for NumericIndexType. """

    # make sure the type is one of the numeric ones
    assert ty.dtype in (types.int64, types.uint64, types.float64)

    # get the data
    data = context.get_constant_generic(
        builder, types.Array(ty.dtype, 1, "C"), pyval.values
    )
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    index_val = cgutils.create_struct_proxy(ty)(context, builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    dtype = ty.dtype
    index_val.dict = context.compile_internal(
        builder,
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )  # pragma: no cover

    return index_val._getvalue()


@lower_constant(StringIndexType)
def lower_constant_string_index(context, builder, ty, pyval):
    """ Constant lowering for StringIndexType. """
    data = context.get_constant_generic(builder, string_array_type, pyval.values)
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    index_val = cgutils.create_struct_proxy(ty)(context, builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    index_val.dict = context.compile_internal(
        builder,
        lambda: numba.typed.Dict.empty(string_type, types.int64),
        types.DictType(string_type, types.int64)(),
        [],
    )  # pragma: no cover

    return index_val._getvalue()


@lower_builtin("getiter", RangeIndexType)
def getiter_range_index(context, builder, sig, args):
    """
    Support for getiter with Index types. Influenced largely by
    numba.np.arrayobj.getiter_array:
    https://github.com/numba/numba/blob/dbc71b78c0686314575a516db04ab3856852e0f5/numba/np/arrayobj.py#L256
    and numba.cpython.range_obj.RangeIter.from_range_state:
    https://github.com/numba/numba/blob/dbc71b78c0686314575a516db04ab3856852e0f5/numba/cpython/rangeobj.py#L107
    """
    [indexty] = sig.args
    [index] = args
    indexobj = context.make_helper(builder, indexty, value=index)

    iterobj = context.make_helper(builder, sig.return_type)

    iterptr = cgutils.alloca_once_value(builder, indexobj.start)

    zero = context.get_constant(types.intp, 0)
    countptr = cgutils.alloca_once_value(builder, zero)

    iterobj.iter = iterptr
    iterobj.stop = indexobj.stop
    iterobj.step = indexobj.step
    iterobj.count = countptr

    diff = builder.sub(indexobj.stop, indexobj.start)
    one = context.get_constant(types.intp, 1)
    pos_diff = builder.icmp(lc.ICMP_SGT, diff, zero)
    pos_step = builder.icmp(lc.ICMP_SGT, indexobj.step, zero)
    sign_same = builder.not_(builder.xor(pos_diff, pos_step))

    with builder.if_then(sign_same):
        rem = builder.srem(diff, indexobj.step)
        rem = builder.select(pos_diff, rem, builder.neg(rem))
        uneven = builder.icmp(lc.ICMP_SGT, rem, zero)
        newcount = builder.add(
            builder.sdiv(diff, indexobj.step), builder.select(uneven, one, zero)
        )
        builder.store(newcount, countptr)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    out = impl_ret_new_ref(context, builder, sig.return_type, res)
    return out


def getiter_index(context, builder, sig, args):
    """
    Support for getiter with Index types. Extracts the stored array and
    calls numba.np.arrayobj.getiter_array.
    """
    [indexty] = sig.args
    [index] = args
    indexobj = context.make_helper(builder, indexty, value=index)
    return numba.np.arrayobj.getiter_array(
        context, builder, signature(sig.return_type, sig.args[0].data), (indexobj.data,)
    )


def _install_index_getiter():
    """install an overload that raises BodoError for unsupported methods of pd.Index"""
    index_types = [
        NumericIndexType,
        StringIndexType,
        TimedeltaIndexType,
        DatetimeIndexType,
    ]

    for typ in index_types:
        lower_builtin("getiter", typ)(getiter_index)


_install_index_getiter()


index_unsupported = [
    "all",
    "any",
    "append",
    "argmax",
    "argmin",
    "argsort",
    "asof",
    "asof_locs",
    "astype",
    "delete",
    "difference",
    "drop",
    "drop_duplicates",
    "droplevel",
    "dropna",
    "duplicated",
    "equals",
    "factorize",
    "fillna",
    "format",
    "get_indexer",
    "get_indexer_for",
    "get_indexer_non_unique",
    "get_level_values",
    "get_slice_bound",
    "get_value",
    "groupby",
    "holds_integer",
    "identical",
    "insert",
    "intersection",
    "is_",
    "is_boolean",
    "is_categorical",
    "is_floating",
    "is_integer",
    "is_interval",
    "is_mixed",
    "is_numeric",
    "is_object",
    "is_type_compatible",
    "isin",
    "item",
    "join",
    "memory_usage",
    "notna",
    "notnull",
    "nunique",
    "putmask",
    "ravel",
    "reindex",
    "rename",
    "repeat",
    "searchsorted",
    "set_names",
    "set_value",
    "shift",
    "slice_indexer",
    "slice_locs",
    "sort",
    "sort_values",
    "sortlevel",
    "str",
    "symmetric_difference",
    "to_flat_index",
    "to_frame",
    "to_list",
    "to_native_types",
    "to_numpy",
    "to_series",
    "tolist",
    "transpose",
    "union",
    "unique",
    "value_counts",
    "view",
    "where",
]


def _install_index_unsupported():
    """install an overload that raises BodoError for unsupported methods of pd.Index"""
    index_types = [
        ("NumericIndexType.", NumericIndexType),
        ("StringIndexType.", StringIndexType),
        ("TimedeltaIndexType.", TimedeltaIndexType),
        ("PeriodIndexType.", PeriodIndexType),
        ("DatetimeIndexType.", DatetimeIndexType),
    ]

    for fname in index_unsupported:
        for t_name, typ in index_types:
            overload_method(typ, fname, no_unliteral=True)(
                create_unsupported_overload(t_name + fname)
            )


_install_index_unsupported()
