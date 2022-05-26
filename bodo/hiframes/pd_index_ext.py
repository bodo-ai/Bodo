# Copyright (C) 2019 Bodo Inc. All rights reserved.
import datetime
import operator
import warnings

import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
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
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import pd_timestamp_type
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    create_unsupported_overload,
    dtype_to_array_type,
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
    is_str_arr_type,
    parse_dtype,
    raise_bodo_error,
)
from bodo.utils.utils import is_null_value

_dt_index_data_typ = types.Array(types.NPDatetime("ns"), 1, "C")
_timedelta_index_data_typ = types.Array(types.NPTimedelta("ns"), 1, "C")
iNaT = pd._libs.tslibs.iNaT
NaT = types.NPDatetime("ns")("NaT")  # TODO: pd.NaT

# used in the various index copy overloads for error checking
idx_cpy_arg_defaults = dict(deep=False, dtype=None, names=None)

# maps index_types to a format string of how we refer to the index type in error messages.
# for example:
# RangeIndexType --> "pandas.RangeIndex.{}"
# StringIndexType --> "pandas.Index.{} with string data"

# Initialized at the bottom of this file, after all the index types have been declared
idx_typ_to_format_str_map = dict()


@typeof_impl.register(pd.Index)
def typeof_pd_index(val, c):
    if val.inferred_type == "string" or pd._libs.lib.infer_dtype(val, True) == "string":
        # Index.inferred_type doesn't skip NAs so we call infer_dtype with
        # skipna=True
        return StringIndexType(get_val_type_maybe_str_literal(val.name))

    if val.inferred_type == "bytes" or pd._libs.lib.infer_dtype(val, True) == "bytes":
        # Index.inferred_type doesn't skip NAs so we call infer_dtype with
        # skipna=True
        return BinaryIndexType(get_val_type_maybe_str_literal(val.name))

    # XXX: assume string data type for empty Index with object dtype
    if val.equals(pd.Index([])):
        return StringIndexType(get_val_type_maybe_str_literal(val.name))

    # TODO: Replace with a specific type for DateIndex, so these can be boxed
    if val.inferred_type == "date":
        return DatetimeIndexType(get_val_type_maybe_str_literal(val.name))

    # Pandas uses object dtype for nullable int arrays
    if (
        val.inferred_type == "integer"
        or pd._libs.lib.infer_dtype(val, True) == "integer"
    ):
        # At least some index values contain the actual dtype in
        # Pandas 1.4.
        if isinstance(val.dtype, pd.core.arrays.integer._IntegerDtype):
            # Get the numpy dtype
            numpy_dtype = val.dtype.numpy_dtype
            # Convert the numpy dtype to the Numba type
            dtype = numba.np.numpy_support.from_dtype(numpy_dtype)
        else:
            # we don't have the dtype default to int64
            dtype = types.int64
        return NumericIndexType(
            dtype,
            get_val_type_maybe_str_literal(val.name),
            IntegerArrayType(dtype),
        )
    if (
        val.inferred_type == "boolean"
        or pd._libs.lib.infer_dtype(val, True) == "boolean"
    ):
        return NumericIndexType(
            types.bool_,
            get_val_type_maybe_str_literal(val.name),
            boolean_array,
        )
    # catch-all for non-supported Index types
    # RangeIndex is directly supported (TODO: make sure this is not called)
    raise NotImplementedError(f"unsupported pd.Index type {val}")


# -------------------------  DatetimeIndex ------------------------------


class DatetimeIndexType(types.IterableType, types.ArrayCompatible):
    """type class for DatetimeIndex objects."""

    def __init__(self, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like freq/dtype/yearfirst?
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = types.Array(bodo.datetime64ns, 1, "C") if data is None else data
        super(DatetimeIndexType, self).__init__(
            name=f"DatetimeIndex({name_typ}, {self.data})"
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def tzval(self):
        return self.data.tz if isinstance(self.data, bodo.DatetimeArrayType) else None

    def copy(self):
        return DatetimeIndexType(self.name_typ, self.data)

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)

    @property
    def pandas_type_name(self):
        return self.data.dtype.type_name

    @property
    def numpy_type_name(self):
        return str(self.data.dtype)


types.datetime_index = DatetimeIndexType()


@typeof_impl.register(pd.DatetimeIndex)
def typeof_datetime_index(val, c):
    # TODO: check value for freq, etc. and raise error since unsupported
    if isinstance(val.dtype, pd.DatetimeTZDtype):
        return DatetimeIndexType(
            get_val_type_maybe_str_literal(val.name), DatetimeArrayType(val.tz)
        )
    return DatetimeIndexType(get_val_type_maybe_str_literal(val.name))


@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: use payload to support mutable name
        members = [
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(_dt_index_data_typ.dtype, types.int64)),
        ]
        super(DatetimeIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DatetimeIndexType, "data", "_data")
make_attribute_wrapper(DatetimeIndexType, "name", "_name")
make_attribute_wrapper(DatetimeIndexType, "dict", "_dict")


@overload_method(DatetimeIndexType, "copy", no_unliteral=True)
def overload_datetime_index_copy(A, name=None, deep=False, dtype=None, names=None):
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    err_str = idx_typ_to_format_str_map[DatetimeIndexType].format("copy()")
    check_unsupported_args(
        "copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_datetime_index(A._data.copy(), name)

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_datetime_index(
                A._data.copy(), A._name
            )

    return impl


@box(DatetimeIndexType)
def box_dt_index(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    dt_index = numba.core.cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    c.context.nrt.incref(c.builder, typ.data, dt_index.data)
    arr_obj = c.pyapi.from_native_value(typ.data, dt_index.data, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_typ, dt_index.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, dt_index.name, c.env_manager)

    # call pd.DatetimeIndex(arr, name=name)
    args = c.pyapi.tuple_pack([arr_obj])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "DatetimeIndex")
    kws = c.pyapi.dict_pack([("name", name_obj)])
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
    if isinstance(typ.data, DatetimeArrayType):
        data_obj = c.pyapi.object_getattr_string(val, "array")
    else:
        data_obj = c.pyapi.object_getattr_string(val, "values")
    data = c.pyapi.to_native_value(typ.data, data_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value

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

    c.pyapi.decref(data_obj)
    c.pyapi.decref(name_obj)

    return NativeValue(index_val._getvalue())


@intrinsic
def init_datetime_index(typingctx, data, name):
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

    ret_typ = DatetimeIndexType(name, data)
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
    func_text += "        val = A[i]\n"
    func_text += "        ts = bodo.utils.conversion.box_if_dt64(val)\n"
    if field in [
        "weekday",
    ]:
        func_text += "        S[i] = ts." + field + "()\n"
    else:
        func_text += "        S[i] = ts." + field + "\n"
    func_text += "    return bodo.hiframes.pd_index_ext.init_numeric_index(S, name)\n"
    loc_vars = {}
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
            val = A[i]
            ts = bodo.utils.conversion.box_if_dt64(val)
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
            val = A[i]
            ts = bodo.utils.conversion.box_if_dt64(val)
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
    check_unsupported_args(
        "DatetimeIndex.min",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(dti, "Index.min()")

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
    check_unsupported_args(
        "DatetimeIndex.max",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(dti, "Index.max()")

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


@overload_method(DatetimeIndexType, "tz_convert", no_unliteral=True)
def overload_pd_datetime_tz_convert(A, tz):
    def impl(A, tz):
        return init_datetime_index(A._data.tz_convert(tz), A._name)

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
    tz=None,
    normalize=False,
    closed=None,
    ambiguous="raise",
    dayfirst=False,
    yearfirst=False,
    dtype=None,
    copy=False,
    name=None,
):
    # TODO: check/handle other input
    if is_overload_none(data):
        raise BodoError("data argument in pd.DatetimeIndex() expected")

    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        data, "pandas.DatetimeIndex()"
    )

    unsupported_args = dict(
        freq=freq,
        tz=tz,
        normalize=normalize,
        closed=closed,
        ambiguous=ambiguous,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        dtype=dtype,
        copy=copy,
    )
    arg_defaults = dict(
        freq=None,
        tz=None,
        normalize=False,
        closed=None,
        ambiguous="raise",
        dayfirst=False,
        yearfirst=False,
        dtype=None,
        copy=False,
    )
    check_unsupported_args(
        "pandas.DatetimeIndex",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    def f(
        data=None,
        freq=None,
        tz=None,
        normalize=False,
        closed=None,
        ambiguous="raise",
        dayfirst=False,
        yearfirst=False,
        dtype=None,
        copy=False,
        name=None,
    ):  # pragma: no cover
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        S = bodo.utils.conversion.convert_to_dt64ns(data_arr)
        return bodo.hiframes.pd_index_ext.init_datetime_index(S, name)

    return f


def overload_sub_operator_datetime_index(lhs, rhs):
    # DatetimeIndex - Timestamp
    if (
        isinstance(lhs, DatetimeIndexType)
        and rhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type
    ):
        timedelta64_dtype = np.dtype("timedelta64[ns]")

        def impl(lhs, rhs):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            in_arr = bodo.hiframes.pd_index_ext.get_index_data(lhs)
            name = bodo.hiframes.pd_index_ext.get_index_name(lhs)
            n = len(in_arr)
            S = np.empty(n, timedelta64_dtype)
            tsint = rhs.value
            for i in numba.parfors.parfor.internal_prange(n):
                S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                    bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i]) - tsint
                )
            return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

        return impl

    # Timestamp - DatetimeIndex
    if (
        isinstance(rhs, DatetimeIndexType)
        and lhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_type
    ):
        timedelta64_dtype = np.dtype("timedelta64[ns]")

        def impl(lhs, rhs):  # pragma: no cover
            numba.parfors.parfor.init_prange()
            in_arr = bodo.hiframes.pd_index_ext.get_index_data(rhs)
            name = bodo.hiframes.pd_index_ext.get_index_name(rhs)
            n = len(in_arr)
            S = np.empty(n, timedelta64_dtype)
            tsint = lhs.value
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

    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(data, "pandas.Index()")

    # Todo: support Categorical dtype, Interval dtype, Period dtype, MultiIndex (?)
    # Todo: Extension dtype (?)

    # unliteral e.g. Tuple(Literal[int](3), Literal[int](1)) to UniTuple(int64 x 2)
    # NOTE: unliteral of LiteralList is Poison type in Numba
    data = types.unliteral(data) if not isinstance(data, types.LiteralList) else data

    data_dtype = getattr(data, "dtype", None)
    if not is_overload_none(dtype):
        elem_type = parse_dtype(dtype, "pandas.Index")
    else:
        elem_type = data_dtype

    # Add a special error message for object dtypes
    if isinstance(elem_type, types.misc.PyObject):
        raise BodoError(
            "pd.Index() object 'dtype' is not specific enough for typing. Please provide a more exact type (e.g. str)."
        )

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
        if isinstance(elem_type, (types.Integer, types.Float, types.Boolean)):

            def impl(
                data=None, dtype=None, copy=False, name=None, tupleize_cols=True
            ):  # pragma: no cover
                data_arr = bodo.utils.conversion.coerce_to_array(data)
                data_coerced = bodo.utils.conversion.fix_arr_dtype(data_arr, elem_type)
                return bodo.hiframes.pd_index_ext.init_numeric_index(data_coerced, name)

        # String/Binary index:
        elif elem_type in [types.string, bytes_type]:

            def impl(
                data=None, dtype=None, copy=False, name=None, tupleize_cols=True
            ):  # pragma: no cover
                return bodo.hiframes.pd_index_ext.init_binary_str_index(
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
                val = dti_arr[ind]
                return bodo.utils.conversion.box_if_dt64(val)

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


@overload(operator.getitem, no_unliteral=True)
def overload_categorical_index_getitem(I, ind):
    """getitem overload for CategoricalIndex"""
    if not isinstance(I, CategoricalIndexType):
        return

    if isinstance(ind, types.Integer):

        def impl(I, ind):  # pragma: no cover
            cat_arr = bodo.hiframes.pd_index_ext.get_index_data(I)
            val = cat_arr[ind]
            return val

        return impl

    if isinstance(ind, types.SliceType):

        def impl(I, ind):  # pragma: no cover
            cat_arr = bodo.hiframes.pd_index_ext.get_index_data(I)
            name = bodo.hiframes.pd_index_ext.get_index_name(I)
            new_arr = cat_arr[ind]
            return bodo.hiframes.pd_index_ext.init_categorical_index(new_arr, name)

        return impl

    raise BodoError(f"pd.CategoricalIndex.__getitem__: unsupported index type {ind}")


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
    # Handle optional types
    if isinstance(val, types.Optional):

        def impl(val):  # pragma: no cover
            if val is None:
                return 0
            return bodo.utils.indexing.unoptional(val)

        return impl

    return lambda val: val  # pragma: no cover


@overload(pd.date_range, inline="always")
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

    unsupported_args = dict(tz=tz, normalize=normalize, closed=closed)
    arg_defaults = dict(tz=None, normalize=False, closed=None)
    check_unsupported_args(
        "pandas.date_range",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="General",
    )

    if not is_overload_none(tz):
        raise_bodo_error("pd.date_range(): tz argument not supported yet")

    freq_set = ""
    if is_overload_none(freq) and any(
        is_overload_none(t) for t in (start, end, periods)
    ):
        freq = "D"  # change just to enable checks below
        freq_set = "  freq = 'D'\n"

    # exactly three parameters should be provided
    if sum(not is_overload_none(t) for t in (start, end, periods, freq)) != 3:
        raise_bodo_error(
            "Of the four parameters: start, end, periods, "
            "and freq, exactly three must be specified"
        )

    # TODO [BE-2499]: enable check when closed is supported
    # closed requires one of start and end to be not None
    # if is_overload_none(start) and is_overload_none(end) and not is_overload_none(closed):
    #     raise_bodo_error(
    #         "Closed has to be None if not both of start and end are defined"
    #     )

    # TODO: check start and end for NaT

    func_text = "def f(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, closed=None):\n"

    func_text += freq_set

    if is_overload_none(start):
        # dummy value for typing
        func_text += "  start_t = pd.Timestamp('1800-01-03')\n"
    else:
        func_text += "  start_t = pd.Timestamp(start)\n"

    if is_overload_none(end):
        # dummy value for typing
        func_text += "  end_t = pd.Timestamp('1800-01-03')\n"
    else:
        func_text += "  end_t = pd.Timestamp(end)\n"

    # freq provided
    if not is_overload_none(freq):
        func_text += "  stride = bodo.hiframes.pd_index_ext.to_offset_value(freq)\n"
        if is_overload_none(periods):
            func_text += "  b = start_t.value\n"
            func_text += (
                "  e = b + (end_t.value - b) // stride * stride + stride // 2 + 1\n"
            )
        elif not is_overload_none(start):
            func_text += "  b = start_t.value\n"
            func_text += "  addend = np.int64(periods) * np.int64(stride)\n"
            func_text += "  e = np.int64(b) + addend\n"
        elif not is_overload_none(end):
            func_text += "  e = end_t.value + stride\n"
            func_text += "  addend = np.int64(periods) * np.int64(-stride)\n"
            func_text += "  b = np.int64(e) + addend\n"
        else:
            raise_bodo_error(
                "at least 'start' or 'end' should be specified "
                "if a 'period' is given."
            )
        # TODO: handle overflows
        func_text += "  arr = np.arange(b, e, stride, np.int64)\n"
    # freq is None
    else:
        # TODO: fix Numba's linspace to support dtype
        # arr = np.linspace(
        #     0, end_t.value - start_t.value,
        #     periods, dtype=np.int64) + start.value

        # using Numpy's linspace algorithm
        func_text += "  delta = end_t.value - start_t.value\n"
        func_text += "  step = delta / (periods - 1)\n"
        func_text += "  arr1 = np.arange(0, periods, 1, np.float64)\n"
        func_text += "  arr1 *= step\n"
        func_text += "  arr1 += start_t.value\n"
        func_text += "  arr = arr1.astype(np.int64)\n"
        func_text += "  arr[-1] = end_t.value\n"

    # TODO [BE-2499]: support closed when distributed pass can handle this
    # func_text += "  left_closed, right_closed = bodo.hiframes.pd_index_ext.validate_endpoints(closed)\n"
    # func_text += "  if not left_closed and len(arr) and arr[0] == start_t.value:\n"
    # func_text += "    arr = arr[1:]\n"
    # func_text += "  if not right_closed and len(arr) and arr[-1] == end_t.value:\n"
    # func_text += "    arr = arr[:-1]\n"

    func_text += "  A = bodo.utils.conversion.convert_to_dt64ns(arr)\n"
    func_text += "  return bodo.hiframes.pd_index_ext.init_datetime_index(A, name)\n"

    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np, "pd": pd}, loc_vars)
    f = loc_vars["f"]

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
            stride = _dummy_convert_none_to_int(freq)
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
            ) = bodo.utils.conversion.box_if_dt64(A[i]).isocalendar()
        return bodo.hiframes.pd_dataframe_ext.init_dataframe(
            (years, weeks, days), idx, ("year", "week", "day")
        )

    return impl


# ------------------------------ Timedelta ---------------------------


# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType, types.ArrayCompatible):
    """Temporary type class for TimedeltaIndex objects."""

    def __init__(self, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like unit/freq?
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        # NOTE: data array can have flags like readonly
        self.data = types.Array(bodo.timedelta64ns, 1, "C") if data is None else data
        super(TimedeltaIndexType, self).__init__(
            name=f"TimedeltaIndexType({name_typ}, {self.data})"
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
        return self.name_typ, self.data

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)

    @property
    def pandas_type_name(self):
        return "timedelta"

    @property
    def numpy_type_name(self):
        return "timedelta64[ns]"


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
    #     return signature(pd_timestamp_type, *args)

    # @bound_function("timedelta_index.min", no_unliteral=True)
    # def resolve_min(self, ary, args, kws):
    #     assert not kws
    #     return signature(pd_timestamp_type, *args)


make_attribute_wrapper(TimedeltaIndexType, "data", "_data")
make_attribute_wrapper(TimedeltaIndexType, "name", "_name")
make_attribute_wrapper(TimedeltaIndexType, "dict", "_dict")


@overload_method(TimedeltaIndexType, "copy", no_unliteral=True)
def overload_timedelta_index_copy(A, name=None, deep=False, dtype=None, names=None):
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    err_str = idx_typ_to_format_str_map[TimedeltaIndexType].format("copy()")
    check_unsupported_args(
        "TimedeltaIndex.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_timedelta_index(A._data.copy(), name)

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_timedelta_index(
                A._data.copy(), A._name
            )

    return impl


@overload_method(TimedeltaIndexType, "min", inline="always", no_unliteral=True)
def overload_timedelta_index_min(tdi, axis=None, skipna=True):
    unsupported_args = dict(axis=axis, skipna=skipna)
    arg_defaults = dict(axis=None, skipna=True)
    check_unsupported_args(
        "TimedeltaIndex.min",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

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
    check_unsupported_args(
        "TimedeltaIndex.max",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

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
    dtype=None,
    copy=False,
    name=None,
):
    # TODO handle dtype=dtype('<m8[ns]') default
    # TODO: check/handle other input
    if is_overload_none(data):
        raise BodoError("data argument in pd.TimedeltaIndex() expected")

    unsupported_args = dict(
        unit=unit,
        freq=freq,
        dtype=dtype,
        copy=copy,
    )

    arg_defaults = dict(
        unit=None,
        freq=None,
        dtype=None,
        copy=False,
    )

    check_unsupported_args(
        "pandas.TimedeltaIndex",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    def impl(
        data=None,
        unit=None,
        freq=None,
        dtype=None,
        copy=False,
        name=None,
    ):  # pragma: no cover
        data_arr = bodo.utils.conversion.coerce_to_array(data)
        S = bodo.utils.conversion.convert_to_td64ns(data_arr)
        return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

    return impl


# ---------------- RangeIndex -------------------


# pd.RangeIndex(): simply keep start/stop/step/name
class RangeIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.RangeIndex() objects."""

    def __init__(self, name_typ=None):
        if name_typ is None:
            name_typ = types.none
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
def overload_range_index_copy(A, name=None, deep=False, dtype=None, names=None):
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    err_str = idx_typ_to_format_str_map[RangeIndexType].format("copy()")
    check_unsupported_args(
        "RangeIndex.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_range_index(
                A._start, A._stop, A._step, name
            )

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_range_index(
                A._start, A._stop, A._step, A._name
            )

    return impl


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
    return lir.Constant.literal_struct([start, stop, step, name])


@overload(pd.RangeIndex, no_unliteral=True, inline="always")
def range_index_overload(
    start=None,
    stop=None,
    step=None,
    dtype=None,
    copy=False,
    name=None,
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
            raise BodoError(msg.format(value=value, field=field))

    _ensure_int_or_none(start, "start")
    _ensure_int_or_none(stop, "stop")
    _ensure_int_or_none(step, "step")

    # all none error case
    if is_overload_none(start) and is_overload_none(stop) and is_overload_none(step):
        msg = "RangeIndex(...) must be called with integers"
        raise BodoError(msg)

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

    func_text = "def _pd_range_index_imp(start=None, stop=None, step=None, dtype=None, copy=False, name=None):\n"
    func_text += "  return init_range_index({}, {}, {}, name)\n".format(
        _start, _stop, _step
    )
    loc_vars = {}
    exec(func_text, {"init_range_index": init_range_index}, loc_vars)
    _pd_range_index_imp = loc_vars["_pd_range_index_imp"]
    return _pd_range_index_imp


@overload(pd.CategoricalIndex, no_unliteral=True, inline="always")
def categorical_index_overload(
    data=None, categories=None, ordered=None, dtype=None, copy=False, name=None
):
    raise BodoError("pd.CategoricalIndex() initializer not yet supported.")


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
class PeriodIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.PeriodIndex. Contains frequency as constant string"""

    def __init__(self, freq, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.freq = freq
        self.name_typ = name_typ
        super(PeriodIndexType, self).__init__(
            name="PeriodIndexType({}, {})".format(freq, name_typ)
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return PeriodIndexType(self.freq, self.name_typ)

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)

    @property
    def pandas_type_name(self):
        return "object"

    @property
    def numpy_type_name(self):
        return f"period[{self.freq}]"


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
            ("data", bodo.IntegerArrayType(types.int64)),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(types.int64, types.int64)),
        ]
        super(PeriodIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(PeriodIndexType, "data", "_data")
make_attribute_wrapper(PeriodIndexType, "name", "_name")
make_attribute_wrapper(PeriodIndexType, "dict", "_dict")


@overload_method(PeriodIndexType, "copy", no_unliteral=True)
def overload_period_index_copy(A, name=None, deep=False, dtype=None, names=None):
    freq = A.freq
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    err_str = idx_typ_to_format_str_map[PeriodIndexType].format("copy()")
    check_unsupported_args(
        "PeriodIndex.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_period_index(
                A._data.copy(), name, freq
            )

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_period_index(
                A._data.copy(), A._name, freq
            )

    return impl


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


@box(PeriodIndexType)
def box_period_index(typ, val, c):
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    c.context.nrt.incref(c.builder, bodo.IntegerArrayType(types.int64), index_val.data)
    data_obj = c.pyapi.from_native_value(
        bodo.IntegerArrayType(types.int64), index_val.data, c.env_manager
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
    arr_typ = bodo.IntegerArrayType(types.int64)
    asi8_obj = c.pyapi.object_getattr_string(val, "asi8")
    isna_obj = c.pyapi.call_method(val, "isna", ())
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)
    arr_mod_obj = c.pyapi.object_getattr_string(pd_class_obj, "arrays")
    data_obj = c.pyapi.call_method(arr_mod_obj, "IntegerArray", (asi8_obj, isna_obj))
    data = c.pyapi.to_native_value(arr_typ, data_obj).value

    c.pyapi.decref(asi8_obj)
    c.pyapi.decref(isna_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(arr_mod_obj)
    c.pyapi.decref(data_obj)

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


# ------------------------------ CategoricalIndex ---------------------------


class CategoricalIndexType(types.IterableType, types.ArrayCompatible):
    """data type for CategoricalIndex values"""

    def __init__(self, data, name_typ=None):
        from bodo.hiframes.pd_categorical_ext import CategoricalArrayType

        assert isinstance(
            data, CategoricalArrayType
        ), "CategoricalIndexType expects CategoricalArrayType"
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        self.data = data
        super(CategoricalIndexType, self).__init__(
            name=f"CategoricalIndexType(data={self.data}, name={name_typ})"
        )

    ndim = 1

    def copy(self):
        return CategoricalIndexType(self.data, self.name_typ)

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def key(self):
        return self.data, self.name_typ

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    @property
    def pandas_type_name(self):
        return "categorical"

    @property
    def numpy_type_name(self):
        from bodo.hiframes.pd_categorical_ext import get_categories_int_type

        return str(get_categories_int_type(self.dtype))

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self, self.dtype.elem_type)


@register_model(CategoricalIndexType)
class CategoricalIndexTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        from bodo.hiframes.pd_categorical_ext import get_categories_int_type

        code_int_type = get_categories_int_type(fe_type.data.dtype)
        members = [
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            # assuming category codes are key in dict
            (
                "dict",
                types.DictType(code_int_type, types.int64),
            ),
        ]
        super(CategoricalIndexTypeModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(pd.CategoricalIndex)
def typeof_categorical_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return CategoricalIndexType(
        bodo.typeof(val.values), get_val_type_maybe_str_literal(val.name)
    )


@box(CategoricalIndexType)
def box_categorical_index(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    categorical_index = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, val
    )

    # box CategoricalArray
    c.context.nrt.incref(c.builder, typ.data, categorical_index.data)
    arr_obj = c.pyapi.from_native_value(typ.data, categorical_index.data, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_typ, categorical_index.name)
    name_obj = c.pyapi.from_native_value(
        typ.name_typ, categorical_index.name, c.env_manager
    )

    # call pd.CategoricalIndex(arr, name=name)
    args = c.pyapi.tuple_pack([arr_obj])
    kws = c.pyapi.dict_pack([("name", name_obj)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "CategoricalIndex")
    res = c.pyapi.call(const_call, args, kws)

    c.pyapi.decref(arr_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(CategoricalIndexType)
def unbox_categorical_index(typ, val, c):
    from bodo.hiframes.pd_categorical_ext import get_categories_int_type

    # get data and name attributes
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
    dtype = get_categories_int_type(typ.data.dtype)
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@intrinsic
def init_categorical_index(typingctx, data, name=None):
    """Create a CategoricalIndex with provided data and name values."""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        from bodo.hiframes.pd_categorical_ext import get_categories_int_type

        data_val, name_val = args
        # create categorical_index struct and store values
        categorical_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        categorical_index.data = data_val
        categorical_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], name_val)

        # create empty dict for get_loc hashmap
        dtype = get_categories_int_type(signature.return_type.data.dtype)
        categorical_index.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(dtype, types.int64),
            types.DictType(dtype, types.int64)(),
            [],
        )  # pragma: no cover

        return categorical_index._getvalue()

    ret_typ = CategoricalIndexType(data, name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_categorical_index = (
    init_index_equiv
)

make_attribute_wrapper(CategoricalIndexType, "data", "_data")
make_attribute_wrapper(CategoricalIndexType, "name", "_name")
make_attribute_wrapper(CategoricalIndexType, "dict", "_dict")


@overload_method(CategoricalIndexType, "copy", no_unliteral=True)
def overload_categorical_index_copy(A, name=None, deep=False, dtype=None, names=None):
    err_str = idx_typ_to_format_str_map[CategoricalIndexType].format("copy()")
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    check_unsupported_args(
        "CategoricalIndex.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_categorical_index(
                A._data.copy(), name
            )

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_categorical_index(
                A._data.copy(), A._name
            )

    return impl


# ------------------------------ IntervalIndex ---------------------------


class IntervalIndexType(types.ArrayCompatible):
    """data type for IntervalIndex values"""

    def __init__(self, data, name_typ=None):
        from bodo.libs.interval_arr_ext import IntervalArrayType

        assert isinstance(
            data, IntervalArrayType
        ), "IntervalIndexType expects IntervalArrayType"
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        self.data = data
        super(IntervalIndexType, self).__init__(
            name=f"IntervalIndexType(data={self.data}, name={name_typ})"
        )

    ndim = 1

    def copy(self):
        return IntervalIndexType(self.data, self.name_typ)

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    @property
    def key(self):
        return self.data, self.name_typ

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    @property
    def pandas_type_name(self):
        return "object"

    @property
    def numpy_type_name(self):
        return f"interval[{self.data.arr_type.dtype}, right]"  # TODO: Support for left and both intervals


@register_model(IntervalIndexType)
class IntervalIndexTypeModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            # assuming a tuple of left/right values is key in dict
            (
                "dict",
                types.DictType(
                    types.UniTuple(fe_type.data.arr_type.dtype, 2), types.int64
                ),
            ),
            # TODO(ehsan): support closed (assuming "right" for now)
        ]
        super(IntervalIndexTypeModel, self).__init__(dmm, fe_type, members)


@typeof_impl.register(pd.IntervalIndex)
def typeof_interval_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return IntervalIndexType(
        bodo.typeof(val.values), get_val_type_maybe_str_literal(val.name)
    )


@box(IntervalIndexType)
def box_interval_index(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module_noblock(mod_name)

    interval_index = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, val
    )

    # box IntervalArray
    c.context.nrt.incref(c.builder, typ.data, interval_index.data)
    arr_obj = c.pyapi.from_native_value(typ.data, interval_index.data, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_typ, interval_index.name)
    name_obj = c.pyapi.from_native_value(
        typ.name_typ, interval_index.name, c.env_manager
    )

    # call pd.IntervalIndex(arr, name=name)
    args = c.pyapi.tuple_pack([arr_obj])
    kws = c.pyapi.dict_pack([("name", name_obj)])
    const_call = c.pyapi.object_getattr_string(pd_class_obj, "IntervalIndex")
    res = c.pyapi.call(const_call, args, kws)

    c.pyapi.decref(arr_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(pd_class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return res


@unbox(IntervalIndexType)
def unbox_interval_index(typ, val, c):
    # get data and name attributes
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
    dtype = types.UniTuple(typ.data.arr_type.dtype, 2)
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@intrinsic
def init_interval_index(typingctx, data, name=None):
    """Create a IntervalIndex with provided data and name values."""
    name = types.none if name is None else name

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create interval_index struct and store values
        interval_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        interval_index.data = data_val
        interval_index.name = name_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], data_val)
        context.nrt.incref(builder, signature.args[1], name_val)

        # create empty dict for get_loc hashmap
        dtype = types.UniTuple(data.arr_type.dtype, 2)
        interval_index.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(dtype, types.int64),
            types.DictType(dtype, types.int64)(),
            [],
        )  # pragma: no cover

        return interval_index._getvalue()

    ret_typ = IntervalIndexType(data, name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_interval_index = (
    init_index_equiv
)

make_attribute_wrapper(IntervalIndexType, "data", "_data")
make_attribute_wrapper(IntervalIndexType, "name", "_name")
make_attribute_wrapper(IntervalIndexType, "dict", "_dict")


# ---------------- NumericIndex -------------------


# represents numeric indices (excluding RangeIndex):
#   Int64Index, UInt64Index, Float64Index
class NumericIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.Int64Index/UInt64Index/Float64Index objects."""

    def __init__(self, dtype, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        self.dtype = dtype
        self.name_typ = name_typ
        data = dtype_to_array_type(dtype) if data is None else data
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
        return NumericIndexType(self.dtype, self.name_typ, self.data)

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)

    @property
    def pandas_type_name(self):
        return str(self.dtype)

    @property
    def numpy_type_name(self):
        return str(self.dtype)


# Pandas 1.4+ has deprecated pd.<type>Index in favor of pd.Index(dtype=<type>),
# but we still need to support older versions.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Int64Index = pd.Int64Index
    UInt64Index = pd.UInt64Index
    Float64Index = pd.Float64Index


@typeof_impl.register(Int64Index)
def typeof_pd_int64_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return NumericIndexType(types.int64, get_val_type_maybe_str_literal(val.name))


@typeof_impl.register(UInt64Index)
def typeof_pd_uint64_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return NumericIndexType(types.uint64, get_val_type_maybe_str_literal(val.name))


@typeof_impl.register(Float64Index)
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
def overload_numeric_index_copy(A, name=None, deep=False, dtype=None, names=None):
    err_str = idx_typ_to_format_str_map[NumericIndexType].format("copy()")
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    check_unsupported_args(
        "Index.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_numeric_index(A._data.copy(), name)

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_numeric_index(
                A._data.copy(), A._name
            )

    return impl


@box(NumericIndexType)
def box_numeric_index(typ, val, c):
    """Box NumericIndexType values by calling pd.Index(data).
    Bodo supports all numberic dtypes (e.g. int32) but Pandas is limited to
    Int64/UInt64/Float64. pd.Index() will convert to the available Index type.
    """
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    c.context.nrt.incref(c.builder, typ.data, index_val.data)
    data_obj = c.pyapi.from_native_value(typ.data, index_val.data, c.env_manager)
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)

    dtype_obj = c.pyapi.make_none()
    copy_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))

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


def create_numeric_constructor(func, func_str, default_dtype):
    def overload_impl(data=None, dtype=None, copy=False, name=None):
        # TODO: I'm not entirely certain what the dtype argument even does.
        # playing around with it in ipython, it seems to have no effect unless it
        # is not castable to the default type, in which case it throws an error.
        # for right now, I'm just going to say we don't support it.
        # TODO: read through the pandas source code
        unsupported_args_dict = dict(dtype=dtype)
        default_dict = dict(dtype=None)
        check_unsupported_args(
            func_str,
            unsupported_args_dict,
            default_dict,
            package_name="pandas",
            module_name="Index",
        )
        if is_overload_false(copy):
            # if copy is False for sure, specialize to avoid branch

            def impl(data=None, dtype=None, copy=False, name=None):  # pragma: no cover
                data_arr = bodo.utils.conversion.coerce_to_ndarray(data)
                data_res = bodo.utils.conversion.fix_arr_dtype(
                    data_arr, np.dtype(default_dtype)
                )
                return bodo.hiframes.pd_index_ext.init_numeric_index(data_res, name)

        else:

            def impl(
                data=None,
                dtype=None,
                copy=False,
                name=None,
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
    for func, func_str, default_dtype in (
        (Int64Index, "pandas.Int64Index", np.int64),
        (UInt64Index, "pandas.UInt64Index", np.uint64),
        (Float64Index, "pandas.Float64Index", np.float64),
    ):
        overload_impl = create_numeric_constructor(func, func_str, default_dtype)
        overload(func, no_unliteral=True)(overload_impl)


_install_numeric_constructors()


# ---------------- StringIndex -------------------


# represents string index, which doesn't have direct Pandas type
# pd.Index() infers string
class StringIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.Index() objects with 'string' as inferred_dtype."""

    def __init__(self, name_typ=None, data_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = string_array_type if data_typ is None else data_typ
        super(StringIndexType, self).__init__(
            name=f"StringIndexType({name_typ}, {self.data})",
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return StringIndexType(self.name_typ, self.data)

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
        return bodo.utils.typing.BodoArrayIterator(self)


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(StringIndexType)
class StringIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            # TODO(ehsan): optimize get_loc() handling for dict-encoded str array case
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(string_type, types.int64)),
        ]
        super(StringIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(StringIndexType, "data", "_data")
make_attribute_wrapper(StringIndexType, "name", "_name")
make_attribute_wrapper(StringIndexType, "dict", "_dict")


# ---------------- BinaryIndex -------------------


# represents binary index, which doesn't have direct Pandas type
# pd.Index() infers binary
# Largely copied from the StringIndexType class
class BinaryIndexType(types.IterableType, types.ArrayCompatible):
    """type class for pd.Index() objects with 'binary' as inferred_dtype."""

    def __init__(self, name_typ=None, data_typ=None):
        # data_typ is added just for compatibility with StringIndexType
        assert (
            data_typ is None or data_typ == binary_array_type
        ), "data_typ must be binary_array_type"
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = binary_array_type
        super(BinaryIndexType, self).__init__(
            name="BinaryIndexType({})".format(name_typ)
        )

    ndim = 1

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 1, "C")

    def copy(self):
        return BinaryIndexType(self.name_typ)

    @property
    def dtype(self):
        return bytes_type

    @property
    def pandas_type_name(self):
        return "bytes"

    @property
    def numpy_type_name(self):
        return "object"

    @property
    def iterator_type(self):
        return bodo.utils.typing.BodoArrayIterator(self)


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(BinaryIndexType)
class BinaryIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("data", binary_array_type),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(bytes_type, types.int64)),
        ]
        super(BinaryIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(BinaryIndexType, "data", "_data")
make_attribute_wrapper(BinaryIndexType, "name", "_name")
make_attribute_wrapper(BinaryIndexType, "dict", "_dict")


# ---------------- Helper fns common to both String/Binary index types -------------------


@unbox(BinaryIndexType)
@unbox(StringIndexType)
def unbox_binary_str_index(typ, val, c):
    """
    helper function that handles unboxing for both binary and string index types
    """

    array_type = typ.data
    scalar_type = typ.data.dtype

    # get data and name attributes
    # TODO: use to_numpy()
    values_obj = c.pyapi.object_getattr_string(val, "values")
    data = c.pyapi.to_native_value(array_type, values_obj).value
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
        lambda: numba.typed.Dict.empty(scalar_type, types.int64),
        types.DictType(scalar_type, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@box(BinaryIndexType)
@box(StringIndexType)
def box_binary_str_index(typ, val, c):
    """
    helper function that handles boxing for both binary and string index types
    """
    array_type = typ.data
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    c.context.nrt.incref(c.builder, array_type, index_val.data)
    data_obj = c.pyapi.from_native_value(array_type, index_val.data, c.env_manager)
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
def init_binary_str_index(typingctx, data, name=None):
    """Create StringIndex or BinaryIndex object"""
    name = types.none if name is None else name

    sig = type(bodo.utils.typing.get_index_type_from_dtype(data.dtype))(name, data)(
        data, name
    )
    cg = get_binary_str_codegen(is_binary=data.dtype == bytes_type)
    return sig, cg


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_binary_str_index = (
    init_index_equiv
)


def get_binary_str_codegen(is_binary=False):
    """
    helper function that returns the codegen for initializing a binary/string index
    """

    if is_binary:
        scalar_dtype_string = "bytes_type"
    else:
        scalar_dtype_string = "string_type"

    func_text = "def impl(context, builder, signature, args):\n"
    func_text += "    assert len(args) == 2\n"
    func_text += "    index_typ = signature.return_type\n"
    func_text += (
        "    index_val = cgutils.create_struct_proxy(index_typ)(context, builder)\n"
    )
    func_text += "    index_val.data = args[0]\n"
    func_text += "    index_val.name = args[1]\n"
    func_text += "    # increase refcount of stored values\n"
    func_text += "    context.nrt.incref(builder, signature.args[0], args[0])\n"
    func_text += "    context.nrt.incref(builder, index_typ.name_typ, args[1])\n"
    func_text += "    # create empty dict for get_loc hashmap\n"
    func_text += "    index_val.dict = context.compile_internal(\n"
    func_text += "       builder,\n"
    func_text += (
        f"       lambda: numba.typed.Dict.empty({scalar_dtype_string}, types.int64),\n"
    )
    func_text += f"        types.DictType({scalar_dtype_string}, types.int64)(), [],)\n"
    func_text += "    return index_val._getvalue()\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "signature": signature,
            "cgutils": cgutils,
            "numba": numba,
            "types": types,
            "bytes_type": bytes_type,
            "string_type": string_type,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@overload_method(BinaryIndexType, "copy", no_unliteral=True)
@overload_method(StringIndexType, "copy", no_unliteral=True)
def overload_binary_string_index_copy(A, name=None, deep=False, dtype=None, names=None):

    typ = type(A)

    err_str = idx_typ_to_format_str_map[typ].format("copy()")
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    check_unsupported_args(
        "Index.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_binary_str_index(
                A._data.copy(), name
            )

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_binary_str_index(
                A._data.copy(), A._name
            )

    return impl


# ---------------- Common Index fns -------------------


@overload_attribute(BinaryIndexType, "name")
@overload_attribute(StringIndexType, "name")
@overload_attribute(DatetimeIndexType, "name")
@overload_attribute(TimedeltaIndexType, "name")
@overload_attribute(RangeIndexType, "name")
@overload_attribute(PeriodIndexType, "name")
@overload_attribute(NumericIndexType, "name")
@overload_attribute(IntervalIndexType, "name")
@overload_attribute(CategoricalIndexType, "name")
@overload_attribute(MultiIndexType, "name")
def Index_get_name(i):
    def impl(i):  # pragma: no cover
        return i._name

    return impl


@overload(operator.getitem, no_unliteral=True)
def overload_index_getitem(I, ind):
    # output of integer indexing is scalar value
    if isinstance(
        I, (NumericIndexType, StringIndexType, BinaryIndexType)
    ) and isinstance(ind, types.Integer):
        return lambda I, ind: bodo.hiframes.pd_index_ext.get_index_data(I)[
            ind
        ]  # pragma: no cover

    # output of slice, bool array ... indexing is pd.Index
    if isinstance(I, NumericIndexType):
        return lambda I, ind: bodo.hiframes.pd_index_ext.init_numeric_index(
            bodo.hiframes.pd_index_ext.get_index_data(I)[ind],
            bodo.hiframes.pd_index_ext.get_index_name(I),
        )  # pragma: no cover

    if isinstance(I, (StringIndexType, BinaryIndexType)):
        return lambda I, ind: bodo.hiframes.pd_index_ext.init_binary_str_index(
            bodo.hiframes.pd_index_ext.get_index_data(I)[ind],
            bodo.hiframes.pd_index_ext.get_index_name(I),
        )  # pragma: no cover


# similar to index_from_array()
def array_type_to_index(arr_typ, name_typ=None):
    """convert array type to a corresponding Index type"""
    if is_str_arr_type(arr_typ):
        return StringIndexType(name_typ, arr_typ)
    if arr_typ == bodo.binary_array_type:
        return BinaryIndexType(name_typ)

    assert isinstance(
        arr_typ, (types.Array, IntegerArrayType, bodo.CategoricalArrayType)
    ) or arr_typ in (
        bodo.datetime_date_array_type,
        bodo.boolean_array,
    ), f"Converting array type {arr_typ} to index not supported"

    # TODO: Pandas keeps datetime_date Index as a generic Index(, dtype=object)
    # Fix this implementation to match.
    if arr_typ == bodo.datetime_date_array_type or arr_typ.dtype == types.NPDatetime(
        "ns"
    ):
        return DatetimeIndexType(name_typ)

    if isinstance(arr_typ, bodo.DatetimeArrayType):
        return DatetimeIndexType(name_typ, arr_typ)

    # categorical array
    if isinstance(arr_typ, bodo.CategoricalArrayType):
        return CategoricalIndexType(arr_typ, name_typ)

    if arr_typ.dtype == types.NPTimedelta("ns"):
        return TimedeltaIndexType(name_typ)

    if isinstance(arr_typ.dtype, (types.Integer, types.Float, types.Boolean)):
        return NumericIndexType(arr_typ.dtype, name_typ, arr_typ)

    raise BodoError(f"invalid index type {arr_typ}")


def is_pd_index_type(t):
    return isinstance(
        t,
        (
            NumericIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            IntervalIndexType,
            CategoricalIndexType,
            PeriodIndexType,
            StringIndexType,
            BinaryIndexType,
            RangeIndexType,
            HeterogeneousIndexType,
        ),
    )


# TODO: test
@overload_method(RangeIndexType, "take", no_unliteral=True)
@overload_method(NumericIndexType, "take", no_unliteral=True)
@overload_method(StringIndexType, "take", no_unliteral=True)
@overload_method(BinaryIndexType, "take", no_unliteral=True)
@overload_method(CategoricalIndexType, "take", no_unliteral=True)
@overload_method(PeriodIndexType, "take", no_unliteral=True)
@overload_method(DatetimeIndexType, "take", no_unliteral=True)
@overload_method(TimedeltaIndexType, "take", no_unliteral=True)
def overload_index_take(I, indices, axis=0, allow_fill=True, fill_value=None):
    unsupported_args = dict(axis=axis, allow_fill=allow_fill, fill_value=fill_value)
    default_args = dict(axis=0, allow_fill=True, fill_value=None)
    check_unsupported_args(
        "Index.take",
        unsupported_args,
        default_args,
        package_name="pandas",
        module_name="Index",
    )
    return lambda I, indices: I[indices]  # pragma: no cover


def _init_engine(I, ban_unique=True):
    pass


@overload(_init_engine)
def overload_init_engine(I, ban_unique=True):
    """initialize the Index hashmap engine (just a simple dict for now)"""
    if isinstance(I, CategoricalIndexType):

        def impl(I, ban_unique=True):  # pragma: no cover
            if len(I) > 0 and not I._dict:
                arr = bodo.utils.conversion.coerce_to_array(I)
                for i in range(len(arr)):
                    if not bodo.libs.array_kernels.isna(arr, i):
                        val = bodo.hiframes.pd_categorical_ext.get_code_for_value(
                            arr.dtype, arr[i]
                        )
                        if ban_unique and val in I._dict:
                            raise ValueError(
                                "Index.get_loc(): non-unique Index not supported yet"
                            )
                        I._dict[val] = i

        return impl
    else:

        def impl(I, ban_unique=True):  # pragma: no cover
            if len(I) > 0 and not I._dict:
                arr = bodo.utils.conversion.coerce_to_array(I)
                for i in range(len(arr)):
                    if not bodo.libs.array_kernels.isna(arr, i):
                        val = arr[i]
                        if ban_unique and val in I._dict:
                            raise ValueError(
                                "Index.get_loc(): non-unique Index not supported yet"
                            )
                        I._dict[val] = i

        return impl


@overload(operator.contains, no_unliteral=True)
def index_contains(I, val):
    """support for "val in I" operator. Uses the Index hashmap for faster results."""
    if not is_index_type(I):  # pragma: no cover
        return

    if isinstance(I, RangeIndexType):
        return lambda I, val: range_contains(
            I.start, I.stop, I.step, val
        )  # pragma: no cover

    if isinstance(I, CategoricalIndexType):

        def impl(I, val):  # pragma: no cover
            key = bodo.utils.conversion.unbox_if_timestamp(val)
            if not is_null_value(I._dict):
                _init_engine(I, False)
                arr = bodo.utils.conversion.coerce_to_array(I)
                code = bodo.hiframes.pd_categorical_ext.get_code_for_value(
                    arr.dtype, key
                )
                return code in I._dict
            else:
                # TODO(ehsan): support raising a proper BodoWarning object
                msg = "Global Index objects can be slow (pass as argument to JIT function for better performance)."
                warnings.warn(msg)
                arr = bodo.utils.conversion.coerce_to_array(I)
                ind = -1
                for i in range(len(arr)):
                    if not bodo.libs.array_kernels.isna(arr, i):
                        if arr[i] == key:
                            ind = i
            return ind != -1

        return impl

    # Note: does not work on implicit Timedelta via string
    # i.e. "1 days" in pd.TimedeltaIndex(["1 days", "2 hours"])
    def impl(I, val):  # pragma: no cover
        key = bodo.utils.conversion.unbox_if_timestamp(val)
        if not is_null_value(I._dict):
            _init_engine(I, False)
            return key in I._dict
        else:
            # TODO(ehsan): support raising a proper BodoWarning object
            msg = "Global Index objects can be slow (pass as argument to JIT function for better performance)."
            warnings.warn(msg)
            arr = bodo.utils.conversion.coerce_to_array(I)
            ind = -1
            for i in range(len(arr)):
                if not bodo.libs.array_kernels.isna(arr, i):
                    if arr[i] == key:
                        ind = i
        return ind != -1

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
@overload_method(BinaryIndexType, "get_loc", no_unliteral=True)
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
    check_unsupported_args(
        "Index.get_loc",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    # Timestamp/Timedelta types are handled the same as datetime64/timedelta64
    key = types.unliteral(key)
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        I, "DatetimeIndex.get_loc"
    )
    if key == pd_timestamp_type:
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

        key = bodo.utils.conversion.unbox_if_timestamp(key)
        # build the index dict if not initialized yet
        if not is_null_value(I._dict):
            _init_engine(I)
            ind = I._dict.get(key, -1)
        else:
            # TODO(ehsan): support raising a proper BodoWarning object
            msg = "Index.get_loc() can be slow for global Index objects (pass as argument to JIT function for better performance)."
            warnings.warn(msg)
            arr = bodo.utils.conversion.coerce_to_array(I)
            ind = -1
            for i in range(len(arr)):
                if arr[i] == key:
                    if ind != -1:
                        raise ValueError(
                            "Index.get_loc(): non-unique Index not supported yet"
                        )
                    ind = i

        if ind == -1:
            raise KeyError("Index.get_loc(): key not found")
        return ind

    return impl


def create_isna_specific_method(overload_name):
    def overload_index_isna_specific_method(I):
        """Generic implementation for Index.isna() and Index.notna()."""
        cond_when_isna = overload_name in {"isna", "isnull"}

        if isinstance(I, RangeIndexType):
            # TODO: parallelize np.full in PA
            # return lambda I: np.full(len(I), <cond>, np.bool_)
            def impl(I):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                n = len(I)
                out_arr = np.empty(n, np.bool_)
                for i in numba.parfors.parfor.internal_prange(n):
                    out_arr[i] = not cond_when_isna
                return out_arr

            return impl

        func_text = (
            "def impl(I):\n"
            "    numba.parfors.parfor.init_prange()\n"
            "    arr = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
            "    n = len(arr)\n"
            "    out_arr = np.empty(n, np.bool_)\n"
            "    for i in numba.parfors.parfor.internal_prange(n):\n"
            f"       out_arr[i] = {'' if cond_when_isna else 'not '}"
            "bodo.libs.array_kernels.isna(arr, i)\n"
            "    return out_arr\n"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "np": np, "numba": numba}, loc_vars)
        impl = loc_vars["impl"]
        return impl

    return overload_index_isna_specific_method


isna_overload_types = (
    RangeIndexType,
    NumericIndexType,
    StringIndexType,
    BinaryIndexType,
    CategoricalIndexType,
    PeriodIndexType,
    DatetimeIndexType,
    TimedeltaIndexType,
)


isna_specific_methods = (
    "isna",
    "notna",
    "isnull",
    "notnull",
)


def _install_isna_specific_methods():
    for overload_type in isna_overload_types:
        for overload_name in isna_specific_methods:
            overload_impl = create_isna_specific_method(overload_name)
            overload_method(
                overload_type,
                overload_name,
                no_unliteral=True,
                inline="always",
            )(overload_impl)


_install_isna_specific_methods()


@overload_attribute(RangeIndexType, "values")
@overload_attribute(NumericIndexType, "values")
@overload_attribute(StringIndexType, "values")
@overload_attribute(BinaryIndexType, "values")
@overload_attribute(CategoricalIndexType, "values")
@overload_attribute(PeriodIndexType, "values")
@overload_attribute(DatetimeIndexType, "values")
@overload_attribute(TimedeltaIndexType, "values")
def overload_values(I):
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(I, "Index.values")
    return lambda I: bodo.utils.conversion.coerce_to_array(I)  # pragma: no cover


@overload(len, no_unliteral=True)
def overload_index_len(I):
    if isinstance(
        I,
        (
            NumericIndexType,
            StringIndexType,
            BinaryIndexType,
            PeriodIndexType,
            IntervalIndexType,
            CategoricalIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            HeterogeneousIndexType,
        ),
    ):
        # TODO: test
        return lambda I: len(
            bodo.hiframes.pd_index_ext.get_index_data(I)
        )  # pragma: no cover


@overload_attribute(DatetimeIndexType, "shape")
@overload_attribute(NumericIndexType, "shape")
@overload_attribute(StringIndexType, "shape")
@overload_attribute(BinaryIndexType, "shape")
@overload_attribute(PeriodIndexType, "shape")
@overload_attribute(TimedeltaIndexType, "shape")
@overload_attribute(IntervalIndexType, "shape")
@overload_attribute(CategoricalIndexType, "shape")
def overload_index_shape(s):
    return lambda s: (
        len(bodo.hiframes.pd_index_ext.get_index_data(s)),
    )  # pragma: no cover


@overload_attribute(RangeIndexType, "shape")
def overload_range_index_shape(s):
    return lambda s: (len(s),)  # pragma: no cover


@overload_attribute(NumericIndexType, "is_monotonic", inline="always")
@overload_attribute(RangeIndexType, "is_monotonic", inline="always")
@overload_attribute(DatetimeIndexType, "is_monotonic", inline="always")
@overload_attribute(TimedeltaIndexType, "is_monotonic", inline="always")
@overload_attribute(NumericIndexType, "is_monotonic_increasing", inline="always")
@overload_attribute(RangeIndexType, "is_monotonic_increasing", inline="always")
@overload_attribute(DatetimeIndexType, "is_monotonic_increasing", inline="always")
@overload_attribute(TimedeltaIndexType, "is_monotonic_increasing", inline="always")
def overload_index_is_montonic(I):
    """
    Implementation of is_monotonic and is_monotonic_increasing attributes for Int64Index,
    UInt64Index, Float64Index, DatetimeIndex, TimedeltaIndex, and RangeIndex types.
    """
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        I, "Index.is_monotonic_increasing"
    )
    if isinstance(I, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType)):

        def impl(I):  # pragma: no cover
            arr = bodo.hiframes.pd_index_ext.get_index_data(I)
            return bodo.libs.array_kernels.series_monotonicity(arr, 1)

        return impl

    elif isinstance(I, RangeIndexType):

        def impl(I):  # pragma: no cover
            # Implementation matches pandas.RangeIndex.is_monotonic:
            # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/indexes/range.py#L356-L362
            return I._step > 0 or len(I) <= 1

        return impl


@overload_attribute(NumericIndexType, "is_monotonic_decreasing", inline="always")
@overload_attribute(RangeIndexType, "is_monotonic_decreasing", inline="always")
@overload_attribute(DatetimeIndexType, "is_monotonic_decreasing", inline="always")
@overload_attribute(TimedeltaIndexType, "is_monotonic_decreasing", inline="always")
def overload_index_is_montonic_decreasing(I):
    """
    Implementation of is_monotonic_decreasing attribute for Int64Index,
    UInt64Index, Float64Index, DatetimeIndex, TimedeltaIndex, and RangeIndex.
    """
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(
        I, "Index.is_monotonic_decreasing"
    )
    if isinstance(I, (NumericIndexType, DatetimeIndexType, TimedeltaIndexType)):

        def impl(I):  # pragma: no cover
            arr = bodo.hiframes.pd_index_ext.get_index_data(I)
            return bodo.libs.array_kernels.series_monotonicity(arr, 2)

        return impl
    elif isinstance(I, RangeIndexType):

        def impl(I):  # pragma: no cover
            # Implementation matches pandas.RangeIndex.is_monotonic_decreasing:
            # https://github.com/pandas-dev/pandas/blob/66e3805b8cabe977f40c05259cc3fcf7ead5687d/pandas/core/indexes/range.py#L356-L362
            return I._step < 0 or len(I) <= 1

        return impl


@overload_method(NumericIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(DatetimeIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(TimedeltaIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(StringIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(PeriodIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(CategoricalIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(BinaryIndexType, "duplicated", inline="always", no_unliteral=True)
@overload_method(RangeIndexType, "duplicated", inline="always", no_unliteral=True)
def overload_index_duplicated(I, keep="first"):
    """
    Implementation of Index.duplicated() for all supported index types.
    """

    if isinstance(I, RangeIndexType):

        def impl(I, keep="first"):  # pragma: no cover
            return np.zeros(len(I), np.bool_)

        return impl

    def impl(I, keep="first"):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        out_arr = bodo.libs.array_kernels.duplicated((arr,))
        return out_arr

    return impl


@overload_method(RangeIndexType, "drop_duplicates", no_unliteral=True, inline="always")
@overload_method(
    NumericIndexType, "drop_duplicates", no_unliteral=True, inline="always"
)
@overload_method(StringIndexType, "drop_duplicates", no_unliteral=True, inline="always")
@overload_method(BinaryIndexType, "drop_duplicates", no_unliteral=True, inline="always")
@overload_method(
    CategoricalIndexType, "drop_duplicates", no_unliteral=True, inline="always"
)
@overload_method(PeriodIndexType, "drop_duplicates", no_unliteral=True, inline="always")
@overload_method(
    DatetimeIndexType, "drop_duplicates", no_unliteral=True, inline="always"
)
@overload_method(
    TimedeltaIndexType, "drop_duplicates", no_unliteral=True, inline="always"
)
def overload_index_drop_duplicates(I, keep="first"):
    """Overload `Index.drop_duplicates` method for all index types."""
    unsupported_args = dict(keep=keep)
    arg_defaults = dict(keep="first")
    check_unsupported_args(
        "Index.drop_duplicates",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if isinstance(I, RangeIndexType):
        return lambda I, keep="first": I.copy()  # pragma: no cover

    func_text = (
        "def impl(I, keep='first'):\n"
        "    data = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
        "    arr = bodo.libs.array_kernels.drop_duplicates_array(data)\n"
        "    name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    )
    if isinstance(I, PeriodIndexType):
        func_text += f"    return bodo.hiframes.pd_index_ext.init_period_index(arr, name, '{I.freq}')\n"
    else:
        func_text += "    return bodo.utils.conversion.index_from_array(arr, name)"

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


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
    ("init_binary_str_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("init_categorical_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func


# array analysis extension
def get_index_data_equiv(self, scope, equiv_set, loc, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    # avoid returning shape for tuple input (results in dimension mismatch error)
    if isinstance(self.typemap[var.name], (HeterogeneousIndexType, MultiIndexType)):
        return None
    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=var, pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_get_index_data = (
    get_index_data_equiv
)


@overload_method(RangeIndexType, "map", inline="always", no_unliteral=True)
@overload_method(NumericIndexType, "map", inline="always", no_unliteral=True)
@overload_method(StringIndexType, "map", inline="always", no_unliteral=True)
@overload_method(BinaryIndexType, "map", inline="always", no_unliteral=True)
@overload_method(CategoricalIndexType, "map", inline="always", no_unliteral=True)
@overload_method(PeriodIndexType, "map", inline="always", no_unliteral=True)
@overload_method(DatetimeIndexType, "map", inline="always", no_unliteral=True)
@overload_method(TimedeltaIndexType, "map", inline="always", no_unliteral=True)
def overload_index_map(I, mapper, na_action=None):

    if not is_const_func_type(mapper):
        raise BodoError("Index.map(): 'mapper' should be a function")

    unsupported_args = dict(
        na_action=na_action,
    )
    map_defaults = dict(
        na_action=None,
    )
    check_unsupported_args(
        "Index.map",
        unsupported_args,
        map_defaults,
        package_name="pandas",
        module_name="Index",
    )

    dtype = I.dtype
    # getitem returns Timestamp for dt_index (TODO: pd.Timedelta when available)
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(I, "DatetimeIndex.map")
    if dtype == types.NPDatetime("ns"):
        dtype = pd_timestamp_type
    if dtype == types.NPTimedelta("ns"):
        dtype = pd_timedelta_type
    if isinstance(dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        dtype = dtype.elem_type

    # get output element type
    typing_context = numba.core.registry.cpu_target.typing_context
    target_context = numba.core.registry.cpu_target.target_context
    try:
        f_return_type = get_const_func_output_type(
            mapper, (dtype,), {}, typing_context, target_context
        )
    except Exception as e:
        raise_bodo_error(get_udf_error_msg("Index.map()", e))

    out_arr_type = get_udf_out_arr_type(f_return_type)

    # Just default to ignore?
    func = get_overload_const_func(mapper, None)
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
@lower_builtin(operator.is_, BinaryIndexType, BinaryIndexType)
@lower_builtin(operator.is_, PeriodIndexType, PeriodIndexType)
@lower_builtin(operator.is_, DatetimeIndexType, DatetimeIndexType)
@lower_builtin(operator.is_, TimedeltaIndexType, TimedeltaIndexType)
@lower_builtin(operator.is_, IntervalIndexType, IntervalIndexType)
@lower_builtin(operator.is_, CategoricalIndexType, CategoricalIndexType)
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
    def overload_index_binary_op(lhs, rhs):

        # left arg is Index
        if is_index_type(lhs):
            func_text = (
                "def impl(lhs, rhs):\n"
                "  arr = bodo.utils.conversion.coerce_to_array(lhs)\n"
            )
            if rhs in [
                bodo.hiframes.pd_timestamp_ext.pd_timestamp_type,
                bodo.hiframes.pd_timestamp_ext.pd_timedelta_type,
            ]:
                func_text += (
                    "  dt = bodo.utils.conversion.unbox_if_timestamp(rhs)\n"
                    "  return op(arr, dt)\n"
                )
            else:
                func_text += (
                    "  rhs_arr = bodo.utils.conversion.get_array_if_series_or_index(rhs)\n"
                    "  return op(arr, rhs_arr)\n"
                )
            loc_vars = {}
            exec(
                func_text,
                {"bodo": bodo, "op": op},
                loc_vars,
            )
            impl = loc_vars["impl"]
            return impl

        # right arg is Index
        if is_index_type(rhs):
            func_text = (
                "def impl(lhs, rhs):\n"
                "  arr = bodo.utils.conversion.coerce_to_array(rhs)\n"
            )
            if lhs in [
                bodo.hiframes.pd_timestamp_ext.pd_timestamp_type,
                bodo.hiframes.pd_timestamp_ext.pd_timedelta_type,
            ]:
                func_text += (
                    "  dt = bodo.utils.conversion.unbox_if_timestamp(lhs)\n"
                    "  return op(dt, arr)\n"
                )
            else:
                func_text += (
                    "  lhs_arr = bodo.utils.conversion.get_array_if_series_or_index(lhs)\n"
                    "  return op(lhs_arr, arr)\n"
                )
            loc_vars = {}
            exec(
                func_text,
                {"bodo": bodo, "op": op},
                loc_vars,
            )
            impl = loc_vars["impl"]
            return impl

        if isinstance(lhs, HeterogeneousIndexType):
            # handle as regular array data if not actually heterogeneous
            if not is_heterogeneous_tuple_type(lhs.data):

                def impl3(lhs, rhs):  # pragma: no cover
                    data = bodo.utils.conversion.coerce_to_array(lhs)
                    arr = bodo.utils.conversion.coerce_to_array(data)
                    rhs_arr = bodo.utils.conversion.get_array_if_series_or_index(rhs)
                    out_arr = op(arr, rhs_arr)
                    return out_arr

                return impl3

            count = len(lhs.data.types)
            # TODO(ehsan): return Numpy array (fix Numba errors)
            func_text = "def f(lhs, rhs):\n"
            func_text += "  return [{}]\n".format(
                ",".join(
                    "op(lhs[{}], rhs{})".format(
                        i, f"[{i}]" if is_iterable_type(rhs) else ""
                    )
                    for i in range(count)
                ),
            )
            loc_vars = {}
            exec(func_text, {"op": op, "np": np}, loc_vars)
            impl = loc_vars["f"]
            return impl

        if isinstance(rhs, HeterogeneousIndexType):
            # handle as regular array data if not actually heterogeneous
            if not is_heterogeneous_tuple_type(rhs.data):

                def impl4(lhs, rhs):  # pragma: no cover
                    data = bodo.hiframes.pd_index_ext.get_index_data(rhs)
                    arr = bodo.utils.conversion.coerce_to_array(data)
                    rhs_arr = bodo.utils.conversion.get_array_if_series_or_index(lhs)
                    out_arr = op(rhs_arr, arr)
                    return out_arr

                return impl4

            count = len(rhs.data.types)
            # TODO(ehsan): return Numpy array (fix Numba errors)
            func_text = "def f(lhs, rhs):\n"
            func_text += "  return [{}]\n".format(
                ",".join(
                    "op(lhs{}, rhs[{}])".format(
                        f"[{i}]" if is_iterable_type(lhs) else "", i
                    )
                    for i in range(count)
                ),
            )
            loc_vars = {}
            exec(func_text, {"op": op, "np": np}, loc_vars)
            impl = loc_vars["f"]
            return impl

    return overload_index_binary_op


# operators taken care of in binops_ext.py
skips = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.pow,
    operator.mod,
]


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        if op in skips:
            continue
        overload_impl = create_binary_op_overload(op)
        overload(op, inline="always")(overload_impl)


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
            BinaryIndexType,
            PeriodIndexType,
            DatetimeIndexType,
            TimedeltaIndexType,
            IntervalIndexType,
            CategoricalIndexType,
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

    def __init__(self, data=None, name_typ=None):

        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        super(HeterogeneousIndexType, self).__init__(
            name=f"heter_index({data}, {name_typ})"
        )

    def copy(self):
        return HeterogeneousIndexType(self.data, self.name_typ)

    @property
    def key(self):
        return self.data, self.name_typ

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    @property
    def pandas_type_name(self):
        return "object"

    @property
    def numpy_type_name(self):
        return "object"


# even though name attribute is mutable, we don't handle it for now
# TODO: create refcounted payload to handle mutable name
@register_model(HeterogeneousIndexType)
class HeterogeneousIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("data", fe_type.data), ("name", fe_type.name_typ)]
        super(HeterogeneousIndexModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(HeterogeneousIndexType, "data", "_data")
make_attribute_wrapper(HeterogeneousIndexType, "name", "_name")


@overload_method(HeterogeneousIndexType, "copy", no_unliteral=True)
def overload_heter_index_copy(A, name=None, deep=False, dtype=None, names=None):
    err_str = idx_typ_to_format_str_map[HeterogeneousIndexType].format("copy()")
    idx_cpy_unsupported_args = dict(deep=deep, dtype=dtype, names=names)
    check_unsupported_args(
        "Index.copy",
        idx_cpy_unsupported_args,
        idx_cpy_arg_defaults,
        fn_str=err_str,
        package_name="pandas",
        module_name="Index",
    )

    # NOTE: assuming data is immutable
    if not is_overload_none(name):

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_numeric_index(A._data.copy(), name)

    else:

        def impl(A, name=None, deep=False, dtype=None, names=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.init_numeric_index(
                A._data.copy(), A._name
            )

    return impl


# TODO(ehsan): test
@box(HeterogeneousIndexType)
def box_heter_index(typ, val, c):  # pragma: no cover
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    class_obj = c.pyapi.import_module_noblock(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)
    c.context.nrt.incref(c.builder, typ.data, index_val.data)
    data_obj = c.pyapi.from_native_value(typ.data, index_val.data, c.env_manager)
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
        context.nrt.incref(builder, index_typ.name_typ, args[1])
        return index_val._getvalue()

    return HeterogeneousIndexType(data, name)(data, name), codegen


@overload_attribute(HeterogeneousIndexType, "name")
def heter_index_get_name(i):
    def impl(i):  # pragma: no cover
        return i._name

    return impl


@overload_attribute(NumericIndexType, "nbytes")
@overload_attribute(DatetimeIndexType, "nbytes")
@overload_attribute(TimedeltaIndexType, "nbytes")
@overload_attribute(RangeIndexType, "nbytes")
@overload_attribute(StringIndexType, "nbytes")
@overload_attribute(BinaryIndexType, "nbytes")
@overload_attribute(CategoricalIndexType, "nbytes")
@overload_attribute(PeriodIndexType, "nbytes")
def overload_nbytes(I):
    """Add support for Index.nbytes by computing underlying arrays nbytes"""
    # Note: Pandas have a different underlying data structure
    # Hence, we get different number from Pandas RangeIndex.nbytes
    if isinstance(I, RangeIndexType):

        def _impl_nbytes(I):  # pragma: no cover
            return (
                bodo.io.np_io.get_dtype_size(type(I._start))
                + bodo.io.np_io.get_dtype_size(type(I._step))
                + bodo.io.np_io.get_dtype_size(type(I._stop))
            )

        return _impl_nbytes
    else:

        def _impl_nbytes(I):  # pragma: no cover
            return I._data.nbytes

        return _impl_nbytes


@overload_method(NumericIndexType, "rename", inline="always")
@overload_method(DatetimeIndexType, "rename", inline="always")
@overload_method(TimedeltaIndexType, "rename", inline="always")
@overload_method(RangeIndexType, "rename", inline="always")
@overload_method(StringIndexType, "rename", inline="always")
@overload_method(BinaryIndexType, "rename", inline="always")
@overload_method(CategoricalIndexType, "rename", inline="always")
@overload_method(PeriodIndexType, "rename", inline="always")
@overload_method(IntervalIndexType, "rename", inline="always")
@overload_method(HeterogeneousIndexType, "rename", inline="always")
def overload_rename(I, name, inplace=False):
    """Add support for Index.rename"""
    if is_overload_true(inplace):
        raise BodoError("Index.rename(): inplace index renaming unsupported")

    return init_index_from_index(I, name)


def init_index_from_index(I, name):
    """Creates an Index value using data of input Index 'I' and new name value 'name'"""
    # TODO: add more possible initializer types
    standard_init_map = {
        NumericIndexType: bodo.hiframes.pd_index_ext.init_numeric_index,
        DatetimeIndexType: bodo.hiframes.pd_index_ext.init_datetime_index,
        TimedeltaIndexType: bodo.hiframes.pd_index_ext.init_timedelta_index,
        StringIndexType: bodo.hiframes.pd_index_ext.init_binary_str_index,
        BinaryIndexType: bodo.hiframes.pd_index_ext.init_binary_str_index,
        CategoricalIndexType: bodo.hiframes.pd_index_ext.init_categorical_index,
        IntervalIndexType: bodo.hiframes.pd_index_ext.init_interval_index,
    }

    if type(I) in standard_init_map:
        init_func = standard_init_map[type(I)]
        return lambda I, name, inplace=False: init_func(
            bodo.hiframes.pd_index_ext.get_index_data(I).copy(), name
        )  # pragma: no cover

    if isinstance(I, RangeIndexType):
        # Distributed Pass currently assumes init_range_index is using integers
        # that are equal on all cores. Since we can't interpret the distributed
        # behavior from just scalars, we call copy instead.
        return lambda I, name, inplace=False: I.copy(name=name)  # pragma: no cover

    if isinstance(I, PeriodIndexType):
        freq = I.freq
        return (
            lambda I, name, inplace=False: bodo.hiframes.pd_index_ext.init_period_index(
                bodo.hiframes.pd_index_ext.get_index_data(I).copy(),
                name,
                freq,
            )
        )  # pragma: no cover

    if isinstance(I, HeterogeneousIndexType):
        return (
            lambda I, name, inplace=False: bodo.hiframes.pd_index_ext.init_heter_index(
                bodo.hiframes.pd_index_ext.get_index_data(I),
                name,
            )
        )  # pragma: no cover

    raise_bodo_error(f"init_index(): Unknown type {type(I)}")


def get_index_constructor(I):
    """Returns the constructor for a corresponding Index type"""
    standard_constructors = {
        NumericIndexType: bodo.hiframes.pd_index_ext.init_numeric_index,
        DatetimeIndexType: bodo.hiframes.pd_index_ext.init_datetime_index,
        TimedeltaIndexType: bodo.hiframes.pd_index_ext.init_timedelta_index,
        StringIndexType: bodo.hiframes.pd_index_ext.init_binary_str_index,
        BinaryIndexType: bodo.hiframes.pd_index_ext.init_binary_str_index,
        CategoricalIndexType: bodo.hiframes.pd_index_ext.init_categorical_index,
        IntervalIndexType: bodo.hiframes.pd_index_ext.init_interval_index,
        RangeIndexType: bodo.hiframes.pd_index_ext.init_range_index,
    }

    if type(I) in standard_constructors:  # pragma: no cover
        return standard_constructors[type(I)]

    raise BodoError(
        f"Unsupported type for standard Index constructor: {type(I)}"
    )  # pragma: no cover


@overload_method(NumericIndexType, "unique", no_unliteral=True, inline="always")
@overload_method(BinaryIndexType, "unique", no_unliteral=True, inline="always")
@overload_method(StringIndexType, "unique", no_unliteral=True, inline="always")
@overload_method(CategoricalIndexType, "unique", no_unliteral=True, inline="always")
# Does not work if the intervals are distinct but share a start-value
# (i.e. [(1, 2), (2, 3), (1, 3)]).
# Does not work for time-based intervals.
# See [BE-2813]
@overload_method(IntervalIndexType, "unique", no_unliteral=True, inline="always")
@overload_method(DatetimeIndexType, "unique", no_unliteral=True, inline="always")
@overload_method(TimedeltaIndexType, "unique", no_unliteral=True, inline="always")
def overload_index_unique(I):
    """Add support for Index.unique() on most Index types"""
    constructor = get_index_constructor(I)

    def impl(I):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        name = bodo.hiframes.pd_index_ext.get_index_name(I)
        uni = bodo.libs.array_kernels.unique(arr)
        return constructor(uni, name)

    return impl


@overload_method(RangeIndexType, "unique", no_unliteral=True)
def overload_range_index_unique(I):
    """Add support for Index.unique() on RangeIndex"""

    def impl(I):  # pragma: no cover
        return I.copy()

    return impl


@overload_method(PeriodIndexType, "unique", no_unliteral=True)
@overload_method(MultiIndexType, "unique", no_unliteral=True)
@overload_method(HeterogeneousIndexType, "unique", no_unliteral=True)
def overload_unsupported_index_unique(I):
    """Add support for Index.unique() on unsupported Index types"""
    raise BodoError(
        f"Index.unique(): {type(I).__name__} supported yet"
    )  # pragma: no cover


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
    """Constant lowering for DatetimeIndexType and TimedeltaIndexType."""
    if isinstance(ty.data, bodo.DatetimeArrayType):
        # TODO [BE-2441]: Unify?
        data = context.get_constant_generic(builder, ty.data, pyval.array)
    else:
        data = context.get_constant_generic(
            builder, types.Array(types.int64, 1, "C"), pyval.values.view(np.int64)
        )
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    # set the dictionary to null since we can't create it without memory leak (BE-2114)
    dtype = ty.dtype
    dict_null = context.get_constant_null(types.DictType(dtype, types.int64))
    return lir.Constant.literal_struct([data, name, dict_null])


@lower_constant(PeriodIndexType)
def lower_constant_period_index(context, builder, ty, pyval):
    """Constant lowering for PeriodIndexType."""
    data = context.get_constant_generic(
        builder,
        bodo.IntegerArrayType(types.int64),
        pd.arrays.IntegerArray(pyval.asi8, pyval.isna()),
    )
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    # set the dictionary to null since we can't create it without memory leak (BE-2114)
    dict_null = context.get_constant_null(types.DictType(types.int64, types.int64))
    return lir.Constant.literal_struct([data, name, dict_null])


@lower_constant(NumericIndexType)
def lower_constant_numeric_index(context, builder, ty, pyval):
    """Constant lowering for NumericIndexType."""

    # make sure the type is one of the numeric ones
    assert isinstance(ty.dtype, (types.Integer, types.Float, types.Boolean))

    # get the data
    data = context.get_constant_generic(
        builder, types.Array(ty.dtype, 1, "C"), pyval.values
    )
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    dtype = ty.dtype
    # set the dictionary to null since we can't create it without memory leak (BE-2114)
    dict_null = context.get_constant_null(types.DictType(dtype, types.int64))
    return lir.Constant.literal_struct([data, name, dict_null])


@lower_constant(StringIndexType)
@lower_constant(BinaryIndexType)
def lower_constant_binary_string_index(context, builder, ty, pyval):
    """Helper functon that handles constant lowering for Binary/String IndexType."""
    array_type = ty.data
    scalar_type = ty.data.dtype

    data = context.get_constant_generic(builder, array_type, pyval.values)
    name = context.get_constant_generic(builder, ty.name_typ, pyval.name)

    # set the dictionary to null since we can't create it without memory leak (BE-2114)
    dict_null = context.get_constant_null(types.DictType(scalar_type, types.int64))
    return lir.Constant.literal_struct([data, name, dict_null])


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
    pos_diff = builder.icmp_signed(">", diff, zero)
    pos_step = builder.icmp_signed(">", indexobj.step, zero)
    sign_same = builder.not_(builder.xor(pos_diff, pos_step))

    with builder.if_then(sign_same):
        rem = builder.srem(diff, indexobj.step)
        rem = builder.select(pos_diff, rem, builder.neg(rem))
        uneven = builder.icmp_signed(">", rem, zero)
        newcount = builder.add(
            builder.sdiv(diff, indexobj.step), builder.select(uneven, one, zero)
        )
        builder.store(newcount, countptr)

    res = iterobj._getvalue()

    # Note: a decref on the iterator will dereference all internal MemInfo*
    out = impl_ret_new_ref(context, builder, sig.return_type, res)
    return out


def _install_index_getiter():
    """install iterators for Index types"""
    index_types = [
        NumericIndexType,
        StringIndexType,
        BinaryIndexType,
        CategoricalIndexType,
        TimedeltaIndexType,
        DatetimeIndexType,
    ]

    for typ in index_types:
        lower_builtin("getiter", typ)(numba.np.arrayobj.getiter_array)


_install_index_getiter()

index_unsupported_methods = [
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
    "droplevel",
    "dropna",
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
    "nunique",
    "putmask",
    "ravel",
    "reindex",
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
    "value_counts",
    "view",
    "where",
]

index_unsupported_atrs = [
    "T",
    "array",
    "asi8",
    "dtype",
    "has_duplicates",
    "hasnans",
    "inferred_type",
    "is_all_dates",
    "is_unique",
    "ndim",
    "nlevels",
    "size",
    "names",
    "empty",
]

# unsupported RangeIndex class methods (handled in untyped pass)
# from_range

cat_idx_unsupported_atrs = [
    "codes",
    "categories",
    "ordered",
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

cat_idx_unsupported_methods = [
    "rename_categories",
    "reorder_categories",
    "add_categories",
    "remove_categories",
    "remove_unused_categories",
    "set_categories",
    "as_ordered",
    "as_unordered",
    "get_loc",
]


interval_idx_unsupported_atrs = [
    "closed",
    "is_empty",
    "is_non_overlapping_monotonic",
    "is_overlapping",
    "left",
    "right",
    "mid",
    "length",
    "values",
    "shape",
    "nbytes",
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

# unsupported Interval class methods (handled in untyped pass)
# from_arrays
# from_tuples
# from_breaks


interval_idx_unsupported_methods = [
    "contains",
    "copy",
    "overlaps",
    "set_closed",
    "to_tuples",
    "take",
    "get_loc",
    "isna",
    "isnull",
    "map",
]


multi_index_unsupported_atrs = [
    "levshape",
    "levels",
    "codes",
    "dtypes",
    "values",
    "shape",
    "nbytes",
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

# unsupported multi-index class methods (handled in untyped pass)
# from_arrays
# from_tuples
# from_frame


multi_index_unsupported_methods = [
    "copy",
    "set_levels",
    "set_codes",
    "swaplevel",
    "reorder_levels",
    "remove_unused_levels",
    "get_loc",
    "get_locs",
    "get_loc_level",
    "take",
    "isna",
    "isnull",
    "map",
]


dt_index_unsupported_atrs = [
    "time",
    "timez",
    "tz",
    "freq",
    "freqstr",
    "inferred_freq",
]

dt_index_unsupported_methods = [
    "normalize",
    "strftime",
    "snap",
    "tz_localize",
    "round",
    "floor",
    "ceil",
    "to_period",
    "to_perioddelta",
    "to_pydatetime",
    "month_name",
    "day_name",
    "mean",
    "indexer_at_time",
    "indexer_between",
    "indexer_between_time",
]


td_index_unsupported_atrs = [
    "components",
    "inferred_freq",
]

td_index_unsupported_methods = [
    "to_pydatetime",
    "round",
    "floor",
    "ceil",
    "mean",
]


period_index_unsupported_atrs = [
    "day",
    "dayofweek",
    "day_of_week",
    "dayofyear",
    "day_of_year",
    "days_in_month",
    "daysinmonth",
    "freq",
    "freqstr",
    "hour",
    "is_leap_year",
    "minute",
    "month",
    "quarter",
    "second",
    "week",
    "weekday",
    "weekofyear",
    "year",
    "end_time",
    "qyear",
    "start_time",
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

string_index_unsupported_atrs = [
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

binary_index_unsupported_atrs = [
    "is_monotonic",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

period_index_unsupported_methods = [
    "asfreq",
    "strftime",
    "to_timestamp",
]

index_types = [
    ("pandas.RangeIndex.{}", RangeIndexType),
    (
        "pandas.Index.{} with numeric data",
        NumericIndexType,
    ),
    (
        "pandas.Index.{} with string data",
        StringIndexType,
    ),
    (
        "pandas.Index.{} with binary data",
        BinaryIndexType,
    ),
    ("pandas.TimedeltaIndex.{}", TimedeltaIndexType),
    ("pandas.IntervalIndex.{}", IntervalIndexType),
    ("pandas.CategoricalIndex.{}", CategoricalIndexType),
    ("pandas.PeriodIndex.{}", PeriodIndexType),
    ("pandas.DatetimeIndex.{}", DatetimeIndexType),
    ("pandas.MultiIndex.{}", MultiIndexType),
]

for name, typ in index_types:
    idx_typ_to_format_str_map[typ] = name


def _install_index_unsupported():
    """install an overload that raises BodoError for unsupported methods of pd.Index"""

    # install unsupported methods that are common to all idx types
    for fname in index_unsupported_methods:
        for format_str, typ in index_types:
            overload_method(typ, fname, no_unliteral=True)(
                create_unsupported_overload(format_str.format(fname + "()"))
            )

    # install unsupported attributes that are common to all idx types
    for attr_name in index_unsupported_atrs:
        for format_str, typ in index_types:
            overload_attribute(typ, attr_name, no_unliteral=True)(
                create_unsupported_overload(format_str.format(attr_name))
            )

    unsupported_attrs_list = [
        (StringIndexType, string_index_unsupported_atrs),
        (BinaryIndexType, binary_index_unsupported_atrs),
        (CategoricalIndexType, cat_idx_unsupported_atrs),
        (IntervalIndexType, interval_idx_unsupported_atrs),
        (MultiIndexType, multi_index_unsupported_atrs),
        (DatetimeIndexType, dt_index_unsupported_atrs),
        (TimedeltaIndexType, td_index_unsupported_atrs),
        (PeriodIndexType, period_index_unsupported_atrs),
    ]

    unsupported_methods_list = [
        (CategoricalIndexType, cat_idx_unsupported_methods),
        (IntervalIndexType, interval_idx_unsupported_methods),
        (MultiIndexType, multi_index_unsupported_methods),
        (DatetimeIndexType, dt_index_unsupported_methods),
        (TimedeltaIndexType, td_index_unsupported_methods),
        (PeriodIndexType, period_index_unsupported_methods),
    ]

    # install unsupported methods for the individual idx types
    for typ, cur_typ_unsupported_methods_list in unsupported_methods_list:
        format_str = idx_typ_to_format_str_map[typ]
        for fn_name in cur_typ_unsupported_methods_list:
            overload_method(typ, fn_name, no_unliteral=True)(
                create_unsupported_overload(format_str.format(fn_name + "()"))
            )

    # install unsupported attributes for the individual idx types
    for typ, cur_typ_unsupported_attrs_list in unsupported_attrs_list:
        format_str = idx_typ_to_format_str_map[typ]
        for attr_name in cur_typ_unsupported_attrs_list:
            overload_attribute(typ, attr_name, no_unliteral=True)(
                create_unsupported_overload(format_str.format(attr_name))
            )

    # max/min only supported for TimedeltaIndexType, DatetimeIndexType
    for idx_typ in [
        RangeIndexType,
        NumericIndexType,
        StringIndexType,
        BinaryIndexType,
        IntervalIndexType,
        CategoricalIndexType,
        PeriodIndexType,
        MultiIndexType,
    ]:
        for fn_name in ["max", "min"]:
            format_str = idx_typ_to_format_str_map[idx_typ]
            overload_method(idx_typ, fn_name, no_unliteral=True)(
                create_unsupported_overload(format_str.format(fn_name + "()"))
            )


_install_index_unsupported()
