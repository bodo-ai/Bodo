import datetime
import operator
import warnings
from abc import ABC, abstractmethod

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
import bodo.pandas as bd
import bodo.utils.conversion
from bodo.hiframes.datetime_timedelta_ext import pd_timedelta_type
from bodo.hiframes.pd_multi_index_ext import IndexNameType, MultiIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import (
    PandasTimestampType,
    pd_timestamp_tz_naive_type,
)
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)
from bodo.libs.binary_arr_ext import binary_array_type, bytes_type
from bodo.libs.bool_arr_ext import boolean_array_type
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.pd_datetime_arr_ext import DatetimeArrayType, PandasDatetimeTZDtype
from bodo.libs.str_arr_ext import string_array_type
from bodo.libs.str_ext import string_type
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    check_unsupported_args,
    dtype_to_array_type,
    get_overload_const_func,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_tuple,
    get_udf_error_msg,
    get_udf_out_arr_type,
    get_val_type_maybe_str_literal,
    is_const_func_type,
    is_heterogeneous_tuple_type,
    is_iterable_type,
    is_overload_bool,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_nan,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    is_str_arr_type,
    parse_dtype,
    raise_bodo_error,
)
from bodo.utils.utils import (
    bodo_exec,
    is_null_value,
)

_dt_index_data_typ = types.Array(types.NPDatetime("ns"), 1, "C")
iNaT = pd._libs.tslibs.iNaT
NaT = types.NPDatetime("ns")("NaT")  # TODO: pd.NaT

# used in the various index copy overloads for error checking
idx_cpy_arg_defaults = {"deep": False, "dtype": None, "names": None}

# maps index_types to a format string of how we refer to the index type in error messages.
# for example:
# RangeIndexType --> "pandas.RangeIndex.{}"
# StringIndexType --> "pandas.Index.{} with string data"

# Initialized at the bottom of this file, after all the index types have been declared
idx_typ_to_format_str_map = {}


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
    if val.equals(pd.Index([])) and val.dtype == np.object_:
        return StringIndexType(get_val_type_maybe_str_literal(val.name))

    # Pandas uses object dtype for nullable int arrays
    if (
        val.inferred_type == "integer"
        or pd._libs.lib.infer_dtype(val, True) == "integer"
    ):
        # At least some index values contain the actual dtype in
        # Pandas 1.4.
        if isinstance(val.dtype, (pd.core.arrays.integer.IntegerDtype, pd.ArrowDtype)):
            # Get the numpy dtype
            numpy_dtype = val.dtype.numpy_dtype
            # Convert the numpy dtype to the Numba type
            dtype = numba.np.numpy_support.from_dtype(numpy_dtype)
            arr_type = IntegerArrayType(dtype)
        else:
            try:
                # dtype could be Numpy dtype
                dtype = numba.np.numpy_support.from_dtype(val.dtype)
                arr_type = types.Array(dtype, 1, "C")
            except numba.core.errors.NumbaNotImplementedError:
                # we don't have the dtype default to int64
                dtype = types.int64
                arr_type = IntegerArrayType(dtype)
        return NumericIndexType(
            dtype,
            get_val_type_maybe_str_literal(val.name),
            arr_type,
        )
    if val.inferred_type == "date" or pd._libs.lib.infer_dtype(val, True) == "date":
        dtype = bodo.types.datetime_date_type
        arr_type = bodo.types.datetime_date_array_type
        return NumericIndexType(
            dtype,
            get_val_type_maybe_str_literal(val.name),
            arr_type,
        )
    # handle nullable float Index
    if (
        val.inferred_type == "floating"
        or pd._libs.lib.infer_dtype(val, True) == "floating"
    ):
        # At least some index values contain the actual dtype in
        # Pandas 1.4.
        if isinstance(val.dtype, (pd.Float32Dtype, pd.Float64Dtype, pd.ArrowDtype)):
            # Get the numpy dtype
            numpy_dtype = val.dtype.numpy_dtype
            # Convert the numpy dtype to the Numba type
            dtype = numba.np.numpy_support.from_dtype(numpy_dtype)
            arr_type = FloatingArrayType(dtype)
        else:
            try:
                # dtype could be Numpy dtype
                dtype = numba.np.numpy_support.from_dtype(val.dtype)
                arr_type = types.Array(dtype, 1, "C")
            except numba.core.errors.NumbaNotImplementedError:
                # we don't have the dtype default to float64
                dtype = types.float64
                arr_type = FloatingArrayType(dtype)
        return NumericIndexType(
            dtype,
            get_val_type_maybe_str_literal(val.name),
            arr_type,
        )
    if (
        val.inferred_type == "boolean"
        or pd._libs.lib.infer_dtype(val, True) == "boolean"
    ):
        return NumericIndexType(
            types.bool_,
            get_val_type_maybe_str_literal(val.name),
            boolean_array_type,
        )

    if (
        val.inferred_type == "timedelta64"
        or pd._libs.lib.infer_dtype(val, True) == "timedelta64"
    ):
        return TimedeltaIndexType(
            get_val_type_maybe_str_literal(val.name), bodo.typeof(val.values)
        )

    # catch-all for all remaining Index types
    arr_typ = bodo.hiframes.boxing._infer_series_arr_type(val)
    if arr_typ == bodo.types.datetime_date_array_type or isinstance(
        arr_typ,
        (
            bodo.types.DecimalArrayType,
            bodo.types.DatetimeArrayType,
            bodo.types.TimeArrayType,
        ),
    ):
        return NumericIndexType(
            arr_typ.dtype,
            get_val_type_maybe_str_literal(val.name),
            arr_typ,
        )

    # catch-all for non-supported Index types
    # RangeIndex is directly supported (TODO: make sure this is not called)
    raise NotImplementedError(f"unsupported pd.Index type {val}")


# -------------------------  Base Index Type ------------------------------
class SingleIndexType(ABC):
    name_typ: IndexNameType

    @property
    @abstractmethod
    def pandas_type_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def numpy_type_name(self):
        raise NotImplementedError

    @property
    def nlevels(self):
        return 1


# -------------------------  DatetimeIndex ------------------------------


class DatetimeIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """type class for DatetimeIndex objects."""

    def __init__(self, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like freq/dtype/yearfirst?
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = (
            types.Array(numba.core.types.NPDatetime("ns"), 1, "C")
            if data is None
            else data
        )
        super().__init__(name=f"DatetimeIndex({name_typ}, {self.data})")

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
        return (
            self.data.tz
            if isinstance(self.data, bodo.types.DatetimeArrayType)
            else None
        )

    def copy(self):
        return DatetimeIndexType(self.name_typ, self.data)

    @property
    def iterator_type(self):
        # The underlying array is a datetime64, but the data is
        # (and should be) boxed as a pd.Timestamp
        return bodo.utils.typing.BodoArrayIterator(
            self, bodo.hiframes.pd_timestamp_ext.PandasTimestampType(self.tzval)
        )

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

    res = DatetimeIndexType(get_val_type_maybe_str_literal(val.name))
    return res


@register_model(DatetimeIndexType)
class DatetimeIndexModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # TODO: use payload to support mutable name
        members = [
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(_dt_index_data_typ.dtype, types.int64)),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(DatetimeIndexType, "data", "_data")
make_attribute_wrapper(DatetimeIndexType, "name", "_name")
make_attribute_wrapper(DatetimeIndexType, "dict", "_dict")


@overload_method(
    DatetimeIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_datetime_index_copy(A, name=None, deep=False, dtype=None, names=None):
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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
    pd_class_obj = c.pyapi.import_module(mod_name)

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
        # Call A.values.astype('datetime64[ns]', copy=False)
        data_obj_us = c.pyapi.object_getattr_string(val, "values")
        data_obj_us_astype = c.pyapi.object_getattr_string(data_obj_us, "astype")
        dtype_obj = c.pyapi.string_from_constant_string("datetime64[ns]")
        false_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))
        args = c.pyapi.tuple_pack([dtype_obj])
        kws = c.pyapi.dict_pack(
            [
                ("copy", false_obj),
            ]
        )
        data_obj = c.pyapi.call(data_obj_us_astype, args, kws)
        c.pyapi.decref(data_obj_us)
        c.pyapi.decref(data_obj_us_astype)
        c.pyapi.decref(dtype_obj)
        c.pyapi.decref(false_obj)
        c.pyapi.decref(args)
        c.pyapi.decref(kws)

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


@intrinsic(prefer_literal=True)
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
    func_text = "def bodo_gen_dti_field(dti):\n"
    func_text += "    numba.parfors.parfor.init_prange()\n"
    func_text += "    A = bodo.hiframes.pd_index_ext.get_index_data(dti)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(dti)\n"
    func_text += "    n = len(A)\n"
    # all datetimeindex fields return int32 as of Pandas 2.0.3
    # https://github.com/pandas-dev/pandas/blob/0f437949513225922d851e9581723d82120684a6/pandas/_libs/tslibs/fields.pyx
    func_text += "    S = np.empty(n, np.int32)\n"
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
    return bodo_exec(func_text, {"numba": numba, "np": np, "bodo": bodo}, {}, __name__)


def _install_dti_field_overload(field):
    """get field implementation and call overload_attribute()
    NOTE: This has to be a separate function to avoid unexpected free variable updates
    """
    impl = gen_dti_field_impl(field)
    overload_attribute(DatetimeIndexType, field)(lambda dti: impl)


def _install_dti_date_fields():
    for field in bodo.hiframes.pd_timestamp_ext.date_fields:
        if field in [
            "is_leap_year",
        ]:
            continue
        _install_dti_field_overload(field)


_install_dti_date_fields()


@overload_attribute(DatetimeIndexType, "is_leap_year", jit_options={"cache": True})
def overload_datetime_index_is_leap_year(dti):
    def impl(dti):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        A = bodo.hiframes.pd_index_ext.get_index_data(dti)
        n = len(A)
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(A, i):
                bodo.libs.array_kernels.setna(out_arr, i)
                continue
            val = A[i]
            ts = bodo.utils.conversion.box_if_dt64(val)
            out_arr[i] = bodo.hiframes.pd_timestamp_ext.is_leap_year(ts.year)
        return out_arr

    return impl


@overload_attribute(DatetimeIndexType, "date", jit_options={"cache": True})
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


@overload_method(
    DatetimeIndexType, "min", no_unliteral=True, jit_options={"cache": True}
)
def overload_datetime_index_min(dti, axis=None, skipna=True):
    dti_is_tz_aware = isinstance(
        dti.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
    )
    tz = dti.dtype.tz if dti_is_tz_aware else None
    # TODO skipna = False
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": None, "skipna": True}
    check_unsupported_args(
        "DatetimeIndex.min",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    def impl(dti, axis=None, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        in_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
        s = numba.cpython.builtins.get_type_max_value(numba.core.types.int64)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(in_arr)):
            if not bodo.libs.array_kernels.isna(in_arr, i):
                if dti_is_tz_aware:
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(
                        in_arr[i].tz_localize(None).value
                    )
                else:
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                s = min(s, val)
                count += 1

        if dti_is_tz_aware:
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count).tz_localize(
                tz
            )
        else:
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

    return impl


# TODO: refactor min/max
@overload_method(
    DatetimeIndexType, "max", no_unliteral=True, jit_options={"cache": True}
)
def overload_datetime_index_max(dti, axis=None, skipna=True):
    dti_is_tz_aware = isinstance(
        dti.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
    )
    tz = dti.dtype.tz if dti_is_tz_aware else None
    # TODO skipna = False
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": None, "skipna": True}
    check_unsupported_args(
        "DatetimeIndex.max",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    def impl(dti, axis=None, skipna=True):  # pragma: no cover
        numba.parfors.parfor.init_prange()
        in_arr = bodo.hiframes.pd_index_ext.get_index_data(dti)
        s = numba.cpython.builtins.get_type_min_value(numba.core.types.int64)
        count = 0
        for i in numba.parfors.parfor.internal_prange(len(in_arr)):
            if not bodo.libs.array_kernels.isna(in_arr, i):
                if dti_is_tz_aware:
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(
                        in_arr[i].tz_localize(None).value
                    )
                else:
                    val = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(in_arr[i])
                s = max(s, val)
                count += 1

        if dti_is_tz_aware:
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count).tz_localize(
                tz
            )
        else:
            return bodo.hiframes.pd_index_ext._dti_val_finalize(s, count)

    return impl


@overload_method(
    DatetimeIndexType, "tz_convert", no_unliteral=True, jit_options={"cache": True}
)
def overload_pd_datetime_tz_convert(A, tz):
    def impl(A, tz):
        return init_datetime_index(A._data.tz_convert(tz), A._name)

    return impl


@infer_getattr
class DatetimeIndexAttribute(AttributeTemplate):
    key = DatetimeIndexType

    def resolve_values(self, dt_index):
        return dt_index.data


@overload(pd.DatetimeIndex, no_unliteral=True, jit_options={"cache": True})
@overload(bd.DatetimeIndex, no_unliteral=True, jit_options={"cache": True})
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

    unsupported_args = {
        "freq": freq,
        "tz": tz,
        "normalize": normalize,
        "closed": closed,
        "ambiguous": ambiguous,
        "dayfirst": dayfirst,
        "yearfirst": yearfirst,
        "dtype": dtype,
        "copy": copy,
    }
    arg_defaults = {
        "freq": None,
        "tz": None,
        "normalize": False,
        "closed": None,
        "ambiguous": "raise",
        "dayfirst": False,
        "yearfirst": False,
        "dtype": None,
        "copy": False,
    }
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
        and rhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
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
        and lhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
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


# binop of DatetimeIndex and string
def gen_dti_str_binop_impl(op, is_lhs_dti):
    # is_arg1_dti: is the first argument DatetimeIndex and second argument str
    op_str = numba.core.utils.OPERATORS_TO_BUILTINS[op]
    func_text = "def bodo_gen_dti_str_binop(lhs, rhs):\n"
    if is_lhs_dti:
        func_text += "  dt_index, _str = lhs, rhs\n"
        comp = f"arr[i] {op_str} other"
    else:
        func_text += "  dt_index, _str = rhs, lhs\n"
        comp = f"other {op_str} arr[i]"
    func_text += "  arr = bodo.hiframes.pd_index_ext.get_index_data(dt_index)\n"
    func_text += "  l = len(arr)\n"
    func_text += "  other = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(_str)\n"
    func_text += "  S = bodo.libs.bool_arr_ext.alloc_bool_array(l)\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(l):\n"
    func_text += f"    S[i] = {comp}\n"
    func_text += "  return S\n"
    return bodo_exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, {}, __name__)


def overload_binop_dti_str(op):
    def overload_impl(lhs, rhs):
        if isinstance(lhs, DatetimeIndexType) and types.unliteral(rhs) == string_type:
            return gen_dti_str_binop_impl(op, True)
        if isinstance(rhs, DatetimeIndexType) and types.unliteral(lhs) == string_type:
            return gen_dti_str_binop_impl(op, False)

    return overload_impl


@overload(pd.Index, inline="always", no_unliteral=True, jit_options={"cache": True})
@overload(bd.Index, inline="always", no_unliteral=True, jit_options={"cache": True})
def pd_index_overload(data=None, dtype=None, copy=False, name=None, tupleize_cols=True):
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(data, "pandas.Index()")

    # Todo: support Categorical dtype, Interval dtype, Period dtype, MultiIndex (?)
    # Todo: Extension dtype (?)

    # unliteral e.g. Tuple(Literal[int](3), Literal[int](1)) to UniTuple(int64 x 2)
    # NOTE: unliteral of LiteralList is Poison type in Numba
    data = types.unliteral(data) if not isinstance(data, types.LiteralList) else data

    if not is_overload_none(dtype):
        elem_type = parse_dtype(dtype, "pandas.Index")
        # Specifies whether the dtype was provided
        dtype_provided = False
    else:
        elem_type = getattr(data, "dtype", None)
        dtype_provided = True

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
    elif (
        isinstance(data, DatetimeIndexType)
        or elem_type == types.NPDatetime("ns")
        or isinstance(elem_type, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype)
    ):

        def impl(
            data=None, dtype=None, copy=False, name=None, tupleize_cols=True
        ):  # pragma: no cover
            return pd.DatetimeIndex(data, name=name)

    # Timedelta index:
    elif (
        isinstance(data, TimedeltaIndexType)
        or elem_type == types.NPTimedelta("ns")
        or elem_type == bodo.types.pd_timedelta_type
    ):

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
        if (
            isinstance(
                elem_type,
                (
                    types.Integer,
                    types.Float,
                    types.Boolean,
                    bodo.types.TimeType,
                    bodo.types.Decimal128Type,
                ),
            )
            or elem_type == bodo.types.datetime_date_type
        ):
            if dtype_provided:

                def impl(
                    data=None, dtype=None, copy=False, name=None, tupleize_cols=True
                ):  # pragma: no cover
                    data_arr = bodo.utils.conversion.coerce_to_array(data)
                    return bodo.hiframes.pd_index_ext.init_numeric_index(data_arr, name)

            else:

                def impl(
                    data=None, dtype=None, copy=False, name=None, tupleize_cols=True
                ):  # pragma: no cover
                    data_arr = bodo.utils.conversion.coerce_to_array(data)
                    fixed_arr = bodo.utils.conversion.fix_arr_dtype(data_arr, elem_type)
                    return bodo.hiframes.pd_index_ext.init_numeric_index(
                        fixed_arr, name
                    )

        # String/Binary index:
        elif elem_type in [types.string, bytes_type]:

            def impl(
                data=None, dtype=None, copy=False, name=None, tupleize_cols=True
            ):  # pragma: no cover
                return bodo.hiframes.pd_index_ext.init_binary_str_index(
                    bodo.utils.conversion.coerce_to_array(data), name
                )

        # Categorical index:
        elif isinstance(elem_type, bodo.types.PDCategoricalDtype):
            if dtype_provided:

                def impl(
                    data=None, dtype=None, copy=False, name=None, tupleize_cols=True
                ):  # pragma: no cover
                    data_arr = bodo.utils.conversion.coerce_to_array(data)
                    return bodo.hiframes.pd_index_ext.init_categorical_index(
                        data_arr, name
                    )

            else:

                def impl(
                    data=None, dtype=None, copy=False, name=None, tupleize_cols=True
                ):  # pragma: no cover
                    data_arr = bodo.utils.conversion.coerce_to_array(data)
                    fixed_arr = bodo.utils.conversion.fix_arr_dtype(data_arr, elem_type)
                    return bodo.hiframes.pd_index_ext.init_categorical_index(
                        fixed_arr, name
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


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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


@overload(pd.date_range, inline="always", jit_options={"cache": True})
@overload(bd.date_range, inline="always", jit_options={"cache": True})
def pd_date_range_overload(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize=False,
    name=None,
    inclusive="both",
    unit=None,
):
    # TODO: check/handle other input
    # check unsupported, TODO: normalize, dayfirst, yearfirst, ...

    unsupported_args = {"tz": tz, "normalize": normalize, "inclusive": inclusive}
    arg_defaults = {"tz": None, "normalize": False, "inclusive": "both"}
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

    func_text = "def bodo_pd_date_range_overload(start=None, end=None, periods=None, freq=None, tz=None, normalize=False, name=None, inclusive='both', unit=None):\n"

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
                "at least 'start' or 'end' should be specified if a 'period' is given."
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

    return bodo_exec(func_text, {"bodo": bodo, "np": np, "pd": pd}, {}, __name__)


@overload(pd.timedelta_range, no_unliteral=True, jit_options={"cache": True})
@overload(bd.timedelta_range, no_unliteral=True, jit_options={"cache": True})
def pd_timedelta_range_overload(
    start=None,
    end=None,
    periods=None,
    freq=None,
    name=None,
    closed=None,
    unit=None,
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
        unit=None,
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

        S = bodo.utils.conversion.convert_to_td64ns(arr)
        return bodo.hiframes.pd_index_ext.init_timedelta_index(S, name)

    return f


@overload_method(
    DatetimeIndexType,
    "isocalendar",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_pd_timestamp_isocalendar(idx):
    __col_name_meta_value_pd_timestamp_isocalendar = ColNamesMetaType(
        ("year", "week", "day")
    )

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
            (years, weeks, days), idx, __col_name_meta_value_pd_timestamp_isocalendar
        )

    return impl


# ------------------------------ Timedelta ---------------------------


# similar to DatetimeIndex
class TimedeltaIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """Temporary type class for TimedeltaIndex objects."""

    def __init__(self, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        # TODO: support other properties like unit/freq?
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        # NOTE: data array can have flags like readonly
        self.data = (
            types.Array(numba.core.types.NPTimedelta("ns"), 1, "C")
            if data is None
            else data
        )
        super().__init__(name=f"TimedeltaIndexType({name_typ}, {self.data})")

    ndim = 1

    def copy(self):
        return TimedeltaIndexType(self.name_typ, self.data)

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
        # The underlying array is a timedelta64, but the data is
        # (and should be) boxed as a pd.Timedelta
        return bodo.utils.typing.BodoArrayIterator(self, bodo.types.pd_timedelta_type)

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
            ("data", fe_type.data),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(types.NPTimedelta("ns"), types.int64)),
        ]
        super().__init__(dmm, fe_type, members)


@typeof_impl.register(pd.TimedeltaIndex)
def typeof_timedelta_index(val, c):
    # keep string literal value in type since reset_index() may need it
    return TimedeltaIndexType(get_val_type_maybe_str_literal(val.name))


@box(TimedeltaIndexType)
def box_timedelta_index(typ, val, c):
    """"""
    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)

    timedelta_index = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder, val
    )

    c.context.nrt.incref(c.builder, typ.data, timedelta_index.data)
    arr_obj = c.pyapi.from_native_value(typ.data, timedelta_index.data, c.env_manager)
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

    # Call A.values.astype('timedelta64[ns]', copy=False)
    data_obj_us = c.pyapi.object_getattr_string(val, "values")
    data_obj_us_astype = c.pyapi.object_getattr_string(data_obj_us, "astype")
    dtype_obj = c.pyapi.string_from_constant_string("timedelta64[ns]")
    false_obj = c.pyapi.bool_from_bool(c.context.get_constant(types.bool_, False))
    args = c.pyapi.tuple_pack([dtype_obj])
    kws = c.pyapi.dict_pack(
        [
            ("copy", false_obj),
        ]
    )
    data_obj = c.pyapi.call(data_obj_us_astype, args, kws)
    c.pyapi.decref(data_obj_us)
    c.pyapi.decref(data_obj_us_astype)
    c.pyapi.decref(dtype_obj)
    c.pyapi.decref(false_obj)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)

    data = c.pyapi.to_native_value(typ.data, data_obj).value
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value
    c.pyapi.decref(data_obj)
    c.pyapi.decref(name_obj)

    # create index struct
    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    index_val.data = data
    index_val.name = name
    # create empty dict for get_loc hashmap
    dtype = types.NPTimedelta("ns")
    _is_error, ind_dict = c.pyapi.call_jit_code(
        lambda: numba.typed.Dict.empty(dtype, types.int64),
        types.DictType(dtype, types.int64)(),
        [],
    )
    index_val.dict = ind_dict
    return NativeValue(index_val._getvalue())


@intrinsic(prefer_literal=True)
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
        dtype = types.NPTimedelta("ns")
        timedelta_index.dict = context.compile_internal(
            builder,
            lambda: numba.typed.Dict.empty(dtype, types.int64),
            types.DictType(dtype, types.int64)(),
            [],
        )  # pragma: no cover

        return timedelta_index._getvalue()

    ret_typ = TimedeltaIndexType(name, data)
    sig = signature(ret_typ, data, name)
    return sig, codegen


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_index_ext_init_timedelta_index = (
    init_index_equiv
)


@infer_getattr
class TimedeltaIndexAttribute(AttributeTemplate):
    key = TimedeltaIndexType

    def resolve_values(self, ary):
        return ary.data

    # TODO: support pd.Timedelta
    # @bound_function("timedelta_index.max", no_unliteral=True)
    # def resolve_max(self, ary, args, kws):
    #     assert not kws
    #     return signature(pd_timestamp_tz_naive_type, *args)

    # @bound_function("timedelta_index.min", no_unliteral=True)
    # def resolve_min(self, ary, args, kws):
    #     assert not kws
    #     return signature(pd_timestamp_tz_naive_type, *args)


make_attribute_wrapper(TimedeltaIndexType, "data", "_data")
make_attribute_wrapper(TimedeltaIndexType, "name", "_name")
make_attribute_wrapper(TimedeltaIndexType, "dict", "_dict")


@overload_method(
    TimedeltaIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_timedelta_index_copy(A, name=None, deep=False, dtype=None, names=None):
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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


@overload_method(
    TimedeltaIndexType,
    "min",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_timedelta_index_min(tdi, axis=None, skipna=True):
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": None, "skipna": True}
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


@overload_method(
    TimedeltaIndexType,
    "max",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_timedelta_index_max(tdi, axis=None, skipna=True):
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": None, "skipna": True}
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
    func_text = "def bodo_gen_tdi_field(tdi):\n"
    func_text += "    numba.parfors.parfor.init_prange()\n"
    func_text += "    A = bodo.hiframes.pd_index_ext.get_index_data(tdi)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(tdi)\n"
    func_text += "    n = len(A)\n"
    # days field returns int64 but others return int32
    # https://github.com/pandas-dev/pandas/blob/0f437949513225922d851e9581723d82120684a6/pandas/_libs/tslibs/fields.pyx#L562
    # https://github.com/pandas-dev/pandas/blob/0f437949513225922d851e9581723d82120684a6/pandas/_libs/tslibs/fields.pyx#L509
    dtype_str = "np.int64" if field == "days" else "np.int32"
    func_text += f"    S = np.empty(n, {dtype_str})\n"
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
    return bodo_exec(func_text, {"numba": numba, "np": np, "bodo": bodo}, {}, __name__)


def _install_tdi_field_overload(field):
    """get field implementation and call overload_attribute()
    NOTE: This has to be a separate function to avoid unexpected free variable updates
    """
    impl = gen_tdi_field_impl(field)
    overload_attribute(TimedeltaIndexType, field)(lambda tdi: impl)


def _install_tdi_time_fields():
    for field in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
        _install_tdi_field_overload(field)


_install_tdi_time_fields()


@overload(pd.TimedeltaIndex, no_unliteral=True, jit_options={"cache": True})
@overload(bd.TimedeltaIndex, no_unliteral=True, jit_options={"cache": True})
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

    unsupported_args = {
        "unit": unit,
        "freq": freq,
        "dtype": dtype,
        "copy": copy,
    }

    arg_defaults = {
        "unit": None,
        "freq": None,
        "dtype": None,
        "copy": False,
    }

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
class RangeIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """type class for pd.RangeIndex() objects."""

    def __init__(self, name_typ=None):
        if name_typ is None:
            name_typ = types.none
        self.name_typ = name_typ
        super().__init__(name=f"RangeIndexType({name_typ})")

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
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(RangeIndexType, "start", "_start")
make_attribute_wrapper(RangeIndexType, "stop", "_stop")
make_attribute_wrapper(RangeIndexType, "step", "_step")
make_attribute_wrapper(RangeIndexType, "name", "_name")


@overload_method(RangeIndexType, "copy", no_unliteral=True, jit_options={"cache": True})
def overload_range_index_copy(A, name=None, deep=False, dtype=None, names=None):
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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
    class_obj = c.pyapi.import_module(mod_name)
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


@intrinsic(prefer_literal=True)
def init_range_index(typingctx, start, stop, step, name=None):
    """Create RangeIndex object"""
    name = types.none if name is None else name

    # Compile time check of step = 0
    literal_zero = is_overload_constant_int(step) and get_overload_const_int(step) == 0

    def codegen(context, builder, signature, args):
        assert len(args) == 4

        if literal_zero:
            raise_bodo_error("Step must not be zero")

        step_zero = cgutils.is_scalar_zero(builder, args[2])
        pyapi = context.get_python_api(builder)

        # Runtime check of step = 0
        with builder.if_then(step_zero):
            pyapi.err_format("PyExc_ValueError", "Step must not be zero")
            val = context.get_constant(types.int32, -1)
            builder.ret(val)

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


@overload(
    pd.RangeIndex, no_unliteral=True, inline="always", jit_options={"cache": True}
)
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

    func_text = "def bodo_pd_range_index(start=None, stop=None, step=None, dtype=None, copy=False, name=None):\n"
    func_text += f"  return init_range_index({_start}, {_stop}, {_step}, name)\n"
    return bodo_exec(func_text, {"init_range_index": init_range_index}, {}, __name__)


@overload(
    pd.CategoricalIndex, no_unliteral=True, inline="always", jit_options={"cache": True}
)
def categorical_index_overload(
    data=None, categories=None, ordered=None, dtype=None, copy=False, name=None
):
    raise BodoError("pd.CategoricalIndex() initializer not yet supported.")


@overload_attribute(RangeIndexType, "start", jit_options={"cache": True})
def rangeIndex_get_start(ri):
    def impl(ri):  # pragma: no cover
        return ri._start

    return impl


@overload_attribute(RangeIndexType, "stop", jit_options={"cache": True})
def rangeIndex_get_stop(ri):
    def impl(ri):  # pragma: no cover
        return ri._stop

    return impl


@overload_attribute(RangeIndexType, "step", jit_options={"cache": True})
def rangeIndex_get_step(ri):
    def impl(ri):  # pragma: no cover
        return ri._step

    return impl


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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


@overload(len, no_unliteral=True, jit_options={"cache": True})
def overload_range_len(r):
    if isinstance(r, RangeIndexType):
        # TODO: test
        return lambda r: max(0, -(-(r._stop - r._start) // r._step))  # pragma: no cover


# ---------------- PeriodIndex -------------------


# Simple type for PeriodIndex for now, freq is saved as a constant string
class PeriodIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """type class for pd.PeriodIndex. Contains frequency as constant string"""

    def __init__(self, freq, name_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.freq = freq
        self.name_typ = name_typ
        super().__init__(name=f"PeriodIndexType({freq}, {name_typ})")

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
            ("data", bodo.types.IntegerArrayType(types.int64)),
            ("name", fe_type.name_typ),
            ("dict", types.DictType(types.int64, types.int64)),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(PeriodIndexType, "data", "_data")
make_attribute_wrapper(PeriodIndexType, "name", "_name")
make_attribute_wrapper(PeriodIndexType, "dict", "_dict")


@overload_method(
    PeriodIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_period_index_copy(A, name=None, deep=False, dtype=None, names=None):
    freq = A.freq
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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


@intrinsic(prefer_literal=True)
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
    class_obj = c.pyapi.import_module(mod_name)

    index_val = cgutils.create_struct_proxy(typ)(c.context, c.builder, val)

    c.context.nrt.incref(
        c.builder, bodo.types.IntegerArrayType(types.int64), index_val.data
    )
    data_obj = c.pyapi.from_native_value(
        bodo.types.IntegerArrayType(types.int64), index_val.data, c.env_manager
    )
    c.context.nrt.incref(c.builder, typ.name_typ, index_val.name)
    name_obj = c.pyapi.from_native_value(typ.name_typ, index_val.name, c.env_manager)
    freq_obj = c.pyapi.string_from_constant_string(typ.freq)

    # call pd.PeriodIndex(ordinal=data, name=name, freq=freq)
    args = c.pyapi.tuple_pack([])
    kws = c.pyapi.dict_pack(
        [("ordinals", data_obj), ("name", name_obj), ("freq", freq_obj)]
    )
    const_call = c.pyapi.object_getattr_string(class_obj, "PeriodIndex")
    ord_call = c.pyapi.object_getattr_string(const_call, "from_ordinals")
    index_obj = c.pyapi.call(ord_call, args, kws)

    c.pyapi.decref(data_obj)
    c.pyapi.decref(name_obj)
    c.pyapi.decref(freq_obj)
    c.pyapi.decref(class_obj)
    c.pyapi.decref(const_call)
    c.pyapi.decref(ord_call)
    c.pyapi.decref(args)
    c.pyapi.decref(kws)
    c.context.nrt.decref(c.builder, typ, val)
    return index_obj


@unbox(PeriodIndexType)
def unbox_period_index(typ, val, c):
    # get data and name attributes
    arr_typ = bodo.types.IntegerArrayType(types.int64)
    asi8_obj = c.pyapi.object_getattr_string(val, "asi8")
    isna_obj = c.pyapi.call_method(val, "isna", ())
    name_obj = c.pyapi.object_getattr_string(val, "name")
    name = c.pyapi.to_native_value(typ.name_typ, name_obj).value

    mod_name = c.context.insert_const_string(c.builder.module, "pandas")
    pd_class_obj = c.pyapi.import_module(mod_name)
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


class CategoricalIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """data type for CategoricalIndex values"""

    def __init__(self, data, name_typ=None):
        from bodo.hiframes.pd_categorical_ext import CategoricalArrayType

        assert isinstance(data, CategoricalArrayType), (
            "CategoricalIndexType expects CategoricalArrayType"
        )
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        self.data = data
        super().__init__(
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
        super().__init__(dmm, fe_type, members)


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
    pd_class_obj = c.pyapi.import_module(mod_name)

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


@intrinsic(prefer_literal=True)
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


@overload_method(
    CategoricalIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_categorical_index_copy(A, name=None, deep=False, dtype=None, names=None):
    err_str = idx_typ_to_format_str_map[CategoricalIndexType].format("copy()")
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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


class IntervalIndexType(types.ArrayCompatible, SingleIndexType):
    """data type for IntervalIndex values"""

    def __init__(self, data, name_typ=None):
        from bodo.libs.interval_arr_ext import IntervalArrayType

        assert isinstance(data, IntervalArrayType), (
            "IntervalIndexType expects IntervalArrayType"
        )
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        self.data = data
        super().__init__(name=f"IntervalIndexType(data={self.data}, name={name_typ})")

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
        super().__init__(dmm, fe_type, members)


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
    pd_class_obj = c.pyapi.import_module(mod_name)

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


@intrinsic(prefer_literal=True)
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


# Represents numeric indices (excluding RangeIndex)
class NumericIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """type class for pd.Index objects with numeric dtypes."""

    def __init__(self, dtype, name_typ=None, data=None):
        name_typ = types.none if name_typ is None else name_typ
        self.dtype = dtype
        self.name_typ = name_typ
        data = dtype_to_array_type(dtype) if data is None else data
        self.data = data
        super().__init__(name=f"NumericIndexType({dtype}, {name_typ}, {data})")

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
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(NumericIndexType, "data", "_data")
make_attribute_wrapper(NumericIndexType, "name", "_name")
make_attribute_wrapper(NumericIndexType, "dict", "_dict")


@overload_method(
    NumericIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_numeric_index_copy(A, name=None, deep=False, dtype=None, names=None):
    err_str = idx_typ_to_format_str_map[NumericIndexType].format("copy()")
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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
    class_obj = c.pyapi.import_module(mod_name)
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


@intrinsic(prefer_literal=True)
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


# ---------------- StringIndex -------------------


# represents string index, which doesn't have direct Pandas type
# pd.Index() infers string
class StringIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """type class for pd.Index() objects with 'string' as inferred_dtype."""

    def __init__(self, name_typ=None, data_typ=None):
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = string_array_type if data_typ is None else data_typ
        super().__init__(
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
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(StringIndexType, "data", "_data")
make_attribute_wrapper(StringIndexType, "name", "_name")
make_attribute_wrapper(StringIndexType, "dict", "_dict")


# ---------------- BinaryIndex -------------------


# represents binary index, which doesn't have direct Pandas type
# pd.Index() infers binary
# Largely copied from the StringIndexType class
class BinaryIndexType(types.IterableType, types.ArrayCompatible, SingleIndexType):
    """type class for pd.Index() objects with 'binary' as inferred_dtype."""

    def __init__(self, name_typ=None, data_typ=None):
        # data_typ is added just for compatibility with StringIndexType
        assert data_typ is None or data_typ == binary_array_type, (
            "data_typ must be binary_array_type"
        )
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        # Add a .data field for consistency with other index types
        self.data = binary_array_type
        super().__init__(name=f"BinaryIndexType({name_typ})")

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
        super().__init__(dmm, fe_type, members)


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
    class_obj = c.pyapi.import_module(mod_name)

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


@intrinsic(prefer_literal=True)
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

    func_text = "def bodo_get_binary_str_codegen(context, builder, signature, args):\n"
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

    return bodo_exec(
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
        {},
        __name__,
    )


@overload_method(
    BinaryIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_binary_string_index_copy(A, name=None, deep=False, dtype=None, names=None):
    typ = type(A)

    err_str = idx_typ_to_format_str_map[typ].format("copy()")
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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


@overload_attribute(BinaryIndexType, "name", jit_options={"cache": True})
@overload_attribute(StringIndexType, "name", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "name", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "name", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "name", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "name", jit_options={"cache": True})
@overload_attribute(NumericIndexType, "name", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "name", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "name", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "name", jit_options={"cache": True})
def Index_get_name(i):
    def impl(i):  # pragma: no cover
        return i._name

    return impl


@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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
    if arr_typ == bodo.types.binary_array_type:
        return BinaryIndexType(name_typ)

    assert isinstance(
        arr_typ,
        (
            types.Array,
            IntegerArrayType,
            FloatingArrayType,
            bodo.types.CategoricalArrayType,
            bodo.types.DecimalArrayType,
            bodo.types.TimeArrayType,
            bodo.types.DatetimeArrayType,
        ),
    ) or arr_typ in (
        bodo.types.datetime_date_array_type,
        bodo.types.boolean_array_type,
        bodo.types.timedelta_array_type,
    ), f"Converting array type {arr_typ} to index not supported"

    # TODO: Pandas keeps datetime_date Index as a generic Index(, dtype=object)
    # Fix this implementation to match.
    if arr_typ.dtype == types.NPDatetime("ns"):
        return DatetimeIndexType(name_typ)

    if isinstance(arr_typ, bodo.types.DatetimeArrayType):
        return DatetimeIndexType(name_typ, arr_typ)

    # categorical array
    if isinstance(arr_typ, bodo.types.CategoricalArrayType):
        return CategoricalIndexType(arr_typ, name_typ)

    if arr_typ.dtype in (types.NPTimedelta("ns"), bodo.types.pd_timedelta_type):
        return TimedeltaIndexType(name_typ, arr_typ)

    if (
        isinstance(
            arr_typ.dtype,
            (types.Integer, types.Float, types.Boolean, bodo.types.TimeType),
        )
        or arr_typ == bodo.types.datetime_date_array_type
    ):
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


def _verify_setop_compatible(func_name, I, other):
    """Verifies that index I and value other can be combined with the
    set operation provided.

    Args:
        func_name (string): union, intersection, difference or symmetric difference
        I (pd.Index): the Index that is to be combined with other
        other (any): the value whose elements are trying to be combined with
        I using a set operation

    Raises:
        BodoError: if other is unsupported or incompatible with I for set operations
    """

    if not is_pd_index_type(other) and not isinstance(other, (SeriesType, types.Array)):
        raise BodoError(
            f"pd.Index.{func_name}(): unsupported type for argument other: {other}"
        )

    # Verify that the two values can be combined
    # TODO: make more permissive, potentially with get_common_scalar_dtype [BE-3017]
    t1 = I.dtype if not isinstance(I, RangeIndexType) else types.int64
    t2 = other.dtype if not isinstance(other, RangeIndexType) else types.int64
    if t1 != t2:
        raise BodoError(f"Index.{func_name}(): incompatible types {t1} and {t2}")


@overload_method(
    NumericIndexType, "union", inline="always", jit_options={"cache": True}
)
@overload_method(StringIndexType, "union", inline="always", jit_options={"cache": True})
@overload_method(BinaryIndexType, "union", inline="always", jit_options={"cache": True})
@overload_method(
    DatetimeIndexType, "union", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "union", inline="always", jit_options={"cache": True}
)
@overload_method(RangeIndexType, "union", inline="always", jit_options={"cache": True})
def overload_index_union(I, other, sort=None):
    """Adds support for a modified version of pd.Index.union() on tagged Index types.

    Args:
        I (pd.Index): the first index in the union
        other (iterable with matching type): array/Index/Series with values that
        are union-ed with I. Must have an underlying type that can be
        reconciled with the type of I.
        sort (boolean, optional): not supported. Defaults to None.

    Raises:
        BodoError: if I and other are not compatible for a union

    Returns:
        pd.Index: the elements of both indices, in the order that they first
        ocurred, without any duplicates.
    """
    unsupported_args = {"sort": sort}
    default_args = {"sort": None}
    check_unsupported_args(
        "Index.union",
        unsupported_args,
        default_args,
        package_name="pandas",
        module_name="Index",
    )

    _verify_setop_compatible("union", I, other)

    constructor = (
        get_index_constructor(I)
        if not isinstance(I, RangeIndexType)
        else init_numeric_index
    )

    # Recycles logic from bodo.libs.array_kernels.overload_union1d: concatenates
    # the two underlying arrays and obtains the unique values from the result
    def impl(I, other, sort=None):  # pragma: no cover
        A1 = bodo.utils.conversion.coerce_to_array(I)
        A2 = bodo.utils.conversion.coerce_to_array(other)
        merged_array = bodo.libs.array_kernels.concat((A1, A2))
        unique_array = bodo.libs.array_kernels.unique(merged_array)
        return constructor(unique_array, None)

    return impl


@overload_method(
    NumericIndexType, "intersection", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "intersection", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "intersection", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "intersection", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "intersection", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "intersection", inline="always", jit_options={"cache": True}
)
def overload_index_intersection(I, other, sort=None):
    """Adds support for a modified version of pd.Index.intersection() on tagged Index types.

    Args:
        I (pd.Index): the first index in the intersection
        other (iterable with matching type): array/Index/Series with values that
        are intersection-ed with I. Must have an underlying type that can be
        reconciled with the type of I.
        sort (boolean, optional): not supported. Defaults to None.

    Raises:
        BodoError: if I and other are not compatible for a intersection

    Returns:
        pd.Index: the elements of both indices, in sorted order.
    """
    unsupported_args = {"sort": sort}
    default_args = {"sort": None}
    check_unsupported_args(
        "Index.intersection",
        unsupported_args,
        default_args,
        package_name="pandas",
        module_name="Index",
    )

    _verify_setop_compatible("intersection", I, other)

    constructor = (
        get_index_constructor(I)
        if not isinstance(I, RangeIndexType)
        else init_numeric_index
    )

    # Recycles logic from bodo.libs.array_kernels.overload_intersect1d: obtains
    # the unique values from each underlying array, combines and sorts them,
    # and keeps each value that is the same as the value after it (since that
    # means it appeared in both of the unique arrays)
    def impl(I, other, sort=None):  # pragma: no cover
        A1 = bodo.utils.conversion.coerce_to_array(I)
        A2 = bodo.utils.conversion.coerce_to_array(other)
        unique_A1 = bodo.libs.array_kernels.unique(A1)
        unique_A2 = bodo.libs.array_kernels.unique(A2)
        merged_array = bodo.libs.array_kernels.concat((unique_A1, unique_A2))
        sorted_array = pd.Series(merged_array).sort_values().values
        mask = bodo.libs.array_kernels.intersection_mask(sorted_array)
        return constructor(sorted_array[mask], None)

    return impl


@overload_method(
    NumericIndexType, "difference", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "difference", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "difference", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "difference", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "difference", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "difference", inline="always", jit_options={"cache": True}
)
def overload_index_difference(I, other, sort=None):
    """Adds support for a modified version of pd.Index.difference() on tagged Index types.

    Args:
        I (pd.Index): the first index in the difference
        other (iterable with matching type): array/Index/Series with values that
        are differenced from I. Must have an underlying type that can be
        reconciled with the type of I.
        sort (boolean, optional): not supported. Defaults to None.

    Raises:
        BodoError: if I and other are not compatible for a difference

    Returns:
        pd.Index: the elements I that are not in other, in sorted order.
    """
    unsupported_args = {"sort": sort}
    default_args = {"sort": None}
    check_unsupported_args(
        "Index.difference",
        unsupported_args,
        default_args,
        package_name="pandas",
        module_name="Index",
    )

    _verify_setop_compatible("difference", I, other)

    constructor = (
        get_index_constructor(I)
        if not isinstance(I, RangeIndexType)
        else init_numeric_index
    )

    # Modifies logic from overload_index_isin: obtains the unique values from
    # the LHS, uses the isin utility to create a mask of all values from the
    # RHS that are in the LHS array, uses the inverse mask to drop those elems
    def impl(I, other, sort=None):  # pragma: no cover
        # setting use_nullable_array since array_isin expects same array types
        A1 = bodo.utils.conversion.coerce_to_array(I, use_nullable_array=True)
        A2 = bodo.utils.conversion.coerce_to_array(other, use_nullable_array=True)
        # Obtains the unique values from A2 for consistency with symmetric_difference
        # TODO: investigate whether this is better or worse for performance
        # than just calling array_isin with A2.
        unique_A1 = bodo.libs.array_kernels.unique(A1)
        unique_A2 = bodo.libs.array_kernels.unique(A2)
        mask = bodo.libs.bool_arr_ext.alloc_false_bool_array(len(unique_A1))
        bodo.libs.array.array_isin(mask, unique_A1, unique_A2, False)
        return constructor(unique_A1[~mask], None)

    return impl


@overload_method(
    NumericIndexType,
    "symmetric_difference",
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "symmetric_difference",
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "symmetric_difference",
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "symmetric_difference",
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "symmetric_difference",
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType, "symmetric_difference", inline="always", jit_options={"cache": True}
)
def overload_index_symmetric_difference(I, other, result_name=None, sort=None):
    """Adds support for a modified version of pd.Index.difference() on tagged Index types.

    Args:
        I (pd.Index): the first index in the symmetric_difference
        other (iterable with matching type): array/Index/Series with values that
        are symmetric_differenced from I. Must have an underlying type that can be
        reconciled with the type of I.
        result_name (string, optional): not supported. Defaults to None.
        sort (boolean, optional): not supported. Defaults to None.

    Raises:
        BodoError: if I and other are not compatible for a symmetric_difference

    Returns:
        pd.Index: the elements I that are not in other, in sorted order.
    """
    unsupported_args = {"result_name": result_name, "sort": sort}
    default_args = {"result_name": None, "sort": None}
    check_unsupported_args(
        "Index.symmetric_difference",
        unsupported_args,
        default_args,
        package_name="pandas",
        module_name="Index",
    )

    _verify_setop_compatible("symmetric_difference", I, other)

    constructor = (
        get_index_constructor(I)
        if not isinstance(I, RangeIndexType)
        else init_numeric_index
    )

    # Modifies logic from overload_index_isin: obtains the unique values from
    # the each array, uses the isin utility to create a mask of all values
    # from each array that are in the other, uses the inverse masks to drop
    # those elems from each array and combines the results
    def impl(I, other, result_name=None, sort=None):  # pragma: no cover
        # setting use_nullable_array since array_isin expects same array types
        A1 = bodo.utils.conversion.coerce_to_array(I, use_nullable_array=True)
        A2 = bodo.utils.conversion.coerce_to_array(other, use_nullable_array=True)
        unique_A1 = bodo.libs.array_kernels.unique(A1)
        unique_A2 = bodo.libs.array_kernels.unique(A2)
        mask1 = bodo.libs.bool_arr_ext.alloc_false_bool_array(len(unique_A1))
        mask2 = bodo.libs.bool_arr_ext.alloc_false_bool_array(len(unique_A2))
        bodo.libs.array.array_isin(mask1, unique_A1, unique_A2, False)
        bodo.libs.array.array_isin(mask2, unique_A2, unique_A1, False)
        combined_arr = bodo.libs.array_kernels.concat(
            (unique_A1[~mask1], unique_A2[~mask2])
        )
        return constructor(combined_arr, None)

    return impl


# TODO: test
@overload_method(RangeIndexType, "take", no_unliteral=True, jit_options={"cache": True})
@overload_method(
    NumericIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "take", no_unliteral=True, jit_options={"cache": True}
)
def overload_index_take(I, indices, axis=0, allow_fill=True, fill_value=None):
    unsupported_args = {
        "axis": axis,
        "allow_fill": allow_fill,
        "fill_value": fill_value,
    }
    default_args = {"axis": 0, "allow_fill": True, "fill_value": None}
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


@overload(_init_engine, jit_options={"cache": True})
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
    elif (
        isinstance(I, TimedeltaIndexType) and I.data == bodo.types.timedelta_array_type
    ):

        def impl(I, ban_unique=True):  # pragma: no cover
            if len(I) > 0 and not I._dict:
                arr = bodo.utils.conversion.coerce_to_array(I)
                for i in range(len(arr)):
                    if not bodo.libs.array_kernels.isna(arr, i):
                        val = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                            arr[i].value
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


@overload(operator.contains, no_unliteral=True, jit_options={"cache": True})
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
            key = bodo.utils.conversion.unbox_if_tz_naive_timestamp(val)
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
        key = bodo.utils.conversion.unbox_if_tz_naive_timestamp(val)
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


@overload_method(
    RangeIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    NumericIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "get_loc", no_unliteral=True, jit_options={"cache": True}
)
def overload_index_get_loc(I, key, method=None, tolerance=None):
    """simple get_loc implementation intended for cases with small Index like
    df.columns.get_loc(c). Only supports Index with unique values (scalar return).
    TODO(ehsan): use a proper hash engine like Pandas inside Index objects
    """
    unsupported_args = {"method": method, "tolerance": tolerance}
    arg_defaults = {"method": None, "tolerance": None}
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
    if key == pd_timestamp_tz_naive_type:
        key = bodo.types.datetime64ns
    if key == pd_timedelta_type:
        key = bodo.types.timedelta64ns

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
        key = bodo.utils.conversion.unbox_if_tz_naive_timestamp(key)
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
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfors.parfor.internal_prange(n):
                    out_arr[i] = not cond_when_isna
                return out_arr

            return impl

        func_text = (
            "def bodo_index_isna_specific_method(I):\n"
            "    numba.parfors.parfor.init_prange()\n"
            "    arr = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
            "    n = len(arr)\n"
            "    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)\n"
            "    for i in numba.parfors.parfor.internal_prange(n):\n"
            f"       out_arr[i] = {'' if cond_when_isna else 'not '}"
            "bodo.libs.array_kernels.isna(arr, i)\n"
            "    return out_arr\n"
        )
        return bodo_exec(
            func_text, {"bodo": bodo, "np": np, "numba": numba}, {}, __name__
        )

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


def _install_isna_impl(overload_type, overload_name):
    """install isna call for Index type
    NOTE: This has to be a separate function to avoid unexpected free variable updates
    """
    overload_impl = create_isna_specific_method(overload_name)
    overload_method(
        overload_type,
        overload_name,
        no_unliteral=True,
        inline="always",
    )(overload_impl)


def _install_isna_specific_methods():
    for overload_type in isna_overload_types:
        for overload_name in isna_specific_methods:
            _install_isna_impl(overload_type, overload_name)


_install_isna_specific_methods()


@overload_attribute(RangeIndexType, "values", jit_options={"cache": True})
@overload_attribute(NumericIndexType, "values", jit_options={"cache": True})
@overload_attribute(StringIndexType, "values", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "values", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "values", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "values", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "values", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "values", jit_options={"cache": True})
def overload_values(I):
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(I, "Index.values")
    return lambda I: bodo.utils.conversion.coerce_to_array(I)  # pragma: no cover


@overload(len, no_unliteral=True, jit_options={"cache": True})
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


@overload(len, no_unliteral=True, jit_options={"cache": True})
def overload_multi_index_len(I):
    if isinstance(I, MultiIndexType):
        return lambda I: len(
            bodo.hiframes.pd_index_ext.get_index_data(I)[0]
        )  # pragma: no cover


@overload_attribute(DatetimeIndexType, "shape", jit_options={"cache": True})
@overload_attribute(NumericIndexType, "shape", jit_options={"cache": True})
@overload_attribute(StringIndexType, "shape", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "shape", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "shape", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "shape", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "shape", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "shape", jit_options={"cache": True})
def overload_index_shape(s):
    return lambda s: (
        len(bodo.hiframes.pd_index_ext.get_index_data(s)),
    )  # pragma: no cover


@overload_attribute(RangeIndexType, "shape", jit_options={"cache": True})
def overload_range_index_shape(s):
    return lambda s: (len(s),)  # pragma: no cover


@overload_attribute(MultiIndexType, "shape", jit_options={"cache": True})
def overload_index_shape(s):
    return lambda s: (
        len(bodo.hiframes.pd_index_ext.get_index_data(s)[0]),
    )  # pragma: no cover


@overload_attribute(
    NumericIndexType,
    "is_monotonic_increasing",
    inline="always",
    jit_options={"cache": True},
)
@overload_attribute(
    RangeIndexType,
    "is_monotonic_increasing",
    inline="always",
    jit_options={"cache": True},
)
@overload_attribute(
    DatetimeIndexType,
    "is_monotonic_increasing",
    inline="always",
    jit_options={"cache": True},
)
@overload_attribute(
    TimedeltaIndexType,
    "is_monotonic_increasing",
    inline="always",
    jit_options={"cache": True},
)
def overload_index_is_montonic(I):
    """
    Implementation of is_monotonic_increasing attributes for Int64Index,
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


@overload_attribute(
    NumericIndexType,
    "is_monotonic_decreasing",
    inline="always",
    jit_options={"cache": True},
)
@overload_attribute(
    RangeIndexType,
    "is_monotonic_decreasing",
    inline="always",
    jit_options={"cache": True},
)
@overload_attribute(
    DatetimeIndexType,
    "is_monotonic_decreasing",
    inline="always",
    jit_options={"cache": True},
)
@overload_attribute(
    TimedeltaIndexType,
    "is_monotonic_decreasing",
    inline="always",
    jit_options={"cache": True},
)
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


@overload_method(
    NumericIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    PeriodIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "duplicated",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_index_duplicated(I, keep="first"):
    """
    Implementation of Index.duplicated() for all supported index types.
    """

    if isinstance(I, RangeIndexType):

        def impl(I, keep="first"):  # pragma: no cover
            return bodo.libs.bool_arr_ext.alloc_false_bool_array(len(I))

        return impl

    def impl(I, keep="first"):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        out_arr = bodo.libs.array_kernels.duplicated((arr,))
        return out_arr

    return impl


@overload_method(
    NumericIndexType,
    "any",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "any",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "any",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "any",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_any(I):
    if isinstance(I, RangeIndexType):

        def impl(I):  # pragma: no cover
            # Must not be empty, and if the start is 0 then its length must be > 1
            return len(I) > 0 and (I._start != 0 or len(I) > 1)

        return impl

    def impl(I):  # pragma: no cover
        A = bodo.hiframes.pd_index_ext.get_index_data(I)
        return bodo.libs.array_ops.array_op_any(A)

    return impl


@overload_method(
    NumericIndexType,
    "all",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "all",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "all",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "all",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_all(I):
    if isinstance(I, RangeIndexType):

        def impl(I):  # pragma: no cover
            # Must be empty or not contain zero
            return (
                # If the range is empty, then it does not contain zero
                len(I) == 0
                or
                # If the step is positive and 0 is not between start and stop,
                # then it does not contain zero
                (I._step > 0 and (I._start > 0 or I._stop <= 0))
                or
                # If the step is negative and 0 is not between stop and start,
                # then it does not contain zero
                (I._step < 0 and (I._start < 0 or I._stop >= 0))
                or
                # If the start is not a multiple of the step, then it does not contain zero
                (I._start % I._step) != 0
            )

        return impl

    def impl(I):  # pragma: no cover
        A = bodo.hiframes.pd_index_ext.get_index_data(I)
        return bodo.libs.array_ops.array_op_all(A)

    return impl


@overload_method(
    RangeIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    NumericIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    PeriodIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "drop_duplicates",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_drop_duplicates(I, keep="first"):
    """Overload `Index.drop_duplicates` method for all index types."""
    unsupported_args = {"keep": keep}
    arg_defaults = {"keep": "first"}
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
        "def bodo_index_drop_duplicates(I, keep='first'):\n"
        "    data = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
        "    arr = bodo.libs.array_kernels.drop_duplicates_array(data)\n"
        "    name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    )
    if isinstance(I, PeriodIndexType):
        func_text += f"    return bodo.hiframes.pd_index_ext.init_period_index(arr, name, '{I.freq}')\n"
    else:
        func_text += "    return bodo.utils.conversion.index_from_array(arr, name)"

    return bodo_exec(func_text, {"bodo": bodo}, {}, __name__)


@numba.generated_jit(cache=True, nopython=True)
def get_index_data(S):
    return lambda S: S._data  # pragma: no cover


@numba.generated_jit(cache=True, nopython=True)
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


@overload_method(
    RangeIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    NumericIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    PeriodIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "map",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_index_map(I, mapper, na_action=None):
    if not is_const_func_type(mapper):
        raise BodoError("Index.map(): 'mapper' should be a function")

    unsupported_args = {
        "na_action": na_action,
    }
    map_defaults = {
        "na_action": None,
    }
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
        dtype = pd_timestamp_tz_naive_type
    if dtype == types.NPTimedelta("ns"):
        dtype = pd_timedelta_type
    if isinstance(dtype, PandasDatetimeTZDtype):
        dtype = PandasTimestampType(dtype.tz)
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
    func_text = "def bodo_overload_index_map(I, mapper, na_action=None):\n"
    func_text += "  name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    func_text += "  A = bodo.utils.conversion.coerce_to_array(I)\n"
    func_text += "  numba.parfors.parfor.init_prange()\n"
    func_text += "  n = len(A)\n"
    func_text += "  S = bodo.utils.utils.alloc_type(n, _arr_typ, (-1,))\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "    t2 = bodo.utils.conversion.box_if_dt64(A[i])\n"
    func_text += "    v = map_func(t2)\n"
    func_text += "    S[i] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(v)\n"
    func_text += "  return bodo.utils.conversion.index_from_array(S, name)\n"

    map_func = bodo.compiler.udf_jit(func)

    return bodo_exec(
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
        {},
        __name__,
    )


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
                "def bodo_index_binary_op_lhs(lhs, rhs):\n"
                "  arr = bodo.utils.conversion.coerce_to_array(lhs)\n"
            )
            if rhs in [
                bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type,
                bodo.hiframes.pd_timestamp_ext.pd_timedelta_type,
            ]:
                func_text += (
                    "  dt = bodo.utils.conversion.unbox_if_tz_naive_timestamp(rhs)\n"
                    "  return op(arr, dt)\n"
                )
            else:
                func_text += (
                    "  rhs_arr = bodo.utils.conversion.get_array_if_series_or_index(rhs)\n"
                    "  return op(arr, rhs_arr)\n"
                )
            return bodo_exec(
                func_text,
                {"bodo": bodo, "op": op},
                {},
                __name__,
            )

        # right arg is Index
        if is_index_type(rhs):
            func_text = (
                "def bodo_index_binary_op_rhs(lhs, rhs):\n"
                "  arr = bodo.utils.conversion.coerce_to_array(rhs)\n"
            )
            if lhs in [
                bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type,
                bodo.hiframes.pd_timestamp_ext.pd_timedelta_type,
            ]:
                func_text += (
                    "  dt = bodo.utils.conversion.unbox_if_tz_naive_timestamp(lhs)\n"
                    "  return op(dt, arr)\n"
                )
            else:
                func_text += (
                    "  lhs_arr = bodo.utils.conversion.get_array_if_series_or_index(lhs)\n"
                    "  return op(lhs_arr, arr)\n"
                )
            return bodo_exec(
                func_text,
                {"bodo": bodo, "op": op},
                {},
                __name__,
            )

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
            func_text = "def bodo_index_binary_op(lhs, rhs):\n"
            func_text += "  return [{}]\n".format(
                ",".join(
                    "op(lhs[{}], rhs{})".format(
                        i, f"[{i}]" if is_iterable_type(rhs) else ""
                    )
                    for i in range(count)
                ),
            )
            return bodo_exec(func_text, {"op": op, "np": np}, {}, __name__)

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
            func_text = "def bodo_index_binary_op(lhs, rhs):\n"
            func_text += "  return [{}]\n".format(
                ",".join(
                    "op(lhs{}, rhs[{}])".format(
                        f"[{i}]" if is_iterable_type(lhs) else "", i
                    )
                    for i in range(count)
                ),
            )
            return bodo_exec(func_text, {"op": op, "np": np}, {}, __name__)

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


def _install_binop_overload(op):
    """Install overload for binop
    NOTE: This has to be a separate function to avoid unexpected free variable updates
    """
    overload_impl = create_binary_op_overload(op)
    overload(op, inline="always")(overload_impl)


def _install_binary_ops():
    # install binary ops such as add, sub, pow, eq, ...
    for op in bodo.hiframes.pd_series_ext.series_binary_ops:
        if op in skips:
            continue
        _install_binop_overload(op)


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


@numba.njit(no_cpython_wrapper=True)
def range_index_to_numeric(I):  # pragma: no cover
    """Convert RangeIndex to equivalent NumericIndex

    Args:
        I (RangeIndexType): RangeIndex input

    Returns:
        NumericIndexType: NumericIndex output
    """
    return init_numeric_index(
        np.arange(I._start, I._stop, I._step),
        bodo.hiframes.pd_index_ext.get_index_name(I),
    )


class HeterogeneousIndexType(types.Type, SingleIndexType):
    """
    Type class for Index objects with potentially heterogeneous but limited number of
    values (e.g. pd.Index([1, 'A']))
    """

    ndim = 1

    def __init__(self, data=None, name_typ=None):
        self.data = data
        name_typ = types.none if name_typ is None else name_typ
        self.name_typ = name_typ
        super().__init__(name=f"heter_index({data}, {name_typ})")

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
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(HeterogeneousIndexType, "data", "_data")
make_attribute_wrapper(HeterogeneousIndexType, "name", "_name")


@overload_method(
    HeterogeneousIndexType, "copy", no_unliteral=True, jit_options={"cache": True}
)
def overload_heter_index_copy(A, name=None, deep=False, dtype=None, names=None):
    err_str = idx_typ_to_format_str_map[HeterogeneousIndexType].format("copy()")
    idx_cpy_unsupported_args = {"deep": deep, "dtype": dtype, "names": names}
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
    class_obj = c.pyapi.import_module(mod_name)

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


@intrinsic(prefer_literal=True)
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


@overload_attribute(HeterogeneousIndexType, "name", jit_options={"cache": True})
def heter_index_get_name(i):
    def impl(i):  # pragma: no cover
        return i._name

    return impl


@overload_attribute(NumericIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(StringIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "nbytes", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "nbytes", jit_options={"cache": True})
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
    elif isinstance(I, MultiIndexType):
        func_text = "def bodo_impl_nbytes(I):\n"
        func_text += "    total = 0\n"
        func_text += "    data = I._data\n"
        for i in range(I.nlevels):
            func_text += f"    total += data[{i}].nbytes\n"
        func_text += "    return total\n"
        return bodo_exec(func_text, {}, {}, __name__)

    else:

        def _impl_nbytes(I):  # pragma: no cover
            return I._data.nbytes

        return _impl_nbytes


@overload_method(
    NumericIndexType, "to_series", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "to_series", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "to_series", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "to_series", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "to_series", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "to_series", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "to_series", inline="always", jit_options={"cache": True}
)
def overload_index_to_series(I, index=None, name=None):
    """Supports pd.Index.to_series() on tagged Index types.

    Args:
        I (pd.Index): The Index that is being converted to a Series.
        index (iterable, optional): The index for the new Series. Can be an Index,
        list, tuple, or Series. If not provided, uses I as the index. Defaults to None.
        name (string, optional): the name of the new Series. If not provided, uses
        the name of I. Defaults to None.

    Returns:
        pd.Series: a Series with the Index values as its values
    """

    if not (
        is_overload_constant_str(name)
        or is_overload_constant_int(name)
        or is_overload_none(name)
    ):
        raise_bodo_error(
            "Index.to_series(): only constant string/int are supported for argument name"
        )

    if is_overload_none(name):
        name_str = "bodo.hiframes.pd_index_ext.get_index_name(I)"
    else:
        name_str = "name"

    func_text = "def bodo_index_to_series(I, index=None, name=None):\n"
    func_text += "    data = bodo.utils.conversion.index_to_array(I)\n"
    if is_overload_none(index):
        func_text += "    new_index = I\n"
    else:
        if is_pd_index_type(index):
            func_text += "    new_index = index\n"
        elif isinstance(index, SeriesType):
            func_text += "    arr = bodo.utils.conversion.coerce_to_array(index)\n"
            func_text += (
                "    index_name = bodo.hiframes.pd_series_ext.get_series_name(index)\n"
            )
            func_text += "    new_index = bodo.utils.conversion.index_from_array(arr, index_name)\n"
        elif bodo.utils.utils.is_array_typ(index, False):
            func_text += (
                "    new_index = bodo.utils.conversion.index_from_array(index)\n"
            )
        elif isinstance(index, (types.List, types.BaseTuple)):
            func_text += "    arr = bodo.utils.conversion.coerce_to_array(index)\n"
            func_text += "    new_index = bodo.utils.conversion.index_from_array(arr)\n"
        else:
            raise_bodo_error(
                f"Index.to_series(): unsupported type for argument index: {type(index).__name__}"
            )

    func_text += f"    new_name = {name_str}\n"
    func_text += (
        "    return bodo.hiframes.pd_series_ext.init_series(data, new_index, new_name)"
    )
    return bodo_exec(func_text, {"bodo": bodo, "np": np}, {}, __name__)


@overload_method(
    NumericIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(DatetimeIndexType, "to_frame", inline="always", no_unliteral=True)
@overload_method(
    TimedeltaIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_index_to_frame(I, index=True, name=None):
    """Supports pd.Index.to_frame() on tagged Index types.

    Args:
        I (pd.Index): the Index that is being converted to a DataFrame
        index (bool, optional): if True, I is also the Index of the new DataFrame.
        If False, the DataFrame's Index is a RangeIndex. Defaults to True.
        name (string, optional): if provided, the name of the column of the new
        DataFrame. If not provided, uses the name of the Index. If the Index's
        name is also None, uses 0. Defaults to None.

    Returns:
        pd.DataFrame: a DataFrame with the Index's values as its only column.
    """

    if is_overload_true(index):
        index_str = "I"
    elif is_overload_false(index):
        index_str = "bodo.hiframes.pd_index_ext.init_range_index(0, len(I), 1, None)"
    elif not isinstance(index, types.Boolean):
        raise_bodo_error("Index.to_frame(): index argument must be a constant boolean")
    else:
        raise_bodo_error(
            "Index.to_frame(): index argument must be a compile time constant"
        )

    func_text = "def bodo_index_to_frame(I, index=True, name=None):\n"
    func_text += "    data = bodo.utils.conversion.index_to_array(I)\n"
    func_text += f"    new_index = {index_str}\n"

    # If no name provided and the Index itself has no name, the column name is 0
    if is_overload_none(name) and I.name_typ == types.none:
        columns = ColNamesMetaType((0,))

    # Otherwise, the column name is either the name provided or the name of the Index
    else:
        if is_overload_none(name):
            columns = ColNamesMetaType((I.name_typ,))
        elif is_overload_constant_str(name):
            columns = ColNamesMetaType((get_overload_const_str(name),))
        elif is_overload_constant_int(name):
            columns = ColNamesMetaType((get_overload_const_int(name),))
        else:
            raise_bodo_error(
                "Index.to_frame(): only constant string/int are supported for argument name"
            )

    func_text += "    return bodo.hiframes.pd_dataframe_ext.init_dataframe((data,), new_index, __col_name_meta_value)\n"

    return bodo_exec(
        func_text,
        {
            "bodo": bodo,
            "np": np,
            "__col_name_meta_value": columns,
        },
        {},
        __name__,
    )


@overload_method(
    MultiIndexType,
    "to_frame",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_multi_index_to_frame(I, index=True, name=None):
    """Supports pd.Index.to_frame() for MultiIndex

    Args:
        I (pd.Index): the Index that is being converted to a DataFrame
        index (bool, optional): if True, I is also the Index of the new DataFrame.
        If False, the DataFrame's Index is a RangeIndex. Defaults to True.
        name (string list, optional): if provided, the names of the columns of the
        new DataFrame. If not provided, uses the name of the Index. If the Index's
        names is also None, uses [0, 1, ...]. Defaults to None.

    Returns:
        pd.DataFrame: a DataFrame with each column of the Index as a column
    """

    if is_overload_true(index):
        index_str = "I"
    elif is_overload_false(index):
        index_str = "bodo.hiframes.pd_index_ext.init_range_index(0, len(I), 1, None)"
    elif not isinstance(index, types.Boolean):
        raise_bodo_error(
            "MultiIndex.to_frame(): index argument must be a constant boolean"
        )
    else:
        raise_bodo_error(
            "MultiIndex.to_frame(): index argument must be a compile time constant"
        )

    func_text = "def bodo_multi_index_to_frame(I, index=True, name=None):\n"
    func_text += "    data = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
    func_text += f"    new_index = {index_str}\n"

    # If no name provided and the Index itself has no name, the column names
    # are 0...n-1
    n_fields = len(I.array_types)
    if is_overload_none(name) and I.names_typ == (types.none,) * n_fields:
        columns = ColNamesMetaType(tuple(range(n_fields)))

    # Otherwise, the column name is either the name provided or the name of the Index
    else:
        if is_overload_none(name):
            columns = ColNamesMetaType(I.names_typ)
        else:
            if is_overload_constant_tuple(name) or is_overload_constant_list(name):
                if is_overload_constant_list(name):
                    names = tuple(get_overload_const_list(name))
                else:
                    names = get_overload_const_tuple(name)
                if n_fields != len(names):
                    raise_bodo_error(
                        f"MultiIndex.to_frame(): expected {n_fields} names, not {len(names)}"
                    )
                if all(
                    is_overload_constant_str(v) or is_overload_constant_int(v)
                    for v in names
                ):
                    columns = ColNamesMetaType(names)
                else:
                    raise_bodo_error(
                        "MultiIndex.to_frame(): only constant string/int list/tuple are supported for argument name"
                    )
            else:
                raise_bodo_error(
                    "MultiIndex.to_frame(): only constant string/int list/tuple are supported for argument name"
                )

    func_text += "    return bodo.hiframes.pd_dataframe_ext.init_dataframe(data, new_index, __col_name_meta_value,)\n"
    return bodo_exec(
        func_text,
        {"bodo": bodo, "np": np, "__col_name_meta_value": columns},
        {},
        __name__,
    )


@overload_method(
    NumericIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "to_numpy", inline="always", jit_options={"cache": True}
)
def overload_index_to_numpy(I, dtype=None, copy=False, na_value=None):
    """Supports pd.Index.to_numpy() on tagged Index types.

    Args:
        I (pd.Index): the Index that is being converted to a numpy array.
        dtype (str or np.dtype, optional): not supported. Defaults to None.
        copy (bool, optional): if True, guarantees that the returned array
        does not alias to the underlying array of the index. Defaults to False.
        na_value (any, optional): not supported Defaults to None.

    Returns:
        np.ndarray: a numpy array with the same underlying values as I.
    """

    unsupported_args = {"dtype": dtype, "na_value": na_value}
    arg_defaults = {"dtype": None, "na_value": None}
    check_unsupported_args(
        "Index.to_numpy",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_bool(copy):
        raise_bodo_error("Index.to_numpy(): copy argument must be a boolean")

    if isinstance(I, RangeIndexType):

        def impl(I, dtype=None, copy=False, na_value=None):  # pragma: no cover
            return np.arange(I._start, I._stop, I._step)

        return impl

    # Force copy to be True or False at runtime for other Index types (RangeIndex
    # is always a copy so it doesn't matter)

    if is_overload_true(copy):

        def impl(I, dtype=None, copy=False, na_value=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.get_index_data(I).copy()

        return impl

    if is_overload_false(copy):

        def impl(I, dtype=None, copy=False, na_value=None):  # pragma: no cover
            return bodo.hiframes.pd_index_ext.get_index_data(I)

        return impl

    def impl(I, dtype=None, copy=False, na_value=None):  # pragma: no cover
        data = bodo.hiframes.pd_index_ext.get_index_data(I)
        return data.copy() if copy else data

    return impl


@overload_method(
    NumericIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "to_list", inline="always", jit_options={"cache": True}
)
@overload_method(
    NumericIndexType, "tolist", inline="always", jit_options={"cache": True}
)
@overload_method(RangeIndexType, "tolist", inline="always", jit_options={"cache": True})
@overload_method(
    StringIndexType, "tolist", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "tolist", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "tolist", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "tolist", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "tolist", inline="always", jit_options={"cache": True}
)
def overload_index_to_list(I):
    """Supported pd.Index.to_list() on tagged Index types

    Args:
        I (pd.Index): the Index being converted to a list

    Returns:
        list: values of the Index in a Python list
    """

    if isinstance(I, RangeIndexType):

        def impl(I):  # pragma: no cover
            l = []
            for i in range(I._start, I._stop, I.step):
                l.append(i)
            return l

        return impl

    # Supported for all Index types that have a supported iterator
    def impl(I):  # pragma: no cover
        l = []
        for i in range(len(I)):
            l.append(I[i])
        return l

    return impl


@overload_attribute(NumericIndexType, "T", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "T", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "T", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "T", jit_options={"cache": True})
@overload_attribute(StringIndexType, "T", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "T", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "T", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "T", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "T", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "T", jit_options={"cache": True})
def overload_T(I):
    """Adds support for Index.T

    Args:
        I (pd.Index): the index whose transpose is being found

    Returns:
        pd.Index: an the same index
    """

    return lambda I: I  # pragma: no cover


@overload_attribute(NumericIndexType, "size", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "size", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "size", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "size", jit_options={"cache": True})
@overload_attribute(StringIndexType, "size", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "size", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "size", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "size", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "size", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "size", jit_options={"cache": True})
def overload_size(I):
    """Adds support for Index.size

    Args:
        I (pd.Index): the index whose size is being found

    Returns:
        integer: the length of the index
    """
    return lambda I: len(I)  # pragma: no cover


@overload_attribute(NumericIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(StringIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "ndim", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "ndim", jit_options={"cache": True})
def overload_ndim(I):
    """Adds support for Index.ndim

    Args:
        I (pd.Index): the index whose ndim is being found

    Returns:
        integer: the number of dimensions of the index
    """

    return lambda I: 1  # pragma: no cover


@overload_attribute(NumericIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(StringIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "nlevels", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "nlevels", jit_options={"cache": True})
def overload_nlevels(I):
    """Adds support for Index.nlevels

    Args:
        I (pd.Index): the index whose nlevels is being found

    Returns:
        integer: the number of levels of the index
    """
    if isinstance(I, MultiIndexType):
        return lambda I: len(I._data)  # pragma: no cover

    return lambda I: 1  # pragma: no cover


@overload_attribute(NumericIndexType, "empty", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "empty", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "empty", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "empty", jit_options={"cache": True})
@overload_attribute(StringIndexType, "empty", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "empty", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "empty", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "empty", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "empty", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "empty", jit_options={"cache": True})
def overload_empty(I):
    """Adds support for Index.empty

    Args:
        I (pd.Index): the index whose empty status is being found

    Returns:
        boolean: whether or not the index is empty
    """
    return lambda I: len(I) == 0  # pragma: no cover


@overload_attribute(NumericIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(StringIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "inferred_type", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "inferred_type", jit_options={"cache": True})
def overload_inferred_type(I):
    """Adds support for Index.inferred_type

    Args:
        I (pd.Index): the index whose inferred type is being found

    Returns:
        string: a user-friendly representation of the underlying type of the Index
    """
    if isinstance(I, NumericIndexType):
        if isinstance(I.dtype, types.Integer):
            return lambda I: "integer"  # pragma: no cover
        elif isinstance(I.dtype, types.Float):
            return lambda I: "floating"  # pragma: no cover
        elif isinstance(I.dtype, types.Boolean):
            return lambda I: "boolean"  # pragma: no cover
        return

    if isinstance(I, StringIndexType):

        def impl(I):  # pragma: no cover
            if len(I._data) == 0:
                return "empty"
            return "string"

        return impl

    inferred_types_map = {
        DatetimeIndexType: "datetime64",
        TimedeltaIndexType: "timedelta64",
        RangeIndexType: "integer",
        BinaryIndexType: "bytes",
        CategoricalIndexType: "categorical",
        PeriodIndexType: "period",
        IntervalIndexType: "interval",
        MultiIndexType: "mixed",
    }
    inferred_type = inferred_types_map[type(I)]
    return lambda I: inferred_type  # pragma: no cover


@overload_attribute(NumericIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(StringIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "dtype", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "dtype", jit_options={"cache": True})
def overload_inferred_type(I):
    """Adds support for Index.dtype

    Args:
        I (pd.Index): the index whose dtype type is being found

    Returns:
        np.dtype: the dtype of the Index's data
    """
    # Note: does not return the correct type if the underlying data came
    # from a pd.array
    if isinstance(I, NumericIndexType):
        dtype = I.dtype
        return lambda I: dtype  # pragma: no cover

    if isinstance(I, CategoricalIndexType):
        dtype = bodo.utils.utils.create_categorical_type(
            I.dtype.categories, I.data, I.dtype.ordered
        )
        return lambda I: dtype  # pragma: no cover

    dtype_map = {
        DatetimeIndexType: np.dtype("datetime64[ns]"),
        TimedeltaIndexType: np.dtype("timedelta64[ns]"),
        RangeIndexType: np.dtype("int64"),
        StringIndexType: np.dtype("O"),
        BinaryIndexType: np.dtype("O"),
        MultiIndexType: np.dtype("O"),
    }
    dtype = dtype_map[type(I)]
    return lambda I: dtype  # pragma: no cover


@overload_attribute(NumericIndexType, "names", jit_options={"cache": True})
@overload_attribute(DatetimeIndexType, "names", jit_options={"cache": True})
@overload_attribute(TimedeltaIndexType, "names", jit_options={"cache": True})
@overload_attribute(RangeIndexType, "names", jit_options={"cache": True})
@overload_attribute(StringIndexType, "names", jit_options={"cache": True})
@overload_attribute(BinaryIndexType, "names", jit_options={"cache": True})
@overload_attribute(CategoricalIndexType, "names", jit_options={"cache": True})
@overload_attribute(IntervalIndexType, "names", jit_options={"cache": True})
@overload_attribute(PeriodIndexType, "names", jit_options={"cache": True})
@overload_attribute(MultiIndexType, "names", jit_options={"cache": True})
def overload_names(I):
    """Adds support for Index.names. Diverges from the pandas API by returning
       a tuple instead of a FrozenList.

    Args:
        I (pd.Index): the Index whose name(s) are being extracted.

    Returns:
        (string option tuple): tuple of all the names (or of Nones)
    """

    if isinstance(I, MultiIndexType):
        return lambda I: I._names  # pragma: no cover

    return lambda I: (I._name,)  # pragma: no cover


@overload_method(
    NumericIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(RangeIndexType, "rename", inline="always", jit_options={"cache": True})
@overload_method(
    StringIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "rename", inline="always", jit_options={"cache": True}
)
@overload_method(
    HeterogeneousIndexType, "rename", inline="always", jit_options={"cache": True}
)
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


@overload_method(
    NumericIndexType,
    "min",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "min",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "min",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_min(I, axis=None, skipna=True):
    """Supports pd.Index.min() on tagged Index types

    Args:
        I (pd.Index): the Index whose minimum value is being found
        axis (int, optional): not supported. Defaults to None.
        skipna (bool, optional): not supported. Defaults to True.

    Returns:
        any: the minimum value of the Index
    """
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": None, "skipna": True}
    check_unsupported_args(
        "Index.min",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if isinstance(I, RangeIndexType):

        def impl(I, axis=None, skipna=True):  # pragma: no cover
            size = len(I)
            if size == 0:
                return np.nan
            if I._step < 0:
                return I._start + I._step * (size - 1)
            else:
                return I._start

        return impl

    if isinstance(I, CategoricalIndexType):
        if not I.dtype.ordered:  # pragma: no cover
            raise BodoError("Index.min(): only ordered categoricals are possible")

    def impl(I, axis=None, skipna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        return bodo.libs.array_ops.array_op_min(arr)

    return impl


@overload_method(
    NumericIndexType,
    "max",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "max",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "max",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_max(I, axis=None, skipna=True):
    """Supports pd.Index.max() on tagged Index types

    Args:
        I (pd.Index): the Index whose maximum value is being found
        axis (int, optional): not supported. Defaults to None.
        skipna (bool, optional): not supported. Defaults to True.

    Returns:
        any: the maximum value of the Index
    """
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": None, "skipna": True}
    check_unsupported_args(
        "Index.max",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if isinstance(I, RangeIndexType):

        def impl(I, axis=None, skipna=True):  # pragma: no cover
            size = len(I)
            if size == 0:
                return np.nan
            if I._step > 0:
                return I._start + I._step * (size - 1)
            else:
                return I._start

        return impl

    if isinstance(I, CategoricalIndexType):
        if not I.dtype.ordered:  # pragma: no cover
            raise BodoError("Index.max(): only ordered categoricals are possible")

    def impl(I, axis=None, skipna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        return bodo.libs.array_ops.array_op_max(arr)

    return impl


@overload_method(
    NumericIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    PeriodIndexType,
    "argmin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_argmin(I, axis=0, skipna=True):
    """Support for Index.argmin() on tagged Index types

    Args:
        I (pd.Index): the Index whose argmin is being found.
        axis (int, optional): Not supported. Defaults to 0.
        skipna (bool, optional): Not supported. Defaults to True.

    Raises:
        BodoError: if an unordered CategoricalIndex is provided.

    Returns:
        int: the location of the minimum value of the index
    """
    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": 0, "skipna": True}
    check_unsupported_args(
        "Index.argmin",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if isinstance(I, RangeIndexType):

        def impl(I, axis=0, skipna=True):  # pragma: no cover
            # If step is positive, returns zero. If step is negative, returns size-1.
            return (I._step < 0) * (len(I) - 1)

        return impl

    if isinstance(I, CategoricalIndexType) and not I.dtype.ordered:
        raise BodoError("Index.argmin(): only ordered categoricals are possible")

    def impl(I, axis=0, skipna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        index = init_numeric_index(np.arange(len(arr)))
        return bodo.libs.array_ops.array_op_idxmin(arr, index)

    return impl


@overload_method(
    NumericIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    PeriodIndexType,
    "argmax",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_argmax(I, axis=0, skipna=True):
    """Support for Index.argmax() on tagged Index types

    Args:
        I (pd.Index): the Index whose argmax is being found.
        axis (int, optional): Not supported. Defaults to 0.
        skipna (bool, optional): Not supported. Defaults to True.

    Raises:
        BodoError: if an unordered CategoricalIndex is provided.

    Returns:
        int: the location of the maximum value of the index
    """

    unsupported_args = {"axis": axis, "skipna": skipna}
    arg_defaults = {"axis": 0, "skipna": True}
    check_unsupported_args(
        "Index.argmax",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if isinstance(I, RangeIndexType):

        def impl(I, axis=0, skipna=True):  # pragma: no cover
            # If step is negative, returns zero. If step is positive, returns size-1.
            return (I._step > 0) * (len(I) - 1)

        return impl

    if isinstance(I, CategoricalIndexType) and not I.dtype.ordered:
        raise BodoError("Index.argmax(): only ordered categoricals are possible")

    def impl(I, axis=0, skipna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        index = np.arange(len(arr))
        return bodo.libs.array_ops.array_op_idxmax(arr, index)

    return impl


@overload_method(
    NumericIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
# Does not work if the intervals are distinct but share a start-value
# (i.e. [(1, 2), (2, 3), (1, 3)]).
# Does not work for time-based intervals.
# See [BE-2813]
@overload_method(
    IntervalIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_unique(I):
    """Add support for Index.unique() on most Index types"""
    constructor = get_index_constructor(I)

    def impl(I):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        name = bodo.hiframes.pd_index_ext.get_index_name(I)
        uni = bodo.libs.array_kernels.unique(arr)
        return constructor(uni, name)

    return impl


@overload_method(
    RangeIndexType,
    "unique",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_range_index_unique(I):
    """Add support for Index.unique() on RangeIndex"""

    def impl(I):  # pragma: no cover
        return I.copy()

    return impl


@overload_method(
    NumericIndexType, "nunique", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "nunique", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "nunique", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "nunique", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "nunique", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "nunique", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "nunique", inline="always", jit_options={"cache": True}
)
def overload_index_nunique(I, dropna=True):
    """Add support for Index.nunique() on tagged Index types"""

    def impl(I, dropna=True):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        n = bodo.libs.array_kernels.nunique(arr, dropna)
        return n

    return impl


@overload_method(
    RangeIndexType, "nunique", inline="always", jit_options={"cache": True}
)
def overload_range_index_nunique(I, dropna=True):
    """Add support for Index.nunique() on RangeIndex by calculating the
    number of elements in the range."""

    def impl(I, dropna=True):  # pragma: no cover
        start = I._start
        stop = I._stop
        step = I._step
        return max(0, -((-(stop - start)) // step))

    return impl


@overload_method(
    NumericIndexType,
    "isin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "isin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "isin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "isin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "isin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_isin(I, values):
    # if input is Series or array, special implementation is necessary since it may
    # require hash-based shuffling of both inputs for parallelization
    if bodo.utils.utils.is_array_typ(values):

        def impl_arr(I, values):  # pragma: no cover
            values_arr = bodo.utils.conversion.coerce_to_array(values)
            A = bodo.hiframes.pd_index_ext.get_index_data(I)
            n = len(A)
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)
            bodo.libs.array.array_isin(out_arr, A, values_arr, False)
            return out_arr

        return impl_arr

    # 'values' should be a set or list, TODO: support other list-likes such as Array
    if not isinstance(values, (types.Set, types.List)):  # pragma: no cover
        raise BodoError("Index.isin(): 'values' parameter should be a set or a list")

    def impl(I, values):  # pragma: no cover
        A = bodo.hiframes.pd_index_ext.get_index_data(I)
        out_arr = bodo.libs.array_ops.array_op_isin(A, values)
        return out_arr

    return impl


@overload_method(
    RangeIndexType,
    "isin",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_range_index_isin(I, values):
    # if input is Series or array, special implementation is necessary since it may
    # require hash-based shuffling of both inputs for parallelization
    if bodo.utils.utils.is_array_typ(values):

        def impl_arr(I, values):  # pragma: no cover
            values_arr = bodo.utils.conversion.coerce_to_array(values)
            A = np.arange(I.start, I.stop, I.step)
            n = len(A)
            out_arr = bodo.libs.bool_arr_ext.alloc_false_bool_array(n)
            # TODO: design special kernel operator at C++ level to optimize
            # this operation just for ranges [BE-2836]
            bodo.libs.array.array_isin(out_arr, A, values_arr, False)
            return out_arr

        return impl_arr

    # 'values' should be a set or list, TODO: support other list-likes such as Array
    if not isinstance(values, (types.Set, types.List)):
        raise BodoError("Index.isin(): 'values' parameter should be a set or a list")

    def impl(I, values):  # pragma: no cover
        A = np.arange(I.start, I.stop, I.step)
        # TODO: design special kernel operator at C++ level to optimize
        # this operation just for ranges [BE-2836]
        out_arr = bodo.libs.array_ops.array_op_isin(A, values)
        return out_arr

    return impl


@register_jitable
def order_range(I, ascending):  # pragma: no cover
    """If I is a RangeIndex that is increasing and ascending is True, or if
    it is descending and ascending is False, then the RangeIndex is cloned.
    Otherwise, its direction is flipped."""
    step = I._step
    # If the range is already sorted in the correct direction, leave alone
    if ascending == (step > 0):
        return I.copy()
    # Otherwise, flip the step sign and calculate new start/end points
    else:
        start = I._start
        name = get_index_name(I)
        size = len(I)
        last_value = start + step * (size - 1)
        new_stop = last_value - step * size
        return init_range_index(last_value, new_stop, -step, name)


@overload_method(
    NumericIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "sort_values",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_sort_values(
    I, return_indexer=False, ascending=True, na_position="last", key=None
):
    """Supports pd.sort_values() on tagged Index types.

    Args:
        I (pd.Index): the Index that is being sorted
        return_indexer (bool, optional): not supported. Defaults to False.
        ascending (bool, optional): whether the values should be sorted in
        increasing or decreasing order. Defaults to True.
        na_position (str, optional): whether null values should be placed
        at the begining or end of the Index. Defaults to "last".
        key (function, optional): not supported. Defaults to None.

    Raises:
        BodoError: if unsupported arguments are provided.

    Returns:
        pd.Index: the Index with its values sorted.
    """
    unsupported_args = {
        "return_indexer": return_indexer,
        "key": key,
    }
    arg_defaults = {
        "return_indexer": False,
        "key": None,
    }
    check_unsupported_args(
        "Index.sort_values",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if not is_overload_bool(ascending):
        raise BodoError(
            "Index.sort_values(): 'ascending' parameter must be of type bool"
        )

    if not is_overload_constant_str(na_position) or get_overload_const_str(
        na_position
    ) not in ("first", "last"):
        raise_bodo_error(
            "Index.sort_values(): 'na_position' should either be 'first' or 'last'"
        )

    if isinstance(I, RangeIndexType):

        def impl(
            I, return_indexer=False, ascending=True, na_position="last", key=None
        ):  # pragma: no cover
            return order_range(I, ascending)

        return impl

    constructor = get_index_constructor(I)

    meta_data = ColNamesMetaType(("$_bodo_col_",))

    # reusing dataframe sort_values() in implementation.
    def impl(
        I, return_indexer=False, ascending=True, na_position="last", key=None
    ):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        name = get_index_name(I)
        index = init_range_index(0, len(arr), 1, None)
        df = bodo.hiframes.pd_dataframe_ext.init_dataframe((arr,), index, meta_data)
        sorted_df = df.sort_values(
            ["$_bodo_col_"],
            ascending=ascending,
            inplace=False,
            na_position=na_position,
        )
        out_arr = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(sorted_df, 0)
        return constructor(out_arr, name)

    return impl


@overload_method(
    NumericIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    PeriodIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "argsort",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_argsort(I, axis=0, kind="quicksort", order=None):
    """Supports pd.argsort() on tagged Index types.

    Args:
        I (pd.Index): the Index that is being sorted
        axis (int, optional): Not supported. Defaults to 0.
        kind (str, optional): Not supported. Defaults to "quicksort".
        order (str, optional): Not supported. Defaults to None.

    Returns:
        np.ndarray: the locations of each element in the original index
        if they were to be sorted.
    """
    unsupported_args = {"axis": axis, "kind": kind, "order": order}
    arg_defaults = {"axis": 0, "kind": "quicksort", "order": None}
    check_unsupported_args(
        "Index.argsort",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )

    if isinstance(I, RangeIndexType):

        def impl(I, axis=0, kind="quicksort", order=None):  # pragma: no cover
            # If the range is already sorted, construct a regular enumeration
            if I._step > 0:
                return np.arange(0, len(I), 1)
            # Otherwise, flip the direction of the enumeration
            else:
                return np.arange(len(I) - 1, -1, -1)

        return impl

    def impl(I, axis=0, kind="quicksort", order=None):  # pragma: no cover
        arr = bodo.hiframes.pd_index_ext.get_index_data(I)
        out_arr = bodo.hiframes.series_impl.argsort(arr)
        return out_arr

    return impl


@overload_method(
    NumericIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
# [BE-2910]: Only works if the elements from other are the same as (or a subset of) the categories
@overload_method(
    CategoricalIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "where",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_where(I, cond, other=np.nan):
    """Supports pd.Index.where() on tagged Index types.

    Args:
        I (pd.Index): the Index that is being transformed
        cond (boolean array): specifies which locations in I to replace with other
        other (iterbale): a scalar/array/Index/array with the values that are injected
        into I if the corresponding location of cond is True. Must have an
        underlying type that can be reconciled with the type of I. If a scalar
        is provided, then all values that are replaced are replaced with that
        value. Defaults to np.nan.

    Returns:
        pd.Index: a copy of I with locations that are False in cond replaced with
        the corresponding value from other.
    """

    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(I, "Index.where()")
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(other, "Index.where()")
    bodo.hiframes.series_impl._validate_arguments_mask_where(
        "where",
        "Index",
        I,
        cond,
        other,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    )

    if is_overload_constant_nan(other):
        other_str = "None"
    else:
        other_str = "other"

    func_text = "def bodo_index_where(I, cond, other=np.nan):\n"
    if isinstance(I, RangeIndexType):
        func_text += "  arr = np.arange(I._start, I._stop, I._step)\n"
        constructor = "init_numeric_index"
    else:
        func_text += "  arr = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
    func_text += "  name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    func_text += (
        f"  out_arr = bodo.hiframes.series_impl.where_impl(cond, arr, {other_str})\n"
    )
    func_text += "  return constructor(out_arr, name)\n"
    constructor = (
        init_numeric_index
        if isinstance(I, RangeIndexType)
        else get_index_constructor(I)
    )
    return bodo_exec(
        func_text, {"bodo": bodo, "np": np, "constructor": constructor}, {}, __name__
    )


@overload_method(
    NumericIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    BinaryIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
# [BE-2910]: Only works if the elements from other are the same as (or a subset of) the categories
@overload_method(
    CategoricalIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "putmask",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_putmask(I, cond, other):
    """Supports pd.Index.putmask() on tagged Index types.

    Args:
        I (pd.Index): the Index that is being transformed
        cond (boolean array): specifies which locations in I to replace with other
        other (iterbale): a scalar/array/Index/array with the values that are injected
        into I if the corresponding location of cond is True. Must have an
        underlying type that can be reconciled with the type of I. If a scalar
        is provided, then all values that are replaced are replaced with that
        value.

    Returns:
        pd.Index: a copy of I with locations that are True in cond replaced with
        the corresponding value from other.
    """

    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(I, "Index.putmask()")
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(other, "Index.putmask()")
    bodo.hiframes.series_impl._validate_arguments_mask_where(
        "putmask",
        "Index",
        I,
        cond,
        other,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    )

    if is_overload_constant_nan(other):
        other_str = "None"
    else:
        other_str = "other"

    func_text = "def bodo_index_putmask(I, cond, other):\n"
    func_text += "  cond = ~cond\n"
    if isinstance(I, RangeIndexType):
        func_text += "  arr = np.arange(I._start, I._stop, I._step)\n"
    else:
        func_text += "  arr = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
    func_text += "  name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    func_text += (
        f"  out_arr = bodo.hiframes.series_impl.where_impl(cond, arr, {other_str})\n"
    )
    func_text += "  return constructor(out_arr, name)\n"
    constructor = (
        init_numeric_index
        if isinstance(I, RangeIndexType)
        else get_index_constructor(I)
    )
    return bodo_exec(
        func_text, {"bodo": bodo, "np": np, "constructor": constructor}, {}, __name__
    )


@overload_method(
    NumericIndexType,
    "repeat",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    StringIndexType,
    "repeat",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    CategoricalIndexType,
    "repeat",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    DatetimeIndexType,
    "repeat",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    TimedeltaIndexType,
    "repeat",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
@overload_method(
    RangeIndexType,
    "repeat",
    no_unliteral=True,
    inline="always",
    jit_options={"cache": True},
)
def overload_index_repeat(I, repeats, axis=None):
    """Supports pd.Index.repeats() on tagged index type

    Args:
        I (pd.Index): the Index whose elements are going to be repeated
        repeats (int or int iterable): the number of times each element is repeated
        (must be non-negative). If iterable, then each element specifies the
        number of times a specific value of I is repeated.
        axis (any, optional): not supported. Defaults to None.

    Returns:
        pd.Index: a version of I with its values repeated
    """

    unsupported_args = {"axis": axis}
    arg_defaults = {"axis": None}
    check_unsupported_args(
        "Index.repeat",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Index",
    )
    bodo.hiframes.pd_timestamp_ext.check_tz_aware_unsupported(I, "Index.repeat()")

    # Repeats can be int or array of int
    if not (
        isinstance(repeats, types.Integer)
        or (is_iterable_type(repeats) and isinstance(repeats.dtype, types.Integer))
    ):  # pragma: no cover
        raise BodoError(
            "Index.repeat(): 'repeats' should be an integer or array of integers"
        )

    func_text = "def bodo_index_repeat(I, repeats, axis=None):\n"
    if not isinstance(repeats, types.Integer):
        func_text += "    repeats = bodo.utils.conversion.coerce_to_array(repeats)\n"
    if isinstance(I, RangeIndexType):
        func_text += "    arr = np.arange(I._start, I._stop, I._step)\n"
    else:
        func_text += "    arr = bodo.hiframes.pd_index_ext.get_index_data(I)\n"
    func_text += "    name = bodo.hiframes.pd_index_ext.get_index_name(I)\n"
    func_text += "    out_arr = bodo.libs.array_kernels.repeat_kernel(arr, repeats)\n"
    func_text += "    return constructor(out_arr, name)"

    constructor = (
        init_numeric_index
        if isinstance(I, RangeIndexType)
        else get_index_constructor(I)
    )
    return bodo_exec(
        func_text, {"bodo": bodo, "np": np, "constructor": constructor}, {}, __name__
    )


@overload_method(
    NumericIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
def overload_is_integer_numeric(I):
    truth = isinstance(I.dtype, types.Integer)
    return lambda I: truth  # pragma: no cover


@overload_method(
    NumericIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
def overload_is_floating_numeric(I):
    truth = isinstance(I.dtype, types.Float)
    return lambda I: truth  # pragma: no cover


@overload_method(
    NumericIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
def overload_is_boolean_numeric(I):
    truth = isinstance(I.dtype, types.Boolean)
    return lambda I: truth  # pragma: no cover


@overload_method(
    NumericIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
def overload_is_numeric_numeric(I):
    truth = not isinstance(I.dtype, types.Boolean)
    return lambda I: truth  # pragma: no cover


# TODO: fix for cases where I came from a pd.array of booleans


@overload_method(
    NumericIndexType, "is_object", inline="always", jit_options={"cache": True}
)
def overload_is_object_numeric(I):
    return lambda I: False  # pragma: no cover


@overload_method(
    StringIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_object", inline="always", jit_options={"cache": True}
)
def overload_is_methods_true(I):
    return lambda I: True  # pragma: no cover


@overload_method(
    NumericIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    NumericIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    StringIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    BinaryIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    DatetimeIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    TimedeltaIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    RangeIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    IntervalIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    CategoricalIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
@overload_method(
    PeriodIndexType, "is_object", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_boolean", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_floating", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_categorical", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_integer", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_interval", inline="always", jit_options={"cache": True}
)
@overload_method(
    MultiIndexType, "is_numeric", inline="always", jit_options={"cache": True}
)
def overload_is_methods_false(I):
    return lambda I: False  # pragma: no cover


# TODO(ehsan): test
@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
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
    if isinstance(ty.data, bodo.types.DatetimeArrayType):
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
        bodo.types.IntegerArrayType(types.int64),
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
    "append",
    "asof",
    "asof_locs",
    "astype",
    "delete",
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
    "is_",
    "is_mixed",
    "is_type_compatible",
    "item",
    "join",
    "memory_usage",
    "ravel",
    "reindex",
    "searchsorted",
    "set_names",
    "set_value",
    "shift",
    "slice_indexer",
    "slice_locs",
    "sort",
    "sortlevel",
    "str",
    "to_flat_index",
    "to_native_types",
    "transpose",
    "value_counts",
    "view",
]

index_unsupported_atrs = [
    "array",
    "asi8",
    "has_duplicates",
    "hasnans",
    "is_unique",
]

# unsupported RangeIndex class methods (handled in untyped pass)
# from_range

cat_idx_unsupported_atrs = [
    "codes",
    "categories",
    "ordered",
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
    "isin",
    "all",
    "any",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
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
    "nbytes",
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
    "dtype",
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
    "isin",
    "all",
    "any",
    "argsort",
    "sort_values",
    "argmax",
    "argmin",
    "where",
    "putmask",
    "nunique",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "to_series",
    "to_frame",
    "to_list",
    "tolist",
    "repeat",
    "min",
    "max",
]


multi_index_unsupported_atrs = [
    "levshape",
    "levels",
    "codes",
    "dtypes",
    "values",
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
    "isin",
    "unique",
    "all",
    "any",
    "argsort",
    "sort_values",
    "argmax",
    "argmin",
    "where",
    "putmask",
    "nunique",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "to_series",
    "to_list",
    "tolist",
    "to_numpy",
    "repeat",
    "min",
    "max",
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
    "all",
    "any",
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
    "all",
    "any",
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
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
    "dtype",
]

period_index_unsupported_methods = [
    "asfreq",
    "strftime",
    "to_timestamp",
    "isin",
    "unique",
    "all",
    "any",
    "where",
    "putmask",
    "sort_values",
    "union",
    "intersection",
    "difference",
    "symmetric_difference",
    "to_series",
    "to_frame",
    "to_numpy",
    "to_list",
    "tolist",
    "repeat",
    "min",
    "max",
]

string_index_unsupported_atrs = [
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

string_index_unsupported_methods = ["min", "max"]

binary_index_unsupported_atrs = [
    "is_monotonic_increasing",
    "is_monotonic_decreasing",
]

binary_index_unsupported_methods = ["repeat", "min", "max"]

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


def _split_idx_format_str(format_str):
    """splits format string from idx_typ_to_format_str_map into path_name, extra_info"""
    if " " not in format_str:
        return format_str, ""

    path_name = format_str[: format_str.index(" ")]
    extra_info = format_str[format_str.index(" ") :]

    return path_name, extra_info


def _install_index_unsupported():
    """install an overload that raises BodoError for unsupported methods of pd.Index"""

    # install unsupported methods that are common to all idx types
    for fname in index_unsupported_methods:
        for format_str, typ in index_types:
            format_str, extra_info = _split_idx_format_str(format_str)
            overload_unsupported_method(
                typ, fname, format_str.format(fname), extra_info=extra_info
            )

    # install unsupported attributes that are common to all idx types
    for attr_name in index_unsupported_atrs:
        for format_str, typ in index_types:
            format_str, extra_info = _split_idx_format_str(format_str)
            overload_unsupported_attribute(
                typ, attr_name, format_str.format(attr_name), extra_info=extra_info
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
        (BinaryIndexType, binary_index_unsupported_methods),
        (StringIndexType, string_index_unsupported_methods),
    ]

    # install unsupported methods for the individual idx types
    for typ, cur_typ_unsupported_methods_list in unsupported_methods_list:
        format_str = idx_typ_to_format_str_map[typ]
        for fname in cur_typ_unsupported_methods_list:
            format_str, extra_info = _split_idx_format_str(format_str)
            overload_unsupported_method(
                typ, fname, format_str.format(fname), extra_info=extra_info
            )

    # install unsupported attributes for the individual idx types
    for typ, cur_typ_unsupported_attrs_list in unsupported_attrs_list:
        format_str = idx_typ_to_format_str_map[typ]
        for attr_name in cur_typ_unsupported_attrs_list:
            format_str, extra_info = _split_idx_format_str(format_str)
            overload_unsupported_attribute(
                typ, attr_name, format_str.format(attr_name), extra_info=extra_info
            )


_install_index_unsupported()
