"""
Support for Series.dt attributes and methods
"""

import datetime
import operator

import numba
import numpy as np
from numba.core import cgutils, types
from numba.extending import (
    intrinsic,
    make_attribute_wrapper,
    models,
    overload_attribute,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_series_ext import (
    SeriesType,
    get_series_data,
    get_series_index,
    get_series_name,
    init_series,
)
from bodo.ir.argument_checkers import (
    DatetimeLikeSeriesArgumentChecker,
    OverloadArgumentsChecker,
    OverloadAttributeChecker,
)
from bodo.ir.declarative_templates import (
    overload_attribute_declarative,
    overload_method_declarative,
)
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)
from bodo.libs.pd_datetime_arr_ext import PandasDatetimeTZDtype
from bodo.utils.typing import (
    BodoError,
    ColNamesMetaType,
    check_unsupported_args,
    raise_bodo_error,
)

# global dtypes to use in allocations throughout this file
dt64_dtype = np.dtype("datetime64[ns]")
timedelta64_dtype = np.dtype("timedelta64[ns]")


class SeriesDatetimePropertiesType(types.Type):
    """accessor for datetime64/timedelta64 values
    (same as DatetimeProperties/TimedeltaProperties objects of Pandas)
    """

    # TODO: Timedelta and Period accessors
    def __init__(self, stype):
        self.stype = stype
        name = f"SeriesDatetimePropertiesType({stype})"
        super().__init__(name)


@register_model(SeriesDatetimePropertiesType)
class SeriesDtModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesDatetimePropertiesType, "obj", "_obj")


@intrinsic
def init_series_dt_properties(typingctx, obj):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        dt_properties_type = signature.return_type

        dt_properties_val = cgutils.create_struct_proxy(dt_properties_type)(
            context, builder
        )
        dt_properties_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return dt_properties_val._getvalue()

    return SeriesDatetimePropertiesType(obj)(obj), codegen


@overload_attribute(SeriesType, "dt", jit_options={"cache": True})
def overload_series_dt(s):
    if not (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(s)
        or bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(s)
    ):
        raise_bodo_error("Can only use .dt accessor with datetimelike values.")
    return lambda s: bodo.hiframes.series_dt_impl.init_series_dt_properties(
        s
    )  # pragma: no cover


def create_date_field_overload(field):
    def overload_field(S_dt):
        has_tz_aware_data = isinstance(S_dt.stype.dtype, PandasDatetimeTZDtype)

        func_text = f"def bodo_overload_field_{field}(S_dt):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(arr)\n"
        if field in (
            "is_leap_year",
            "is_month_start",
            "is_month_end",
            "is_quarter_start",
            "is_quarter_end",
            "is_year_start",
            "is_year_end",
        ):
            func_text += "    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)\n"
        else:
            func_text += (
                "    out_arr = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)\n"
            )

        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(arr, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(out_arr, i)\n"
        func_text += "            continue\n"
        if not has_tz_aware_data:
            func_text += "        dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])\n"
            func_text += "        ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n"
            if field == "weekday":
                func_text += "        out_arr[i] = ts.weekday()\n"
            else:
                func_text += "        out_arr[i] = ts." + field + "\n"
        else:
            call_params = "()" if field == "weekday" else ""
            func_text += f"        out_arr[i] = arr[i].{field}{call_params}\n"

        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
        )
        return bodo.utils.utils.bodo_exec(
            func_text, {"bodo": bodo, "numba": numba, "np": np}, {}, __name__
        )

    return overload_field


def overload_datetime_field_declarative(field, overload_impl):
    """
    Use declarative overload template to create an overload of dt fields that are only
    implemented for datetimes. This check is performed at compile time and is
    documented using overload_attribute_declarative.
    """
    overload_attribute_declarative(
        SeriesDatetimePropertiesType,
        field,
        path=f"pd.Series.dt.{field}",
        arg_checker=OverloadAttributeChecker(
            DatetimeLikeSeriesArgumentChecker("S_dt", type="datetime"),
        ),
        description=None,
        inline="always",
        jit_options={"cache": True},
    )(overload_impl)


def _install_date_fields():
    for field in bodo.hiframes.pd_timestamp_ext.date_fields:
        overload_impl = create_date_field_overload(field)
        # Using overload_attribute_declarative performs check that the Series data is
        # datetime at compile time and can be used to generate documentation.
        overload_datetime_field_declarative(field, overload_impl)


_install_date_fields()


def create_date_method_overload(method, is_str_method):
    if is_str_method:
        # Only string methods both have locale as an argument.
        func_text = "def overload_method(S_dt, locale=None):\n"
    else:
        func_text = "def overload_method(S_dt):\n"
    if is_str_method:
        func_text += "    def impl(S_dt, locale=None):\n"
    else:
        func_text += "    def impl(S_dt):\n"
    func_text += "        S = S_dt._obj\n"
    func_text += "        arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "        index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "        name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    func_text += "        numba.parfors.parfor.init_prange()\n"
    func_text += "        n = len(arr)\n"
    if is_str_method:
        func_text += "        out_arr = bodo.utils.utils.alloc_type(n, bodo.types.string_array_type, (-1,))\n"
    else:
        func_text += "        out_arr = np.empty(n, np.dtype('datetime64[ns]'))\n"
    func_text += "        for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "            if bodo.libs.array_kernels.isna(arr, i):\n"
    func_text += "                bodo.libs.array_kernels.setna(out_arr, i)\n"
    func_text += "                continue\n"
    func_text += "            ts = bodo.utils.conversion.box_if_dt64(arr[i])\n"
    func_text += f"            method_val = ts.{method}()\n"
    if is_str_method:
        func_text += "            out_arr[i] = method_val\n"
    else:
        func_text += "            out_arr[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(method_val.value)\n"
    func_text += (
        "        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )
    func_text += "    return impl\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, loc_vars)
    overload_method = loc_vars["overload_method"]
    return overload_method


def overload_datetime_method_declarative(method, overload_impl, unsupported_args):
    """
    Use declarative overload template to create an overload of dt method that are only
    implemented for datetimes. This check is performed at compile time and is
    documenteded using overload_method_declarative.
    """
    overload_method_declarative(
        SeriesDatetimePropertiesType,
        method,
        path=f"pd.Series.dt.{method}",
        unsupported_args=unsupported_args,
        method_args_checker=OverloadArgumentsChecker(
            [
                DatetimeLikeSeriesArgumentChecker(
                    "S_dt", type="datetime", is_self=True
                ),
            ]
        ),
        description=None,
        inline="always",
    )(overload_impl)


def _install_date_methods():
    for method in bodo.hiframes.pd_timestamp_ext.date_methods:
        is_str_method = method in ["day_name", "month_name"]
        overload_impl = create_date_method_overload(method, is_str_method)
        # Only string methods have locale as an argument.
        unsupported_args = {"locale"} if is_str_method else set()
        # Using overload_method_declarative performs check that the Series data is
        # datetime at compile time and can be used to generate documentation.
        overload_datetime_method_declarative(method, overload_impl, unsupported_args)


_install_date_methods()


@overload_attribute(SeriesDatetimePropertiesType, "date", jit_options={"cache": True})
def series_dt_date_overload(S_dt):
    if not (
        S_dt.stype.dtype == types.NPDatetime("ns")
        or isinstance(
            S_dt.stype.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
        )
    ):  # pragma: no cover
        return

    def impl(S_dt):  # pragma: no cover
        S = S_dt._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
        for i in numba.parfors.parfor.internal_prange(n):
            dt64_val = arr[i]
            ts = bodo.utils.conversion.box_if_dt64(dt64_val)
            out_arr[i] = datetime.date(ts.year, ts.month, ts.day)
        #        S[i] = datetime.date(ts.year, ts.month, ts.day)\n'
        #        S[i] = ts.day + (ts.month << 16) + (ts.year << 32)\n'
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


def create_series_dt_df_output_overload(attr):
    def series_dt_df_output_overload(S_dt):
        if not (
            (
                attr == "components"
                and S_dt.stype.dtype
                in (types.NPTimedelta("ns"), bodo.types.pd_timedelta_type)
            )
            or (
                attr == "isocalendar"
                and (
                    S_dt.stype.dtype == types.NPDatetime("ns")
                    or isinstance(S_dt.stype.dtype, PandasDatetimeTZDtype)
                )
            )
        ):  # pragma: no cover
            return

        has_tz_aware_data = isinstance(S_dt.stype.dtype, PandasDatetimeTZDtype)

        if attr == "components":
            fields = [
                "days",
                "hours",
                "minutes",
                "seconds",
                "milliseconds",
                "microseconds",
                "nanoseconds",
            ]
            convert = (
                "convert_numpy_timedelta64_to_pd_timedelta"
                if S_dt.stype.dtype == types.NPTimedelta("ns")
                else ""
            )
            int_type = "np.empty(n, np.int64)"
            attr_call = attr
        elif attr == "isocalendar":
            fields = ["year", "week", "day"]
            if has_tz_aware_data:
                # We only convert tz-naive data
                convert = None
            else:
                convert = "convert_datetime64_to_timestamp"
            int_type = "bodo.libs.int_arr_ext.alloc_int_array(n, np.uint32)"
            attr_call = attr + "()"

        func_text = "def impl(S_dt):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(arr)\n"
        for field in fields:
            func_text += f"    {field} = {int_type}\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(arr, i):\n"
        for field in fields:
            func_text += f"            bodo.libs.array_kernels.setna({field}, i)\n"
        func_text += "            continue\n"
        tuple_vals = "(" + "[i], ".join(fields) + "[i])"
        if convert:
            getitem_val = f"bodo.hiframes.pd_timestamp_ext.{convert}(arr[i])"
        else:
            getitem_val = "arr[i]"
        func_text += f"        {tuple_vals} = {getitem_val}.{attr_call}\n"
        arr_args = "(" + ", ".join(fields) + ")"
        func_text += f"    return bodo.hiframes.pd_dataframe_ext.init_dataframe({arr_args}, index, __col_name_meta_value_series_dt_df_output)\n"
        loc_vars = {}
        exec(
            func_text,
            {
                "bodo": bodo,
                "numba": numba,
                "np": np,
                "__col_name_meta_value_series_dt_df_output": ColNamesMetaType(
                    tuple(fields)
                ),
            },
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    return series_dt_df_output_overload


def _install_df_output_overload():
    df_outputs = [("components", overload_attribute), ("isocalendar", overload_method)]
    for attr, overload_type in df_outputs:
        overload_impl = create_series_dt_df_output_overload(attr)
        overload_type(SeriesDatetimePropertiesType, attr, inline="always")(
            overload_impl
        )


_install_df_output_overload()


# support Timedelta fields such as S.dt.days
def create_timedelta_field_overload(field):
    def overload_field(S_dt):
        if S_dt.stype.dtype not in (
            types.NPTimedelta("ns"),
            bodo.types.pd_timedelta_type,
        ):  # pragma: no cover
            return
        # TODO: refactor with TimedeltaIndex?
        func_text = "def impl(S_dt):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    A = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(A)\n"
        # all timedelta fields return int64
        func_text += "    B = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(A, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(B, i)\n"
        func_text += "            continue\n"
        func_text += "        td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(A[i])\n"
        if field == "nanoseconds":
            func_text += "        B[i] = td64 % 1000\n"
        elif field == "microseconds":
            func_text += "        B[i] = td64 // 1000 % 1000000\n"
        elif field == "seconds":
            func_text += "        B[i] = td64 // (1000 * 1000000) % (60 * 60 * 24)\n"
        elif field == "days":
            func_text += "        B[i] = td64 // (1000 * 1000000 * 60 * 60 * 24)\n"
        else:  # pragma: no cover
            assert False, "invalid timedelta field"
        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(B, index, name)\n"
        )
        loc_vars = {}
        exec(func_text, {"numba": numba, "np": np, "bodo": bodo}, loc_vars)
        impl = loc_vars["impl"]
        return impl

    return overload_field


# support Timedelta methods such as S.dt.total_seconds()
def create_timedelta_method_overload(method):
    def overload_method(S_dt):
        if S_dt.stype.dtype not in (
            types.NPTimedelta("ns"),
            bodo.types.pd_timedelta_type,
        ):  # pragma: no cover
            return
        # TODO: refactor with TimedeltaIndex?
        func_text = "def impl(S_dt):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    A = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(A)\n"
        # total_seconds returns a float64
        if method == "total_seconds":
            func_text += "    B = np.empty(n, np.float64)\n"
        # Only other method is to_pytimedelta, which is an arr of datetimes
        else:
            func_text += "    B = bodo.hiframes.datetime_timedelta_ext.alloc_timedelta_array(n)\n"

        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(A, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(B, i)\n"
        func_text += "            continue\n"
        # Convert the timedelta to its integer representation.
        # Then convert to a float
        func_text += "        td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(A[i])\n"
        if method == "total_seconds":
            func_text += "        B[i] = td64 / (1000.0 * 1000000.0)\n"
        elif method == "to_pytimedelta":
            # Convert td64 to microseconds
            func_text += (
                "        B[i] = datetime.timedelta(microseconds=td64 // 1000)\n"
            )
        else:  # pragma: no cover
            assert False, "invalid timedelta method"
        if method == "total_seconds":
            func_text += (
                "    return bodo.hiframes.pd_series_ext.init_series(B, index, name)\n"
            )
        else:
            func_text += "    return B\n"
        loc_vars = {}
        exec(
            func_text,
            {"numba": numba, "np": np, "bodo": bodo, "datetime": datetime},
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    return overload_method


def _install_S_dt_timedelta_fields():
    for field in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
        overload_impl = create_timedelta_field_overload(field)
        overload_attribute(SeriesDatetimePropertiesType, field)(overload_impl)


_install_S_dt_timedelta_fields()


def _install_S_dt_timedelta_methods():
    for method in bodo.hiframes.pd_timestamp_ext.timedelta_methods:
        overload_impl = create_timedelta_method_overload(method)
        overload_method(SeriesDatetimePropertiesType, method, inline="always")(
            overload_impl
        )


_install_S_dt_timedelta_methods()


@overload_method(
    SeriesDatetimePropertiesType,
    "strftime",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def dt_strftime(S_dt, date_format):
    if not (
        S_dt.stype.dtype == types.NPDatetime("ns")
        or isinstance(
            S_dt.stype.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
        )
    ):  # pragma: no cover
        return

    if types.unliteral(date_format) != types.unicode_type:
        raise BodoError(
            "Series.str.strftime(): 'date_format' argument must be a string"
        )

    def impl(S_dt, date_format):  # pragma: no cover
        S = S_dt._obj
        A = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(A)
        B = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
        for j in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(A, j):
                bodo.libs.array_kernels.setna(B, j)
                continue
            B[j] = bodo.utils.conversion.box_if_dt64(A[j]).strftime(date_format)
        return bodo.hiframes.pd_series_ext.init_series(B, index, name)

    return impl


@overload_method(
    SeriesDatetimePropertiesType,
    "tz_convert",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def overload_dt_tz_convert(S_dt, tz):
    def impl(S_dt, tz):
        S = S_dt._obj
        data = get_series_data(S).tz_convert(tz)
        index = get_series_index(S)
        name = get_series_name(S)
        return init_series(data, index, name)

    return impl


def create_timedelta_freq_overload(method):
    def freq_overload(S_dt, freq, ambiguous="raise", nonexistent="raise"):
        if (
            S_dt.stype.dtype
            not in (types.NPTimedelta("ns"), bodo.types.pd_timedelta_type)
            and S_dt.stype.dtype != types.NPDatetime("ns")
            and not isinstance(
                S_dt.stype.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
            )
        ):  # pragma: no cover
            return
        is_tz_aware = isinstance(
            S_dt.stype.dtype, bodo.libs.pd_datetime_arr_ext.PandasDatetimeTZDtype
        )
        unsupported_args = {"ambiguous": ambiguous, "nonexistent": nonexistent}
        floor_defaults = {"ambiguous": "raise", "nonexistent": "raise"}
        check_unsupported_args(
            f"Series.dt.{method}",
            unsupported_args,
            floor_defaults,
            package_name="pandas",
            module_name="Series",
        )
        func_text = "def impl(S_dt, freq, ambiguous='raise', nonexistent='raise'):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    A = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(A)\n"
        if S_dt.stype.dtype in (types.NPTimedelta("ns"), bodo.types.pd_timedelta_type):
            func_text += "    B = np.empty(n, np.dtype('timedelta64[ns]'))\n"
        elif is_tz_aware:
            func_text += "    B = bodo.libs.pd_datetime_arr_ext.alloc_pd_datetime_array(n, tz_literal)\n"
        else:
            func_text += "    B = np.empty(n, np.dtype('datetime64[ns]'))\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(A, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(B, i)\n"
        func_text += "            continue\n"
        if S_dt.stype.dtype in (types.NPTimedelta("ns"), bodo.types.pd_timedelta_type):
            front_convert = "bodo.hiframes.pd_timestamp_ext.convert_numpy_timedelta64_to_pd_timedelta"
            back_convert = "bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64"
        else:
            front_convert = (
                "bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp"
            )
            back_convert = "bodo.hiframes.pd_timestamp_ext.integer_to_dt64"
        if is_tz_aware:
            func_text += f"        B[i] = A[i].{method}(freq)\n"
        else:
            func_text += f"        B[i] = {back_convert}({front_convert}(A[i]).{method}(freq).value)\n"
        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(B, index, name)\n"
        )
        loc_vars = {}
        # Add the tz_literal to the globals for tz-aware data.
        tz_literal = None
        if is_tz_aware:
            tz_literal = S_dt.stype.dtype.tz
        exec(
            func_text,
            {"numba": numba, "np": np, "bodo": bodo, "tz_literal": tz_literal},
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    return freq_overload


def _install_S_dt_timedelta_freq_methods():
    freq_methods = ["ceil", "floor", "round"]
    for method in freq_methods:
        overload_impl = create_timedelta_freq_overload(method)
        overload_method(SeriesDatetimePropertiesType, method, inline="always")(
            overload_impl
        )


_install_S_dt_timedelta_freq_methods()


def create_bin_op_overload(op):
    """create overload function for binary operators
    with series(dt64)/series(timedelta) type
    """

    def overload_series_dt_binop(lhs, rhs):
        # lhs is series(dt64) and rhs is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            lhs
        ) and bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr1 = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr1 = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr1)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                data_arr2 = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr2 = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr2)
                n = len(arr1)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)

                for i in numba.parfors.parfor.internal_prange(n):
                    int_time1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr1[i])
                    int_time2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr2[i])
                    if int_time1 == nat_int or int_time2 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_time1, int_time2)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is series(dt64) and rhs is series(timedelta64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            lhs
        ) and bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                n = len(arr)
                S = np.empty(n, dt64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)

                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr2[i]
                    )
                    if int_dt64 == nat_int or int_td64 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64, int_td64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(ret_val)
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is series(timedelta64) and rhs is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            rhs
        ) and bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                n = len(arr)
                S = np.empty(n, dt64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)

                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr2[i]
                    )
                    if int_dt64 == nat_int or int_td64 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64, int_td64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(ret_val)
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is series(dt64) and rhs is timestamp
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs)
            and rhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                tsint = rhs.value
                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if int_dt64 == nat_int or tsint == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64, tsint)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is timestamp and rhs is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs)
            and lhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                tsint = lhs.value
                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if tsint == nat_int or int_dt64 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(tsint, int_dt64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is series(dt64) and rhs is datetime.timedelta
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs)
            and rhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                n = len(arr)
                S = np.empty(n, dt64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    rhs
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if int_dt64 == nat_int or int_td64 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64, int_td64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(ret_val)
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is datetime.timedelta and rhs is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs)
            and lhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                n = len(arr)
                S = np.empty(n, dt64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    lhs
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if int_dt64 == nat_int or int_td64 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64, int_td64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(ret_val)
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is series(dt64) and rhs is datetime.datetime
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs)
            and rhs == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                dt64 = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(rhs)
                int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dt64)
                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if int_dt64_2 == nat_int or int_dt64 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64_2, int_dt64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is datetime.datetime and rhs is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs)
            and lhs == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                dt64 = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(lhs)
                int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dt64)
                for i in numba.parfors.parfor.internal_prange(n):
                    int_dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if int_dt64 == nat_int or int_dt64_2 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_dt64, int_dt64_2)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is series(timedelta64) and rhs is datetime.timedelta
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs)
            and rhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = bodo.types.timedelta64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    rhs
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfors.parfor.internal_prange(n):
                    int_td64_2 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr[i]
                    )
                    if int_td64 == nat_int or int_td64_2 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_td64_2, int_td64)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # lhs is datetime.timedelta and rhs is series(timedelta64)
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs)
            and lhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = bodo.types.timedelta64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    lhs
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfors.parfor.internal_prange(n):
                    int_td64_2 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr[i]
                    )
                    if int_td64 == nat_int or int_td64_2 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_td64, int_td64_2)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        ret_val
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        raise BodoError(f"{op} not supported for data types {lhs} and {rhs}.")

    return overload_series_dt_binop


def create_cmp_op_overload(op):
    """create overload function for comparison operators with series(dt64) type"""

    def overload_series_dt64_cmp(lhs, rhs):
        if op == operator.ne:
            default_value = True
        else:
            default_value = False

        # lhs is series(timedelta) and rhs is timedelta
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(lhs)
            and rhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = bodo.types.timedelta64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(nat)
                td64_pre_2 = (
                    bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                        rhs
                    )
                )
                dt64_2 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                    td64_pre_2
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr[i]
                    )
                    if dt64_1 == nat_int or dt64_2 == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64_1, dt64_2)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # rhs is series(timedelta) and lhs is timedelta
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(rhs)
            and lhs == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = bodo.types.timedelta64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(nat)
                td64_pre_1 = (
                    bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                        lhs
                    )
                )
                dt64_1 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                    td64_pre_1
                )
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_2 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr[i]
                    )
                    if dt64_1 == nat_int or dt64_2 == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64_1, dt64_2)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # lhs is series(dt64) and rhs is timestamp
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs)
            and rhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64_1 == nat_int or rhs.value == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64_1, rhs.value)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # lhs is timestamp and rhs is series(dt64)
        if (
            lhs == bodo.hiframes.pd_timestamp_ext.pd_timestamp_tz_naive_type
            and bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs)
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64_2 == nat_int or lhs.value == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(lhs.value, dt64_2)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # lhs is series(dt64) and rhs is string
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs) and (
            rhs == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(rhs)
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(lhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(lhs)
                numba.parfors.parfor.init_prange()
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                string_to_dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(rhs)
                date = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(string_to_dt64)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64_1 == nat_int or date == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64_1, date)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # lhs is string and rhs is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs) and (
            lhs == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(lhs)
        ):
            nat = bodo.types.datetime64ns("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                data_arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
                arr = bodo.libs.pd_datetime_arr_ext.unwrap_tz_array(data_arr)
                index = bodo.hiframes.pd_series_ext.get_series_index(rhs)
                name = bodo.hiframes.pd_series_ext.get_series_name(rhs)
                numba.parfors.parfor.init_prange()
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                string_to_dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(lhs)
                date = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(string_to_dt64)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64 == nat_int or date == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(date, dt64)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        raise BodoError(f"{op} operator not supported for data types {lhs} and {rhs}.")

    return overload_series_dt64_cmp


series_dt_unsupported_methods = {
    "to_period",
    "to_pydatetime",
    "tz_localize",
    "asfreq",
    "to_timestamp",
    "as_unit",
}

series_dt_unsupported_attrs = {
    # attributes
    "time",
    "timetz",
    "tz",
    "freq",
    # Properties
    "qyear",
    "start_time",
    "end_time",
    "unit",
}


def _install_series_dt_unsupported():
    """install an overload that raises BodoError for unsupported methods of Series.dt"""

    for attr_name in series_dt_unsupported_attrs:
        full_name = "Series.dt." + attr_name
        overload_unsupported_attribute(
            SeriesDatetimePropertiesType, attr_name, full_name
        )

    for fname in series_dt_unsupported_methods:
        full_name = "Series.dt." + fname
        overload_unsupported_method(SeriesDatetimePropertiesType, fname, full_name)


_install_series_dt_unsupported()
