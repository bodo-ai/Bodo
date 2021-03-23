# Copyright (C) 2019 Bodo Inc. All rights reserved.
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
    overload,
    overload_attribute,
    overload_method,
    register_model,
)

import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    create_unsupported_overload,
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
        name = "SeriesDatetimePropertiesType({})".format(stype)
        super(SeriesDatetimePropertiesType, self).__init__(name)


@register_model(SeriesDatetimePropertiesType)
class SeriesDtModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super(SeriesDtModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesDatetimePropertiesType, "obj", "_obj")


@intrinsic
def init_series_dt_properties(typingctx, obj=None):
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


@overload_attribute(SeriesType, "dt")
def overload_series_dt(s):
    if not (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(s)
        or bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(s)
    ):
        raise_bodo_error("Can only use .dt accessor with datetimelike values.")
    return lambda s: bodo.hiframes.series_dt_impl.init_series_dt_properties(s)


def create_date_field_overload(field):
    def overload_field(S_dt):
        if not S_dt.stype.dtype == types.NPDatetime("ns"):  # pragma: no cover
            return
        func_text = "def impl(S_dt):\n"
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
            func_text += "    out_arr = np.empty(n, np.bool_)\n"
        else:
            func_text += (
                "    out_arr = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)\n"
            )

        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(arr, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(out_arr, i)\n"
        func_text += "            continue\n"
        func_text += (
            "        dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])\n"
        )
        # extract year, month, day faster without conversion to Timestamp
        if field in ("year", "month", "day"):
            func_text += "        dt, year, days = bodo.hiframes.pd_timestamp_ext.extract_year_days(dt64)\n"
            if field in ("month", "day"):
                func_text += "        month, day = bodo.hiframes.pd_timestamp_ext.get_month_day(year, days)\n"
            func_text += "        out_arr[i] = {}\n".format(field)
        elif field in (
            "dayofyear",
            "dayofweek",
            "weekday",
        ):
            funcdict = {
                "dayofyear": "get_day_of_year",
                "dayofweek": "get_day_of_week",
                "weekday": "get_day_of_week",
            }
            func_text += "        dt, year, days = bodo.hiframes.pd_timestamp_ext.extract_year_days(dt64)\n"
            func_text += "        month, day = bodo.hiframes.pd_timestamp_ext.get_month_day(year, days)\n"
            func_text += "        out_arr[i] = bodo.hiframes.pd_timestamp_ext.{}(year, month, day)\n".format(
                funcdict[field]
            )
        elif field == "is_leap_year":
            func_text += "        dt, year, days = bodo.hiframes.pd_timestamp_ext.extract_year_days(dt64)\n"
            func_text += "        out_arr[i] = bodo.hiframes.pd_timestamp_ext.is_leap_year(year)\n"
        elif field in ("daysinmonth", "days_in_month"):
            funcdict = {
                "days_in_month": "get_days_in_month",
                "daysinmonth": "get_days_in_month",
            }
            func_text += "        dt, year, days = bodo.hiframes.pd_timestamp_ext.extract_year_days(dt64)\n"
            func_text += "        month, day = bodo.hiframes.pd_timestamp_ext.get_month_day(year, days)\n"
            func_text += "        out_arr[i] = bodo.hiframes.pd_timestamp_ext.{}(year, month)\n".format(
                funcdict[field]
            )
        else:
            func_text += "        ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)\n"
            func_text += "        out_arr[i] = ts." + field + "\n"
        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, loc_vars)
        impl = loc_vars["impl"]
        return impl

    return overload_field


def _install_date_fields():
    for field in bodo.hiframes.pd_timestamp_ext.date_fields:
        overload_impl = create_date_field_overload(field)
        overload_attribute(SeriesDatetimePropertiesType, field)(overload_impl)


_install_date_fields()


def create_date_method_overload(method):
    def overload_method(S_dt):
        """.dt methods on only datetime64s. Only .normalize is supported so far."""
        if not S_dt.stype.dtype == types.NPDatetime("ns"):  # pragma: no cover
            return
        func_text = "def impl(S_dt):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(arr)\n"
        func_text += "    out_arr = np.empty(n, np.dtype('datetime64[ns]'))\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(arr, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(out_arr, i)\n"
        func_text += "            continue\n"
        func_text += "        ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(arr[i])\n"
        func_text += f"        out_arr[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(ts.{method}().value)\n"
        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, loc_vars)
        impl = loc_vars["impl"]
        return impl

    return overload_method


def _install_date_methods():
    for method in bodo.hiframes.pd_timestamp_ext.date_methods:
        overload_impl = create_date_method_overload(method)
        overload_method(SeriesDatetimePropertiesType, method, inline="always")(
            overload_impl
        )


_install_date_methods()


@overload_attribute(SeriesDatetimePropertiesType, "date")
def series_dt_date_overload(S_dt):
    if not S_dt.stype.dtype == types.NPDatetime("ns"):  # pragma: no cover
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
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            out_arr[i] = datetime.date(ts.year, ts.month, ts.day)
        #        S[i] = datetime.date(ts.year, ts.month, ts.day)\n'
        #        S[i] = ts.day + (ts.month << 16) + (ts.year << 32)\n'
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


def create_series_dt_df_output_overload(attr):
    def series_dt_df_output_overload(S_dt):
        if not (
            (attr == "components" and S_dt.stype.dtype == types.NPTimedelta("ns"))
            or (attr == "isocalendar" and S_dt.stype.dtype == types.NPDatetime("ns"))
        ):  # pragma: no cover
            return

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
            convert = "convert_numpy_timedelta64_to_pd_timedelta"
            int_type = "np.empty(n, np.int64)"
            attr_call = attr
        elif attr == "isocalendar":
            fields = ["year", "week", "day"]
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
            func_text += "    {} = {}\n".format(field, int_type)
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(arr, i):\n"
        for field in fields:
            func_text += "            bodo.libs.array_kernels.setna({}, i)\n".format(
                field
            )
        func_text += "            continue\n"
        tuple_vals = "(" + "[i], ".join(fields) + "[i])"
        func_text += (
            "        {} = bodo.hiframes.pd_timestamp_ext.{}(arr[i]).{}\n".format(
                tuple_vals, convert, attr_call
            )
        )
        arr_args = "(" + ", ".join(fields) + ")"
        str_args = "('" + "', '".join(fields) + "')"
        func_text += "    return bodo.hiframes.pd_dataframe_ext.init_dataframe({}, index, {})\n".format(
            arr_args, str_args
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "numba": numba, "np": np}, loc_vars)
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
        if not S_dt.stype.dtype == types.NPTimedelta("ns"):  # pragma: no cover
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
        if not S_dt.stype.dtype == types.NPTimedelta("ns"):  # pragma: no cover
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
            func_text += "    B = bodo.hiframes.datetime_timedelta_ext.alloc_datetime_timedelta_array(n)\n"

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
    SeriesDatetimePropertiesType, "strftime", inline="always", no_unliteral=True
)
def dt_strftime(S_dt, format_str):
    if S_dt.stype.dtype != types.NPDatetime("ns"):  # pragma: no cover
        return

    def impl(S_dt, format_str):  # pragma: no cover
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
            B[j] = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(
                A[j]
            ).strftime(format_str)
        return bodo.hiframes.pd_series_ext.init_series(B, index, name)

    return impl


def create_timedelta_freq_overload(method):
    def freq_overload(S_dt, freq, ambiguous="raise", nonexistent="raise"):
        if S_dt.stype.dtype != types.NPTimedelta(
            "ns"
        ) and S_dt.stype.dtype != types.NPDatetime(
            "ns"
        ):  # pragma: no cover
            return
        unsupported_args = dict(ambiguous=ambiguous, nonexistent=nonexistent)
        floor_defaults = dict(ambiguous="raise", nonexistent="raise")
        check_unsupported_args("floor", unsupported_args, floor_defaults)
        func_text = "def impl(S_dt, freq, ambiguous='raise', nonexistent='raise'):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    A = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(A)\n"
        if S_dt.stype.dtype == types.NPTimedelta("ns"):
            func_text += "    B = np.empty(n, np.dtype('timedelta64[ns]'))\n"
        else:
            func_text += "    B = np.empty(n, np.dtype('datetime64[ns]'))\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(A, i):\n"
        func_text += "            bodo.libs.array_kernels.setna(B, i)\n"
        func_text += "            continue\n"
        if S_dt.stype.dtype == types.NPTimedelta("ns"):
            front_convert = "bodo.hiframes.pd_timestamp_ext.convert_numpy_timedelta64_to_pd_timedelta"
            back_convert = "bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64"
        else:
            front_convert = (
                "bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp"
            )
            back_convert = "bodo.hiframes.pd_timestamp_ext.integer_to_dt64"
        func_text += "        B[i] = {}({}(A[i]).{}(freq).value)\n".format(
            back_convert, front_convert, method
        )
        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(B, index, name)\n"
        )
        loc_vars = {}
        exec(
            func_text,
            {"numba": numba, "np": np, "bodo": bodo},
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

    def overload_series_dt_binop(A1, A2):

        # A1 is series(dt64) and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A1
        ) and bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr1 = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(A2)
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

        # A1 is series(dt64) and A2 is series(timedelta64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A1
        ) and bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A2):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(A2)
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

        # A1 is series(timedelta64) and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A2
        ) and bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A1):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(A1)
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

        # A1 is series(dt64) and A2 is timestamp
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
        ):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                tsint = A2.value
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

        # A1 is timestamp and A2 is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
            and A1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
        ):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                tsint = A1.value
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

        # A1 is series(dt64) and A2 is datetime.timedelta
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = np.empty(n, dt64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A2
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

        # A1 is datetime.timedelta and A2 is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
            and A1 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = np.empty(n, dt64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A1
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

        # A1 is series(dt64) and A2 is datetime.datetime
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type
        ):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                dt64 = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(A2)
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

        # A1 is datetime.datetime and A2 is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
            and A1 == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type
        ):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                dt64 = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(A1)
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

        # A1 is series(timedelta64) and A2 is datetime.timedelta
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A1)
            and A2 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A2
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

        # A1 is datetime.timedelta and A2 is series(timedelta64)
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A2)
            and A1 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = np.empty(n, timedelta64_dtype)
                nat_int = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(nat)
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A1
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
            nat = lhs.dtype("NaT")

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
            nat = rhs.dtype("NaT")

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
            and rhs == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
        ):
            nat = lhs.dtype("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
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
            lhs == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
            and bodo.hiframes.pd_series_ext.is_dt64_series_typ(rhs)
        ):
            nat = rhs.dtype("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
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
                        ret_val = op(dt64_2, lhs.value)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # lhs is series(dt64) and rhs is string
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(lhs) and (
            rhs == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(rhs)
        ):
            nat = lhs.dtype("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(lhs)
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
            nat = rhs.dtype("NaT")

            def impl(lhs, rhs):  # pragma: no cover
                arr = bodo.hiframes.pd_series_ext.get_series_data(rhs)
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


def _install_bin_ops():
    """install overloads for operators with series(dt64) type"""
    for op in (operator.add, operator.sub):
        overload_impl = create_bin_op_overload(op)
        overload(op, no_unliteral=True)(overload_impl)


_install_bin_ops()

series_dt_unsupported_methods = {
    "asfreq",
    "day_name",
    "month_name",
    "normalize",
    "to_period",
    "to_pydatetime",
    "to_timestamp",
    "tz_convert",
    "tz_localize",
}

series_dt_unsupported_attrs = {
    "end_time",
    "freq",
    "qyear",
    "start_time",
    "time",
    "timetz",
    "tz",
    "week",
    "weekday",
    "weekofyear",
}


def _install_series_dt_unsupported():
    """install an overload that raises BodoError for unsupported methods of Series.dt """

    for attr_name in series_dt_unsupported_attrs:
        full_name = "Series.dt." + attr_name
        overload_attribute(SeriesDatetimePropertiesType, attr_name)(
            create_unsupported_overload(full_name)
        )

    for fname in series_dt_unsupported_methods:
        full_name = "Series.dt." + fname
        overload_method(SeriesDatetimePropertiesType, fname, no_unliteral=True)(
            create_unsupported_overload(full_name)
        )


_install_series_dt_unsupported()
