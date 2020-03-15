# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Support for Series.dt attributes and methods
"""
import operator
import datetime
import numpy as np
import pandas as pd
import numba
from numba import types, cgutils
from numba.extending import (
    models,
    register_model,
    infer_getattr,
    overload,
    overload_method,
    make_attribute_wrapper,
    intrinsic,
    overload_attribute,
)
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import (
    pandas_timestamp_type,
    convert_datetime64_to_timestamp,
    integer_to_dt64,
)
from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
from bodo.utils.typing import (
    BodoError,
    is_list_like_index_type,
    is_overload_false,
    is_overload_true,
)
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type, pre_alloc_string_array


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
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return dt_properties_val._getvalue()

    return SeriesDatetimePropertiesType(obj)(obj), codegen


@overload_attribute(SeriesType, "dt")
def overload_series_dt(s):
    if not (
        bodo.hiframes.pd_series_ext.is_dt64_series_typ(s)
        or bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(s)
    ):
        raise BodoError("Can only use .dt accessor with datetimelike values.")
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
        func_text += "    numba.parfor.init_prange()\n"
        func_text += "    n = len(arr)\n"
        func_text += "    out_arr = np.empty(n, np.int64)\n"
        func_text += "    for i in numba.parfor.internal_prange(n):\n"
        func_text += (
            "        dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])\n"
        )
        # extract year, month, day faster without conversion to Timestamp
        if field in ("year", "month", "day"):
            func_text += "        dt, year, days = bodo.hiframes.pd_timestamp_ext.extract_year_days(dt64)\n"
            if field in ("month", "day"):
                func_text += "        month, day = bodo.hiframes.pd_timestamp_ext.get_month_day(year, days)\n"
            func_text += "        out_arr[i] = {}\n".format(field)
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


@overload_attribute(SeriesDatetimePropertiesType, "date")
def series_dt_date_overload(S_dt):
    if not S_dt.stype.dtype == types.NPDatetime("ns"):  # pragma: no cover
        return

    def impl(S_dt):  # pragma: no cover
        S = S_dt._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfor.init_prange()
        n = len(arr)
        out_arr = bodo.hiframes.datetime_date_ext.alloc_datetime_date_array(n)
        for i in numba.parfor.internal_prange(n):
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            out_arr[i] = datetime.date(ts.year, ts.month, ts.day)
        #        S[i] = datetime.date(ts.year, ts.month, ts.day)\n'
        #        S[i] = ts.day + (ts.month << 16) + (ts.year << 32)\n'
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


# support Timedelta fields such as S.dt.days
def create_timedelta_field_overload(field):
    def overload_field(S_dt):
        if not S_dt.stype.dtype == types.NPTimedelta("ns"):  # pragma: no cover
            return
        # TODO: refactor with TimedeltaIndex?
        # TODO: NAs
        func_text = "def impl(S_dt):\n"
        func_text += "    S = S_dt._obj\n"
        func_text += "    A = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfor.init_prange()\n"
        func_text += "    n = len(A)\n"
        # all timedelta fields return int64
        func_text += "    B = np.empty(n, np.int64)\n"
        func_text += "    for i in numba.parfor.internal_prange(n):\n"
        func_text += "        td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(A[i])\n"
        if field == "nanoseconds":
            func_text += "        B[i] = td64 % 1000\n"
        elif field == "microseconds":
            func_text += "        B[i] = td64 // 1000 % 100000\n"
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


def _install_S_dt_timedelta_fields():
    for field in bodo.hiframes.pd_timestamp_ext.timedelta_fields:
        overload_impl = create_timedelta_field_overload(field)
        overload_attribute(SeriesDatetimePropertiesType, field)(overload_impl)


_install_S_dt_timedelta_fields()


def create_bin_op_overload(op):
    """create overload function for binary operators 
    with series(dt64)/series(timedelta) type
    """

    def overload_series_dt_binop(A1, A2):

        # A1 is series(dt64) and A2 is series(timedelta64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A1
        ) and bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A2):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(A2)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))

                for i in numba.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr2[i]
                    )
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        op(int_dt64, int_td64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is series(timedelta64) and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A2
        ) and bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A1):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                arr2 = bodo.hiframes.pd_series_ext.get_series_data(A1)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))

                for i in numba.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr2[i]
                    )
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        op(int_dt64, int_td64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is series(dt64) and A2 is timestamp
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                tsint = A2.value
                for i in numba.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        op(int_dt64, tsint)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is timestamp and A2 is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
            and A1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                tsint = A1.value
                for i in numba.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        op(tsint, int_dt64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is series(dt64) and A2 is datetime.timedelta
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A2
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        op(int_dt64, int_td64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is datetime.timedelta and A2 is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
            and A1 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A1
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfor.internal_prange(n):
                    int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(
                        op(int_dt64, int_td64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is series(dt64) and A2 is datetime.datetime
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                dt64 = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(A2)
                int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dt64)
                for i in numba.parfor.internal_prange(n):
                    int_dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        op(int_dt64_2, int_dt64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is datetime.datetime and A2 is series(dt64)
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
            and A1 == bodo.hiframes.datetime_datetime_ext.datetime_datetime_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                dt64 = bodo.hiframes.pd_timestamp_ext.datetime_datetime_to_dt64(A1)
                int_dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(dt64)
                for i in numba.parfor.internal_prange(n):
                    int_dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        op(int_dt64, int_dt64_2)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is series(timedelta64) and A2 is datetime.timedelta
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A1)
            and A2 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A2
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfor.internal_prange(n):
                    int_td64_2 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr[i]
                    )
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        op(int_td64_2, int_td64)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

        # A1 is datetime.timedelta and A2 is series(timedelta64)
        if (
            bodo.hiframes.pd_series_ext.is_timedelta64_series_typ(A2)
            and A1 == bodo.hiframes.datetime_timedelta_ext.datetime_timedelta_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                S = numba.unsafe.ndarray.empty_inferred((n,))
                td64 = bodo.hiframes.pd_timestamp_ext.datetime_timedelta_to_timedelta64(
                    A1
                )
                int_td64 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(td64)
                for i in numba.parfor.internal_prange(n):
                    int_td64_2 = bodo.hiframes.pd_timestamp_ext.timedelta64_to_integer(
                        arr[i]
                    )
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_timedelta64(
                        op(int_td64, int_td64_2)
                    )
                return bodo.hiframes.pd_series_ext.init_series(S, index, name)

            return impl

    return overload_series_dt_binop


def create_cmp_op_overload(op):
    """create overload function for comparison operators with series(dt64) type
    """

    def overload_series_dt64_cmp(A1, A2):
        # A1 is series(dt64) and A2 is timestamp
        if (
            bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1)
            and A2 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    out_arr[i] = op(dt64, A2.value)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is timestamp and A2 is series(dt64)
        if (
            A1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
            and bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    out_arr[i] = op(dt64, A1.value)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is series(dt64) and A2 is string
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1) and (
            A2 == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(A2)
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                string_to_dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(A2)
                date = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(string_to_dt64)
                for i in numba.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    out_arr[i] = op(dt64, date)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is string and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2) and (
            A1 == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(A1)
        ):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                string_to_dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(A1)
                date = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(string_to_dt64)
                for i in numba.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    out_arr[i] = op(date, dt64)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is series(dt64) and A2 is dt64
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A1
        ) and A2 == types.NPDatetime("ns"):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A2)
                    out_arr[i] = op(dt64_1, dt64_2)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is dt64 and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A2
        ) and A1 == types.NPDatetime("ns"):

            def impl(A1, A2):  # pragma: no cover
                numba.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                for i in numba.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A1)
                    out_arr[i] = op(dt64_2, dt64_1)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

    return overload_series_dt64_cmp


def _install_cmp_ops():
    """install overloads for comparison operators with series(dt64) type
    """
    for op in (
        operator.eq,
        operator.ne,
        operator.ge,
        operator.gt,
        operator.le,
        operator.lt,
    ):
        overload_impl = create_cmp_op_overload(op)
        overload(op)(overload_impl)


_install_cmp_ops()


def _install_bin_ops():
    """install overloads for operators with series(dt64) type
    """
    for op in (operator.add, operator.sub):
        overload_impl = create_bin_op_overload(op)
        overload(op)(overload_impl)


_install_bin_ops()
