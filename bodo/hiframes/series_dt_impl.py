# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Support for Series.dt attributes and methods
"""
import operator
import datetime
import numpy as np
import numba
from numba.core import types, cgutils
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
from numba.core.typing.templates import (
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
from bodo.utils.typing import (
    BodoError,
    is_list_like_index_type,
    is_overload_false,
    is_overload_true,
)


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
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(arr)\n"
        func_text += "    out_arr = np.empty(n, np.int64)\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
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
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(A)\n"
        # all timedelta fields return int64
        func_text += "    B = np.empty(n, np.int64)\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)

                for i in numba.parfors.parfor.internal_prange(n):
                    int_time1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr1[i])
                    int_time2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr2[i])
                    if int_time1 == nat_int or int_time2 == nat_int:
                        ret_val = nat_int
                    else:
                        ret_val = op(int_time1, int_time2)
                    S[i] = bodo.hiframes.pd_timestamp_ext.integer_to_dt64(ret_val)
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
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
                S = numba.np.unsafe.ndarray.empty_inferred((n,))
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
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
    """create overload function for comparison operators with series(dt64) type
    """

    def overload_series_dt64_cmp(A1, A2):
        if op == operator.ne:
            default_value = True
        else:
            default_value = False

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
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64 == nat_int or A2.value == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64, A2.value)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is timestamp and A2 is series(dt64)
        if (
            A1 == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type
            and bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2)
        ):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64 == nat_int or A1.value == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64, A1.value)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is series(dt64) and A2 is string
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(A1) and (
            A2 == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(A2)
        ):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                string_to_dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(A2)
                date = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(string_to_dt64)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    if dt64 == nat_int or date == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64, date)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is string and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(A2) and (
            A1 == bodo.libs.str_ext.string_type
            or bodo.utils.typing.is_overload_constant_str(A1)
        ):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                string_to_dt64 = bodo.hiframes.pd_timestamp_ext.parse_datetime_str(A1)
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

        # A1 is series(dt64) and A2 is dt64
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A1
        ) and A2 == types.NPDatetime("ns"):
            nat = A1.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A1)
                index = bodo.hiframes.pd_series_ext.get_series_index(A1)
                name = bodo.hiframes.pd_series_ext.get_series_name(A1)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A2)
                    if dt64_1 == nat_int or dt64_2 == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64_1, dt64_2)
                    out_arr[i] = ret_val
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return impl

        # A1 is dt64 and A2 is series(dt64)
        if bodo.hiframes.pd_series_ext.is_dt64_series_typ(
            A2
        ) and A1 == types.NPDatetime("ns"):
            nat = A2.dtype("NaT")

            def impl(A1, A2):  # pragma: no cover
                numba.parfors.parfor.init_prange()
                arr = bodo.hiframes.pd_series_ext.get_series_data(A2)
                index = bodo.hiframes.pd_series_ext.get_series_index(A2)
                name = bodo.hiframes.pd_series_ext.get_series_name(A2)
                n = len(arr)
                out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(n)
                nat_int = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(nat)
                for i in numba.parfors.parfor.internal_prange(n):
                    dt64_1 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
                    dt64_2 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(A1)
                    if dt64_1 == nat_int or dt64_2 == nat_int:
                        ret_val = default_value
                    else:
                        ret_val = op(dt64_2, dt64_1)
                    out_arr[i] = ret_val
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
