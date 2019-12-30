# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Support for Series.dt methods
"""
import operator
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
    convert_timestamp_to_datetime64,
    integer_to_dt64,
)
from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
from bodo.utils.typing import (
    is_list_like_index_type,
    is_overload_false,
    is_overload_true,
)
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import (
    string_array_type,
    pre_alloc_string_array,
    get_utf8_size,
)


class SeriesDatetimePropertiesType(types.Type):
    """accessor for datetime64 values (same as DatetimeProperties object of Pandas)
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
        obj_val, = args
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
    return lambda s: bodo.hiframes.series_dt_impl.init_series_dt_properties(s)


def create_date_field_overload(field):
    def overload_field(S_dt):
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
    def impl(S_dt):  # pragma: no cover
        S = S_dt._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfor.init_prange()
        n = len(arr)
        out_arr = numba.unsafe.ndarray.empty_inferred((n,))
        for i in numba.parfor.internal_prange(n):
            dt64 = bodo.hiframes.pd_timestamp_ext.dt64_to_integer(arr[i])
            ts = bodo.hiframes.pd_timestamp_ext.convert_datetime64_to_timestamp(dt64)
            out_arr[i] = bodo.hiframes.pd_timestamp_ext.datetime_date_ctor(
                ts.year, ts.month, ts.day
            )
        #        S[i] = datetime.date(ts.year, ts.month, ts.day)\n'
        #        S[i] = ts.day + (ts.month << 16) + (ts.year << 32)\n'
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl
