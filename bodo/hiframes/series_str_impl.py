"""
Support for Series.str methods
"""
import operator
import numpy as np
import pandas as pd
import numba
from numba import types, cgutils
from numba.extending import (models, register_model, infer_getattr,
    overload, overload_method, make_attribute_wrapper, intrinsic,
    overload_attribute)
from numba.typing.templates import (infer_global, AbstractTemplate, signature,
    AttributeTemplate, bound_function)
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.pd_timestamp_ext import (pandas_timestamp_type,
    convert_datetime64_to_timestamp, convert_timestamp_to_datetime64,
    integer_to_dt64)
from bodo.hiframes.pd_index_ext import NumericIndexType, RangeIndexType
from bodo.utils.typing import is_list_like_index_type


str2str_methods = ('capitalize', 'lower', 'lstrip', 'rstrip',
            'strip', 'swapcase', 'title', 'upper')


class SeriesStrMethodType(types.Type):
    def __init__(self, stype):
        # keeping Series type since string data representation can be varied
        self.stype = stype
        name = "SeriesStrMethodType({})".format(stype)
        super(SeriesStrMethodType, self).__init__(name)


@register_model(SeriesStrMethodType)
class SeriesStrModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('obj', fe_type.stype),
        ]
        super(SeriesStrModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesStrMethodType, 'obj', '_obj')


@intrinsic
def init_series_str_method(typingctx, obj=None):

    def codegen(context, builder, signature, args):
        obj_val, = args
        str_method_type = signature.return_type

        str_method_val = cgutils.create_struct_proxy(str_method_type)(
            context, builder)
        str_method_val.obj = obj_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return str_method_val._getvalue()

    return SeriesStrMethodType(obj)(obj), codegen


@overload_attribute(SeriesType, 'str')
def overload_series_str(s):
    return lambda s: bodo.hiframes.series_str_impl.init_series_str_method(s)


@overload_method(SeriesStrMethodType, 'len')
def overload_str_method_len(S_str):
    def impl(S_str):
        S = S_str._obj
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        numba.parfor.init_prange()
        n = len(arr)
        n_bytes = (n + 7) >> 3
        out_arr = np.empty(n, np.int64)
        bitmap = np.empty(n_bytes, np.uint8)
        for i in numba.parfor.internal_prange(n):
            if bodo.hiframes.api.isna(arr, i):
                out_arr[i] = 1
                bodo.libs.int_arr_ext.set_bit_to_arr(
                        bitmap, i, 0)
            else:
                # TODO: optimize str len on string array
                out_arr[i] = len(arr[i])
                bodo.libs.int_arr_ext.set_bit_to_arr(
                            bitmap, i, 1)

        return bodo.hiframes.api.init_series(
            bodo.libs.int_arr_ext.init_integer_array(out_arr, bitmap),
            index, name)

    return impl
