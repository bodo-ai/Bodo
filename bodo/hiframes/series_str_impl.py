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
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import (string_array_type, pre_alloc_string_array,
    get_utf8_size)
from bodo.hiframes.split_impl import (string_array_split_view_type,
    getitem_c_arr, get_array_ctypes_ptr,
    get_split_view_index, get_split_view_data_ptr)


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


@overload_method(SeriesStrMethodType, 'split')
def overload_str_method_split(S_str, pat=None, n=-1, expand=False):
    # TODO: support or just check n and expand arguments
    # TODO: support distributed

    # use split view if sep is a string of length 1
    if isinstance(pat, types.StringLiteral) and len(pat.literal_value) == 1:
        def _str_split_view_impl(S_str, pat=None, n=-1, expand=False):
            S = S_str._obj
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            out_arr = bodo.hiframes.split_impl.compute_split_view(arr, pat)
            return bodo.hiframes.api.init_series(out_arr, index, name)

        return _str_split_view_impl

    def _str_split_impl(S_str, pat=None, n=-1, expand=False):
        S = S_str._obj
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        numba.parfor.init_prange()
        n = len(arr)
        out_arr = bodo.libs.str_ext.alloc_list_list_str(n)
        for i in numba.parfor.internal_prange(n):
            in_str = arr[i]
            out_arr[i] = in_str.split(pat)

        return bodo.hiframes.api.init_series(out_arr, index, name)

    return _str_split_impl


@overload_method(SeriesStrMethodType, 'get')
def overload_str_method_get(S_str, i):
    arr_typ = S_str.stype.data
    # XXX only supports get for list(list(str)) input and split view
    assert (arr_typ == types.List(types.List(string_type))
        or arr_typ == string_array_split_view_type)

    # TODO: support and test NA
    # TODO: support distributed

    if arr_typ == string_array_split_view_type:
        # TODO: refactor and enable distributed
        def _str_get_split_impl(S_str, i):
            S = S_str._obj
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            numba.parfor.init_prange()
            n = len(arr)
            n_total_chars = 0
            for k in numba.parfor.internal_prange(n):
                data_start, length = get_split_view_index(arr, k, i)
                n_total_chars += length
            numba.parfor.init_prange()
            out_arr = pre_alloc_string_array(n, n_total_chars)
            for j in numba.parfor.internal_prange(n):
                data_start, length = get_split_view_index(arr, j, i)
                ptr = get_split_view_data_ptr(arr, data_start)
                bodo.libs.str_arr_ext.setitem_str_arr_ptr(
                    out_arr, j, ptr, length)
            return bodo.hiframes.api.init_series(out_arr, index, name)
        return _str_get_split_impl

    def _str_get_impl(S_str, i):
        S = S_str._obj
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        numba.parfor.init_prange()
        n = len(arr)
        n_total_chars = 0
        str_list = bodo.libs.str_ext.alloc_str_list(n)
        for k in numba.parfor.internal_prange(n):
            # TODO: support NAN
            in_list_str = arr[k]
            out_str = in_list_str[i]
            str_list[k] = out_str
            n_total_chars += get_utf8_size(out_str)
        numba.parfor.init_prange()
        out_arr = pre_alloc_string_array(n, n_total_chars)
        for j in numba.parfor.internal_prange(n):
            _str = str_list[j]
            out_arr[j] = _str
        return bodo.hiframes.api.init_series(out_arr, index, name)

    return _str_get_impl
