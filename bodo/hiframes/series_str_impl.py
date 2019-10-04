"""
Support for Series.str methods
"""
import operator
import numpy as np
import pandas as pd
import re
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
from bodo.utils.typing import (is_list_like_index_type, is_overload_false,
    is_overload_true)
from bodo.libs.str_ext import string_type, str_findall_count
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
        l = len(arr)
        out_arr = bodo.libs.str_ext.alloc_list_list_str(l)
        for i in numba.parfor.internal_prange(l):
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


@overload_method(SeriesStrMethodType, 'replace')
def overload_str_method_replace(pat, repl, n=-1, case=None, flags=0,
                                                                   regex=True):
    # TODO: support other arguments
    # TODO: support dynamic values for regex
    if is_overload_true(regex):
        def _str_replace_regex_impl(S_str, pat, repl, n=-1, case=None, flags=0,
                                                                   regex=True):
            S = S_str._obj
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            numba.parfor.init_prange()
            e = re.compile(pat)
            l = len(arr)
            n_total_chars = 0
            str_list = bodo.libs.str_ext.alloc_str_list(l)
            for i in numba.parfor.internal_prange(l):
                if bodo.hiframes.api.isna(arr, i):
                    continue
                out_str = e.sub(repl, arr[i])
                str_list[i] = out_str
                n_total_chars += get_utf8_size(out_str)
            numba.parfor.init_prange()
            out_arr = pre_alloc_string_array(l, n_total_chars)
            for j in numba.parfor.internal_prange(l):
                if bodo.hiframes.api.isna(arr, j):
                    out_arr[j] = ''
                    bodo.ir.join.setitem_arr_nan(out_arr, j)
                    continue
                _str = str_list[j]
                out_arr[j] = _str
            return bodo.hiframes.api.init_series(out_arr, index, name)
        return _str_replace_regex_impl

    if not is_overload_false(regex):
        raise ValueError(
            "regex argument for Series.str.replace should be constant")

    def _str_replace_noregex_impl(S_str, pat, repl, n=-1, case=None, flags=0,
                                                                   regex=True):
        S = S_str._obj
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        numba.parfor.init_prange()
        l = len(arr)
        n_total_chars = 0
        str_list = bodo.libs.str_ext.alloc_str_list(l)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(arr, i):
                continue
            out_str = arr[i].replace(pat, repl)
            str_list[i] = out_str
            n_total_chars += get_utf8_size(out_str)
        numba.parfor.init_prange()
        out_arr = pre_alloc_string_array(l, n_total_chars)
        for j in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(arr, j):
                out_arr[j] = ''
                bodo.ir.join.setitem_arr_nan(out_arr, j)
                continue
            _str = str_list[j]
            out_arr[j] = _str
        return bodo.hiframes.api.init_series(out_arr, index, name)
    return _str_replace_noregex_impl


@overload_method(SeriesStrMethodType, 'contains')
def overload_str_method_contains(S_str, pat, case=True, flags=0, na=np.nan, regex=True):
    # TODO: support other arguments
    # TODO: support dynamic values for regex
    if is_overload_true(regex):
        def _str_contains_regex_impl(S_str, pat, case=True, flags=0, na=np.nan,
                                               regex=True):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.api.get_series_data(S)
            index = bodo.hiframes.api.get_series_index(S)
            name = bodo.hiframes.api.get_series_name(S)
            numba.parfor.init_prange()
            e = bodo.libs.str_ext.compile_regex(pat)
            l = len(arr)
            out_arr = np.empty(l, dtype=np.bool_)
            nulls = np.empty((l + 7) >> 3, dtype=np.uint8)
            for i in numba.parfor.internal_prange(l):
                if bodo.hiframes.api.isna(arr, i):
                    out_arr[i] = False
                    bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 0)
                else:
                    out_arr[i] = bodo.libs.str_ext.contains_regex(arr[i], e)
                    bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)
            return bodo.hiframes.api.init_series(
                bodo.libs.bool_arr_ext.init_bool_array(out_arr, nulls),
                index, name)
        return _str_contains_regex_impl

    if not is_overload_false(regex):
        raise ValueError(
            "regex argument for Series.str.replace should be constant")

    def _str_contains_noregex_impl(S_str, pat, case=True, flags=0, na=np.nan,
                                            regex=True):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.api.get_series_data(S)
        index = bodo.hiframes.api.get_series_index(S)
        name = bodo.hiframes.api.get_series_name(S)
        numba.parfor.init_prange()
        l = len(arr)
        out_arr = np.empty(l, dtype=np.bool_)
        nulls = np.empty((l + 7) >> 3, dtype=np.uint8)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(arr, i):
                out_arr[i] = False
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 0)
            else:
                out_arr[i] = bodo.libs.str_ext.contains_noregex(arr[i], pat)
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)
        return bodo.hiframes.api.init_series(
            bodo.libs.bool_arr_ext.init_bool_array(out_arr, nulls),
            index, name)
    return _str_contains_noregex_impl


@overload_method(SeriesStrMethodType, 'count')
def overload_str_method_count(S_str, pat, flags=0):
    # python str.count() and pandas str.count() are different
    def impl(S_str, pat, flags=0):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        e = re.compile(pat)
        numba.parfor.init_prange()
        l = len(str_arr)
        out_arr = np.empty(l, dtype=np.int64)
        bitmap = np.empty((l+7)>>3, np.uint8)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                out_arr[i] = 1
                bodo.libs.int_arr_ext.set_bit_to_arr(
                        bitmap, i, 0)
            else:
                out_arr[i] = str_findall_count(e, str_arr[i])
                bodo.libs.int_arr_ext.set_bit_to_arr(
                        bitmap, i, 1)
        return bodo.hiframes.api.init_series(
            bodo.libs.int_arr_ext.init_integer_array(out_arr, bitmap),
            index, name)
    return impl


@overload_method(SeriesStrMethodType, 'find')
def overload_str_method_find(S_str, sub):
    # not supporting start,end as arguments
    def impl(S_str, sub):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        out_arr = np.empty(l, dtype=np.int64)
        bitmap = np.empty((l+7)>>3, np.uint8)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                out_arr[i] = 1
                bodo.libs.int_arr_ext.set_bit_to_arr(
                        bitmap, i, 0)
            else:
                out_arr[i] = str_arr[i].find(sub)
                bodo.libs.int_arr_ext.set_bit_to_arr(
                        bitmap, i, 1)
        return bodo.hiframes.api.init_series(
            bodo.libs.int_arr_ext.init_integer_array(out_arr, bitmap),
            index, name)
    return impl


@overload_method(SeriesStrMethodType, 'center')
def overload_str_method_center(S_str, width, fillchar=' '):
    def impl(S_str, width, fillchar=' '):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(str_arr[i].center(width, fillchar))
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, j):
                out_arr[j] = ''
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].center(width, fillchar)
        return bodo.hiframes.api.init_series(out_arr, index, name)
    return impl


@overload_method(SeriesStrMethodType, 'ljust')
def overload_str_method_ljust(S_str, width, fillchar=' '):
    def impl(S_str, width, fillchar=' '):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(str_arr[i].ljust(width, fillchar))
            num_chars+=s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, j):
                out_arr[j] = ''
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].ljust(width, fillchar)
        return bodo.hiframes.api.init_series(out_arr, index, name)
    return impl


@overload_method(SeriesStrMethodType, 'rjust')
def overload_str_method_rjust(S_str, width, fillchar=' '):
    def impl(S_str, width, fillchar=' '):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(str_arr[i].rjust(width, fillchar))
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, j):
                out_arr[j] = ''
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].rjust(width, fillchar)
        return bodo.hiframes.api.init_series(out_arr, index, name)
    return impl


@overload_method(SeriesStrMethodType, 'zfill')
def overload_str_method_zfill(S_str, width):
    def impl(S_str, width):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(str_arr[i].zfill(width))
            num_chars+=s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, j):
                out_arr[j] = ''
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].zfill(width)
        return bodo.hiframes.api.init_series(out_arr, index, name)
    return impl


@overload_method(SeriesStrMethodType, 'startswith')
def overload_str_method_startswith(S_str, pat, na=np.nan):
    def impl(S_str, pat, na=np.nan):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        nulls = np.empty((l + 7) >> 3, dtype=np.uint8)
        out_arr = np.empty(l, dtype=np.bool_)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                out_arr[i] = False
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 0)
            else:
                out_arr[i] = str_arr[i].startswith(pat)
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)
        return bodo.hiframes.api.init_series(
            bodo.libs.bool_arr_ext.init_bool_array(out_arr, nulls),
            index, name)
    return impl


@overload_method(SeriesStrMethodType, 'endswith')
def overload_str_method_endswith(S_str, pat, na=np.nan):
    def impl(S_str, pat, na=np.nan):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        nulls = np.empty((l + 7) >> 3, dtype=np.uint8)
        out_arr = np.empty(l, dtype=np.bool_)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                out_arr[i] = False
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 0)
            else:
                out_arr[i] = str_arr[i].endswith(pat)
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)
        return bodo.hiframes.api.init_series(
            bodo.libs.bool_arr_ext.init_bool_array(out_arr, nulls),
            index, name)
    return impl


@overload_method(SeriesStrMethodType, 'isupper')
def overload_str_method_isupper(S_str):
    def impl(S_str):
        S = S_str._obj
        str_arr = bodo.hiframes.api.get_series_data(S)
        name = bodo.hiframes.api.get_series_name(S)
        index = bodo.hiframes.api.get_series_index(S)
        numba.parfor.init_prange()
        l = len(str_arr)
        nulls = np.empty((l + 7) >> 3, dtype=np.uint8)
        out_arr = np.empty(l, dtype=np.bool_)
        for i in numba.parfor.internal_prange(l):
            if bodo.hiframes.api.isna(str_arr, i):
                out_arr[i] = False
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 0)
            else:
                out_arr[i] = str_arr[i].isupper()
                bodo.libs.int_arr_ext.set_bit_to_arr(nulls, i, 1)
        return bodo.hiframes.api.init_series(
            bodo.libs.bool_arr_ext.init_bool_array(out_arr, nulls),
            index, name)
    return impl


def create_str2str_methods_overload(func_name):
    def overload_str2str_methods(S_str):
        func_text = 'def f(S_str):\n'
        func_text += '    S = S_str._obj\n'
        func_text += '    str_arr = bodo.hiframes.api.get_series_data(S)\n'
        func_text += '    index = bodo.hiframes.api.get_series_index(S)\n'
        func_text += '    name = bodo.hiframes.api.get_series_name(S)\n'
        func_text += '    numba.parfor.init_prange()\n'
        func_text += '    n = len(str_arr)\n'
        # functions that don't change the number of characters
        if func_name in ('capitalize', 'lower', 'swapcase', 'title', 'upper'):
            func_text += '    num_chars = num_total_chars(str_arr)\n'
        else:
            func_text += '    num_chars = 0\n'
            func_text += '    for i in numba.parfor.internal_prange(n):\n'
            func_text += '        if bodo.hiframes.api.isna(str_arr, i):\n'
            func_text += '            l = 0\n'
            func_text += '        else:\n'
            func_text += '            l = get_utf8_size(str_arr[i].{}())\n'.format(func_name)
            func_text += '        num_chars += l\n'
        func_text += '    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, num_chars)\n'
        func_text += '    for j in numba.parfor.internal_prange(n):\n'
        func_text += '        if bodo.hiframes.api.isna(str_arr, j):\n'
        func_text += '            out_arr[j] = ""\n'
        func_text += '            bodo.ir.join.setitem_arr_nan(out_arr, j)\n'
        func_text += '        else:\n'
        func_text += '            out_arr[j] = str_arr[j].{}()\n'.format(func_name)
        func_text += '    return bodo.hiframes.api.init_series(out_arr, index, name)\n'
        loc_vars = {}
        # print(func_text)
        exec(func_text, {'bodo': bodo, 'numba': numba,
            'num_total_chars': bodo.libs.str_arr_ext.num_total_chars,
            'get_utf8_size': bodo.libs.str_arr_ext.get_utf8_size}, loc_vars)
        f = loc_vars['f']
        return f

    return overload_str2str_methods


def _install_str2str_methods():
    # install methods that just transform the string into another string
    for op in bodo.hiframes.pd_series_ext.str2str_methods:
        overload_impl = create_str2str_methods_overload(op)
        overload_method(SeriesStrMethodType, op)(overload_impl)


_install_str2str_methods()
