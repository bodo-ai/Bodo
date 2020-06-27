# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Support for Series.str methods
"""
import operator
import numpy as np
import re
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
from bodo.hiframes.pd_index_ext import StringIndexType
from bodo.utils.typing import is_overload_false, is_overload_true
from bodo.libs.str_ext import str_findall_count
from bodo.libs.str_arr_ext import (
    string_array_type,
    pre_alloc_string_array,
    get_utf8_size,
)
from bodo.hiframes.split_impl import (
    string_array_split_view_type,
    getitem_c_arr,
    get_array_ctypes_ptr,
    get_split_view_index,
    get_split_view_data_ptr,
)
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.utils.typing import (
    BodoError,
    is_overload_none,
    get_overload_const_list,
    is_overload_true,
    is_overload_false,
    is_overload_zero,
    is_overload_constant_bool,
    is_overload_constant_str,
    get_overload_const_str,
    get_overload_const_str_len,
    is_overload_constant_int,
    get_overload_const_int,
)


class SeriesStrMethodType(types.Type):
    def __init__(self, stype):
        # keeping Series type since string data representation can be varied
        self.stype = stype
        name = "SeriesStrMethodType({})".format(stype)
        super(SeriesStrMethodType, self).__init__(name)


@register_model(SeriesStrMethodType)
class SeriesStrModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super(SeriesStrModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesStrMethodType, "obj", "_obj")


@intrinsic
def init_series_str_method(typingctx, obj=None):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        str_method_type = signature.return_type

        str_method_val = cgutils.create_struct_proxy(str_method_type)(context, builder)
        str_method_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return str_method_val._getvalue()

    return SeriesStrMethodType(obj)(obj), codegen


def str_arg_check(func_name, arg_name, arg):
    """
    Helper function to raise BodoError 
    when the argument is NOT a string(UnicodeType) or const string
    """
    if not isinstance(arg, types.UnicodeType) and not is_overload_constant_str(arg):
        raise BodoError(
            "Series.str.{}(): parameter '{}' expected a string object, not {}".format(
                func_name, arg_name, arg
            )
        )


def int_arg_check(func_name, arg_name, arg):
    """
    Helper function to raise BodoError 
    when the argument is NOT an Integer type
    """
    if not isinstance(arg, types.Integer) and not is_overload_constant_int(arg):
        raise BodoError(
            "Series.str.{}(): parameter '{}' expected an int object, not {}".format(
                func_name, arg_name, arg
            )
        )


def not_supported_arg_check(func_name, arg_name, arg, defval):
    """
    Helper function to raise BodoError 
    when not supported argument is provided by users
    """
    if arg_name == "na":
        if not isinstance(arg, types.Omitted) and (
            not isinstance(arg, float) or not np.isnan(arg)
        ):
            raise BodoError(
                "Series.str.{}(): parameter '{}' is not supported, default: np.nan".format(
                    func_name, arg_name
                )
            )
    else:
        if not isinstance(arg, types.Omitted) and arg != defval:
            raise BodoError(
                "Series.str.{}(): parameter '{}' is not supported, default: {}".format(
                    func_name, arg_name, defval
                )
            )


def common_validate_padding(func_name, width, fillchar):
    """
    Helper function to raise BodoError 
    for checking arguments' types of ljust,rjust,center,padding
    """
    if is_overload_constant_str(fillchar):
        if get_overload_const_str_len(fillchar) != 1:
            raise BodoError(
                "Series.str.{}(): fillchar must be a character, not str".format(
                    func_name
                )
            )
    elif not isinstance(fillchar, types.UnicodeType):
        raise BodoError(
            "Series.str.{}(): fillchar must be a character, not {}".format(
                func_name, fillchar
            )
        )

    int_arg_check(func_name, "width", width)


@overload_attribute(SeriesType, "str")
def overload_series_str(S):
    if not isinstance(S, SeriesType) or not S.data in (
        string_array_type,
        string_array_split_view_type,
        ArrayItemArrayType(string_array_type),
    ):
        raise BodoError(
            "Series.str(): input should be a series of string or list string or string view"
        )
    return lambda S: bodo.hiframes.series_str_impl.init_series_str_method(S)


@overload_method(SeriesStrMethodType, "len", inline="always", no_unliteral=True)
def overload_str_method_len(S_str):
    def impl(S_str):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(n, np.int64)
        for i in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                # TODO: optimize str len on string array (count unicode chars inplace)
                out_arr[i] = len(arr[i])

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "split", inline="always", no_unliteral=True)
def overload_str_method_split(S_str, pat=None, n=-1, expand=False):
    # TODO: support or just check n and expand arguments
    if not is_overload_none(pat):
        str_arg_check("split", "pat", pat)
    not_supported_arg_check("split", "n", n, -1)
    not_supported_arg_check("split", "expand", expand, False)

    # TODO: support distributed
    # TODO: support regex

    # use split view if sep is a string of length 1
    if isinstance(pat, types.StringLiteral) and len(pat.literal_value) == 1:

        def _str_split_view_impl(
            S_str, pat=None, n=-1, expand=False
        ):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.hiframes.split_impl.compute_split_view(arr, pat)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_split_view_impl

    # TODO: optimize!
    def _str_split_impl(S_str, pat=None, n=-1, expand=False):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        # not inlining loops since fusion optimization doesn't seem likely
        out_arr = bodo.libs.str_ext.str_split(arr, pat, n)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_split_impl


@overload_method(SeriesStrMethodType, "get", no_unliteral=True)
def overload_str_method_get(S_str, i):
    arr_typ = S_str.stype.data
    if (
        arr_typ != string_array_split_view_type
        and arr_typ != ArrayItemArrayType(string_array_type)
        and arr_typ != string_array_type
    ):
        raise BodoError(
            "Series.str.get(): only supports input type of Series(list(str)) "
            "and Series(str)"
        )
    int_arg_check("get", "i", i)
    # TODO: support and test NA
    # TODO: support distributed

    if arr_typ == string_array_split_view_type:
        # TODO: refactor and enable distributed
        def _str_get_split_impl(S_str, i):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            numba.parfors.parfor.init_prange()
            n = len(arr)
            n_total_chars = 0
            for k in numba.parfors.parfor.internal_prange(n):
                _, _, length = get_split_view_index(arr, k, i)
                n_total_chars += length
            numba.parfors.parfor.init_prange()
            out_arr = pre_alloc_string_array(n, n_total_chars)
            for j in numba.parfors.parfor.internal_prange(n):
                status, data_start, length = get_split_view_index(arr, j, i)
                if status == 0:
                    bodo.ir.join.setitem_arr_nan(out_arr, j)
                    ptr = get_split_view_data_ptr(arr, 0)
                else:
                    ptr = get_split_view_data_ptr(arr, data_start)
                bodo.libs.str_arr_ext.setitem_str_arr_ptr(out_arr, j, ptr, length)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_get_split_impl

    def _str_get_impl(S_str, i):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        n_total_chars = 0
        str_list = bodo.libs.str_ext.alloc_random_access_string_array(n)
        na_map = np.empty(n, np.bool_)
        for k in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(arr, k):
                na_map[k] = True
                str_list[k] = ""
                continue
            in_list_str = arr[k]
            if not (len(in_list_str) > i >= -len(in_list_str)):
                na_map[k] = True
                str_list[k] = ""
                continue
            out_str = in_list_str[i]
            str_list[k] = out_str
            na_map[k] = False
            n_total_chars += get_utf8_size(out_str)
        numba.parfors.parfor.init_prange()
        out_arr = pre_alloc_string_array(n, n_total_chars)
        for j in numba.parfors.parfor.internal_prange(n):
            if na_map[j]:
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                _str = str_list[j]
                out_arr[j] = _str
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_get_impl


@overload_method(SeriesStrMethodType, "join", inline="always", no_unliteral=True)
def overload_str_method_join(S_str, sep):
    arr_typ = S_str.stype.data
    if (
        arr_typ != string_array_split_view_type
        and arr_typ != ArrayItemArrayType(string_array_type)
        and arr_typ != string_array_type
    ):
        raise BodoError(
            "Series.str.join(): only supports input type of Series(list(str)) "
            "and Series(str)"
        )
    str_arg_check("join", "sep", sep)

    def impl(S_str, sep):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        n = len(str_arr)
        n_total_chars = 0
        for k in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(str_arr, k):
                s = 0
            else:
                in_list_str = str_arr[k]
                s = bodo.libs.str_arr_ext.get_utf8_size(sep.join(in_list_str))
            n_total_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, n_total_chars)
        for j in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                in_list_str = str_arr[j]
                out_arr[j] = sep.join(in_list_str)

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "replace", inline="always", no_unliteral=True)
def overload_str_method_replace(S_str, pat, repl, n=-1, case=None, flags=0, regex=True):
    not_supported_arg_check("replace", "n", n, -1)
    not_supported_arg_check("replace", "case", case, None)
    str_arg_check("replace", "pat", pat)
    str_arg_check("replace", "repl", repl)
    int_arg_check("replace", "flags", flags)
    # TODO: support other arguments
    # TODO: support dynamic values for regex
    if is_overload_true(regex):

        def _str_replace_regex_impl(
            S_str, pat, repl, n=-1, case=None, flags=0, regex=True
        ):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            numba.parfors.parfor.init_prange()
            e = re.compile(pat, flags)
            l = len(arr)
            n_total_chars = 0
            str_list = bodo.libs.str_ext.alloc_random_access_string_array(l)
            for i in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(arr, i):
                    continue
                out_str = e.sub(repl, arr[i])
                str_list[i] = out_str
                n_total_chars += get_utf8_size(out_str)
            numba.parfors.parfor.init_prange()
            out_arr = pre_alloc_string_array(l, n_total_chars)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(arr, j):
                    out_arr[j] = ""
                    bodo.ir.join.setitem_arr_nan(out_arr, j)
                    continue
                _str = str_list[j]
                out_arr[j] = _str
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_replace_regex_impl

    if not is_overload_false(regex):
        raise BodoError("Series.str.replace(): regex argument should be bool")

    def _str_replace_noregex_impl(
        S_str, pat, repl, n=-1, case=None, flags=0, regex=True
    ):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        l = len(arr)
        n_total_chars = 0
        str_list = bodo.libs.str_ext.alloc_random_access_string_array(l)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(arr, i):
                continue
            out_str = arr[i].replace(pat, repl)
            str_list[i] = out_str
            n_total_chars += get_utf8_size(out_str)
        numba.parfors.parfor.init_prange()
        out_arr = pre_alloc_string_array(l, n_total_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
                continue
            _str = str_list[j]
            out_arr[j] = _str
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_replace_noregex_impl


@overload_method(SeriesStrMethodType, "contains", no_unliteral=True)
def overload_str_method_contains(S_str, pat, case=True, flags=0, na=np.nan, regex=True):
    not_supported_arg_check("contains", "case", case, True)
    not_supported_arg_check("contains", "na", na, np.nan)
    str_arg_check("contains", "pat", pat)
    int_arg_check("contians", "flags", flags)
    # TODO: support other arguments
    # TODO: support dynamic values for regex
    if is_overload_true(regex):

        def _str_contains_regex_impl(
            S_str, pat, case=True, flags=0, na=np.nan, regex=True
        ):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            numba.parfors.parfor.init_prange()
            e = re.compile(pat, flags)
            l = len(arr)
            out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)
            for i in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(arr, i):
                    bodo.ir.join.setitem_arr_nan(out_arr, i)
                else:
                    out_arr[i] = bodo.libs.str_ext.contains_regex(e, arr[i])
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_contains_regex_impl

    if not is_overload_false(regex):
        raise BodoError("Series.str.contains(): regex argument should be bool")

    def _str_contains_noregex_impl(
        S_str, pat, case=True, flags=0, na=np.nan, regex=True
    ):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        l = len(arr)
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                out_arr[i] = pat in arr[i]
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_contains_noregex_impl


@overload_method(SeriesStrMethodType, "count", inline="always", no_unliteral=True)
def overload_str_method_count(S_str, pat, flags=0):
    # python str.count() and pandas str.count() are different
    str_arg_check("count", "pat", pat)
    int_arg_check("count", "flags", flags)

    def impl(S_str, pat, flags=0):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        e = re.compile(pat, flags)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(l, np.int64)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                out_arr[i] = str_findall_count(e, str_arr[i])
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "find", inline="always", no_unliteral=True)
def overload_str_method_find(S_str, sub):
    # not supporting start,end as arguments
    str_arg_check("find", "sub", sub)

    def impl(S_str, sub):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(l, np.int64)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                out_arr[i] = str_arr[i].find(sub)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "rfind", inline="always", no_unliteral=True)
def overload_str_method_rfind(S_str, sub, start=0, end=None):
    str_arg_check("rfind", "sub", sub)
    if start != 0:
        int_arg_check("rfind", "start", start)
    if not is_overload_none(end):
        int_arg_check("rfind", "end", end)

    def impl(S_str, sub, start=0, end=None):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(l, np.int64)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                out_arr[i] = str_arr[i].rfind(sub, start, end)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "center", inline="always", no_unliteral=True)
def overload_str_method_center(S_str, width, fillchar=" "):
    common_validate_padding("center", width, fillchar)

    def impl(S_str, width, fillchar=" "):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(
                    str_arr[i].center(width, fillchar)
                )
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].center(width, fillchar)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "ljust", inline="always", no_unliteral=True)
def overload_str_method_ljust(S_str, width, fillchar=" "):
    common_validate_padding("ljust", width, fillchar)

    def impl(S_str, width, fillchar=" "):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(
                    str_arr[i].ljust(width, fillchar)
                )
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].ljust(width, fillchar)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "rjust", inline="always", no_unliteral=True)
def overload_str_method_rjust(S_str, width, fillchar=" "):
    common_validate_padding("rjust", width, fillchar)

    def impl(S_str, width, fillchar=" "):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(
                    str_arr[i].rjust(width, fillchar)
                )
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].rjust(width, fillchar)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "pad", no_unliteral=True)
def overload_str_method_pad(S_str, width, side="left", fillchar=" "):
    common_validate_padding("pad", width, fillchar)
    if is_overload_constant_str(side):
        if get_overload_const_str(side) not in [
            "left",
            "right",
            "both",
        ]:  # numba does not catch this case. Causes SegFault
            raise BodoError("Series.str.pad(): Invalid Side")
    else:
        raise BodoError("Series.str.pad(): Invalid Side")

    def impl(S_str, width, side="left", fillchar=" "):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                s = 0
            else:
                if side == "left":
                    s = bodo.libs.str_arr_ext.get_utf8_size(
                        str_arr[i].rjust(width, fillchar)
                    )
                elif side == "right":
                    s = bodo.libs.str_arr_ext.get_utf8_size(
                        str_arr[i].ljust(width, fillchar)
                    )
                elif side == "both":
                    s = bodo.libs.str_arr_ext.get_utf8_size(
                        str_arr[i].center(width, fillchar)
                    )
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                if side == "left":
                    out_arr[j] = str_arr[j].rjust(width, fillchar)
                elif side == "right":
                    out_arr[j] = str_arr[j].ljust(width, fillchar)
                elif side == "both":
                    out_arr[j] = str_arr[j].center(width, fillchar)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "zfill", inline="always", no_unliteral=True)
def overload_str_method_zfill(S_str, width):
    int_arg_check("zfill", "width", width)

    def impl(S_str, width):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(str_arr[i].zfill(width))
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j].zfill(width)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "slice", no_unliteral=True)
def overload_str_method_slice(S_str, start=None, stop=None, step=None):
    if not is_overload_none(start):
        int_arg_check("slice", "start", start)
    if not is_overload_none(stop):
        int_arg_check("slice", "stop", stop)
    if not is_overload_none(step):
        int_arg_check("slice", "step", step)

    def impl(S_str, start=None, stop=None, step=None):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        num_chars = 0
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                s = 0
            else:
                s = bodo.libs.str_arr_ext.get_utf8_size(str_arr[i][start:stop:step])
            num_chars += s
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, num_chars)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.ir.join.setitem_arr_nan(out_arr, j)
            else:
                out_arr[j] = str_arr[j][start:stop:step]
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "startswith", inline="always", no_unliteral=True)
def overload_str_method_startswith(S_str, pat, na=np.nan):
    not_supported_arg_check("startswith", "na", na, np.nan)
    str_arg_check("startswith", "pat", pat)

    def impl(S_str, pat, na=np.nan):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                out_arr[i] = str_arr[i].startswith(pat)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "endswith", inline="always", no_unliteral=True)
def overload_str_method_endswith(S_str, pat, na=np.nan):
    not_supported_arg_check("endswith", "na", na, np.nan)
    str_arg_check("endswith", "pat", pat)

    def impl(S_str, pat, na=np.nan):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.ir.join.setitem_arr_nan(out_arr, i)
            else:
                out_arr[i] = str_arr[i].endswith(pat)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload(operator.getitem, no_unliteral=True)
def overload_str_method_getitem(S_str, ind):
    if not isinstance(S_str, SeriesStrMethodType):
        return

    if not isinstance(types.unliteral(ind), (types.SliceType, types.Integer)):
        raise BodoError("index input to Series.str[] should be a slice or an integer")

    if isinstance(ind, types.SliceType):
        return lambda S_str, ind: S_str.slice(ind.start, ind.stop, ind.step)

    if isinstance(types.unliteral(ind), types.Integer):
        return lambda S_str, ind: S_str.get(ind)


@overload_method(SeriesStrMethodType, "extract", inline="always", no_unliteral=True)
def overload_str_method_extract(S_str, pat, flags=0, expand=True):

    if not is_overload_constant_bool(expand):
        raise BodoError(
            "Series.str.extract(): 'expand' argument should be a constant bool"
        )

    columns, regex = _get_column_names_from_regex(pat, flags, "extract")
    n_cols = len(columns)

    # generate one loop for finding character count and another for computation
    # TODO: avoid multiple loops if possible, or even avoid inlined loops if needed
    func_text = "def impl(S_str, pat, flags=0, expand=True):\n"
    func_text += "  regex = re.compile(pat, flags=flags)\n"
    func_text += "  S = S_str._obj\n"
    func_text += "  str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "  index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    func_text += "  numba.parfors.parfor.init_prange()\n"
    func_text += "  n = len(str_arr)\n"
    for i in range(n_cols):
        func_text += "  num_chars_{} = 0\n".format(i)
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "      if bodo.libs.array_kernels.isna(str_arr, i):\n"
    for i in range(n_cols):
        func_text += "          l_{} = 0\n".format(i)
    func_text += "      else:\n"
    func_text += "          m = regex.search(str_arr[i])\n"
    func_text += "          if not m:\n"
    for i in range(n_cols):
        func_text += "            l_{} = 0\n".format(i)
    func_text += "          else:\n"
    func_text += "            g = m.groups()\n"
    for i in range(n_cols):
        func_text += "            l_{0} = get_utf8_size(g[{0}])\n".format(i)
    for i in range(n_cols):
        func_text += "      num_chars_{0} += l_{0}\n".format(i)
    for i in range(n_cols):
        func_text += "  out_arr_{0} = bodo.libs.str_arr_ext.pre_alloc_string_array(n, num_chars_{0})\n".format(
            i
        )
    func_text += "  for j in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "      if bodo.libs.array_kernels.isna(str_arr, j):\n"
    for i in range(n_cols):
        func_text += "          out_arr_{}[j] = ''\n".format(i)
        func_text += "          bodo.ir.join.setitem_arr_nan(out_arr_{}, j)\n".format(i)
    func_text += "      else:\n"
    func_text += "          m = regex.search(str_arr[j])\n"
    func_text += "          if m:\n"
    func_text += "            g = m.groups()\n"
    for i in range(n_cols):
        func_text += "            out_arr_{0}[j] = g[{0}]\n".format(i)
    func_text += "          else:\n"
    for i in range(n_cols):
        func_text += "            out_arr_{}[j] = ''\n".format(i)
        func_text += "            bodo.ir.join.setitem_arr_nan(out_arr_{}, j)\n".format(
            i
        )

    # no expand case
    if is_overload_false(expand) and regex.groups == 1:
        name = (
            "'{}'".format(list(regex.groupindex.keys()).pop())
            if len(regex.groupindex.keys()) > 0
            else "name"
        )
        func_text += "  return bodo.hiframes.pd_series_ext.init_series(out_arr_0, index, {})\n".format(
            name
        )
        loc_vars = {}
        exec(
            func_text,
            {"re": re, "bodo": bodo, "numba": numba, "get_utf8_size": get_utf8_size},
            loc_vars,
        )
        impl = loc_vars["impl"]
        return impl

    data_args = ", ".join("out_arr_{}".format(i) for i in range(n_cols))
    impl = bodo.hiframes.dataframe_impl._gen_init_df(
        func_text,
        columns,
        data_args,
        "index",
        extra_globals={"get_utf8_size": get_utf8_size, "re": re},
    )
    return impl


@overload_method(SeriesStrMethodType, "extractall", inline="always", no_unliteral=True)
def overload_str_method_extractall(S_str, pat, flags=0, expand=True):

    columns, _ = _get_column_names_from_regex(pat, flags, "extractall")
    n_cols = len(columns)
    is_index_string = isinstance(S_str.stype.index, StringIndexType)

    # generate one loop for finding character count and another for computation
    # TODO: avoid multiple loops if possible, or even avoid inlined loops if needed
    func_text = "def impl(S_str, pat, flags=0, expand=True):\n"
    func_text += "  regex = re.compile(pat, flags=flags)\n"
    func_text += "  S = S_str._obj\n"
    func_text += "  str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "  index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    # TODO: support MultiIndex in input Series
    func_text += "  index_arr = bodo.utils.conversion.index_to_array(index)\n"
    func_text += "  index_name = bodo.hiframes.pd_index_ext.get_index_name(index)\n"
    # TODO: string index char count
    func_text += "  numba.parfors.parfor.init_prange()\n"
    func_text += "  n = len(str_arr)\n"
    # using a list wrapper for integer to avoid reduction machinery (we need local size)
    func_text += "  out_n_l = [0]\n"
    for i in range(n_cols):
        func_text += "  num_chars_{} = 0\n".format(i)
    if is_index_string:
        func_text += "  index_num_chars = 0\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    if is_index_string:
        func_text += "      index_num_chars += get_utf8_size(index_arr[i])\n"
    func_text += "      if bodo.libs.array_kernels.isna(str_arr, i):\n"
    func_text += "          continue\n"  # extractall just skips NAs
    func_text += "      m = regex.findall(str_arr[i])\n"
    func_text += "      out_n_l[0] += len(m)\n"
    for i in range(n_cols):
        func_text += "      l_{} = 0\n".format(i)
    func_text += "      for s in m:\n"
    for i in range(n_cols):
        func_text += "        l_{} += get_utf8_size(s{})\n".format(
            i, "[{}]".format(i) if n_cols > 1 else ""
        )
    for i in range(n_cols):
        func_text += "      num_chars_{0} += l_{0}\n".format(i)
    # using a sentinel function to specify that the arrays are local and no need for
    # distributed transformation
    func_text += (
        "  out_n = bodo.libs.distributed_api.local_alloc_size(out_n_l[0], str_arr)\n"
    )
    for i in range(n_cols):
        func_text += "  out_arr_{0} = bodo.libs.str_arr_ext.pre_alloc_string_array(out_n, num_chars_{0})\n".format(
            i
        )
    if is_index_string:
        func_text += "  out_ind_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(out_n, index_num_chars)\n"
    else:
        func_text += "  out_ind_arr = np.empty(out_n, index_arr.dtype)\n"
    func_text += "  out_match_arr = np.empty(out_n, np.int64)\n"
    func_text += "  out_ind = 0\n"
    func_text += "  for j in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "      if bodo.libs.array_kernels.isna(str_arr, j):\n"
    func_text += "          continue\n"  # extractall just skips NAs
    func_text += "      m = regex.findall(str_arr[j])\n"
    func_text += "      for k, s in enumerate(m):\n"
    for i in range(n_cols):
        # using set_arr_local() to avoid distributed transformation of setitem
        func_text += "        bodo.libs.distributed_api.set_arr_local(out_arr_{}, out_ind, s{})\n".format(
            i, "[{}]".format(i) if n_cols > 1 else ""
        )
    func_text += "        bodo.libs.distributed_api.set_arr_local(out_ind_arr, out_ind, index_arr[j])\n"
    func_text += (
        "        bodo.libs.distributed_api.set_arr_local(out_match_arr, out_ind, k)\n"
    )
    func_text += "        out_ind += 1\n"
    func_text += "  out_index = bodo.hiframes.pd_multi_index_ext.init_multi_index(\n"
    func_text += "    (out_ind_arr, out_match_arr), (index_name, 'match'))\n"

    # TODO: support dead code elimination with local distribution sentinels
    data_args = ", ".join("out_arr_{}".format(i) for i in range(n_cols))
    impl = bodo.hiframes.dataframe_impl._gen_init_df(
        func_text,
        columns,
        data_args,
        "out_index",
        extra_globals={"get_utf8_size": get_utf8_size, "re": re},
    )
    return impl


def _get_column_names_from_regex(pat, flags, func_name):
    """get output dataframe's column names from constant regular expression in
    extract/extractall calls
    """
    # error checking
    # regex arguments have to be constant for "extract", since evaluation of regex in
    # compilation time is required for determining output type.
    if not is_overload_constant_str(pat):
        raise BodoError(
            "Series.str.{}(): 'pat' argument should be a constant string".format(
                func_name
            )
        )

    if not is_overload_constant_int(flags):
        raise BodoError(
            "Series.str.{}(): 'flags' argument should be a constant int".format(
                func_name
            )
        )

    # get column names similar to pd.core.strings._str_extract_frame()
    pat = get_overload_const_str(pat)
    flags = get_overload_const_int(flags)
    regex = re.compile(pat, flags=flags)
    if regex.groups == 0:
        raise BodoError(
            "Series.str.{}(): pattern {} contains no capture groups".format(
                func_name, pat
            )
        )
    names = dict(zip(regex.groupindex.values(), regex.groupindex.keys()))
    columns = [names.get(1 + i, i) for i in range(regex.groups)]
    return columns, regex


def create_str2str_methods_overload(func_name):
    def overload_str2str_methods(S_str):
        func_text = "def f(S_str):\n"
        func_text += "    S = S_str._obj\n"
        func_text += "    str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    n = len(str_arr)\n"
        # functions that don't change the number of characters
        if func_name in ("capitalize", "lower", "swapcase", "title", "upper"):
            func_text += "    num_chars = num_total_chars(str_arr)\n"
        else:
            func_text += "    num_chars = 0\n"
            func_text += "    for i in numba.parfors.parfor.internal_prange(n):\n"
            func_text += "        if bodo.libs.array_kernels.isna(str_arr, i):\n"
            func_text += "            l = 0\n"
            func_text += "        else:\n"
            func_text += "            l = get_utf8_size(str_arr[i].{}())\n".format(
                func_name
            )
            func_text += "        num_chars += l\n"
        func_text += (
            "    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, num_chars)\n"
        )
        func_text += "    for j in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "        if bodo.libs.array_kernels.isna(str_arr, j):\n"
        func_text += '            out_arr[j] = ""\n'
        func_text += "            bodo.ir.join.setitem_arr_nan(out_arr, j)\n"
        func_text += "        else:\n"
        func_text += "            out_arr[j] = str_arr[j].{}()\n".format(func_name)
        func_text += (
            "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
        )
        loc_vars = {}
        # print(func_text)
        exec(
            func_text,
            {
                "bodo": bodo,
                "numba": numba,
                "num_total_chars": bodo.libs.str_arr_ext.num_total_chars,
                "get_utf8_size": bodo.libs.str_arr_ext.get_utf8_size,
            },
            loc_vars,
        )
        f = loc_vars["f"]
        return f

    return overload_str2str_methods


def create_str2bool_methods_overload(func_name):
    def overload_str2bool_methods(S_str):
        func_text = "def f(S_str):\n"
        func_text += "    S = S_str._obj\n"
        func_text += "    str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "    numba.parfors.parfor.init_prange()\n"
        func_text += "    l = len(str_arr)\n"
        func_text += "    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)\n"
        func_text += "    for i in numba.parfors.parfor.internal_prange(l):\n"
        func_text += "        if bodo.libs.array_kernels.isna(str_arr, i):\n"
        func_text += "            bodo.ir.join.setitem_arr_nan(out_arr, i)\n"
        func_text += "        else:\n"
        func_text += "            out_arr[i] = str_arr[i].{}()\n".format(func_name)
        func_text += "    return bodo.hiframes.pd_series_ext.init_series(\n"
        func_text += "      out_arr,index, name)\n"
        loc_vars = {}
        # print(func_text)
        exec(
            func_text, {"bodo": bodo, "numba": numba, "np": np,}, loc_vars,
        )
        f = loc_vars["f"]
        return f

    return overload_str2bool_methods


def _install_str2str_methods():
    # install methods that just transform the string into another string
    for op in bodo.hiframes.pd_series_ext.str2str_methods:
        overload_impl = create_str2str_methods_overload(op)
        overload_method(SeriesStrMethodType, op, inline="always", no_unliteral=True)(
            overload_impl
        )


def _install_str2bool_methods():
    # install methods that just transform the string into another boolean
    for op in bodo.hiframes.pd_series_ext.str2bool_methods:
        overload_impl = create_str2bool_methods_overload(op)
        overload_method(SeriesStrMethodType, op, inline="always", no_unliteral=True)(
            overload_impl
        )


_install_str2str_methods()
_install_str2bool_methods()


@overload_attribute(SeriesType, "cat")
def overload_series_cat(s):
    if not isinstance(s.dtype, bodo.hiframes.pd_categorical_ext.PDCategoricalDtype):
        raise BodoError("Can only use .cat accessor with categorical values.")
    return lambda s: bodo.hiframes.series_str_impl.init_series_cat_method(s)


class SeriesCatMethodType(types.Type):
    def __init__(self, stype):
        self.stype = stype
        name = "SeriesCatMethodType({})".format(stype)
        super(SeriesCatMethodType, self).__init__(name)


@register_model(SeriesCatMethodType)
class SeriesCatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super(SeriesCatModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesCatMethodType, "obj", "_obj")


@intrinsic
def init_series_cat_method(typingctx, obj=None):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        cat_method_type = signature.return_type

        cat_method_val = cgutils.create_struct_proxy(cat_method_type)(context, builder)
        cat_method_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return cat_method_val._getvalue()

    return SeriesCatMethodType(obj)(obj), codegen


@overload_attribute(SeriesCatMethodType, "codes")
def series_cat_codes_overload(S_dt):
    def impl(S_dt):  # pragma: no cover
        S = S_dt._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        # Pandas ignores Series name for some reason currently
        # name = bodo.hiframes.pd_series_ext.get_series_name(S)
        name = None
        return bodo.hiframes.pd_series_ext.init_series(arr.codes, index, name)

    return impl
