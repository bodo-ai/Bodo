# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Support for Series.str methods
"""
import operator
import re

import numba
import numpy as np
import pandas as pd
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
from bodo.hiframes.generic_pandas_coverage import (
    generate_series_to_df_impl,
    generate_simple_series_impl,
)
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.hiframes.pd_index_ext import StringIndexType
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.split_impl import (
    get_split_view_data_ptr,
    get_split_view_index,
    string_array_split_view_type,
)
from bodo.libs.array import get_search_regex
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.str_arr_ext import (
    get_utf8_size,
    pre_alloc_string_array,
    string_array_type,
)
from bodo.libs.str_ext import str_findall_count
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_str_len,
    is_bin_arr_type,
    is_list_like_index_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    is_str_arr_type,
    raise_bodo_error,
)
from bodo.utils.utils import synchronize_error_njit


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
def init_series_str_method(typingctx, obj):
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
        raise_bodo_error(
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
    if not (
        is_str_arr_type(S.data)
        or S.data == string_array_split_view_type
        or isinstance(S.data, ArrayItemArrayType)
        or is_bin_arr_type(S.data)
    ):
        raise_bodo_error(
            "Series.str: input should be a series of string/binary or arrays"
        )
    return lambda S: bodo.hiframes.series_str_impl.init_series_str_method(S)


@overload_method(SeriesStrMethodType, "len", inline="always", no_unliteral=True)
def overload_str_method_len(S_str):
    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_len_dict_impl(S_str):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_len(arr)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_len_dict_impl

    def impl(S_str):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        out_arr = bodo.libs.array_kernels.get_arr_lens(arr, False)

        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "split", inline="always", no_unliteral=True)
def overload_str_method_split(S_str, pat=None, n=-1, expand=False):
    # TODO: support or just check n and expand arguments
    if not is_overload_none(pat):
        str_arg_check("split", "pat", pat)
    int_arg_check("split", "n", n)
    not_supported_arg_check("split", "expand", expand, False)

    # TODO: support distributed
    # TODO: support regex

    # use split view if sep is a string of length 1 and n == -1
    if (
        is_overload_constant_str(pat)
        and len(get_overload_const_str(pat)) == 1
        # _str_split assumes an ascii character
        and get_overload_const_str(pat).isascii()
        and is_overload_constant_int(n)
        and get_overload_const_int(n) == -1
        # only works on regular string arrays
        and S_str.stype.data == string_array_type
    ):

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

    use_default_pat = is_overload_none(pat) and not (
        is_overload_constant_int(n) and get_overload_const_int(n) < 1
    )

    def _str_split_impl(S_str, pat=None, n=-1, expand=False):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        # Not inlining loops since fusion optimization doesn't seem likely
        if use_default_pat and n >= 1:
            # Avoiding passing in None since the implementation of split does not
            # do the correct pandas behavior when pat=None and n>=1, but does when
            # passed in the corresponding regex pattern.
            out_arr = bodo.libs.str_ext.str_split_empty_n(arr, n)
        else:
            out_arr = bodo.libs.str_ext.str_split(arr, pat, n)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_split_impl


@overload_method(SeriesStrMethodType, "get", no_unliteral=True)
def overload_str_method_get(S_str, i):
    arr_typ = S_str.stype.data
    if (
        arr_typ != string_array_split_view_type and not is_str_arr_type(arr_typ)
    ) and not isinstance(arr_typ, ArrayItemArrayType):
        raise_bodo_error(
            "Series.str.get(): only supports input type of Series(array(item)) "
            "and Series(str)"
        )
    int_arg_check("get", "i", i)
    # TODO: support and test NA
    # TODO: support distributed

    if isinstance(arr_typ, ArrayItemArrayType):

        def _str_get_array_impl(S_str, i):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.array_kernels.get(arr, i)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_get_array_impl

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
                    bodo.libs.array_kernels.setna(out_arr, j)
                    ptr = get_split_view_data_ptr(arr, 0)
                else:
                    bodo.libs.str_arr_ext.str_arr_set_not_na(out_arr, j)
                    ptr = get_split_view_data_ptr(arr, data_start)
                bodo.libs.str_arr_ext.setitem_str_arr_ptr(out_arr, j, ptr, length)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_get_split_impl

    # optimized version for dictionary encode arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_get_dict_impl(S_str, i):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_get(arr, i)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_get_dict_impl

    def _str_get_impl(S_str, i):  # pragma: no cover
        S = S_str._obj
        arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        numba.parfors.parfor.init_prange()
        n = len(arr)
        numba.parfors.parfor.init_prange()
        out_arr = pre_alloc_string_array(n, -1)
        for j in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(arr, j) or not (
                len(arr[j]) > i >= -len(arr[j])
            ):
                out_arr[j] = ""
                bodo.libs.array_kernels.setna(out_arr, j)
            else:
                out_arr[j] = arr[j][i]
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_get_impl


@overload_method(SeriesStrMethodType, "join", inline="always", no_unliteral=True)
def overload_str_method_join(S_str, sep):
    arr_typ = S_str.stype.data
    if (
        arr_typ != string_array_split_view_type
        and arr_typ != ArrayItemArrayType(string_array_type)
        and not is_str_arr_type(arr_typ)
    ):
        raise_bodo_error(
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
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)
        for j in numba.parfors.parfor.internal_prange(n):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.libs.array_kernels.setna(out_arr, j)
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

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_replace_dict_impl(
            S_str, pat, repl, n=-1, case=None, flags=0, regex=True
        ):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_replace(arr, pat, repl, flags, regex)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_replace_dict_impl

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
            out_arr = pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(arr, j):
                    out_arr[j] = ""
                    bodo.libs.array_kernels.setna(out_arr, j)
                    continue
                out_arr[j] = e.sub(repl, arr[j])
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
        numba.parfors.parfor.init_prange()
        out_arr = pre_alloc_string_array(l, -1)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(arr, j):
                out_arr[j] = ""
                bodo.libs.array_kernels.setna(out_arr, j)
                continue
            out_arr[j] = arr[j].replace(pat, repl)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return _str_replace_noregex_impl


@overload_method(
    SeriesStrMethodType, "removeprefix", inline="always", no_unliteral=True
)
def overload_str_method_removeprefix(S, prefix):
    str_arg_check("removeprefix", "prefix", prefix)
    scalar_text = "if bodo.libs.array_kernels.isna(data, i):\n"
    scalar_text += "  bodo.libs.array_kernels.setna(result, i)\n"
    scalar_text += "else:\n"
    scalar_text += " data_str = data[i]\n"
    scalar_text += " if data_str.startswith(prefix):\n"
    scalar_text += "   result[i] = data_str[len(prefix):]\n"
    scalar_text += " else:\n"
    scalar_text += "   result[i] = data_str\n"
    return generate_simple_series_impl(
        ("S", "prefix"), (S, prefix), S.stype, scalar_text
    )


@overload_method(SeriesStrMethodType, "casefold", inline="always")
def overload_str_method_casefold(S):
    scalar_text = "if bodo.libs.array_kernels.isna(data, i):\n"
    scalar_text += " bodo.libs.array_kernels.setna(result, i)\n"
    scalar_text += " continue\n"
    scalar_text += "result[i] = data[i].casefold()\n"
    return generate_simple_series_impl(("S",), (S,), S.stype, scalar_text)


@overload_method(
    SeriesStrMethodType, "removesuffix", inline="always", no_unliteral=True
)
def overload_str_method_removesuffix(S, suffix):
    str_arg_check("removesuffix", "suffix", suffix)
    scalar_text = "if bodo.libs.array_kernels.isna(data, i):\n"
    scalar_text += "  bodo.libs.array_kernels.setna(result, i)\n"
    scalar_text += "else:\n"
    scalar_text += " data_str = data[i]\n"
    scalar_text += " if data_str.endswith(suffix):\n"
    scalar_text += "   result[i] = data_str[:-len(suffix)]\n"
    scalar_text += " else:\n"
    scalar_text += "   result[i] = data_str\n"
    return generate_simple_series_impl(
        ("S", "suffix"), (S, suffix), S.stype, scalar_text
    )


@overload_method(SeriesStrMethodType, "partition", inline="always", no_unliteral=True)
def overload_str_method_partition(S, sep=" ", expand=True):
    str_arg_check("partition", "sep", sep)
    if not is_overload_constant_bool(expand):
        raise_bodo_error(
            "pd.Series.str.partition: requires expand to be a constant boolean"
        )
    unsupported_args = dict(expand=expand)
    arg_defaults = dict(expand=True)
    check_unsupported_args(
        "Series.str.partition",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="Series",
    )

    # Returns a 3-column Dataframe
    scalar_text = "if bodo.libs.array_kernels.isna(data, i):\n"
    scalar_text += "  bodo.libs.array_kernels.setna(res0, i)\n"
    scalar_text += "  bodo.libs.array_kernels.setna(res1, i)\n"
    scalar_text += "  bodo.libs.array_kernels.setna(res2, i)\n"
    scalar_text += "else:\n"
    scalar_text += " data_str = data[i]\n"
    scalar_text += " if sep in data_str:\n"
    scalar_text += "  sep_idx = data_str.index(sep)\n"
    scalar_text += "  res0[i] = data_str[:sep_idx]\n"
    scalar_text += "  res1[i] = sep\n"
    scalar_text += "  res2[i] = data_str[sep_idx+len(sep):]\n"
    scalar_text += " else:\n"
    scalar_text += "  res0[i] = data_str\n"
    scalar_text += "  res1[i] = ''\n"
    scalar_text += "  res2[i] = ''\n"
    return generate_series_to_df_impl(
        ("S", "sep", "expand"),
        (None, "' '", "True"),
        (S, sep, expand),
        (0, 1, 2),
        (S.stype.data, S.stype.data, S.stype.data),
        scalar_text,
    )


@numba.njit
def series_contains_regex(S, pat, case, flags, na, regex):  # pragma: no cover
    with bodo.objmode(out_arr=bodo.boolean_array_type):
        out_arr = pd.array(S.array, "string")._str_contains(pat, case, flags, na, regex)
    return out_arr


@numba.njit
def series_match_regex(S, pat, case, flags, na):  # pragma: no cover
    with bodo.objmode(out_arr=bodo.boolean_array_type):
        out_arr = S.array._str_match(pat, case, flags, na)
    return out_arr


@numba.njit
def series_fullmatch_regex(S, pat, case, flags, na):  # pragma: no cover
    with bodo.objmode(out_arr=bodo.boolean_array_type):
        out_arr = S.array._str_fullmatch(pat, case, flags, na)
    return out_arr


def is_regex_unsupported(pat):
    """Check if it's constant and any of the below patterns are in the regex-pattern."""
    # Based on https://docs.python.org/3/library/re.html,
    # all patterns are supported by boost::xpressive except use of flags
    # as part of the pattern (i.e (?aiLmsux)).
    # See test_re_syntax for examples
    # To keep code simple, this treats escaped \(? as unsupported case as well.
    # These are flags that are used as part of the regular expression
    # and not supported in C++ directly.
    # TODO: [BE-1204] match flags in Python with boost::xpressive::regex_constants
    unsupported_regex = ["(?a", "(?i", "(?L", "(?m", "(?s", "(?u", "(?x", "(?#"]
    if is_overload_constant_str(pat):
        if isinstance(pat, types.StringLiteral):
            pat = pat.literal_value
        return any([x in pat for x in unsupported_regex])
    else:
        return True


@overload_method(SeriesStrMethodType, "contains", no_unliteral=True)
def overload_str_method_contains(S_str, pat, case=True, flags=0, na=np.nan, regex=True):
    not_supported_arg_check("contains", "na", na, np.nan)
    str_arg_check("contains", "pat", pat)
    int_arg_check("contains", "flags", flags)
    # TODO: support other arguments
    # TODO: support dynamic values for regex

    # Error checking for regex argument
    if not is_overload_constant_bool(regex):
        raise BodoError(
            "Series.str.contains(): 'regex' argument should be a constant boolean"
        )
    # Error checking for case argument
    if not is_overload_constant_bool(case):
        raise BodoError(
            "Series.str.contains(): 'case' argument should be a constant boolean"
        )

    # Get value of re.IGNORECASE. It cannot be computed inside the impl
    # since this is a custom enum class (not regular Enum) and numba doesn't
    # support it. https://numba.readthedocs.io/en/stable/reference/pysupported.html#enum
    re_ignorecase_value = re.IGNORECASE.value

    func_text = "def impl(\n"
    func_text += "    S_str, pat, case=True, flags=0, na=np.nan, regex=True\n"
    func_text += "):\n"
    func_text += "  S = S_str._obj\n"
    func_text += "  arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "  index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    func_text += "  l = len(arr)\n"
    func_text += "  out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)\n"

    if is_overload_true(regex):
        # If not known at compile-time or the pattern isn't supported,
        # use Python's re.search in objmode
        if is_regex_unsupported(pat) or flags:
            # optimized version for dictionary encoded arrays
            if S_str.stype.data == bodo.dict_str_arr_type:
                func_text += "  out_arr = bodo.libs.dict_arr_ext.str_series_contains_regex(arr, pat, case, flags, na, regex)\n"
            else:
                func_text += "  out_arr = bodo.hiframes.series_str_impl.series_contains_regex(S, pat, case, flags, na, regex)\n"
        else:
            # get_search_regex handles dictionary encoded arrays as well
            func_text += "  get_search_regex(arr, case, False, bodo.libs.str_ext.unicode_to_utf8(pat), out_arr)\n"

    else:
        # optimized version for dictionary encoded arrays
        if S_str.stype.data == bodo.dict_str_arr_type:
            func_text += "  out_arr = bodo.libs.dict_arr_ext.str_contains_non_regex(arr, pat, case)\n"
        else:
            func_text += "  numba.parfors.parfor.init_prange()\n"
            # Only needed for the non-regex case-insensitive case
            if is_overload_false(case):
                func_text += "  upper_pat = pat.upper()\n"

            func_text += "  for i in numba.parfors.parfor.internal_prange(l):\n"
            func_text += "      if bodo.libs.array_kernels.isna(arr, i):\n"
            func_text += "          bodo.libs.array_kernels.setna(out_arr, i)\n"
            func_text += "      else: \n"
            if is_overload_true(case):
                func_text += "          out_arr[i] = pat in arr[i]\n"
            else:
                func_text += "          out_arr[i] = upper_pat in arr[i].upper()\n"
    func_text += (
        "  return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )
    loc_vars = {}
    exec(
        func_text,
        {
            "re": re,
            "bodo": bodo,
            "numba": numba,
            "np": np,
            "re_ignorecase_value": re_ignorecase_value,
            "get_search_regex": get_search_regex,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


def gen_str_match_impl(S_str, pat, do_full_match, flags):
    """Generates an implementation of str.match or str.full_match depending on the do_full_match flag."""
    # Get value of re.IGNORECASE. It cannot be computed inside the impl
    # since this is a custom enum class (not regular Enum) and numba doesn't
    # support it. https://numba.readthedocs.io/en/stable/reference/pysupported.html#enum
    re_ignorecase_value = re.IGNORECASE.value

    func_text = "def impl(S_str, pat, case=True, flags=0, na=np.nan):\n"
    func_text += "        S = S_str._obj\n"
    func_text += "        arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "        l = len(arr)\n"
    func_text += "        index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "        name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    if not is_regex_unsupported(pat) and flags == 0:
        func_text += "        out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)\n"
        func_text += f"        get_search_regex(arr, case, True, bodo.libs.str_ext.unicode_to_utf8(pat), out_arr, do_full_match={do_full_match})\n"
    # optimized version for dictionary encoded array
    elif S_str.stype.data == bodo.dict_str_arr_type:
        func_text += f"        out_arr = bodo.libs.dict_arr_ext.str_match(arr, pat, case, flags, na, do_full_match={do_full_match})\n"
    else:
        func_text += "        out_arr = series_match_impl(S, pat, case, flags, na)\n"
    func_text += (
        "        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )

    loc_vars = {}
    exec(
        func_text,
        {
            "re": re,
            "bodo": bodo,
            "numba": numba,
            "np": np,
            "re_ignorecase_value": re_ignorecase_value,
            "get_search_regex": get_search_regex,
            "series_match_impl": series_fullmatch_regex
            if do_full_match
            else series_match_regex,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@overload_method(SeriesStrMethodType, "fullmatch", inline="always", no_unliteral=True)
def overload_str_method_fullmatch(S_str, pat, case=True, flags=0, na=np.nan):
    not_supported_arg_check("fullmatch", "na", na, np.nan)
    str_arg_check("fullmatch", "pat", pat)
    int_arg_check("fullmatch", "flags", flags)

    # Error checking for case argument
    if not is_overload_constant_bool(case):
        raise BodoError(
            "Series.str.fullmatch(): 'case' argument should be a constant boolean"
        )

    return gen_str_match_impl(S_str, pat, True, flags)


@overload_method(SeriesStrMethodType, "match", inline="always", no_unliteral=True)
def overload_str_method_match(S_str, pat, case=True, flags=0, na=np.nan):
    not_supported_arg_check("match", "na", na, np.nan)
    str_arg_check("match", "pat", pat)
    int_arg_check("match", "flags", flags)

    # Error checking for case argument
    if not is_overload_constant_bool(case):
        raise BodoError(
            "Series.str.match(): 'case' argument should be a constant boolean"
        )

    return gen_str_match_impl(S_str, pat, False, flags)


@overload_method(SeriesStrMethodType, "cat", no_unliteral=True)
def overload_str_method_cat(S_str, others=None, sep=None, na_rep=None, join="left"):
    # only DataFrame input is currently supported (TODO: support Series/Index/array)
    if not isinstance(others, DataFrameType):
        raise_bodo_error("Series.str.cat(): 'others' must be a DataFrame currently")

    if not is_overload_none(sep):
        str_arg_check("cat", "sep", sep)

    if not is_overload_constant_str(join) or get_overload_const_str(join) != "left":
        raise_bodo_error("Series.str.cat(): 'join' not supported yet")

    func_text = "def impl(S_str, others=None, sep=None, na_rep=None, join='left'):\n"
    func_text += "  S = S_str._obj\n"
    func_text += "  arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "  index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    func_text += "  l = len(arr)\n"

    for i in range(len(others.columns)):
        func_text += f"  data{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(others, {i})\n"

    # optimized path for dictionar-encoded string arrays
    if S_str.stype.data == bodo.dict_str_arr_type and all(
        t == bodo.dict_str_arr_type for t in others.data
    ):
        in_data = ", ".join(f"data{i}" for i in range(len(others.columns)))
        func_text += (
            f"  out_arr = bodo.libs.dict_arr_ext.cat_dict_str((arr, {in_data}), sep)\n"
        )

    else:
        na_check = " or ".join(
            ["bodo.libs.array_kernels.isna(arr, i)"]
            + [
                f"bodo.libs.array_kernels.isna(data{i}, i)"
                for i in range(len(others.columns))
            ]
        )

        func_text += "  out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)\n"
        func_text += "  numba.parfors.parfor.init_prange()\n"

        func_text += "  for i in numba.parfors.parfor.internal_prange(l):\n"
        func_text += f"      if {na_check}:\n"
        func_text += "          bodo.libs.array_kernels.setna(out_arr, i)\n"
        func_text += "          continue\n"

        str_list = ", ".join(
            ["arr[i]"] + [f"data{i}[i]" for i in range(len(others.columns))]
        )

        sep_str = "''" if is_overload_none(sep) else "sep"
        func_text += f"      out_arr[i] = {sep_str}.join([{str_list}])\n"

    func_text += (
        "  return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )

    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    return impl


@overload_method(SeriesStrMethodType, "count", inline="always", no_unliteral=True)
def overload_str_method_count(S_str, pat, flags=0):
    # python str.count() and pandas str.count() are different
    str_arg_check("count", "pat", pat)
    int_arg_check("count", "flags", flags)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_count_dict_impl(S_str, pat, flags=0):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_count(arr, pat, flags)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_count_dict_impl

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
                bodo.libs.array_kernels.setna(out_arr, i)
            else:
                out_arr[i] = str_findall_count(e, str_arr[i])
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "find", inline="always", no_unliteral=True)
def overload_str_method_find(S_str, sub, start=0, end=None):
    str_arg_check("find", "sub", sub)
    int_arg_check("find", "start", start)
    if not is_overload_none(end):
        int_arg_check("find", "end", end)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_find_dict_impl(S_str, sub, start=0, end=None):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_find(arr, sub, start, end)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_find_dict_impl

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
                bodo.libs.array_kernels.setna(out_arr, i)
            else:
                out_arr[i] = str_arr[i].find(sub, start, end)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "rfind", inline="always", no_unliteral=True)
def overload_str_method_rfind(S_str, sub, start=0, end=None):
    str_arg_check("rfind", "sub", sub)
    if start != 0:
        int_arg_check("rfind", "start", start)
    if not is_overload_none(end):
        int_arg_check("rfind", "end", end)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_rfind_dict_impl(S_str, sub, start=0, end=None):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_rfind(arr, sub, start, end)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_rfind_dict_impl

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
                bodo.libs.array_kernels.setna(out_arr, i)
            else:
                out_arr[i] = str_arr[i].rfind(sub, start, end)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "index", inline="always", no_unliteral=True)
def overload_str_method_index(S_str, sub, start=0, end=None):
    """returns the implementation for Series.str.index based on whether the
    underlying data is dictionary-encoded or not. To facilitate error
    synchronization across ranks, we call find instead of index on the
    each string and raise error when -1 is present

    Args:
        S_str: input string series
        sub (string): substring being searched
        start (int, optional): left edge index. Defaults to 0.
        end (int, optional): right edge index. Defaults to None.
    """
    str_arg_check("index", "sub", sub)
    int_arg_check("index", "start", start)
    if not is_overload_none(end):
        int_arg_check("index", "end", end)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_index_dict_impl(S_str, sub, start=0, end=None):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_index(arr, sub, start, end)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_index_dict_impl

    def impl(S_str, sub, start=0, end=None):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        l = len(str_arr)
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(l, np.int64)
        numba.parfors.parfor.init_prange()
        error_flag = False
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.libs.array_kernels.setna(out_arr, i)
            else:
                # index raises ValueError when substring is not found
                # try...except does not work with numba prange so we do not call synchronize_error
                # We work around this by calling find and raise error when -1 is present
                out_arr[i] = str_arr[i].find(sub, start, end)
                if out_arr[i] == -1:
                    error_flag = True
        error_message = "substring not found" if error_flag else ""
        synchronize_error_njit("ValueError", error_message)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "rindex", inline="always", no_unliteral=True)
def overload_str_method_rindex(S_str, sub, start=0, end=None):
    """returns the implementation for Series.str.rindex based on whether the
    underlying data is dictionary-encoded or not. To facilitate error
    synchronization across ranks, we call find instead of index on the
    each string and raise error when -1 is present

    Args:
        S_str: input string series
        sub (string): substring being searched
        start (int, optional): left edge index. Defaults to 0.
        end (int, optional): right edge index. Defaults to None.
    """
    str_arg_check("rindex", "sub", sub)
    int_arg_check("rindex", "start", start)
    if not is_overload_none(end):
        int_arg_check("rindex", "end", end)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_rindex_dict_impl(S_str, sub, start=0, end=None):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_rindex(arr, sub, start, end)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_rindex_dict_impl

    def impl(S_str, sub, start=0, end=None):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        l = len(str_arr)
        out_arr = bodo.libs.int_arr_ext.alloc_int_array(l, np.int64)
        numba.parfors.parfor.init_prange()
        error_flag = False
        for i in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, i):
                bodo.libs.array_kernels.setna(out_arr, i)
            else:
                # rindex raises ValueError when substring is not found
                out_arr[i] = str_arr[i].rindex(sub, start, end)
                if out_arr[i] == -1:
                    error_flag = True
        error_message = "substring not found" if error_flag else ""
        synchronize_error_njit("ValueError", error_message)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(
    SeriesStrMethodType, "slice_replace", inline="always", no_unliteral=True
)
def overload_str_method_slice_replace(S_str, start=0, stop=None, repl=""):
    int_arg_check("slice_replace", "start", start)
    if not is_overload_none(stop):
        int_arg_check("slice_replace", "stop", stop)
    str_arg_check("slice_replace", "repl", repl)

    def impl(S_str, start=0, stop=None, repl=""):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                bodo.libs.array_kernels.setna(out_arr, j)
            else:
                if stop is not None:
                    ending = str_arr[j][stop:]
                else:
                    ending = ""
                out_arr[j] = str_arr[j][:start] + repl + ending
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "repeat", inline="always", no_unliteral=True)
def overload_str_method_repeat(S_str, repeats):
    if isinstance(repeats, types.Integer) or is_overload_constant_int(repeats):
        # optimized version for dictionary encode arrays
        if S_str.stype.data == bodo.dict_str_arr_type:

            def _str_repeat_int_dict_impl(S_str, repeats):  # pragma: no cover
                S = S_str._obj
                arr = bodo.hiframes.pd_series_ext.get_series_data(S)
                index = bodo.hiframes.pd_series_ext.get_series_index(S)
                name = bodo.hiframes.pd_series_ext.get_series_name(S)
                out_arr = bodo.libs.dict_arr_ext.str_repeat_int(arr, repeats)
                return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

            return _str_repeat_int_dict_impl

        def impl(S_str, repeats):  # pragma: no cover
            S = S_str._obj
            str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            numba.parfors.parfor.init_prange()
            l = len(str_arr)
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(str_arr, j):
                    bodo.libs.array_kernels.setna(out_arr, j)
                else:
                    out_arr[j] = str_arr[j] * repeats
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return impl
    elif is_overload_constant_list(repeats):
        list_vals = get_overload_const_list(repeats)
        legal_array_input = all([isinstance(val, int) for val in list_vals])
    elif is_list_like_index_type(repeats) and isinstance(repeats.dtype, types.Integer):
        legal_array_input = True
    else:  # pragma: no cover
        legal_array_input = False

    if legal_array_input:
        # when S_str is a heterogeneous sequence of integers, dictionary encoding array
        # does not provide any benefits over regular arrays, thus
        # str_repeat_seq is not implemented.
        def impl(S_str, repeats):  # pragma: no cover
            S = S_str._obj
            str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            repeat_arr = bodo.utils.conversion.coerce_to_array(repeats)
            numba.parfors.parfor.init_prange()
            l = len(str_arr)
            # TODO(Nick): Check that repeats and str_arr are the same size.
            out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
            for j in numba.parfors.parfor.internal_prange(l):
                if bodo.libs.array_kernels.isna(str_arr, j):
                    bodo.libs.array_kernels.setna(out_arr, j)
                else:
                    out_arr[j] = str_arr[j] * repeat_arr[j]
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return impl
    else:  # pragma: no cover
        raise BodoError(
            "Series.str.repeat(): repeats argument must either be an integer or a sequence of integers"
        )


def create_ljust_rjust_center_overload(func_name):
    func_text = (
        "def dict_impl(S_str, width, fillchar=' '):\n"
        "    S = S_str._obj\n"
        "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        f"    out_arr = bodo.libs.dict_arr_ext.str_{func_name}(arr, width, fillchar)\n"
        "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
        "def impl(S_str, width, fillchar=' '):\n"
        "    S = S_str._obj\n"
        "    str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        "    numba.parfors.parfor.init_prange()\n"
        "    l = len(str_arr)\n"
        "    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)\n"
        "    for j in numba.parfors.parfor.internal_prange(l):\n"
        "        if bodo.libs.array_kernels.isna(str_arr, j):\n"
        "            bodo.libs.array_kernels.setna(out_arr, j)\n"
        "        else:\n"
        f"            out_arr[j] = str_arr[j].{func_name}(width, fillchar)\n"
        "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )
    loc_vars = {}
    glob_vals = {
        "bodo": bodo,
        "numba": numba,
    }
    exec(func_text, glob_vals, loc_vars)
    impl = loc_vars["impl"]
    dict_impl = loc_vars["dict_impl"]

    def overload_ljust_rjust_center_method(S_str, width, fillchar=" "):
        common_validate_padding(func_name, width, fillchar)

        if S_str.stype.data == bodo.dict_str_arr_type:
            return dict_impl
        return impl

    return overload_ljust_rjust_center_method


def _install_ljust_rjust_center():
    # install ljust/rjust/center
    for func in ["ljust", "rjust", "center"]:
        impl = create_ljust_rjust_center_overload(func)
        overload_method(SeriesStrMethodType, func, inline="always", no_unliteral=True)(
            impl
        )


_install_ljust_rjust_center()


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

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_pad_dict_impl(
            S_str, width, side="left", fillchar=" "
        ):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            if side == "left":
                out_arr = bodo.libs.dict_arr_ext.str_rjust(arr, width, fillchar)
            elif side == "right":
                out_arr = bodo.libs.dict_arr_ext.str_ljust(arr, width, fillchar)
            elif side == "both":
                out_arr = bodo.libs.dict_arr_ext.str_center(arr, width, fillchar)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_pad_dict_impl

    def impl(S_str, width, side="left", fillchar=" "):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.libs.array_kernels.setna(out_arr, j)
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

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_zfill_dict_impl(S_str, width):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_zfill(arr, width)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_zfill_dict_impl

    def impl(S_str, width):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.libs.array_kernels.setna(out_arr, j)
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

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_slice_dict_impl(
            S_str, start=None, stop=None, step=None
        ):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_slice(arr, start, stop, step)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_slice_dict_impl

    def impl(S_str, start=None, stop=None, step=None):  # pragma: no cover
        S = S_str._obj
        str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)
        name = bodo.hiframes.pd_series_ext.get_series_name(S)
        index = bodo.hiframes.pd_series_ext.get_series_index(S)
        numba.parfors.parfor.init_prange()
        l = len(str_arr)
        out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(l, -1)
        for j in numba.parfors.parfor.internal_prange(l):
            if bodo.libs.array_kernels.isna(str_arr, j):
                out_arr[j] = ""
                bodo.libs.array_kernels.setna(out_arr, j)
            else:
                out_arr[j] = str_arr[j][start:stop:step]
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "startswith", inline="always", no_unliteral=True)
def overload_str_method_startswith(S_str, pat, na=np.nan):
    not_supported_arg_check("startswith", "na", na, np.nan)
    str_arg_check("startswith", "pat", pat)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_startswith_dict_impl(S_str, pat, na=np.nan):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_startswith(arr, pat, na)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_startswith_dict_impl

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
                bodo.libs.array_kernels.setna(out_arr, i)
            else:
                out_arr[i] = str_arr[i].startswith(pat)
        return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

    return impl


@overload_method(SeriesStrMethodType, "endswith", inline="always", no_unliteral=True)
def overload_str_method_endswith(S_str, pat, na=np.nan):
    not_supported_arg_check("endswith", "na", na, np.nan)
    str_arg_check("endswith", "pat", pat)

    # optimized version for dictionary encoded arrays
    if S_str.stype.data == bodo.dict_str_arr_type:

        def _str_endswith_dict_impl(S_str, pat, na=np.nan):  # pragma: no cover
            S = S_str._obj
            arr = bodo.hiframes.pd_series_ext.get_series_data(S)
            index = bodo.hiframes.pd_series_ext.get_series_index(S)
            name = bodo.hiframes.pd_series_ext.get_series_name(S)
            out_arr = bodo.libs.dict_arr_ext.str_endswith(arr, pat, na)
            return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)

        return _str_endswith_dict_impl

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
                bodo.libs.array_kernels.setna(out_arr, i)
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

    # check if the array is dictionary encoded
    if S_str.stype.data == bodo.dict_str_arr_type:
        # optimized version for dictionary encoded arrays
        func_text = "def impl(S_str, pat, flags=0, expand=True):\n"
        func_text += "  S = S_str._obj\n"
        func_text += "  arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "  index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += f"  out_arr_list = bodo.libs.dict_arr_ext.str_extract(arr, pat, flags, {n_cols})\n"
        for i in range(n_cols):
            func_text += f"  out_arr_{i} = out_arr_list[{i}]\n"
    else:
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
            func_text += "  out_arr_{0} = bodo.libs.str_arr_ext.pre_alloc_string_array(n, -1)\n".format(
                i
            )
        func_text += "  for j in numba.parfors.parfor.internal_prange(n):\n"
        func_text += "      if bodo.libs.array_kernels.isna(str_arr, j):\n"
        for i in range(n_cols):
            func_text += "          out_arr_{}[j] = ''\n".format(i)
            func_text += (
                "          bodo.libs.array_kernels.setna(out_arr_{}, j)\n".format(i)
            )
        func_text += "      else:\n"
        func_text += "          m = regex.search(str_arr[j])\n"
        func_text += "          if m:\n"
        func_text += "            g = m.groups()\n"
        for i in range(n_cols):
            func_text += "            out_arr_{0}[j] = g[{0}]\n".format(i)
        func_text += "          else:\n"
        for i in range(n_cols):
            func_text += "            out_arr_{}[j] = ''\n".format(i)
            func_text += (
                "            bodo.libs.array_kernels.setna(out_arr_{}, j)\n".format(i)
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
def overload_str_method_extractall(S_str, pat, flags=0):
    columns, _ = _get_column_names_from_regex(pat, flags, "extractall")
    n_cols = len(columns)
    is_index_string = isinstance(S_str.stype.index, StringIndexType)
    is_multi_group = n_cols > 1
    multi_group = "_multi" if is_multi_group else ""

    # check if the string array is dictionary encoded
    if S_str.stype.data == bodo.dict_str_arr_type:
        # optimized version for dictionary encoded arrays
        func_text = "def impl(S_str, pat, flags=0):\n"
        func_text += "  S = S_str._obj\n"
        func_text += "  arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        func_text += "  index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        func_text += "  name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        func_text += "  index_arr = bodo.utils.conversion.index_to_array(index)\n"
        func_text += "  index_name = bodo.hiframes.pd_index_ext.get_index_name(index)\n"
        func_text += "  regex = re.compile(pat, flags=flags)\n"
        func_text += "  out_ind_arr, out_match_arr, out_arr_list = "
        func_text += f"bodo.libs.dict_arr_ext.str_extractall{multi_group}(\n"
        func_text += f"arr, regex, {n_cols}, index_arr)\n"
        for i in range(n_cols):
            func_text += f"  out_arr_{i} = out_arr_list[{i}]\n"
        func_text += (
            "  out_index = bodo.hiframes.pd_multi_index_ext.init_multi_index(\n"
        )
        func_text += "    (out_ind_arr, out_match_arr), (index_name, 'match'))\n"
    else:
        # generate one loop for finding character count and another for computation
        # TODO: avoid multiple loops if possible, or even avoid inlined loops if needed
        func_text = "def impl(S_str, pat, flags=0):\n"
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
        # TODO: refactor with arr_builder
        # using a sentinel function to specify that the arrays are local and no need for
        # distributed transformation
        func_text += "  out_n = bodo.libs.distributed_api.local_alloc_size(out_n_l[0], str_arr)\n"
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
        func_text += "        bodo.libs.distributed_api.set_arr_local(out_match_arr, out_ind, k)\n"
        func_text += "        out_ind += 1\n"
        func_text += (
            "  out_index = bodo.hiframes.pd_multi_index_ext.init_multi_index(\n"
        )
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
    # All of the functions except for strip take no arguments.
    # Strip takes one optional argument, which is the character(s) to strip.
    # In order to resolve this with minmal code duplication, we create/exec the func text
    # outside of the overload, and then
    # return the function with two different overload declarations

    # func_text for regular string arrays
    is_strip = func_name in ["lstrip", "rstrip", "strip"]
    func_text = (
        f"def f({'S_str, to_strip=None' if is_strip else 'S_str'}):\n"
        "    S = S_str._obj\n"
        "    str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        "    str_arr = decode_if_dict_array(str_arr)\n"
        "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        "    numba.parfors.parfor.init_prange()\n"
        "    n = len(str_arr)\n"
        f"    num_chars = {'-1' if is_strip else 'num_total_chars(str_arr)'}\n"
        "    out_arr = bodo.libs.str_arr_ext.pre_alloc_string_array(n, num_chars)\n"
        "    for j in numba.parfors.parfor.internal_prange(n):\n"
        "        if bodo.libs.array_kernels.isna(str_arr, j):\n"
        '            out_arr[j] = ""\n'
        "            bodo.libs.array_kernels.setna(out_arr, j)\n"
        "        else:\n"
        f"            out_arr[j] = str_arr[j].{func_name}({'to_strip' if is_strip else ''})\n"
        "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )

    # func_text for dictionary-encoded string array
    func_text += (
        f"def _dict_impl({'S_str, to_strip=None' if is_strip else 'S_str'}):\n"
        "    S = S_str._obj\n"
        "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
        "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
        "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
        f"    out_arr = bodo.libs.dict_arr_ext.str_{func_name}({'arr, to_strip' if is_strip else 'arr'})\n"
        "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "num_total_chars": bodo.libs.str_arr_ext.num_total_chars,
            "get_utf8_size": bodo.libs.str_arr_ext.get_utf8_size,
            "decode_if_dict_array": bodo.utils.typing.decode_if_dict_array,
        },
        loc_vars,
    )
    f = loc_vars["f"]
    _dict_impl = loc_vars["_dict_impl"]

    if is_strip:

        def overload_strip_method(S_str, to_strip=None):
            if not is_overload_none(to_strip):
                str_arg_check(func_name, "to_strip", to_strip)
            if S_str.stype.data == bodo.dict_str_arr_type:
                return _dict_impl
            return f

        return overload_strip_method
    else:

        def overload_str_method_dict_supported(S_str):
            if S_str.stype.data == bodo.dict_str_arr_type:
                return _dict_impl
            return f

        return overload_str_method_dict_supported


def create_str2bool_methods_overload(func_name):
    func_text = "def dict_impl(S_str):\n"
    func_text += "    S = S_str._obj\n"
    func_text += "    arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    func_text += f"    out_arr = bodo.libs.dict_arr_ext.str_{func_name}(arr)\n"
    func_text += (
        "    return bodo.hiframes.pd_series_ext.init_series(out_arr, index, name)\n"
    )
    func_text += "def impl(S_str):\n"
    func_text += "    S = S_str._obj\n"
    func_text += "    str_arr = bodo.hiframes.pd_series_ext.get_series_data(S)\n"
    func_text += "    index = bodo.hiframes.pd_series_ext.get_series_index(S)\n"
    func_text += "    name = bodo.hiframes.pd_series_ext.get_series_name(S)\n"
    func_text += "    numba.parfors.parfor.init_prange()\n"
    func_text += "    l = len(str_arr)\n"
    func_text += "    out_arr = bodo.libs.bool_arr_ext.alloc_bool_array(l)\n"
    func_text += "    for i in numba.parfors.parfor.internal_prange(l):\n"
    func_text += "        if bodo.libs.array_kernels.isna(str_arr, i):\n"
    func_text += "            bodo.libs.array_kernels.setna(out_arr, i)\n"
    func_text += "        else:\n"
    func_text += "            out_arr[i] = np.bool_(str_arr[i].{}())\n".format(
        func_name
    )
    func_text += "    return bodo.hiframes.pd_series_ext.init_series(\n"
    func_text += "      out_arr,index, name)\n"
    loc_vars = {}
    exec(
        func_text,
        {
            "bodo": bodo,
            "numba": numba,
            "np": np,
        },
        loc_vars,
    )
    impl = loc_vars["impl"]
    dict_impl = loc_vars["dict_impl"]

    def overload_str2bool_methods(S_str):
        if S_str.stype.data == bodo.dict_str_arr_type:
            return dict_impl
        return impl

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

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


@register_model(SeriesCatMethodType)
class SeriesCatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.stype)]
        super(SeriesCatModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(SeriesCatMethodType, "obj", "_obj")


@intrinsic
def init_series_cat_method(typingctx, obj):
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
        return bodo.hiframes.pd_series_ext.init_series(
            bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(arr), index, name
        )

    return impl


unsupported_cat_attrs = {
    "categories",
    "ordered",
}

unsupported_cat_methods = {
    "rename_categories",
    "reorder_categories",
    "add_categories",
    "remove_categories",
    "remove_unused_categories",
    "set_categories",
    "as_ordered",
    "as_unordered",
}


def _install_catseries_unsupported():
    """install an overload that raises BodoError for unsupported Series cat methods"""

    for attr_name in unsupported_cat_attrs:
        full_name = "Series.cat." + attr_name
        bodo.overload_unsupported_attribute(SeriesCatMethodType, attr_name, full_name)

    for fname in unsupported_cat_methods:
        full_name = "Series.cat." + fname
        bodo.overload_unsupported_method(SeriesCatMethodType, fname, full_name)


_install_catseries_unsupported()


unsupported_str_methods = {
    "decode",
    "encode",
    "findall",
    "normalize",
    "rpartition",
    "rsplit",
    "translate",
    "wrap",
    "get_dummies",
}


def _install_strseries_unsupported():
    """install an overload that raises BodoError for unsupported Series str methods"""

    for fname in unsupported_str_methods:
        full_name = "Series.str." + fname
        bodo.overload_unsupported_method(SeriesStrMethodType, fname, full_name)


_install_strseries_unsupported()
