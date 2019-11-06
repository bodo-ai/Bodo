# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""Old code for some auxiliary functionality. Needs to be refactored.
"""
from collections import namedtuple
import pandas as pd
import numpy as np

import numba
from numba import ir
from numba.ir_utils import mk_unique_var
from numba import types, cgutils
import numba.array_analysis
from numba.typing import signature
from numba.typing.templates import infer_global, AbstractTemplate
from numba.extending import overload, intrinsic
from numba.targets.imputils import (
    impl_ret_new_ref,
    impl_ret_borrowed,
    iternext_impl,
    RefType,
)
from numba.targets.arrayobj import _getitem_array1d
from numba.extending import register_model, models

import bodo
from bodo.libs.str_ext import string_type
from bodo.libs.str_arr_ext import string_array_type, is_str_arr_typ
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.bool_arr_ext import boolean_array

from bodo.utils.utils import build_set
from numba.targets.imputils import lower_builtin, impl_ret_untracked
from bodo.hiframes.pd_series_ext import (
    SeriesType,
    SeriesPayloadType,
    if_arr_to_series_type,
    if_series_to_array_type,
)
from bodo.hiframes.pd_index_ext import DatetimeIndexType, TimedeltaIndexType
from bodo.ir.sort import (
    alltoallv_tup,
    finalize_shuffle_meta,
    update_shuffle_meta,
    alloc_pre_shuffle_metadata,
)
from bodo.ir.join import write_send_buff

from numba.targets.arrayobj import make_array
from bodo.utils.utils import unliteral_all
import llvmlite.llvmpy.core as lc


def get_series_payload(context, builder, series_type, value):
    meminfo = cgutils.create_struct_proxy(series_type)(context, builder, value).meminfo
    payload_type = SeriesPayloadType(series_type)
    payload = context.nrt.meminfo_data(builder, meminfo)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)


@intrinsic
def _get_series_data(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(context, builder, series_typ.data, series_payload.data)

    ret_typ = series_typ.data
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def update_series_data(typingctx, series_typ, arr_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        series_payload.data = args[1]
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[1], args[1])
        return

    sig = types.none(series_typ, arr_typ)
    return sig, codegen


@intrinsic
def update_series_index(typingctx, series_typ, arr_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        series_payload.index = args[1]
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[1], args[1])
        return

    sig = types.none(series_typ, arr_typ)
    return sig, codegen


@intrinsic
def _get_series_index(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, series_typ.index, series_payload.index
        )

    ret_typ = series_typ.index
    sig = signature(ret_typ, series_typ)
    return sig, codegen


@intrinsic
def _get_series_name(typingctx, series_typ=None):
    def codegen(context, builder, signature, args):
        series_payload = get_series_payload(
            context, builder, signature.args[0], args[0]
        )
        # TODO: is borrowing None reference ok here?
        return impl_ret_borrowed(
            context, builder, signature.return_type, series_payload.name
        )

    sig = signature(series_typ.name_typ, series_typ)
    return sig, codegen


# this function should be used for getting S._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_data(S):
    return lambda S: _get_series_data(S)


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_index(S):
    return lambda S: _get_series_index(S)


@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_series_name(S):
    return lambda S: _get_series_name(S)


@numba.generated_jit(nopython=True)
def get_index_data(S):
    return lambda S: S._data


@numba.generated_jit(nopython=True)
def get_index_name(S):
    return lambda S: S._name


# array analysis extension
def get_series_data_equiv(self, scope, equiv_set, args, kws):
    assert len(args) == 1 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


from numba.array_analysis import ArrayAnalysis

ArrayAnalysis._analyze_op_call_bodo_hiframes_api_get_series_data = get_series_data_equiv


def init_series_equiv(self, scope, equiv_set, args, kws):
    assert len(args) >= 1 and not kws
    # TODO: add shape for index
    var = args[0]
    if equiv_set.has_shape(var):
        return var, []
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_api_init_series = init_series_equiv


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


def alias_ext_init_series(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    if len(args) > 1:  # has index
        numba.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


def alias_ext_init_integer_array(lhs_name, args, alias_map, arg_aliases):
    assert len(args) == 2
    numba.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    numba.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.ir_utils.alias_func_extensions[
    ("init_series", "bodo.hiframes.api")
] = alias_ext_init_series
numba.ir_utils.alias_func_extensions[
    ("get_series_data", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_series_index", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_datetime_index", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_timedelta_index", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_numeric_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_string_index", "bodo.hiframes.pd_index_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_index_data", "bodo.hiframes.api")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func
# TODO: init_dataframe
numba.ir_utils.alias_func_extensions[
    ("init_integer_array", "bodo.libs.int_arr_ext")
] = alias_ext_init_integer_array
numba.ir_utils.alias_func_extensions[
    ("get_int_arr_data", "bodo.libs.int_arr_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_int_arr_bitmap", "bodo.libs.int_arr_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("init_bool_array", "bodo.libs.bool_arr_ext")
] = alias_ext_init_integer_array
numba.ir_utils.alias_func_extensions[
    ("get_bool_arr_data", "bodo.libs.bool_arr_ext")
] = alias_ext_dummy_func
numba.ir_utils.alias_func_extensions[
    ("get_bool_arr_bitmap", "bodo.libs.bool_arr_ext")
] = alias_ext_dummy_func


def construct_series(context, builder, series_type, data_val, index_val, name_val):
    # create payload struct and store values
    payload_type = SeriesPayloadType(series_type)
    series_payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    series_payload.data = data_val
    series_payload.index = index_val
    series_payload.name = name_val

    # create meminfo and store payload
    payload_ll_type = context.get_data_type(payload_type)
    payload_size = context.get_abi_sizeof(payload_ll_type)
    meminfo = context.nrt.meminfo_alloc(
        builder, context.get_constant(types.uintp, payload_size)
    )
    meminfo_data_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_data_ptr, payload_ll_type.as_pointer())
    builder.store(series_payload._getvalue(), meminfo_data_ptr)

    # create Series struct
    series = cgutils.create_struct_proxy(series_type)(context, builder)
    series.meminfo = meminfo
    # Set parent to NULL
    series.parent = cgutils.get_null_value(series.parent.type)
    return series._getvalue()


@intrinsic
def init_series(typingctx, data, index=None, name=None):
    """Create a Series with provided data, index and name values.
    Used as a single constructor for Series and assigning its data, so that
    optimization passes can look for init_series() to see if underlying
    data has changed, and get the array variables from init_series() args if
    not changed.
    """

    index = types.none if index is None else index
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, index_val, name_val = args
        series_type = signature.return_type

        series_val = construct_series(
            context, builder, series_type, data_val, index_val, name_val
        )

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], index_val)
            context.nrt.incref(builder, signature.args[2], name_val)

        return series_val

    dtype = data.dtype
    # XXX pd.DataFrame() calls init_series for even Series since it's untyped
    data = if_series_to_array_type(data)
    ret_typ = SeriesType(dtype, data, index, name)
    sig = signature(ret_typ, data, index, name)
    return sig, codegen


@intrinsic
def init_datetime_index(typingctx, data, name=None):
    """Create a DatetimeIndex with provided data and name values.
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create dt_index struct and store values
        dt_index = cgutils.create_struct_proxy(signature.return_type)(context, builder)
        dt_index.data = data_val
        dt_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], name_val)

        return dt_index._getvalue()

    ret_typ = DatetimeIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen


@intrinsic
def init_timedelta_index(typingctx, data, name=None):
    """Create a TimedeltaIndex with provided data and name values.
    """
    name = types.none if name is None else name
    name = types.unliteral(name)

    def codegen(context, builder, signature, args):
        data_val, name_val = args
        # create timedelta_index struct and store values
        timedelta_index = cgutils.create_struct_proxy(signature.return_type)(
            context, builder
        )
        timedelta_index.data = data_val
        timedelta_index.name = name_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], data_val)
            context.nrt.incref(builder, signature.args[1], name_val)

        return timedelta_index._getvalue()

    ret_typ = TimedeltaIndexType(name)
    sig = signature(ret_typ, data, name)
    return sig, codegen
