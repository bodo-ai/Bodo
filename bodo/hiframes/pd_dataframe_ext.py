# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implement pd.DataFrame typing and data model handling.
"""
import operator
import warnings
import json
from collections import namedtuple
import pandas as pd
import numpy as np
import numba
from numba.core import types, cgutils
from bodo.hiframes.pd_index_ext import StringIndexType
from numba.extending import (
    models,
    register_model,
    lower_cast,
    infer_getattr,
    type_callable,
    infer,
    overload,
    make_attribute_wrapper,
    intrinsic,
    lower_builtin,
    overload_method,
)
from numba.core.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
from numba.parfors.array_analysis import ArrayAnalysis
from numba.core.imputils import impl_ret_borrowed, lower_constant
from llvmlite import ir as lir

import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.hiframes.series_indexing import SeriesIlocType
from bodo.hiframes.pd_index_ext import RangeIndexType, NumericIndexType
from bodo.libs.str_ext import string_type, unicode_to_char_ptr
from bodo.utils.typing import (
    BodoWarning,
    BodoError,
    is_overload_none,
    is_overload_constant_bool,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_constant_list,
    is_overload_true,
    is_overload_false,
    is_overload_zero,
    is_dtype_nullable,
    get_overload_const_str,
    get_overload_const_list,
    is_overload_bool_list,
    get_index_names,
    get_index_data_arr_types,
    raise_const_error,
    is_overload_constant_tuple,
    get_overload_const_tuple,
    get_overload_const_int,
    is_overload_constant_int,
    raise_bodo_error,
    check_unsupported_args,
    ensure_constant_arg,
    ensure_constant_values,
    create_unsupported_overload,
    get_overload_const,
)
from bodo.utils.transform import (
    get_const_func_output_type,
    gen_const_tup,
    get_const_tup_vals,
)
from bodo.utils.conversion import index_to_array
from bodo.libs.array import array_to_info, arr_info_list_to_table
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.libs.str_arr_ext import string_array_type, str_arr_from_sequence
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.distributed_api import bcast_scalar, bcast
from bodo.hiframes.pd_index_ext import is_pd_index_type
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.io import csv_cpp, json_cpp
import llvmlite.binding as ll


_csv_write = types.ExternalFunction(
    "csv_write",
    types.void(types.voidptr, types.voidptr, types.int64, types.int64, types.bool_),
)
ll.add_symbol("csv_write", csv_cpp.csv_write)

_json_write = types.ExternalFunction(
    "json_write",
    types.void(
        types.voidptr, types.voidptr, types.int64, types.int64, types.bool_, types.bool_
    ),
)
ll.add_symbol("json_write", json_cpp.json_write)


class DataFrameType(types.ArrayCompatible):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects.
    """

    def __init__(self, data=None, index=None, columns=None, has_parent=False):
        # data is tuple of Array types (not Series)
        # index is Index obj (not Array type)
        # columns is a tuple of column names (strings, ints, or tuples in case of
        # MultiIndex)

        self.data = data
        if index is None:
            index = RangeIndexType(types.none)
        self.index = index
        self.columns = columns
        # keeping whether it is unboxed from Python to enable reflection of new
        # columns
        self.has_parent = has_parent
        super(DataFrameType, self).__init__(
            name="dataframe({}, {}, {}, {})".format(data, index, columns, has_parent)
        )

    def copy(self, index=None, has_parent=None):
        # XXX is copy necessary?
        if index is None:
            index = self.index.copy()
        data = tuple(a.copy() for a in self.data)
        if has_parent is None:
            has_parent = self.has_parent
        return DataFrameType(data, index, self.columns, has_parent)

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 2, "C")

    @property
    def key(self):
        # needed?
        return self.data, self.index, self.columns, self.has_parent

    def unify(self, typingctx, other):
        """unifies two possible dataframe types into a single type
        see test_dataframe.py::test_df_type_unify_error
        """
        if (
            isinstance(other, DataFrameType)
            and len(other.data) == len(self.data)
            and other.columns == self.columns
            and other.has_parent == self.has_parent
        ):
            new_index = self.index.unify(typingctx, other.index)
            data = tuple(a.unify(typingctx, b) for a, b in zip(self.data, other.data))
            # NOTE: unification is an extreme corner case probably, since arrays can
            # be unified only if just their layout or alignment is different.
            # That doesn't happen in df case since all arrays are 1D and C layout.
            # see: https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/types/npytypes.py#L436
            if new_index is not None and None not in data:  # pragma: no cover
                return DataFrameType(data, new_index, self.columns, self.has_parent)

    def can_convert_to(self, typingctx, other):
        return
        # overload resolution tries to convert for even get_dataframe_data()
        # TODO: find valid conversion possibilities
        # if (isinstance(other, DataFrameType)
        #         and len(other.data) == len(self.data)
        #         and other.columns == self.columns):
        #     data_convert = max(a.can_convert_to(typingctx, b)
        #                         for a,b in zip(self.data, other.data))
        #     if self.index == types.none and other.index == types.none:
        #         return data_convert
        #     if self.index != types.none and other.index != types.none:
        #         return max(data_convert,
        #             self.index.can_convert_to(typingctx, other.index))

    def is_precise(self):
        return all(a.is_precise() for a in self.data) and self.index.is_precise()


# payload type inside meminfo so that mutation are seen by all references
class DataFramePayloadType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        super(DataFramePayloadType, self).__init__(
            name="DataFramePayloadType({})".format(df_type)
        )


# TODO: encapsulate in meminfo since dataframe is mutible, for example:
# df = pd.DataFrame({'A': A})
# df2 = df
# if cond:
#    df['A'] = B
# df2.A
# TODO: meminfo for reference counting of dataframes
@register_model(DataFramePayloadType)
class DataFramePayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        n_cols = len(fe_type.df_type.columns)
        members = [
            ("data", types.Tuple(fe_type.df_type.data)),
            ("index", fe_type.df_type.index),
            # for lazy unboxing of df coming from Python (usually argument)
            # list of flags noting which columns and index are unboxed
            # index flag is last
            ("unboxed", types.UniTuple(types.int8, n_cols + 1)),
            ("parent", types.pyobject),
        ]
        super(DataFramePayloadModel, self).__init__(dmm, fe_type, members)


@register_model(DataFrameType)
class DataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        payload_type = DataFramePayloadType(fe_type)
        # payload_type = types.Opaque('Opaque.DataFrame')
        # TODO: does meminfo decref content when object is deallocated?
        members = [
            ("columns", numba.typeof(fe_type.columns)),
            ("meminfo", types.MemInfoPointer(payload_type)),
            # for boxed DataFrames, enables updating original DataFrame object
            ("parent", types.pyobject),
        ]
        super(DataFrameModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameType, "columns", "_columns")
make_attribute_wrapper(DataFrameType, "parent", "_parent")


@infer_getattr
class DataFrameAttribute(AttributeTemplate):
    key = DataFrameType

    @bound_function("df.apply", no_unliteral=True)
    def resolve_apply(self, df, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws.get("func", None)

        # check axis
        axis = args[1] if len(args) > 1 else kws.get("axis", None)
        if not (is_overload_constant_int(axis) and get_overload_const_int(axis) == 1):
            raise BodoError("only apply() with axis=1 supported")

        # using NamedTuple instead of Series, TODO: pass Series
        Row = namedtuple("R", df.columns)

        # the data elements come from getitem of Series to perform conversion
        # e.g. dt64 to timestamp in TestDate.test_ts_map_date2
        dtypes = []
        for arr_typ in df.data:
            series_typ = SeriesType(arr_typ.dtype, arr_typ, df.index, string_type)
            # iloc necessary since Series getitem may not be supported for df.index
            el_typ = self.context.resolve_function_type(
                operator.getitem, (SeriesIlocType(series_typ), types.int64), {}
            ).return_type
            dtypes.append(el_typ)

        row_typ = types.NamedTuple(dtypes, Row)
        try:
            f_return_type = get_const_func_output_type(func, (row_typ,), self.context)
        except:
            raise BodoError("DataFrame.apply(): user-defined function not supported")

        # unbox Timestamp to dt64 in Series (TODO: timedelta64)
        if f_return_type == bodo.hiframes.pd_timestamp_ext.pandas_timestamp_type:
            f_return_type = types.NPDatetime("ns")
        return signature(SeriesType(f_return_type, index=df.index), *args)

    def generic_resolve(self, df, attr):
        # column selection
        if attr in df.columns:
            ind = df.columns.index(attr)
            arr_typ = df.data[ind]
            return SeriesType(arr_typ.dtype, arr_typ, df.index, string_type)

        # level selection in multi-level df
        if len(df.columns) > 0 and isinstance(df.columns[0], tuple):
            new_names = []
            new_data = []
            # make sure attr is actually in the levels, not something like df.shape
            level_found = False
            for i, v in enumerate(df.columns):
                if v[0] != attr:
                    continue
                level_found = True
                # output names are str in 2 level case, not tuple
                # TODO: test more than 2 levels
                new_names.append(v[1] if len(v) == 2 else v[1:])
                new_data.append(df.data[i])
            if level_found:
                return DataFrameType(tuple(new_data), df.index, tuple(new_names))


# don't convert literal types to non-literal and rerun the typing template
DataFrameAttribute._no_unliteral = True


def decref_df_data(context, builder, payload, df_type):
    """call decref() on all data arrays and index of dataframe
    """
    # decref all unboxed arrays
    for i in range(len(df_type.data)):
        unboxed = builder.extract_value(payload.unboxed, i)
        is_unboxed = builder.icmp_unsigned("==", unboxed, lir.Constant(unboxed.type, 1))

        with builder.if_then(is_unboxed):
            arr = builder.extract_value(payload.data, i)
            context.nrt.decref(builder, df_type.data[i], arr)

    # decref index
    # NOTE: currently, Index is always unboxed so no check of unboxed flag, TODO: fix
    context.nrt.decref(builder, df_type.index, payload.index)
    # last unboxed flag is for index
    # index_unboxed = builder.extract_value(payload.unboxed, len(df_type.data))
    # is_index_unboxed = builder.icmp_unsigned(
    #     "==", index_unboxed, lir.Constant(index_unboxed.type, 1)
    # )
    # with builder.if_then(is_index_unboxed):
    #     context.nrt.decref(builder, df_type.index, payload.index)


def define_df_dtor(context, builder, df_type, payload_type):
    """
    Define destructor for dataframe type if not already defined
    Similar to Numba's List dtor:
    https://github.com/numba/numba/blob/cc7e7c7cfa6389b54d3b5c2c95751c97eb531a96/numba/targets/listobj.py#L273
    """
    mod = builder.module
    # Declare dtor
    fnty = lir.FunctionType(lir.VoidType(), [cgutils.voidptr_t])
    # TODO(ehsan): do we need to sanitize the name in any case?
    fn = mod.get_or_insert_function(fnty, name=".dtor.df.{}".format(df_type))

    # End early if the dtor is already defined
    if not fn.is_declaration:
        return fn

    fn.linkage = "linkonce_odr"
    # Populate the dtor
    builder = lir.IRBuilder(fn.append_basic_block())
    base_ptr = fn.args[0]  # void*

    # get payload struct
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_data_helper(builder, payload_type, ref=payload_ptr)

    decref_df_data(context, builder, payload, df_type)

    # decref parent object
    has_parent = cgutils.is_not_null(builder, payload.parent)
    with builder.if_then(has_parent):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()  # acquire GIL
        pyapi.decref(payload.parent)
        pyapi.gil_release(gil_state)  # release GIL

    builder.ret_void()
    return fn


def construct_dataframe(
    context, builder, df_type, data_tup, index_val, column_tup, unboxed_tup, parent=None
):

    # create payload struct and store values
    payload_type = DataFramePayloadType(df_type)
    dataframe_payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    dataframe_payload.data = data_tup
    dataframe_payload.index = index_val
    dataframe_payload.unboxed = unboxed_tup

    # create meminfo and store payload
    payload_ll_type = context.get_data_type(payload_type)
    payload_size = context.get_abi_sizeof(payload_ll_type)
    dtor_fn = define_df_dtor(context, builder, df_type, payload_type)
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, payload_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, payload_ll_type.as_pointer())

    # create dataframe struct
    dataframe = cgutils.create_struct_proxy(df_type)(context, builder)
    dataframe.columns = column_tup
    dataframe.meminfo = meminfo
    if parent is None:
        # Set parent to NULL
        dataframe.parent = cgutils.get_null_value(dataframe.parent.type)
    else:
        dataframe.parent = parent
        dataframe_payload.parent = parent
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()  # acquire GIL
        pyapi.incref(parent)
        pyapi.gil_release(gil_state)  # release GIL

    builder.store(dataframe_payload._getvalue(), meminfo_data_ptr)
    return dataframe._getvalue()


@intrinsic
def init_dataframe(typingctx, data_tup_typ, index_typ, col_names_typ=None):
    """Create a DataFrame with provided data, index and columns values.
    Used as a single constructor for DataFrame and assigning its data, so that
    optimization passes can look for init_dataframe() to see if underlying
    data has changed, and get the array variables from init_dataframe() args if
    not changed.
    """
    assert is_pd_index_type(index_typ) or isinstance(index_typ, MultiIndexType)

    n_cols = len(data_tup_typ.types)
    if n_cols == 0:
        column_names = ()
    else:
        # using 'get_const_tup_vals' since column names are generated using
        # 'gen_const_tup' which requires special handling for nested tuples
        column_names = get_const_tup_vals(col_names_typ)

    assert len(column_names) == n_cols

    def codegen(context, builder, signature, args):
        df_type = signature.return_type
        data_tup = args[0]
        index_val = args[1]
        columns_type = numba.typeof(column_names)

        # column names
        columns_tup = context.get_constant_generic(builder, columns_type, column_names)

        # set unboxed flags to 1 so that dtor decrefs all arrays
        one = context.get_constant(types.int8, 1)
        unboxed_tup = context.make_tuple(
            builder, types.UniTuple(types.int8, n_cols + 1), [one] * (n_cols + 1)
        )

        dataframe_val = construct_dataframe(
            context, builder, df_type, data_tup, index_val, columns_tup, unboxed_tup
        )

        # increase refcount of stored values
        context.nrt.incref(builder, data_tup_typ, data_tup)
        context.nrt.incref(builder, index_typ, index_val)
        context.nrt.incref(builder, columns_type, columns_tup)

        return dataframe_val

    ret_typ = DataFrameType(data_tup_typ.types, index_typ, column_names)
    sig = signature(ret_typ, data_tup_typ, index_typ, col_names_typ)
    return sig, codegen


@intrinsic
def has_parent(typingctx, df=None):
    def codegen(context, builder, sig, args):
        dataframe = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        return cgutils.is_not_null(builder, dataframe.parent)

    return signature(types.bool_, df), codegen


def get_dataframe_payload(context, builder, df_type, value):
    meminfo = cgutils.create_struct_proxy(df_type)(context, builder, value).meminfo
    payload_type = DataFramePayloadType(df_type)
    payload = context.nrt.meminfo_data(builder, meminfo)
    ptrty = context.get_data_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_data_helper(builder, payload_type, ref=payload)


@intrinsic
def _get_dataframe_unboxed(typingctx, df_typ=None):

    n_cols = len(df_typ.columns)
    ret_typ = types.UniTuple(types.int8, n_cols + 1)

    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(context, builder, ret_typ, dataframe_payload.unboxed)

    sig = signature(ret_typ, df_typ)
    return sig, codegen


@intrinsic
def _get_dataframe_data(typingctx, df_typ=None):

    ret_typ = types.Tuple(df_typ.data)

    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(context, builder, ret_typ, dataframe_payload.data)

    sig = signature(ret_typ, df_typ)
    return sig, codegen


@intrinsic
def _get_dataframe_index(typingctx, df_typ=None):
    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, df_typ.index, dataframe_payload.index
        )

    ret_typ = df_typ.index
    sig = signature(ret_typ, df_typ)
    return sig, codegen


# this function should be used for getting df._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_data(df, i):
    def _impl(df, i):  # pragma: no cover
        if has_parent(df) and _get_dataframe_unboxed(df)[i] == 0:
            bodo.hiframes.boxing.unbox_dataframe_column(df, i)
        return _get_dataframe_data(df)[i]

    return _impl


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_index(df):
    return lambda df: _get_dataframe_index(df)


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_dataframe_index", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func


def alias_ext_init_dataframe(lhs_name, args, alias_map, arg_aliases):
    assert len(args) == 3
    # add alias for data tuple
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)
    # add alias for index
    numba.core.ir_utils._add_alias(lhs_name, args[1].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("init_dataframe", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_init_dataframe


def init_dataframe_equiv(self, scope, equiv_set, loc, args, kws):
    """shape analysis for init_dataframe() calls. All input arrays have the same shape,
    which is the same as output dataframe's shape.
    """
    assert len(args) == 3 and not kws
    data_tup = args[0]
    # TODO: add shape for index (requires full shape support for indices)
    if equiv_set.has_shape(data_tup):
        data_shapes = equiv_set.get_shape(data_tup)
        # all data arrays have the same shape
        if len(data_shapes) > 1:
            equiv_set.insert_equiv(*data_shapes)
        if len(data_shapes) > 0:
            return (data_shapes[0], len(data_shapes)), []
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_init_dataframe = (
    init_dataframe_equiv
)


def get_dataframe_data_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for get_dataframe_data(). output array has the same shape as input
    dataframe.
    """
    assert len(args) == 2 and not kws
    var = args[0]
    if equiv_set.has_shape(var):
        return equiv_set.get_shape(var)[0], []
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_data = (
    get_dataframe_data_equiv
)


@intrinsic
def set_dataframe_data(typingctx, df_typ, c_ind_typ, arr_typ=None):
    col_ind = c_ind_typ.literal_value

    def codegen(context, builder, signature, args):
        # TODO: fix refcount
        df_arg, _, arr_arg = args
        dataframe_payload = get_dataframe_payload(context, builder, df_typ, df_arg)
        # assign array and set unboxed flag
        dataframe_payload.data = builder.insert_value(
            dataframe_payload.data, arr_arg, col_ind
        )
        dataframe_payload.unboxed = builder.insert_value(
            dataframe_payload.unboxed, context.get_constant(types.int8, 1), col_ind
        )

        context.nrt.incref(builder, arr_typ, arr_arg)

        # store payload
        dataframe = cgutils.create_struct_proxy(df_typ)(context, builder, value=df_arg)
        payload_type = DataFramePayloadType(df_typ)
        payload_ptr = context.nrt.meminfo_data(builder, dataframe.meminfo)
        ptrty = context.get_data_type(payload_type).as_pointer()
        payload_ptr = builder.bitcast(payload_ptr, ptrty)
        builder.store(dataframe_payload._getvalue(), payload_ptr)
        return impl_ret_borrowed(context, builder, df_typ, df_arg)

    sig = signature(df_typ, df_typ, c_ind_typ, arr_typ)
    return sig, codegen


@intrinsic
def set_df_index(typingctx, df_t, index_t=None):
    """used in very limited cases like distributed to_csv() to create a new
    dataframe with index
    """
    # TODO: make inplace when dfs are full objects

    def codegen(context, builder, signature, args):
        in_df_arg = args[0]
        index_val = args[1]
        df_typ = signature.args[0]
        in_df = cgutils.create_struct_proxy(df_typ)(context, builder, value=in_df_arg)
        in_df_payload = get_dataframe_payload(context, builder, df_typ, in_df_arg)

        dataframe = construct_dataframe(
            context,
            builder,
            signature.return_type,
            in_df_payload.data,
            index_val,
            in_df.columns,
            in_df_payload.unboxed,
            in_df.parent,
        )

        # increase refcount of stored values
        context.nrt.incref(builder, index_t, index_val)
        # TODO: refcount
        context.nrt.incref(builder, types.Tuple(df_t.data), in_df_payload.data)
        context.nrt.incref(
            builder, types.UniTuple(string_type, len(df_t.columns)), in_df.columns
        )

        return dataframe

    ret_typ = DataFrameType(df_t.data, index_t, df_t.columns)
    sig = signature(ret_typ, df_t, index_t)
    return sig, codegen


@intrinsic
def set_df_column_with_reflect(typingctx, df, cname, arr, inplace=None):
    """Set df column and reflect to parent Python object
    return a new df.
    """
    assert isinstance(inplace, bodo.utils.typing.BooleanLiteral)
    is_inplace = inplace.literal_value
    col_name = cname.literal_value
    n_cols = len(df.columns)
    new_n_cols = n_cols
    data_typs = df.data
    column_names = df.columns
    index_typ = df.index
    is_new_col = col_name not in df.columns
    col_ind = n_cols
    if is_new_col:
        data_typs += (arr,)
        column_names += (col_name,)
        new_n_cols += 1
    else:
        col_ind = df.columns.index(col_name)
        data_typs = tuple(
            (arr if i == col_ind else data_typs[i]) for i in range(n_cols)
        )

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg, _ = args

        in_dataframe_payload = get_dataframe_payload(context, builder, df, df_arg)
        in_dataframe = cgutils.create_struct_proxy(df)(context, builder, value=df_arg)

        data_arrs = [
            builder.extract_value(in_dataframe_payload.data, i)
            if i != col_ind
            else arr_arg
            for i in range(n_cols)
        ]
        if is_new_col:
            data_arrs.append(arr_arg)

        zero = context.get_constant(types.int8, 0)
        one = context.get_constant(types.int8, 1)
        unboxed_vals = [
            builder.extract_value(in_dataframe_payload.unboxed, i)
            if i != col_ind
            else one
            for i in range(n_cols)
        ]

        if is_new_col:
            unboxed_vals.append(one)  # for new data array
        unboxed_vals.append(zero)  # for index

        index_val = in_dataframe_payload.index

        data_tup = context.make_tuple(builder, types.Tuple(data_typs), data_arrs)

        # column names
        columns_type = numba.typeof(column_names)
        columns_tup = context.get_constant_generic(builder, columns_type, column_names)
        unboxed_tup = context.make_tuple(
            builder, types.UniTuple(types.int8, new_n_cols + 1), unboxed_vals
        )

        # TODO: refcount of parent?
        out_dataframe = construct_dataframe(
            context,
            builder,
            signature.return_type,
            data_tup,
            index_val,
            columns_tup,
            unboxed_tup,
            in_dataframe.parent,
        )

        # increase refcount of stored values
        context.nrt.incref(builder, index_typ, index_val)
        for var, typ in zip(data_arrs, data_typs):
            context.nrt.incref(builder, typ, var)
        context.nrt.incref(builder, columns_type, columns_tup)

        # TODO: test this
        # test_set_column_cond3 doesn't test it for some reason
        if is_inplace:
            # TODO: test refcount properly
            # old data arrays will be replaced so need a decref
            decref_df_data(context, builder, in_dataframe_payload, df)
            # store payload
            payload_type = DataFramePayloadType(df)
            payload_ptr = context.nrt.meminfo_data(builder, in_dataframe.meminfo)
            ptrty = context.get_data_type(payload_type).as_pointer()
            payload_ptr = builder.bitcast(payload_ptr, ptrty)
            out_dataframe_payload = get_dataframe_payload(
                context, builder, df, out_dataframe
            )
            builder.store(out_dataframe_payload._getvalue(), payload_ptr)

            # incref data again since there will be too references updated
            # TODO: incref only unboxed arrays to be safe?
            context.nrt.incref(builder, index_typ, index_val)
            for var, typ in zip(data_arrs, data_typs):
                context.nrt.incref(builder, typ, var)
            context.nrt.incref(builder, columns_type, columns_tup)

        # set column of parent
        # get boxed array
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()  # acquire GIL
        env_manager = context.get_env_manager(builder)

        context.nrt.incref(builder, arr, arr_arg)

        # call boxing for array data
        # TODO: check complex data types possible for Series for dataframes set column here
        c = numba.core.pythonapi._BoxContext(context, builder, pyapi, env_manager)
        py_arr = bodo.hiframes.boxing._box_series_data(arr.dtype, arr, arr_arg, c)

        # get column as string or int obj
        if isinstance(col_name, str):
            cstr = context.insert_const_string(builder.module, col_name)
            cstr_obj = pyapi.string_from_string(cstr)
        else:
            assert isinstance(col_name, int)
            cstr_obj = pyapi.long_from_longlong(
                context.get_constant(types.intp, col_name)
            )

        # set column array
        pyapi.object_setitem(in_dataframe.parent, cstr_obj, py_arr)

        pyapi.decref(py_arr)
        pyapi.decref(cstr_obj)

        pyapi.gil_release(gil_state)  # release GIL

        return out_dataframe

    ret_typ = DataFrameType(data_typs, index_typ, column_names, True)
    sig = signature(ret_typ, df, cname, arr, inplace)
    return sig, codegen


@lower_constant(DataFrameType)
def lower_constant_dataframe(context, builder, df_type, pyval):
    """embed constant DataFrame value but getting constant values for data arrays and
    Index.
    """
    n_cols = len(pyval.columns)
    data_tup = context.get_constant_generic(
        builder,
        types.Tuple(df_type.data),
        tuple(pyval.iloc[:, i].values for i in range(n_cols)),
    )
    index_val = context.get_constant_generic(builder, df_type.index, pyval.index)
    columns_tup = context.get_constant_generic(
        builder, numba.typeof(df_type.columns), df_type.columns
    )

    # set unboxed flags to 1 for all arrays
    one = context.get_constant(types.int8, 1)
    unboxed_tup = context.make_tuple(
        builder, types.UniTuple(types.int8, n_cols + 1), [one] * (n_cols + 1)
    )

    dataframe_val = construct_dataframe(
        context, builder, df_type, data_tup, index_val, columns_tup, unboxed_tup
    )

    return dataframe_val


@overload(pd.DataFrame, inline="always", no_unliteral=True)
def pd_dataframe_overload(data=None, index=None, columns=None, dtype=None, copy=False):
    # TODO: support other input combinations
    # TODO: error checking
    if not is_overload_constant_bool(copy):  # pragma: no cover
        raise BodoError("pd.DataFrame(): copy argument should be constant")

    copy = get_overload_const(copy)

    col_args, data_args, index_arg = _get_df_args(data, index, columns, dtype, copy)
    col_var = gen_const_tup(col_args)

    func_text = (
        "def _init_df(data=None, index=None, columns=None, dtype=None, copy=False):\n"
    )
    func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe({}, {}, {})\n".format(
        data_args, index_arg, col_var
    )
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    _init_df = loc_vars["_init_df"]
    return _init_df


def _get_df_args(data, index, columns, dtype, copy):
    """
    Check pd.DataFrame() arguments and return column and data arguments
    (as text) for init_dataframe().
    Also applies options and fixes input if necessary.
    """
    # dtype argument
    astype_str = ""
    if not is_overload_none(dtype):
        astype_str = ".astype(dtype)"

    index_is_none = is_overload_none(index)
    index_arg = "bodo.utils.conversion.convert_to_index(index)"

    # data is sentinel tuple (converted from dictionary)
    if isinstance(data, types.BaseTuple):
        # first element is sentinel
        if not data.types[0] == types.StringLiteral("__bodo_tup"):
            raise ValueError("pd.DataFrame tuple input data not supported yet")
        n_cols = (len(data.types) - 1) // 2
        data_keys = [t.literal_value for t in data.types[1 : n_cols + 1]]
        data_arrs = [
            "bodo.utils.conversion.coerce_to_array(data[{}], True){}".format(
                i, astype_str
            )
            for i in range(n_cols + 1, 2 * n_cols + 1)
        ]
        data_dict = dict(zip(data_keys, data_arrs))
        # if no index provided and there are Series inputs, get index from them
        # XXX cannot handle alignment of multiple Series
        if is_overload_none(index):
            for i, t in enumerate(data.types[n_cols + 1 :]):
                if isinstance(t, SeriesType):
                    index_arg = "bodo.hiframes.pd_series_ext.get_series_index(data[{}])".format(
                        n_cols + 1 + i
                    )
                    index_is_none = False
                    break
    # empty dataframe
    elif is_overload_none(data):
        data_dict = {}
    else:
        # ndarray case
        # checks for 2d and column args
        # TODO: error checking
        if not (isinstance(data, types.Array) and data.ndim == 2):  # pragma: no cover
            raise BodoError(
                "pd.DataFrame() supports constant dictionary and ndarray input"
            )
        if is_overload_none(columns):  # pragma: no cover
            raise BodoError(
                "pd.DataFrame() column argument is required when"
                "ndarray is passed as data"
            )
        if copy:
            astype_str += ".copy()"
        columns_consts = get_overload_const_list(columns)
        n_cols = len(columns_consts)
        data_arrs = ["data[:,{}]{}".format(i, astype_str) for i in range(n_cols)]
        data_dict = dict(zip(columns_consts, data_arrs))

    if is_overload_none(columns):
        col_names = data_dict.keys()
    else:
        col_names = get_overload_const_list(columns)

    df_len = _get_df_len_from_info(data_dict, col_names, index_is_none, index_arg)
    _fill_null_arrays(data_dict, col_names, df_len, dtype)

    # set default RangeIndex if index argument is None and data argument isn't Series
    if index_is_none:
        # empty df has object Index in Pandas which correponds to our StringIndex
        if is_overload_none(data):
            index_arg = "bodo.hiframes.pd_index_ext.init_string_index(bodo.libs.str_arr_ext.pre_alloc_string_array(0, 0))"
        else:
            index_arg = "bodo.hiframes.pd_index_ext.init_range_index(0, {}, 1, None)".format(
                df_len
            )

    data_args = "({},)".format(", ".join(data_dict[c] for c in col_names))
    if len(col_names) == 0:
        data_args = "()"

    return col_names, data_args, index_arg


def _get_df_len_from_info(data_dict, col_names, index_is_none, index_arg):
    """return generated text for length of dataframe, given the input info in the
    pd.DataFrame() call
    """
    df_len = "0"
    for c in col_names:
        if c in data_dict:
            df_len = "len({})".format(data_dict[c])
            break

    if df_len is None and not index_is_none:
        df_len = "len({})".format(index_arg)  # TODO: test

    return df_len


def _fill_null_arrays(data_dict, col_names, df_len, dtype):
    """Fills data_dict with Null arrays if there are columns that are not
    available in data_dict.
    """
    # no null array needed
    if all(c in data_dict for c in col_names):
        return

    # object array of NaNs if dtype not specified
    if is_overload_none(dtype):
        dtype = "bodo.string_type"
    else:
        dtype = "dtype"

    # array with NaNs
    null_arr = "bodo.libs.array_kernels.gen_na_array({}, {})".format(df_len, dtype)
    for c in col_names:
        if c not in data_dict:
            data_dict[c] = null_arr


@overload(len, no_unliteral=True)  # TODO: avoid lowering?
def df_len_overload(df):
    if not isinstance(df, DataFrameType):
        return

    if len(df.columns) == 0:  # empty df
        return lambda df: 0
    return lambda df: len(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, 0))


# dummy lowering for filter (TODO: use proper overload and avoid this)
@lower_builtin(operator.getitem, DataFrameType, types.Array(types.bool_, 1, "C"))
@lower_builtin(operator.getitem, DataFrameType, SeriesType)
def lower_getitem_filter_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return dataframe._getvalue()


# handle getitem for Tuples because sometimes df._data[i] in
# get_dataframe_data() doesn't translate to 'static_getitem' which causes
# Numba to fail. See TestDataFrame.test_unbox1, TODO: find root cause in Numba
# adapted from typing/builtins.py
@infer_global(operator.getitem)
class GetItemTuple(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        tup, idx = args
        if not isinstance(tup, types.BaseTuple) or not isinstance(
            idx, types.IntegerLiteral
        ):
            return
        idx_val = idx.literal_value
        if isinstance(idx_val, int):
            ret = tup.types[idx_val]
        elif isinstance(idx_val, slice):
            ret = types.BaseTuple.from_types(tup.types[idx_val])

        return signature(ret, *args)


# adapted from targets/tupleobj.py
@lower_builtin(operator.getitem, types.BaseTuple, types.IntegerLiteral)
@lower_builtin(operator.getitem, types.BaseTuple, types.SliceLiteral)
def getitem_tuple_lower(context, builder, sig, args):
    tupty, idx = sig.args
    idx = idx.literal_value
    tup, _ = args
    if isinstance(idx, int):
        if idx < 0:
            idx += len(tupty)
        if not 0 <= idx < len(tupty):
            raise IndexError("cannot index at %d in %s" % (idx, tupty))
        res = builder.extract_value(tup, idx)
    elif isinstance(idx, slice):
        items = cgutils.unpack_tuple(builder, tup)[idx]
        res = context.make_tuple(builder, sig.return_type, items)
    else:
        raise NotImplementedError("unexpected index %r for %s" % (idx, sig.args[0]))
    return impl_ret_borrowed(context, builder, sig.return_type, res)


def validate_unicity_output_column_names(
    suffix_x, suffix_y, left_keys, right_keys, left_columns, right_columns
):
    """Raise a BodoError if the column in output of the join operation collide """
    comm_keys = set(left_keys) & set(right_keys)
    comm_data = set(left_columns) & set(right_columns)
    add_suffix = comm_data - comm_keys
    other_left = set(left_columns) - comm_data
    other_right = set(right_columns) - comm_data

    NatureLR = {}

    def insertOutColumn(col_name):
        if col_name in NatureLR:
            raise BodoError(
                "join(): two columns happen to have the same name : {}".format(col_name)
            )
        NatureLR[col_name] = 0

    for eVar in comm_keys:
        insertOutColumn(eVar)

    for eVar in add_suffix:
        eVarX = eVar + suffix_x
        eVarY = eVar + suffix_y
        insertOutColumn(eVarX)
        insertOutColumn(eVarY)

    for eVar in other_left:
        insertOutColumn(eVar)

    for eVar in other_right:
        insertOutColumn(eVar)


@overload_method(DataFrameType, "merge", inline="always", no_unliteral=True)
@overload(pd.merge, inline="always", no_unliteral=True)
def merge_overload(
    left,
    right,
    how="inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    sort=False,
    suffixes=("_x", "_y"),
    copy=True,
    indicator=False,
    validate=None,
):
    validate_merge_spec(
        left,
        right,
        how,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        sort,
        suffixes,
        copy,
        indicator,
        validate,
    )

    how = get_overload_const_str(how)
    # NOTE: using sorted to avoid inconsistent ordering across processors
    comm_cols = tuple(sorted(set(left.columns) & set(right.columns)))

    if not is_overload_none(on):
        left_on = right_on = on

    if (
        is_overload_none(on)
        and is_overload_none(left_on)
        and is_overload_none(right_on)
        and is_overload_false(left_index)
        and is_overload_false(right_index)
    ):
        left_keys = comm_cols
        right_keys = comm_cols
    else:
        if is_overload_true(left_index):
            left_keys = ["$_bodo_index_"]
        else:
            left_keys = get_overload_const_list(left_on)
            # make sure all left_keys is a valid column in left
            validate_keys(left_keys, left.columns)
        if is_overload_true(right_index):
            right_keys = ["$_bodo_index_"]
        else:
            right_keys = get_overload_const_list(right_on)
            # make sure all right_keys is a valid column in right
            validate_keys(right_keys, right.columns)

    validate_keys_length(
        left_on, right_on, left_index, right_index, left_keys, right_keys
    )
    validate_keys_dtypes(
        left, right, left_on, right_on, left_index, right_index, left_keys, right_keys
    )

    # The suffixes
    if is_overload_constant_tuple(suffixes):
        suffixes_val = get_overload_const_tuple(suffixes)
    if is_overload_constant_list(suffixes):
        suffixes_val = list(get_overload_const_list(suffixes))

    suffix_x = suffixes_val[0]
    suffix_y = suffixes_val[1]
    validate_unicity_output_column_names(
        suffix_x, suffix_y, left_keys, right_keys, left.columns, right.columns
    )

    left_keys = gen_const_tup(left_keys)
    right_keys = gen_const_tup(right_keys)

    # generating code since typers can't find constants easily
    func_text = "def _impl(left, right, how='inner', on=None, left_on=None,\n"
    func_text += "    right_on=None, left_index=False, right_index=False, sort=False,\n"
    func_text += (
        "    suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):\n"
    )
    func_text += "  return bodo.hiframes.pd_dataframe_ext.join_dummy(left, right, {}, {}, '{}', '{}', '{}', False)\n".format(
        left_keys, right_keys, how, suffix_x, suffix_y
    )

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    _impl = loc_vars["_impl"]
    return _impl


def common_validate_merge_merge_asof_spec(
    name_func, left, right, on, left_on, right_on, left_index, right_index, suffixes
):
    """Validate checks that are common to merge and merge_asof"""
    # make sure left and right are dataframes
    if not isinstance(left, DataFrameType) or not isinstance(right, DataFrameType):
        raise BodoError(name_func + "() requires dataframe inputs")
    if (
        (not is_overload_none(on))
        and (not is_overload_constant_list(on))
        and (not is_overload_constant_str(on))
    ):
        raise_const_error(name_func + "(): 'on' must be of type str or str list")
    # make sure left_on is of type str or strlist
    if (
        (not is_overload_none(left_on))
        and (not is_overload_constant_list(left_on))
        and (not is_overload_constant_str(left_on))
    ):
        raise_const_error(name_func + "(): left_on must be of type str or str list")
    # make sure right_on is of type str or strlist
    if (
        (not is_overload_none(right_on))
        and (not is_overload_constant_list(right_on))
        and (not is_overload_constant_str(right_on))
    ):
        raise_const_error(name_func + "(): right_on must be of type str or str list")

    # make sure leftindex is of type bool
    ensure_constant_arg(name_func, "left_index", left_index, bool)
    ensure_constant_arg(name_func, "right_index", right_index, bool)

    # make sure suffixes is not passed in
    # make sure on is of type str or strlist
    if (not is_overload_constant_tuple(suffixes)) and (
        not is_overload_constant_list(suffixes)
    ):
        raise_const_error(
            name_func + "(): suffixes parameters should be ['_left', '_right']"
        )

    if is_overload_constant_tuple(suffixes):
        suffixes_val = get_overload_const_tuple(suffixes)
    if is_overload_constant_list(suffixes):
        suffixes_val = list(get_overload_const_list(suffixes))

    if len(suffixes_val) != 2:
        raise BodoError(name_func + "(): The number of suffixes should be exactly 2")

    comm_cols = tuple(set(left.columns) & set(right.columns))
    if not is_overload_none(on):
        # make sure two dataframes have common columns
        if len(comm_cols) == 0:
            raise_bodo_error(
                name_func + "(): No common columns to perform merge on. "
                "Merge options: left_on={lon}, right_on={ron}, "
                "left_index={lidx}, right_index={ridx}".format(
                    lon=is_overload_true(left_on),
                    ron=is_overload_true(right_on),
                    lidx=is_overload_true(left_index),
                    ridx=is_overload_true(right_index),
                )
            )
        # make sure "on" does not coexist with left_on or right_on
        if (not is_overload_none(left_on)) or (not is_overload_none(right_on)):
            raise BodoError(
                name_func + '(): Can only pass argument "on" OR "left_on" '
                'and "right_on", not a combination of both.'
            )

    # make sure right_on, right_index, left_on, left_index are speciefied properly
    if (
        (is_overload_true(left_index) or not is_overload_none(left_on))
        and is_overload_none(right_on)
        and not is_overload_true(right_index)
    ):
        raise BodoError(name_func + "(): Must pass right_on or right_index=True")
    if (
        (is_overload_true(right_index) or not is_overload_none(right_on))
        and is_overload_none(left_on)
        and not is_overload_true(left_index)
    ):
        raise BodoError(name_func + "(): Must pass left_on or left_index=True")


def validate_merge_spec(
    left,
    right,
    how,
    on,
    left_on,
    right_on,
    left_index,
    right_index,
    sort,
    suffixes,
    copy,
    indicator,
    validate,
):
    """validate arguments to merge()
    """
    common_validate_merge_merge_asof_spec(
        "merge", left, right, on, left_on, right_on, left_index, right_index, suffixes
    )

    unsupported_args = dict(
        sort=sort, copy=copy, indicator=indicator, validate=validate
    )
    merge_defaults = dict(sort=False, copy=True, indicator=False, validate=None)
    check_unsupported_args("merge", unsupported_args, merge_defaults)

    # make sure how is constant and one of ("left", "right", "outer", "inner")
    ensure_constant_values("merge", "how", how, ("left", "right", "outer", "inner"))


def validate_merge_asof_spec(
    left,
    right,
    on,
    left_on,
    right_on,
    left_index,
    right_index,
    by,
    left_by,
    right_by,
    suffixes,
    tolerance,
    allow_exact_matches,
    direction,
):
    """validate checks of the merge_asof() function"""
    common_validate_merge_merge_asof_spec(
        "merge_asof",
        left,
        right,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        suffixes,
    )
    if not is_overload_true(allow_exact_matches):
        raise BodoError(
            "merge_asof(): allow_exact_matches parameter only supports default value True"
        )
    # make sure validate is None
    if not is_overload_none(tolerance):
        raise BodoError(
            "merge_asof(): tolerance parameter only supports default value None"
        )
    if not is_overload_none(by):
        raise BodoError("merge_asof(): by parameter only supports default value None")
    if not is_overload_none(left_by):
        raise BodoError(
            "merge_asof(): left_by parameter only supports default value None"
        )
    if not is_overload_none(right_by):
        raise BodoError(
            "merge_asof(): right_by parameter only supports default value None"
        )
    if not is_overload_constant_str(direction):
        raise BodoError("merge_asof(): direction parameter should be of type str")
    else:
        direction = get_overload_const_str(direction)
        if direction != "backward":
            raise BodoError(
                "merge_asof(): direction parameter only supports default value 'backward'"
            )


def validate_merge_asof_keys_length(
    left_on, right_on, left_index, right_index, left_keys, right_keys
):
    # make sure right_keys and left_keys have the same size
    if (not is_overload_true(left_index)) and (not is_overload_true(right_index)):
        if len(right_keys) != len(left_keys):
            raise BodoError("merge(): len(right_on) must equal len(left_on)")
    if not is_overload_none(left_on) and is_overload_true(right_index):
        raise BodoError(
            "merge(): right_index = True and specifying left_on is not suppported yet."
        )
    if not is_overload_none(right_on) and is_overload_true(left_index):
        raise BodoError(
            "merge(): left_index = True and specifying right_on is not suppported yet."
        )


def validate_keys_length(
    left_on, right_on, left_index, right_index, left_keys, right_keys
):
    # make sure right_keys and left_keys have the same size
    if (not is_overload_true(left_index)) and (not is_overload_true(right_index)):
        if len(right_keys) != len(left_keys):
            raise BodoError("merge(): len(right_on) must equal len(left_on)")
    if is_overload_true(right_index):
        if len(left_keys) != 1:
            raise BodoError(
                "merge(): len(left_on) must equal the number "
                'of levels in the index of "right", which is 1'
            )
    if is_overload_true(left_index):
        if len(right_keys) != 1:
            raise BodoError(
                "merge(): len(right_on) must equal the number "
                'of levels in the index of "left", which is 1'
            )


def validate_keys_dtypes(
    left, right, left_on, right_on, left_index, right_index, left_keys, right_keys
):
    # make sure left keys and right keys have comparable dtypes

    typing_context = numba.core.registry.cpu_target.typing_context

    if is_overload_true(left_index) or is_overload_true(right_index):
        # cases where index is used in merging
        if is_overload_true(left_index) and is_overload_true(right_index):
            lk_type = left.index
            is_l_str = isinstance(lk_type, StringIndexType)
            rk_type = right.index
            is_r_str = isinstance(rk_type, StringIndexType)
        elif is_overload_true(left_index):
            lk_type = left.index
            is_l_str = isinstance(lk_type, StringIndexType)
            rk_type = right.data[right.columns.index(right_keys[0])]
            is_r_str = rk_type.dtype == string_type
        elif is_overload_true(right_index):
            lk_type = left.data[left.columns.index(left_keys[0])]
            is_l_str = lk_type.dtype == string_type
            rk_type = right.index
            is_r_str = isinstance(rk_type, StringIndexType)

        if is_l_str and is_r_str:
            return
        lk_type = lk_type.dtype
        rk_type = rk_type.dtype
        try:
            ret_dtype = typing_context.resolve_function_type(
                operator.eq, (lk_type, rk_type), {}
            )
        except:
            raise_bodo_error(
                "merge: You are trying to merge on {lk_dtype} and "
                "{rk_dtype} columns. If you wish to proceed "
                "you should use pd.concat".format(lk_dtype=lk_type, rk_dtype=rk_type)
            )
    else:  # cases where only columns are used in merge
        for lk, rk in zip(left_keys, right_keys):
            lk_type = left.data[left.columns.index(lk)].dtype
            lk_arr_type = left.data[left.columns.index(lk)]
            rk_type = right.data[right.columns.index(rk)].dtype
            rk_arr_type = right.data[right.columns.index(rk)]

            if lk_arr_type == rk_arr_type:
                continue

            msg = (
                "merge: You are trying to merge on column {lk} of {lk_dtype} and "
                "column {rk} of {rk_dtype}. If you wish to proceed "
                "you should use pd.concat"
            ).format(lk=lk, lk_dtype=lk_type, rk=rk, rk_dtype=rk_type)

            # Make sure non-string columns are not merged with string columns.
            # As of Numba 0.47, string comparison with non-string works and is always
            # False, so using type inference below doesn't work
            # TODO: check all incompatible key types similar to Pandas in
            # _maybe_coerce_merge_keys
            l_is_str = lk_type == string_type
            r_is_str = rk_type == string_type
            if l_is_str ^ r_is_str:
                raise_bodo_error(msg)

            try:
                ret_dtype = typing_context.resolve_function_type(
                    operator.eq, (lk_type, rk_type), {}
                )
            except:  # pragma: no cover
                # TODO: cover this case in unittests
                raise_bodo_error(msg)


def validate_keys(keys, columns):
    if len(set(keys).difference(set(columns))) > 0:
        raise_bodo_error(
            "merge(): invalid key {} for on/left_on/right_on".format(
                set(keys).difference(set(columns))
            )
        )


@overload_method(DataFrameType, "join", inline="always", no_unliteral=True)
def join_overload(left, other, on=None, how="left", lsuffix="", rsuffix="", sort=False):
    validate_join_spec(left, other, on, how, lsuffix, rsuffix, sort)

    how = get_overload_const_str(how)

    if not is_overload_none(on):
        left_keys = get_overload_const_list(on)
    else:
        left_keys = ["$_bodo_index_"]

    right_keys = ["$_bodo_index_"]

    left_keys = gen_const_tup(left_keys)
    right_keys = gen_const_tup(right_keys)

    # generating code since typers can't find constants easily
    func_text = "def _impl(left, other, on=None, how='left',\n"
    func_text += "    lsuffix='', rsuffix='', sort=False):\n"
    func_text += "  return bodo.hiframes.pd_dataframe_ext.join_dummy(left, other, {}, {}, '{}', '{}', '{}', True)\n".format(
        left_keys, right_keys, how, lsuffix, rsuffix
    )

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    _impl = loc_vars["_impl"]
    return _impl


def validate_join_spec(left, other, on, how, lsuffix, rsuffix, sort):
    # make sure left and other are dataframes
    if not isinstance(other, DataFrameType):
        raise BodoError("join() requires dataframe inputs")

    # make sure how is constant and one of ("left", "right", "outer", "inner")
    ensure_constant_values("merge", "how", how, ("left", "right", "outer", "inner"))

    # make sure on is of type str or strlist
    if (
        (not is_overload_none(on))
        and (not is_overload_constant_list(on))
        and (not is_overload_constant_str(on))
    ):
        raise_const_error("join(): 'on' must be of type str or str list")
    # make sure 'on' has length 1 since we don't support Multiindex
    if not is_overload_none(on) and len(get_overload_const_list(on)) != 1:
        raise BodoError("join(): len(on) must equals to 1 when specified.")
    # make sure 'on' is a valid column in other
    if not is_overload_none(on):
        on_keys = get_overload_const_list(on)
        validate_keys(on_keys, left.columns)
    # make sure sort is the default value, sort=True not supported
    if not is_overload_false(sort):
        raise BodoError("join(): sort parameter only supports default value False")

    comm_cols = tuple(set(left.columns) & set(other.columns))
    if len(comm_cols) > 0:
        # make sure two dataframes do not have common columns
        # because we are not supporting lsuffix and rsuffix
        raise BodoError(
            "join(): not supporting joining on overlapping columns:"
            "{cols} Use DataFrame.merge() instead.".format(cols=comm_cols)
        )


# a dummy join function that will be replace in dataframe_pass
def join_dummy(
    left_df, right_df, left_on, right_on, how, suffix_x, suffix_y, is_join
):  # pragma: no cover
    return left_df


@infer_global(join_dummy)
class JoinTyper(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_dataframe_ext import DataFrameType
        from bodo.utils.typing import is_overload_str

        assert not kws
        (
            left_df,
            right_df,
            left_on,
            right_on,
            how_var,
            suffix_x,
            suffix_y,
            is_join,
        ) = args

        left_on = get_overload_const_list(left_on)
        right_on = get_overload_const_list(right_on)

        # columns with common name that are not common keys will get a suffix
        comm_keys = set(left_on) & set(right_on)
        comm_data = set(left_df.columns) & set(right_df.columns)
        add_suffix = comm_data - comm_keys

        # Those two variables have the same values as the "left_index" in argument
        # to "merge" even if the index has a name.
        left_index = "$_bodo_index_" in left_on
        right_index = "$_bodo_index_" in right_on

        how = get_overload_const_str(how_var)
        is_left = how in {"left", "outer"}
        is_right = how in {"right", "outer"}
        columns = []
        data = []
        # In the case of merging on one index and a column we have to add another
        # column to the output. This is in the case of a column showing up also
        # on the other side.
        if left_index and not right_index and not is_join.literal_value:
            right_key = right_on[0]
            if right_key in left_df.columns:
                columns.append(right_key)
                data.append(right_df.data[right_df.columns.index(right_key)])
        if right_index and not left_index and not is_join.literal_value:
            left_key = left_on[0]
            if left_key in right_df.columns:
                columns.append(left_key)
                data.append(left_df.data[left_df.columns.index(left_key)])

        def map_data_type(in_type, need_nullable):
            if (
                isinstance(in_type, types.Array)
                and not is_dtype_nullable(in_type.dtype)
                and need_nullable
            ):
                return IntegerArrayType(in_type.dtype)
            else:
                return in_type

        # The left side. All of it got included.
        for in_type, col in zip(left_df.data, left_df.columns):
            columns.append(col + suffix_x.literal_value if col in add_suffix else col)
            if col in comm_keys:
                # For a common key we take either from left or right, so no additional NaN occurs.
                data.append(in_type)
            else:
                # For a key that is not common OR data column, we have to plan for a NaN column
                data.append(map_data_type(in_type, is_right))
        # The right side
        # common keys are added only once so avoid adding them
        for in_type, col in zip(right_df.data, right_df.columns):
            if col not in comm_keys:
                # a key column that is not common needs to plan for NaN.
                # Same for a data column of course.
                columns.append(
                    col + suffix_y.literal_value if col in add_suffix else col
                )
                data.append(map_data_type(in_type, is_left))
        # In the case of merging with left_index=True or right_index=True then
        # the index is coming from the other index. And so we need to set it adequately.
        index_typ = RangeIndexType(types.none)
        if left_index and right_index and not is_overload_str(how, "asof"):
            index_typ = left_df.index
            if isinstance(index_typ, bodo.hiframes.pd_index_ext.RangeIndexType):
                index_typ = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
        elif left_index and not right_index:
            index_typ = right_df.index
            if isinstance(index_typ, bodo.hiframes.pd_index_ext.RangeIndexType):
                index_typ = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
        elif right_index and not left_index:
            index_typ = left_df.index
            if isinstance(index_typ, bodo.hiframes.pd_index_ext.RangeIndexType):
                index_typ = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)

        out_df = DataFrameType(tuple(data), index_typ, tuple(columns))
        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(join_dummy, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return dataframe._getvalue()


@overload(pd.merge_asof, inline="always", no_unliteral=True)
def merge_asof_overload(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_index=False,
    right_index=False,
    by=None,
    left_by=None,
    right_by=None,
    suffixes=("_x", "_y"),
    tolerance=None,
    allow_exact_matches=True,
    direction="backward",
):

    validate_merge_asof_spec(
        left,
        right,
        on,
        left_on,
        right_on,
        left_index,
        right_index,
        by,
        left_by,
        right_by,
        suffixes,
        tolerance,
        allow_exact_matches,
        direction,
    )

    # TODO: support 'by' argument

    # XXX copied from merge, TODO: refactor
    # make sure left and right are dataframes
    if not isinstance(left, DataFrameType) or not isinstance(right, DataFrameType):
        raise TypeError("merge_asof() requires dataframe inputs")

    # NOTE: using sorted to avoid inconsistent ordering across processors
    comm_cols = tuple(sorted(set(left.columns) & set(right.columns)))

    if not is_overload_none(on):
        left_on = right_on = on

    if (
        is_overload_none(on)
        and is_overload_none(left_on)
        and is_overload_none(right_on)
        and is_overload_false(left_index)
        and is_overload_false(right_index)
    ):
        left_keys = comm_cols
        right_keys = comm_cols
    else:
        if is_overload_true(left_index):
            left_keys = ["$_bodo_index_"]
        else:
            left_keys = get_overload_const_list(left_on)
            validate_keys(left_keys, left.columns)
        if is_overload_true(right_index):
            right_keys = ["$_bodo_index_"]
        else:
            right_keys = get_overload_const_list(right_on)
            validate_keys(right_keys, right.columns)

    validate_merge_asof_keys_length(
        left_on, right_on, left_index, right_index, left_keys, right_keys
    )
    validate_keys_dtypes(
        left, right, left_on, right_on, left_index, right_index, left_keys, right_keys
    )
    left_keys = gen_const_tup(left_keys)
    right_keys = gen_const_tup(right_keys)
    # The suffixes
    if isinstance(suffixes, tuple):
        suffixes_val = suffixes
    if is_overload_constant_list(suffixes):
        suffixes_val = list(get_overload_const_list(suffixes))
    if isinstance(suffixes, types.Omitted):
        suffixes_val = suffixes.value

    suffix_x = suffixes_val[0]
    suffix_y = suffixes_val[1]

    # generating code since typers can't find constants easily
    func_text = "def _impl(left, right, on=None, left_on=None, right_on=None,\n"
    func_text += "    left_index=False, right_index=False, by=None, left_by=None,\n"
    func_text += "    right_by=None, suffixes=('_x', '_y'), tolerance=None,\n"
    func_text += "    allow_exact_matches=True, direction='backward'):\n"
    func_text += "  suffix_x = suffixes[0]\n"
    func_text += "  suffix_y = suffixes[1]\n"
    func_text += "  return bodo.hiframes.pd_dataframe_ext.join_dummy(left, right, {}, {}, 'asof', '{}', '{}', False)\n".format(
        left_keys, right_keys, suffix_x, suffix_y
    )

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    _impl = loc_vars["_impl"]
    return _impl


@overload_method(DataFrameType, "pivot_table", no_unliteral=True)
def pivot_table_overload(
    df,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name="All",
    _pivot_values=None,
):
    if aggfunc == "mean":

        def _impl(
            df,
            values=None,
            index=None,
            columns=None,
            aggfunc="mean",
            fill_value=None,
            margins=False,
            dropna=True,
            margins_name="All",
            _pivot_values=None,
        ):  # pragma: no cover

            return bodo.hiframes.pd_groupby_ext.pivot_table_dummy(
                df, values, index, columns, "mean", _pivot_values
            )

        return _impl

    def _impl(
        df,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
        _pivot_values=None,
    ):  # pragma: no cover

        return bodo.hiframes.pd_groupby_ext.pivot_table_dummy(
            df, values, index, columns, aggfunc, _pivot_values
        )

    return _impl


@overload(pd.crosstab, inline="always", no_unliteral=True)
def crosstab_overload(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins=False,
    margins_name="All",
    dropna=True,
    normalize=False,
    _pivot_values=None,
):
    # TODO: hanlde multiple keys (index args)
    # TODO: handle values and aggfunc options
    def _impl(
        index,
        columns,
        values=None,
        rownames=None,
        colnames=None,
        aggfunc=None,
        margins=False,
        margins_name="All",
        dropna=True,
        normalize=False,
        _pivot_values=None,
    ):  # pragma: no cover
        return bodo.hiframes.pd_groupby_ext.crosstab_dummy(
            index, columns, _pivot_values
        )

    return _impl


@overload(pd.concat, inline="always", no_unliteral=True)
def concat_overload(
    objs,
    axis=0,
    join="outer",
    join_axes=None,
    ignore_index=False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity=False,
    sort=None,
    copy=True,
):
    # TODO: handle options
    # TODO: support Index
    return lambda objs, axis=0, join="outer", join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=None, copy=True: bodo.hiframes.pd_dataframe_ext.concat_dummy(
        objs, axis
    )


def concat_dummy(objs):  # pragma: no cover
    return pd.concat(objs)


@infer_global(concat_dummy)
class ConcatDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        objs = args[0]
        axis = 0

        if isinstance(args[1], types.IntegerLiteral):
            axis = args[1].literal_value

        if isinstance(objs, types.List):
            assert axis == 0
            assert isinstance(objs.dtype, (SeriesType, DataFrameType))
            # TODO: support Index in append/concat
            ret_typ = objs.dtype.copy(index=RangeIndexType(types.none))
            if isinstance(ret_typ, DataFrameType):
                ret_typ = ret_typ.copy(
                    has_parent=False, index=RangeIndexType(types.none)
                )
            return signature(ret_typ, *args)

        if not isinstance(objs, types.BaseTuple):
            raise ValueError("Tuple argument for pd.concat expected")
        assert len(objs.types) > 0

        if axis == 1:
            data = []
            names = []
            col_no = 0
            for obj in objs.types:
                assert isinstance(obj, (SeriesType, DataFrameType))
                if isinstance(obj, SeriesType):
                    # TODO: handle names of SeriesTypes
                    data.append(obj.data)
                    names.append(str(col_no))
                    col_no += 1
                else:  # DataFrameType
                    # TODO: test
                    data.extend(obj.data)
                    names.extend(obj.columns)

            ret_typ = DataFrameType(
                tuple(data), RangeIndexType(types.none), tuple(names)
            )
            return signature(ret_typ, *args)

        assert axis == 0
        # dataframe case
        if isinstance(objs.types[0], DataFrameType):
            assert all(isinstance(t, DataFrameType) for t in objs.types)
            # get output column names
            all_colnames = []
            for df in objs.types:
                all_colnames.extend(df.columns)
            # TODO: verify how Pandas sorts column names
            # remove duplicates but keep original order
            all_colnames = list(dict.fromkeys(all_colnames).keys())

            # get output data types
            all_data = []
            for cname in all_colnames:
                # arguments to the generated function
                arr_args = [
                    df.data[df.columns.index(cname)]
                    for df in objs.types
                    if cname in df.columns
                ]
                # XXX we add arrays of float64 NaNs if an Integer column is missing
                # so add a dummy array of float64 for accurate typing
                # e.g. int to float conversion
                # TODO: use nullable integer array when pandas switches
                # TODO: fix NA column additions for other types
                if len(arr_args) < len(objs.types) and all(
                    isinstance(t.dtype, types.Integer) for t in arr_args
                ):
                    arr_args.append(types.Array(types.float64, 1, "C"))
                # use bodo.libs.array_kernels.concat() typer
                concat_typ = self.context.resolve_function_type(
                    bodo.libs.array_kernels.concat, (types.Tuple(arr_args),), {}
                ).return_type
                all_data.append(concat_typ)

            ret_typ = DataFrameType(
                tuple(all_data), RangeIndexType(types.none), tuple(all_colnames)
            )
            return signature(ret_typ, *args)

        # series case
        elif isinstance(objs.types[0], SeriesType):
            assert all(isinstance(t, SeriesType) for t in objs.types)
            arr_args = [S.data for S in objs.types]
            concat_typ = self.context.resolve_function_type(
                bodo.libs.array_kernels.concat, (types.Tuple(arr_args),), {}
            ).return_type
            ret_typ = SeriesType(concat_typ.dtype, concat_typ)
            return signature(ret_typ, *args)
        # TODO: handle other iterables like arrays, lists, ...


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(concat_dummy, types.VarArg(types.Any))
def lower_concat_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "sort_values", inline="always", no_unliteral=True)
def sort_values_overload(
    df,
    by,
    axis=0,
    ascending=True,
    inplace=False,
    kind="quicksort",
    na_position="last",
    _bodo_transformed=False,
):
    # df type can change if inplace is set (e.g. RangeIndex to Int64Index)
    handle_inplace_df_type_change(inplace, _bodo_transformed, "sort_values")

    validate_sort_values_spec(df, by, axis, ascending, inplace, kind, na_position)

    def _impl(
        df,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        _bodo_transformed=False,
    ):  # pragma: no cover

        return bodo.hiframes.pd_dataframe_ext.sort_values_dummy(
            df, by, ascending, inplace, na_position
        )

    return _impl


def validate_sort_values_spec(df, by, axis, ascending, inplace, kind, na_position):
    """validates sort_values spec
    Note that some checks are due to unsupported functionalities
    """

    # whether 'by' is supplied is checked by numba
    # make sure 'by' is a const str or str list
    if not is_overload_constant_str(by) and not is_overload_constant_list(by):
        raise_const_error(
            "sort_values(): 'by' parameter only supports "
            "a constant column label or column labels. by={}".format(by)
        )
    # make sure by has valid label(s)
    set_possible_keys = set(df.columns)
    if is_overload_constant_str(df.index.name_typ):
        set_possible_keys.add(get_overload_const_str(df.index.name_typ))
    if len(set(get_overload_const_list(by)).difference(set_possible_keys)) > 0:
        raise_bodo_error(
            "sort_values(): invalid key {} for by.".format(
                set_possible_keys.difference(set(get_overload_const_list(by)))
            )
        )

    # make sure axis has default value 0
    if not is_overload_zero(axis):
        raise BodoError(
            "sort_values(): 'axis' parameter only " "supports integer value 0."
        )

    # make sure 'ascending' is of type bool
    if not is_overload_bool(ascending) and not is_overload_bool_list(ascending):
        raise BodoError(
            "sort_values(): 'ascending' parameter must be of type bool or list of bool, "
            "not {}.".format(ascending)
        )

    # make sure 'inplace' is of type bool
    if not is_overload_bool(inplace):
        raise BodoError(
            "sort_values(): 'inplace' parameter must be of type bool, "
            "not {}.".format(inplace)
        )

    # make sure 'kind' is not specified
    if kind != "quicksort" and not isinstance(kind, types.Omitted):
        warnings.warn(
            BodoWarning(
                "sort_values(): specifying sorting algorithm "
                "is not supported in Bodo. Bodo uses stable sort."
            )
        )

    # make sure 'na_position' is correctly specified
    if not is_overload_constant_str(na_position):
        raise_const_error(
            "sort_values(): na_position parameter must be a literal constant of type str, not "
            "{na_position}".format(na_position=na_position)
        )

    na_position = get_overload_const_str(na_position)
    if na_position not in ["first", "last"]:
        raise BodoError("sort_values(): na_position should either be 'first' or 'last'")


def sort_values_dummy(df, by, ascending, inplace, na_position):  # pragma: no cover
    return df.sort_values(
        by, ascending=ascending, inplace=inplace, na_position=na_position
    )


@infer_global(sort_values_dummy)
class SortDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, by, ascending, inplace, na_position = args

        index = df.index
        if isinstance(index, bodo.hiframes.pd_index_ext.RangeIndexType):
            index = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
        ret_typ = df.copy(index=index, has_parent=False)
        return signature(ret_typ, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(sort_values_dummy, types.VarArg(types.Any))
def lower_sort_values_dummy(context, builder, sig, args):
    if sig.return_type == types.none:
        return

    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "sort_index", inline="always", no_unliteral=True)
def sort_index_overload(
    df,
    axis=0,
    level=None,
    ascending=True,
    inplace=False,
    kind="quicksort",
    na_position="last",
    sort_remaining=True,
    by=None,
):
    def _impl(
        df,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
        by=None,
    ):  # pragma: no cover

        return bodo.hiframes.pd_dataframe_ext.sort_values_dummy(
            df, "$_bodo_index_", ascending, inplace, na_position
        )

    return _impl


# dummy function to change the df type to have set_parent=True
# used in sort_values(inplace=True) hack
def set_parent_dummy(df):  # pragma: no cover
    return df


@infer_global(set_parent_dummy)
class ParentDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (df,) = args
        ret = DataFrameType(df.data, df.index, df.columns, True)
        return signature(ret, *args)


@lower_builtin(set_parent_dummy, types.VarArg(types.Any))
def lower_set_parent_dummy(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# TODO: jitoptions for overload_method and infer_global
# (no_cpython_wrapper to avoid error for iterator object)
@overload_method(DataFrameType, "itertuples", inline="always", no_unliteral=True)
def itertuples_overload(df, index=True, name="Pandas"):
    def _impl(df, index=True, name="Pandas"):  # pragma: no cover
        return bodo.hiframes.pd_dataframe_ext.itertuples_dummy(df)

    return _impl


def itertuples_dummy(df):  # pragma: no cover
    return df


@infer_global(itertuples_dummy)
class ItertuplesDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (df,) = args
        # XXX index handling, assuming implicit index
        assert "Index" not in df.columns
        columns = ("Index",) + df.columns
        arr_types = (types.Array(types.int64, 1, "C"),) + df.data
        iter_typ = bodo.hiframes.dataframe_impl.DataFrameTupleIterator(
            columns, arr_types
        )
        return signature(iter_typ, *args)


@lower_builtin(itertuples_dummy, types.VarArg(types.Any))
def lower_itertuples_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "fillna", inline="always", no_unliteral=True)
def fillna_overload(
    df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None
):
    # TODO: handle possible **kwargs options?

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent that has a string column (reflection)
    def _impl(
        df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None
    ):  # pragma: no cover
        return bodo.hiframes.pd_dataframe_ext.fillna_dummy(df, value, inplace)

    return _impl


def fillna_dummy(df, n):  # pragma: no cover
    return df


@infer_global(fillna_dummy)
class FillnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, value, inplace = args
        # inplace value
        if isinstance(inplace, bodo.utils.typing.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        if not inplace:
            # copy type to sethas_parent False, TODO: data always copied?
            out_df = DataFrameType(df.data, df.index, df.columns)
            return signature(out_df, *args)
        return signature(types.none, *args)


@lower_builtin(fillna_dummy, types.VarArg(types.Any))
def lower_fillna_dummy(context, builder, sig, args):
    if sig.return_type == types.none:
        return
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "reset_index", inline="always", no_unliteral=True)
def reset_index_overload(
    df,
    level=None,
    drop=False,
    inplace=False,
    col_level=0,
    col_fill="",
    _bodo_transformed=False,
):

    handle_inplace_df_type_change(inplace, _bodo_transformed, "reset_index")

    # make sure 'drop' is a constant bool
    if not is_overload_constant_bool(drop):
        raise BodoError(
            "reset_index(): 'drop' parameter should be a constant boolean value"
        )

    # make sure 'inplace' is a constant bool
    if not is_overload_constant_bool(inplace):
        raise BodoError(
            "reset_index(): 'inplace' parameter should be a constant boolean value"
        )

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(
        df,
        level=None,
        drop=False,
        inplace=False,
        col_level=0,
        col_fill="",
        _bodo_transformed=False,
    ):  # pragma: no cover
        return bodo.hiframes.pd_dataframe_ext.reset_index_dummy(df, drop, inplace)

    return _impl


def reset_index_dummy(df, n):  # pragma: no cover
    return df


@infer_global(reset_index_dummy)
class ResetIndexDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, drop, inplace = args
        # safe to just get const values here, since error checking is done in
        # reset_index overload
        drop = is_overload_true(drop)

        # default output index is simple integer index with no name
        # TODO: handle MultiIndex and `level` argument case
        index = RangeIndexType(types.none)
        data = df.data
        columns = df.columns
        if not drop:
            # pandas assigns "level_0" if "index" is already used as a column name
            # https://github.com/pandas-dev/pandas/blob/08b70d837dd017d49d2c18e02369a15272b662b2/pandas/core/frame.py#L4547
            default_name = "index" if "index" not in columns else "level_0"
            index_names = get_index_names(
                df.index, "DataFrame.reset_index()", default_name
            )
            columns = index_names + columns
            data = get_index_data_arr_types(df.index) + data

        out_df = DataFrameType(data, index, columns)
        return signature(out_df, *args)


@lower_builtin(reset_index_dummy, types.VarArg(types.Any))
def lower_reset_index_dummy(context, builder, sig, args):
    if sig.return_type is types.none:
        return context.get_dummy_value()
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "dropna", inline="always", no_unliteral=True)
def dropna_overload(df, axis=0, how="any", thresh=None, subset=None, inplace=False):

    # error-checking for inplace=True
    if not is_overload_constant_bool(inplace) or is_overload_true(inplace):
        raise BodoError("DataFrame.dropna(): inplace=True is not supported")

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(
        df, axis=0, how="any", thresh=None, subset=None, inplace=False
    ):  # pragma: no cover
        return bodo.hiframes.pd_dataframe_ext.dropna_dummy(df)

    return _impl


def dropna_dummy(df, n):  # pragma: no cover
    return df


@infer_global(dropna_dummy)
class DropnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        (df,) = args
        # copy type to set has_parent False
        index = df.index
        if isinstance(index, RangeIndexType):
            index = NumericIndexType(types.int64)
        out_df = DataFrameType(df.data, index, df.columns)
        return signature(out_df, *args)


@lower_builtin(dropna_dummy, types.VarArg(types.Any))
def lower_dropna_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "drop", inline="always", no_unliteral=True)
def drop_overload(
    df,
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
    _bodo_transformed=False,
):

    handle_inplace_df_type_change(inplace, _bodo_transformed, "drop")

    if not is_overload_constant_bool(inplace):  # pragma: no cover
        raise_bodo_error(
            "DataFrame.drop(): 'inplace' parameter should be a constant bool"
        )

    if not is_overload_none(labels):
        # make sure axis=1
        if (
            not is_overload_constant_int(axis) or get_overload_const_int(axis) != 1
        ):  # pragma: no cover
            raise_bodo_error("only axis=1 supported for df.drop()")
        # get 'labels' column list
        if is_overload_constant_str(labels):
            drop_cols = (get_overload_const_str(labels),)
        elif is_overload_constant_list(labels):  # pragma: no cover
            drop_cols = get_overload_const_list(labels)
        else:  # pragma: no cover
            raise_bodo_error(
                "constant list of columns expected for labels in df.drop()"
            )
    else:
        assert not is_overload_none(columns)
        # TODO: error checking
        if is_overload_constant_str(columns):  # pragma: no cover
            drop_cols = (get_overload_const_str(columns),)
        elif is_overload_constant_list(columns):
            drop_cols = get_overload_const_list(columns)
        else:  # pragma: no cover
            raise_bodo_error(
                "constant list of columns expected for labels in df.drop()"
            )

    # check drop columns to be in df schema
    for c in drop_cols:
        if c not in df.columns:
            raise_bodo_error(
                "DataFrame.drop(): column {} not in DataFrame columns {}".format(
                    c, df.columns
                )
            )

    inplace = is_overload_true(inplace)
    # TODO: inplace of df with parent (reflection)

    new_cols = tuple(c for c in df.columns if c not in drop_cols)
    data_args = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}){}".format(
            df.columns.index(c), ".copy()" if not inplace else ""
        )
        for c in new_cols
    )

    func_text = "def impl(df, labels=None, axis=0, index=None, columns=None,\n"
    func_text += (
        "     level=None, inplace=False, errors='raise', _bodo_transformed=False):\n"
    )
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"
    # return new df even for inplace case, since typing pass replaces input variable
    # using output of the call
    return bodo.hiframes.dataframe_impl._gen_init_df(
        func_text, new_cols, data_args, index
    )


def query_dummy(df, expr):  # pragma: no cover
    return df.eval(expr)


@infer_global(query_dummy)
class QueryDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(
            SeriesType(types.bool_, index=RangeIndexType(types.none)), *args
        )


@lower_builtin(query_dummy, types.VarArg(types.Any))
def lower_query_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


def val_isin_dummy(S, vals):  # pragma: no cover
    return S in vals


def val_notin_dummy(S, vals):  # pragma: no cover
    return S not in vals


@infer_global(val_isin_dummy)
@infer_global(val_notin_dummy)
class ValIsinTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        return signature(SeriesType(types.bool_, index=args[0].index), *args)


@lower_builtin(val_isin_dummy, types.VarArg(types.Any))
@lower_builtin(val_notin_dummy, types.VarArg(types.Any))
def lower_val_isin_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "append", inline="always", no_unliteral=True)
def append_overload(df, other, ignore_index=False, verify_integrity=False, sort=None):
    if isinstance(other, DataFrameType):
        return lambda df, other, ignore_index=False, verify_integrity=False, sort=None: pd.concat(
            (df, other)
        )

    # TODO: tuple case
    # TODO: non-homogenous build_list case
    if isinstance(other, types.List) and isinstance(other.dtype, DataFrameType):
        return lambda df, other, ignore_index=False, verify_integrity=False, sort=None: pd.concat(
            [df] + other
        )

    raise ValueError(
        "invalid df.append() input. Only dataframe and list" " of dataframes supported"
    )


def gen_pandas_parquet_metadata(
    df, write_non_range_index_to_metadata, write_rangeindex_to_metadata
):
    # returns dict with pandas dataframe metadata for parquet storage.
    # For more information, see:
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html#storing-pandas-dataframe-objects-in-apache-parquet-format

    pandas_metadata = {}

    pandas_metadata["columns"] = []

    for col_name, col_type in zip(df.columns, df.data):
        if isinstance(col_type, types.Array) or col_type == boolean_array:
            pandas_type = numpy_type = col_type.dtype.name
            if numpy_type.startswith("datetime"):
                pandas_type = "datetime"
        elif col_type == string_array_type:
            pandas_type = "unicode"
            numpy_type = "object"
        elif isinstance(col_type, DecimalArrayType):
            pandas_type = numpy_type = "object"
        elif isinstance(col_type, IntegerArrayType):
            dtype_name = col_type.dtype.name
            if dtype_name.startswith("int"):
                pandas_type = "Int" + dtype_name[3:]
            elif dtype_name.startswith("uint"):
                pandas_type = "UInt" + dtype_name[4:]
            else:  # pragma: no cover
                raise BodoError(
                    "to_parquet(): unknown dtype in nullable Integer column {} {}".format(
                        col_name, col_type
                    )
                )
            numpy_type = col_type.dtype.name
        elif col_type == datetime_date_array_type:
            pandas_type = "datetime"
            numpy_type = "object"
        else:  # pragma: no cover
            raise BodoError(
                "to_parquet(): unsupported column type for metadata generation : {} {}".format(
                    col_name, col_type
                )
            )

        col_metadata = {
            "name": col_name,
            "field_name": col_name,
            "pandas_type": pandas_type,
            "numpy_type": numpy_type,
            "metadata": None,
        }
        pandas_metadata["columns"].append(col_metadata)

    if write_non_range_index_to_metadata:
        # TODO multi-level
        if "none" in df.index.name:
            _idxname = "__index_level_0__"
            _colidxname = None
        else:
            _idxname = "%s"
            _colidxname = "%s"

        pandas_metadata["index_columns"] = [_idxname]

        # add index column metadata
        pandas_metadata["columns"].append(
            {
                "name": _colidxname,
                "field_name": _idxname,
                "pandas_type": df.index.pandas_type_name,
                "numpy_type": df.index.numpy_type_name,
                "metadata": None,
            }
        )
    elif write_rangeindex_to_metadata:
        pandas_metadata["index_columns"] = [
            {"kind": "range", "name": "%s", "start": "%d", "stop": "%d", "step": "%d"}
        ]
    else:
        pandas_metadata["index_columns"] = []

    pandas_metadata["pandas_version"] = pd.__version__

    return pandas_metadata


@overload_method(DataFrameType, "to_parquet", no_unliteral=True)
def to_parquet_overload(
    df,
    fname,
    engine="auto",
    compression="snappy",
    index=None,
    partition_cols=None,
    # TODO handle possible **kwargs options?
    _is_parallel=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    if not is_overload_none(partition_cols):
        raise BodoError(
            "to_parquet(): Bodo does not currently support partition_cols option"
        )

    if not is_overload_none(compression) and get_overload_const_str(
        compression
    ) not in {"snappy", "gzip", "brotli"}:
        raise BodoError(
            "to_parquet(): Unsupported compression: "
            + str(get_overload_const_str(compression))
        )

    if not is_overload_none(index) and not is_overload_constant_bool(index):
        raise BodoError("to_parquet(): index must be a constant bool or None")

    from bodo.io.parquet_pio import parquet_write_table_cpp

    # if index=False, we don't write index to the parquet file
    # if index=True we write index to the parquet file even if the index is trivial RangeIndex.
    # if index=None and sequential and RangeIndex:
    #    do not write index value, and write dict to metadata
    # if index=None and sequential and non-RangeIndex:
    #    write index to the parquet file and write non-dict to metadata
    # if index=None and parallel:
    #    write index to the parquet file and write non-dict to metadata regardless of index type
    is_range_index = isinstance(df.index, bodo.hiframes.pd_index_ext.RangeIndexType)
    write_non_rangeindex = (df.index is not None) and (
        is_overload_true(_is_parallel)
        or (not is_overload_true(_is_parallel) and not is_range_index)
    )

    # we write index to metadata always if index=True
    write_non_range_index_to_metadata = is_overload_true(index) or (
        is_overload_none(index)
        and (not is_range_index or is_overload_true(_is_parallel))
    )

    write_rangeindex_to_metadata = (
        is_overload_none(index)
        and is_range_index
        and not is_overload_true(_is_parallel)
    )

    # write pandas metadata for the parquet file
    pandas_metadata_str = json.dumps(
        gen_pandas_parquet_metadata(
            df, write_non_range_index_to_metadata, write_rangeindex_to_metadata
        )
    )
    if not is_overload_true(_is_parallel) and is_range_index:
        pandas_metadata_str = pandas_metadata_str.replace('"%d"', "%d")
        if df.index.name == "RangeIndexType(none)":
            # if the index name is None then we need to write just "null" to the metadata file
            # without quotation marks(null). But if a name is provided we need to
            # wrap the name with quotation mark to indicate it is a string
            pandas_metadata_str = pandas_metadata_str.replace('"%s"', "%s")

    # convert dataframe columns to array_info
    data_args = ", ".join(
        "array_to_info(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}))".format(
            i
        )
        for i in range(len(df.columns))
    )

    col_names_text = ", ".join('"{}"'.format(col_name) for col_name in df.columns)

    func_text = "def df_to_parquet(df, fname, engine='auto', compression='snappy', index=None, partition_cols=None, _is_parallel=False):\n"
    # put arrays in table_info
    func_text += "    info_list = [{}]\n".format(data_args)
    func_text += "    table = arr_info_list_to_table(info_list)\n"
    func_text += "    col_names = array_to_info(str_arr_from_sequence([{}]))\n".format(
        col_names_text
    )
    if is_overload_true(index) or (is_overload_none(index) and write_non_rangeindex):
        func_text += "    index_col = array_to_info(index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)))\n"
        write_index = True
    else:
        func_text += "    index_col = array_to_info(np.empty(0))\n"
        write_index = False
    func_text += '    metadata = """' + pandas_metadata_str + '"""\n'
    func_text += "    if compression is None:\n"
    func_text += "        compression = 'none'\n"
    func_text += "    if df.index.name is not None:\n"
    func_text += "        name_ptr = df.index.name\n"
    func_text += "    else:\n"
    func_text += "        name_ptr = 'null'\n"
    if write_rangeindex_to_metadata:
        func_text += "    parquet_write_table_cpp(unicode_to_char_ptr(fname),\n"
        func_text += "                            table, col_names, index_col,\n"
        func_text += "                            " + str(write_index) + ",\n"
        func_text += "                            unicode_to_char_ptr(metadata),\n"
        func_text += "                            unicode_to_char_ptr(compression),\n"
        func_text += "                            _is_parallel, 1, df.index.start,\n"
        func_text += "                            df.index.stop, df.index.step,\n"
        func_text += "                            unicode_to_char_ptr(name_ptr))\n"
    else:
        func_text += "    parquet_write_table_cpp(unicode_to_char_ptr(fname),\n"
        func_text += "                            table, col_names, index_col,\n"
        func_text += "                            " + str(write_index) + ",\n"
        func_text += "                            unicode_to_char_ptr(metadata),\n"
        func_text += "                            unicode_to_char_ptr(compression),\n"
        func_text += "                            _is_parallel, 0, 0, 0, 0,\n"
        func_text += "                            unicode_to_char_ptr(name_ptr))\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "bodo": bodo,
            "unicode_to_char_ptr": unicode_to_char_ptr,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "str_arr_from_sequence": str_arr_from_sequence,
            "parquet_write_table_cpp": parquet_write_table_cpp,
            "index_to_array": index_to_array,
        },
        loc_vars,
    )
    df_to_parquet = loc_vars["df_to_parquet"]
    return df_to_parquet


def to_sql_exception_guard(
    df,
    name,
    con,
    schema=None,
    if_exists="fail",
    index=True,
    index_label=None,
    chunksize=None,
    dtype=None,
    method=None,
):  # pragma: no cover
    """Call of to_sql and guard the exception and return it as string if error happens"""
    err_msg = "all_ok"
    try:
        df.to_sql(
            name, con, schema, if_exists, index, index_label, chunksize, dtype, method,
        )
    except ValueError as e:
        err_msg = e.args[0]
    return err_msg


@numba.njit
def to_sql_exception_guard_encaps(
    df,
    name,
    con,
    schema=None,
    if_exists="fail",
    index=True,
    index_label=None,
    chunksize=None,
    dtype=None,
    method=None,
):  # pragma: no cover
    with numba.objmode(out="unicode_type"):
        out = to_sql_exception_guard(
            df,
            name,
            con,
            schema,
            if_exists,
            index,
            index_label,
            chunksize,
            dtype,
            method,
        )
    return out


@overload_method(DataFrameType, "to_sql")
def to_sql_overload(
    df,
    name,
    con,
    schema=None,
    if_exists="fail",
    index=True,
    index_label=None,
    chunksize=None,
    dtype=None,
    method=None,
    # Additional entry
    _is_parallel=False,
):
    if not is_overload_none(chunksize):
        raise BodoError("to_sql(): chunksize option is not supported")

    def _impl(
        df,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
        _is_parallel=False,
    ):  # pragma: no cover
        """Nodes number 0 does the first initial insertion into the database.
        Following nodes do the insertion of the rest if no error happened.
        The bcast_scalar is used to synchronize the process between 0 and the rest.
        """
        rank = bodo.libs.distributed_api.get_rank()
        err_msg = "unset"
        if rank != 0:
            if_exists = "append"  # For other nodes, we append to the existing data set.
            err_msg = bcast_scalar(err_msg)
        # The writing of the data.
        if rank == 0 or (_is_parallel and err_msg == "all_ok"):
            err_msg = to_sql_exception_guard_encaps(
                df,
                name,
                con,
                schema,
                if_exists,
                index,
                index_label,
                chunksize,
                dtype,
                method,
            )
        if rank == 0:
            err_msg = bcast_scalar(err_msg)
        if err_msg != "all_ok":
            # TODO: We cannot do a simple raise ValueError(err_msg).
            print("err_msg=", err_msg)
            raise ValueError("error in to_sql() operation")

    return _impl


dummy_use = numba.njit(lambda a: None)


# TODO: other Pandas versions (0.24 defaults are different than 0.23)
@overload_method(DataFrameType, "to_csv", no_unliteral=True)
def to_csv_overload(
    df,
    path_or_buf=None,
    sep=",",
    na_rep="",
    float_format=None,
    columns=None,
    header=True,
    index=True,
    index_label=None,
    mode="w",
    encoding=None,
    compression="infer",
    quoting=None,
    quotechar='"',
    line_terminator=None,
    chunksize=None,
    date_format=None,
    doublequote=True,
    escapechar=None,
    decimal=".",
):
    # TODO: refactor when objmode() can understand global string constant
    # String output case
    if path_or_buf is None or path_or_buf == types.none:

        def _impl(
            df,
            path_or_buf=None,
            sep=",",
            na_rep="",
            float_format=None,
            columns=None,
            header=True,
            index=True,
            index_label=None,
            mode="w",
            encoding=None,
            compression="infer",
            quoting=None,
            quotechar='"',
            line_terminator=None,
            chunksize=None,
            date_format=None,
            doublequote=True,
            escapechar=None,
            decimal=".",
        ):  # pragma: no cover
            with numba.objmode(D="unicode_type"):
                D = df.to_csv(
                    path_or_buf,
                    sep,
                    na_rep,
                    float_format,
                    columns,
                    header,
                    index,
                    index_label,
                    mode,
                    encoding,
                    compression,
                    quoting,
                    quotechar,
                    line_terminator,
                    chunksize,
                    date_format,
                    doublequote,
                    escapechar,
                    decimal,
                )
            return D

        return _impl

    def _impl(
        df,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression="infer",
        quoting=None,
        quotechar='"',
        line_terminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
    ):  # pragma: no cover
        # passing None for the first argument returns a string
        # containing contents to write to csv
        with numba.objmode(D="unicode_type"):
            D = df.to_csv(
                None,
                sep,
                na_rep,
                float_format,
                columns,
                header,
                index,
                index_label,
                mode,
                encoding,
                compression,
                quoting,
                quotechar,
                line_terminator,
                chunksize,
                date_format,
                doublequote,
                escapechar,
                decimal,
            )
        _csv_write(unicode_to_char_ptr(path_or_buf), D._data, 0, len(D), False)
        # ensure path_or_buf and D are not deleted before call to _csv_write completes
        dummy_use(path_or_buf)
        dummy_use(D)

    return _impl


@overload_method(DataFrameType, "to_json", no_unliteral=True)
def to_json_overload(
    df,
    path_or_buf=None,
    orient="columns",
    date_format=None,
    double_precision=10,
    force_ascii=True,
    date_unit="ms",
    default_handler=None,
    lines=False,
    compression="infer",
    index=True,
    indent=None,
):
    # TODO: refactor when objmode() can understand global string constant
    # String output case
    if path_or_buf is None or path_or_buf == types.none:

        def _impl(
            df,
            path_or_buf=None,
            orient="columns",
            date_format=None,
            double_precision=10,
            force_ascii=True,
            date_unit="ms",
            default_handler=None,
            lines=False,
            compression="infer",
            index=True,
            indent=None,
        ):  # pragma: no cover
            with numba.objmode(D="unicode_type"):
                D = df.to_json(
                    path_or_buf,
                    orient,
                    date_format,
                    double_precision,
                    force_ascii,
                    date_unit,
                    default_handler,
                    lines,
                    compression,
                    index,
                    indent,
                )
            return D

        return _impl

    def _impl(
        df,
        path_or_buf=None,
        orient="columns",
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=False,
        compression="infer",
        index=True,
        indent=None,
    ):  # pragma: no cover
        # passing None for the first argument returns a string
        # containing contents to write to json
        with numba.objmode(D="unicode_type"):
            D = df.to_json(
                None,
                orient,
                date_format,
                double_precision,
                force_ascii,
                date_unit,
                default_handler,
                lines,
                compression,
                index,
                indent,
            )
        if lines and orient == "records":
            _json_write(
                unicode_to_char_ptr(path_or_buf), D._data, 0, len(D), False, True
            )
        else:
            _json_write(
                unicode_to_char_ptr(path_or_buf), D._data, 0, len(D), False, False
            )
        # ensure path_or_buf and D are not deleted before call to _json_write completes
        dummy_use(path_or_buf)
        dummy_use(D)

    return _impl


def handle_inplace_df_type_change(inplace, _bodo_transformed, func_name):
    """df type can change for functions like drop, rename, etc. if inplace is set, so
    variable replacement in typing pass is necessary for type stability.
    This returns control to typing pass to handle it using a normal exception.
    typing pass sets _bodo_transformed if variable replacement is done already
    """
    if (
        is_overload_false(_bodo_transformed)
        and bodo.transforms.typing_pass.in_partial_typing
        and (is_overload_true(inplace) or not is_overload_constant_bool(inplace))
    ):
        bodo.transforms.typing_pass.typing_transform_required = True
        raise Exception(
            "DataFrame.{}(): transform necessary for inplace".format(func_name)
        )


# Throw BodoError for top-level unsupported functions in Pandas
pd_unsupported = (
    # Input/output
    pd.read_pickle,
    pd.read_table,
    pd.read_fwf,
    pd.read_clipboard,
    pd.ExcelWriter,
    pd.json_normalize,
    pd.read_html,
    pd.read_hdf,
    pd.read_feather,
    pd.read_orc,  # TODO: support
    pd.read_sas,
    pd.read_spss,
    pd.read_sql_table,
    pd.read_sql_query,
    pd.read_gbq,
    pd.read_stata,
    # General functions
    ## Data manipulations
    pd.melt,
    pd.pivot,
    pd.cut,
    pd.qcut,
    pd.merge_ordered,
    pd.get_dummies,
    pd.factorize,
    pd.wide_to_long,
    ## Top-level dealing with datetimelike
    pd.bdate_range,
    pd.period_range,
    pd.timedelta_range,
    pd.infer_freq,
    ## Top-level dealing with intervals
    pd.interval_range,
    ## Top-level evaluation
    pd.eval,
    ## Hashing
    pd.util.hash_array,
    pd.util.hash_pandas_object,
    # Testing
    pd.test,
)


def _install_pd_unsupported():
    """install an overload that raises BodoError for unsupported functions
    """
    for f in pd_unsupported:
        fname = "pd." + f.__name__
        overload(f)(create_unsupported_overload(fname))


_install_pd_unsupported()
