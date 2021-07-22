# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implement pd.DataFrame typing and data model handling.
"""
import json
import operator

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
    infer_global,
    signature,
)
from numba.extending import (
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.pd_index_ext import (
    HeterogeneousIndexType,
    NumericIndexType,
    RangeIndexType,
    is_pd_index_type,
)
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.series_indexing import SeriesIlocType
from bodo.io import json_cpp
from bodo.libs.array import arr_info_list_to_table, array_to_info
from bodo.libs.array_item_arr_ext import ArrayItemArrayType
from bodo.libs.binary_arr_ext import binary_array_type
from bodo.libs.bool_arr_ext import boolean_array
from bodo.libs.decimal_arr_ext import DecimalArrayType
from bodo.libs.distributed_api import bcast_scalar
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import str_arr_from_sequence, string_array_type
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.libs.struct_arr_ext import StructArrayType
from bodo.utils.conversion import index_to_array
from bodo.utils.transform import (
    gen_const_tup,
    get_const_func_output_type,
    get_const_tup_vals,
)
from bodo.utils.typing import (
    BodoError,
    check_unsupported_args,
    create_unsupported_overload,
    dtype_to_array_type,
    get_index_data_arr_types,
    get_literal_value,
    get_overload_const,
    get_overload_const_bool,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_udf_error_msg,
    get_udf_out_arr_type,
    is_heterogeneous_tuple_type,
    is_iterable_type,
    is_literal_type,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_str,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    is_tuple_like_type,
    raise_bodo_error,
    to_nullable_type,
)

_json_write = types.ExternalFunction(
    "json_write",
    types.void(
        types.voidptr,
        types.voidptr,
        types.int64,
        types.int64,
        types.bool_,
        types.bool_,
        types.voidptr,
    ),
)
ll.add_symbol("json_write", json_cpp.json_write)


class DataFrameType(types.ArrayCompatible):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects."""

    ndim = 2

    def __init__(self, data=None, index=None, columns=None):
        # data is tuple of Array types (not Series) or tuples (for df.describe)
        # index is Index obj (not Array type)
        # columns is a tuple of column names (strings, ints, or tuples in case of
        # MultiIndex)

        self.data = data
        if index is None:
            index = RangeIndexType(types.none)
        self.index = index
        self.columns = columns
        super(DataFrameType, self).__init__(
            name="dataframe({}, {}, {})".format(data, index, columns)
        )

    def copy(self, index=None):
        # XXX is copy necessary?
        if index is None:
            index = self.index.copy()
        data = tuple(a.copy() for a in self.data)
        return DataFrameType(data, index, self.columns)

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 2, "C")

    @property
    def key(self):
        # needed?
        return self.data, self.index, self.columns

    def unify(self, typingctx, other):
        """unifies two possible dataframe types into a single type
        see test_dataframe.py::test_df_type_unify_error
        """
        if (
            isinstance(other, DataFrameType)
            and len(other.data) == len(self.data)
            and other.columns == self.columns
        ):
            new_index = self.index.unify(typingctx, other.index)
            data = tuple(
                a.unify(typingctx, b) if a != b else a
                for a, b in zip(self.data, other.data)
            )
            # NOTE: unification is an extreme corner case probably, since arrays can
            # be unified only if just their layout or alignment is different.
            # That doesn't happen in df case since all arrays are 1D and C layout.
            # see: https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/types/npytypes.py#L436
            if new_index is not None and None not in data:  # pragma: no cover
                return DataFrameType(data, new_index, self.columns)

        # convert empty dataframe to any other dataframe to support important common
        # cases (see test_append_empty_df), even though it's not fully accurate.
        # TODO: detect and handle wrong corner cases (or raise warning) in compiler
        # passes
        if isinstance(other, DataFrameType) and len(self.data) == 0:
            return other

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
            ("meminfo", types.MemInfoPointer(payload_type)),
            # for boxed DataFrames, enables updating original DataFrame object
            ("parent", types.pyobject),
        ]
        super(DataFrameModel, self).__init__(dmm, fe_type, members)


@infer_getattr
class DataFrameAttribute(AttributeTemplate):
    key = DataFrameType

    @bound_function("df.pipe", no_unliteral=True)
    def resolve_pipe(self, df, args, kws):
        return bodo.hiframes.pd_groupby_ext.resolve_obj_pipe(
            self, df, args, kws, "DataFrame"
        )

    @bound_function("df.apply", no_unliteral=True)
    def resolve_apply(self, df, args, kws):
        kws = dict(kws)
        # pop apply() arguments from kws so only UDF kws remain
        func = args[0] if len(args) > 0 else kws.pop("func", None)
        axis = args[1] if len(args) > 1 else kws.pop("axis", types.literal(0))
        raw = args[2] if len(args) > 2 else kws.pop("raw", types.literal(False))
        result_type = args[3] if len(args) > 3 else kws.pop("result_type", types.none)
        f_args = args[4] if len(args) > 4 else kws.pop("args", types.Tuple([]))

        # check axis
        if not (is_overload_constant_int(axis) and get_overload_const_int(axis) == 1):
            raise BodoError("only Dataframe.apply() with axis=1 supported")

        unsupported_args = dict(raw=raw, result_type=result_type)
        merge_defaults = dict(raw=False, result_type=None)
        check_unsupported_args("Dataframe.apply", unsupported_args, merge_defaults)

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

        # each row is passed as a Series to UDF
        # TODO: pass df_index[i] as row name (after issue with RangeIndex getitem in
        # test_df_apply_assertion is resolved)
        # # name of the Series is the dataframe index value of the row
        # name_type = self.context.resolve_function_type(
        #     operator.getitem, (df.index, types.int64), {}
        # ).return_type
        name_type = types.none
        # the Index has constant column name values
        index_type = HeterogeneousIndexType(
            types.BaseTuple.from_types(tuple(types.literal(c) for c in df.columns)),
            None,
        )
        data_type = types.BaseTuple.from_types(dtypes)
        name_dtype = df.index.dtype
        if name_dtype == types.NPDatetime("ns"):
            name_dtype = bodo.pd_timestamp_type
        if name_dtype == types.NPTimedelta("ns"):
            name_dtype = bodo.pd_timedelta_type
        if is_heterogeneous_tuple_type(data_type):
            row_typ = HeterogeneousSeriesType(data_type, index_type, name_dtype)
        else:
            row_typ = SeriesType(data_type.dtype, data_type, index_type, name_dtype)
        arg_typs = (row_typ,)
        if f_args is not None:
            arg_typs += tuple(f_args.types)
        try:
            f_return_type = get_const_func_output_type(
                func, arg_typs, kws, self.context
            )
        except Exception as e:
            raise_bodo_error(get_udf_error_msg("DataFrame.apply()", e))

        if (
            isinstance(f_return_type, (SeriesType, HeterogeneousSeriesType))
            and f_return_type.const_info is None
        ):
            raise BodoError(
                "Invalid Series output in UDF (Series with constant length and constant Index value expected)"
            )

        # output is dataframe if UDF returns a Series
        if isinstance(f_return_type, HeterogeneousSeriesType):
            # NOTE: get_const_func_output_type() adds const_info attribute for Series
            # output
            _, index_vals = f_return_type.const_info
            arrs = tuple(dtype_to_array_type(t) for t in f_return_type.data.types)
            ret_type = DataFrameType(arrs, df.index, index_vals)
        elif isinstance(f_return_type, SeriesType):
            n_cols, index_vals = f_return_type.const_info
            arrs = tuple(
                dtype_to_array_type(f_return_type.dtype) for _ in range(n_cols)
            )
            ret_type = DataFrameType(arrs, df.index, index_vals)
        else:
            data_arr = get_udf_out_arr_type(f_return_type)
            ret_type = SeriesType(data_arr.dtype, data_arr, df.index, None)

        # add dummy default value for UDF kws to avoid errors
        kw_names = ", ".join("{} = ''".format(a) for a in kws.keys())
        func_text = f"def apply_stub(func, axis=0, raw=False, result_type=None, args=(), {kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        apply_stub = loc_vars["apply_stub"]

        pysig = numba.core.utils.pysignature(apply_stub)
        new_args = (func, axis, raw, result_type, f_args) + tuple(kws.values())
        return signature(ret_type, *new_args).replace(pysig=pysig)

    def generic_resolve(self, df, attr):
        # column selection
        if attr in df.columns:
            ind = df.columns.index(attr)
            arr_typ = df.data[ind]
            return SeriesType(
                arr_typ.dtype, arr_typ, df.index, types.StringLiteral(attr)
            )

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


# workaround to support row["A"] case in df.apply()
# implements getitem for namedtuples if generated by Bodo
@overload(operator.getitem, no_unliteral=True)
def namedtuple_getitem_overload(tup, idx):
    if isinstance(tup, types.BaseNamedTuple) and is_overload_constant_str(idx):
        field_idx = get_overload_const_str(idx)
        val_ind = tup.instance_class._fields.index(field_idx)
        return lambda tup, idx: tup[val_ind]  # pragma: no cover


def decref_df_data(context, builder, payload, df_type):
    """call decref() on all data arrays and index of dataframe"""
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
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload_ptr = builder.bitcast(base_ptr, ptrty)
    payload = context.make_helper(builder, payload_type, ref=payload_ptr)

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
    context, builder, df_type, data_tup, index_val, unboxed_tup, parent=None
):

    # create payload struct and store values
    payload_type = DataFramePayloadType(df_type)
    dataframe_payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    dataframe_payload.data = data_tup
    dataframe_payload.index = index_val
    dataframe_payload.unboxed = unboxed_tup

    # create meminfo and store payload
    payload_ll_type = context.get_value_type(payload_type)
    payload_size = context.get_abi_sizeof(payload_ll_type)
    dtor_fn = define_df_dtor(context, builder, df_type, payload_type)
    meminfo = context.nrt.meminfo_alloc_dtor(
        builder, context.get_constant(types.uintp, payload_size), dtor_fn
    )
    meminfo_void_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_void_ptr, payload_ll_type.as_pointer())

    # create dataframe struct
    dataframe = cgutils.create_struct_proxy(df_type)(context, builder)
    dataframe.meminfo = meminfo
    if parent is None:
        # Set parent to NULL
        dataframe.parent = cgutils.get_null_value(dataframe.parent.type)
    else:
        dataframe.parent = parent
        dataframe_payload.parent = parent
        # incref parent dataframe object if not null (not fully known until runtime)
        has_parent = cgutils.is_not_null(builder, parent)
        with builder.if_then(has_parent):
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
    assert is_pd_index_type(index_typ) or isinstance(
        index_typ, MultiIndexType
    ), "init_dataframe(): invalid index type"

    n_cols = len(data_tup_typ.types)
    if n_cols == 0:
        column_names = ()
    else:
        # using 'get_const_tup_vals' since column names are generated using
        # 'gen_const_tup' which requires special handling for nested tuples
        column_names = get_const_tup_vals(col_names_typ)

    assert (
        len(column_names) == n_cols
    ), "init_dataframe(): number of column names does not match number of columns"

    def codegen(context, builder, signature, args):
        df_type = signature.return_type
        data_tup = args[0]
        index_val = args[1]

        # set unboxed flags to 1 so that dtor decrefs all arrays
        one = context.get_constant(types.int8, 1)
        unboxed_tup = context.make_tuple(
            builder, types.UniTuple(types.int8, n_cols + 1), [one] * (n_cols + 1)
        )

        dataframe_val = construct_dataframe(
            context, builder, df_type, data_tup, index_val, unboxed_tup
        )
        # increase refcount of stored values
        context.nrt.incref(builder, data_tup_typ, data_tup)
        context.nrt.incref(builder, index_typ, index_val)
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
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_helper(builder, payload_type, ref=payload)


@intrinsic
def _get_dataframe_unboxed(typingctx, df_typ=None):

    n_cols = len(df_typ.columns)
    ret_typ = types.UniTuple(types.int8, n_cols + 1)
    sig = signature(ret_typ, df_typ)

    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, signature.return_type, dataframe_payload.unboxed
        )

    return sig, codegen


@intrinsic
def _get_dataframe_data(typingctx, df_typ=None):
    ret_typ = types.Tuple(df_typ.data)
    sig = signature(ret_typ, df_typ)

    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, signature.return_type, dataframe_payload.data
        )

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


def get_dataframe_data(df, i):  # pragma: no cover
    return df[i]


@infer_global(get_dataframe_data)
class GetDataFrameDataInfer(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        assert is_overload_constant_int(args[1])
        df = args[0]
        i = get_overload_const_int(args[1])
        ret = df.data[i]
        return ret(*args)


def get_dataframe_data_impl(df, i):
    def _impl(df, i):  # pragma: no cover
        if has_parent(df) and _get_dataframe_unboxed(df)[i] == 0:
            bodo.hiframes.boxing.unbox_dataframe_column(df, i)
        return _get_dataframe_data(df)[i]

    return _impl


@lower_builtin(get_dataframe_data, DataFrameType, types.IntegerLiteral)
def lower_get_dataframe_data(context, builder, sig, args):
    impl = get_dataframe_data_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_index(df):
    return lambda df: _get_dataframe_index(df)  # pragma: no cover


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
    from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

    assert len(args) == 3 and not kws
    data_tup = args[0]
    index = args[1]

    # avoid returning shape for tuple data (can result in Numba errors)
    data_type = self.typemap[data_tup.name]
    if any(is_tuple_like_type(t) for t in data_type.types):
        return None

    if equiv_set.has_shape(data_tup):
        data_shapes = equiv_set.get_shape(data_tup)
        # all data arrays have the same shape
        if len(data_shapes) > 1:
            equiv_set.insert_equiv(*data_shapes)
        if len(data_shapes) > 0:
            # index and data have the same length (avoid tuple index)
            index_type = self.typemap[index.name]
            if not isinstance(
                index_type, HeterogeneousIndexType
            ) and equiv_set.has_shape(index):
                equiv_set.insert_equiv(data_shapes[0], index)
            return ArrayAnalysis.AnalyzeResult(
                shape=(data_shapes[0], len(data_shapes)), pre=[]
            )
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

    # avoid returning shape for tuple data (can result in Numba errors)
    data_types = self.typemap[var.name].data
    if any(is_tuple_like_type(t) for t in data_types):
        return None

    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(var)[0], pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_data = (
    get_dataframe_data_equiv
)


def get_dataframe_index_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for get_dataframe_index(). output Index has the same length as
    input dataframe.
    """
    from bodo.hiframes.pd_index_ext import HeterogeneousIndexType

    assert len(args) == 1 and not kws
    var = args[0]

    # avoid returning shape for tuple data (can result in Numba errors)
    index_type = self.typemap[var.name].index
    if isinstance(index_type, HeterogeneousIndexType):
        return None

    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(var)[0], pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_index = (
    get_dataframe_index_equiv
)


@intrinsic
def set_dataframe_data(typingctx, df_typ, c_ind_typ, arr_typ=None):
    """set column data of a dataframe inplace"""
    assert is_overload_constant_int(c_ind_typ)
    col_ind = get_overload_const_int(c_ind_typ)

    # make sure dataframe column data type is not changed (avoids lowering error)
    if df_typ.data[col_ind] != arr_typ:
        raise BodoError(
            "Changing dataframe column data type inplace is not supported in conditionals/loops or for dataframe arguments"
        )

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg = args
        dataframe_payload = get_dataframe_payload(context, builder, df_typ, df_arg)

        # decref existing data column if valid (unboxed)
        unboxed = builder.extract_value(dataframe_payload.unboxed, col_ind)
        is_unboxed = builder.icmp_unsigned("==", unboxed, lir.Constant(unboxed.type, 1))
        with builder.if_then(is_unboxed):
            arr = builder.extract_value(dataframe_payload.data, col_ind)
            context.nrt.decref(builder, df_typ.data[col_ind], arr)

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
        ptrty = context.get_value_type(payload_type).as_pointer()
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
            in_df_payload.unboxed,
            in_df.parent,
        )

        # increase refcount of stored values
        context.nrt.incref(builder, index_t, index_val)
        context.nrt.incref(builder, types.Tuple(df_t.data), in_df_payload.data)
        return dataframe

    ret_typ = DataFrameType(df_t.data, index_t, df_t.columns)
    sig = signature(ret_typ, df_t, index_t)
    return sig, codegen


@intrinsic
def set_df_column_with_reflect(typingctx, df_type, cname_type, arr_type=None):
    """Set df column and reflect to parent Python object
    return a new df.
    """
    assert is_literal_type(cname_type), "constant column name expected"
    col_name = get_literal_value(cname_type)
    n_cols = len(df_type.columns)
    new_n_cols = n_cols
    data_typs = df_type.data
    column_names = df_type.columns
    index_typ = df_type.index
    is_new_col = col_name not in df_type.columns
    col_ind = n_cols
    if is_new_col:
        data_typs += (arr_type,)
        column_names += (col_name,)
        new_n_cols += 1
    else:
        col_ind = df_type.columns.index(col_name)
        data_typs = tuple(
            (arr_type if i == col_ind else data_typs[i]) for i in range(n_cols)
        )

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg = args

        in_dataframe_payload = get_dataframe_payload(context, builder, df_type, df_arg)
        in_dataframe = cgutils.create_struct_proxy(df_type)(
            context, builder, value=df_arg
        )

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
            unboxed_tup,
            in_dataframe.parent,
        )

        # increase refcount of stored values
        context.nrt.incref(builder, index_typ, index_val)
        for var, typ in zip(data_arrs, data_typs):
            context.nrt.incref(builder, typ, var)

        # update existing native dataframe inplace if possible (not a new column name
        # and data type matches existing column)
        # see test_set_column_native_reflect
        if not is_new_col and arr_type == df_type.data[col_ind]:
            # old data arrays will be replaced so need a decref
            decref_df_data(context, builder, in_dataframe_payload, df_type)
            # store payload
            payload_type = DataFramePayloadType(df_type)
            payload_ptr = context.nrt.meminfo_data(builder, in_dataframe.meminfo)
            ptrty = context.get_value_type(payload_type).as_pointer()
            payload_ptr = builder.bitcast(payload_ptr, ptrty)
            out_dataframe_payload = get_dataframe_payload(
                context, builder, df_type, out_dataframe
            )
            builder.store(out_dataframe_payload._getvalue(), payload_ptr)

            # incref data again since there will be too references updated
            # TODO: incref only unboxed arrays to be safe?
            context.nrt.incref(builder, index_typ, index_val)
            for var, typ in zip(data_arrs, data_typs):
                context.nrt.incref(builder, typ, var)

        # set column of parent if not null, which is not fully known until runtime
        # see test_set_column_reflect_error
        has_parent = cgutils.is_not_null(builder, in_dataframe.parent)
        with builder.if_then(has_parent):
            # get boxed array
            pyapi = context.get_python_api(builder)
            gil_state = pyapi.gil_ensure()  # acquire GIL
            env_manager = context.get_env_manager(builder)

            context.nrt.incref(builder, arr_type, arr_arg)

            # call boxing for array data
            # TODO: check complex data types possible for Series for dataframes set column here
            c = numba.core.pythonapi._BoxContext(context, builder, pyapi, env_manager)
            py_arr = c.pyapi.from_native_value(arr_type, arr_arg, c.env_manager)

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

    ret_typ = DataFrameType(data_typs, index_typ, column_names)
    sig = signature(ret_typ, df_type, cname_type, arr_type)
    return sig, codegen


@lower_constant(DataFrameType)
def lower_constant_dataframe(context, builder, df_type, pyval):
    """embed constant DataFrame value by getting constant values for data arrays and
    Index.
    """
    n_cols = len(pyval.columns)
    data_tup = context.get_constant_generic(
        builder,
        types.Tuple(df_type.data),
        tuple(pyval.iloc[:, i].values for i in range(n_cols)),
    )
    index_val = context.get_constant_generic(builder, df_type.index, pyval.index)

    # set unboxed flags to 1 for all arrays
    one = context.get_constant(types.int8, 1)
    unboxed_tup = context.make_tuple(
        builder, types.UniTuple(types.int8, n_cols + 1), [one] * (n_cols + 1)
    )

    dataframe_val = construct_dataframe(
        context, builder, df_type, data_tup, index_val, unboxed_tup
    )

    return dataframe_val


@lower_cast(DataFrameType, DataFrameType)
def cast_df_to_df(context, builder, fromty, toty, val):
    """
    Support dataframe casting cases:
    1) convert RangeIndex to Int64Index
    2) cast empty dataframe to another dataframe
    (common pattern, see test_append_empty_df)
    """
    # RangeIndex to Int64Index case
    if (
        len(fromty.data) == len(toty.data)
        and isinstance(fromty.index, RangeIndexType)
        and isinstance(toty.index, NumericIndexType)
    ):
        dataframe_payload = get_dataframe_payload(context, builder, fromty, val)
        new_index = context.cast(
            builder, dataframe_payload.index, fromty.index, toty.index
        )
        new_data = dataframe_payload.data
        context.nrt.incref(builder, types.BaseTuple.from_types(fromty.data), new_data)

        df = construct_dataframe(
            context,
            builder,
            toty,
            new_data,
            new_index,
            dataframe_payload.unboxed,
            dataframe_payload.parent,
        )
        # TODO: fix casting refcount in Numba since Numba increfs value after cast
        return df

    # only empty dataframe case supported from this point
    if not len(fromty.data) == 0:
        raise BodoError(f"Invalid dataframe cast from {fromty} to {toty}")

    # generate empty dataframe with target type using empty arrays for data columns and
    # index
    extra_globals = {}
    # TODO: support MultiIndex
    if isinstance(toty.index, RangeIndexType):
        index = "bodo.hiframes.pd_index_ext.init_range_index(0, 0, 1, None)"
    else:
        index_arr_type = get_index_data_arr_types(toty.index)[0]
        n_extra_sizes = bodo.utils.transform.get_type_alloc_counts(index_arr_type) - 1
        extra_sizes = ", ".join("0" for _ in range(n_extra_sizes))
        index = "bodo.utils.conversion.index_from_array(bodo.utils.utils.alloc_type(0, index_arr_type, ({}{})))".format(
            extra_sizes, ", " if n_extra_sizes == 1 else ""
        )
        extra_globals["index_arr_type"] = index_arr_type

    data_args = []
    for i, arr_typ in enumerate(toty.data):
        n_extra_sizes = bodo.utils.transform.get_type_alloc_counts(arr_typ) - 1
        extra_sizes = ", ".join("0" for _ in range(n_extra_sizes))
        empty_arr = "bodo.utils.utils.alloc_type(0, arr_type{}, ({}{}))".format(
            i, extra_sizes, ", " if n_extra_sizes == 1 else ""
        )
        data_args.append(empty_arr)
        extra_globals[f"arr_type{i}"] = arr_typ
    data_args = ", ".join(data_args)

    func_text = "def impl():\n"
    gen_func = bodo.hiframes.dataframe_impl._gen_init_df(
        func_text, toty.columns, data_args, index, extra_globals
    )
    df = context.compile_internal(builder, gen_func, toty(), [])
    # TODO: fix casting refcount in Numba since Numba increfs value after cast
    return df


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
    func_text += (
        "  return bodo.hiframes.pd_dataframe_ext.init_dataframe({}, {}, {})\n".format(
            data_args, index_arg, col_var
        )
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
            raise BodoError("pd.DataFrame tuple input data not supported yet")
        assert len(data.types) % 2 == 1, "invalid const dict tuple structure"
        n_cols = (len(data.types) - 1) // 2
        data_keys = [t.literal_value for t in data.types[1 : n_cols + 1]]
        data_val_types = dict(zip(data_keys, data.types[n_cols + 1 :]))
        data_arrs = ["data[{}]".format(i) for i in range(n_cols + 1, 2 * n_cols + 1)]
        data_dict = dict(zip(data_keys, data_arrs))
        # if no index provided and there are Series inputs, get index from them
        # XXX cannot handle alignment of multiple Series
        if is_overload_none(index):
            for i, t in enumerate(data.types[n_cols + 1 :]):
                if isinstance(t, SeriesType):
                    index_arg = (
                        "bodo.hiframes.pd_series_ext.get_series_index(data[{}])".format(
                            n_cols + 1 + i
                        )
                    )
                    index_is_none = False
                    break
    # empty dataframe
    elif is_overload_none(data):
        data_dict = {}
        data_val_types = {}
    else:
        # ndarray case
        # checks for 2d and column args
        # TODO: error checking
        if not (isinstance(data, types.Array) and data.ndim == 2):  # pragma: no cover
            raise BodoError(
                "pd.DataFrame() supports constant dictionary and array input"
            )
        if is_overload_none(columns):  # pragma: no cover
            raise BodoError(
                "pd.DataFrame() 'columns' argument is required when"
                " an array is passed as data"
            )
        copy_str = ".copy()" if copy else ""
        columns_consts = get_overload_const_list(columns)
        n_cols = len(columns_consts)
        data_val_types = {c: data.copy(ndim=1) for c in columns_consts}
        data_arrs = ["data[:,{}]{}".format(i, copy_str) for i in range(n_cols)]
        data_dict = dict(zip(columns_consts, data_arrs))

    if is_overload_none(columns):
        col_names = data_dict.keys()
    else:
        col_names = get_overload_const_list(columns)

    df_len = _get_df_len_from_info(
        data_dict, data_val_types, col_names, index_is_none, index_arg
    )
    _fill_null_arrays(data_dict, col_names, df_len, dtype)

    # set default RangeIndex if index argument is None and data argument isn't Series
    if index_is_none:
        # empty df has object Index in Pandas which correponds to our StringIndex
        if is_overload_none(data):
            index_arg = "bodo.hiframes.pd_index_ext.init_string_index(bodo.libs.str_arr_ext.pre_alloc_string_array(0, 0))"
        else:
            index_arg = (
                "bodo.hiframes.pd_index_ext.init_range_index(0, {}, 1, None)".format(
                    df_len
                )
            )

    data_args = "({},)".format(
        ", ".join(
            "bodo.utils.conversion.coerce_to_array({}, True, scalar_to_arr_len={}){}".format(
                data_dict[c], df_len, astype_str
            )
            for c in col_names
        )
    )
    if len(col_names) == 0:
        data_args = "()"

    return col_names, data_args, index_arg


def _get_df_len_from_info(
    data_dict, data_val_types, col_names, index_is_none, index_arg
):
    """return generated text for length of dataframe, given the input info in the
    pd.DataFrame() call
    """
    df_len = "0"
    for c in col_names:
        if c in data_dict and is_iterable_type(data_val_types[c]):
            df_len = "len({})".format(data_dict[c])
            break

    # If we haven't found a length, rely on the index
    if df_len == "0" and not index_is_none:
        df_len = f"len({index_arg})"

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
        dtype = "bodo.string_array_type"
    else:
        dtype = "bodo.utils.conversion.array_type_from_dtype(dtype)"

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
        return lambda df: 0  # pragma: no cover
    return lambda df: len(
        bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, 0)
    )  # pragma: no cover


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


# a dummy join function that will be replace in dataframe_pass
def join_dummy(
    left_df,
    right_df,
    left_on,
    right_on,
    how,
    suffix_x,
    suffix_y,
    is_join,
    indicator,
    _bodo_na_equal,
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
            indicator,
            _bodo_na_equal,
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

        # The left side. All of it got included.
        for in_type, col in zip(left_df.data, left_df.columns):
            columns.append(
                str(col) + suffix_x.literal_value if col in add_suffix else col
            )
            if col in comm_keys:
                # For a common key we take either from left or right, so no additional NaN occurs.
                data.append(in_type)
            else:
                # For a key that is not common OR data column, we have to plan for a NaN column
                data.append(to_nullable_type(in_type) if is_right else in_type)
        # The right side
        # common keys are added only once so avoid adding them
        for in_type, col in zip(right_df.data, right_df.columns):
            if col not in comm_keys:
                # a key column that is not common needs to plan for NaN.
                # Same for a data column of course.
                columns.append(
                    str(col) + suffix_y.literal_value if col in add_suffix else col
                )
                data.append(to_nullable_type(in_type) if is_left else in_type)

        # If indicator=True, add a column called "_merge", which is categorical
        # with Categories: ['left_only', 'right_only', 'both']
        indicator_value = get_overload_const_bool(indicator)
        if indicator_value:
            columns.append("_merge")
            data.append(
                bodo.CategoricalArrayType(
                    bodo.PDCategoricalDtype(
                        ("left_only", "right_only", "both"), bodo.string_type, False
                    )
                )
            )
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


JoinTyper._no_unliteral = True


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(join_dummy, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return dataframe._getvalue()


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
    axis = get_overload_const_int(axis)
    ignore_index = is_overload_true(ignore_index)

    unsupported_args = dict(
        join=join,
        join_axes=join_axes,
        keys=keys,
        levels=levels,
        names=names,
        verify_integrity=verify_integrity,
        sort=sort,
        copy=copy,
    )

    arg_defaults = dict(
        join="outer",
        join_axes=None,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        sort=None,
        copy=True,
    )
    check_unsupported_args("pd.concat", unsupported_args, arg_defaults)

    func_text = (
        "def impl(objs, axis=0, join='outer', join_axes=None, "
        "ignore_index=False, keys=None, levels=None, names=None, "
        "verify_integrity=False, sort=None, copy=True):\n"
    )

    # concat of columns into a dataframe
    if axis == 1:
        if not isinstance(objs, types.BaseTuple):
            # using raise_bodo_error() since typing pass may transform list to tuple
            raise_bodo_error("Only tuple argument for pd.concat(axis=1) expected")
        index = "bodo.hiframes.pd_index_ext.init_range_index(0, len(objs[0]), 1, None)"
        col_no = 0
        data_args = []
        names = []
        for i, obj in enumerate(objs.types):
            assert isinstance(obj, (SeriesType, DataFrameType))
            if isinstance(obj, SeriesType):
                # TODO: use Series name if possible
                names.append(str(col_no))
                col_no += 1
                data_args.append(
                    "bodo.hiframes.pd_series_ext.get_series_data(objs[{}])".format(i)
                )
            else:  # DataFrameType
                names.extend(obj.columns)
                for j in range(len(obj.data)):
                    data_args.append(
                        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(objs[{}], {})".format(
                            i, j
                        )
                    )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text, names, ", ".join(data_args), index
        )

    assert axis == 0

    # dataframe tuples case
    if isinstance(objs, types.BaseTuple) and isinstance(objs.types[0], DataFrameType):
        assert all(isinstance(t, DataFrameType) for t in objs.types)

        # get output column names
        all_colnames = []
        for df in objs.types:
            all_colnames.extend(df.columns)

        # remove duplicates but keep original order
        all_colnames = list(dict.fromkeys(all_colnames).keys())

        # get array types for all output columns (for NA generation for missing columns)
        arr_types = {}
        for col_no, c in enumerate(all_colnames):
            for df in objs.types:
                if c in df.columns:
                    arr_types["arr_typ{}".format(col_no)] = df.data[df.columns.index(c)]
                    break
        assert len(arr_types) == len(all_colnames)

        # generate concat for each output column
        out_data = []
        for col_no, c in enumerate(all_colnames):
            args = []
            for i, df in enumerate(objs.types):
                if c in df.columns:
                    col_ind = df.columns.index(c)
                    args.append(
                        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(objs[{}], {})".format(
                            i, col_ind
                        )
                    )
                else:
                    args.append(
                        "bodo.libs.array_kernels.gen_na_array(len(objs[{}]), arr_typ{})".format(
                            i, col_no
                        )
                    )
            func_text += "  A{} = bodo.libs.array_kernels.concat(({},))\n".format(
                col_no, ", ".join(args)
            )
        if ignore_index:
            index = "bodo.hiframes.pd_index_ext.init_range_index(0, len(A0), 1, None)"
        else:
            index = "bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(({},)))\n".format(
                ", ".join(
                    "bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(objs[{}]))".format(
                        i
                    )
                    # ignore dummy string index of empty dataframes (test_append_empty_df)
                    for i in range(len(objs.types))
                    if len(objs[i].columns) > 0
                )
            )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text,
            all_colnames,
            ", ".join("A{}".format(i) for i in range(len(all_colnames))),
            index,
            arr_types,
        )

    # series tuples case
    if isinstance(objs, types.BaseTuple) and isinstance(objs.types[0], SeriesType):
        assert all(isinstance(t, SeriesType) for t in objs.types)
        # TODO: index and name
        func_text += "  out_arr = bodo.libs.array_kernels.concat(({},))\n".format(
            ", ".join(
                "bodo.hiframes.pd_series_ext.get_series_data(objs[{}])".format(i)
                for i in range(len(objs.types))
            )
        )
        if ignore_index:
            func_text += "  index = bodo.hiframes.pd_index_ext.init_range_index(0, len(out_arr), 1, None)\n"
        else:
            func_text += "  index = bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(({},)))\n".format(
                ", ".join(
                    "bodo.utils.conversion.index_to_array(bodo.hiframes.pd_series_ext.get_series_index(objs[{}]))".format(
                        i
                    )
                    for i in range(len(objs.types))
                )
            )
        func_text += (
            "  return bodo.hiframes.pd_series_ext.init_series(out_arr, index)\n"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "np": np, "numba": numba}, loc_vars)
        return loc_vars["impl"]

    # list of dataframes
    if isinstance(objs, types.List) and isinstance(objs.dtype, DataFrameType):
        # TODO(ehsan): index
        df_type = objs.dtype
        for col_no, c in enumerate(df_type.columns):
            func_text += "  arrs{} = []\n".format(col_no)
            func_text += "  for i in range(len(objs)):\n"
            func_text += "    df = objs[i]\n"
            func_text += "    arrs{0}.append(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {0}))\n".format(
                col_no
            )
            func_text += (
                "  out_arr{0} = bodo.libs.array_kernels.concat(arrs{0})\n".format(
                    col_no
                )
            )
        if ignore_index:
            index = (
                "bodo.hiframes.pd_index_ext.init_range_index(0, len(out_arr0), 1, None)"
            )
        else:
            func_text += "  arrs_index = []\n"
            func_text += "  for i in range(len(objs)):\n"
            func_text += "    df = objs[i]\n"
            func_text += "    arrs_index.append(bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)))\n"
            index = "bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(arrs_index))\n"
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text,
            df_type.columns,
            ", ".join("out_arr{}".format(i) for i in range(len(df_type.columns))),
            index,
        )

    # list of Series
    if isinstance(objs, types.List) and isinstance(objs.dtype, SeriesType):
        func_text += "  arrs = []\n"
        func_text += "  for i in range(len(objs)):\n"
        func_text += (
            "    arrs.append(bodo.hiframes.pd_series_ext.get_series_data(objs[i]))\n"
        )
        func_text += "  out_arr = bodo.libs.array_kernels.concat(arrs)\n"
        if ignore_index:
            func_text += "  index = bodo.hiframes.pd_index_ext.init_range_index(0, len(out_arr), 1, None)\n"
        else:
            func_text += "  arrs_index = []\n"
            func_text += "  for i in range(len(objs)):\n"
            func_text += "    S = objs[i]\n"
            func_text += "    arrs_index.append(bodo.utils.conversion.index_to_array(bodo.hiframes.pd_series_ext.get_series_index(S)))\n"
            func_text += "  index = bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(arrs_index))\n"
        func_text += (
            "  return bodo.hiframes.pd_series_ext.init_series(out_arr, index)\n"
        )
        loc_vars = {}
        exec(func_text, {"bodo": bodo, "np": np, "numba": numba}, loc_vars)
        return loc_vars["impl"]

    # TODO: handle other iterables like arrays, lists, ...
    raise BodoError("pd.concat(): input type {} not supported yet".format(objs))


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
        ret_typ = df.copy(index=index)
        return signature(ret_typ, *args)


SortDummyTyper._no_unliteral = True


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(sort_values_dummy, types.VarArg(types.Any))
def lower_sort_values_dummy(context, builder, sig, args):
    if sig.return_type == types.none:
        return

    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


# dummy function to change the df type to have set_parent=True
# used in sort_values(inplace=True) hack
def set_parent_dummy(df):  # pragma: no cover
    return df


@infer_global(set_parent_dummy)
class ParentDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (df,) = args
        ret = DataFrameType(df.data, df.index, df.columns)
        return signature(ret, *args)


@lower_builtin(set_parent_dummy, types.VarArg(types.Any))
def lower_set_parent_dummy(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# TODO: jitoptions for overload_method and infer_global
# (no_cpython_wrapper to avoid error for iterator object)
@overload_method(DataFrameType, "itertuples", inline="always", no_unliteral=True)
def itertuples_overload(df, index=True, name="Pandas"):

    unsupported_args = dict(index=index, name=name)
    arg_defaults = dict(index=True, name="Pandas")
    check_unsupported_args("DataFrame.itertuples", unsupported_args, arg_defaults)

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


def gen_pandas_parquet_metadata(
    df,
    write_non_range_index_to_metadata,
    write_rangeindex_to_metadata,
    partition_cols=None,
):
    # returns dict with pandas dataframe metadata for parquet storage.
    # For more information, see:
    # https://pandas.pydata.org/pandas-docs/stable/development/developer.html#storing-pandas-dataframe-objects-in-apache-parquet-format

    pandas_metadata = {}

    pandas_metadata["columns"] = []

    if partition_cols is None:
        partition_cols = []
    for col_name, col_type in zip(df.columns, df.data):
        if col_name in partition_cols:
            # partition columns are not written to parquet files, and don't appear
            # in pandas metadata
            continue
        if isinstance(col_type, types.Array) or col_type == boolean_array:
            pandas_type = numpy_type = col_type.dtype.name
            if numpy_type.startswith("datetime"):
                pandas_type = "datetime"
        elif col_type == string_array_type:
            pandas_type = "unicode"
            numpy_type = "object"
        elif col_type == binary_array_type:
            pandas_type = "bytes"
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
        elif isinstance(col_type, (StructArrayType, ArrayItemArrayType)):
            # TODO: provide meaningful pandas_type when possible.
            # For example "pandas_type": "list[list[int64]]", "numpy_type": "object"
            # can occur
            pandas_type = "object"
            numpy_type = "object"
        # TODO: metadata for categorical arrays
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
    if not is_overload_none(compression) and get_overload_const_str(
        compression
    ) not in {"snappy", "gzip", "brotli"}:
        raise BodoError(
            "to_parquet(): Unsupported compression: "
            + str(get_overload_const_str(compression))
        )

    if not is_overload_none(partition_cols):
        partition_cols = get_overload_const_list(partition_cols)
        part_col_idxs = []
        for part_col_name in partition_cols:
            try:
                idx = df.columns.index(part_col_name)
            except ValueError:
                raise BodoError(f"Partition column {part_col_name} is not in dataframe")
            part_col_idxs.append(idx)
    else:
        partition_cols = None

    if not is_overload_none(index) and not is_overload_constant_bool(index):
        raise BodoError("to_parquet(): index must be a constant bool or None")

    from bodo.io.parquet_pio import (
        parquet_write_table_cpp,
        parquet_write_table_partitioned_cpp,
    )

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
            df,
            write_non_range_index_to_metadata,
            write_rangeindex_to_metadata,
            partition_cols=partition_cols,
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
    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += f"    bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(fname, parallel=_is_parallel)\n"
    if partition_cols:
        # We need the values of the categories for any partition columns that
        # are categorical arrays, because they are used to generate the
        # output directory name
        categories_args = ", ".join(
            f"array_to_info(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i}).dtype.categories.values)"
            for i in range(len(df.columns))
            if isinstance(df.data[i], CategoricalArrayType) and (i in part_col_idxs)
        )
        if categories_args:
            func_text += "    cat_info_list = [{}]\n".format(categories_args)
            func_text += "    cat_table = arr_info_list_to_table(cat_info_list)\n"
        else:
            func_text += "    cat_table = table\n"  # hack to avoid typing issue
        col_names_no_partitions_text = ", ".join(
            '"{}"'.format(col_name)
            for col_name in df.columns
            if col_name not in partition_cols
        )
        func_text += "    col_names_no_partitions = array_to_info(str_arr_from_sequence([{}]))\n".format(
            col_names_no_partitions_text
        )
        func_text += f"    part_cols_idxs = np.array({part_col_idxs}, dtype=np.int32)\n"
        func_text += "    parquet_write_table_partitioned_cpp(unicode_to_utf8(fname),\n"
        func_text += "                            table, col_names, col_names_no_partitions, cat_table,\n"
        func_text += (
            "                            part_cols_idxs.ctypes, len(part_cols_idxs),\n"
        )
        func_text += "                            unicode_to_utf8(compression),\n"
        func_text += "                            _is_parallel,\n"
        func_text += "                            unicode_to_utf8(bucket_region))\n"
    elif write_rangeindex_to_metadata:
        func_text += "    parquet_write_table_cpp(unicode_to_utf8(fname),\n"
        func_text += "                            table, col_names, index_col,\n"
        func_text += "                            " + str(write_index) + ",\n"
        func_text += "                            unicode_to_utf8(metadata),\n"
        func_text += "                            unicode_to_utf8(compression),\n"
        func_text += "                            _is_parallel, 1, df.index.start,\n"
        func_text += "                            df.index.stop, df.index.step,\n"
        func_text += "                            unicode_to_utf8(name_ptr),\n"
        func_text += "                            unicode_to_utf8(bucket_region))\n"
    else:
        func_text += "    parquet_write_table_cpp(unicode_to_utf8(fname),\n"
        func_text += "                            table, col_names, index_col,\n"
        func_text += "                            " + str(write_index) + ",\n"
        func_text += "                            unicode_to_utf8(metadata),\n"
        func_text += "                            unicode_to_utf8(compression),\n"
        func_text += "                            _is_parallel, 0, 0, 0, 0,\n"
        func_text += "                            unicode_to_utf8(name_ptr),\n"
        func_text += "                            unicode_to_utf8(bucket_region))\n"

    loc_vars = {}
    exec(
        func_text,
        {
            "np": np,
            "bodo": bodo,
            "unicode_to_utf8": unicode_to_utf8,
            "array_to_info": array_to_info,
            "arr_info_list_to_table": arr_info_list_to_table,
            "str_arr_from_sequence": str_arr_from_sequence,
            "parquet_write_table_cpp": parquet_write_table_cpp,
            "parquet_write_table_partitioned_cpp": parquet_write_table_partitioned_cpp,
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
    unsupported_args = dict(chunksize=chunksize)
    arg_defaults = dict(chunksize=None)
    check_unsupported_args("to_sql", unsupported_args, arg_defaults)

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
    if not (
        is_overload_none(path_or_buf)
        or is_overload_constant_str(path_or_buf)
        or path_or_buf == string_type
    ):
        raise BodoError(
            "DataFrame.to_csv(): 'path_or_buf' argument should be None or string"
        )
    # TODO: refactor when objmode() can understand global string constant
    # String output case
    if is_overload_none(path_or_buf):

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

        bodo.io.fs_io.csv_write(path_or_buf, D)

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

        # Assuming that path_or_buf is a string
        bucket_region = bodo.io.fs_io.get_s3_bucket_region_njit(
            path_or_buf, parallel=False
        )

        if lines and orient == "records":
            bodo.hiframes.pd_dataframe_ext._json_write(
                unicode_to_utf8(path_or_buf),
                unicode_to_utf8(D),
                0,
                len(D),
                False,
                True,
                unicode_to_utf8(bucket_region),
            )
            # Check if there was an error in the C++ code. If so, raise it.
            bodo.utils.utils.check_and_propagate_cpp_exception()
        else:
            bodo.hiframes.pd_dataframe_ext._json_write(
                unicode_to_utf8(path_or_buf),
                unicode_to_utf8(D),
                0,
                len(D),
                False,
                False,
                unicode_to_utf8(bucket_region),
            )
            # Check if there was an error in the C++ code. If so, raise it.
            bodo.utils.utils.check_and_propagate_cpp_exception()

    return _impl


@overload(pd.get_dummies, inline="always", no_unliteral=True)
def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
):
    args_dict = {
        "prefix": prefix,
        "prefix_sep": prefix_sep,
        "dummy_na": dummy_na,
        "columns": columns,
        "sparse": sparse,
        "drop_first": drop_first,
        "dtype": dtype,
    }
    args_default_dict = {
        "prefix": None,
        "prefix_sep": "_",
        "dummy_na": False,
        "columns": None,
        "sparse": False,
        "drop_first": False,
        "dtype": None,
    }
    check_unsupported_args("pd.get_dummies", args_dict, args_default_dict)
    if not categorical_can_construct_dataframe(data):
        raise BodoError(
            "pd.get_dummies() only support categorical data types with explicitly known categories"
        )

    func_text = "def impl(data, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False, dtype=None,):\n"
    if isinstance(data, SeriesType):
        categories = data.data.dtype.categories
        func_text += (
            "  data_values = bodo.hiframes.pd_series_ext.get_series_data(data)\n"
        )
    else:
        categories = data.dtype.categories
        func_text += "  data_values = data\n"

    n_cols = len(categories)

    # Pandas implementation:
    func_text += "  codes = bodo.hiframes.pd_categorical_ext.get_categorical_arr_codes(data_values)\n"
    func_text += "  numba.parfors.parfor.init_prange()\n"
    func_text += "  n = len(data_values)\n"
    for i in range(n_cols):
        func_text += "  data_arr_{} = np.empty(n, np.uint8)\n".format(i)
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "      if bodo.libs.array_kernels.isna(data_values, i):\n"
    for j in range(n_cols):
        func_text += "          data_arr_{}[i] = 0\n".format(j)
    func_text += "      else:\n"
    for k in range(n_cols):
        func_text += "          data_arr_{0}[i] = codes[i] == {0}\n".format(k)
    data_args = ", ".join(f"data_arr_{i}" for i in range(n_cols))
    index = "bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)"

    # TODO(Nick): Replace categories with categorical index type
    return bodo.hiframes.dataframe_impl._gen_init_df(
        func_text, categories, data_args, index
    )


def categorical_can_construct_dataframe(val):
    """Helper function that returns if a datatype is categorical and has constant
    values that can be used as column names for dataframes
    """
    if isinstance(val, CategoricalArrayType):
        return val.dtype.categories is not None
    elif isinstance(val, SeriesType) and isinstance(val.data, CategoricalArrayType):
        return val.data.dtype.categories is not None
    return False


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
    pd.factorize,
    pd.wide_to_long,
    ## Top-level dealing with datetimelike
    pd.bdate_range,
    pd.period_range,
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

dataframe_unsupported = {
    "to_latex",
    "from_dict",
    "reindex_like",
    "pivot",
    "clip",
    "slice_shift",
    "tz_convert",
    "combine",
    "convert_dtypes",
    "floordiv",
    "eval",
    "applymap",
    "nlargest",
    "to_markdown",
    "rmul",
    "pad",
    "sparse",
    "combine_first",
    "kurt",
    "at_time",
    "mad",
    "mask",
    "to_html",
    "unstack",
    "iteritems",
    "between_time",
    "mod",
    "to_gbq",
    "rank",
    "round",
    "mode",
    "multiply",
    "value_counts",
    "corrwith",
    "set_axis",
    "nsmallest",
    "to_dict",
    "to_feather",
    "cummax",
    "to_stata",
    "ne",
    "ewm",
    "first",
    "expanding",
    "droplevel",
    "truncate",
    "asof",
    "pow",
    "reorder_levels",
    "mul",
    "last",
    "agg",
    "le",
    "any",
    "xs",
    "explode",
    "equals",
    "asfreq",
    "pop",
    "iterrows",
    "rename_axis",
    "resample",
    "to_xarray",
    "items",
    "radd",
    "tshift",
    "rsub",
    "align",
    "add",
    "squeeze",
    "swapaxes",
    "to_pickle",
    "to_timestamp",
    "interpolate",
    "eq",
    "bool",
    "skew",
    "rdiv",
    "div",
    "sem",
    "tz_localize",
    "lt",
    "bfill",
    "last_valid_index",
    "to_records",
    "keys",
    "to_clipboard",
    "transform",
    "dot",
    "truediv",
    "gt",
    "add_prefix",
    "divide",
    "lookup",
    "infer_objects",
    "melt",
    "rmod",
    "aggregate",
    "from_records",
    "rpow",
    "to_excel",
    "subtract",
    "rfloordiv",
    "ffill",
    "to_hdf",
    "update",
    "sub",
    "hist",
    "ge",
    "get",
    "all",
    "plot",
    "backfill",
    "stack",
    "where",
    "transpose",
    "T",
    "rtruediv",
    "cummin",
    "swaplevel",
    "first_valid_index",
    "compare",
    "boxplot",
    "to_period",
    "add_suffix",
    "kurtosis",
    "reindex",
    # Indexing, iteration
    "at",
    "__iter__",
}


dataframe_unsupported_attrs = (
    "axes",
    "at",
    "attrs",
)


def _install_pd_unsupported():
    """install an overload that raises BodoError for unsupported functions"""
    for f in pd_unsupported:
        fname = "pd." + f.__name__
        overload(f, no_unliteral=True)(create_unsupported_overload(fname))


def _install_dataframe_unsupported():
    """install an overload that raises BodoError for unsupported Dataframe methods"""

    for attr_name in dataframe_unsupported_attrs:
        full_name = "DataFrame." + attr_name
        overload_attribute(DataFrameType, attr_name)(
            create_unsupported_overload(full_name)
        )
    for fname in dataframe_unsupported:
        full_name = "Dataframe." + fname
        overload_method(DataFrameType, fname)(create_unsupported_overload(full_name))


_install_pd_unsupported()
_install_dataframe_unsupported()
