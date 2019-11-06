# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Implement pd.DataFrame typing and data model handling.
"""
import operator
from collections import namedtuple
import pandas as pd
import numpy as np
import numba
from numba import types, cgutils
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
from numba.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
from numba.targets.imputils import impl_ret_new_ref, impl_ret_borrowed
import bodo
from bodo.hiframes.pd_series_ext import SeriesType
from bodo.libs.str_ext import string_type
from bodo.utils.typing import (
    BodoError,
    is_overload_none,
    is_overload_constant_bool,
    is_overload_constant_str,
    is_overload_constant_str_list,
    is_overload_true,
    is_overload_false,
    is_overload_zero,
    get_overload_const_str,
    get_const_str_list,
)


class DataFrameType(types.Type):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects.
    """

    def __init__(self, data=None, index=None, columns=None, has_parent=False):
        # data is tuple of Array types (not Series)
        # index is Index obj (not Array type)
        # columns is tuple of strings

        self.data = data
        if index is None:
            index = types.none
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
            index = types.none if self.index == types.none else self.index.copy()
        data = tuple(a.copy() for a in self.data)
        if has_parent is None:
            has_parent = self.has_parent
        return DataFrameType(data, index, self.columns, has_parent)

    @property
    def key(self):
        # needed?
        return self.data, self.index, self.columns, self.has_parent

    def unify(self, typingctx, other):
        if (
            isinstance(other, DataFrameType)
            and len(other.data) == len(self.data)
            and other.columns == self.columns
            and other.has_parent == self.has_parent
        ):
            new_index = types.none
            if self.index != types.none and other.index != types.none:
                new_index = self.index.unify(typingctx, other.index)
            elif other.index != types.none:
                new_index = other.index
            elif self.index != types.none:
                new_index = self.index

            data = tuple(a.unify(typingctx, b) for a, b in zip(self.data, other.data))
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
        ]
        super(DataFramePayloadModel, self).__init__(dmm, fe_type, members)


@register_model(DataFrameType)
class DataFrameModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        n_cols = len(fe_type.columns)
        payload_type = DataFramePayloadType(fe_type)
        # payload_type = types.Opaque('Opaque.DataFrame')
        # TODO: does meminfo decref content when object is deallocated?
        members = [
            ("columns", types.UniTuple(string_type, n_cols)),
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

    def resolve_shape(self, ary):
        return types.UniTuple(types.intp, 2)

    def resolve_iat(self, ary):
        return DataFrameIatType(ary)

    def resolve_iloc(self, ary):
        return DataFrameILocType(ary)

    def resolve_loc(self, ary):
        return DataFrameLocType(ary)

    @bound_function("df.apply")
    def resolve_apply(self, df, args, kws):
        kws = dict(kws)
        func = args[0] if len(args) > 0 else kws.get("func", None)
        # check lambda
        if not isinstance(func, types.MakeFunctionLiteral):
            raise ValueError("df.apply(): lambda not found")

        # check axis
        axis = args[1] if len(args) > 1 else kws.get("axis", None)
        if (
            axis is None
            or not isinstance(axis, types.IntegerLiteral)
            or axis.literal_value != 1
        ):
            raise ValueError("only apply() with axis=1 supported")

        # using NamedTuple instead of Series, TODO: pass Series
        Row = namedtuple("R", df.columns)

        # the data elements come from getitem of Series to perform conversion
        # e.g. dt64 to timestamp in TestDate.test_ts_map_date2
        dtypes = []
        for arr_typ in df.data:
            series_typ = SeriesType(arr_typ.dtype, arr_typ, df.index, string_type)
            el_typ = self.context.resolve_function_type(
                operator.getitem, (series_typ, types.int64), {}
            ).return_type
            dtypes.append(el_typ)

        row_typ = types.NamedTuple(dtypes, Row)
        code = func.literal_value.code
        f_ir = numba.ir_utils.get_ir_of_code({"np": np}, code)
        _, f_return_type, _ = numba.typed_passes.type_inference_stage(
            self.context, f_ir, (row_typ,), None
        )

        return signature(SeriesType(f_return_type, index=df.index), *args)

    def generic_resolve(self, df, attr):
        if attr in df.columns:
            ind = df.columns.index(attr)
            arr_typ = df.data[ind]
            return SeriesType(arr_typ.dtype, arr_typ, df.index, string_type)


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
    meminfo = context.nrt.meminfo_alloc(
        builder, context.get_constant(types.uintp, payload_size)
    )
    meminfo_data_ptr = context.nrt.meminfo_data(builder, meminfo)
    meminfo_data_ptr = builder.bitcast(meminfo_data_ptr, payload_ll_type.as_pointer())
    builder.store(dataframe_payload._getvalue(), meminfo_data_ptr)

    # create dataframe struct
    dataframe = cgutils.create_struct_proxy(df_type)(context, builder)
    dataframe.columns = column_tup
    dataframe.meminfo = meminfo
    if parent is None:
        # Set parent to NULL
        dataframe.parent = cgutils.get_null_value(dataframe.parent.type)
    else:
        dataframe.parent = parent
    return dataframe._getvalue()


@intrinsic
def init_dataframe(typingctx, data_tup_typ, index_typ, col_names_typ=None):
    """Create a DataFrame with provided data, index and columns values.
    Used as a single constructor for DataFrame and assigning its data, so that
    optimization passes can look for init_dataframe() to see if underlying
    data has changed, and get the array variables from init_dataframe() args if
    not changed.
    """

    n_cols = len(data_tup_typ.types)
    # assert all(isinstance(t, types.StringLiteral) for t in col_names_typ.types)
    column_names = col_names_typ.consts
    assert len(column_names) == n_cols

    def codegen(context, builder, signature, args):
        df_type = signature.return_type
        data_tup = args[0]
        index_val = args[1]
        column_strs = [
            numba.unicode.make_string_from_constant(context, builder, string_type, c)
            for c in column_names
        ]

        column_tup = context.make_tuple(
            builder, types.UniTuple(string_type, n_cols), column_strs
        )
        zero = context.get_constant(types.int8, 0)
        unboxed_tup = context.make_tuple(
            builder, types.UniTuple(types.int8, n_cols + 1), [zero] * (n_cols + 1)
        )

        dataframe_val = construct_dataframe(
            context, builder, df_type, data_tup, index_val, column_tup, unboxed_tup
        )

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, data_tup_typ, data_tup)
            context.nrt.incref(builder, index_typ, index_val)
            for var in column_strs:
                context.nrt.incref(builder, string_type, var)

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


# TODO: alias analysis
# this function should be used for getting df._data for alias analysis to work
# no_cpython_wrapper since Array(DatetimeDate) cannot be boxed
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_data(df, i):
    def _impl(df, i):
        if has_parent(df) and _get_dataframe_unboxed(df)[i] == 0:
            bodo.hiframes.boxing.unbox_dataframe_column(df, i)
        return _get_dataframe_data(df)[i]

    return _impl


# TODO: use separate index type instead of just storing array
@numba.generated_jit(nopython=True, no_cpython_wrapper=True)
def get_dataframe_index(df):
    return lambda df: _get_dataframe_index(df)


@intrinsic
def set_dataframe_data(typingctx, df_typ, c_ind_typ, arr_typ=None):
    col_ind = c_ind_typ.literal_value

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg = args
        dataframe_payload = get_dataframe_payload(context, builder, df_typ, df_arg)
        # assign array and set unboxed flag
        dataframe_payload.data = builder.insert_value(
            dataframe_payload.data, arr_arg, col_ind
        )
        dataframe_payload.unboxed = builder.insert_value(
            dataframe_payload.unboxed, context.get_constant(types.int8, 1), col_ind
        )

        if context.enable_nrt:
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
        if context.enable_nrt:
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
    assert isinstance(inplace, bodo.utils.utils.BooleanLiteral)
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

        column_strs = [
            numba.unicode.make_string_from_constant(context, builder, string_type, c)
            for c in column_names
        ]

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
        column_tup = context.make_tuple(
            builder, types.UniTuple(string_type, new_n_cols), column_strs
        )
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
            column_tup,
            unboxed_tup,
            in_dataframe.parent,
        )

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, index_typ, index_val)
            for var, typ in zip(data_arrs, data_typs):
                context.nrt.incref(builder, typ, var)
            for var in column_strs:
                context.nrt.incref(builder, string_type, var)

        # TODO: test this
        # test_set_column_cond3 doesn't test it for some reason
        if is_inplace:
            # store payload
            payload_type = DataFramePayloadType(df)
            payload_ptr = context.nrt.meminfo_data(builder, in_dataframe.meminfo)
            ptrty = context.get_data_type(payload_type).as_pointer()
            payload_ptr = builder.bitcast(payload_ptr, ptrty)
            out_dataframe_payload = get_dataframe_payload(
                context, builder, df, out_dataframe
            )
            builder.store(out_dataframe_payload._getvalue(), payload_ptr)

        # set column of parent
        # get boxed array
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()  # acquire GIL
        env_manager = context.get_env_manager(builder)

        if context.enable_nrt:
            context.nrt.incref(builder, arr, arr_arg)

        # call boxing for array data
        # TODO: check complex data types possible for Series for dataframes set column here
        c = numba.pythonapi._BoxContext(context, builder, pyapi, env_manager)
        py_arr = bodo.hiframes.boxing._box_series_data(arr.dtype, arr, arr_arg, c)

        # get column as string obj
        cstr = context.insert_const_string(builder.module, col_name)
        cstr_obj = pyapi.string_from_string(cstr)

        # set column array
        pyapi.object_setitem(in_dataframe.parent, cstr_obj, py_arr)

        pyapi.decref(py_arr)
        pyapi.decref(cstr_obj)

        pyapi.gil_release(gil_state)  # release GIL

        return out_dataframe

    ret_typ = DataFrameType(data_typs, index_typ, column_names, True)
    sig = signature(ret_typ, df, cname, arr, inplace)
    return sig, codegen


# TODO: fix default/kws arg handling in inline_closurecall
# e.g. bodo/tests/test_dataframe.py::TestDataFrame::test_create_dtype1
# @overload(pd.DataFrame, inline='always')
@overload(pd.DataFrame)
def pd_dataframe_overload(data=None, index=None, columns=None, dtype=None, copy=False):
    # TODO: support other input combinations
    if not isinstance(copy, (bool, bodo.utils.utils.BooleanLiteral, types.Omitted)):
        raise ValueError("pd.DataFrame(): copy argument should be constant")

    # get value of copy
    copy = getattr(copy, "literal_value", copy)  # literal type
    copy = getattr(copy, "value", copy)  # ommited type
    assert isinstance(copy, bool)

    col_args, data_args, index_arg = _get_df_args(data, index, columns, dtype, copy)
    col_var = "bodo.utils.typing.add_consts_to_type([{}], {})".format(
        col_args, col_args
    )

    func_text = (
        "def _init_df(data=None, index=None, columns=None, dtype=None, copy=False):\n"
    )
    func_text += "  return bodo.hiframes.pd_dataframe_ext.init_dataframe(({},), {}, {})\n".format(
        data_args, index_arg, col_var
    )
    loc_vars = {}
    exec(func_text, {"bodo": bodo, "np": np}, loc_vars)
    # print(func_text)
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

    index_arg = "bodo.utils.conversion.convert_to_index(index)"

    # data is sentinel tuple (converted from dictionary)
    if isinstance(data, types.Tuple):
        # first element is sentinel
        if not data.types[0] == types.StringLiteral("__bodo_tup"):
            raise ValueError("pd.DataFrame tuple input data not supported yet")
        n_cols = (len(data.types) - 1) // 2
        data_keys = [t.literal_value for t in data.types[1 : n_cols + 1]]
        data_arrs = [
            "bodo.utils.conversion.coerce_to_array(data[{}], True, True){}".format(
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
                    index_arg = "bodo.hiframes.api.get_series_index(data[{}])".format(
                        n_cols + 1 + i
                    )
                    break
    else:
        # ndarray case
        # checks for 2d and column args
        if not (isinstance(data, types.Array) and data.ndim == 2):
            raise ValueError(
                "pd.DataFrame() supports constant dictionary and" "ndarray input"
            )
        if is_overload_none(columns):
            raise ValueError(
                "pd.DataFrame() column argument is required when"
                "ndarray is passed as data"
            )
        if copy:
            astype_str += ".copy()"
        n_cols = len(columns.consts)
        data_arrs = ["data[:,{}]{}".format(i, astype_str) for i in range(n_cols)]
        data_dict = dict(zip(columns.consts, data_arrs))

    if is_overload_none(columns):
        col_names = data_dict.keys()
    else:
        col_names = columns.consts

    _fill_null_arrays(data_dict, col_names, index)

    data_args = ", ".join(data_dict[c] for c in col_names)
    col_args = ", ".join("'{}'".format(c) for c in col_names)
    return col_args, data_args, index_arg


def _fill_null_arrays(data_dict, col_names, index):
    """Fills data_dict with Null arrays if there are columns that are not
    available in data_dict.
    """
    # no null array needed
    if all(c in data_dict for c in col_names):
        return

    # get null array, needs index or an array available for length
    df_len = None
    for c in col_names:
        if c in data_dict:
            df_len = "len({})".format(data_dict[c])
            break

    if df_len is None and not is_overload_none(index):
        df_len = "len(index)"  # TODO: test

    assert df_len is not None, "empty dataframe with null arrays"  # TODO

    # TODO: object array with NaN (use StringArray?)
    null_arr = "np.full({}, np.nan)".format(df_len)
    for c in col_names:
        if c not in data_dict:
            data_dict[c] = null_arr

    return


@overload(len)  # TODO: avoid lowering?
def df_len_overload(df):
    if not isinstance(df, DataFrameType):
        return

    if len(df.columns) == 0:  # empty df
        return lambda df: 0
    return lambda df: len(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, 0))


@overload(operator.getitem)  # TODO: avoid lowering?
def df_getitem_overload(df, ind):
    if isinstance(df, DataFrameType) and isinstance(ind, types.StringLiteral):
        index = df.columns.index(ind.literal_value)
        return lambda df, ind: bodo.hiframes.pd_series_ext.init_series(
            get_dataframe_data(df, index), _get_dataframe_index(df), df._columns[index]
        )


@infer_global(operator.getitem)
class GetItemDataFrame(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        # df1 = df[df.A > .5]
        if (
            isinstance(df, DataFrameType)
            and isinstance(idx, (SeriesType, types.Array))
            and idx.dtype == types.bool_
        ):
            index = df.index
            if index is types.none or isinstance(
                index, bodo.hiframes.pd_index_ext.RangeIndexType
            ):
                index = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
            return signature(df.copy(has_parent=False, index=index), *args)


@infer
class StaticGetItemDataFrame(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        df, idx = args
        if (
            isinstance(df, DataFrameType)
            and isinstance(idx, list)
            and all(isinstance(c, str) for c in idx)
        ):
            data_typs = tuple(df.data[df.columns.index(c)] for c in idx)
            columns = tuple(idx)
            ret_typ = DataFrameType(data_typs, df.index, columns)
            return signature(ret_typ, *args)


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


# TODO: handle dataframe pass
# df.ia[] type
class DataFrameIatType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameIatType({})".format(df_type)
        super(DataFrameIatType, self).__init__(name)


# df.iloc[] type
class DataFrameILocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameILocType({})".format(df_type)
        super(DataFrameILocType, self).__init__(name)


# df.loc[] type
class DataFrameLocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameLocType({})".format(df_type)
        super(DataFrameLocType, self).__init__(name)


@infer
class StaticGetItemDataFrameIat(AbstractTemplate):
    key = "static_getitem"

    def generic(self, args, kws):
        df, idx = args
        # TODO: handle df.at[]
        if isinstance(df, DataFrameIatType):
            # df.iat[3,1]
            if (
                isinstance(idx, tuple)
                and len(idx) == 2
                and isinstance(idx[0], int)
                and isinstance(idx[1], int)
            ):
                col_no = idx[1]
                data_typ = df.df_type.data[col_no]
                return signature(data_typ.dtype, *args)


@infer_global(operator.getitem)
class GetItemDataFrameIat(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        # TODO: handle df.at[]
        if isinstance(df, DataFrameIatType):
            # df.iat[n,1]
            if (
                isinstance(idx, types.Tuple)
                and len(idx) == 2
                and isinstance(idx.types[1], types.IntegerLiteral)
            ):
                col_no = idx.types[1].literal_value
                data_typ = df.df_type.data[col_no]
                return signature(data_typ.dtype, *args)


@infer_global(operator.setitem)
class SetItemDataFrameIat(AbstractTemplate):
    key = operator.setitem

    def generic(self, args, kws):
        df, idx, val = args
        # TODO: handle df.at[]
        if isinstance(df, DataFrameIatType):
            # df.iat[n,1] = 3
            if (
                isinstance(idx, types.Tuple)
                and len(idx) == 2
                and isinstance(idx.types[1], types.IntegerLiteral)
            ):
                col_no = idx.types[1].literal_value
                data_typ = df.df_type.data[col_no]
                return signature(types.none, data_typ, idx.types[0], val)


@infer_global(operator.getitem)
class GetItemDataFrameLoc(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        # handling df.loc similar to df.iloc as temporary hack
        # TODO: handle proper labeled indexes
        if isinstance(df, DataFrameLocType):
            # df1 = df.loc[df.A > .5], df1 = df.loc[np.array([1,2,3])]
            if isinstance(idx, (SeriesType, types.Array, types.List)) and (
                idx.dtype == types.bool_ or isinstance(idx.dtype, types.Integer)
            ):
                return signature(df.df_type, *args)
            # df.loc[1:n]
            if isinstance(idx, types.SliceType):
                return signature(df.df_type, *args)
            # df.loc[1:n,'A']
            if (
                isinstance(idx, types.Tuple)
                and len(idx) == 2
                and isinstance(idx.types[1], types.StringLiteral)
            ):
                col_name = idx.types[1].literal_value
                col_no = df.df_type.columns.index(col_name)
                data_typ = df.df_type.data[col_no]
                # TODO: index
                ret_typ = SeriesType(data_typ.dtype, data_typ, None, bodo.string_type)
                return signature(ret_typ, *args)


@infer_global(operator.getitem)
class GetItemDataFrameILoc(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        df, idx = args
        if isinstance(df, DataFrameILocType):
            # df1 = df.iloc[df.A > .5], df1 = df.iloc[np.array([1,2,3])]
            if isinstance(idx, (SeriesType, types.Array, types.List)) and (
                idx.dtype == types.bool_ or isinstance(idx.dtype, types.Integer)
            ):
                return signature(df.df_type, *args)
            # df.iloc[1:n]
            if isinstance(idx, types.SliceType):
                return signature(df.df_type, *args)
            # df.iloc[1:n,0]
            if (
                isinstance(idx, types.Tuple)
                and len(idx) == 2
                and isinstance(idx.types[1], types.IntegerLiteral)
            ):
                col_no = idx.types[1].literal_value
                data_typ = df.df_type.data[col_no]
                # TODO: index
                ret_typ = SeriesType(data_typ.dtype, data_typ, None, bodo.string_type)
                return signature(ret_typ, *args)


@overload_method(DataFrameType, "merge")
@overload(pd.merge)
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
    comm_cols = tuple(set(left.columns) & set(right.columns))

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
            left_keys = get_const_str_list(left_on)
            # make sure all left_keys is a valid column in left
            validate_keys(left_keys, left.columns)
        if is_overload_true(right_index):
            right_keys = ["$_bodo_index_"]
        else:
            right_keys = get_const_str_list(right_on)
            # make sure all right_keys is a valid column in right
            validate_keys(right_keys, right.columns)

    validate_keys_length(
        left_on, right_on, left_index, right_index, left_keys, right_keys
    )
    validate_keys_dtypes(
        left, right, left_on, right_on, left_index, right_index, left_keys, right_keys
    )

    left_keys = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
        ", ".join("'{}'".format(c) for c in left_keys)
    )
    right_keys = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
        ", ".join("'{}'".format(c) for c in right_keys)
    )

    # generating code since typers can't find constants easily
    func_text = "def _impl(left, right, how='inner', on=None, left_on=None,\n"
    func_text += "    right_on=None, left_index=False, right_index=False, sort=False,\n"
    func_text += (
        "    suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):\n"
    )
    func_text += "  return bodo.hiframes.pd_dataframe_ext.join_dummy(left, right, {}, {}, '{}')\n".format(
        left_keys, right_keys, how
    )

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    # print(func_text)
    _impl = loc_vars["_impl"]
    return _impl

    # TODO: use regular implementation when constants work in typers
    # # merge on common columns if no key is provided
    # # TODO: use generic impl below when branch pruning is fixed
    # if (is_overload_none(on) and is_overload_none(left_on)
    #         and is_overload_none(right_on) and is_overload_false(left_index)
    #         and is_overload_false(right_index)):
    #     def _impl(left, right, how='inner', on=None, left_on=None,
    #             right_on=None, left_index=False, right_index=False, sort=False,
    #             suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):

    #         return bodo.hiframes.pd_dataframe_ext.join_dummy(
    #             left, right, comm_cols, comm_cols, how)

    #     return _impl

    # def _impl(left, right, how='inner', on=None, left_on=None,
    #         right_on=None, left_index=False, right_index=False, sort=False,
    #         suffixes=('_x', '_y'), copy=True, indicator=False, validate=None):
    #     if on is not None:
    #         left_on = right_on = on

    #     # if left_on is None and right_on is None and left_index == False and right_index == False:
    #     #     left_on = right_on = comm_cols

    #     return bodo.hiframes.pd_dataframe_ext.join_dummy(
    #         left, right, left_on, right_on, how)

    # return _impl


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
    # make sure left and right are dataframes
    if not isinstance(left, DataFrameType) or not isinstance(right, DataFrameType):
        raise BodoError("merge() requires dataframe inputs")
    # make sure how is of type str
    if not is_overload_constant_str(how):
        raise BodoError(
            "merge(): how parameter must be of type str, not "
            "{how}".format(how=type(how))
        )
    how = get_overload_const_str(how)
    # make sure how is one of ["left", "right", "outer", "inner"]
    if how not in ["left", "right", "outer", "inner"]:
        raise BodoError("merge(): invalid key '{}' for how".format(how))
    # make sure on is of type str or strlist
    if (
        (not is_overload_none(on))
        and (not is_overload_constant_str_list(on))
        and (not is_overload_constant_str(on))
    ):
        raise BodoError("merge(): on must be of type str or str list")
    # make sure left_on is of type str or strlist
    if (
        (not is_overload_none(left_on))
        and (not is_overload_constant_str_list(left_on))
        and (not is_overload_constant_str(left_on))
    ):
        raise BodoError("merge(): left_on must be of type str or str list")
    # make sure right_on is of type str or strlist
    if (
        (not is_overload_none(right_on))
        and (not is_overload_constant_str_list(right_on))
        and (not is_overload_constant_str(right_on))
    ):
        raise BodoError("merge(): right_on must be of type str or str list")
    # make sure leftindex is of type bool
    if not is_overload_constant_bool(left_index):
        raise BodoError(
            "merge(): left_index parameter must be of type bool, not "
            "{left_index}".format(left_index=type(left_index))
        )
    # make sure rightindex is of type bool
    if not is_overload_constant_bool(right_index):
        raise BodoError(
            "merge(): right_index parameter must be of type bool, not "
            "{right_index}".format(right_index=type(right_index))
        )
    # make sure sort is the default value, sort=True not supported
    if not is_overload_false(sort):
        raise BodoError("merge(): sort parameter only supports default value False")
    # make sure suffixes is not passed in
    if suffixes != ("_x", "_y"):
        raise BodoError(
            "merge(): suffixes parameter cannot be specified. "
            "Default value is ('_x', '_y')"
        )
    # make sure copy is the default value, copy=False not supported
    if not is_overload_true(copy):
        raise BodoError("merge(): copy parameter only supports default value True")
    # make sure copy is the default value, copy=False not supported
    if not is_overload_false(indicator):
        raise BodoError(
            "merge(): indicator parameter only supports default value False"
        )
    # make sure validate is None
    if not is_overload_none(validate):
        raise BodoError("merge(): validate parameter only supports default value None")

    comm_cols = tuple(set(left.columns) & set(right.columns))
    if not is_overload_none(on):
        # make sure two dataframes have common columns
        if len(comm_cols) == 0:
            raise BodoError(
                "merge(): No common columns to perform merge on. "
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
                'merge(): Can only pass argument "on" OR "left_on" '
                'and "right_on", not a combination of both.'
            )

    # make sure right_on, right_index, left_on, left_index are speciefied properly
    if (
        (is_overload_true(left_index) or not is_overload_none(left_on))
        and is_overload_none(right_on)
        and not is_overload_true(right_index)
    ):
        raise BodoError("merge(): Must pass right_on or right_index=True")
    if (
        (is_overload_true(right_index) or not is_overload_none(right_on))
        and is_overload_none(left_on)
        and not is_overload_true(left_index)
    ):
        raise BodoError("merge(): Must pass left_on or left_index=True")


def validate_keys_length(
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
        # error checking after specifying left_on and right_index = True is supported
        # if len(left_keys) != 1:
        #     raise BodoError(
        #         "merge(): len(left_on) must equal the number "
        #         'of levels in the index of "right", which is 1'
        #     )
    if not is_overload_none(right_on) and is_overload_true(left_index):
        raise BodoError(
            "merge(): left_index = True and specifying right_on is not suppported yet."
        )
        # error checking after specifying right_on and left_index = True is supported
        # if len(right_keys) != 1:
        #     raise BodoError(
        #         "merge(): len(right_on) must equal the number "
        #         'of levels in the index of "left", which is 1'
        #     )


def validate_keys_dtypes(
    left, right, left_on, right_on, left_index, right_index, left_keys, right_keys
):
    # make sure left keys and right keys have comparable dtypes

    typing_context = numba.targets.registry.cpu_target.typing_context

    if is_overload_true(left_index) or is_overload_true(right_index):
        # cases where index is used in merging
        if is_overload_true(left_index) and is_overload_true(right_index):
            lk_type = left.index.dtype
            rk_type = right.index.dtype
        elif is_overload_true(left_index):
            lk_type = left.index.dtype
            rk_type = right.data[right.columns.index(right_keys[0])].dtype
        elif is_overload_true(right_index):
            lk_type = left.data[left.columns.index(left_keys[0])].dtype
            rk_type = right.index.dtype

        try:
            ret_dtype = typing_context.resolve_function_type(
                operator.eq, (lk_type, rk_type), {}
            )
        except:
            raise BodoError(
                "merge: You are trying to merge on {lk_dtype} and "
                "{rk_dtype} columns. If you wish to proceed "
                "you should use pd.concat".format(lk_dtype=lk_type, rk_dtype=rk_type)
            )
    else:  # cases where only columns are used in merge
        for lk, rk in zip(left_keys, right_keys):
            lk_type = left.data[left.columns.index(lk)].dtype
            rk_type = right.data[right.columns.index(rk)].dtype

            try:
                ret_dtype = typing_context.resolve_function_type(
                    operator.eq, (lk_type, rk_type), {}
                )
            except:
                raise BodoError(
                    "merge: You are trying to merge on column {lk} of {lk_dtype} and "
                    "column {rk} of {rk_dtype}. If you wish to proceed "
                    "you should use pd.concat".format(
                        lk=lk, lk_dtype=lk_type, rk=rk, rk_dtype=rk_type
                    )
                )


def validate_keys(keys, columns):
    if len(set(keys).difference(set(columns))) > 0:
        raise BodoError(
            "merge(): invalid key {} for on/left_on/right_on".format(
                set(keys).difference(set(columns))
            )
        )


@overload_method(DataFrameType, "join")
def join_overload(left, other, on=None, how="left", lsuffix="", rsuffix="", sort=False):

    # make sure left and right are dataframes
    if not isinstance(left, DataFrameType) or not isinstance(other, DataFrameType):
        raise TypeError("join() requires dataframe inputs")

    if not is_overload_none(on):
        left_keys = get_const_str_list(on)
    else:
        left_keys = ["$_bodo_index_"]

    right_keys = ["$_bodo_index_"]

    left_keys = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
        ", ".join("'{}'".format(c) for c in left_keys)
    )
    right_keys = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
        ", ".join("'{}'".format(c) for c in right_keys)
    )

    # generating code since typers can't find constants easily
    func_text = "def _impl(left, other, on=None, how='left',\n"
    func_text += "    lsuffix='', rsuffix='', sort=False):\n"
    func_text += "  return bodo.hiframes.pd_dataframe_ext.join_dummy(left, other, {}, {}, '{}')\n".format(
        left_keys, right_keys, how
    )

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    # print(func_text)
    _impl = loc_vars["_impl"]
    return _impl


# a dummy join function that will be replace in dataframe_pass
def join_dummy(left_df, right_df, left_on, right_on, how):
    return left_df


@infer_global(join_dummy)
class JoinTyper(AbstractTemplate):
    def generic(self, args, kws):
        from bodo.hiframes.pd_dataframe_ext import DataFrameType
        from bodo.utils.typing import is_overload_str

        assert not kws
        left_df, right_df, left_on, right_on, how = args

        # columns with common name that are not common keys will get a suffix
        comm_keys = set(left_on.consts) & set(right_on.consts)
        comm_data = set(left_df.columns) & set(right_df.columns)
        add_suffix = comm_data - comm_keys

        columns = [(c + "_x" if c in add_suffix else c) for c in left_df.columns]
        # common keys are added only once so avoid adding them
        columns += [
            (c + "_y" if c in add_suffix else c)
            for c in right_df.columns
            if c not in comm_keys
        ]
        data = list(left_df.data)
        data += [
            right_df.data[right_df.columns.index(c)]
            for c in right_df.columns
            if c not in comm_keys
        ]

        # TODO: unify left/right indices if necessary (e.g. RangeIndex/Int64)
        index_typ = types.none
        left_index = "$_bodo_index_" in left_on.consts
        right_index = "$_bodo_index_" in right_on.consts
        if left_index and right_index and not is_overload_str(how, "asof"):
            index_typ = left_df.index
            if isinstance(index_typ, bodo.hiframes.pd_index_ext.RangeIndexType):
                index_typ = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
        elif right_index and is_overload_str(how, "left"):
            index_typ = left_df.index
        elif left_index and is_overload_str(how, "right"):
            index_typ = right_df.index

        out_df = DataFrameType(tuple(data), index_typ, tuple(columns))
        return signature(out_df, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(join_dummy, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return dataframe._getvalue()


@overload(pd.merge_asof)
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

    # TODO: support 'by' argument

    # XXX copied from merge, TODO: refactor
    # make sure left and right are dataframes
    if not isinstance(left, DataFrameType) or not isinstance(right, DataFrameType):
        raise TypeError("merge_asof() requires dataframe inputs")

    comm_cols = tuple(set(left.columns) & set(right.columns))

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
            left_keys = get_const_str_list(left_on)
        if is_overload_true(right_index):
            right_keys = ["$_bodo_index_"]
        else:
            right_keys = get_const_str_list(right_on)

    left_keys = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
        ", ".join("'{}'".format(c) for c in left_keys)
    )
    right_keys = "bodo.utils.typing.add_consts_to_type([{0}], {0})".format(
        ", ".join("'{}'".format(c) for c in right_keys)
    )

    # generating code since typers can't find constants easily
    func_text = "def _impl(left, right, on=None, left_on=None, right_on=None,\n"
    func_text += "    left_index=False, right_index=False, by=None, left_by=None,\n"
    func_text += "    right_by=None, suffixes=('_x', '_y'), tolerance=None,\n"
    func_text += "    allow_exact_matches=True, direction='backward'):\n"
    func_text += "  return bodo.hiframes.pd_dataframe_ext.join_dummy(left, right, {}, {}, 'asof')\n".format(
        left_keys, right_keys
    )

    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    # print(func_text)
    _impl = loc_vars["_impl"]
    return _impl

    # def _impl(left, right, on=None, left_on=None, right_on=None,
    #         left_index=False, right_index=False, by=None, left_by=None,
    #         right_by=None, suffixes=('_x', '_y'), tolerance=None,
    #         allow_exact_matches=True, direction='backward'):
    #     if on is not None:
    #         left_on = right_on = on

    #     return bodo.hiframes.pd_dataframe_ext.join_dummy(
    #         left, right, left_on, right_on, 'asof')

    # return _impl


@overload_method(DataFrameType, "pivot_table")
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
    ):

        return bodo.hiframes.pd_groupby_ext.pivot_table_dummy(
            df, values, index, columns, aggfunc, _pivot_values
        )

    return _impl


@overload(pd.crosstab)
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
    ):
        return bodo.hiframes.pd_groupby_ext.crosstab_dummy(
            index, columns, _pivot_values
        )

    return _impl


@overload(pd.concat)
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


def concat_dummy(objs):
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
            ret_typ = objs.dtype.copy(index=types.none)
            if isinstance(ret_typ, DataFrameType):
                ret_typ = ret_typ.copy(has_parent=False, index=types.none)
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

            ret_typ = DataFrameType(tuple(data), None, tuple(names))
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
            all_colnames = sorted(set(all_colnames))

            # get output data types
            all_data = []
            for cname in all_colnames:
                # arguments to the generated function
                arr_args = [
                    df.data[df.columns.index(cname)]
                    for df in objs.types
                    if cname in df.columns
                ]
                # XXX we add arrays of float64 NaNs if a column is missing
                # so add a dummy array of float64 for accurate typing
                # e.g. int to float conversion
                # TODO: fix NA column additions for other types
                if len(arr_args) < len(objs.types):
                    arr_args.append(types.Array(types.float64, 1, "C"))
                # use bodo.libs.array_kernels.concat() typer
                concat_typ = self.context.resolve_function_type(
                    bodo.libs.array_kernels.concat, (types.Tuple(arr_args),), {}).return_type
                all_data.append(concat_typ)

            ret_typ = DataFrameType(tuple(all_data), None, tuple(all_colnames))
            return signature(ret_typ, *args)

        # series case
        elif isinstance(objs.types[0], SeriesType):
            assert all(isinstance(t, SeriesType) for t in objs.types)
            arr_args = [S.data for S in objs.types]
            concat_typ = self.context.resolve_function_type(
                    bodo.libs.array_kernels.concat, (types.Tuple(arr_args),), {}).return_type
            ret_typ = SeriesType(concat_typ.dtype, concat_typ)
            return signature(ret_typ, *args)
        # TODO: handle other iterables like arrays, lists, ...


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(concat_dummy, types.VarArg(types.Any))
def lower_concat_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "sort_values")
def sort_values_overload(
    df, by, axis=0, ascending=True, inplace=False, kind="quicksort", na_position="last"
):
    def _impl(
        df,
        by,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
    ):

        return bodo.hiframes.pd_dataframe_ext.sort_values_dummy(
            df, by, ascending, inplace
        )

    return _impl


def sort_values_dummy(df, by, ascending, inplace):
    return df.sort_values(by, ascending=ascending, inplace=inplace)


@infer_global(sort_values_dummy)
class SortDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, by, ascending, inplace = args

        # inplace value
        if isinstance(inplace, bodo.utils.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        index = df.index
        if index is types.none or isinstance(
            index, bodo.hiframes.pd_index_ext.RangeIndexType
        ):
            index = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)
        ret_typ = df.copy(index=index, has_parent=False)
        # TODO: handle cases where untyped pass inplace replacement is not
        # possible and none should be returned
        # if inplace:
        #     ret_typ = types.none
        return signature(ret_typ, *args)


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(sort_values_dummy, types.VarArg(types.Any))
def lower_sort_values_dummy(context, builder, sig, args):
    if sig.return_type == types.none:
        return

    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "sort_index")
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
    ):

        return bodo.hiframes.pd_dataframe_ext.sort_values_dummy(
            df, "$_bodo_index_", ascending, inplace
        )

    return _impl


# dummy function to change the df type to have set_parent=True
# used in sort_values(inplace=True) hack
def set_parent_dummy(df):
    return df


@infer_global(set_parent_dummy)
class ParentDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, = args
        ret = DataFrameType(df.data, df.index, df.columns, True)
        return signature(ret, *args)


@lower_builtin(set_parent_dummy, types.VarArg(types.Any))
def lower_set_parent_dummy(context, builder, sig, args):
    return impl_ret_borrowed(context, builder, sig.return_type, args[0])


# TODO: jitoptions for overload_method and infer_global
# (no_cpython_wrapper to avoid error for iterator object)
@overload_method(DataFrameType, "itertuples")
def itertuples_overload(df, index=True, name="Pandas"):
    def _impl(df, index=True, name="Pandas"):
        return bodo.hiframes.pd_dataframe_ext.itertuples_dummy(df)

    return _impl


def itertuples_dummy(df):
    return df


@infer_global(itertuples_dummy)
class ItertuplesDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df, = args
        # XXX index handling, assuming implicit index
        assert "Index" not in df.columns
        columns = ("Index",) + df.columns
        arr_types = (types.Array(types.int64, 1, "C"),) + df.data
        iter_typ = bodo.hiframes.dataframe_impl.DataFrameTupleIterator(columns, arr_types)
        return signature(iter_typ, *args)


@lower_builtin(itertuples_dummy, types.VarArg(types.Any))
def lower_itertuples_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "fillna")
def fillna_overload(
    df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None
):
    # TODO: handle possible **kwargs options?

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent that has a string column (reflection)
    def _impl(
        df, value=None, method=None, axis=None, inplace=False, limit=None, downcast=None
    ):
        return bodo.hiframes.pd_dataframe_ext.fillna_dummy(df, value, inplace)

    return _impl


def fillna_dummy(df, n):
    return df


@infer_global(fillna_dummy)
class FillnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, value, inplace = args
        # inplace value
        if isinstance(inplace, bodo.utils.utils.BooleanLiteral):
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
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "reset_index")
def reset_index_overload(
    df, level=None, drop=False, inplace=False, col_level=0, col_fill=""
):

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(df, level=None, drop=False, inplace=False, col_level=0, col_fill=""):
        return bodo.hiframes.pd_dataframe_ext.reset_index_dummy(df, inplace)

    return _impl


def reset_index_dummy(df, n):
    return df


@infer_global(reset_index_dummy)
class ResetIndexDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, inplace = args
        # inplace value
        if isinstance(inplace, bodo.utils.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        if not inplace:
            out_df = DataFrameType(df.data, None, df.columns)
            return signature(out_df, *args)
        return signature(types.none, *args)


@lower_builtin(reset_index_dummy, types.VarArg(types.Any))
def lower_reset_index_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "dropna")
def dropna_overload(df, axis=0, how="any", thresh=None, subset=None, inplace=False):

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(df, axis=0, how="any", thresh=None, subset=None, inplace=False):
        return bodo.hiframes.pd_dataframe_ext.dropna_dummy(df, inplace)

    return _impl


def dropna_dummy(df, n):
    return df


@infer_global(dropna_dummy)
class DropnaDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, inplace = args
        # inplace value
        if isinstance(inplace, bodo.utils.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        if not inplace:
            # copy type to set has_parent False
            # TODO: support Index
            out_df = DataFrameType(df.data, types.none, df.columns)
            return signature(out_df, *args)
        return signature(types.none, *args)


@lower_builtin(dropna_dummy, types.VarArg(types.Any))
def lower_dropna_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "drop")
def drop_overload(
    df,
    labels=None,
    axis=0,
    index=None,
    columns=None,
    level=None,
    inplace=False,
    errors="raise",
):

    # TODO: avoid dummy and generate func here when inlining is possible
    # TODO: inplace of df with parent (reflection)
    def _impl(
        df,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        return bodo.hiframes.pd_dataframe_ext.drop_dummy(
            df, labels, axis, columns, inplace
        )

    return _impl


def drop_dummy(df, labels, axis, columns, inplace):
    return df


@infer_global(drop_dummy)
class DropDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        df, labels, axis, columns, inplace = args

        if labels != types.none:
            if (
                not isinstance(axis, types.IntegerLiteral)
                or not axis.literal_value == 1
            ):
                raise ValueError("only axis=1 supported for df.drop()")
            if isinstance(labels, types.StringLiteral):
                drop_cols = (labels.literal_value,)
            elif hasattr(labels, "consts"):
                drop_cols = labels.consts
            else:
                raise ValueError(
                    "constant list of columns expected for labels in df.drop()"
                )
        else:
            assert columns != types.none
            if isinstance(columns, types.StringLiteral):
                drop_cols = (columns.literal_value,)
            elif hasattr(columns, "consts"):
                drop_cols = columns.consts
            else:
                raise ValueError(
                    "constant list of columns expected for labels in df.drop()"
                )

        assert all(c in df.columns for c in drop_cols)
        new_cols = tuple(c for c in df.columns if c not in drop_cols)
        new_data = tuple(df.data[df.columns.index(c)] for c in new_cols)

        # inplace value
        if isinstance(inplace, bodo.utils.utils.BooleanLiteral):
            inplace = inplace.literal_value
        else:
            # XXX inplace type is just bool when value not passed. Therefore,
            # we assume the default False value.
            # TODO: more robust fix or just check
            inplace = False

        # TODO: reflection
        has_parent = False  # df.has_parent
        # if not inplace:
        #     has_parent = False  # data is copied

        out_df = DataFrameType(new_data, df.index, new_cols, has_parent)
        return signature(out_df, *args)


@lower_builtin(drop_dummy, types.VarArg(types.Any))
def lower_drop_dummy(context, builder, sig, args):
    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


@overload_method(DataFrameType, "append")
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


# TODO: other Pandas versions (0.24 defaults are different than 0.23)
@overload_method(DataFrameType, "to_csv")
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
        ):
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
    ):
        with numba.objmode:
            df.to_csv(
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

    return _impl
