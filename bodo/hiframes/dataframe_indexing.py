# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Indexing support for pd.DataFrame type.
"""
import operator
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
from bodo.hiframes.pd_index_ext import RangeIndexType
from bodo.utils.typing import (
    BodoWarning,
    BodoError,
    is_overload_none,
    is_overload_constant_bool,
    is_overload_bool,
    is_overload_constant_str,
    is_overload_constant_str_list,
    is_overload_true,
    is_overload_false,
    is_overload_zero,
    is_overload_constant_int,
    get_overload_const_str,
    get_const_str_list,
    get_overload_const_int,
    is_overload_bool_list,
    get_index_names,
    get_index_data_arr_types,
    raise_const_error,
    is_overload_constant_tuple,
    get_overload_const_tuple,
    is_list_like_index_type,
)
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.hiframes.pd_dataframe_ext import DataFrameType


# DataFrame getitem
@overload(operator.getitem)
def df_getitem_overload(df, ind):
    if not isinstance(df, DataFrameType):
        return

    # A = df["column"]
    if is_overload_constant_str(ind):
        ind_str = get_overload_const_str(ind)
        # df with multi-level column names returns a lower level dataframe
        if isinstance(df.columns[0], tuple):
            new_names = []
            new_data = []
            for i, v in enumerate(df.columns):
                if v[0] != ind_str:
                    continue
                # output names are str in 2 level case, not tuple
                # TODO: test more than 2 levels
                new_names.append(v[1] if len(v) == 2 else v[1:])
                new_data.append(
                    "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})".format(
                        i
                    )
                )
            func_text = "def impl(df, ind):\n"
            index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"
            return bodo.hiframes.dataframe_impl._gen_init_df(
                func_text, new_names, ", ".join(new_data), index
            )

        # regular single level case
        if ind_str not in df.columns:
            raise BodoError(
                "dataframe {} does not include column {}".format(df, ind_str)
            )
        col_no = df.columns.index(ind_str)
        return lambda df, ind: bodo.hiframes.pd_series_ext.init_series(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_no),
            bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df),
            ind_str,
        )

    # A = df[["C1", "C2"]]
    if is_overload_constant_str_list(ind):
        ind_columns = get_const_str_list(ind)
        # error checking, TODO: test
        for c in ind_columns:
            if c not in df.columns:
                raise BodoError(
                    "Column {} not found in dataframe columns {}".format(c, df.columns)
                )
        new_data = ", ".join(
            "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {}).copy()".format(
                df.columns.index(c)
            )
            for c in ind_columns
        )
        func_text = "def impl(df, ind):\n"
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text, ind_columns, new_data, index
        )

    # df1 = df[df.A > .5]
    if is_list_like_index_type(ind) and ind.dtype == types.bool_:
        # implement using array filtering (not using the old Filter node)
        # TODO: create an IR node for enforcing same dist for all columns and ind array
        func_text = "def impl(df, ind):\n"
        func_text += "  idx = bodo.utils.conversion.coerce_to_ndarray(ind)\n"
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx]"
        new_data = ", ".join(
            "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})[idx]".format(
                df.columns.index(c)
            )
            for c in df.columns
        )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text, df.columns, new_data, index
        )

    # TODO: error-checking test
    raise BodoError(
        "df[] getitem using {} not supported".format(ind)
    )  # pragma: no cover


##################################  df.iloc  ##################################


# df.iloc[] type
class DataFrameILocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameILocType({})".format(df_type)
        super(DataFrameILocType, self).__init__(name)


@register_model(DataFrameILocType)
class DataFrameILocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.df_type)]
        super(DataFrameILocModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameILocType, "obj", "_obj")


@intrinsic
def init_dataframe_iloc(typingctx, obj=None):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        iloc_type = signature.return_type

        iloc_val = cgutils.create_struct_proxy(iloc_type)(context, builder)
        iloc_val.obj = obj_val

        # increase refcount of stored values
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return iloc_val._getvalue()

    return DataFrameILocType(obj)(obj), codegen


@overload_attribute(DataFrameType, "iloc")
def overload_series_iloc(s):
    return lambda s: bodo.hiframes.dataframe_indexing.init_dataframe_iloc(s)


# df.iloc[] getitem
@overload(operator.getitem)
def overload_iloc_getitem(I, idx):
    if not isinstance(I, DataFrameILocType):
        return

    df = I.df_type

    # Integer case returns Series(object) which is not supported
    # TODO: error checking test
    if isinstance(idx, types.Integer):  # pragma: no cover
        # TODO: support cases that can be typed, e.g. all float64
        # TODO: return namedtuple instead of Series?
        raise BodoError(
            "df.iloc[] with integer index is not supported since output Series cannot be typed"
        )

    # df.iloc[idx]
    # array of bools/ints, or slice
    if (
        is_list_like_index_type(idx)
        and isinstance(idx.dtype, (types.Integer, types.Boolean))
    ) or isinstance(idx, types.SliceType):
        # TODO: refactor with df filter
        func_text = "def impl(I, idx):\n"
        func_text += "  df = I._obj\n"
        if isinstance(idx, types.SliceType):
            func_text += "  idx_t = idx\n"
        else:
            func_text += "  idx_t = bodo.utils.conversion.coerce_to_ndarray(idx)\n"
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx_t]"
        new_data = ", ".join(
            "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})[idx_t]".format(
                df.columns.index(c)
            )
            for c in df.columns
        )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text, df.columns, new_data, index
        )

    # df.iloc[1:n,0], df.iloc[1,0]
    if isinstance(idx, types.BaseTuple) and len(idx) == 2 and is_overload_constant_int(idx.types[1]):
        # create Series from column data and reuse Series.iloc[]
        col_ind = get_overload_const_int(idx.types[1])
        col_name = df.columns[col_ind]
        def impl_col_ind(I, idx):
            df = I._obj
            index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            return bodo.hiframes.pd_series_ext.init_series(data, index, col_name).iloc[idx[0]]

        return impl_col_ind

    # TODO: error-checking test
    raise BodoError(
        "df.iloc[] getitem using {} not supported".format(idx)
    )  # pragma: no cover


# TODO: handle dataframe pass
# df.ia[] type
class DataFrameIatType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameIatType({})".format(df_type)
        super(DataFrameIatType, self).__init__(name)


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
                isinstance(idx, types.BaseTuple)
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
                isinstance(idx, types.BaseTuple)
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
            if isinstance(
                idx, (SeriesType, types.Array, types.List, BooleanArrayType)
            ) and (idx.dtype == types.bool_ or isinstance(idx.dtype, types.Integer)):
                return signature(df.df_type, *args)
            # df.loc[1:n]
            if isinstance(idx, types.SliceType):
                return signature(df.df_type, *args)
            # df.loc[1:n,"A"]
            if (
                isinstance(idx, types.BaseTuple)
                and len(idx) == 2
                and isinstance(idx.types[1], types.StringLiteral)
            ):
                col_name = idx.types[1].literal_value
                col_no = df.df_type.columns.index(col_name)
                data_typ = df.df_type.data[col_no]
                # TODO: index
                ret_typ = SeriesType(data_typ.dtype, data_typ, None, bodo.string_type)
                return signature(ret_typ, *args)
