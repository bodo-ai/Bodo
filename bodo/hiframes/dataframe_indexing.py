# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Indexing support for pd.DataFrame type.
"""
import operator

import numpy as np
import pandas as pd
from numba.core import cgutils, types
from numba.core.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    bound_function,
    infer_global,
    signature,
)
from numba.extending import (
    infer,
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    overload_method,
    register_model,
    type_callable,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import DataFrameType
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    get_index_data_arr_types,
    get_index_names,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    get_overload_const_tuple,
    is_list_like_index_type,
    is_overload_bool,
    is_overload_bool_list,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_constant_tuple,
    is_overload_false,
    is_overload_none,
    is_overload_true,
    is_overload_zero,
    raise_bodo_error,
    raise_const_error,
)


# DataFrame getitem
@overload(operator.getitem, no_unliteral=True)
def df_getitem_overload(df, ind):
    if not isinstance(df, DataFrameType):
        return

    # A = df["column"]
    if is_overload_constant_str(ind) or is_overload_constant_int(ind):
        ind_val = (
            get_overload_const_str(ind)
            if is_overload_constant_str(ind)
            else get_overload_const_int(ind)
        )
        # df with multi-level column names returns a lower level dataframe
        if isinstance(df.columns[0], tuple):
            new_names = []
            new_data = []
            for i, v in enumerate(df.columns):
                if v[0] != ind_val:
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
        if ind_val not in df.columns:
            raise_bodo_error(
                "dataframe {} does not include column {}".format(df, ind_val)
            )
        col_no = df.columns.index(ind_val)
        return lambda df, ind: bodo.hiframes.pd_series_ext.init_series(
            bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_no),
            bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df),
            ind_val,
        )  # pragma: no cover

    # A = df[["C1", "C2"]]
    # TODO: support int names
    if is_overload_constant_list(ind):
        ind_columns = get_overload_const_list(ind)
        # error checking, TODO: test
        for c in ind_columns:
            if c not in df.columns:
                raise_bodo_error(
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
    raise_bodo_error(
        "df[] getitem using {} not supported".format(ind)
    )  # pragma: no cover


# DataFrame setitem
@overload(operator.setitem, no_unliteral=True)
def df_setitem_overload(df, idx, val):
    if not isinstance(df, DataFrameType):
        return

    # df["B"] = A
    # handle in typing pass since the dataframe type can change
    # TODO: better error checking here
    raise_bodo_error("DataFrame setitem: transform necessary")


##################################  df.iloc  ##################################


# df.iloc[] type
class DataFrameILocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameILocType({})".format(df_type)
        super(DataFrameILocType, self).__init__(name)

    ndim = 2


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
        context.nrt.incref(builder, signature.args[0], obj_val)

        return iloc_val._getvalue()

    return DataFrameILocType(obj)(obj), codegen


@overload_attribute(DataFrameType, "iloc")
def overload_series_iloc(s):
    return lambda s: bodo.hiframes.dataframe_indexing.init_dataframe_iloc(s)


# df.iloc[] getitem
@overload(operator.getitem, no_unliteral=True)
def overload_iloc_getitem(I, idx):
    if not isinstance(I, DataFrameILocType):
        return

    df = I.df_type

    # Integer case returns Series(object) which is not supported
    # TODO: error checking test
    if isinstance(types.unliteral(idx), types.Integer):  # pragma: no cover
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
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and is_overload_constant_int(idx.types[1])
    ):
        # create Series from column data and reuse Series.iloc[]
        col_ind = get_overload_const_int(idx.types[1])
        col_name = df.columns[col_ind]

        def impl_col_ind(I, idx):
            df = I._obj
            index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            return bodo.hiframes.pd_series_ext.init_series(data, index, col_name).iloc[
                idx[0]
            ]

        return impl_col_ind

    # df.iloc[:,1:] case requires typing pass transform since slice info not available
    # here. TODO: refactor when SliceLiteral of Numba has all the info.
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx.types) == 2
        and isinstance(idx.types[1], types.SliceType)
    ):
        raise_bodo_error("Invalid df.iloc[] getitem using (slice, slice)")

    # TODO: error-checking test
    raise BodoError(
        "df.iloc[] getitem using {} not supported".format(idx)
    )  # pragma: no cover


##################################  df.loc  ##################################


# df.loc[] type
class DataFrameLocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameLocType({})".format(df_type)
        super(DataFrameLocType, self).__init__(name)

    ndim = 2


@register_model(DataFrameLocType)
class DataFrameLocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.df_type)]
        super(DataFrameLocModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameLocType, "obj", "_obj")


@intrinsic
def init_dataframe_loc(typingctx, obj=None):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        loc_type = signature.return_type

        loc_val = cgutils.create_struct_proxy(loc_type)(context, builder)
        loc_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return loc_val._getvalue()

    return DataFrameLocType(obj)(obj), codegen


@overload_attribute(DataFrameType, "loc")
def overload_series_loc(s):
    return lambda s: bodo.hiframes.dataframe_indexing.init_dataframe_loc(s)


# df.loc[] getitem
@overload(operator.getitem, no_unliteral=True)
def overload_loc_getitem(I, idx):
    if not isinstance(I, DataFrameLocType):
        return

    df = I.df_type

    # df.loc[idx] with array of bools
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        # TODO: refactor with df filter
        func_text = "def impl(I, idx):\n"
        func_text += "  df = I._obj\n"
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

    # df.loc[idx, "A"]
    if isinstance(idx, types.BaseTuple) and len(idx) == 2:
        col_idx = idx.types[1]

        # df.loc[idx, "A"]
        if is_overload_constant_str(col_idx):
            # TODO: support non-str dataframe names
            # TODO: error checking
            # create Series from column data and reuse Series.loc[]
            col_name = get_overload_const_str(col_idx)
            col_ind = df.columns.index(col_name)

            def impl_col_name(I, idx):
                df = I._obj
                index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)
                data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
                return bodo.hiframes.pd_series_ext.init_series(
                    data, index, col_name
                ).loc[idx[0]]

            return impl_col_name

        # df.loc[idx, ["A", "B"]] or df.loc[idx, [True, False, True]]
        if is_overload_constant_list(col_idx):
            # get column list (could be list of strings or bools)
            col_idx_list = get_overload_const_list(col_idx)
            return gen_df_loc_col_select_impl(df, col_idx_list)

    # TODO: error-checking test
    raise_bodo_error(
        "DataFrame.loc[] getitem (location-based indexing) using {} not supported yet.".format(
            idx
        )
    )  # pragma: no cover


def gen_df_loc_col_select_impl(df, col_idx_list):
    """generate implementation for cases like df.loc[:, ["A", "B"]] and
    df.loc[:, [True, False, True]]
    """
    # get column names if bool list
    if len(col_idx_list) > 0 and isinstance(col_idx_list[0], (bool, np.bool_)):
        col_idx_list = list(np.array(df.columns)[col_idx_list])

    # create a new dataframe, create new data/index using idx
    new_data = ", ".join(
        "bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {})[idx[0]]".format(
            df.columns.index(c)
        )
        for c in col_idx_list
    )
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx[0]]"
    func_text = "def impl(I, idx):\n"
    func_text += "  df = I._obj\n"
    return bodo.hiframes.dataframe_impl._gen_init_df(
        func_text, col_idx_list, new_data, index
    )


##################################  df.iat  ##################################


# df.ia[] type
class DataFrameIatType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameIatType({})".format(df_type)
        super(DataFrameIatType, self).__init__(name)

    ndim = 2


@register_model(DataFrameIatType)
class DataFrameIatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.df_type)]
        super(DataFrameIatModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameIatType, "obj", "_obj")


@intrinsic
def init_dataframe_iat(typingctx, obj=None):
    def codegen(context, builder, signature, args):
        (obj_val,) = args
        iat_type = signature.return_type

        iat_val = cgutils.create_struct_proxy(iat_type)(context, builder)
        iat_val.obj = obj_val

        # increase refcount of stored values
        context.nrt.incref(builder, signature.args[0], obj_val)

        return iat_val._getvalue()

    return DataFrameIatType(obj)(obj), codegen


@overload_attribute(DataFrameType, "iat")
def overload_series_iat(s):
    return lambda s: bodo.hiframes.dataframe_indexing.init_dataframe_iat(s)


# df.iat[] getitem
@overload(operator.getitem, no_unliteral=True)
def overload_iat_getitem(I, idx):
    if not isinstance(I, DataFrameIatType):
        return

    df = I.df_type

    # df.iat[1,0]
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and is_overload_constant_int(idx.types[1])
    ):
        col_ind = get_overload_const_int(idx.types[1])

        def impl_col_ind(I, idx):
            df = I._obj
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            return data[idx[0]]

        return impl_col_ind

    # TODO: error-checking test
    raise BodoError(
        "df.iat[] getitem using {} not supported".format(idx)
    )  # pragma: no cover


# df.iat[] setitem
@overload(operator.setitem, no_unliteral=True)
def overload_iat_setitem(I, idx, val):
    if not isinstance(I, DataFrameIatType):
        return

    df = I.df_type

    # df.iat[1,0]
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and is_overload_constant_int(idx.types[1])
    ):
        col_ind = get_overload_const_int(idx.types[1])

        def impl_col_ind(I, idx, val):
            df = I._obj
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            data[idx[0]] = val

        return impl_col_ind

    # TODO: error-checking test
    raise BodoError(
        "df.iat[] setitem using {} not supported".format(idx)
    )  # pragma: no cover
