# Copyright (C) 2019 Bodo Inc. All rights reserved.
"""
Indexing support for pd.DataFrame type.
"""
import operator
from numba.core import types, cgutils
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
from numba.core.typing.templates import (
    infer_global,
    AbstractTemplate,
    signature,
    AttributeTemplate,
    bound_function,
)
import bodo
from bodo.utils.typing import (
    BodoWarning,
    BodoError,
    raise_bodo_error,
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
from bodo.hiframes.pd_dataframe_ext import DataFrameType


# DataFrame getitem
@overload(operator.getitem)
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
    raise_bodo_error(
        "df[] getitem using {} not supported".format(ind)
    )  # pragma: no cover


# DataFrame setitem
@overload(operator.setitem)
def df_setitem_overload(df, idx, val):
    if not isinstance(df, DataFrameType):
        return

    # df["B"] = A
    # handle in typing pass since the dataframe type can change
    # TODO: better error checking here
    bodo.transforms.typing_pass.typing_transform_required = True
    raise Exception("DataFrame setitem: transform necessary")


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
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return loc_val._getvalue()

    return DataFrameLocType(obj)(obj), codegen


@overload_attribute(DataFrameType, "loc")
def overload_series_loc(s):
    return lambda s: bodo.hiframes.dataframe_indexing.init_dataframe_loc(s)


# df.loc[] getitem
@overload(operator.getitem)
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
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and is_overload_constant_str(idx.types[1])
    ):
        # TODO: support non-str dataframe names
        # TODO: error checking
        # create Series from column data and reuse Series.loc[]
        col_name = get_overload_const_str(idx.types[1])
        col_ind = df.columns.index(col_name)

        def impl_col_name(I, idx):
            df = I._obj
            index = bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            return bodo.hiframes.pd_series_ext.init_series(data, index, col_name).loc[
                idx[0]
            ]

        return impl_col_name

    # TODO: error-checking test
    raise BodoError(
        "DataFrame.loc[] getitem (location-based indexing) using {} not supported yet.".format(
            idx
        )
    )  # pragma: no cover


##################################  df.iat  ##################################


# df.ia[] type
class DataFrameIatType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameIatType({})".format(df_type)
        super(DataFrameIatType, self).__init__(name)


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
        if context.enable_nrt:
            context.nrt.incref(builder, signature.args[0], obj_val)

        return iat_val._getvalue()

    return DataFrameIatType(obj)(obj), codegen


@overload_attribute(DataFrameType, "iat")
def overload_series_iat(s):
    return lambda s: bodo.hiframes.dataframe_indexing.init_dataframe_iat(s)


# df.iat[] getitem
@overload(operator.getitem)
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
@overload(operator.setitem)
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
