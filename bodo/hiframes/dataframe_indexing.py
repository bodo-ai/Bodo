# Copyright (C) 2022 Bodo Inc. All rights reserved.
"""
Indexing support for pd.DataFrame type.
"""
import operator

import numpy as np
import pandas as pd
from numba.core import cgutils, types
from numba.core.typing.templates import AbstractTemplate, infer_global
from numba.extending import (
    intrinsic,
    lower_builtin,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_attribute,
    register_model,
)

import bodo
from bodo.hiframes.pd_dataframe_ext import (
    DataFrameType,
    check_runtime_cols_unsupported,
)
from bodo.utils.transform import gen_const_tup
from bodo.utils.typing import (
    BodoError,
    get_overload_const_int,
    get_overload_const_list,
    get_overload_const_str,
    is_immutable_array,
    is_list_like_index_type,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    raise_bodo_error,
)


@infer_global(operator.getitem)
class DataFrameGetItemTemplate(AbstractTemplate):
    """
    Split DataFrame GetItem into separate
    Typing and Lowering to Reduce Compilation Time
    Currently this is only implemented for boolean arrays/lists
    to handle filtering use cases.

    TODO: Extend to other getitem operations
    """

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 2
        check_runtime_cols_unsupported(args[0], "DataFrame getitem (df[])")
        if isinstance(args[0], DataFrameType):
            return self.typecheck_df_getitem(args)
        elif isinstance(args[0], DataFrameLocType):
            return self.typecheck_loc_getitem(args)
        else:
            return

    def typecheck_loc_getitem(self, args):
        # Currently, typechecks the following uses of df.loc:
        # df.loc[bool_ary],
        # df.loc[slice/bool_ary, col_ind],
        # df.loc[int, scalar col_ind] for dataframe with RangeIndex,

        # As these were the test cases in dataframe_part2
        # This may need to be expanded in

        I = args[0]
        idx = args[1]

        df = I.df_type

        # df with multi-level column names returns a lower level dataframe
        # Not currently supported
        if isinstance(df.columns[0], tuple):
            raise_bodo_error(
                "DataFrame.loc[] getitem (location-based indexing) with multi-indexed columns not supported yet"
            )

        # df.loc[idx] with idx = array of bools
        # TODO: add slice to this case when it's supported in the overload
        if (
            is_list_like_index_type(idx) and idx.dtype == types.bool_
        ):  # or isinstance(idx, types.SliceType):
            df_idx_inds = idx
            new_data_type = df.data
            new_columns = df.columns
            # Update index type for boolean indexing since that converts
            # range to numeric
            new_index_type = self.replace_range_with_numeric_idx_if_needed(
                df, df_idx_inds
            )
            ret = DataFrameType(
                new_data_type,
                new_index_type,
                new_columns,
                is_table_format=df.is_table_format,
            )
            return ret(*args)

        # df.loc[idx, col_ind]
        if isinstance(idx, types.BaseTuple) and len(idx) == 2:
            df_index_indexer_type = idx.types[0]
            df_columns_indexer_type = idx.types[1]

            # df.loc[scalar int, col_ind] (for range index only)
            if isinstance(df_index_indexer_type, types.Integer):
                if not isinstance(df.index, bodo.hiframes.pd_index_ext.RangeIndexType):
                    raise_bodo_error(
                        "Dataframe.loc[int, col_ind] getitem only supported for dataframes with RangeIndexes"
                    )

                # df.loc[scalar idx, "A"]
                if is_overload_constant_str(df_columns_indexer_type):
                    col_index_value = get_overload_const_str(df_columns_indexer_type)

                    if col_index_value not in df.columns:
                        raise_bodo_error(
                            "dataframe {} does not include column {}".format(
                                df, col_index_value
                            )
                        )
                    col_num = df.columns.index(col_index_value)
                    return (df.data[col_num].dtype)(*args)

                if isinstance(df_columns_indexer_type, types.UnicodeType):
                    raise_bodo_error(
                        f"DataFrame.loc[] getitem (location-based indexing) requires constant column names. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
                    )  # pragma: no cover

                # TODO: support df.loc[scalar int, ["A", "B"]] or df.loc[scalar int, [True, False, True]]
                else:
                    # In this case, we don't need a constant column name error warning, since we don't support df.loc[scalar int, ["A", "B"]] with either constant or non constant cols.
                    raise_bodo_error(
                        f"DataFrame.loc[] getitem (location-based indexing) using {idx} not supported yet."
                    )  # pragma: no cover

            if (
                is_list_like_index_type(df_index_indexer_type)
                and df_index_indexer_type.dtype == types.bool_
            ) or isinstance(df_index_indexer_type, types.SliceType):
                new_index_type = self.replace_range_with_numeric_idx_if_needed(
                    df, df_index_indexer_type
                )
                # df.loc[slice/bool_ary, "A"]
                if is_overload_constant_str(df_columns_indexer_type):
                    column_index_val = get_overload_const_str(df_columns_indexer_type)

                    if column_index_val not in df.columns:
                        raise_bodo_error(
                            f"dataframe {df} does not include column {column_index_val}"
                        )
                    col_num = df.columns.index(column_index_val)
                    data_type = df.data[col_num]
                    dtype = data_type.dtype
                    name_typ = types.literal(df.columns[col_num])
                    ret = bodo.SeriesType(dtype, data_type, new_index_type, name_typ)
                    return ret(*args)

                if isinstance(df_columns_indexer_type, types.UnicodeType):
                    raise_bodo_error(
                        f"DataFrame.loc[] getitem (location-based indexing) requires constant column names. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
                    )  # pragma: no cover

                # df.loc[slice/bool_ary, ["A", "B"]] or df.loc[slice/bool_ary, [True, False, True]]
                elif is_overload_constant_list(df_columns_indexer_type):
                    df_col_inds_literal = get_overload_const_list(
                        df_columns_indexer_type
                    )

                    unliteral_lst_typ = types.unliteral(df_columns_indexer_type)

                    if unliteral_lst_typ.dtype == types.bool_:
                        if len(df.columns) != len(df_col_inds_literal):
                            raise_bodo_error(
                                f"dataframe {df} has {len(df.columns)} columns, but boolean array used with DataFrame.loc[] {df_col_inds_literal} has {len(df_col_inds_literal)} values"
                            )  # pragma: no cover

                        new_names = []
                        new_data = []
                        for i in range(len(df_col_inds_literal)):
                            if df_col_inds_literal[i]:
                                new_names.append(df.columns[i])
                                new_data.append(df.data[i])
                        new_cols = tuple()

                        use_table_format = (
                            df.is_table_format
                            and len(new_names) > 0
                            and len(new_names)
                            >= bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
                        )

                        ret = DataFrameType(
                            tuple(new_data),
                            new_index_type,
                            tuple(new_names),
                            is_table_format=use_table_format,
                        )
                        return ret(*args)

                    elif unliteral_lst_typ.dtype == bodo.string_type:
                        (new_cols, new_data) = get_df_getitem_kept_cols_and_data(
                            df, df_col_inds_literal
                        )
                        use_table_format = (
                            df.is_table_format
                            and len(df_col_inds_literal) > 0
                            and len(df_col_inds_literal)
                            >= bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
                        )
                        ret = DataFrameType(
                            new_data,
                            new_index_type,
                            new_cols,
                            is_table_format=use_table_format,
                        )
                        return ret(*args)

        # this needs to have a constant warning, as the non constant list could either be indexing with column names or bool indexing
        raise_bodo_error(
            f"DataFrame.loc[] getitem (location-based indexing) using {idx} not supported yet. If you are trying to select a subset of the columns by passing a list of column names, that list must be a compile time constant. See https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
        )  # pragma: no cover

    def typecheck_df_getitem(self, args):
        df = args[0]
        ind = args[1]

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
                    new_data.append(df.data[i])
                data_type = tuple(new_data)
                index_type = df.index
                columns = tuple(new_names)
                ret = DataFrameType(data_type, index_type, columns)
                return ret(*args)
            # regular single level case
            else:
                if ind_val not in df.columns:
                    raise_bodo_error(
                        "dataframe {} does not include column {}".format(df, ind_val)
                    )
                col_num = df.columns.index(ind_val)
                data_type = df.data[col_num]
                dtype = data_type.dtype
                index_type = df.index
                name_typ = types.literal(df.columns[col_num])
                ret = bodo.SeriesType(dtype, data_type, index_type, name_typ)
                return ret(*args)

        if isinstance(ind, types.Integer) or isinstance(ind, types.UnicodeType):
            # if we have non constant integer/string getitem, raise an error
            raise_bodo_error(
                "df[] getitem selecting a subset of columns requires providing constant column names. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
            )  # pragma: no cover

        # df1 = df[df.A > 5] or df1 = df[:n]
        if (is_list_like_index_type(ind) and ind.dtype == types.bool_) or isinstance(
            ind, types.SliceType
        ):
            data_type = df.data
            # Update index type for boolean indexing since that converts
            # range to numeric
            index_type = self.replace_range_with_numeric_idx_if_needed(df, ind)
            columns = df.columns
            ret = DataFrameType(
                data_type, index_type, columns, is_table_format=df.is_table_format
            )
            return ret(*args)
        # A = df[["C1", "C2"]]
        elif is_overload_constant_list(ind):
            # Check that all columns named are in the dataframe
            ind_columns = get_overload_const_list(ind)
            (columns, data_type) = get_df_getitem_kept_cols_and_data(df, ind_columns)
            index_type = df.index
            use_table_format = (
                df.is_table_format
                and len(ind_columns) > 0
                and len(ind_columns) >= bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
            )
            ret = DataFrameType(
                data_type, index_type, columns, is_table_format=use_table_format
            )
            return ret(*args)

        # In this case, we need to raise a more general error, as a non constant list
        # could be an attempt by the user to select a subset of rows with an integer list,
        # or an attempt to select a subset of columns with a list of non constant column names
        # TODO: error-checking test
        raise_bodo_error(
            f"df[] getitem using {ind} not supported. If you are trying to select a subset of the columns, you must provide the column names you are selecting as a constant. See https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
        )  # pragma: no cover

    def replace_range_with_numeric_idx_if_needed(self, df, ind):
        """
        helper function for getitem typing. This function is used to get the output
        index type of a getitem on a Series/Dataframe with a rangeIndex, using a
        slice or a list of bools on the rows.
        Using a list of bools will result in a numeric index instead of a range index.
        """
        new_index_type = (
            bodo.hiframes.pd_index_ext.NumericIndexType(types.int64, df.index.name_typ)
            if not isinstance(ind, types.SliceType)
            and isinstance(df.index, bodo.hiframes.pd_index_ext.RangeIndexType)
            else df.index
        )
        return new_index_type


DataFrameGetItemTemplate._no_unliteral = True


def get_df_getitem_kept_cols_and_data(df, cols_to_keep_list):
    """helper function for getitem typing. Takes a dataframe, and a list of columns to keep,
    and returns the data_type, and the columns for a dataframe containing only those columns.
    Throws an error if cols_to_keep_list contains a column that is not present in the input
    dataframe."""
    # Check that all columns named are in the dataframe
    for c in cols_to_keep_list:
        if c not in df.column_index:
            raise_bodo_error(
                "Column {} not found in dataframe columns {}".format(c, df.columns)
            )
    columns = tuple(cols_to_keep_list)
    data_type = tuple(df.data[df.column_index[name]] for name in columns)
    return (columns, data_type)


# lowering is necessary since df filtering is used in the train_test_split()
# implementation which is not inlined and uses the regular Numba pipeline
# see bodo/tests/test_sklearn_part2.py::test_train_test_split_df
@lower_builtin(operator.getitem, DataFrameType, types.Any)
def getitem_df_lower(context, builder, sig, args):
    impl = df_getitem_overload(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def df_getitem_overload(df, ind):
    # This check shouldn't be needeed, but it can't hurt to keep it in
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
            if c not in df.column_index:
                raise_bodo_error(
                    "Column {} not found in dataframe columns {}".format(c, df.columns)
                )
        extra_globals = None
        if (
            df.is_table_format
            and len(ind_columns) > 0
            and len(ind_columns) >= bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
        ):
            # This assumes ind_columns are always names.
            col_nums = [df.column_index[c] for c in ind_columns]
            # Pass the column numbers as a MetaType to avoid putting
            # a constant in the IR.
            extra_globals = {
                "col_nums_meta": bodo.utils.typing.MetaType(tuple(col_nums))
            }
            new_data = f"bodo.hiframes.table.table_subset(bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df), col_nums_meta, True)"
        else:
            new_data = ", ".join(
                f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {df.column_index[c]}).copy()"
                for c in ind_columns
            )
        func_text = "def impl(df, ind):\n"
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)"
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text, ind_columns, new_data, index, extra_globals=extra_globals
        )

    # df1 = df[df.A > .5] or df[:n]
    if (is_list_like_index_type(ind) and ind.dtype == types.bool_) or isinstance(
        ind, types.SliceType
    ):
        # implement using array filtering (not using the old Filter node)
        # TODO: create an IR node for enforcing same dist for all columns and ind array
        func_text = "def impl(df, ind):\n"
        if not isinstance(ind, types.SliceType):
            func_text += "  ind = bodo.utils.conversion.coerce_to_array(ind)\n"
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[ind]"
        if df.is_table_format:
            new_data = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)[ind]"
        else:
            new_data = ", ".join(
                f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {df.column_index[c]})[ind]"
                for c in df.columns
            )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text,
            df.columns,
            new_data,
            index,
        )

    # TODO: error-checking test
    raise_bodo_error(
        "df[] getitem using {} not supported".format(ind)
    )  # pragma: no cover


# DataFrame setitem
@overload(operator.setitem, no_unliteral=True)
def df_setitem_overload(df, idx, val):
    check_runtime_cols_unsupported(df, "DataFrame setitem (df[])")
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

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 2


@register_model(DataFrameILocType)
class DataFrameILocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.df_type)]
        super(DataFrameILocModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameILocType, "obj", "_obj")


@intrinsic
def init_dataframe_iloc(typingctx, obj):
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
def overload_dataframe_iloc(df):
    check_runtime_cols_unsupported(df, "DataFrame.iloc")
    return lambda df: bodo.hiframes.dataframe_indexing.init_dataframe_iloc(
        df
    )  # pragma: no cover


# df.iloc[] getitem
@overload(operator.getitem, no_unliteral=True)
def overload_iloc_getitem(I, idx):
    if not isinstance(I, DataFrameILocType):
        return

    df = I.df_type

    # df.iloc[1], returns row as Series
    if isinstance(idx, types.Integer):
        return _gen_iloc_getitem_row_impl(df, df.columns, "idx")

    # df.iloc[idx, [1, 2]], selection with column index
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and not isinstance(idx[1], types.SliceType)
    ):
        if not (
            is_overload_constant_list(idx.types[1])
            or is_overload_constant_int(idx.types[1])
        ):
            raise_bodo_error(
                "idx2 in df.iloc[idx1, idx2] should be a constant integer or constant list of integers. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
            )

        num_cols = len(df.data)
        if is_overload_constant_int(idx.types[1]):
            is_out_series = True
            col_ind = get_overload_const_int(idx.types[1])
            # Test out of bounds indices
            if col_ind < 0 or col_ind >= num_cols:
                raise BodoError(
                    "df.iloc: column integer must refer to a valid column number"
                )
            col_inds = [col_ind]
        else:
            is_out_series = False
            col_inds = get_overload_const_list(idx.types[1])
            # Test invalid list type, and out of bounds indices
            if any(
                not isinstance(ind, int) or ind < 0 or ind >= num_cols
                for ind in col_inds
            ):
                raise BodoError(
                    "df.iloc: column list must be integers referring to a valid column number"
                )

        # NOTE: using pd.Series instead of np.array to avoid automatic value conversion
        # see: test_groupby_dead_col_multifunc
        col_names = tuple(pd.Series(df.columns, dtype=object)[col_inds])
        if isinstance(idx.types[0], types.Integer):
            # df.iloc[3, 1] case, output is a scalar value
            if isinstance(idx.types[1], types.Integer):
                col_ind = col_inds[0]

                def impl(I, idx):
                    df = I._obj
                    return bodo.utils.conversion.box_if_dt64(
                        bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)[
                            idx[0]
                        ]
                    )

                return impl

            return _gen_iloc_getitem_row_impl(df, col_names, "idx[0]")

        if (
            is_list_like_index_type(idx.types[0])
            and isinstance(idx.types[0].dtype, (types.Integer, types.Boolean))
        ) or isinstance(idx.types[0], types.SliceType):
            return _gen_iloc_getitem_bool_slice_impl(
                df, col_names, idx.types[0], "idx[0]", is_out_series
            )

    # df.iloc[idx]
    # array of bools/ints, or slice
    if (
        is_list_like_index_type(idx)
        and isinstance(idx.dtype, (types.Integer, types.Boolean))
    ) or isinstance(idx, types.SliceType):
        return _gen_iloc_getitem_bool_slice_impl(df, df.columns, idx, "idx", False)

    # TODO: error-checking test
    # df.iloc[:,1:] case requires typing pass transform since slice info not available
    # here. TODO: refactor when SliceLiteral of Numba has all the info.
    if (
        isinstance(idx, types.BaseTuple)
        and len(idx) == 2
        and isinstance(idx[0], types.SliceType)
        and isinstance(idx[1], types.SliceType)
    ):  # pragma: no cover
        raise_bodo_error(
            "slice2 in df.iloc[slice1,slice2] should be constant. For more information, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
        )

    raise_bodo_error(f"df.iloc[] getitem using {idx} not supported")  # pragma: no cover


def _gen_iloc_getitem_bool_slice_impl(df, col_names, idx_typ, idx, is_out_series):
    """generate df.iloc getitem implementation for cases with bool or slice index"""
    func_text = "def impl(I, idx):\n"
    func_text += "  df = I._obj\n"
    if isinstance(idx_typ, types.SliceType):
        func_text += f"  idx_t = {idx}\n"
    else:
        func_text += f"  idx_t = bodo.utils.conversion.coerce_to_array({idx})\n"
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx_t]"
    extra_globals = None
    if df.is_table_format and not is_out_series:
        col_nums = [df.column_index[c] for c in col_names]
        # Pass the column numbers as a MetaType to avoid putting
        # a constant in the IR.
        extra_globals = {"col_nums_meta": bodo.utils.typing.MetaType(tuple(col_nums))}
        new_data = f"bodo.hiframes.table.table_subset(bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df), col_nums_meta, False)[idx_t]"
    else:
        new_data = ", ".join(
            f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {df.column_index[c]})[idx_t]"
            for c in col_names
        )
    if is_out_series:
        c_name = (
            f"'{col_names[0]}'" if isinstance(col_names[0], str) else f"{col_names[0]}"
        )
        func_text += f"  return bodo.hiframes.pd_series_ext.init_series({new_data}, {index}, {c_name})\n"
        loc_vars = {}
        exec(func_text, {"bodo": bodo}, loc_vars)
        return loc_vars["impl"]

    return bodo.hiframes.dataframe_impl._gen_init_df(
        func_text, col_names, new_data, index, extra_globals=extra_globals
    )


def _gen_iloc_getitem_row_impl(df, col_names, idx):
    """generate df.iloc getitem implementation for cases that return a single row"""
    func_text = "def impl(I, idx):\n"
    func_text += "  df = I._obj\n"
    row_args = ", ".join(
        f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {df.column_index[c]})[{idx}]"
        for c in col_names
    )
    func_text += f"  row_idx = bodo.hiframes.pd_index_ext.init_heter_index({gen_const_tup(col_names)}, None)\n"
    # TODO: pass df_index[i] as row name (after issue with RangeIndex getitem in
    # test_df_apply_assertion is resolved)
    func_text += f"  return bodo.hiframes.pd_series_ext.init_series(({row_args},), row_idx, None)\n"
    loc_vars = {}
    exec(func_text, {"bodo": bodo}, loc_vars)
    impl = loc_vars["impl"]
    return impl


# DataFrame.iloc[] setitem
@overload(operator.setitem, no_unliteral=True)
def df_iloc_setitem_overload(df, idx, val):
    if not isinstance(df, DataFrameILocType):
        return

    # df.iloc[cond, 1] = A
    # handle in typing pass since the dataframe type can change
    # TODO: better error checking here
    raise_bodo_error(
        f"DataFrame.iloc setitem unsupported for dataframe {df.df_type}, index {idx}, value {val}"
    )


##################################  df.loc  ##################################


# df.loc[] type
class DataFrameLocType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameLocType({})".format(df_type)
        super(DataFrameLocType, self).__init__(name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 2


@register_model(DataFrameLocType)
class DataFrameLocModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.df_type)]
        super(DataFrameLocModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameLocType, "obj", "_obj")


@intrinsic
def init_dataframe_loc(typingctx, obj):
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
def overload_dataframe_loc(df):
    check_runtime_cols_unsupported(df, "DataFrame.loc")
    return lambda df: bodo.hiframes.dataframe_indexing.init_dataframe_loc(
        df
    )  # pragma: no cover


# df.loc[] getitem
@lower_builtin(operator.getitem, DataFrameLocType, types.Any)
def loc_getitem_lower(context, builder, sig, args):
    impl = overload_loc_getitem(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def overload_loc_getitem(I, idx):
    # This check shouldn't be needeed, but it can't hurt to keep it in
    if not isinstance(I, DataFrameLocType):
        return

    df = I.df_type

    # df.loc[idx] with array of bools
    if is_list_like_index_type(idx) and idx.dtype == types.bool_:
        func_text = "def impl(I, idx):\n"
        func_text += "  df = I._obj\n"
        func_text += "  idx_t = bodo.utils.conversion.coerce_to_array(idx)\n"
        index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx_t]"
        if df.is_table_format:
            new_data = f"bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)[idx_t]"
        else:
            new_data = ", ".join(
                f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {df.column_index[c]})[idx_t]"
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

            def impl_col_name(I, idx):  # pragma: no cover
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
            # check selected columns to be in dataframe schema
            # may require schema change, see test_loc_col_select (impl4)
            if (
                len(col_idx_list) > 0
                and not isinstance(col_idx_list[0], (bool, np.bool_))
                and not all(c in df.column_index for c in col_idx_list)
            ):
                raise_bodo_error(
                    f"DataFrame.loc[]: invalid column list {col_idx_list}; not all in dataframe columns {df.columns}"
                )
            return gen_df_loc_col_select_impl(df, col_idx_list)

    # TODO: error-checking test
    raise_bodo_error(
        f"DataFrame.loc[] getitem (location-based indexing) using {idx} not supported yet."
    )  # pragma: no cover


def gen_df_loc_col_select_impl(df, col_idx_list):
    """generate implementation for cases like df.loc[:, ["A", "B"]] and
    df.loc[:, [True, False, True]]
    """
    col_names = []
    col_inds = []
    # get column names if bool list
    if len(col_idx_list) > 0 and isinstance(col_idx_list[0], (bool, np.bool_)):
        for i, kept in enumerate(col_idx_list):
            if kept:
                col_inds.append(i)
                col_names.append(df.columns[i])
    else:
        col_names = col_idx_list
        col_inds = [df.column_index[c] for c in col_idx_list]
    extra_globals = None
    if (
        df.is_table_format
        and len(col_idx_list) > 0
        and len(col_idx_list) >= bodo.hiframes.boxing.TABLE_FORMAT_THRESHOLD
    ):
        extra_globals = {"col_nums_meta": bodo.utils.typing.MetaType(tuple(col_inds))}
        new_data = f"bodo.hiframes.table.table_subset(bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df), col_nums_meta, False)[idx[0]]"
    else:
        # create a new dataframe, create new data/index using idx
        new_data = ", ".join(
            f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {ind})[idx[0]]"
            for ind in col_inds
        )
    index = "bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)[idx[0]]"
    func_text = "def impl(I, idx):\n"
    func_text += "  df = I._obj\n"
    return bodo.hiframes.dataframe_impl._gen_init_df(
        func_text, col_names, new_data, index, extra_globals=extra_globals
    )


# DataFrame.loc[] setitem
@overload(operator.setitem, no_unliteral=True)
def df_loc_setitem_overload(df, idx, val):
    if not isinstance(df, DataFrameLocType):
        return

    # df.loc[cond, "B"] = A
    # handle in typing pass since the dataframe type can change
    # TODO: better error checking here
    raise_bodo_error(
        f"DataFrame.loc setitem unsupported for dataframe {df.df_type}, index {idx}, value {val}"
    )


##################################  df.iat  ##################################


# df.iat[] type
class DataFrameIatType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        name = "DataFrameIatType({})".format(df_type)
        super(DataFrameIatType, self).__init__(name)

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    ndim = 2


@register_model(DataFrameIatType)
class DataFrameIatModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("obj", fe_type.df_type)]
        super(DataFrameIatModel, self).__init__(dmm, fe_type, members)


make_attribute_wrapper(DataFrameIatType, "obj", "_obj")


@intrinsic
def init_dataframe_iat(typingctx, obj):
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
def overload_dataframe_iat(df):
    check_runtime_cols_unsupported(df, "DataFrame.iat")
    return lambda df: bodo.hiframes.dataframe_indexing.init_dataframe_iat(
        df
    )  # pragma: no cover


# df.iat[] getitem
@overload(operator.getitem, no_unliteral=True)
def overload_iat_getitem(I, idx):
    if not isinstance(I, DataFrameIatType):
        return

    # df.iat[1,0]
    if isinstance(idx, types.BaseTuple) and len(idx) == 2:
        if not isinstance(idx.types[0], types.Integer):
            raise BodoError(
                "DataFrame.iat: iAt based indexing can only have integer indexers"
            )
        if not is_overload_constant_int(idx.types[1]):
            raise_bodo_error(
                "DataFrame.iat getitem: column index must be a constant integer. For more informaton, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
            )
        col_ind = get_overload_const_int(idx.types[1])

        def impl_col_ind(I, idx):  # pragma: no cover
            df = I._obj
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            return bodo.utils.conversion.box_if_dt64(data[idx[0]])

        return impl_col_ind

    raise BodoError(
        "df.iat[] getitem using {} not supported".format(idx)
    )  # pragma: no cover


# df.iat[] setitem
@overload(operator.setitem, no_unliteral=True)
def overload_iat_setitem(I, idx, val):
    if not isinstance(I, DataFrameIatType):
        return

    # df.iat[1,0]
    if isinstance(idx, types.BaseTuple) and len(idx) == 2:
        if not isinstance(idx.types[0], types.Integer):
            raise BodoError(
                "DataFrame.iat: iAt based indexing can only have integer indexers"
            )
        if not is_overload_constant_int(idx.types[1]):
            raise_bodo_error(
                "DataFrame.iat setitem: column index must be a constant integer. For more informaton, see https://docs.bodo.ai/latest/bodo_parallelism/typing_considerations/#require_constants."
            )
        col_ind = get_overload_const_int(idx.types[1])

        # Bodo Restriction, cannot set item with immutable array.
        if is_immutable_array(I.df_type.data[col_ind]):
            raise BodoError(
                f"DataFrame setitem not supported for column with immutable array type {I.df_type.data}"
            )

        def impl_col_ind(I, idx, val):  # pragma: no cover
            df = I._obj
            data = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, col_ind)
            data[idx[0]] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(val)

        return impl_col_ind

    # TODO: error-checking test
    raise BodoError(
        "df.iat[] setitem using {} not supported".format(idx)
    )  # pragma: no cover


@lower_cast(DataFrameIatType, DataFrameIatType)
@lower_cast(DataFrameILocType, DataFrameILocType)
@lower_cast(DataFrameLocType, DataFrameLocType)
def cast_series_iat(context, builder, fromty, toty, val):
    """cast indexing objects since 'dist' in Series/DataFrame data types can change
    in distributed analysis.
    See test_get_list_string
    """
    # just cast the underlying dataframe object
    iat_val = cgutils.create_struct_proxy(fromty)(context, builder, val)
    new_df = context.cast(builder, iat_val.obj, fromty.df_type, toty.df_type)
    new_iat_val = cgutils.create_struct_proxy(toty)(context, builder)
    new_iat_val.obj = new_df
    return new_iat_val._getvalue()
