"""
Implement pd.DataFrame typing and data model handling.
"""

from __future__ import annotations

import operator
import time
import typing as pt
from collections.abc import Sequence
from functools import cached_property

import llvmlite.binding as ll
import numba
import numpy as np
import pandas as pd
from llvmlite import ir as lir
from numba.core import cgutils, types
from numba.core.imputils import impl_ret_borrowed, lower_constant
from numba.core.typing.templates import (
    AbstractTemplate,
    bound_function,
    infer_global,
    signature,
)
from numba.cpython.listobj import ListInstance
from numba.extending import (
    infer_getattr,
    intrinsic,
    lower_builtin,
    lower_cast,
    make_attribute_wrapper,
    models,
    overload,
    overload_method,
    register_model,
)
from numba.parfors.array_analysis import ArrayAnalysis

import bodo
import bodo.io.utils
import bodo.pandas as bd
from bodo.hiframes.datetime_date_ext import datetime_date_array_type
from bodo.hiframes.pd_categorical_ext import CategoricalArrayType
from bodo.hiframes.pd_index_ext import (
    HeterogeneousIndexType,
    NumericIndexType,
    RangeIndexType,
    SingleIndexType,
    is_pd_index_type,
)
from bodo.hiframes.pd_multi_index_ext import MultiIndexType
from bodo.hiframes.pd_series_ext import HeterogeneousSeriesType, SeriesType
from bodo.hiframes.series_indexing import SeriesIlocType
from bodo.hiframes.table import (
    Table,
    TableType,
    decode_if_dict_table,
    get_table_data,
    set_table_data_codegen,
)
from bodo.io import json_cpp
from bodo.ir.unsupported_method_template import (
    overload_unsupported_attribute,
    overload_unsupported_method,
)
from bodo.libs.array import (
    append_arr_info_list_to_cpp_table,
    arr_info_list_to_table,
    array_from_cpp_table,
    array_to_info,
    delete_table,
    py_table_to_cpp_table,
    shuffle_table,
)
from bodo.libs.bool_arr_ext import BooleanArrayType
from bodo.libs.float_arr_ext import FloatingArrayType
from bodo.libs.int_arr_ext import IntegerArrayType
from bodo.libs.str_arr_ext import str_arr_from_sequence
from bodo.libs.str_ext import string_type, unicode_to_utf8
from bodo.utils import tracing
from bodo.utils.cg_helpers import is_ll_eq
from bodo.utils.conversion import fix_arr_dtype, index_to_array, index_to_array_list
from bodo.utils.templates import OverloadedKeyAttributeTemplate
from bodo.utils.transform import get_const_func_output_type
from bodo.utils.typing import (
    BodoError,
    BodoWarning,
    ColNamesMetaType,
    assert_bodo_error,
    check_unsupported_args,
    create_unsupported_overload,
    decode_if_dict_array,
    dtype_to_array_type,
    error_on_unsupported_streaming_arrays,
    get_castable_arr_dtype,
    get_common_scalar_dtype,
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
    is_nullable,
    is_overload_bool,
    is_overload_const_str_equal,
    is_overload_constant_bool,
    is_overload_constant_int,
    is_overload_constant_list,
    is_overload_constant_str,
    is_overload_false,
    is_overload_int,
    is_overload_none,
    is_overload_true,
    is_str_arr_type,
    is_tuple_like_type,
    raise_bodo_error,
    to_nullable_type,
    to_str_arr_if_dict_array,
    unwrap_typeref,
)
from bodo.utils.utils import (
    bodo_exec,
    is_null_pointer,
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
        types.voidptr,
    ),
)
ll.add_symbol("json_write", json_cpp.json_write)


class DataFrameType(types.ArrayCompatible):  # TODO: IterableType over column names
    """Temporary type class for DataFrame objects."""

    index: SingleIndexType | MultiIndexType

    ndim = 2

    def __init__(
        self,
        data: Sequence[types.ArrayCompatible] | None = None,
        index=None,
        columns: Sequence[str] | None = None,
        dist=None,
        is_table_format=False,
    ):
        # data is tuple of Array types (not Series) or tuples (for df.describe)
        # index is Index obj (not Array type)
        # columns is a tuple of column names (strings, ints, or tuples in case of
        # MultiIndex)
        from bodo.transforms.distributed_analysis import Distribution

        self.data = data
        if index is None:
            index = RangeIndexType(types.none)
        self.index = index
        self.columns = columns
        # 'dist' is the distribution of this dataframe, which may not be accurate in all
        # stages since distribution info is not available before distribution analysis.
        # But it needs to be accurate for argument values before distribution analysis
        # starts, and should become generally accurate afterwards (needed for returns,
        # adjusting distributions in calls to other JIT functions).
        # Using OneD_Var as default to use when calling other JIT functions in type
        # inference stage. This will hopefully avoid extensive recompilation in
        # distributed analysis since distributed dataframes are the most common.
        dist = Distribution.OneD_Var if dist is None else dist
        self.dist = dist
        # flag indicating data is stored in the new Table format (needed for data model)
        self.is_table_format = is_table_format
        # If columns is None, we are determining the number of columns art runtime.
        if columns is None:
            assert is_table_format, (
                "Determining columns at runtime is only supported for DataFrame with table format"
            )
            # If we have columns determined at runtime, we change the arguments to create the table.
            self.table_type = TableType(tuple(data[:-1]), True)
        else:
            # save TableType to avoid recreating it in other places like column unbox & dtor
            self.table_type = TableType(data) if is_table_format else None

        super().__init__(
            name=f"dataframe({data}, {index}, {columns}, {dist}, {is_table_format}, {self.has_runtime_cols})"
        )

    def __str__(self):
        """Returns DataFrame name, if DataFrame has many columns returns compact representation of name."""
        if not self.has_runtime_cols and len(self.columns) > 20:
            data_str = f"{len(self.data)} columns of types {set(self.data)}"
            columns_str = (
                f"('{self.columns[0]}', '{self.columns[1]}', ..., '{self.columns[-1]}')"
            )

            # Note: We include this hash in the string to ensure that __str__ returns a unique string
            # for each dataframe type, to avoid issues with Numba caching. Numba caching isn't an issue
            # for dataframe type itself (as dataframeType has a defined key fn), but it can be an issue
            # for types that contain one or more dataframe's (Loc, Iloc, .dt, .str, etc.), as the
            # default key for a given type is the __str__ method of that type, and the default __str__
            # implementation will call __str__ on all
            # the components. Therefore, we need this __str__ method to still return a unique str
            # per each DataFrameType.
            # Technically, it's possible for two hash's from two different dataframe types to be
            # identical, but it's so unlikely (especially since we're only using this path as a fallback)
            # that we're not considering it.

            key_hash_val = str(hash(super().__str__()))
            return f"dataframe({data_str}, {self.index}, {columns_str}, {self.dist}, {self.is_table_format}, {self.has_runtime_cols}, key_hash={key_hash_val})"

        return super().__str__()

    def copy(
        self,
        data=None,
        index=None,
        columns=None,
        dist=None,
        is_table_format=None,
    ):
        if data is None:
            data = self.data
        if columns is None:
            columns = self.columns
        if index is None:
            index = self.index
        if dist is None:
            dist = self.dist
        if is_table_format is None:
            is_table_format = self.is_table_format

        return DataFrameType(
            data,
            index,
            columns,
            dist,
            is_table_format,
        )

    @property
    def has_runtime_cols(self):
        """
        Is the number of columns contained in this DataFrame
        determined at runtime
        """
        return self.columns is None

    @cached_property
    def column_index(self):
        return {c: i for i, c in enumerate(self.columns)}

    @property
    def runtime_colname_typ(self):
        """
        When the number of columns are determined at runtime then the
        DataFrame also contains the column names. This returns the array
        type for how the names are stored.
        """
        # If we have runtime columns the second element of the data tuple
        # is the column names array.
        return self.data[-1] if self.has_runtime_cols else None

    @property
    def runtime_data_types(self):
        """
        When the number of columns are determined at runtime then data
        contains both the array types and the column name types. This
        returns the tuple of column types.
        """
        return self.data[:-1] if self.has_runtime_cols else self.data

    @property
    def as_array(self):
        # using types.undefined to avoid Array templates for binary ops
        return types.Array(types.undefined, 2, "C")

    @property
    def key(self):
        # needed?
        return (
            self.data,
            self.index,
            self.columns,
            self.dist,
            self.is_table_format,
        )

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)

    def unify(self, typingctx, other):
        """unifies two possible dataframe types into a single type
        see test_dataframe.py::test_df_type_unify_error
        """
        from bodo.transforms.distributed_analysis import Distribution

        if (
            isinstance(other, DataFrameType)
            and len(other.data) == len(self.data)
            and other.columns == self.columns
            and other.has_runtime_cols == self.has_runtime_cols
        ):
            # NOTE: checking equality since Index types may not have unify() implemented
            # TODO: add unify() to all Index types and remove this
            new_index = (
                self.index
                if self.index == other.index
                else self.index.unify(typingctx, other.index)
            )
            data = tuple(
                a.unify(typingctx, b) if a != b else a
                for a, b in zip(self.data, other.data)
            )
            # use the most conservative distribution
            dist = Distribution(min(self.dist.value, other.dist.value))
            # NOTE: unification is an extreme corner case probably, since arrays can
            # be unified only if just their layout or alignment is different.
            # That doesn't happen in df case since all arrays are 1D and C layout.
            # see: https://github.com/numba/numba/blob/13ece9b97e6f01f750e870347f231282325f60c3/numba/core/types/npytypes.py#L436
            if new_index is not None and None not in data:  # pragma: no cover
                return DataFrameType(
                    data,
                    new_index,
                    self.columns,
                    dist,
                    self.is_table_format,
                )

        # convert empty dataframe to any other dataframe to support important common
        # cases (see test_append_empty_df), even though it's not fully accurate.
        # TODO: detect and handle wrong corner cases (or raise warning) in compiler
        # passes
        if (
            isinstance(other, DataFrameType)
            and len(self.data) == 0
            and not self.has_runtime_cols
        ):
            return other

    def can_convert_to(self, typingctx, other):
        from numba.core.typeconv import Conversion

        if (
            isinstance(other, DataFrameType)
            and self.data == other.data
            and self.index == other.index
            and self.columns == other.columns
            and self.dist != other.dist
            and self.has_runtime_cols == other.has_runtime_cols
        ):
            return Conversion.safe

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

    def replace_col_type(self, col_name, new_type):
        """
        Return a new DataFrameType with the given column name's type replaced
        with the new type. The column name must already exist.

        This API is intended to work as a fail safe when bodo.typeof cannot
        properly infer a type based on constant information. For example,
        StructArrayType vs MapArrayType with string keys.
        """
        if col_name not in self.columns:
            raise ValueError(
                f"DataFrameType.replace_col_type replaced column must be found in the DataFrameType. '{col_name}' not found in DataFrameType with columns {self.columns}"
            )
        col_idx = self.columns.index(col_name)
        new_data = tuple(
            list(self.data[:col_idx]) + [new_type] + list(self.data[col_idx + 1 :])
        )
        return DataFrameType(
            new_data,
            self.index,
            self.columns,
            self.dist,
            self.is_table_format,
        )


def check_runtime_cols_unsupported(df, func_name):
    """
    Checks if df is a DataFrameType with has_runtime_cols=True.
    If so it raises a Bodo error for the provided function name.
    """
    if isinstance(df, DataFrameType) and df.has_runtime_cols:
        raise BodoError(
            f"{func_name} on DataFrames with columns determined at runtime is not yet supported. Please return the DataFrame to regular Python to update typing information."
        )


# payload type inside meminfo so that mutation are seen by all references
class DataFramePayloadType(types.Type):
    def __init__(self, df_type):
        self.df_type = df_type
        super().__init__(name=f"DataFramePayloadType({df_type})")

    @property
    def mangling_args(self):
        """
        Avoids long mangled function names in the generated LLVM, which slows down
        compilation time. See [BE-1726]
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/funcdesc.py#L67
        https://github.com/numba/numba/blob/8e6fa5690fbe4138abf69263363be85987891e8b/numba/core/itanium_mangler.py#L219
        """
        return self.__class__.__name__, (self._code,)


# TODO: encapsulate in meminfo since dataframe is mutable, for example:
# df = pd.DataFrame({'A': A})
# df2 = df
# if cond:
#    df['A'] = B
# df2.A
# TODO: meminfo for reference counting of dataframes
@register_model(DataFramePayloadType)
class DataFramePayloadModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        # NOTE: columns are lazily unboxed from Python so some array values may be null
        data_typ = types.Tuple(fe_type.df_type.data)
        if fe_type.df_type.is_table_format:
            data_typ = types.Tuple([fe_type.df_type.table_type])
        members = [
            ("data", data_typ),
            ("index", fe_type.df_type.index),
            ("parent", types.pyobject),
        ]
        if fe_type.df_type.has_runtime_cols:
            members.append(("columns", fe_type.df_type.runtime_colname_typ))
        super().__init__(dmm, fe_type, members)


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
        super().__init__(dmm, fe_type, members)


# Export meminfo for null checks
make_attribute_wrapper(DataFrameType, "meminfo", "_meminfo")


@infer_getattr
class DataFrameAttribute(OverloadedKeyAttributeTemplate):
    key = DataFrameType

    def resolve_shape(self, df):
        return types.Tuple([types.int64, types.int64])

    @bound_function("df.head")
    def resolve_head(self, df, args, kws):
        func_name = "DataFrame.head"
        check_runtime_cols_unsupported(df, f"{func_name}()")

        # Obtain a the pysig and folded args
        arg_names = ("n",)
        arg_defaults = {"n": 5}

        pysig, folded_args = bodo.utils.typing.fold_typing_args(
            func_name, args, kws, arg_names, arg_defaults
        )

        # Check typing on arguments
        n_arg = folded_args[0]
        if not is_overload_int(n_arg):
            raise BodoError(f"{func_name}(): 'n' must be an Integer")

        # Determine the return type
        # Return type is the same as the dataframe
        ret = df.copy()
        # Return the signature
        return ret(*folded_args).replace(pysig=pysig)

    @bound_function("df.corr")
    def resolve_corr(self, df, args, kws):
        func_name = "DataFrame.corr"
        check_runtime_cols_unsupported(df, f"{func_name}()")

        # Obtain a the pysig and folded args
        full_args = (df,) + args
        arg_names = ("df", "method", "min_periods")
        arg_defaults = {"method": "pearson", "min_periods": 1}
        unsupported_arg_names = ("method",)

        pysig, folded_args = bodo.utils.typing.fold_typing_args(
            func_name, full_args, kws, arg_names, arg_defaults, unsupported_arg_names
        )

        # Check typing on arguments
        min_periods_arg = folded_args[2]
        if not is_overload_int(min_periods_arg):
            raise BodoError(f"{func_name}(): 'min_periods' must be an Integer")

        # Determine the return type
        # Return type are float64 operating only on numeric columns
        numeric_col_names = []
        numeric_col_data = []
        for c, d in zip(df.columns, df.data):
            if bodo.utils.typing._is_pandas_numeric_dtype(d.dtype):
                numeric_col_names.append(c)
                # All output columns are always float64. The astype
                # used np.hstack makes this "A" instead of "C"
                numeric_col_data.append(types.Array(types.float64, 1, "A"))
        # TODO: support empty dataframe
        if len(numeric_col_names) == 0:
            raise_bodo_error("DataFrame.corr(): requires non-empty dataframe")
        numeric_col_data = tuple(numeric_col_data)
        numeric_col_names = tuple(numeric_col_names)
        index_typ = bodo.utils.typing.type_col_to_index(numeric_col_names)
        ret = DataFrameType(numeric_col_data, index_typ, numeric_col_names)
        # Return the signature
        return ret(*folded_args).replace(pysig=pysig)

    @bound_function("df.pipe", no_unliteral=True)
    def resolve_pipe(self, df, args, kws):
        check_runtime_cols_unsupported(df, "DataFrame.pipe()")
        return bodo.hiframes.pd_groupby_ext.resolve_obj_pipe(
            self, df, args, kws, "DataFrame"
        )

    @bound_function("df.apply", no_unliteral=True)
    def resolve_apply(self, df, args, kws):
        check_runtime_cols_unsupported(df, "DataFrame.apply()")
        kws = dict(kws)
        # pop apply() arguments from kws so only UDF kws remain
        func = args[0] if len(args) > 0 else kws.pop("func", None)
        axis = args[1] if len(args) > 1 else kws.pop("axis", types.literal(0))
        raw = args[2] if len(args) > 2 else kws.pop("raw", types.literal(False))
        result_type = args[3] if len(args) > 3 else kws.pop("result_type", types.none)
        f_args = args[4] if len(args) > 4 else kws.pop("args", types.Tuple([]))

        unsupported_args = {"raw": raw, "result_type": result_type}
        merge_defaults = {"raw": False, "result_type": None}
        check_unsupported_args(
            "Dataframe.apply",
            unsupported_args,
            merge_defaults,
            package_name="pandas",
            module_name="DataFrame",
        )

        # Is the function a UDF or a builtin
        is_udf = True
        if types.unliteral(func) == types.unicode_type:
            if not is_overload_constant_str(func):
                raise BodoError(
                    "DataFrame.apply(): string argument (for builtins) must be a compile time constant"
                )
            is_udf = False

        if not (is_overload_constant_int(axis)):
            raise BodoError(
                "Dataframe.apply(): axis argument must be a compile time constant."
            )
        axis_number = get_overload_const_int(axis)
        if is_udf and axis_number != 1:
            raise BodoError(
                "Dataframe.apply(): only axis=1 supported for user-defined functions"
            )
        elif axis_number not in (0, 1):
            raise BodoError("Dataframe.apply(): axis must be either 0 or 1")

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
        # the Index has constant column name values
        index_type = HeterogeneousIndexType(
            types.BaseTuple.from_types(tuple(types.literal(c) for c in df.columns)),
            None,
        )
        data_type = types.BaseTuple.from_types(dtypes)
        null_tup_type = types.Tuple([types.bool_] * len(data_type))
        nullable_dtype = bodo.types.NullableTupleType(data_type, null_tup_type)
        name_dtype = df.index.dtype
        if name_dtype == types.NPDatetime("ns"):
            name_dtype = bodo.types.pd_timestamp_tz_naive_type
        if name_dtype == types.NPTimedelta("ns"):
            name_dtype = bodo.types.pd_timedelta_type
        if is_heterogeneous_tuple_type(data_type):
            row_typ = HeterogeneousSeriesType(nullable_dtype, index_type, name_dtype)
        else:
            row_typ = SeriesType(
                data_type.dtype, nullable_dtype, index_type, name_dtype
            )
        arg_typs = (row_typ,)
        if f_args is not None:
            arg_typs += tuple(f_args.types)

        try:
            if not is_udf:
                f_return_type = bodo.utils.transform.get_udf_str_return_type(
                    df,
                    get_overload_const_str(func),
                    self.context,
                    "DataFrame.apply",
                    # Only pass axis if axis=1
                    axis if axis_number == 1 else None,
                )
            else:
                f_return_type = get_const_func_output_type(
                    func,
                    arg_typs,
                    kws,
                    self.context,
                    numba.core.registry.cpu_target.target_context,
                )
        except Exception as e:
            raise_bodo_error(get_udf_error_msg("DataFrame.apply()", e))
        if is_udf:
            # check axis. We only accept axis=0 on builtins.
            if not (
                is_overload_constant_int(axis) and get_overload_const_int(axis) == 1
            ):
                raise BodoError(
                    "Dataframe.apply(): only user-defined functions with axis=1 supported"
                )
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
                # Heterogenous Series should always return a Nullable Tuple in the output type,
                if isinstance(
                    f_return_type.data, bodo.libs.nullable_tuple_ext.NullableTupleType
                ):
                    scalar_types = f_return_type.data.tuple_typ.types
                elif isinstance(f_return_type.data, types.Tuple):
                    # TODO: Confirm if this path ever taken? It shouldn't be.
                    scalar_types = f_return_type.data.types
                else:
                    raise_bodo_error(
                        "df.apply(): Unexpected Series return type for Heterogeneous data"
                    )
                # NOTE: nullable is determined at runtime, so by default always assume nullable type
                # TODO: Support for looking at constant values.
                arrs = tuple(
                    to_nullable_type(dtype_to_array_type(t)) for t in scalar_types
                )
                ret_type = DataFrameType(arrs, df.index, index_vals)
            elif isinstance(f_return_type, SeriesType):
                n_cols, index_vals = f_return_type.const_info
                # Note: For homogenous Series we return a regular tuple, so
                # convert to nullable.
                arrs = tuple(
                    to_nullable_type(dtype_to_array_type(f_return_type.dtype))
                    for _ in range(n_cols)
                )
                ret_type = DataFrameType(arrs, df.index, index_vals)
            else:
                data_arr = get_udf_out_arr_type(f_return_type)
                ret_type = SeriesType(data_arr.dtype, data_arr, df.index, None)
        else:
            # If apply just calls a builtin function we just return the type of that
            # function.
            ret_type = f_return_type

        # add dummy default value for UDF kws to avoid errors
        kw_names = ", ".join(f"{a} = ''" for a in kws.keys())
        func_text = f"def apply_stub(func, axis=0, raw=False, result_type=None, args=(), {kw_names}):\n"
        func_text += "    pass\n"
        loc_vars = {}
        exec(func_text, {}, loc_vars)
        apply_stub = loc_vars["apply_stub"]

        pysig = numba.core.utils.pysignature(apply_stub)
        new_args = (func, axis, raw, result_type, f_args) + tuple(kws.values())
        return signature(ret_type, *new_args).replace(pysig=pysig)

    @bound_function("df.plot", no_unliteral=True)
    def resolve_plot(self, df, args, kws):
        # Obtain a the pysig and folded args
        func_name = "DataFrame.plot"
        check_runtime_cols_unsupported(df, f"{func_name}()")
        arg_names = (
            "x",
            "y",
            "kind",
            "figsize",
            "ax",
            "subplots",
            "sharex",
            "sharey",
            "layout",
            "use_index",
            "title",
            "grid",
            "legend",
            "style",
            "logx",
            "logy",
            "loglog",
            "xticks",
            "yticks",
            "xlim",
            "ylim",
            "rot",
            "fontsize",
            "colormap",
            "table",
            "yerr",
            "xerr",
            "secondary_y",
            "sort_columns",
            "xlabel",
            "ylabel",
            "position",
            "stacked",
            "mark_right",
            "include_bool",
            "backend",
        )
        arg_defaults = {
            "x": None,
            "y": None,
            "kind": "line",
            "figsize": None,
            "ax": None,
            "subplots": False,
            "sharex": None,
            "sharey": False,
            "layout": None,
            "use_index": True,
            "title": None,
            "grid": None,
            "legend": True,
            "style": None,
            "logx": False,
            "logy": False,
            "loglog": False,
            "xticks": None,
            "yticks": None,
            "xlim": None,
            "ylim": None,
            "rot": None,
            "fontsize": None,
            "colormap": None,
            "table": False,
            "yerr": None,
            "xerr": None,
            "secondary_y": False,
            "sort_columns": False,
            "xlabel": None,
            "ylabel": None,
            "position": 0.5,
            "stacked": False,  # True in area plot
            "mark_right": True,
            "include_bool": False,
            "backend": None,
        }
        unsupported_arg_names = (
            "subplots",
            "sharex",
            "sharey",
            "layout",
            "use_index",
            "grid",
            "style",
            "logx",
            "logy",
            "loglog",
            "xlim",
            "ylim",
            "rot",
            "colormap",
            "table",
            "yerr",
            "xerr",
            "sort_columns",
            "secondary_y",
            "colorbar",
            "position",
            "stacked",
            "mark_right",
            "include_bool",
            "backend",
        )

        pysig, folded_args = bodo.utils.typing.fold_typing_args(
            func_name, args, kws, arg_names, arg_defaults, unsupported_arg_names
        )

        kind = folded_args[2]  # default: "line"
        if not is_overload_constant_str(kind):
            raise BodoError(
                f"{func_name}: kind must be a constant string and one of ('line', 'scatter')."
            )

        # Check typing on arguments
        x = folded_args[0]  # None
        # label or position, default None
        if not is_overload_none(x) and not (
            is_overload_int(x) or is_overload_constant_str(x)
        ):
            raise BodoError(
                f"{func_name}: x must be a constant column name, constant integer, or None."
            )
        if is_overload_constant_str(x):
            x_name = get_overload_const_str(x)
            if x_name not in df.columns:
                raise BodoError(f"{func_name}: {x_name} column not found.")
        elif is_overload_int(x):
            x_val = get_overload_const_int(x)
            if x_val > len(df.columns):
                raise BodoError(
                    f"{func_name}: x: {x_val} is out of bounds for axis 0 with size {len(df.columns)}"
                )
            x = df.columns[x]
        # label, position or list of label, positions, default None
        y = folded_args[1]  # None
        if not is_overload_none(y) and not (
            is_overload_int(y) or is_overload_constant_str(y)
        ):
            raise BodoError(
                "df.plot(): y must be a constant column name, constant integer, or None."
            )
        if is_overload_constant_str(y):
            y_name = get_overload_const_str(y)
            if y_name not in df.columns:
                raise BodoError(f"{func_name}: {y_name} column not found.")
        elif is_overload_int(y):
            y_val = get_overload_const_int(y)
            if y_val > len(df.columns):
                raise BodoError(
                    f"{func_name}: y: {y_val} is out of bounds for axis 0 with size {len(df.columns)}"
                )
            y = df.columns[y]

        # A tuple (width, height) in inches
        figsize = folded_args[3]
        if not is_overload_none(figsize) and not is_tuple_like_type(figsize):
            raise BodoError(
                f"{func_name}: figsize must be a constant numeric tuple (width, height) or None."
            )

        title = folded_args[10]
        if not is_overload_none(title) and not is_overload_constant_str(title):
            raise BodoError(f"{func_name}: title must be a constant string or None.")

        legend = folded_args[12]
        if not is_overload_bool(legend):
            raise BodoError(f"{func_name}: legend must be a boolean type.")

        xticks = folded_args[17]
        if not is_overload_none(xticks) and not is_tuple_like_type(xticks):
            raise BodoError(f"{func_name}: xticks must be a constant tuple or None.")
        yticks = folded_args[18]
        if not is_overload_none(yticks) and not is_tuple_like_type(yticks):
            raise BodoError(f"{func_name}: yticks must be a constant tuple or None.")
        fontsize = folded_args[22]
        if not is_overload_none(fontsize) and not is_overload_int(fontsize):
            raise BodoError(f"{func_name}: fontsize must be an integer or None.")

        xlabel = folded_args[29]
        if not is_overload_none(xlabel) and not is_overload_constant_str(xlabel):
            raise BodoError(f"{func_name}: xlabel must be a constant string or None.")
        ylabel = folded_args[30]
        if not is_overload_none(ylabel) and not is_overload_constant_str(ylabel):
            raise BodoError(f"{func_name}: ylabel must be a constant string or None.")

        # default: line
        return_typ = types.List(types.mpl_line_2d_type)
        kind = get_overload_const_str(kind)
        if kind == "scatter":
            if is_overload_none(x) and is_overload_none(y):
                raise BodoError(f"{func_name}: {kind} requires an x and y column.")
            elif is_overload_none(x):
                raise BodoError(f"{func_name}: {kind} x column is missing.")
            elif is_overload_none(y):
                raise BodoError(f"{func_name}: {kind} y column is missing.")

            return_typ = types.mpl_path_collection_type
        elif kind != "line":
            raise BodoError(f"{func_name}: {kind} plot is not supported.")

        ## Return the signature
        return signature(return_typ, *folded_args).replace(pysig=pysig)

    def generic_resolve(self, df, attr):
        if self._is_existing_attr(attr):
            return
        check_runtime_cols_unsupported(df, "Acessing DataFrame columns by attribute")
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
DataFrameAttribute._no_unliteral = True  # type: ignore


# workaround to support row["A"] case in df.apply()
# implements getitem for namedtuples if generated by Bodo
@overload(operator.getitem, no_unliteral=True, jit_options={"cache": True})
def namedtuple_getitem_overload(tup, idx):
    if isinstance(tup, types.BaseNamedTuple) and is_overload_constant_str(idx):
        field_idx = get_overload_const_str(idx)
        val_ind = tup.instance_class._fields.index(field_idx)
        return lambda tup, idx: tup[val_ind]  # pragma: no cover


def decref_df_data(context, builder, payload, df_type):
    """call decref() on all data arrays and index of dataframe"""
    # decref all unboxed arrays
    if df_type.is_table_format:
        # no need to check for null columns since decref ignores nulls:
        # https://github.com/numba/numba/blob/e314821f48bfc1678c9662584eef166fb9d5469c/numba/core/runtime/nrtdynmod.py#L78
        context.nrt.decref(
            builder, df_type.table_type, builder.extract_value(payload.data, 0)
        )
        context.nrt.decref(builder, df_type.index, payload.index)
        if df_type.has_runtime_cols:
            context.nrt.decref(builder, df_type.data[-1], payload.columns)
        return

    for i in range(len(df_type.data)):
        arr = builder.extract_value(payload.data, i)
        context.nrt.decref(builder, df_type.data[i], arr)

    # decref index
    context.nrt.decref(builder, df_type.index, payload.index)


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
    fn = cgutils.get_or_insert_function(mod, fnty, name=f".dtor.df.{df_type}")

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
    context, builder, df_type, data_tup, index_val, parent=None, colnames=None
):
    # create payload struct and store values
    payload_type = DataFramePayloadType(df_type)
    dataframe_payload = cgutils.create_struct_proxy(payload_type)(context, builder)
    dataframe_payload.data = data_tup
    dataframe_payload.index = index_val
    if colnames is not None:
        assert df_type.has_runtime_cols, (
            "construct_dataframe can only provide colnames if columns are determined at runtime"
        )
        dataframe_payload.columns = colnames

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
def init_runtime_cols_dataframe(typingctx, data_typ, index_typ, colnames_index_typ):
    """Create a DataFrame from the provided table, index, and column
    names when the number of columns is determined at runtime.
    """
    assert (
        isinstance(data_typ, types.BaseTuple)
        and isinstance(data_typ.dtype, TableType)
        and data_typ.dtype.has_runtime_cols
    ), (
        "init_runtime_cols_dataframe must be called with a table that determines columns at runtime."
    )
    assert bodo.hiframes.pd_index_ext.is_pd_index_type(
        colnames_index_typ
    ) or isinstance(
        colnames_index_typ, bodo.hiframes.pd_multi_index_ext.MultiIndexType
    ), "Column names must be an index"
    if isinstance(data_typ.dtype.arr_types, types.UniTuple):
        arr_types = [data_typ.dtype.arr_types.dtype] * len(data_typ.dtype.arr_types)
    else:
        arr_types = list(data_typ.dtype.arr_types)

    ret_typ = DataFrameType(
        tuple(arr_types + [colnames_index_typ]),
        index_typ,
        None,
        is_table_format=True,
    )

    def codegen(context, builder, signature, args):
        df_type = signature.return_type
        data_tup, index, col_names = args
        parent = None
        dataframe_val = construct_dataframe(
            context, builder, df_type, data_tup, index, parent, col_names
        )
        # increase refcount of stored values
        context.nrt.incref(builder, data_typ, data_tup)
        context.nrt.incref(builder, index_typ, index)
        context.nrt.incref(builder, colnames_index_typ, col_names)
        return dataframe_val

    sig = signature(ret_typ, data_typ, index_typ, colnames_index_typ)
    return sig, codegen


@intrinsic
def init_dataframe(typingctx, data_tup_typ, index_typ, col_names_typ):
    """Create a DataFrame with provided data, index and columns values.
    Used as a single constructor for DataFrame and assigning its data, so that
    optimization passes can look for init_dataframe() to see if underlying
    data has changed, and get the array variables from init_dataframe() args if
    not changed.
    """
    assert is_pd_index_type(index_typ) or isinstance(index_typ, MultiIndexType), (
        f"init_dataframe(): invalid index type of {index_typ}"
    )

    n_cols = len(data_tup_typ.types)
    if n_cols == 0:
        column_names = ()

    untyperefed_col_names_typ = unwrap_typeref(col_names_typ)

    assert isinstance(untyperefed_col_names_typ, ColNamesMetaType) and isinstance(
        untyperefed_col_names_typ.meta, tuple
    ), (
        "Third argument to init_dataframe must be of type ColNamesMetaType, and must contain a tuple of column names"
    )
    column_names = untyperefed_col_names_typ.meta

    # handle the new table format
    if n_cols == 1 and isinstance(data_tup_typ.types[0], TableType):
        n_cols = len(data_tup_typ.types[0].arr_types)

    assert len(column_names) == n_cols, (
        "init_dataframe(): number of column names does not match number of columns"
    )

    # get data array types for new table format
    is_table_format = False
    data_arrs = data_tup_typ.types
    if n_cols != 0 and isinstance(data_tup_typ.types[0], TableType):
        data_arrs = data_tup_typ.types[0].arr_types
        is_table_format = True

    ret_typ = DataFrameType(
        data_arrs, index_typ, column_names, is_table_format=is_table_format
    )

    def codegen(context, builder, signature, args):
        df_type = signature.return_type
        data_tup = args[0]
        index_val = args[1]
        parent = None
        # set df parent to parent of input table in case unboxing columns is necessary
        if is_table_format:
            table = cgutils.create_struct_proxy(ret_typ.table_type)(
                context, builder, builder.extract_value(data_tup, 0)
            )
            parent = table.parent

        dataframe_val = construct_dataframe(
            context, builder, df_type, data_tup, index_val, parent, None
        )
        # increase refcount of stored values
        context.nrt.incref(builder, data_tup_typ, data_tup)
        context.nrt.incref(builder, index_typ, index_val)
        return dataframe_val

    sig = signature(ret_typ, data_tup_typ, index_typ, col_names_typ)
    return sig, codegen


def pushdown_safe_init_df(table, colNames):  # pragma: no cover
    # Dummy function used for overload
    pass


@overload(pushdown_safe_init_df, inline="never", jit_options={"cache": True})
def overload_pushdown_safe_init_df(table, colNames):
    """
    A wrapper for init_dataframe to coerce a table to a DataFrame while preventing filter pushdown
    from tracking this function call as a "use" of the table variable. This is not to be used
    outside of merge into.

    Args:
        table [TableType]: the table that is to be coerced to a DataFrame
        colNames [ColNamesMetaType]: the names of the columns of the DataFrame

    Returns:
        [DataFrame] the data from the table wrapped in a DataFrame.
    """

    def bodo_pushdown_safe_init_df(table, colNames):
        index = bodo.hiframes.pd_index_ext.init_range_index(0, len(table), 1, None)
        return bodo.hiframes.pd_dataframe_ext.init_dataframe((table,), index, colNames)

    return bodo_pushdown_safe_init_df


@intrinsic
def has_parent(typingctx, df):
    check_runtime_cols_unsupported(df, "has_parent")

    def codegen(context, builder, sig, args):
        dataframe = cgutils.create_struct_proxy(sig.args[0])(
            context, builder, value=args[0]
        )
        return cgutils.is_not_null(builder, dataframe.parent)

    return signature(types.bool_, df), codegen


@intrinsic(prefer_literal=True)
def _column_needs_unboxing(typingctx, df_typ, i_typ):
    check_runtime_cols_unsupported(df_typ, "_column_needs_unboxing")
    assert isinstance(df_typ, DataFrameType) and is_overload_constant_int(i_typ)

    def codegen(context, builder, sig, args):
        dataframe_payload = get_dataframe_payload(context, builder, df_typ, args[0])
        col_ind = get_overload_const_int(i_typ)
        arr_typ = df_typ.data[col_ind]

        if df_typ.is_table_format:
            table = cgutils.create_struct_proxy(df_typ.table_type)(
                context, builder, builder.extract_value(dataframe_payload.data, 0)
            )
            blk = df_typ.table_type.type_to_blk[arr_typ]
            arr_list = getattr(table, f"block_{blk}")
            arr_list_inst = ListInstance(
                context, builder, types.List(arr_typ), arr_list
            )
            offset = context.get_constant(
                types.int64, df_typ.table_type.block_offsets[col_ind]
            )
            arr = arr_list_inst.getitem(offset)
        else:
            arr = builder.extract_value(dataframe_payload.data, col_ind)

        arr_struct_ptr = cgutils.alloca_once_value(builder, arr)
        null_struct_ptr = cgutils.alloca_once_value(
            builder, context.get_constant_null(arr_typ)
        )
        return is_ll_eq(builder, arr_struct_ptr, null_struct_ptr)

    return signature(types.bool_, df_typ, i_typ), codegen


def get_dataframe_payload(context, builder, df_type, value):
    meminfo = cgutils.create_struct_proxy(df_type)(context, builder, value).meminfo
    payload_type = DataFramePayloadType(df_type)
    payload = context.nrt.meminfo_data(builder, meminfo)
    ptrty = context.get_value_type(payload_type).as_pointer()
    payload = builder.bitcast(payload, ptrty)
    return context.make_helper(builder, payload_type, ref=payload)


@intrinsic
def _get_dataframe_data(typingctx, df_typ):
    check_runtime_cols_unsupported(df_typ, "_get_dataframe_data")
    ret_typ = types.Tuple(df_typ.data)
    if df_typ.is_table_format:
        ret_typ = types.Tuple([TableType(df_typ.data)])

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
def get_dataframe_index(typingctx, df_typ):
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
        if not is_overload_constant_int(args[1]):
            raise_bodo_error(
                "Selecting a DataFrame column requires a constant column label"
            )
        df = args[0]
        check_runtime_cols_unsupported(df, "get_dataframe_data")
        i = get_overload_const_int(args[1])
        ret = df.data[i]
        return ret(*args)


GetDataFrameDataInfer.prefer_literal = True


def get_dataframe_data_impl(df, i):
    if df.is_table_format:
        if bodo.hiframes.boxing.UNBOX_DATAFRAME_EAGERLY:

            def _impl(df, i):  # pragma: no cover
                return get_table_data(_get_dataframe_data(df)[0], i)

            return _impl

        def _impl(df, i):  # pragma: no cover
            if has_parent(df) and _column_needs_unboxing(df, i):
                bodo.hiframes.boxing.unbox_dataframe_column(df, i)
            return get_table_data(_get_dataframe_data(df)[0], i)

        return _impl

    def _impl(df, i):  # pragma: no cover
        if has_parent(df) and _column_needs_unboxing(df, i):
            bodo.hiframes.boxing.unbox_dataframe_column(df, i)
        return _get_dataframe_data(df)[i]

    return _impl


@intrinsic
def get_dataframe_table(typingctx, df_typ):
    """return internal data table for dataframe with table format"""
    assert df_typ.is_table_format, "get_dataframe_table() expects table format"

    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context,
            builder,
            df_typ.table_type,
            builder.extract_value(dataframe_payload.data, 0),
        )

    return df_typ.table_type(df_typ), codegen


def get_dataframe_all_data(df):  # pragma: no cover
    return df.data


def get_dataframe_all_data_impl(df):
    """implementation for get_dataframe_all_data(), which returns the underlying data
    of a dataframe (TableType in table format case and tuple of arrays in non-table
    case).
    Should be rarely actually used since get_dataframe_all_data() should be inlined.

    Args:
        df (DataFrameType): input dataframe type

    Returns:
        function (df -> TableType|tuple(array)): implementation for
            get_dataframe_all_data()
    """
    # table format case
    if df.is_table_format:

        def _impl(df):  # pragma: no cover
            return get_dataframe_table(df)

        return _impl

    # tuple of arrays case
    data = ", ".join(
        f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})"
        for i in range(len(df.columns))
    )
    comma = "," if len(df.columns) != 0 else ""
    return eval(f"lambda df: ({data}{comma})", {"bodo": bodo})


@infer_global(get_dataframe_all_data)
class GetDataFrameAllDataInfer(AbstractTemplate):
    """Type inference for get_dataframe_all_data(). Separate from lowering to improve
    compilation time (since get_dataframe_all_data will be inlined almost always).
    """

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        df_type = args[0]
        check_runtime_cols_unsupported(df_type, "get_dataframe_data")
        ret = (
            df_type.table_type
            if df_type.is_table_format
            else types.BaseTuple.from_types(df_type.data)
        )
        return ret(*args)


@lower_builtin(get_dataframe_all_data, DataFrameType)
def lower_get_dataframe_all_data(context, builder, sig, args):
    """lowering for get_dataframe_all_data()
    Should be rarely actually used since get_dataframe_all_data() should be inlined.
    """
    impl = get_dataframe_all_data_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


@intrinsic
def get_dataframe_column_names(typingctx, df_typ):
    """return internal column names for dataframe with runtime columns"""
    assert df_typ.has_runtime_cols, (
        "get_dataframe_column_names() expects columns to be determined at runtime"
    )

    def codegen(context, builder, signature, args):
        dataframe_payload = get_dataframe_payload(
            context, builder, signature.args[0], args[0]
        )
        return impl_ret_borrowed(
            context, builder, df_typ.runtime_colname_typ, dataframe_payload.columns
        )

    return df_typ.runtime_colname_typ(df_typ), codegen


@lower_builtin(get_dataframe_data, DataFrameType, types.IntegerLiteral)
def lower_get_dataframe_data(context, builder, sig, args):
    impl = get_dataframe_data_impl(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def alias_ext_dummy_func(lhs_name, args, alias_map, arg_aliases):
    assert len(args) >= 1
    numba.core.ir_utils._add_alias(lhs_name, args[0].name, alias_map, arg_aliases)


numba.core.ir_utils.alias_func_extensions[
    ("get_dataframe_data", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_dataframe_index", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_dataframe_table", "bodo.hiframes.pd_dataframe_ext")
] = alias_ext_dummy_func
numba.core.ir_utils.alias_func_extensions[
    ("get_dataframe_all_data", "bodo.hiframes.pd_dataframe_ext")
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


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_init_dataframe = (  # type: ignore
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


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_data = (  # type: ignore
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


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_index = (  # type: ignore
    get_dataframe_index_equiv
)


def get_dataframe_table_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for get_dataframe_table(). output table has the same shape as
    input df (rows and columns).
    """
    assert len(args) == 1 and not kws
    var = args[0]

    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(var), pre=[])


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_table = (  # type: ignore
    get_dataframe_table_equiv
)


def get_dataframe_column_names_equiv(self, scope, equiv_set, loc, args, kws):
    """array analysis for get_dataframe_column_names(). The output index's length
    matches the input df's number of columns.
    """
    assert len(args) == 1 and not kws
    var = args[0]

    if equiv_set.has_shape(var):
        return ArrayAnalysis.AnalyzeResult(shape=equiv_set.get_shape(var)[1], pre=[])
    return None


ArrayAnalysis._analyze_op_call_bodo_hiframes_pd_dataframe_ext_get_dataframe_column_names = (  # type: ignore
    get_dataframe_column_names_equiv
)


@intrinsic(prefer_literal=True)
def set_dataframe_data(typingctx, df_typ, c_ind_typ, arr_typ):
    """set column data of a dataframe inplace"""
    check_runtime_cols_unsupported(df_typ, "set_dataframe_data")
    assert_bodo_error(is_overload_constant_int(c_ind_typ))
    col_ind = get_overload_const_int(c_ind_typ)

    # make sure dataframe column data type is not changed (avoids lowering error)
    if df_typ.data[col_ind] != arr_typ:
        raise BodoError(
            "Changing dataframe column data type inplace is not supported in conditionals/loops or for dataframe arguments"
        )

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg = args
        dataframe_payload = get_dataframe_payload(context, builder, df_typ, df_arg)

        if df_typ.is_table_format:
            table = cgutils.create_struct_proxy(df_typ.table_type)(
                context, builder, builder.extract_value(dataframe_payload.data, 0)
            )
            blk = df_typ.table_type.type_to_blk[arr_typ]
            arr_list = getattr(table, f"block_{blk}")
            arr_list_inst = ListInstance(
                context, builder, types.List(arr_typ), arr_list
            )
            offset = context.get_constant(
                types.int64, df_typ.table_type.block_offsets[col_ind]
            )
            arr_list_inst.setitem(offset, arr_arg, True)
        else:
            # decref existing data column
            arr = builder.extract_value(dataframe_payload.data, col_ind)
            context.nrt.decref(builder, df_typ.data[col_ind], arr)

            # assign array
            dataframe_payload.data = builder.insert_value(
                dataframe_payload.data, arr_arg, col_ind
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
def set_df_index(typingctx, df_t, index_t):
    """used in very limited cases like distributed to_csv() to create a new
    dataframe with index
    """
    # TODO: make inplace when dfs are full objects
    check_runtime_cols_unsupported(df_t, "set_df_index")

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
            in_df.parent,
            None,
        )

        # increase refcount of stored values
        context.nrt.incref(builder, index_t, index_val)
        context.nrt.incref(builder, types.Tuple(df_t.data), in_df_payload.data)
        return dataframe

    ret_typ = DataFrameType(
        df_t.data, index_t, df_t.columns, df_t.dist, df_t.is_table_format
    )
    sig = signature(ret_typ, df_t, index_t)
    return sig, codegen


@intrinsic(prefer_literal=True)
def set_df_column_with_reflect(typingctx, df_type, cname_type, arr_type_t):
    """Set df column and reflect to parent Python object
    return a new df.
    """
    check_runtime_cols_unsupported(df_type, "set_df_column_with_reflect")
    assert_bodo_error(is_literal_type(cname_type), "constant column name expected")
    col_name = get_literal_value(cname_type)
    n_cols = len(df_type.columns)
    new_n_cols = n_cols
    data_typs = df_type.data
    column_names = df_type.columns
    index_typ = df_type.index
    is_new_col = col_name not in df_type.columns
    col_ind = n_cols
    cast_arr_to_nullable = False
    arr_type = arr_type_t

    if is_new_col:
        data_typs += (arr_type,)
        column_names += (col_name,)
        new_n_cols += 1
    else:
        col_ind = df_type.columns.index(col_name)
        # handle setting non-nullable array to nullable column
        if (
            isinstance(
                data_typs[col_ind],
                (FloatingArrayType, IntegerArrayType, BooleanArrayType),
            )
            and isinstance(arr_type, types.Array)
            and arr_type.dtype == data_typs[col_ind].dtype
        ):
            arr_type = data_typs[col_ind]
            cast_arr_to_nullable = True
        data_typs = tuple(
            (arr_type if i == col_ind else data_typs[i]) for i in range(n_cols)
        )

    def codegen(context, builder, signature, args):
        df_arg, _, arr_arg = args

        if cast_arr_to_nullable:
            arr_arg = context.compile_internal(
                builder,
                lambda A: bodo.utils.conversion.coerce_to_array(
                    A, use_nullable_array=True
                ),
                arr_type(arr_type_t),
                [arr_arg],
            )

        in_dataframe_payload = get_dataframe_payload(context, builder, df_type, df_arg)
        in_dataframe = cgutils.create_struct_proxy(df_type)(
            context, builder, value=df_arg
        )

        if df_type.is_table_format:
            in_table_type = df_type.table_type
            in_table = builder.extract_value(in_dataframe_payload.data, 0)
            out_table_type = TableType(data_typs)
            out_table = set_table_data_codegen(
                context,
                builder,
                in_table_type,
                in_table,
                out_table_type,
                arr_type,
                arr_arg,
                col_ind,
                is_new_col,
            )
            data_tup = context.make_tuple(
                builder, types.Tuple([out_table_type]), [out_table]
            )
        else:
            data_arrs = [
                builder.extract_value(in_dataframe_payload.data, i)
                if i != col_ind
                else arr_arg
                for i in range(n_cols)
            ]
            if is_new_col:
                data_arrs.append(arr_arg)

            for var, typ in zip(data_arrs, data_typs):
                context.nrt.incref(builder, typ, var)

            data_tup = context.make_tuple(builder, types.Tuple(data_typs), data_arrs)

        index_val = in_dataframe_payload.index
        context.nrt.incref(builder, index_typ, index_val)

        # TODO: refcount of parent?
        out_dataframe = construct_dataframe(
            context,
            builder,
            signature.return_type,
            data_tup,
            index_val,
            in_dataframe.parent,
            None,
        )

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

            # incref data again since there will be two references updated
            context.nrt.incref(builder, index_typ, index_val)
            if df_type.is_table_format:
                context.nrt.incref(
                    builder, out_table_type, builder.extract_value(data_tup, 0)
                )
            else:
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

        if cast_arr_to_nullable:
            context.nrt.decref(builder, arr_type, arr_arg)

        return out_dataframe

    ret_typ = DataFrameType(
        data_typs, index_typ, column_names, df_type.dist, df_type.is_table_format
    )
    sig = signature(ret_typ, df_type, cname_type, arr_type_t)
    return sig, codegen


@lower_constant(DataFrameType)
def lower_constant_dataframe(context, builder, df_type, pyval):
    """embed constant DataFrame value by getting constant values for data arrays and
    Index.
    """
    check_runtime_cols_unsupported(df_type, "lowering a constant DataFrame")
    n_cols = len(pyval.columns)
    data_arrs = []
    for i in range(n_cols):
        col = pyval.iloc[:, i]
        if isinstance(df_type.data[i], bodo.types.DatetimeArrayType):
            # TODO [BE-2441]: Unify?
            py_arr = col.array
        else:
            py_arr = col.values
        data_arrs.append(py_arr)
    data_arrs = tuple(data_arrs)

    if df_type.is_table_format:
        table = Table(data_arrs)
        table = context.get_constant_generic(builder, bodo.typeof(table), table)
        data_tup = lir.Constant.literal_struct([table])
    else:
        # not using get_constant_generic for tuple directly since Numba's tuple lowering
        # doesn't return a proper constant
        data_tup = lir.Constant.literal_struct(
            [
                context.get_constant_generic(builder, df_type.data[i], v)
                for i, v in enumerate(data_arrs)
            ]
        )

    index_val = context.get_constant_generic(builder, df_type.index, pyval.index)

    # create a constant payload with the same data model as DataFramePayloadType
    # "data", "index", "parent"
    parent_null = context.get_constant_null(types.pyobject)
    payload = lir.Constant.literal_struct([data_tup, index_val, parent_null])
    payload = cgutils.global_constant(builder, ".const.payload", payload).bitcast(
        cgutils.voidptr_t
    )

    # create a constant meminfo with the same data model as Numba
    minus_one = context.get_constant(types.int64, -1)
    null_ptr = context.get_constant_null(types.voidptr)
    meminfo = lir.Constant.literal_struct(
        [minus_one, null_ptr, null_ptr, payload, minus_one]
    )
    meminfo = cgutils.global_constant(builder, ".const.meminfo", meminfo).bitcast(
        cgutils.voidptr_t
    )

    # create the dataframe
    return lir.Constant.literal_struct([meminfo, parent_null])


@lower_cast(DataFrameType, DataFrameType)
def cast_df_to_df(context, builder, fromty, toty, val):
    """
    Support dataframe casting cases:
    1) convert RangeIndex to Int64Index
    2) cast empty dataframe to another dataframe
    (common pattern, see test_append_empty_df)
    """

    # trivial cast if only 'dist' is different (no need to change value)
    if (
        fromty.data == toty.data
        and fromty.index == toty.index
        and fromty.columns == toty.columns
        and fromty.is_table_format == toty.is_table_format
        and fromty.dist != toty.dist
        and fromty.has_runtime_cols == toty.has_runtime_cols
    ):
        return val

    # cast empty df with no columns to empty df with columns
    if (
        not fromty.has_runtime_cols
        and not toty.has_runtime_cols
        and len(fromty.data) == 0
        and len(toty.columns)
    ):
        return _cast_empty_df(context, builder, toty)

    # cases below assume data types are the same except DictionaryArray -> StringArrayType (only index and format changes)
    if (
        len(fromty.data) != len(toty.data)
        or (
            fromty.data != toty.data
            and any(
                context.typing_context.unify_pairs(fromty.data[i], toty.data[i]) is None
                for i in range(len(fromty.data))
            )
        )
        or fromty.has_runtime_cols != toty.has_runtime_cols
    ):
        raise BodoError(f"Invalid dataframe cast from {fromty} to {toty}")

    in_dataframe_payload = get_dataframe_payload(context, builder, fromty, val)

    # RangeIndex to Int64Index case
    if isinstance(fromty.index, RangeIndexType) and isinstance(
        toty.index, NumericIndexType
    ):
        new_index = context.cast(
            builder, in_dataframe_payload.index, fromty.index, toty.index
        )
    else:
        new_index = in_dataframe_payload.index
        context.nrt.incref(builder, fromty.index, new_index)

    # data format and content doesn't change
    if fromty.is_table_format == toty.is_table_format and fromty.data == toty.data:
        new_data = in_dataframe_payload.data
        if fromty.is_table_format:
            context.nrt.incref(builder, types.Tuple([fromty.table_type]), new_data)
        else:
            context.nrt.incref(
                builder, types.BaseTuple.from_types(fromty.data), new_data
            )
    # data format or content change.
    else:
        if not fromty.is_table_format and toty.is_table_format:
            new_data = _cast_df_data_to_table_format(
                context, builder, fromty, toty, val, in_dataframe_payload
            )
        elif fromty.is_table_format and not toty.is_table_format:
            new_data = _cast_df_data_to_tuple_format(
                context, builder, fromty, toty, val, in_dataframe_payload
            )
        elif fromty.is_table_format and toty.is_table_format:
            new_data = _cast_df_data_keep_table_format(
                context, builder, fromty, toty, val, in_dataframe_payload
            )
        else:
            new_data = _cast_df_data_keep_tuple_format(
                context, builder, fromty, toty, val, in_dataframe_payload
            )

    return construct_dataframe(
        context, builder, toty, new_data, new_index, in_dataframe_payload.parent, None
    )


def _cast_empty_df(context, builder, toty):
    """cast empty dataframe with no columns to an empty dataframe with columns
    see test_append_empty_df
    """
    # TODO(ehsan): can input df have non-empty index?

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


def _cast_df_data_to_table_format(
    context, builder, fromty, toty, df, in_dataframe_payload
):
    """cast df data from tuple data format to table data format"""
    check_runtime_cols_unsupported(
        toty, "casting traditional DataFrame to table format"
    )
    table_type = toty.table_type
    table = cgutils.create_struct_proxy(table_type)(context, builder)
    table.parent = in_dataframe_payload.parent

    # create blocks in output
    for t, blk in table_type.type_to_blk.items():
        n_arrs = context.get_constant(
            types.int64, len(table_type.block_to_arr_ind[blk])
        )
        _, out_arr_list = ListInstance.allocate_ex(
            context, builder, types.List(t), n_arrs
        )
        out_arr_list.size = n_arrs
        setattr(table, f"block_{blk}", out_arr_list.value)

    # copy array values from input
    for i, t in enumerate(fromty.data):
        out_type = toty.data[i]
        if t != out_type:
            # If the actual type changes we need an explicit cast, so as a result
            # we must ensure the column is unboxed.
            sig_args = (fromty, types.literal(i))
            impl = lambda df, i: bodo.hiframes.boxing.unbox_col_if_needed(df, i)
            sig = types.none(*sig_args)
            args = (df, context.get_constant(types.int64, i))
            context.compile_internal(builder, impl, sig, args)
        arr = builder.extract_value(in_dataframe_payload.data, i)
        # Perform the cast
        if t != out_type:
            new_arr = context.cast(builder, arr, t, out_type)
            should_incref = False
        else:
            new_arr = arr
            should_incref = True
        blk = table_type.type_to_blk[t]
        arr_list = getattr(table, f"block_{blk}")
        arr_list_inst = ListInstance(context, builder, types.List(t), arr_list)
        offset = context.get_constant(types.int64, table_type.block_offsets[i])
        arr_list_inst.setitem(offset, new_arr, should_incref)

    data_tup = context.make_tuple(
        builder, types.Tuple([table_type]), [table._getvalue()]
    )
    return data_tup


def _cast_df_data_keep_tuple_format(
    context, builder, fromty, toty, df, in_dataframe_payload
):
    """cast df data from tuple data format to  tuple format.
    This path is only used when there are types to need to be handled
    through casting.
    """
    check_runtime_cols_unsupported(toty, "casting traditional DataFrame columns")

    data_arrs = []
    for i in range(len(fromty.data)):
        if fromty.data[i] != toty.data[i]:
            # If the types aren't equal we need to cast. This cast requires
            # the array contents so we must unbox the column.
            sig_args = (fromty, types.literal(i))
            impl = lambda df, i: bodo.hiframes.boxing.unbox_col_if_needed(df, i)
            sig = types.none(*sig_args)
            args = (df, context.get_constant(types.int64, i))
            context.compile_internal(builder, impl, sig, args)
            arr = builder.extract_value(in_dataframe_payload.data, i)
            new_arr = context.cast(builder, arr, fromty.data[i], toty.data[i])
            should_incref = False
        else:
            # Otherwise we incref and just copy the data
            new_arr = builder.extract_value(in_dataframe_payload.data, i)
            should_incref = True
        if should_incref:
            context.nrt.incref(builder, toty.data[i], new_arr)
        data_arrs.append(new_arr)

    data_tup = context.make_tuple(builder, types.Tuple(toty.data), data_arrs)
    return data_tup


def _cast_df_data_keep_table_format(
    context, builder, fromty, toty, df, in_dataframe_payload
):
    """cast df data from table format to table format.
    This path is only used when there are types to need to be handled
    through casting.
    """
    check_runtime_cols_unsupported(toty, "casting table format DataFrame columns")

    in_table_type = fromty.table_type
    in_table = cgutils.create_struct_proxy(in_table_type)(
        context, builder, builder.extract_value(in_dataframe_payload.data, 0)
    )
    out_table_type = toty.table_type
    out_table = cgutils.create_struct_proxy(out_table_type)(context, builder)
    out_table.parent = in_dataframe_payload.parent

    # create blocks in output
    for t, blk in out_table_type.type_to_blk.items():
        n_arrs = context.get_constant(
            types.int64, len(out_table_type.block_to_arr_ind[blk])
        )
        _, out_arr_list = ListInstance.allocate_ex(
            context, builder, types.List(t), n_arrs
        )
        out_arr_list.size = n_arrs
        setattr(out_table, f"block_{blk}", out_arr_list.value)

    # copy array values from input
    # TODO: Reduce codegen for table format if all values in a list maps to
    # the same output type.
    # type mapping.
    for i in range(len(fromty.data)):
        in_type = fromty.data[i]
        out_type = toty.data[i]
        if in_type != out_type:
            # If the actual type changes we need an explicit cast, so as a result
            # we must ensure the column is unboxed.
            # TODO: Handle dead columns.
            sig_args = (fromty, types.literal(i))
            impl = lambda df, i: bodo.hiframes.boxing.unbox_col_if_needed(df, i)
            sig = types.none(*sig_args)
            args = (df, context.get_constant(types.int64, i))
            context.compile_internal(builder, impl, sig, args)
        # Get the input data
        in_blk = in_table_type.type_to_blk[in_type]
        in_arr_list = getattr(in_table, f"block_{in_blk}")
        in_arr_list_inst = ListInstance(
            context, builder, types.List(in_type), in_arr_list
        )
        in_offset = context.get_constant(types.int64, in_table_type.block_offsets[i])
        arr = in_arr_list_inst.getitem(in_offset)
        # Perform the cast
        if in_type != out_type:
            # TODO: Handle dead columns?
            new_arr = context.cast(builder, arr, in_type, out_type)
            should_incref = False
        else:
            new_arr = arr
            should_incref = True

        # Store in the output.
        out_blk = out_table_type.type_to_blk[t]
        out_arr_list = getattr(out_table, f"block_{out_blk}")
        out_arr_list_inst = ListInstance(
            context, builder, types.List(out_type), out_arr_list
        )
        out_offset = context.get_constant(types.int64, out_table_type.block_offsets[i])
        out_arr_list_inst.setitem(out_offset, new_arr, should_incref)

    data_tup = context.make_tuple(
        builder, types.Tuple([out_table_type]), [out_table._getvalue()]
    )
    return data_tup


def _cast_df_data_to_tuple_format(
    context, builder, fromty, toty, df, in_dataframe_payload
):
    """cast df data from table data format to tuple data format"""
    check_runtime_cols_unsupported(
        fromty, "casting table format to traditional DataFrame"
    )
    # TODO(ehsan): test
    table_type = fromty.table_type
    table = cgutils.create_struct_proxy(table_type)(
        context, builder, builder.extract_value(in_dataframe_payload.data, 0)
    )

    # copy array values from input
    data_arrs = []
    for i, t in enumerate(toty.data):
        in_type = fromty.data[i]
        if t != in_type:
            # If the actual type changes we need an explicit cast, so as a result
            # we must ensure the column is unboxed.
            sig_args = (fromty, types.literal(i))
            impl = lambda df, i: bodo.hiframes.boxing.unbox_col_if_needed(df, i)
            sig = types.none(*sig_args)
            args = (df, context.get_constant(types.int64, i))
            context.compile_internal(builder, impl, sig, args)

        blk = table_type.type_to_blk[in_type]
        arr_list = getattr(table, f"block_{blk}")
        arr_list_inst = ListInstance(context, builder, types.List(in_type), arr_list)
        offset = context.get_constant(types.int64, table_type.block_offsets[i])
        arr = arr_list_inst.getitem(offset)
        if t != in_type:
            new_arr = context.cast(builder, arr, in_type, t)
        else:
            new_arr = arr
            context.nrt.incref(builder, t, new_arr)
        data_arrs.append(new_arr)

    data_tup = context.make_tuple(builder, types.Tuple(toty.data), data_arrs)
    return data_tup


@overload(pd.DataFrame, inline="always", no_unliteral=True, jit_options={"cache": True})
@overload(bd.DataFrame, inline="always", no_unliteral=True, jit_options={"cache": True})
def pd_dataframe_overload(data=None, index=None, columns=None, dtype=None, copy=False):
    # TODO: support other input combinations
    # TODO: error checking
    if not is_overload_constant_bool(copy):  # pragma: no cover
        raise BodoError("pd.DataFrame(): 'copy' argument should be a constant boolean")

    copy = get_overload_const(copy)

    col_args, data_args, index_arg = _get_df_args(data, index, columns, dtype, copy)
    col_var = ColNamesMetaType(tuple(col_args))

    func_text = "def bodo_init_df(data=None, index=None, columns=None, dtype=None, copy=False):\n"
    func_text += f"  return bodo.hiframes.pd_dataframe_ext.init_dataframe({data_args}, {index_arg}, __col_name_meta_value_pd_overload)\n"
    return bodo_exec(
        func_text,
        {"bodo": bodo, "np": np, "__col_name_meta_value_pd_overload": col_var},
        {},
        __name__,
    )


@intrinsic
def _tuple_to_table_format_decoded(typingctx, df_typ):
    """
    Internal testing function used to convert a
    tuple format to table format and changing dict
    arrays to string arrays. This leads to
    a cast and is used to test casting between formats.
    """
    assert not df_typ.is_table_format, (
        "_tuple_to_table_format requires a tuple format input"
    )

    def codegen(context, builder, signature, args):
        # Force a cast.
        # TODO: Test for incref condition?
        return context.cast(
            builder,
            args[0],
            signature.args[0],
            signature.return_type,
        )

    ret_typ = DataFrameType(
        to_str_arr_if_dict_array(df_typ.data),
        df_typ.index,
        df_typ.columns,
        dist=df_typ.dist,
        is_table_format=True,
    )

    sig = signature(ret_typ, df_typ)
    return sig, codegen


@intrinsic
def _table_to_tuple_format_decoded(typingctx, df_typ):
    """
    Internal testing function used to convert a
    table format to tuple format and changing dict
    arrays to string arrays. This leads to
    a cast and is used to test casting between formats.
    """

    assert df_typ.is_table_format, (
        "_tuple_to_table_format requires a table format input"
    )

    def codegen(context, builder, signature, args):
        # Force a cast
        # TODO: Test for incref condition?
        return context.cast(
            builder,
            args[0],
            signature.args[0],
            signature.return_type,
        )

    ret_typ = DataFrameType(
        to_str_arr_if_dict_array(df_typ.data),
        df_typ.index,
        df_typ.columns,
        dist=df_typ.dist,
        is_table_format=False,
    )

    sig = signature(ret_typ, df_typ)
    return sig, codegen


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
        data_arrs = [f"data[{i}]" for i in range(n_cols + 1, 2 * n_cols + 1)]
        data_dict = dict(zip(data_keys, data_arrs))
        # if no index provided and there are Series inputs, get index from them
        # XXX cannot handle alignment of multiple Series
        if is_overload_none(index):
            for i, t in enumerate(data.types[n_cols + 1 :]):
                if isinstance(t, SeriesType):
                    index_arg = f"bodo.hiframes.pd_series_ext.get_series_index(data[{n_cols + 1 + i}])"
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
                "pd.DataFrame() only supports constant dictionary and array input"
            )
        if is_overload_none(columns):  # pragma: no cover
            raise BodoError(
                "pd.DataFrame() 'columns' argument is required when"
                " an array is passed as data"
            )
        copy_str = ".copy()" if copy else ""
        columns_consts = get_overload_const_list(columns)
        if columns_consts is None:
            raise_bodo_error("pd.DataFrame(): constant column names required")
        n_cols = len(columns_consts)
        data_val_types = {c: data.copy(ndim=1) for c in columns_consts}
        data_arrs = [f"data[:,{i}]{copy_str}" for i in range(n_cols)]
        data_dict = dict(zip(columns_consts, data_arrs))

    if is_overload_none(columns):
        col_names = data_dict.keys()
    else:
        col_names = get_overload_const_list(columns)
        if col_names is None:
            raise_bodo_error("pd.DataFrame(): constant column names required")

    df_len = _get_df_len_from_info(
        data_dict, data_val_types, col_names, index_is_none, index_arg
    )
    _fill_null_arrays(data_dict, col_names, df_len, dtype)

    # set default RangeIndex if index argument is None and data argument isn't Series
    if index_is_none:
        # empty df has object Index in Pandas which correponds to our StringIndex
        if is_overload_none(data):
            index_arg = "bodo.hiframes.pd_index_ext.init_binary_str_index(bodo.libs.str_arr_ext.pre_alloc_string_array(0, 0))"
        else:
            index_arg = (
                f"bodo.hiframes.pd_index_ext.init_range_index(0, {df_len}, 1, None)"
            )
    data_args = "({},)".format(
        ", ".join(
            f"bodo.utils.conversion.coerce_to_array({data_dict[c]}, True, scalar_to_arr_len={df_len}){astype_str}"
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
            df_len = f"len({data_dict[c]})"
            break

    # If we haven't found a length, rely on the index
    if df_len == "0":
        if not index_is_none:
            df_len = f"len({index_arg})"
        elif data_dict:  # pragma: no cover
            # In the case that the dataframe is not empty, throw an error.
            # This shouldn't regularly occur
            raise BodoError(
                "Internal Error: Unable to determine length of DataFrame Index. If this is unexpected, please try passing an index value."
            )

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
        dtype = "bodo.types.string_array_type"
    else:
        dtype = "bodo.utils.conversion.array_type_from_dtype(dtype)"

    # array with NaNs
    null_arr = f"bodo.libs.array_kernels.gen_na_array({df_len}, {dtype})"
    for c in col_names:
        if c not in data_dict:
            data_dict[c] = null_arr


@infer_global(len)
class LenTemplate(AbstractTemplate):
    """
    Split len into separate
    Typing and Lowering to Reduce Compilation Time
    """

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        # TODO: Fuse more templates
        if isinstance(args[0], (DataFrameType, bodo.types.TableType)):
            return types.int64(*args)


@lower_builtin(len, DataFrameType)
def table_len_lower(context, builder, sig, args):
    """
    Implementation for lowering len for DataFrames.
    """
    impl = df_len_overload(*sig.args)
    return context.compile_internal(builder, impl, sig, args)


def df_len_overload(df):
    if not isinstance(df, DataFrameType):
        return

    if df.has_runtime_cols:
        # If columns are determined at runtime we have determine
        # the length through the table.
        def impl(df):  # pragma: no cover
            if is_null_pointer(df._meminfo):
                return 0
            t = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)
            return len(t)

        return impl

    # Note: The 0 column path is never taken because it gets optimized
    # out in DataFrame pass. This implementation is included in case
    # series pass is unable to run, but it is not tested.
    if len(df.columns) == 0:  # pragma: no cover

        def impl(df):  # pragma: no cover
            if is_null_pointer(df._meminfo):
                return 0
            return len(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df))

        return impl

    def impl(df):  # pragma: no cover
        # If for some reason we have a null dataframe,
        # an assumption made in read_csv with chunksize
        # (see test_csv_chunksize_forloop), then return 0
        # because this should be a garbage/unused value.
        if is_null_pointer(df._meminfo):
            return 0
        return len(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, 0))

    return impl


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


GetItemTuple.prefer_literal = True


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
            raise IndexError(f"cannot index at {idx} in {tupty}")
        res = builder.extract_value(tup, idx)
    elif isinstance(idx, slice):
        items = cgutils.unpack_tuple(builder, tup)[idx]
        res = context.make_tuple(builder, sig.return_type, items)
    else:
        raise NotImplementedError(f"unexpected index {idx!r} for {sig.args[0]}")
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
    _bodo_rebalance_output_if_skewed,
    gen_cond,
):  # pragma: no cover
    return left_df


@infer_global(join_dummy)
class JoinTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        (
            left_df,
            right_df,
            left_on,
            right_on,
            how_var,
            suffix_x,
            suffix_y,
            is_join,  # True if this is DataFrame.join
            indicator,
            _,
            _,
            _,
        ) = args

        how = get_overload_const_str(how_var)

        # cross join output has all left/right input columns
        if how == "cross":
            data = left_df.data + right_df.data
            columns = left_df.columns + right_df.columns
            out_df = DataFrameType(
                data, RangeIndexType(types.none), columns, is_table_format=True
            )
            return signature(out_df, *args)

        left_on = get_overload_const_list(left_on)
        right_on = get_overload_const_list(right_on)

        # Map left_on and right_on to dictionaries for finding
        # key numbers later
        left_on_map = {c: i for i, c in enumerate(left_on)}
        right_on_map = {c: i for i, c in enumerate(right_on)}

        # columns with common name that are not common keys will get a suffix
        comm_keys = set(left_on) & set(right_on)
        comm_data = set(left_df.columns) & set(right_df.columns)
        add_suffix = comm_data - comm_keys

        # If $_bodo_index_ is found in either set of keys then we merge on the
        # index.
        left_index = "$_bodo_index_" in left_on
        right_index = "$_bodo_index_" in right_on

        # Determine if this is a left outer join or a right outer join.
        is_left = how in {"left", "outer"}
        is_right = how in {"right", "outer"}

        columns = []
        data = []

        # get key data types for merge on index cases
        if left_index or right_index:
            if left_index:
                left_key_type = bodo.utils.typing.get_index_data_arr_types(
                    left_df.index
                )[0]
            else:
                left_key_type = left_df.data[left_df.column_index[left_on[0]]]
            if right_index:
                right_key_type = bodo.utils.typing.get_index_data_arr_types(
                    right_df.index
                )[0]
            else:
                right_key_type = right_df.data[right_df.column_index[right_on[0]]]

        # merge between left_index and right column requires special
        # handling if the column also exists in left.
        if left_index and not right_index and not is_join.literal_value:
            right_key = right_on[0]
            if right_key in left_df.column_index:
                columns.append(right_key)
                if (
                    right_key_type == bodo.types.dict_str_arr_type
                    and left_key_type == bodo.types.string_array_type
                ):
                    # If we have a merge between a dict_array and a regular string
                    # array, the output needs to be a string array. This is because
                    # we will fall back to a string array for the join.
                    out_col = bodo.types.string_array_type
                else:
                    out_col = right_key_type
                data.append(out_col)

        # merge between right_index and left column requires special
        # handling if the column also exists in right.
        if right_index and not left_index and not is_join.literal_value:
            left_key = left_on[0]
            if left_key in right_df.column_index:
                columns.append(left_key)
                if (
                    left_key_type == bodo.types.dict_str_arr_type
                    and right_key_type == bodo.types.string_array_type
                ):
                    # If we have a merge between a dict_array and a regular string
                    # array, the output needs to be a string array. This is because
                    # we will fall back to a string array for the join.
                    out_col = bodo.types.string_array_type
                else:
                    out_col = left_key_type
                data.append(out_col)

        # The left side. All of it got included.
        for in_type, col in zip(left_df.data, left_df.columns):
            columns.append(
                str(col) + suffix_x.literal_value if col in add_suffix else col
            )
            if col in comm_keys:
                # For a common key we take either from left or right, so no additional NaN occurs.
                if in_type == bodo.types.dict_str_arr_type:
                    # If we have a dict array we need to check that the other table doesn't have a string
                    # array, otherwise we must use a regular string array.
                    in_type = right_df.data[right_df.column_index[col]]
                data.append(in_type)
            else:
                if in_type == bodo.types.dict_str_arr_type and col in left_on_map:
                    # If we have a dict array we need to check that the other table doesn't have a string
                    # array, otherwise we must use a regular string array.
                    if right_index:
                        in_type = right_key_type
                    else:
                        key_num = left_on_map[col]
                        right_key_name = right_on[key_num]
                        in_type = right_df.data[right_df.column_index[right_key_name]]
                if is_right:
                    # For a key that is not common OR data column, we have to plan for a NaN column
                    in_type = to_nullable_type(in_type)
                data.append(in_type)
        # The right side
        # common keys are added only once so avoid adding them
        for in_type, col in zip(right_df.data, right_df.columns):
            if col not in comm_keys:
                columns.append(
                    str(col) + suffix_y.literal_value if col in add_suffix else col
                )
                if in_type == bodo.types.dict_str_arr_type and col in right_on_map:
                    # If we have a dict array we need to check that the other table doesn't have a string
                    # array, otherwise we must use a regular string array.
                    if left_index:
                        in_type = left_key_type
                    else:
                        key_num = right_on_map[col]
                        left_key_name = left_on[key_num]
                        in_type = left_df.data[left_df.column_index[left_key_name]]
                if is_left:
                    # a key column that is not common needs to plan for NaN.
                    # Same for a data column of course.
                    in_type = to_nullable_type(in_type)
                data.append(in_type)

        # If indicator=True, add a column called "_merge", which is categorical
        # with Categories: ['left_only', 'right_only', 'both']
        indicator_value = get_overload_const_bool(indicator)
        if indicator_value:
            columns.append("_merge")
            data.append(
                bodo.types.CategoricalArrayType(
                    bodo.types.PDCategoricalDtype(
                        ("left_only", "right_only", "both"),
                        bodo.types.string_type,
                        False,
                    )
                )
            )
        # In the case of merging with left_index=True or right_index=True then
        # the index is coming from the other index. And so we need to set it adequately.
        index_typ = RangeIndexType(types.none)
        # Convert range index to numeric index
        convert_range = False
        index_to_nullable = False
        if left_index and right_index and not is_overload_const_str_equal(how, "asof"):
            index_typ = left_df.index
            convert_range = True
        elif left_index and not right_index:
            index_typ = right_df.index
            convert_range = True
            if is_left:
                # for left join on left Index, the output array corresponding to right
                # Index becomes the output dataframe's Index, which should be nullable
                # since left join may insert nulls.
                index_to_nullable = True
        elif right_index and not left_index:
            index_typ = left_df.index
            convert_range = True
            if is_right:
                index_to_nullable = True

        if convert_range and isinstance(
            index_typ, bodo.hiframes.pd_index_ext.RangeIndexType
        ):
            # If the index comes from one of the tables it will no longer be a range
            # as the entries will be shuffled
            index_typ = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)

        if index_to_nullable:
            index_typ = to_nullable_type(index_typ)

        out_df = DataFrameType(
            tuple(data), index_typ, tuple(columns), is_table_format=True
        )
        return signature(out_df, *args)


JoinTyper._no_unliteral = True  # type: ignore


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(join_dummy, types.VarArg(types.Any))
def lower_join_dummy(context, builder, sig, args):
    dataframe = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return dataframe._getvalue()


@overload(pd.concat, inline="always", no_unliteral=True)
@overload(bd.concat, inline="always", no_unliteral=True)
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

    # axis and ignore_index should be constant values
    if not is_overload_constant_int(axis):
        raise BodoError("pd.concat(): 'axis' should be a constant integer")
    if not is_overload_constant_bool(ignore_index):
        raise BodoError("pd.concat(): 'ignore_index' should be a constant boolean")

    axis = get_overload_const_int(axis)
    ignore_index = is_overload_true(ignore_index)

    unsupported_args = {
        "join": join,
        "join_axes": join_axes,
        "keys": keys,
        "levels": levels,
        "names": names,
        "verify_integrity": verify_integrity,
        "sort": sort,
        "copy": copy,
    }

    arg_defaults = {
        "join": "outer",
        "join_axes": None,
        "keys": None,
        "levels": None,
        "names": None,
        "verify_integrity": False,
        "sort": None,
        "copy": True,
    }
    check_unsupported_args(
        "pandas.concat",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="General",
    )

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
            check_runtime_cols_unsupported(obj, "pandas.concat()")
            if isinstance(obj, SeriesType):
                # TODO: use Series name if possible
                names.append(str(col_no))
                col_no += 1
                data_args.append(
                    f"bodo.hiframes.pd_series_ext.get_series_data(objs[{i}])"
                )
            else:  # DataFrameType
                names.extend(obj.columns)
                for j in range(len(obj.data)):
                    data_args.append(
                        f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(objs[{i}], {j})"
                    )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text, names, ", ".join(data_args), index
        )

    if axis != 0:
        raise_bodo_error("pd.concat(): axis must be 0 or 1")

    # dataframe tuples case
    if isinstance(objs, types.BaseTuple) and isinstance(objs.types[0], DataFrameType):
        assert all(isinstance(t, DataFrameType) for t in objs.types)

        # get output column names
        all_colnames = []
        for df in objs.types:
            check_runtime_cols_unsupported(df, "pandas.concat()")
            all_colnames.extend(df.columns)

        # remove duplicates but keep original order
        all_colnames = list(dict.fromkeys(all_colnames).keys())

        # get array types for all output columns (for NA generation for missing columns)
        arr_types = {}
        for col_no, c in enumerate(all_colnames):
            for i, df in enumerate(objs.types):
                if c in df.column_index:
                    arr_types[f"arr_typ{col_no}"] = df.data[df.column_index[c]]
                    break
        assert len(arr_types) == len(all_colnames)

        # generate concat for each output column
        for col_no, c in enumerate(all_colnames):
            args = []
            for i, df in enumerate(objs.types):
                if c in df.column_index:
                    col_ind = df.column_index[c]
                    args.append(
                        f"bodo.hiframes.pd_dataframe_ext.get_dataframe_data(objs[{i}], {col_ind})"
                    )
                else:
                    args.append(
                        f"bodo.libs.array_kernels.gen_na_array(len(objs[{i}]), arr_typ{col_no})"
                    )
            func_text += "  A{} = bodo.libs.array_kernels.concat(({},))\n".format(
                col_no, ", ".join(args)
            )
        if ignore_index:
            index = "bodo.hiframes.pd_index_ext.init_range_index(0, len(A0), 1, None)"
        else:
            index = "bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(({},)))\n".format(
                ", ".join(
                    f"bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(objs[{i}]))"
                    # ignore dummy string index of empty dataframes (test_append_empty_df)
                    for i in range(len(objs.types))
                    if len(objs[i].columns) > 0
                )
            )
        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text,
            all_colnames,
            ", ".join(f"A{i}" for i in range(len(all_colnames))),
            index,
            arr_types,
        )

    # series tuples case
    if isinstance(objs, types.BaseTuple) and isinstance(objs.types[0], SeriesType):
        assert all(isinstance(t, SeriesType) for t in objs.types)
        # TODO: index and name
        func_text += "  out_arr = bodo.libs.array_kernels.concat(({},))\n".format(
            ", ".join(
                f"bodo.hiframes.pd_series_ext.get_series_data(objs[{i}])"
                for i in range(len(objs.types))
            )
        )
        if ignore_index:
            func_text += "  index = bodo.hiframes.pd_index_ext.init_range_index(0, len(out_arr), 1, None)\n"
        else:
            func_text += "  index = bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(({},)))\n".format(
                ", ".join(
                    f"bodo.utils.conversion.index_to_array(bodo.hiframes.pd_series_ext.get_series_index(objs[{i}]))"
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
        check_runtime_cols_unsupported(objs.dtype, "pandas.concat()")
        df_type = objs.dtype

        # create output data columns
        if df_type.is_table_format:
            func_text += "  in_tables = []\n"
            func_text += "  for i in range(len(objs)):\n"
            func_text += "    df = objs[i]\n"
            func_text += "    in_tables.append(bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df))\n"
            func_text += (
                "  out_table = bodo.utils.table_utils.concat_tables(in_tables)\n"
            )
            data_args = "out_table"
        else:
            for col_no, c in enumerate(df_type.columns):
                func_text += f"  arrs{col_no} = []\n"
                func_text += "  for i in range(len(objs)):\n"
                func_text += "    df = objs[i]\n"
                func_text += f"    arrs{col_no}.append(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {col_no}))\n"
                func_text += f"  out_arr{col_no} = bodo.libs.array_kernels.concat(arrs{col_no})\n"
            data_args = ", ".join(f"out_arr{i}" for i in range(len(df_type.columns)))

        # create output Index
        if ignore_index:
            out_name = "out_table" if df_type.is_table_format else "out_arr0"
            index = f"bodo.hiframes.pd_index_ext.init_range_index(0, len({out_name}), 1, None)"
        else:
            func_text += "  arrs_index = []\n"
            func_text += "  for i in range(len(objs)):\n"
            func_text += "    df = objs[i]\n"
            func_text += "    arrs_index.append(bodo.utils.conversion.index_to_array(bodo.hiframes.pd_dataframe_ext.get_dataframe_index(df)))\n"
            # TODO: Update index name in all cases
            if objs.dtype.index.name_typ == types.none:
                name = None
            else:
                name = objs.dtype.index.name_typ.literal_value
            index = f"bodo.utils.conversion.index_from_array(bodo.libs.array_kernels.concat(arrs_index), {name!r})\n"

        return bodo.hiframes.dataframe_impl._gen_init_df(
            func_text,
            df_type.columns,
            data_args,
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
    raise BodoError(f"pd.concat(): input type {objs} not supported yet")


def sort_values_dummy(
    df,
    by,
    ascending,
    inplace,
    na_position,
    _bodo_chunk_bounds,
    _bodo_interval_sort,
):  # pragma: no cover
    pass


@infer_global(sort_values_dummy)
class SortDummyTyper(AbstractTemplate):
    def generic(self, args, kws):
        assert not kws
        df = args[0]

        index = df.index
        if isinstance(index, bodo.hiframes.pd_index_ext.RangeIndexType):
            index = bodo.hiframes.pd_index_ext.NumericIndexType(types.int64)

        ret_typ = df.copy(index=index)
        return signature(ret_typ, *args)


SortDummyTyper._no_unliteral = True  # type: ignore


# dummy lowering to avoid overload errors, remove after overload inline PR
# is merged
@lower_builtin(sort_values_dummy, types.VarArg(types.Any))
def lower_sort_values_dummy(context, builder, sig, args):
    if sig.return_type == types.none:
        return

    out_obj = cgutils.create_struct_proxy(sig.return_type)(context, builder)
    return out_obj._getvalue()


# TODO: jitoptions for overload_method and infer_global
# (no_cpython_wrapper to avoid error for iterator object)
@overload_method(
    DataFrameType,
    "itertuples",
    inline="always",
    no_unliteral=True,
    jit_options={"cache": True},
)
def itertuples_overload(df, index=True, name="Pandas"):
    check_runtime_cols_unsupported(df, "DataFrame.itertuples()")
    unsupported_args = {"index": index, "name": name}
    arg_defaults = {"index": True, "name": "Pandas"}
    check_unsupported_args(
        "DataFrame.itertuples",
        unsupported_args,
        arg_defaults,
        package_name="pandas",
        module_name="DataFrame",
    )

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


@numba.generated_jit(nopython=True, no_unliteral=True)
def pivot_impl(
    index_tup,
    columns_tup,
    values_tup,
    pivot_values,
    index_names,
    columns_name,
    value_names,
    check_duplicates=True,
    is_already_shuffled=False,
    _constant_pivot_values=None,
    parallel=False,
):  # pragma: no cover
    """
    Python implementation for pivot. This provides a parallel implementation
    by shuffling the table and constructs an output DataFrame where the columns
    may not be known at runtime.

    Pandas implements this by first creating a new DataFrame from the reduced values
    and then unstacking the DataFrame. We do something similar on each rank, but rather
    than create any intermediate DataFrames we immediately start updating the output arrays.

    https://github.com/pandas-dev/pandas/blob/ad190575aa75962d2d0eade2de81a5fe5a2e285b/pandas/core/reshape/pivot.py#L520
    https://github.com/pandas-dev/pandas/blob/ad190575aa75962d2d0eade2de81a5fe5a2e285b/pandas/core/reshape/reshape.py#L462
    https://github.com/pandas-dev/pandas/blob/ad190575aa75962d2d0eade2de81a5fe5a2e285b/pandas/_libs/reshape.pyx#L22


    index_tup - Tuple of arrays used for the index input.
    columns_tup - Tuple of arrays used for the columns input.
    values_tup - Tuple of arrays used for the values input.
    pivot_values - Replicated arrays of unique column names. Each entry
                   represents a unique output column. These values are already sorted
                   in ascending order on each rank.
    index_names - Typeref of ColNamesMetaType of names for the index arrays.
    columns_name - Typeref of ColNamesMetaType with the names for columns arrays.
    value_names - Typeref of ColNamesMetaType of names for the values arrays.
    check_duplicates - Do we need to add error checking for duplicates. This is a compile
                       time check to generate the bitmaps and can be skipped if we have
                       already done a reduction or removed duplicates (i.e. pivot_table).
    is_already_shuffled - Is the data already shuffled in a way that is consistent with the
                          output of a pivot table. This is a compile time check to skip the shuffle.
    _constant_pivot_values - Either None or Typeref of ColNamesMetaType of names for the output columns.
                             If its a ColNamesMetaType then the columns are known at compile time.
    parallel - Are index_tup, columns_tup, and values_tup distributed or replicated.
    """
    if not is_overload_constant_bool(check_duplicates):
        raise BodoError("pivot_impl(): check_duplicates must be a constant boolean")
    check_duplicates_codegen = get_overload_const_bool(check_duplicates)
    gen_shuffle_codegen = not get_overload_const_bool(is_already_shuffled)

    # Check if we have columns known at compile time.
    has_compile_time_columns = not is_overload_none(_constant_pivot_values)

    # Convert typerefs
    index_names = (
        index_names.instance_type
        if isinstance(index_names, types.TypeRef)
        else index_names
    )
    columns_name = (
        columns_name.instance_type
        if isinstance(columns_name, types.TypeRef)
        else columns_name
    )
    value_names = (
        value_names.instance_type
        if isinstance(value_names, types.TypeRef)
        else value_names
    )
    _constant_pivot_values = (
        _constant_pivot_values.instance_type
        if isinstance(_constant_pivot_values, types.TypeRef)
        else _constant_pivot_values
    )

    # Multi_index occurs when we have > 1 value name.
    use_multi_index = len(value_names) > 1

    # Initial literal names to lower. These will be updated if needed
    # in the generated code.
    index_names_lit = None
    value_names_lit = None
    columns_name_lit = None
    columns_typ = None

    # If we have a unituple of values we can generate a single list.
    use_single_list = isinstance(values_tup, types.UniTuple)
    # Convert the value arrays to nullable for empty values.
    if use_single_list:
        data_arr_typs = [to_str_arr_if_dict_array(to_nullable_type(values_tup.dtype))]
    else:
        data_arr_typs = [
            to_str_arr_if_dict_array(to_nullable_type(typ)) for typ in values_tup
        ]

    func_text = "def impl(\n"
    func_text += "    index_tup, columns_tup, values_tup, pivot_values, index_names, columns_name, value_names, check_duplicates=True, is_already_shuffled=False, _constant_pivot_values=None, parallel=False\n"
    func_text += "):\n"
    func_text += "    ev = tracing.Event('pivot_impl', is_parallel=parallel)\n"
    if gen_shuffle_codegen:
        # If the data is parallel we need to shuffle to get all values
        # in a row on the same rank.
        func_text += "    if parallel:\n"
        func_text += "        ev_shuffle = tracing.Event('shuffle_pivot_index')\n"
        # Shuffle based on index so each rank contains all values for
        # the same index
        array_to_infos = ", ".join(
            [f"array_to_info(index_tup[{i}])" for i in range(len(index_tup))]
            + [f"array_to_info(columns_tup[{i}])" for i in range(len(columns_tup))]
            + [f"array_to_info(values_tup[{i}])" for i in range(len(values_tup))]
        )
        func_text += f"        info_list = [{array_to_infos}]\n"
        func_text += "        cpp_table = arr_info_list_to_table(info_list)\n"
        # NOTE: C++ will delete cpp_table pointer
        func_text += f"        out_cpp_table = shuffle_table(cpp_table, {len(index_tup)}, parallel, 0)\n"
        index_info_to_arrays = ", ".join(
            [
                f"array_from_cpp_table(out_cpp_table, {i}, index_tup[{i}])"
                for i in range(len(index_tup))
            ]
        )
        columns_info_to_arrays = ", ".join(
            [
                f"array_from_cpp_table(out_cpp_table, {i + len(index_tup)}, columns_tup[{i}])"
                for i in range(len(columns_tup))
            ]
        )
        values_info_to_arrays = ", ".join(
            [
                f"array_from_cpp_table(out_cpp_table, {i + len(index_tup) + len(columns_tup)}, values_tup[{i}])"
                for i in range(len(values_tup))
            ]
        )
        func_text += f"        index_tup = ({index_info_to_arrays},)\n"
        func_text += f"        columns_tup = ({columns_info_to_arrays},)\n"
        func_text += f"        values_tup = ({values_info_to_arrays},)\n"
        # Delete the tables
        func_text += "        delete_table(out_cpp_table)\n"
        func_text += "        ev_shuffle.finalize()\n"
    # Load the index and column arrays. Move value arrays to a
    # list since access won't be known at compile time.
    func_text += "    columns_arr = columns_tup[0]\n"
    # If the values_tup is a unituple we can generate code as a list
    if use_single_list:
        func_text += "    values_arrs = [arr for arr in values_tup]\n"
    # Create a map on each rank for the index values.
    func_text += "    ev_unique = tracing.Event('pivot_unique_index_map', is_parallel=parallel)\n"
    func_text += "    unique_index_arr_tup, row_vector = bodo.libs.array_ops.array_unique_vector_map(\n"
    func_text += "        index_tup\n"
    func_text += "    )\n"
    func_text += "    n_rows = len(unique_index_arr_tup[0])\n"
    # Create a map for columns using the unique values
    func_text += "    num_values_arrays = len(values_tup)\n"
    func_text += "    n_unique_pivots = len(pivot_values)\n"
    if use_single_list:
        func_text += "    n_cols = num_values_arrays * n_unique_pivots\n"
    else:
        func_text += "    n_cols = n_unique_pivots\n"
    func_text += "    col_map = {}\n"
    func_text += "    for i in range(n_unique_pivots):\n"
    func_text += "        if bodo.libs.array_kernels.isna(pivot_values, i):\n"
    func_text += "            raise ValueError(\n"
    func_text += "                \"DataFrame.pivot(): NA values in 'columns' array not supported\"\n"
    func_text += "            )\n"
    func_text += "        col_map[pivot_values[i]] = i\n"
    func_text += "    ev_unique.finalize()\n"
    func_text += "    ev_alloc = tracing.Event('pivot_alloc', is_parallel=parallel)\n"
    # If we have a string array then we need to do 2 passes
    has_str_array = False
    for i, data_arr_typ in enumerate(data_arr_typs):
        if is_str_arr_type(data_arr_typ):
            has_str_array = True
            # Allocate arrays for the lengths
            func_text += f"    len_arrs_{i} = [np.zeros(n_rows, np.int64) for _ in range(n_cols)]\n"
            func_text += f"    total_lens_{i} = np.zeros(n_cols, np.int64)\n"
    if has_str_array:
        if check_duplicates_codegen:
            # Strings need to detect duplicates as soon as possible to avoid possible
            # segfaults with setitem.
            func_text += "    nbytes = (n_rows + 7) >> 3\n"
            # Bitmaps can be allocated per unique value rather than per output column.
            func_text += "    seen_bitmaps = [np.zeros(nbytes, np.int8) for _ in range(n_unique_pivots)]\n"
        # Get the lengths for each value
        func_text += "    for i in range(len(columns_arr)):\n"
        func_text += "        col_name = columns_arr[i]\n"
        # Determine which column we are in by checking the map
        func_text += "        pivot_idx = col_map[col_name]\n"
        func_text += "        row_idx = row_vector[i]\n"
        # If this value has already been seen raise an exception.
        if check_duplicates_codegen:
            func_text += "        seen_bitmap = seen_bitmaps[pivot_idx]\n"
            func_text += "        if bodo.libs.int_arr_ext.get_bit_bitmap_arr(seen_bitmap, row_idx):\n"
            func_text += "            raise ValueError(\"DataFrame.pivot(): 'index' contains duplicate entries for the same output column\")\n"
            func_text += "        else:\n"
            func_text += "            bodo.libs.int_arr_ext.set_bit_to_arr(seen_bitmap, row_idx, 1)\n"
        # Compute the lengths
        if use_single_list:
            func_text += "        for j in range(num_values_arrays):\n"
            func_text += "            col_idx = (j * len(pivot_values)) + pivot_idx\n"
            func_text += "            len_arr = len_arrs_0[col_idx]\n"
            func_text += "            values_arr = values_arrs[j]\n"
            func_text += (
                "            if not bodo.libs.array_kernels.isna(values_arr, i):\n"
            )
            func_text += "                str_val_len = bodo.libs.str_arr_ext.get_str_arr_item_length(values_arr, i)\n"
            func_text += "                len_arr[row_idx] = str_val_len\n"
            func_text += "                total_lens_0[col_idx] += str_val_len\n"
        else:
            # If we need to generate multiple lists, generate code per list
            for i, data_arr_typ in enumerate(data_arr_typs):
                if is_str_arr_type(data_arr_typ):
                    func_text += f"        if not bodo.libs.array_kernels.isna(values_tup[{i}], i):\n"
                    func_text += f"            str_val_len_{i} = bodo.libs.str_arr_ext.get_str_arr_item_length(values_tup[{i}], i)\n"
                    func_text += f"            len_arrs_{i}[pivot_idx][row_idx] = str_val_len_{i}\n"
                    func_text += (
                        f"            total_lens_{i}[pivot_idx] += str_val_len_{i}\n"
                    )

    # Allocate the data arrays. If we have string data we use the info from the first pass
    func_text += "    ev_alloc.add_attribute('num_rows', n_rows)\n"
    for i, data_arr_typ in enumerate(data_arr_typs):
        if is_str_arr_type(data_arr_typ):
            func_text += f"    data_arrs_{i} = [\n"
            func_text += "        bodo.libs.str_arr_ext.gen_na_str_array_lens(\n"
            func_text += f"            n_rows, total_lens_{i}[i], len_arrs_{i}[i]\n"
            func_text += "        )\n"
            func_text += "        for i in range(n_cols)\n"
            func_text += "    ]\n"
            func_text += "    if tracing.is_tracing():\n"
            func_text += "         for i in range(n_cols):\n"
            func_text += f"            ev_alloc.add_attribute('total_str_chars_out_column_{i}_' + str(i), total_lens_{i}[i])\n"
        else:
            func_text += f"    data_arrs_{i} = [\n"
            func_text += f"        bodo.libs.array_kernels.gen_na_array(n_rows, data_arr_typ_{i})\n"
            func_text += "        for _ in range(n_cols)\n"
            func_text += "    ]\n"

    if not has_str_array and check_duplicates_codegen:
        # We skip the seen bitmaps for strings because those were computed in the first
        # pass.
        func_text += "    nbytes = (n_rows + 7) >> 3\n"
        # Bitmaps can be allocated per unique value rather than per output column.
        func_text += "    seen_bitmaps = [np.zeros(nbytes, np.int8) for _ in range(n_unique_pivots)]\n"

    func_text += "    ev_alloc.finalize()\n"
    func_text += (
        "    ev_fill = tracing.Event('pivot_fill_data', is_parallel=parallel)\n"
    )

    # Set values that aren't NA
    func_text += "    for i in range(len(columns_arr)):\n"
    func_text += "        col_name = columns_arr[i]\n"
    # Determine which column we are in by checking the map
    func_text += "        pivot_idx = col_map[col_name]\n"
    func_text += "        row_idx = row_vector[i]\n"
    if not has_str_array and check_duplicates_codegen:
        # If this value has already been seen raise an exception.
        func_text += "        seen_bitmap = seen_bitmaps[pivot_idx]\n"
        func_text += "        if bodo.libs.int_arr_ext.get_bit_bitmap_arr(seen_bitmap, row_idx):\n"
        func_text += "            raise ValueError(\"DataFrame.pivot(): 'index' contains duplicate entries for the same output column\")\n"
        func_text += "        else:\n"
        func_text += "            bodo.libs.int_arr_ext.set_bit_to_arr(seen_bitmap, row_idx, 1)\n"
    # Set the Data using the row info.
    if use_single_list:
        func_text += "        for j in range(num_values_arrays):\n"
        func_text += "            col_idx = (j * len(pivot_values)) + pivot_idx\n"
        func_text += "            col_arr = data_arrs_0[col_idx]\n"
        func_text += "            values_arr = values_arrs[j]\n"
        func_text += "            bodo.libs.array_kernels.copy_array_element(col_arr, row_idx, values_arr, i)\n"
    else:
        # If we need to generate multiple lists, generate code per list
        for i, data_arr_typ in enumerate(data_arr_typs):
            func_text += f"        col_arr_{i} = data_arrs_{i}[pivot_idx]\n"
            func_text += f"        bodo.libs.array_kernels.copy_array_element(col_arr_{i}, row_idx, values_tup[{i}], i)\n"
    # Convert the index array to a proper index.
    if len(index_names) == 1:
        func_text += "    index = bodo.utils.conversion.index_from_array(unique_index_arr_tup[0], index_names_lit)\n"
        index_names_lit = index_names.meta[0]

    else:
        func_text += "    index = bodo.hiframes.pd_multi_index_ext.init_multi_index(unique_index_arr_tup, index_names_lit, None)\n"
        index_names_lit = tuple(index_names.meta)
    func_text += "    if tracing.is_tracing():\n"
    func_text += "        index_nbytes = index.nbytes\n"
    func_text += "        ev.add_attribute('index_nbytes', index_nbytes)\n"
    if not has_compile_time_columns:
        columns_name_lit = columns_name.meta[0]
        # Convert the columns to a proper index. if they are not known at compile time.
        if use_multi_index:
            # If we have a multi-index we have to update value_names and pivot_values for each entry.
            func_text += f"    num_rows = {len(value_names)} * len(pivot_values)\n"
            value_names_lit = value_names.meta
            if all(isinstance(c, str) for c in value_names_lit):
                value_names_lit = pd.array(value_names_lit, "string")
            elif all(isinstance(c, int) for c in value_names_lit):
                value_names_lit = np.array(value_names_lit, "int64")
            else:
                raise BodoError(
                    "pivot(): column names selected for 'values' must all share a common int or string type. Please convert your names to a common type using DataFrame.rename()"
                )

            if isinstance(value_names_lit.dtype, pd.StringDtype):
                func_text += "    total_chars = 0\n"
                func_text += f"    for i in range({len(value_names)}):\n"
                func_text += "        value_name_str_len = bodo.libs.str_arr_ext.get_str_arr_item_length(value_names_lit, i)\n"
                func_text += "        total_chars += value_name_str_len\n"
                func_text += "    new_value_names = bodo.libs.str_arr_ext.pre_alloc_string_array(num_rows, total_chars * len(pivot_values))\n"
            else:
                func_text += "    new_value_names = bodo.utils.utils.alloc_type(num_rows, value_names_lit, (-1,))\n"

            if is_str_arr_type(pivot_values):
                func_text += "    total_chars = 0\n"
                func_text += "    for i in range(len(pivot_values)):\n"
                func_text += "        pivot_val_str_len = bodo.libs.str_arr_ext.get_str_arr_item_length(pivot_values, i)\n"
                func_text += "        total_chars += pivot_val_str_len\n"
                func_text += f"    new_pivot_values = bodo.libs.str_arr_ext.pre_alloc_string_array(num_rows, total_chars * {len(value_names)})\n"
            else:
                func_text += "    new_pivot_values = bodo.utils.utils.alloc_type(num_rows, pivot_values, (-1,))\n"

            # Copy the data to the new arrays
            func_text += f"    for i in range({len(value_names)}):\n"
            func_text += "        for j in range(len(pivot_values)):\n"
            func_text += "            new_value_names[(i * len(pivot_values)) + j] = value_names_lit[i]\n"
            func_text += "            new_pivot_values[(i * len(pivot_values)) + j] = pivot_values[j]\n"
            func_text += "    column_index = bodo.hiframes.pd_multi_index_ext.init_multi_index((new_value_names, new_pivot_values), (None, columns_name_lit), None)\n"
        else:
            func_text += "    column_index =  bodo.utils.conversion.index_from_array(pivot_values, columns_name_lit)\n"
    func_text += "    ev_fill.finalize()\n"
    # Create the output Table and DataFrame.
    table_type = None
    if has_compile_time_columns:
        # Generate the column names for computing a DataFrame type.
        if use_multi_index:
            # Names are clustered by value name
            total_names = []
            for pivot_name in _constant_pivot_values.meta:
                for value_name in value_names.meta:
                    total_names.append((pivot_name, value_name))
            column_names = tuple(total_names)
        else:
            column_names = tuple(_constant_pivot_values.meta)
        columns_typ = ColNamesMetaType(column_names)
        # Each original column is repeated once per pivot value
        # and maintains the original location.
        total_typs = []
        for typ in data_arr_typs:
            total_typs.extend([typ] * len(_constant_pivot_values))
        data_values = tuple(total_typs)
        # Create the table type
        table_type = TableType(data_values)
        # Generate a table with constant types.
        func_text += "    table = bodo.hiframes.table.init_table(table_type, False)\n"
        func_text += "    table = bodo.hiframes.table.set_table_len(table, n_rows)\n"
        for i, typ in enumerate(data_arr_typs):
            # We support constant columns with multiple values
            func_text += f"    table = bodo.hiframes.table.set_table_block(table, data_arrs_{i}, {table_type.type_to_blk[typ]})\n"
        func_text += "    result = bodo.hiframes.pd_dataframe_ext.init_dataframe(\n"
        func_text += "        (table,), index, columns_typ\n"
        func_text += "    )\n"
    else:
        data_lists = ", ".join(f"data_arrs_{i}" for i in range(len(data_arr_typs)))
        func_text += f"    table = bodo.hiframes.table.init_runtime_table_from_lists(({data_lists},), n_rows)\n"
        func_text += (
            "    result = bodo.hiframes.pd_dataframe_ext.init_runtime_cols_dataframe(\n"
        )
        func_text += "        (table,), index, column_index\n"
        func_text += "    )\n"
    func_text += "    ev.finalize()\n"
    func_text += "    return result\n"
    loc_vars = {}
    data_types_dict = {
        f"data_arr_typ_{i}": data_arr_typ
        for i, data_arr_typ in enumerate(data_arr_typs)
    }
    glbls = {
        "bodo": bodo,
        "np": np,
        "array_to_info": array_to_info,
        "arr_info_list_to_table": arr_info_list_to_table,
        "shuffle_table": shuffle_table,
        "array_from_cpp_table": array_from_cpp_table,
        "delete_table": delete_table,
        "table_type": table_type,
        "columns_typ": columns_typ,
        "index_names_lit": index_names_lit,
        "value_names_lit": value_names_lit,
        "columns_name_lit": columns_name_lit,
        **data_types_dict,
        "tracing": tracing,
    }
    exec(func_text, glbls, loc_vars)
    impl = loc_vars["impl"]
    return impl


@overload_method(
    DataFrameType,
    "to_parquet",
    no_unliteral=True,
    # jit_options={"cache": True}
)
def to_parquet_overload(
    df,
    path,
    engine="auto",
    compression="snappy",
    index=None,
    partition_cols=None,
    storage_options=None,
    row_group_size=-1,
    _bodo_file_prefix="part-",
    _bodo_timestamp_tz=None,
    # TODO handle possible **kwargs options?
    _is_parallel=False,  # IMPORTANT: this is a Bodo parameter and must be in the last position
):
    check_unsupported_args(
        "DataFrame.to_parquet",
        {
            "storage_options": storage_options,
        },
        {
            "storage_options": None,
        },
        package_name="pandas",
        module_name="IO",
    )
    df = pt.cast(DataFrameType, df)

    # If a DataFrame has runtime columns then you cannot specify
    # partition_cols since the column names aren't known at compile
    # time.
    if df.has_runtime_cols and not is_overload_none(partition_cols):
        raise BodoError(
            "DataFrame.to_parquet(): Providing 'partition_cols' on DataFrames with columns determined at runtime is not yet supported. Please return the DataFrame to regular Python to update typing information."
        )
    if (
        not is_overload_none(engine)
        and is_overload_constant_str(engine)
        and get_overload_const_str(engine)
        not in (
            "auto",
            "pyarrow",
        )
    ):  # pragma: no cover
        raise BodoError("DataFrame.to_parquet(): only pyarrow engine supported")

    if (
        not is_overload_none(compression)
        and is_overload_constant_str(compression)
        and get_overload_const_str(compression) not in {"snappy", "gzip", "brotli"}
    ):
        raise BodoError(
            "to_parquet(): Unsupported compression: "
            + str(get_overload_const_str(compression))
        )

    if not is_overload_none(partition_cols):
        assert_bodo_error(is_overload_constant_list(partition_cols))
        partition_cols = get_overload_const_list(partition_cols)
        part_col_idxs = []
        for part_col_name in partition_cols:
            try:
                # TODO: Support index columns as partition columns
                idx = df.columns.index(part_col_name)
            except ValueError:
                raise BodoError(
                    f"Partition column `{part_col_name}` is not in dataframe"
                )
            part_col_idxs.append(idx)
    else:
        partition_cols = None

    if not is_overload_none(index) and not is_overload_constant_bool(index):
        raise BodoError("to_parquet(): index must be a constant bool or None")

    if not is_overload_int(row_group_size):
        raise BodoError("to_parquet(): row_group_size must be integer")
    # Users can use this to specify timestamp tz string name
    if not is_overload_none(_bodo_timestamp_tz) and (
        not is_overload_constant_str(_bodo_timestamp_tz)
        or not get_overload_const_str(_bodo_timestamp_tz)
    ):
        raise BodoError(
            "to_parquet(): _bodo_timestamp_tz must be None or a constant string"
        )

    from bodo.io.parquet_write import (
        gen_pandas_parquet_metadata,
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

    func_text = "def bodo_df_to_parquet(df, path, engine='auto', compression='snappy', index=None, partition_cols=None, storage_options=None, row_group_size=-1, _bodo_file_prefix='part-', _bodo_timestamp_tz=None, _is_parallel=False):\n"

    # Get all column names that will be written to parquet
    # Note, index columns are added later for the non-partition case
    # TODO: Extend for partitioned writes as well
    if df.has_runtime_cols:
        func_text += "    columns_index = get_dataframe_column_names(df)\n"
        # Note: C++ assumes the array is always a string array.
        func_text += "    col_names_arr = index_to_array(columns_index)\n"
    func_text += "    col_names = array_to_info(col_names_arr)\n"

    # Why we are calling drop_duplicates_local_dictionary on all dict encoded arrays?
    # Arrow doesn't support writing DictionaryArrays with nulls in the dictionary.
    # In most cases, we shouldn't end up with nulls in the dictionary, but it might
    # still happen in some array kernels.
    # `has_unique_local_dictionary` means there's no nulls in the dictionary.
    # It might be false even when there aren't any nulls, but even in those cases,
    # deduplicating locally shouldn't be very expensive.
    # TODO [BE-4383] Handle dict encoded array deduplication for runtime columns case

    # put arrays in table_info
    extra_globals = {}
    if df.is_table_format:
        func_text += "    py_table = get_dataframe_table(df)\n"
        if not df.has_runtime_cols:
            output_arr_typ = types.none
            extra_globals.update({"output_arr_typ": output_arr_typ})
            func_text += "    py_table = bodo.utils.table_utils.generate_mappable_table_func(py_table, 'bodo.libs.array_ops.drop_duplicates_local_dictionary_if_dict', output_arr_typ, False)\n"
        func_text += "    table = py_table_to_cpp_table(py_table, py_table_typ)\n"
    else:
        for i in range(len(df.data)):
            func_text += f"    arr{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i})\n"
            if df.data[i] == bodo.types.dict_str_arr_type:
                func_text += f"    arr{i} = bodo.libs.array.drop_duplicates_local_dictionary(arr{i}, False)\n"
        data_args = ", ".join(f"array_to_info(arr{i})" for i in range(len(df.columns)))
        func_text += f"    info_list = [{data_args}]\n"
        func_text += "    table = arr_info_list_to_table(info_list)\n"

    func_text += "    if compression is None:\n"
    func_text += "        compression = 'none'\n"
    func_text += "    if _bodo_timestamp_tz is None:\n"
    func_text += "        _bodo_timestamp_tz = ''\n"

    # if it's an s3 url, get the region and pass it into the c++ code
    func_text += "    bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(path, parallel=_is_parallel)\n"
    col_names_no_parts_arr = None
    if partition_cols:
        col_names_no_parts_arr = pd.array(
            [col_name for col_name in df.columns if col_name not in partition_cols]
        )
        # We need the values of the categories for any partition columns that
        # are categorical arrays, because they are used to generate the
        # output directory name
        categories_args = ", ".join(
            f"array_to_info(bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df, {i}).dtype.categories.values)"
            for i in range(len(df.columns))
            if isinstance(df.data[i], CategoricalArrayType) and (i in part_col_idxs)
        )
        if categories_args:
            func_text += f"    cat_info_list = [{categories_args}]\n"
            func_text += "    cat_table = arr_info_list_to_table(cat_info_list)\n"
        else:
            func_text += "    cat_table = 0\n"
        func_text += (
            "    col_names_no_partitions = array_to_info(col_names_no_parts_arr)\n"
        )
        func_text += f"    part_cols_idxs = np.array({part_col_idxs}, dtype=np.int32)\n"
        func_text += "    parquet_write_table_partitioned_cpp(unicode_to_utf8(path),\n"
        func_text += "                            table, col_names, col_names_no_partitions, cat_table,\n"
        func_text += (
            "                            part_cols_idxs.ctypes, len(part_cols_idxs),\n"
        )
        func_text += "                            unicode_to_utf8(compression),\n"
        func_text += "                            _is_parallel,\n"
        func_text += "                            unicode_to_utf8(bucket_region),\n"
        func_text += "                            row_group_size,\n"
        func_text += "                            unicode_to_utf8(_bodo_file_prefix),\n"
        func_text += (
            "                            unicode_to_utf8(_bodo_timestamp_tz))\n"
        )
    else:
        # Parquet needs to include all columns, including index columns
        if is_overload_true(index) or (
            is_overload_none(index) and write_non_rangeindex
        ):
            func_text += "    index_arr_list = index_to_array_list(df.index)\n"
            func_text += "    index_info_list = []\n"
            for i in range(df.index.nlevels):
                func_text += (
                    f"    index_info_list.append(array_to_info(index_arr_list[{i}]))\n"
                )
            func_text += (
                "    append_arr_info_list_to_cpp_table(table, index_info_list)\n"
            )
        func_text += (
            # Generate Pandas metadata string to store in the Parquet schema.
            # It requires both compile-time type info and runtime index name info
            f"    metadata, out_names_arr = gen_pandas_parquet_metadata(df, col_names_arr, partition_cols, {write_non_range_index_to_metadata}, {write_rangeindex_to_metadata})\n"
            # Update the columns list as well to include index columns
            "    col_names = array_to_info(out_names_arr)\n"
            # Actual write
            "    parquet_write_table_cpp(\n"
            "        unicode_to_utf8(path),\n"
            "        table, col_names,\n"
            "        unicode_to_utf8(metadata),\n"
            "        unicode_to_utf8(compression),\n"
            "        _is_parallel,\n"
            "        unicode_to_utf8(bucket_region),\n"
            "        row_group_size,\n"
            "        unicode_to_utf8(_bodo_file_prefix),\n"
            "        False,\n"  # convert_timedelta_to_int64
            "        unicode_to_utf8(_bodo_timestamp_tz),\n"
            "        False,\n"  # downcast_time_ns_to_us
            "        True)\n"  # create_dir
        )

    loc_vars = {}

    if df.has_runtime_cols:
        col_names_arr = None
    else:
        # Pandas raises a ValueError if columns aren't strings.
        # Similarly if columns aren't strings we have an a segfault
        # in C++.
        for col in df.columns:
            if not isinstance(col, str):
                # This is the Pandas error message.
                raise BodoError(
                    "DataFrame.to_parquet(): parquet must have string column names"
                )
        col_names_arr = pd.array(df.columns)

    glbls = {
        "np": np,
        "bodo": bodo,
        "unicode_to_utf8": unicode_to_utf8,
        "array_to_info": array_to_info,
        "arr_info_list_to_table": arr_info_list_to_table,
        "str_arr_from_sequence": str_arr_from_sequence,
        "parquet_write_table_cpp": parquet_write_table_cpp,
        "gen_pandas_parquet_metadata": gen_pandas_parquet_metadata,
        "parquet_write_table_partitioned_cpp": parquet_write_table_partitioned_cpp,
        "index_to_array": index_to_array,
        "col_names_arr": col_names_arr,
        "py_table_to_cpp_table": py_table_to_cpp_table,
        "py_table_typ": df.table_type,
        "get_dataframe_table": get_dataframe_table,
        "col_names_no_parts_arr": col_names_no_parts_arr,
        "get_dataframe_column_names": get_dataframe_column_names,
        "fix_arr_dtype": fix_arr_dtype,
        "decode_if_dict_array": decode_if_dict_array,
        "decode_if_dict_table": decode_if_dict_table,
        "index_to_array_list": index_to_array_list,
        "append_arr_info_list_to_cpp_table": append_arr_info_list_to_cpp_table,
    }
    glbls.update(extra_globals)
    return bodo_exec(
        func_text,
        glbls,
        loc_vars,
        __name__,
    )


# -------------------------------------- to_sql ------------------------------------------


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
    _is_table_create=False,
    _is_parallel=False,
):  # pragma: no cover
    """Call of to_sql and guard the exception and return it as string if error happens"""
    ev = tracing.Event("to_sql_exception_guard", is_parallel=_is_parallel)
    err_msg = "all_ok"
    # Find the db_type to determine if we are using Snowflake
    db_type, con_paswd = bodo.io.utils.parse_dbtype(con)

    if _is_parallel and bodo.get_rank() == 0:
        # Default number of rows to write to create the table. This is done in case
        # rank 0 has a large number of rows because that delays writing on other ranks.
        # TODO: Determine a reasonable threshold.
        default_chunksize = 100
        if chunksize is None:
            create_chunksize = default_chunksize
        else:
            # If the user provides a chunksize we then take the min of
            # the chunksize and our small default.
            # TODO: Should users be able to configure this???
            create_chunksize = min(chunksize, default_chunksize)

        # We may be creating a table. Truncate the DataFrame
        # the first default_chunksize rows
        if _is_table_create:
            df = df.iloc[:create_chunksize, :]
        # If have already created the DataFrame, only append the rows after the initial
        # creation
        else:
            df = df.iloc[create_chunksize:, :]
            # If there is no more data to write just return
            if len(df) == 0:
                return err_msg

    df_columns_original = df.columns
    try:
        # Pandas + SQLAlchemy per default save all object (string) columns as CLOB in Oracle DB,
        # which makes insertion extremely slow.
        # Stack overflow suggestion:
        # explicitly specify dtype for all DF columns of object dtype as VARCHAR
        # when saving DataFrames to Oracle DB.
        # See [BE-2770] and
        # https://stackoverflow.com/questions/42727990/speed-up-to-sql-when-writing-pandas-dataframe-to-oracle-database-using-sqlalch
        # If a column's dtype is identified as `object` by Pandas,
        # use Bodo DataFrame type to get its original dtype.
        if db_type == "oracle":
            import os

            import sqlalchemy as sa
            from sqlalchemy.dialects.oracle import VARCHAR2

            disable_varchar2 = os.environ.get("BODO_DISABLE_ORACLE_VARCHAR2", None)

            bodo_df_type = bodo.typeof(df)
            dtyp = {}
            for c, col_dtype in zip(bodo_df_type.columns, bodo_df_type.data):
                if df[c].dtype == "object":
                    if col_dtype == datetime_date_array_type:
                        dtyp[c] = sa.types.Date
                    elif col_dtype in (
                        bodo.types.string_array_type,
                        bodo.types.dict_str_arr_type,
                    ) and (not disable_varchar2 or disable_varchar2 == "0"):
                        dtyp[c] = VARCHAR2(4000)
                # workaround to avoid issue with Oracle and Float values
                # See discussion https://github.com/sqlalchemy/sqlalchemy/discussions/9667
                # Ticket in Pandas https://github.com/pandas-dev/pandas/issues/52715
                elif df[c].dtype in ["float", "float64"]:
                    dtyp[c] = sa.FLOAT
            dtype = dtyp

        try:
            ev_df_to_sql = tracing.Event("df_to_sql", is_parallel=_is_parallel)
            df.to_sql(
                name,
                con,
                schema=schema,
                if_exists=if_exists,
                index=index,
                index_label=index_label,
                chunksize=chunksize,
                dtype=dtype,
                method=method,
            )
            ev_df_to_sql.finalize()
        except Exception as e:
            err_msg = e.args[0]
            if db_type == "oracle" and "ORA-12899" in err_msg:
                err_msg += """
                String is larger than VARCHAR2 maximum length.
                Please set environment variable `BODO_DISABLE_ORACLE_VARCHAR2` to
                disable Bodo's optimziation use of VARCHA2.
                NOTE: Oracle `to_sql` with CLOB datatypes is known to be really slow.
                """
        return err_msg

    finally:
        df.columns = df_columns_original
        ev.finalize()


@numba.njit(cache=True)
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
    _is_table_create=False,
    _is_parallel=False,
):  # pragma: no cover
    ev = tracing.Event("to_sql_exception_guard_encaps", is_parallel=_is_parallel)
    with bodo.ir.object_mode.no_warning_objmode(out="unicode_type"):
        ev_objmode = tracing.Event(
            "to_sql_exception_guard_encaps:objmode", is_parallel=_is_parallel
        )
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
            _is_table_create,
            _is_parallel,
        )
        ev_objmode.finalize()
    ev.finalize()
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
    # Custom Bodo Arguments
    _bodo_allow_downcasting=False,
    _bodo_create_table_type="",
    # Additional entry
    _is_parallel=False,
):
    """
    _bodo_create_table_type: str. Only used when creating a snowflake table. Must be one of "", "TRANSIENT", or "TEMPORARY"
    """

    import warnings

    # Currently to_sql (Iceberg, Snowflake, SQL writes) does not support
    # writing dfs with runtime columns. DataFrameType has the invariant:
    # df.columns is None and df.data is None iff df.has_runtime_columns
    check_runtime_cols_unsupported(df, "DataFrame.to_sql()")
    df: DataFrameType = df
    assert df.columns is not None and df.data is not None

    if is_overload_none(schema):
        if bodo.get_rank() == 0:
            warnings.warn(
                BodoWarning(
                    "DataFrame.to_sql(): schema argument is recommended to avoid permission issues when writing the table."
                )
            )

    if not (is_overload_none(chunksize) or isinstance(chunksize, types.Integer)):
        raise BodoError(
            "DataFrame.to_sql(): 'chunksize' argument must be an integer if provided."
        )

    # Snowflake write imports
    # We need to import so that the types are in numba's type registry
    # when executing the code.
    from bodo.io.parquet_write import parquet_write_table_cpp
    from bodo.io.snowflake import snowflake_connector_cursor_python_type  # noqa

    extra_globals = {}

    # Pandas raises a ValueError if columns aren't strings.
    # Similarly if columns aren't strings we have a segfault in C++.
    for col in df.columns:
        if not isinstance(col, str):
            # This is the Pandas error message
            raise BodoError(
                "DataFrame.to_sql(): input dataframe must have string column names. "
                "Please return the DataFrame with runtime column names to regular "
                "Python to modify column names."
            )
    col_names_arr = pd.array(df.columns)

    func_text = (
        "def df_to_sql(\n"
        "    df, name, con,\n"
        "    schema=None, if_exists='fail', index=True,\n"
        "    index_label=None, chunksize=None, dtype=None,\n"
        '    method=None, _bodo_allow_downcasting=False, _bodo_create_table_type="",\n'
        "    _is_parallel=False,\n"
        "):\n"
    )

    # ------------------------------ Iceberg Write -----------------------------
    func_text += (
        "    if con.startswith('iceberg'):\n"
        "        if schema is None:\n"
        "            raise ValueError('DataFrame.to_sql(): schema must be provided when writing to an Iceberg table.')\n"
        "        if chunksize is not None:\n"
        "            raise ValueError('DataFrame.to_sql(): chunksize not supported for Iceberg tables.')\n"
        "        if index and bodo.get_rank() == 0:\n"
        "            warnings.warn('index is not supported for Iceberg tables.')\n"
        "        if index_label is not None and bodo.get_rank() == 0:\n"
        "            warnings.warn('index_label is not supported for Iceberg tables.')\n"
    )

    # XXX A lot of this is copied from the to_parquet impl, so might be good to refactor

    if df.is_table_format:
        func_text += "        py_table = get_dataframe_table(df)\n"
        if not df.has_runtime_cols:
            output_arr_typ = types.none
            extra_globals.update({"output_arr_typ": output_arr_typ})
            # Call drop_duplicates_local_dictionary on all dict-encoded arrays.
            # See note in `to_parquet_overload` (Why we are calling drop_duplicates_local_dictionary
            # on all dict encoded arrays?) for why this is important.
            func_text += "        py_table = bodo.utils.table_utils.generate_mappable_table_func(py_table, 'bodo.libs.array_ops.drop_duplicates_local_dictionary_if_dict', output_arr_typ, False)\n"
        func_text += "        table = py_table_to_cpp_table(py_table, py_table_typ)\n"
    else:
        for i in range(len(df.data)):
            func_text += f"        arr{i} = get_dataframe_data(df, {i})\n"
            # Call drop_duplicates_local_dictionary on all dict-encoded arrays.
            # See note in `to_parquet_overload` (Why we are calling drop_duplicates_local_dictionary
            # on all dict encoded arrays?) for why this is important.
            if df.data[i] == bodo.types.dict_str_arr_type:
                func_text += f"        arr{i} = bodo.libs.array.drop_duplicates_local_dictionary(arr{i}, False)\n"
        data_args = ", ".join(f"array_to_info(arr{i})" for i in range(len(df.columns)))
        func_text += f"        info_list = [{data_args}]\n"
        func_text += "        table = arr_info_list_to_table(info_list)\n"

    # We don't write pandas metadata for Iceberg (at least for now)
    # Partition columns not supported through this API.
    func_text += (
        "        col_names = array_to_info(col_names_arr)\n"
        "        table_id = name if schema == '' else f'{schema}.{name}'\n"
        "        bodo.io.iceberg.write.iceberg_write(\n"
        "            con, table_id, table, col_names,\n"
        "            if_exists, _is_parallel, pyarrow_table_schema,\n"
        "           _bodo_allow_downcasting,\n"
        "        )\n"
    )

    # ----------------------------- Snowflake Write ----------------------------
    # Design doc link: https://bodo.atlassian.net/wiki/spaces/B/pages/1077280785/Snowflake+Distributed+Write
    func_text += "    elif con.startswith('snowflake'):\n"
    func_text += (
        "        if index and bodo.get_rank() == 0:\n"
        "            warnings.warn('index is not supported for Snowflake tables.')      \n"
        "        if index_label is not None and bodo.get_rank() == 0:\n"
        "            warnings.warn('index_label is not supported for Snowflake tables.')\n"
        "        if _bodo_allow_downcasting and bodo.get_rank() == 0:\n"
        "            warnings.warn('_bodo_allow_downcasting is not supported for Snowflake tables.')\n"
        "        ev = tracing.Event('snowflake_write_impl', sync=False)\n"
    )

    # Compute table location, qualified and quoted
    func_text += "        location = ''\n"
    if not is_overload_none(schema):
        func_text += "        location += '\"' + schema + '\".'\n"
    func_text += "        location += name\n"
    func_text += "        my_rank = bodo.get_rank()\n"

    # In object mode: Connect to snowflake, create internal stage, and
    # get internal stage credentials on each rank
    func_text += (
        "        with bodo.ir.object_mode.no_warning_objmode(\n"
        "            cursor='snowflake_connector_cursor_type',\n"
        "            tmp_folder='temporary_directory_type',\n"
        "            stage_name='unicode_type',\n"
        "            parquet_path='unicode_type',\n"
        "            upload_using_snowflake_put='boolean',\n"
        "            old_creds='DictType(unicode_type, unicode_type)',\n"
        "        ):\n"
        "            (\n"
        "                cursor, tmp_folder, stage_name, parquet_path, upload_using_snowflake_put, old_creds,\n"
        "            ) = bodo.io.snowflake.connect_and_get_upload_info(con)\n"
    )

    # Barrier ensures that internal stage exists before we upload files to it
    func_text += "        bodo.barrier()\n"

    # Estimate chunk size by repeating internal implementation of `df.memory_usage()`.
    # Calling `df.memory_usage()` provides much poorer performance as the call
    # does not seem to get inlined, causing lazy boxing overheads to get incurred
    # within `df.memory_usage()`.
    func_text += "        if chunksize is None:\n"
    func_text += "            ev_estimate_chunksize = tracing.Event('estimate_chunksize')          \n"
    if df.is_table_format and len(df.columns) > 0:
        # Don't use table format if the table is unused
        func_text += (
            f"            nbytes_arr = np.empty({len(df.columns)}, np.int64)\n"
            f"            table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)\n"
            f"            bodo.utils.table_utils.generate_table_nbytes(table, nbytes_arr, 0)\n"
            f"            memory_usage = np.sum(nbytes_arr)\n"
        )
    else:
        data = ", ".join(
            f"bodo.libs.array_ops.array_op_nbytes(get_dataframe_data(df, {i}))"
            for i in range(len(df.columns))
        )
        comma = "," if len(df.columns) == 1 else ""
        func_text += (
            f"            memory_usage = np.array(({data}{comma}), np.int64).sum()\n"
        )
    func_text += (
        "            nsplits = int(max(1, memory_usage / bodo.io.snowflake.SF_WRITE_PARQUET_CHUNK_SIZE))\n"
        "            chunksize = max(1, (len(df) + nsplits - 1) // nsplits)\n"
        "            ev_estimate_chunksize.finalize()\n"
    )

    extra_globals.update(
        {
            "__col_name_meta_value_df_to_sql": ColNamesMetaType(df.columns),
        }
    )
    # Call drop_duplicates_local_dictionary on all dict-encoded arrays.
    # See note in `to_parquet_overload` (Why we are calling drop_duplicates_local_dictionary
    # on all dict encoded arrays?) for why this is important.
    if df.is_table_format:
        if not df.has_runtime_cols:
            output_arr_typ = types.none
            extra_globals.update(
                {
                    "output_arr_typ": output_arr_typ,
                }
            )
            func_text += "        table = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df)\n"
            func_text += "        table = bodo.utils.table_utils.generate_mappable_table_func(table, 'bodo.libs.array_ops.drop_duplicates_local_dictionary_if_dict', output_arr_typ, False)\n"
            func_text += "        df = bodo.hiframes.pd_dataframe_ext.init_dataframe((table,), df.index, __col_name_meta_value_df_to_sql)\n"
    else:
        for i in range(len(df.data)):
            func_text += f"        arr{i} = get_dataframe_data(df, {i})\n"
            if df.data[i] == bodo.types.dict_str_arr_type:
                func_text += f"        arr{i} = bodo.libs.array.drop_duplicates_local_dictionary(arr{i}, False)\n"
        data_args = ", ".join([f"arr{i}" for i in range(len(df.data))])
        func_text += f"        df = bodo.hiframes.pd_dataframe_ext.init_dataframe(({data_args},), df.index, __col_name_meta_value_df_to_sql)\n"

    if df.has_runtime_cols:
        func_text += "        columns_index = get_dataframe_column_names(df)\n"
        func_text += "        names_arr = index_to_array(columns_index)\n"

    func_text += "        bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(parquet_path, parallel=_is_parallel)\n"

    # On all ranks, write local dataframe chunk to S3/ADLS, or a local file
    # upon fallback to snowflake PUT. In the fallback case, we execute the
    # PUT command later to perform the actual upload
    func_text += "        ev_upload_df = tracing.Event('upload_df', is_parallel=False)           \n"
    func_text += "        num_files = len(range(0, len(df), chunksize))\n"
    func_text += "        for chunk_idx, i in enumerate(range(0, len(df), chunksize)):           \n"

    # Create a unique filename for uploaded chunk with quotes/backslashes escaped
    # We ensure that `parquet_path` always has a trailing slash in `connect_and_get_upload_info`
    func_text += "            chunk_name = f'file{chunk_idx}_rank{my_rank}_{bodo.io.helpers.uuid4_helper()}.parquet'\n"
    # Azure paths can have a query string at the end.
    # We need to append the chunk name before the query string.
    func_text += "            if parquet_path.startswith('abfs') and '?' in parquet_path:  # Azure\n"
    func_text += "                container_path, query = parquet_path.split('?')\n"
    func_text += (
        "                chunk_path = container_path + chunk_name + '?' + query\n"
    )
    func_text += "            else:\n"
    func_text += "                chunk_path = parquet_path + chunk_name\n"
    # To escape backslashes, we want to replace ( \ ) with ( \\ ), so the func_text
    # should contain the string literals ( \\ ) and ( \\\\ ). To add these to func_text,
    # we need to write ( \\\\ ) and ( \\\\\\\\ ) here.
    # To escape quotes, we want to replace ( ' ) with ( \' ), so the func_text
    # should contain the string literals ( ' ) and ( \\' ). To add these to func_text,
    # we need to write ( \' ) and ( \\\\\' ) here.
    func_text += '            chunk_path = chunk_path.replace("\\\\", "\\\\\\\\")\n'
    func_text += '            chunk_path = chunk_path.replace("\'", "\\\\\'")\n'

    # Convert dataframe chunk to cpp table
    # TODO Using df.iloc below incurs significant boxing/unboxing overhead.
    # Perhaps we can avoid iloc by creating a single C++ table up front
    # and directly indexing into it.
    func_text += "            ev_to_df_table = tracing.Event(f'to_df_table_{chunk_idx}', is_parallel=False)\n"
    func_text += "            chunk = df.iloc[i : i + chunksize]\n"
    if df.is_table_format:
        func_text += "            py_table_chunk = get_dataframe_table(chunk)\n"
        func_text += "            table_chunk = py_table_to_cpp_table(py_table_chunk, py_table_typ)\n"
    else:
        data_args_chunk = ", ".join(
            f"array_to_info(get_dataframe_data(chunk, {i}))"
            for i in range(len(df.columns))
        )
        func_text += f"            table_chunk = arr_info_list_to_table([{data_args_chunk}])     \n"
    func_text += "            ev_to_df_table.finalize()\n"

    # In C++: Upload dataframe chunks on each rank to internal stage.
    # Compute column names and other required info for parquet_write_table_cpp
    if df.has_runtime_cols:
        func_text += "            col_names = array_to_info(names_arr)\n"
    else:
        func_text += "            col_names = array_to_info(col_names_arr)\n"

    # Dump chunks to parquet file
    # Below, we always pass `is_parallel=False`. Passing `is_parallel=True` would cause
    # `pq_write` to write a directory of partitioned files, rather than a single file,
    # which is what we want as `pq_write` is being called separately and independently
    # from each rank and we're already accounting for the partitioning ourselves..
    func_text += (
        "            ev_pq_write_cpp = tracing.Event(f'pq_write_cpp_{chunk_idx}', is_parallel=False)\n"
        "            ev_pq_write_cpp.add_attribute('chunk_start', i)\n"
        "            ev_pq_write_cpp.add_attribute('chunk_end', i + len(chunk))\n"
        "            ev_pq_write_cpp.add_attribute('chunk_size', len(chunk))\n"
        "            ev_pq_write_cpp.add_attribute('chunk_path', chunk_path)\n"
        "            parquet_write_table_cpp(\n"
        "                unicode_to_utf8(chunk_path),\n"
        "                table_chunk, col_names,\n"
        "                unicode_to_utf8('null'),\n"  # metadata
        "                unicode_to_utf8(bodo.io.snowflake.SF_WRITE_PARQUET_COMPRESSION),\n"
        "                False,\n"  # is_parallel
        "                unicode_to_utf8(bucket_region),\n"
        # We set the row group size equal to chunksize to force this parquet to
        # be written as one row group. Due to prior chunking, the whole parquet
        # file is already a reasonable size for one row group.
        "                chunksize,\n"  # row_group_size
        "                unicode_to_utf8('null'),\n"  # prefix
        "                True,\n"  # Explicitly cast timedelta to int64 in the bodo_array_to_arrow step (convert_timedelta_to_int64)
        "                unicode_to_utf8('UTC'),\n"  # Explicitly set tz='UTC' for snowflake write. see [BE-3530]
        "                True,\n"  # Explicitly downcast nanoseconds to microseconds (See gen_snowflake_schema comment)
        "                True,\n"  # Create directory
        "            )\n"
        "            ev_pq_write_cpp.finalize()\n"
        # If needed, upload local parquet to internal stage using objmode PUT
        "            if upload_using_snowflake_put:\n"
        "                with bodo.ir.object_mode.no_warning_objmode():\n"
        "                    bodo.io.snowflake.do_upload_and_cleanup(\n"
        "                        cursor, chunk_idx, chunk_path, stage_name,\n"
        "                    )\n"
        "        ev_upload_df.finalize()\n"
    )

    # Barrier ensures that files are copied into internal stage before COPY_INTO
    func_text += "        bodo.barrier()\n"

    # Generate snowflake schema from bodo datatypes.
    sf_schema = bodo.io.snowflake.gen_snowflake_schema(df.columns, df.data)

    # Compute the total number of files written.
    func_text += (
        "        sum_op = np.int32(bodo.libs.distributed_api.Reduce_Type.Sum.value)\n"
    )
    func_text += "        num_files_global = bodo.libs.distributed_api.dist_reduce(num_files, np.int32(sum_op))\n"
    # In object mode on rank 0: Create a new table if needed, execute COPY_INTO,
    # and clean up created internal stage.
    func_text += (
        # This is because it seems like globals aren't passed into objmode
        "        df_data_ = df_data\n"
        "        with bodo.ir.object_mode.no_warning_objmode():\n"
        "            bodo.io.snowflake.create_table_copy_into(\n"
        # Creating the dict here instead of passing it in as a globall is necessary because when boxing
        # dicts we check if they contain type references in the value and throw an error, we need typerefs here
        f"                cursor, stage_name, location, {sf_schema}, dict(zip(df.columns, df_data_)),\n"
        "                if_exists, _bodo_create_table_type, num_files_global, old_creds, tmp_folder,\n"
        "            )\n"
    )
    func_text += "        ev.finalize()\n"

    # -------------------------- Default to_sql Impl --------------------------
    func_text += "    else:\n"
    func_text += "        if _bodo_allow_downcasting and bodo.get_rank() == 0:\n"
    func_text += "            warnings.warn('_bodo_allow_downcasting is not supported for SQL tables.')\n"

    # Nodes number 0 does the first initial insertion into the database.
    # Following nodes do the insertion of the rest if no error happened.
    # The bcast_scalar is used to synchronize the process between 0 and the rest.
    func_text += "        rank = bodo.libs.distributed_api.get_rank()\n"
    func_text += "        err_msg = 'unset'\n"

    # Rank 0 writes first and we wait for a response. If this is done in parallel,
    # rank 0 may need to create the table, otherwise if this is replicated only
    # rank 0 writes and the other ranks wait to propagate any error message.
    func_text += "        if rank != 0:\n"
    func_text += "            err_msg = bodo.libs.distributed_api.bcast_scalar(err_msg)          \n"

    # Rank 0 creates the table. This only writes a chunk of the data
    # to enable further parallelism if data isn't replicated.
    func_text += "        elif rank == 0:\n"
    func_text += "            err_msg = to_sql_exception_guard_encaps(\n"
    func_text += "                          df, name, con, schema, if_exists, index, index_label,\n"
    func_text += "                          chunksize, dtype, method,\n"
    func_text += "                          True, _is_parallel,\n"
    func_text += "                      )\n"
    func_text += "            err_msg = bodo.libs.distributed_api.bcast_scalar(err_msg)          \n"

    # For all nodes we append to existing table after rank 0 creates the table.
    func_text += "        if_exists = 'append'\n"

    # The writing of the rest of the data. If data isn't parallel, then
    # rank 0 has already written all of the data in the previous call.
    # TODO: We cannot do a simple raise ValueError(err_msg).
    func_text += "        if _is_parallel and err_msg == 'all_ok':\n"
    func_text += "            err_msg = to_sql_exception_guard_encaps(\n"
    func_text += "                          df, name, con, schema, if_exists, index, index_label,\n"
    func_text += "                          chunksize, dtype, method,\n"
    func_text += "                          False, _is_parallel,\n"
    func_text += "                      )\n"
    func_text += "        if err_msg != 'all_ok':\n"
    func_text += "            print('err_msg=', err_msg)\n"
    func_text += "            raise ValueError('error in to_sql() operation')\n"
    loc_vars = {}
    glbls = globals().copy()
    glbls.update(
        {
            "arr_info_list_to_table": arr_info_list_to_table,
            "array_to_info": array_to_info,
            "bodo": bodo,
            "col_names_arr": col_names_arr,
            "get_dataframe_column_names": get_dataframe_column_names,
            "get_dataframe_data": get_dataframe_data,
            "get_dataframe_table": get_dataframe_table,
            "index_to_array": index_to_array,
            "np": np,
            "parquet_write_table_cpp": parquet_write_table_cpp,
            "py_table_to_cpp_table": py_table_to_cpp_table,
            "py_table_typ": df.table_type,
            "pyarrow_table_schema": bodo.io.helpers.numba_to_pyarrow_schema(
                df, is_iceberg=True
            ),
            "time": time,
            "to_sql_exception_guard_encaps": to_sql_exception_guard_encaps,
            "tracing": tracing,
            "unicode_to_utf8": unicode_to_utf8,
            "warnings": warnings,
            "df_data": df.data,
        }
    )
    glbls.update(extra_globals)
    exec(func_text, glbls, loc_vars)
    _impl = loc_vars["df_to_sql"]
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
    compression=None,  # this is different from pandas, default is 'infer'.
    quoting=None,
    quotechar='"',
    lineterminator=None,
    chunksize=None,
    date_format=None,
    doublequote=True,
    escapechar=None,
    decimal=".",
    errors="strict",
    storage_options=None,
    _bodo_file_prefix="part-",
    # Concatenate string output on rank 0 if set (used in spawn mode)
    _bodo_concat_str_output=False,
):
    check_runtime_cols_unsupported(df, "DataFrame.to_csv()")
    check_unsupported_args(
        "DataFrame.to_csv",
        {
            "encoding": encoding,
            "mode": mode,
            "errors": errors,
            "storage_options": storage_options,
        },
        {
            "encoding": None,
            "mode": "w",
            "errors": "strict",
            "storage_options": None,
        },
        package_name="pandas",
        module_name="IO",
    )

    if not (
        is_overload_none(path_or_buf)
        or is_overload_constant_str(path_or_buf)
        or path_or_buf == string_type
    ):
        raise BodoError(
            "DataFrame.to_csv(): 'path_or_buf' argument should be None or string"
        )
    if not is_overload_none(compression):
        raise BodoError(
            "DataFrame.to_csv(): 'compression' argument supports only None, which is the default in JIT code."
        )
    # best effort warning that compression defaults to None, when users pass in filepaths that would normally be compressed
    # in pandas.
    if is_overload_constant_str(path_or_buf):
        filepath = get_overload_const_str(path_or_buf)
        if filepath.endswith((".gz", ".bz2", ".zip", ".xz")):
            import warnings

            from bodo.utils.typing import BodoWarning

            warnings.warn(
                BodoWarning(
                    "DataFrame.to_csv(): 'compression' argument defaults to None in JIT code, which is the only supported value."
                )
            )
    if not (
        is_overload_none(columns) or isinstance(columns, (types.List, types.Tuple))
    ):
        raise BodoError(
            "DataFrame.to_csv(): 'columns' argument must be list a or tuple type."
        )

    # TODO: refactor when objmode() can understand global string constant
    # String output case
    if is_overload_none(path_or_buf):
        # NOTE: using a separate path for _bodo_concat_str_output since gatherv fails
        # for categorical arrays with non-constant categories so avoiding gatherv
        # compilation as much as possible.
        # See BSE-4713
        assert is_overload_constant_bool(_bodo_concat_str_output), (
            "to_csv: _bodo_concat_str_output should be constant bool"
        )

        if is_overload_true(_bodo_concat_str_output):

            def _impl_concat_str(
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
                compression=None,  # this is different from pandas, default is 'infer'.
                quoting=None,
                quotechar='"',
                lineterminator=None,
                chunksize=None,
                date_format=None,
                doublequote=True,
                escapechar=None,
                decimal=".",
                errors="strict",
                storage_options=None,
                _bodo_file_prefix="part-",
                _bodo_concat_str_output=False,
            ):  # pragma: no cover
                # Return the concatenated string output on rank 0
                # and empty string on all other ranks
                df = bodo.gatherv(df)
                if bodo.get_rank() != 0:
                    return ""

                with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
                    D = df.to_csv(
                        path_or_buf,
                        sep=sep,
                        na_rep=na_rep,
                        float_format=float_format,
                        columns=columns,
                        header=header,
                        index=index,
                        index_label=index_label,
                        mode=mode,
                        encoding=encoding,
                        compression=compression,
                        quoting=quoting,
                        quotechar=quotechar,
                        lineterminator=lineterminator,
                        chunksize=chunksize,
                        date_format=date_format,
                        doublequote=doublequote,
                        escapechar=escapechar,
                        decimal=decimal,
                        errors=errors,
                        storage_options=storage_options,
                    )
                return D

            return _impl_concat_str

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
            compression=None,  # this is different from pandas, default is 'infer'.
            quoting=None,
            quotechar='"',
            lineterminator=None,
            chunksize=None,
            date_format=None,
            doublequote=True,
            escapechar=None,
            decimal=".",
            errors="strict",
            storage_options=None,
            _bodo_file_prefix="part-",
            _bodo_concat_str_output=False,
        ):  # pragma: no cover
            with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
                D = df.to_csv(
                    path_or_buf,
                    sep=sep,
                    na_rep=na_rep,
                    float_format=float_format,
                    columns=columns,
                    header=header,
                    index=index,
                    index_label=index_label,
                    mode=mode,
                    encoding=encoding,
                    compression=compression,
                    quoting=quoting,
                    quotechar=quotechar,
                    lineterminator=lineterminator,
                    chunksize=chunksize,
                    date_format=date_format,
                    doublequote=doublequote,
                    escapechar=escapechar,
                    decimal=decimal,
                    errors=errors,
                    storage_options=storage_options,
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
        compression=None,  # this is different from pandas, default is 'infer'.
        quoting=None,
        quotechar='"',
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors="strict",
        storage_options=None,
        _bodo_file_prefix="part-",
        _bodo_concat_str_output=False,
    ):  # pragma: no cover
        # passing None for the first argument returns a string
        # containing contents to write to csv
        with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
            D = df.to_csv(
                None,
                sep=sep,
                na_rep=na_rep,
                float_format=float_format,
                columns=columns,
                header=header,
                index=index,
                index_label=index_label,
                mode=mode,
                encoding=encoding,
                compression=compression,
                quoting=quoting,
                quotechar=quotechar,
                lineterminator=lineterminator,
                chunksize=chunksize,
                date_format=date_format,
                doublequote=doublequote,
                escapechar=escapechar,
                decimal=decimal,
                errors=errors,
                storage_options=storage_options,
            )

        bodo.io.helpers.csv_write(path_or_buf, D, _bodo_file_prefix)

    return _impl


@overload_method(DataFrameType, "to_json", no_unliteral=True)
def to_json_overload(
    df,
    path_or_buf=None,
    # Pandas default "columns"
    # Change it to match Bodo default for `read_json`
    orient="records",
    date_format=None,
    double_precision=10,
    force_ascii=True,
    date_unit="ms",
    default_handler=None,
    # Pandas default: "False"
    # Change it to match Bodo default for `read_json`
    lines=True,
    compression="infer",
    index=None,
    indent=None,
    storage_options=None,
    mode="w",
    _bodo_file_prefix="part-",
    # Concatenate string output on rank 0 if set (used in spawn mode)
    _bodo_concat_str_output=False,
):
    check_runtime_cols_unsupported(df, "DataFrame.to_json()")
    check_unsupported_args(
        "DataFrame.to_json",
        {
            "storage_options": storage_options,
        },
        {
            "storage_options": None,
        },
        package_name="pandas",
        module_name="IO",
    )

    # TODO: refactor when objmode() can understand global string constant
    # String output case
    if is_overload_none(path_or_buf):

        def _impl(
            df,
            path_or_buf=None,
            orient="records",
            date_format=None,
            double_precision=10,
            force_ascii=True,
            date_unit="ms",
            default_handler=None,
            lines=True,
            compression="infer",
            index=None,
            indent=None,
            storage_options=None,
            mode="w",
            _bodo_file_prefix="part-",
            _bodo_concat_str_output=False,
        ):  # pragma: no cover
            if _bodo_concat_str_output:
                # Return the concatenated string output on rank 0
                # and empty string on all other ranks
                df = bodo.gatherv(df)
                if bodo.get_rank() != 0:
                    return ""
            with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
                D = df.to_json(
                    path_or_buf,
                    orient=orient,
                    date_format=date_format,
                    double_precision=double_precision,
                    force_ascii=force_ascii,
                    date_unit=date_unit,
                    default_handler=default_handler,
                    lines=lines,
                    compression=compression,
                    index=index,
                    indent=indent,
                    storage_options=storage_options,
                    mode=mode,
                )
            return D

        return _impl

    def _impl(
        df,
        path_or_buf=None,
        orient="records",
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=True,
        compression="infer",
        index=None,
        indent=None,
        storage_options=None,
        mode="w",
        _bodo_file_prefix="part-",
        _bodo_concat_str_output=False,
    ):  # pragma: no cover
        # passing None for the first argument returns a string
        # containing contents to write to json
        with bodo.ir.object_mode.no_warning_objmode(D="unicode_type"):
            D = df.to_json(
                None,
                orient=orient,
                date_format=date_format,
                double_precision=double_precision,
                force_ascii=force_ascii,
                date_unit=date_unit,
                default_handler=default_handler,
                lines=lines,
                compression=compression,
                index=index,
                indent=indent,
                storage_options=storage_options,
                mode=mode,
            )

        # Assuming that path_or_buf is a string
        bucket_region = bodo.io.fs_io.get_s3_bucket_region_wrapper(
            path_or_buf, parallel=False
        )

        bodo.hiframes.pd_dataframe_ext._json_write(
            unicode_to_utf8(path_or_buf),
            unicode_to_utf8(D),
            0,
            len(D),
            False,
            lines and orient == "records",
            unicode_to_utf8(bucket_region),
            unicode_to_utf8(_bodo_file_prefix),
        )
        # Check if there was an error in the C++ code. If so, raise it.
        bodo.utils.utils.check_and_propagate_cpp_exception()

    return _impl


@overload(pd.get_dummies, inline="always", no_unliteral=True)
@overload(bd.get_dummies, inline="always", no_unliteral=True)
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
    check_unsupported_args(
        "pandas.get_dummies",
        args_dict,
        args_default_dict,
        package_name="pandas",
        module_name="General",
    )
    if not categorical_can_construct_dataframe(data):
        raise BodoError(
            "pandas.get_dummies() only support categorical data types with explicitly known categories"
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
        func_text += f"  data_arr_{i} = np.empty(n, np.uint8)\n"
    func_text += "  for i in numba.parfors.parfor.internal_prange(n):\n"
    func_text += "      if bodo.libs.array_kernels.isna(data_values, i):\n"
    for j in range(n_cols):
        func_text += f"          data_arr_{j}[i] = 0\n"
    func_text += "      else:\n"
    for k in range(n_cols):
        func_text += f"          data_arr_{k}[i] = codes[i] == {k}\n"
    data_args = ", ".join(f"data_arr_{i}" for i in range(n_cols))
    index = "bodo.hiframes.pd_index_ext.init_range_index(0, n, 1, None)"

    # convert datetime64 categories to Timestamp and timedelta64 to Timedelta
    # to avoid codegen errors
    # TODO(Ehsan): pass column names as dataframe type to avoid these issues
    if isinstance(categories[0], np.datetime64):
        categories = tuple(pd.Timestamp(c) for c in categories)
    elif isinstance(categories[0], np.timedelta64):
        categories = tuple(pd.Timedelta(c) for c in categories)

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
        raise BodoError(f"DataFrame.{func_name}(): transform necessary for inplace")


def union_dataframes(df_tup, drop_duplicates, output_colnames):  # pragma: no cover
    pass


@overload(union_dataframes, inline="always")
def overload_union_dataframes(
    df_tup, drop_duplicates, output_colnames
):  # pragma: no cover
    # Step 1: Verify that all DataFrames have the same number of columns.
    # We don't care about the index or the names.
    df_types: tuple[DataFrameType, ...] = df_tup.types
    if len(df_types) == 0:
        raise BodoError(
            "union_distinct_dataframes must be called with at least one DataFrame"
        )

    for in_table_type in df_types:
        error_on_unsupported_streaming_arrays(in_table_type.table_type)

    num_cols = -1
    col_types = None
    for df_type in df_types:
        if not isinstance(df_type, DataFrameType):
            raise BodoError("union_distinct_dataframes must be called with DataFrames")
        if num_cols == -1:
            num_cols = len(df_type.data)
            col_types = df_type.data
        else:
            if num_cols != len(df_type.data):
                raise BodoError(
                    "union_distinct_dataframes must be called with DataFrames with the same number of columns"
                )
            new_col_types = []
            for i, col_typ in enumerate(col_types):
                other_col_typ = df_type.data[i]
                if col_typ == other_col_typ:
                    new_col_types.append(col_typ)
                elif (
                    col_typ == bodo.types.dict_str_arr_type
                    or other_col_typ == bodo.types.dict_str_arr_type
                ):
                    if col_typ not in (
                        bodo.types.string_array_type,
                        bodo.types.null_array_type,
                    ) and other_col_typ not in (
                        bodo.types.string_array_type,
                        bodo.types.null_array_type,
                    ):
                        # If one column is dict encoded the other column must be a string
                        # or null array.
                        raise BodoError(
                            f"Unable to union table with columns of incompatible types. Found types {col_typ} and {other_col_typ} in column {i}."
                        )
                    # If either array is dict encoded we want the output to be dict encoded.
                    new_col_types.append(bodo.types.dict_str_arr_type)
                else:
                    col_dtype = col_typ.dtype
                    other_col_dtype = other_col_typ.dtype
                    new_dtype, _ = get_common_scalar_dtype([col_dtype, other_col_dtype])
                    if new_dtype is None:
                        raise BodoError(
                            f"Unable to union table with columns of incompatible types. Found types {col_dtype} and {other_col_dtype} in column {i}."
                        )
                    new_col_type = dtype_to_array_type(
                        new_dtype, is_nullable(col_typ) or is_nullable(other_col_typ)
                    )
                    new_col_types.append(new_col_type)
            col_types = tuple(new_col_types)

    func_text = "def impl(df_tup, drop_duplicates, output_colnames):\n"
    glbls = {
        "bodo": bodo,
        "py_table_typ": bodo.types.TableType(col_types),
    }
    # Step 2 generate code to convert each DataFrame to C++.
    for i, df_type in enumerate(df_types):
        # Load the DataFrame
        func_text += f"  df{i} = df_tup[{i}]\n"
        # TODO: If necessary create an astype to unify the types.
        # Convert the DataFrame to a C++ table.
        if df_type.is_table_format:
            func_text += f"  table{i} = bodo.hiframes.pd_dataframe_ext.get_dataframe_table(df{i})\n"
            if df_type.data != col_types:
                # If the types differ we need to cast.
                func_text += f"  arg{i} = bodo.utils.table_utils.table_astype(table{i}, py_table_typ, False, _bodo_nan_to_str=False)\n"
            else:
                func_text += f"  arg{i} = table{i}\n"
        else:
            for j in range(num_cols):
                func_text += f"  arr{i}_{j} = bodo.hiframes.pd_dataframe_ext.get_dataframe_data(df{i}, {j})\n"
                if df_type.data[j] != col_types[j]:
                    glbls[f"arr_typ_{i}_{j}"] = get_castable_arr_dtype(col_types[j])
                    # Cast the array if there is a type mismatch.
                    func_text += f"  arg{i}_{j} = bodo.utils.conversion.fix_arr_dtype(arr{i}_{j}, arr_typ_{i}_{j}, False, nan_to_str=False, from_series=True)\n"
                else:
                    func_text += f"  arg{i}_{j} = arr{i}_{j}\n"
            arrs = [f"arg{i}_{j}" for j in range(num_cols)]
            tuple_inputs = ", ".join(arrs)
            func_text += f"  arg{i} = ({tuple_inputs},)\n"
    df_args = ", ".join([f"arg{i}" for i in range(len(df_types))])
    # Step 3 call the C++ kernel
    func_text += f"  out_py_table = bodo.libs.array.union_tables(({df_args}, ), drop_duplicates, py_table_typ)\n"
    func_text += "  out_df = bodo.hiframes.pd_dataframe_ext.init_dataframe((out_py_table,), bodo.hiframes.pd_index_ext.init_range_index(0, len(out_py_table), 1, None), output_colnames)\n"
    func_text += "  return out_df\n"
    loc_vars = {}
    exec(func_text, glbls, loc_vars)
    return loc_vars["impl"]


# Throw BodoError for top-level unsupported functions in Pandas
pd_unsupported = [
    # Input/output
    pd.read_pickle,
    pd.read_table,
    pd.read_fwf,
    pd.read_clipboard,
    pd.ExcelFile,
    pd.read_html,
    pd.read_xml,
    pd.read_hdf,
    pd.read_feather,
    pd.read_orc,  # TODO: support
    pd.read_sas,
    pd.read_spss,
    pd.read_sql_query,
    pd.read_stata,
    pd.ExcelWriter,
    pd.json_normalize,
    # General functions
    ## Data manipulations
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
    # Testing
    pd.test,
    # GroupBy
    pd.Grouper,
]


try:
    pd_unsupported.append(pd.read_gbq)
except AttributeError:
    # pd.read_gbq is not supported in Pandas > 2.2
    pass


pd_util_unsupported = (
    ## Hashing
    pd.util.hash_array,
    pd.util.hash_pandas_object,
)

dataframe_unsupported = [
    # Attributes and underlying data
    "set_flags",
    # Conversion
    "convert_dtypes",
    "bool",
    # Indexing, iteration
    "__iter__",
    "items",
    "iteritems",
    "keys",
    "iterrows",
    "lookup",
    "pop",
    "xs",
    "get",
    # Binary operator functions
    "add",
    "__add__",
    "sub",
    "mul",
    "div",
    "truediv",
    "floordiv",
    "mod",
    "pow",
    "dot",
    "radd",
    "rsub",
    "rmul",
    "rdiv",
    "rtruediv",
    "rfloordiv",
    "rmod",
    "rpow",
    "lt",
    "gt",
    "le",
    "ge",
    "ne",
    "eq",
    "combine",
    "combine_first",
    "subtract",  # Not in the organized pd docs, putting it here
    "divide",  # Not in the organized pd docs, putting it here
    "multiply",  # Not in the organized pd docs, putting it here
    # Function application, GroupBy & window
    "agg",
    "aggregate",
    "map",
    "applymap",
    "transform",
    "expanding",
    "ewm",
    # Computations / descriptive stats
    "all",
    "any",
    "clip",
    "corrwith",
    "cummax",
    "cummin",
    "eval",
    "kurt",
    "kurtosis",
    "mode",
    "round",
    "sem",
    "skew",
    "value_counts",
    # Reindexing / selection / label manipulation
    "add_prefix",
    "add_suffix",
    "align",
    "at_time",
    "between_time",
    "equals",
    "reindex",
    "reindex_like",
    "rename_axis",
    "set_axis",
    "truncate",
    # Missing data handling
    "backfill",
    "bfill",
    "ffill",
    "interpolate",
    "pad",
    # Reshaping, sorting, transposing
    "droplevel",
    "reorder_levels",
    "nlargest",
    "nsmallest",
    "swaplevel",
    "stack",
    "unstack",
    "swapaxes",
    "squeeze",
    "to_xarray",
    "T",
    "transpose",
    # Combining / comparing / joining / merging
    "compare",
    "update",
    # Time Series-related
    "asfreq",
    "asof",
    "slice_shift",
    "tshift",
    "first_valid_index",
    "last_valid_index",
    "resample",
    "to_period",
    "to_timestamp",
    "tz_convert",
    "tz_localize",
    # Plotting
    # TODO [BSE-3957]: handle df.plot.x
    "boxplot",
    "hist",
    # Serialization / IO / conversion:
    "from_dict",
    "from_records",
    "to_pickle",
    "to_hdf",
    "to_dict",
    "to_excel",
    "to_html",
    "to_feather",
    "to_latex",
    "to_stata",
    "to_gbq",
    "to_records",
    "to_clipboard",
    "to_markdown",
    "to_xml",  # Not in the organized pd docs, putting it here
    "to_orc",
    "__dataframe__",
]

dataframe_unsupported_attrs = [
    "at",
    "attrs",
    "axes",
    "flags",
    # property
    "style",
    # TODO: handle Df.sparse.x
    "sparse",
]

# TODO [BSE-3957]: add proper error messaging for df.plot.x


def _install_pd_unsupported(mod_name, pd_unsupported):
    """install an overload that raises BodoError for unsupported functions"""
    for f in pd_unsupported:
        fname = mod_name + "." + f.__name__
        overload(f, no_unliteral=True)(create_unsupported_overload(fname))


def _install_dataframe_unsupported():
    """install an overload that raises BodoError for unsupported Dataframe methods"""

    for attr_name in dataframe_unsupported_attrs:
        full_name = "DataFrame." + attr_name
        overload_unsupported_attribute(DataFrameType, attr_name, full_name)
    for fname in dataframe_unsupported:
        full_name = "DataFrame." + fname
        overload_unsupported_method(DataFrameType, fname, full_name)


# Run install unsupported for each module to ensure a correct error message.
_install_pd_unsupported("pandas", pd_unsupported)
_install_pd_unsupported("pandas.util", pd_util_unsupported)
_install_dataframe_unsupported()
