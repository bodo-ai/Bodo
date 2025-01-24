"""
Provides a registry of function call handlers for distributed analysis.
"""

import typing as pt

from numba.core import ir, types
from numba.core.ir_utils import guard
from numba.parfors.array_analysis import ShapeEquivSet

from bodo.transforms.distributed_analysis import (
    Distribution,
    _meet_array_dists,
    _set_REP,
    _set_var_dist,
)
from bodo.utils.typing import BodoError, get_overload_const_str, is_overload_none
from bodo.utils.utils import is_distributable_typ


class DistributedAnalysisContext:
    """Distributed analysis context data needed for handling calls"""

    def __init__(
        self,
        typemap: dict[str, types.Type],
        array_dists: dict[str, Distribution],
        equiv_set: ShapeEquivSet,
        func_name: str,
        metadata: dict[str, pt.Any],
        diag_info: list[tuple[str, ir.Loc]],
    ):
        self.typemap = typemap
        self.array_dists = array_dists
        self.equiv_set = equiv_set
        self.func_name = func_name
        self.metadata = metadata
        self.diag_info = diag_info


class DistributedAnalysisCallRegistry:
    """Registry of function call handlers for distributed analysis"""

    def __init__(self):
        # Dictionary format: (function_name, module_name): handler_func
        # module_name is a class name for methods instead.
        # handler_func takes distributed analysis context data and the instruction to
        # handle as arguments.
        self.call_map = {
            # scalar_optional_getitem is used by BodoSQL to load scalars.
            # This doesn't impact the distribution of any array.
            ("scalar_optional_getitem", "bodo.utils.indexing"): no_op_analysis,
            # add_nested_counts is used by ArrayItemArray to add nested counts.
            # This doesn't impact the distribution of any array.
            ("add_nested_counts", "bodo.utils.indexing"): no_op_analysis,
            # scalar_to_array_item_array is used to convert a scalar to ArrayItemArray.
            # This doesn't impact the distribution of any array.
            (
                "scalar_to_array_item_array",
                "bodo.libs.array_item_arr_ext",
            ): no_op_analysis,
            (
                "table_astype",
                "bodo.utils.table_utils",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoRandomForestClassifierType"): meet_first_2_args_analysis,
            ("predict", "BodoRandomForestClassifierType"): meet_out_first_arg_analysis,
            ("score", "BodoRandomForestClassifierType"): meet_first_2_args_analysis,
            (
                "predict_proba",
                "BodoRandomForestClassifierType",
            ): meet_out_first_arg_analysis,
            (
                "predict_log_proba",
                "BodoRandomForestClassifierType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoRandomForestRegressorType"): meet_first_2_args_analysis,
            ("predict", "BodoRandomForestRegressorType"): meet_out_first_arg_analysis,
            ("score", "BodoRandomForestRegressorType"): meet_first_2_args_analysis,
            ("fit", "BodoSGDClassifierType"): meet_first_2_args_analysis,
            ("predict", "BodoSGDClassifierType"): meet_out_first_arg_analysis,
            ("score", "BodoSGDClassifierType"): meet_first_2_args_analysis,
            ("predict_proba", "BodoSGDClassifierType"): meet_out_first_arg_analysis,
            ("predict_log_proba", "BodoSGDClassifierType"): meet_out_first_arg_analysis,
            ("fit", "BodoSGDRegressorType"): meet_first_2_args_analysis,
            ("predict", "BodoSGDRegressorType"): meet_out_first_arg_analysis,
            ("score", "BodoSGDRegressorType"): meet_first_2_args_analysis,
            ("fit", "BodoLogisticRegressionType"): meet_first_2_args_analysis,
            ("predict", "BodoLogisticRegressionType"): meet_out_first_arg_analysis,
            ("score", "BodoLogisticRegressionType"): meet_first_2_args_analysis,
            (
                "predict_proba",
                "BodoLogisticRegressionType",
            ): meet_out_first_arg_analysis,
            (
                "predict_log_proba",
                "BodoLogisticRegressionType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoMultinomialNBType"): meet_first_2_args_analysis,
            ("predict", "BodoMultinomialNBType"): meet_out_first_arg_analysis,
            ("score", "BodoMultinomialNBType"): meet_first_2_args_analysis,
            ("fit", "BodoLassoType"): meet_first_2_args_analysis,
            ("predict", "BodoLassoType"): meet_out_first_arg_analysis,
            ("score", "BodoLassoType"): meet_first_2_args_analysis,
            ("fit", "BodoLinearRegressionType"): meet_first_2_args_analysis,
            ("predict", "BodoLinearRegressionType"): meet_out_first_arg_analysis,
            ("score", "BodoLinearRegressionType"): meet_first_2_args_analysis,
            ("fit", "BodoRidgeType"): meet_first_2_args_analysis,
            ("predict", "BodoRidgeType"): meet_out_first_arg_analysis,
            ("score", "BodoRidgeType"): meet_first_2_args_analysis,
            ("fit", "BodoLinearSVCType"): meet_first_2_args_analysis,
            ("predict", "BodoLinearSVCType"): meet_out_first_arg_analysis,
            ("score", "BodoLinearSVCType"): meet_first_2_args_analysis,
            ("fit", "BodoXGBClassifierType"): meet_first_2_args_analysis_xgb_fit,
            ("predict", "BodoXGBClassifierType"): meet_out_first_arg_analysis,
            ("predict_proba", "BodoXGBClassifierType"): meet_out_first_arg_analysis,
            ("fit", "BodoXGBRegressorType"): meet_first_2_args_analysis_xgb_fit,
            ("predict", "BodoXGBRegressorType"): meet_out_first_arg_analysis,
            (
                "generate_mappable_table_func",
                "bodo.utils.table_utils",
            ): analyze_mappable_table_funcs,
            ("table_subset", "bodo.hiframes.table"): meet_out_first_arg_analysis,
            ("create_empty_table", "bodo.hiframes.table"): analyze_create_table_empty,
            ("table_concat", "bodo.utils.table_utils"): analyze_table_concat,
            (
                "array_to_repeated_array_item_array",
                "bodo.libs.array_item_arr_ext",
            ): analyze_array_to_repeated_array_item_array,
            # Scalar to array functions are similar to allocations and can return
            # distributed data
            (
                "scalar_to_struct_array",
                "bodo.libs.struct_arr_ext",
            ): init_out_1D,
            ("scalar_to_map_array", "bodo.libs.map_arr_ext"): init_out_1D,
            (
                "transform",
                "BodoKMeansClusteringType",
            ): analyze_call_sklearn_cluster_kmeans,
            ("score", "BodoKMeansClusteringType"): analyze_call_sklearn_cluster_kmeans,
            (
                "predict",
                "BodoKMeansClusteringType",
            ): analyze_call_sklearn_cluster_kmeans,
            ("fit", "BodoKMeansClusteringType"): analyze_call_sklearn_cluster_kmeans,
            # Analyze distribution of sklearn.preprocessing.OneHotEncoder, StandardScaler,
            # MaxAbsScaler, MinMaxScaler, RobustScaler, and LabelEncoder functions.
            # Only need to handle fit_transform, transform and inverse_transform. fit is handled automatically.
            ("fit", "BodoPreprocessingOneHotEncoderType"): no_op_analysis,
            ("partial_fit", "BodoPreprocessingOneHotEncoderType"): no_op_analysis,
            (
                "transform",
                "BodoPreprocessingOneHotEncoderType",
            ): meet_out_first_arg_analysis,
            (
                "inverse_transform",
                "BodoPreprocessingOneHotEncoderType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoPreprocessingOneHotEncoderType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoPreprocessingStandardScalerType"): no_op_analysis,
            ("partial_fit", "BodoPreprocessingStandardScalerType"): no_op_analysis,
            (
                "transform",
                "BodoPreprocessingStandardScalerType",
            ): meet_out_first_arg_analysis,
            (
                "inverse_transform",
                "BodoPreprocessingStandardScalerType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoPreprocessingStandardScalerType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoPreprocessingMaxAbsScalerType"): no_op_analysis,
            ("partial_fit", "BodoPreprocessingMaxAbsScalerType"): no_op_analysis,
            (
                "transform",
                "BodoPreprocessingMaxAbsScalerType",
            ): meet_out_first_arg_analysis,
            (
                "inverse_transform",
                "BodoPreprocessingMaxAbsScalerType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoPreprocessingMaxAbsScalerType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoPreprocessingMinMaxScalerType"): no_op_analysis,
            ("partial_fit", "BodoPreprocessingMinMaxScalerType"): no_op_analysis,
            (
                "transform",
                "BodoPreprocessingMinMaxScalerType",
            ): meet_out_first_arg_analysis,
            (
                "inverse_transform",
                "BodoPreprocessingMinMaxScalerType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoPreprocessingMinMaxScalerType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoPreprocessingRobustScalerType"): no_op_analysis,
            ("partial_fit", "BodoPreprocessingRobustScalerType"): no_op_analysis,
            (
                "transform",
                "BodoPreprocessingRobustScalerType",
            ): meet_out_first_arg_analysis,
            (
                "inverse_transform",
                "BodoPreprocessingRobustScalerType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoPreprocessingRobustScalerType",
            ): meet_out_first_arg_analysis,
            ("fit", "BodoPreprocessingLabelEncoderType"): no_op_analysis,
            ("partial_fit", "BodoPreprocessingLabelEncoderType"): no_op_analysis,
            (
                "transform",
                "BodoPreprocessingLabelEncoderType",
            ): meet_out_first_arg_analysis,
            (
                "inverse_transform",
                "BodoPreprocessingLabelEncoderType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoPreprocessingLabelEncoderType",
            ): meet_out_first_arg_analysis,
            # match input and output distributions (y is ignored)
            (
                "fit_transform",
                "BodoFExtractHashingVectorizerType",
            ): meet_out_first_arg_analysis,
            (
                "fit_transform",
                "BodoFExtractCountVectorizerType",
            ): meet_out_first_arg_analysis,
            (
                "shuffle",
                "sklearn.utils",
            ): meet_out_first_arg_analysis,
            (
                "precision_score",
                "sklearn.metrics._classification",
            ): analyze_call_sklearn_metrics,
            (
                "precision_score",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "recall_score",
                "sklearn.metrics._classification",
            ): analyze_call_sklearn_metrics,
            (
                "recall_score",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "f1_score",
                "sklearn.metrics._classification",
            ): analyze_call_sklearn_metrics,
            (
                "f1_score",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "log_loss",
                "sklearn.metrics._classification",
            ): analyze_call_sklearn_metrics,
            (
                "log_loss",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "accuracy_score",
                "sklearn.metrics._classification",
            ): analyze_call_sklearn_metrics,
            (
                "accuracy_score",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "confusion_matrix",
                "sklearn.metrics._classification",
            ): analyze_call_sklearn_metrics,
            (
                "confusion_matrix",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "mean_squared_error",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "mean_squared_error",
                "sklearn.metrics._regression",
            ): analyze_call_sklearn_metrics,
            (
                "mean_absolute_error",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "mean_absolute_error",
                "sklearn.metrics._regression",
            ): analyze_call_sklearn_metrics,
            (
                "r2_score",
                "sklearn.metrics",
            ): analyze_call_sklearn_metrics,
            (
                "r2_score",
                "sklearn.metrics._regression",
            ): analyze_call_sklearn_metrics,
            # Match distribution of X to the output.
            # The output distribution is intended to match X and should ignore Y.
            (
                "cosine_similarity",
                "sklearn.metrics.pairwise",
            ): meet_out_first_arg_analysis,
            (
                "datetime_date_arr_to_dt64_arr",
                "bodo.hiframes.pd_timestamp_ext",
            ): meet_out_first_arg_analysis,
            (
                "unwrap_tz_array",
                "bodo.libs.pd_datetime_arr_ext",
            ): meet_out_first_arg_analysis,
            ("accum_func", "bodo.libs.array_kernels"): meet_out_first_arg_analysis,
            ("parallel_print", "bodo"): no_op_analysis,
            (
                "series_contains_regex",
                "bodo.hiframes.series_str_impl",
            ): meet_out_first_arg_analysis,
            (
                "series_match_regex",
                "bodo.hiframes.series_str_impl",
            ): meet_out_first_arg_analysis,
            (
                "series_fullmatch_regex",
                "bodo.hiframes.series_str_impl",
            ): meet_out_first_arg_analysis,
            ("setna", "bodo.libs.array_kernels"): no_op_analysis,
            (
                "snowflake_writer_append_table",
                "bodo.io.snowflake_write",
            ): meet_first_2_args_analysis,
            (
                "iceberg_writer_append_table",
                "bodo.io.iceberg.stream_iceberg_write",
            ): meet_first_2_args_analysis,
            (
                "parquet_writer_append_table",
                "bodo.io.stream_parquet_write",
            ): meet_first_2_args_analysis,
            (
                "groupby_build_consume_batch",
                "bodo.libs.streaming.groupby",
            ): meet_first_2_args_analysis,
            (
                "groupby_grouping_sets_build_consume_batch",
                "bodo.libs.streaming.groupby",
            ): meet_first_2_args_analysis,
            (
                "window_build_consume_batch",
                "bodo.libs.streaming.window",
            ): meet_first_2_args_analysis,
            (
                "sort_build_consume_batch",
                "bodo.libs.streaming.sort",
            ): meet_first_2_args_analysis,
            (
                "table_builder_finalize",
                "bodo.libs.table_builder",
            ): meet_out_first_arg_analysis,
            (
                "rebalance",
                "bodo",
            ): meet_out_first_arg_analysis,
            (
                "random_shuffle",
                "bodo",
            ): meet_out_first_arg_analysis,
            ("len", "__builtin__"): no_op_analysis,
            ("len", "builtins"): no_op_analysis,
            ("local_len", "bodo.hiframes.table"): no_op_analysis,
            ("quantile", "bodo.libs.array_kernels"): no_op_analysis,
            ("approx_percentile", "bodo.libs.array_kernels"): no_op_analysis,
            ("percentile_cont", "bodo.libs.array_kernels"): no_op_analysis,
            ("percentile_disc", "bodo.libs.array_kernels"): no_op_analysis,
            ("nunique", "bodo.libs.array_kernels"): no_op_analysis,
            ("anyvalue_agg", "bodo.libs.array_kernels"): no_op_analysis,
            ("boolor_agg", "bodo.libs.array_kernels"): no_op_analysis,
            ("booland_agg", "bodo.libs.array_kernels"): no_op_analysis,
            ("boolxor_agg", "bodo.libs.array_kernels"): no_op_analysis,
            ("bitor_agg", "bodo.libs.array_kernels"): no_op_analysis,
            ("bitand_agg", "bodo.libs.array_kernels"): no_op_analysis,
            ("bitxor_agg", "bodo.libs.array_kernels"): no_op_analysis,
            (
                "series_str_dt64_astype",
                "bodo.hiframes.pd_timestamp_ext",
            ): meet_out_first_arg_analysis,
            (
                "series_str_td64_astype",
                "bodo.hiframes.pd_timestamp_ext",
            ): meet_out_first_arg_analysis,
            (
                "cat_replace",
                "bodo.hiframes.pd_categorical_ext",
            ): meet_out_first_arg_analysis,
            (
                "intersection_mask",
                "bodo.libs.array_kernels",
            ): meet_out_first_arg_analysis,
            # Output is a decimal, so we don't need to explicitly set the distribution
            (
                "sum_decimal_array",
                "bodo.libs.decimal_arr_ext",
            ): no_op_analysis,
            ("first_last_valid_index", "bodo.libs.array_kernels"): no_op_analysis,
            (
                "get_valid_entries_from_date_offset",
                "bodo.libs.array_kernels",
            ): no_op_analysis,
            ("shift", "bodo.hiframes.rolling"): meet_out_first_arg_analysis,
            ("pct_change", "bodo.hiframes.rolling"): meet_out_first_arg_analysis,
            ("set_table_data_null", "bodo.hiframes.table"): meet_out_first_arg_analysis,
            (
                "str_arr_encode",
                "bodo.libs.str_arr_ext",
            ): meet_out_first_arg_analysis,
            (
                "pandas_string_array_to_datetime",
                "bodo.hiframes.pd_timestamp_ext",
            ): meet_out_first_arg_analysis,
            (
                "pandas_dict_string_array_to_datetime",
                "bodo.hiframes.pd_timestamp_ext",
            ): meet_out_first_arg_analysis,
            (
                "_table_to_tuple_format_decoded",
                "bodo.hiframes.pd_dataframe_ext",
            ): meet_out_first_arg_analysis,
            (
                "_tuple_to_table_format_decoded",
                "bodo.hiframes.pd_dataframe_ext",
            ): meet_out_first_arg_analysis,
            (
                "pandas_string_array_to_timedelta",
                "bodo.hiframes.pd_timestamp_ext",
            ): meet_out_first_arg_analysis,
            ("get", "bodo.libs.array_kernels"): meet_out_first_arg_analysis,
            ("series_monotonicity", "bodo.libs.array_kernels"): no_op_analysis,
            ("autocorr", "bodo.libs.array_kernels"): no_op_analysis,
            ("array_op_median", "bodo.libs.array_ops"): no_op_analysis,
            ("str_arr_min_max", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("isna", "bodo.libs.array_kernels"): no_op_analysis,
            ("get_str_arr_str_length", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("get_series_name", "bodo.hiframes.pd_series_ext"): no_op_analysis,
            ("get_index_name", "bodo.hiframes.pd_index_ext"): no_op_analysis,
            ("order_range", "bodo.hiframes.pd_index_ext"): meet_out_first_arg_analysis,
            (
                "range_index_to_numeric",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            ("get_int_arr_data", "bodo.libs.int_arr_ext"): meet_out_first_arg_analysis,
            (
                "get_int_arr_bitmap",
                "bodo.libs.int_arr_ext",
            ): meet_out_first_arg_analysis,
            (
                "get_float_arr_data",
                "bodo.libs.float_arr_ext",
            ): meet_out_first_arg_analysis,
            (
                "get_float_arr_bitmap",
                "bodo.libs.float_arr_ext",
            ): meet_out_first_arg_analysis,
            (
                "get_categorical_arr_codes",
                "bodo.hiframes.pd_categorical_ext",
            ): meet_out_first_arg_analysis,
            ("ffill_bfill_arr", "bodo.libs.array_kernels"): meet_out_first_arg_analysis,
            ("get_bit_bitmap_arr", "bodo.libs.int_arr_ext"): no_op_analysis,
            ("set_bit_to_arr", "bodo.libs.int_arr_ext"): no_op_analysis,
            # iceberg_merge_cow_py doesn't have a return value
            # or alter the distribution of any input.
            ("iceberg_merge_cow_py", "bodo.io.iceberg.merge_into"): no_op_analysis,
            ("decode_if_dict_array", "bodo.utils.typing"): meet_out_first_arg_analysis,
            (
                "np_to_nullable_array",
                "bodo.utils.conversion",
            ): meet_out_first_arg_analysis,
            ("get_arr_lens", "bodo.libs.array_kernels"): meet_out_first_arg_analysis,
            ("str_split", "bodo.libs.str_ext"): meet_out_first_arg_analysis,
            ("str_split_empty_n", "bodo.libs.str_ext"): meet_out_first_arg_analysis,
            (
                "init_numeric_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            (
                "init_binary_str_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            (
                "init_categorical_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            (
                "init_datetime_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            (
                "init_timedelta_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            (
                "init_period_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            (
                "init_interval_index",
                "bodo.hiframes.pd_index_ext",
            ): meet_out_first_arg_analysis,
            ("get_table_data", "bodo.hiframes.table"): meet_out_first_arg_analysis,
            (
                "compute_split_view",
                "bodo.hiframes.split_impl",
            ): meet_out_first_arg_analysis,
            # just used in str.get() implementation for now so we know it is
            # parallel
            # TODO: handle index similar to getitem to support more cases
            ("get_split_view_index", "bodo.hiframes.split_impl"): no_op_analysis,
            ("get_split_view_data_ptr", "bodo.hiframes.split_impl"): no_op_analysis,
            ("setitem_str_arr_ptr", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("num_total_chars", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("inplace_eq", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("get_str_arr_item_copy", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("copy_array_element", "bodo.libs.array_kernels"): no_op_analysis,
            ("str_arr_setitem_int_to_str", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("str_arr_setitem_NA_str", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("str_arr_set_not_na", "bodo.libs.str_arr_ext"): no_op_analysis,
            ("set_null_bits_to_value", "bodo.libs.str_arr_ext"): no_op_analysis,
            (
                "str_arr_to_dict_str_arr",
                "bodo.libs.str_arr_ext",
            ): meet_out_first_arg_analysis,
            ("array_op_describe", "bodo.libs.array_ops"): no_op_analysis,
            ("array_op_nbytes", "bodo.libs.array_ops"): no_op_analysis,
            # np.fromfile()
            ("file_read", "bodo.io.np_io"): no_op_analysis,
            ("array_to_string", "bodosql.kernels"): meet_out_first_arg_analysis,
            # TODO: make sure assert_equiv is not generated unnecessarily
            # TODO: fix assert_equiv for np.stack from df.value
            ("assert_equiv", "numba.parfors.array_analysis"): no_op_analysis,
            ("dict_arr_eq", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("dict_arr_ne", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            (
                "convert_dict_arr_to_int",
                "bodo.libs.dict_arr_ext",
            ): meet_out_first_arg_analysis,
            (
                "dict_arr_to_numeric",
                "bodo.libs.dict_arr_ext",
            ): meet_out_first_arg_analysis,
            ("str_replace", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_startswith", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_endswith", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            (
                "str_contains_non_regex",
                "bodo.libs.dict_arr_ext",
            ): meet_out_first_arg_analysis,
            (
                "str_series_contains_regex",
                "bodo.libs.dict_arr_ext",
            ): meet_out_first_arg_analysis,
            ("str_capitalize", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_lower", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_swapcase", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_title", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_upper", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_center", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_extract", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_get", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_repeat_int", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_lstrip", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_rstrip", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_strip", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_ljust", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_rjust", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_zfill", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_find", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_rfind", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_index", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_rindex", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_slice", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_len", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_count", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isalnum", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isalpha", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isdigit", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isspace", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_islower", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isupper", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_istitle", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isnumeric", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_isdecimal", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
            ("str_match", "bodo.libs.dict_arr_ext"): meet_out_first_arg_analysis,
        }

    def analyze_call(self, ctx, inst, fdef):
        """Perform distributed analysis for input call if it's in the registry.
        Return True if handled.

        Args:
            ctx (call_registry): distributed analysis context data
            inst (ir.Assign): call expression
            fdef (tuple[ir.Var|str]): function path in tuple of Var/string format

        Returns:
            bool: True if handled, False otherwise
        """
        handler_func = self.call_map.get(fdef, None)
        if handler_func:
            handler_func(ctx, inst)
            return True

        return False


def no_op_analysis(ctx, inst):
    """Handler that doesn't change any distributions"""
    pass


def meet_out_first_arg_analysis(ctx, inst):
    """Handler that meets distributions of first call argument and output variables"""
    _meet_array_dists(
        ctx.typemap, inst.target.name, inst.value.args[0].name, ctx.array_dists
    )


def meet_first_2_args_analysis(ctx, inst):
    """Handler that meets distributions of the first two call arguments"""
    _meet_array_dists(
        ctx.typemap, inst.value.args[0].name, inst.value.args[1].name, ctx.array_dists
    )


def meet_first_2_args_analysis_xgb_fit(ctx, inst):  # pragma: no cover
    """Handler that meets distributions of the first two call arguments for xgb fit"""
    _meet_array_dists(
        ctx.typemap, inst.value.args[0].name, inst.value.args[1].name, ctx.array_dists
    )
    if ctx.array_dists[inst.value.args[0].name] == Distribution.REP:
        raise BodoError("Arguments of xgboost.fit are not distributed", inst.loc)


def analyze_mappable_table_funcs(ctx, inst):
    """
    Analyze for functions using generate_mappable_table_func.
    Arg0 is a table used for distribution and arg1 is the function
    name. Distributions differ based on arg1.

    Returns True if there was a known implementation.
    """
    lhs = inst.target.name
    rhs = inst.value
    func_name_typ = ctx.typemap[rhs.args[1].name]
    has_func = not is_overload_none(func_name_typ)
    if has_func:
        # XXX: Make this more scalable by recalling the distributed
        # analysis already in this pass for each of the provided
        # func names.
        func_name = guard(get_overload_const_str, func_name_typ)
        # We support mappable prefixes that don't need to be separate functions.
        if func_name[0] == "~":
            func_name = func_name[1:]
        # Note: This isn't an elif because the ~ may modify the name
        if func_name in (
            "bodo.libs.array_ops.array_op_isna",
            "copy",
            "bodo.libs.array_ops.drop_duplicates_local_dictionary_if_dict",
        ):
            # Not currently in the code because it is otherwise inlined.
            # This should be included somewhere.
            _meet_array_dists(ctx.typemap, lhs, rhs.args[0].name, ctx.array_dists)
            return True
    else:
        # If we don't have a func, this is a shallow copy.
        _meet_array_dists(ctx.typemap, lhs, rhs.args[0].name, ctx.array_dists)
        return True

    return False


def analyze_create_table_empty(ctx, inst):
    """distributed analysis for create_empty_table (just initializes the output to
    Distribution.OneD)
    """
    lhs = inst.target.name
    if lhs not in ctx.array_dists:
        _set_var_dist(ctx.typemap, lhs, ctx.array_dists, Distribution.OneD, True)


def analyze_table_concat(ctx, inst):
    """distributed analysis for table_concat"""
    lhs = inst.target.name
    table = inst.value.args[0].name
    out_dist = Distribution.OneD_Var
    if lhs in ctx.array_dists:
        out_dist = Distribution(min(out_dist.value, ctx.array_dists[lhs].value))
    out_dist = Distribution(min(out_dist.value, ctx.array_dists[table].value))
    ctx.array_dists[lhs] = out_dist
    if out_dist != Distribution.OneD_Var:
        ctx.array_dists[table] = out_dist


def analyze_array_to_repeated_array_item_array(ctx, inst):
    """distributed analysis for array_to_repeated_array_item_array"""
    lhs = inst.target.name
    rhs = inst.value
    if lhs not in ctx.array_dists:
        ctx.array_dists[lhs] = Distribution.OneD
    # array_to_repeated_array_item_array is used to create an ArrayItemArray with an array.
    # This requires the input array to be replicated.
    _set_REP(
        ctx.typemap,
        ctx.metadata,
        ctx.diag_info,
        rhs.args[0],
        ctx.array_dists,
        "The scalar array must be duplicated for array_to_repeated_array_item_array.",
        rhs.loc,
    )


def init_out_1D(ctx, inst):
    """initialize output distribution to 1D"""
    lhs = inst.target.name
    if lhs not in ctx.array_dists:
        ctx.array_dists[lhs] = Distribution.OneD


def analyze_call_sklearn_cluster_kmeans(ctx, inst):
    """
    Analyze distribution of sklearn cluster kmeans
    functions (sklearn.cluster.kmeans.func_name)
    """
    lhs = inst.target.name
    rhs = inst.value
    kws = dict(rhs.kws)
    array_dists = ctx.array_dists
    func_name = ctx.func_name
    typemap = ctx.typemap
    if func_name == "fit":
        # match dist of X and sample_weight (if provided)
        X_arg_name = rhs.args[0].name
        if len(rhs.args) >= 3:
            sample_weight_arg_name = rhs.args[2].name
        elif "sample_weight" in kws:
            sample_weight_arg_name = kws["sample_weight"].name
        else:
            sample_weight_arg_name = None

        if sample_weight_arg_name:
            _meet_array_dists(typemap, X_arg_name, sample_weight_arg_name, array_dists)

    elif func_name == "predict":
        # match dist of X and sample_weight (if provided)
        X_arg_name = rhs.args[0].name
        if len(rhs.args) >= 2:
            sample_weight_arg_name = rhs.args[1].name
        elif "sample_weight" in kws:
            sample_weight_arg_name = kws["sample_weight"].name
        else:
            sample_weight_arg_name = None
        if sample_weight_arg_name:
            _meet_array_dists(typemap, X_arg_name, sample_weight_arg_name, array_dists)

        # match input and output distributions
        _meet_array_dists(typemap, lhs, rhs.args[0].name, array_dists)

    elif func_name == "score":
        # match dist of X and sample_weight (if provided)
        X_arg_name = rhs.args[0].name
        if len(rhs.args) >= 3:
            sample_weight_arg_name = rhs.args[2].name
        elif "sample_weight" in kws:
            sample_weight_arg_name = kws["sample_weight"].name
        else:
            sample_weight_arg_name = None
        if sample_weight_arg_name:
            _meet_array_dists(typemap, X_arg_name, sample_weight_arg_name, array_dists)

    elif func_name == "transform":
        # match input (X) and output (X_new) distributions
        _meet_array_dists(typemap, lhs, rhs.args[0].name, array_dists)


def analyze_call_sklearn_metrics(ctx, inst):
    """
    Analyze distribution of sklearn metrics functions
    """
    lhs = inst.target.name
    rhs = inst.value
    kws = dict(rhs.kws)
    array_dists = ctx.array_dists
    func_name = ctx.func_name
    typemap = ctx.typemap
    metadata = ctx.metadata
    diag_info = ctx.diag_info

    if func_name in {"mean_squared_error", "mean_absolute_error", "r2_score"}:
        _set_REP(
            typemap,
            metadata,
            diag_info,
            lhs,
            array_dists,
            f"output of {func_name} is REP",
            rhs.loc,
        )
        _analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
            typemap, lhs, func_name, rhs, kws, array_dists
        )

    if func_name in {"precision_score", "recall_score", "f1_score"}:
        # output is always replicated, and the output can be an array
        # if average=None so we have to set it
        # TODO this shouldn't be done if output is float?
        _set_REP(
            typemap,
            metadata,
            diag_info,
            lhs,
            array_dists,
            f"output of {func_name} is REP",
            rhs.loc,
        )
        dist_arg0 = is_distributable_typ(typemap[rhs.args[0].name])
        dist_arg1 = is_distributable_typ(typemap[rhs.args[1].name])
        if dist_arg0 and dist_arg1:
            _meet_array_dists(typemap, rhs.args[0].name, rhs.args[1].name, array_dists)
        elif not dist_arg0 and dist_arg1:
            _set_REP(
                typemap,
                metadata,
                diag_info,
                rhs.args[1].name,
                array_dists,
                f"first input of {func_name} is non-distributable",
                rhs.loc,
            )
        elif not dist_arg1 and dist_arg0:
            _set_REP(
                typemap,
                metadata,
                diag_info,
                rhs.args[0].name,
                array_dists,
                f"second input of {func_name} is non-distributable",
                rhs.loc,
            )

    if func_name == "log_loss":
        _analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
            typemap, lhs, func_name, rhs, kws, array_dists
        )
        # labels is an optional kw arg, so check if it is provided.
        # if it is provided, then set it to REP
        if "labels" in kws:
            labels_arg_name = kws["labels"].name
            _set_REP(
                typemap,
                metadata,
                diag_info,
                labels_arg_name,
                array_dists,
                "labels when provided are assumed to be REP",
                rhs.loc,
            )

    if func_name == "accuracy_score":
        _analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
            typemap, lhs, func_name, rhs, kws, array_dists
        )

    if func_name == "confusion_matrix":
        # output is always replicated, and the output is an array
        _set_REP(
            typemap,
            metadata,
            diag_info,
            lhs,
            array_dists,
            f"output of {func_name} is REP",
            rhs.loc,
        )
        _analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
            typemap, lhs, func_name, rhs, kws, array_dists
        )
        # labels is an optional kw arg, so check if it is provided.
        # if it is provided, then set it to REP
        if "labels" in kws:
            labels_arg_name = kws["labels"].name
            _set_REP(
                typemap,
                metadata,
                diag_info,
                labels_arg_name,
                array_dists,
                "labels when provided are assumed to be REP",
                rhs.loc,
            )


def _analyze_sklearn_score_err_ytrue_ypred_optional_sample_weight(
    typemap, lhs, func_name, rhs, kws, array_dists
):
    """
    Analyze for sklearn functions like accuracy_score and mean_squared_error.
    In these we have a y_true, y_pred and optionally a sample_weight.
    Distribution of all 3 should match.
    """

    # sample_weight is an optional kw arg, so check if it is provided.
    # if it is provided and it is not none (because if it is none it's
    # as good as not being provided), then get the "name"
    sample_weight_arg_name = None
    if "sample_weight" in kws and (typemap[kws["sample_weight"].name] != types.none):
        sample_weight_arg_name = kws["sample_weight"].name

    # check if all 3 (y_true, y_pred, sample_weight) are distributable types
    dist_y_true = is_distributable_typ(typemap[rhs.args[0].name])
    dist_y_pred = is_distributable_typ(typemap[rhs.args[1].name])
    # if sample_weight is not provided, we can act as if it is distributable
    dist_sample_weight = (
        is_distributable_typ(typemap[sample_weight_arg_name])
        if sample_weight_arg_name
        else True
    )

    # if any of the 3 are not distributable, set top_dist to REP
    top_dist = Distribution.OneD
    if not (dist_y_true and dist_y_pred and dist_sample_weight):
        top_dist = Distribution.REP

    # Match distribution of y_true and y_pred, with top dist as
    # computed above, i.e. set to REP if any of the types
    # are not distributable
    _meet_array_dists(
        typemap,
        rhs.args[0].name,
        rhs.args[1].name,
        array_dists,
        top_dist=top_dist,
    )
    if sample_weight_arg_name:
        # Match distribution of y_true and sample_weight
        _meet_array_dists(
            typemap,
            rhs.args[0].name,
            sample_weight_arg_name,
            array_dists,
            top_dist=top_dist,
        )
        # Match distribution of y_pred and sample_weight
        _meet_array_dists(
            typemap,
            rhs.args[1].name,
            sample_weight_arg_name,
            array_dists,
            top_dist=top_dist,
        )


call_registry = DistributedAnalysisCallRegistry()
