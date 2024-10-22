# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Provides a registry of function call handlers for distributed analysis.
"""

from bodo.transforms.distributed_analysis import _meet_array_dists


class DistributedAnalysisContext:
    """Distributed analysis context data needed for handling calls"""

    def __init__(self, typemap, array_dists, equiv_set):
        self.typemap = typemap
        self.array_dists = array_dists
        self.equiv_set = equiv_set


class DistributedAnalysisCallRegistry:
    """Registry of function call handlers for distributed analysis"""

    def __init__(self):
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


call_registry = DistributedAnalysisCallRegistry()
