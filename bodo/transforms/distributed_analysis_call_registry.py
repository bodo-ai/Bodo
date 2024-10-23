# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Provides a registry of function call handlers for distributed analysis.
"""

from numba.core.ir_utils import guard

from bodo.transforms.distributed_analysis import (
    Distribution,
    _meet_array_dists,
    _set_REP,
    _set_var_dist,
)
from bodo.utils.typing import BodoError, get_overload_const_str, is_overload_none


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
            ("fit", "BodoXGBClassifierType"): meet_first_2_args_analysis_xgb_fit,
            ("predict", "BodoXGBClassifierType"): meet_out_first_arg_analysis,
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
        rhs.args[0],
        ctx.array_dists,
        "The scalar array must be duplicated for array_to_repeated_array_item_array.",
        rhs.loc,
    )


call_registry = DistributedAnalysisCallRegistry()
