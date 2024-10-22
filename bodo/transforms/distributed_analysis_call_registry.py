# Copyright (C) 2024 Bodo Inc. All rights reserved.
"""
Provides a registry of function call handlers for distributed analysis.
"""


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


call_registry = DistributedAnalysisCallRegistry()
