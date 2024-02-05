package com.bodosql.calcite.adapter.pandas

import org.apache.calcite.plan.RelOptRule

class PandasRules private constructor() {
    companion object {
        @JvmField
        val PANDAS_PROJECT_RULE: RelOptRule = PandasProjectRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_FILTER_RULE: RelOptRule = PandasFilterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_MIN_ROW_NUMBER_FILTER_RULE: RelOptRule = PandasMinRowNumberFilterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_AGGREGATE_RULE: RelOptRule = PandasAggregateRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_JOIN_RULE: RelOptRule = PandasJoinRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_SORT_RULE: RelOptRule = PandasSortRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_UNION_RULE: RelOptRule = PandasUnionRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_INTERSECT_RULE: RelOptRule = PandasIntersectRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_MINUS_RULE: RelOptRule = PandasMinusRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_VALUES_RULE: RelOptRule = PandasValuesRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_SAMPLE_RULE: RelOptRule = PandasSampleRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_ROW_SAMPLE_RULE: RelOptRule = PandasRowSampleRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_FLATTEN_RULE: RelOptRule = PandasFlattenRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_WINDOW_RULE: RelOptRule = PandasWindowRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_TABLE_FUNCTION_SCAN_RULE: RelOptRule = PandasTableFunctionScanRule.DEFAULT_CONFIG.toRule()

        // TODO(jsternberg): The following rules aren't necessarily correct.
        // These ones should probably be part of a different section of the codebase
        // as they're physical interactions with outside sources. The current code
        // treats these as logical operations rather than physical operations though
        // so just going to leave them as-is for the sake of transitioning.
        @JvmField
        val PANDAS_TABLE_MODIFY_RULE: RelOptRule = PandasTableModifyRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_TABLE_CREATE_RULE: RelOptRule = PandasTableCreateRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_TABLE_SCAN: RelOptRule = PandasTableScanRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_TARGET_TABLE_SCAN: RelOptRule = PandasTargetTableScanRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_JOIN_REBALANCE_OUTPUT_RULE: RelOptRule = PandasJoinRebalanceOutputRule.Config.DEFAULT_CONFIG.toRule()

        @JvmField
        val PANDAS_RULES: List<RelOptRule> = listOf(
            PANDAS_PROJECT_RULE,
            PANDAS_FILTER_RULE,
            PANDAS_AGGREGATE_RULE,
            PANDAS_JOIN_RULE,
            PANDAS_SORT_RULE,
            PANDAS_UNION_RULE,
            PANDAS_INTERSECT_RULE,
            PANDAS_MINUS_RULE,
            PANDAS_VALUES_RULE,
            PANDAS_SAMPLE_RULE,
            PANDAS_ROW_SAMPLE_RULE,
            PANDAS_FLATTEN_RULE,
            PANDAS_WINDOW_RULE,
            PANDAS_TABLE_MODIFY_RULE,
            PANDAS_TABLE_CREATE_RULE,
            PANDAS_TABLE_SCAN,
            PANDAS_TARGET_TABLE_SCAN,
            PANDAS_TABLE_FUNCTION_SCAN_RULE,
            PANDAS_MIN_ROW_NUMBER_FILTER_RULE,
        )

        fun rules(): List<RelOptRule> = PANDAS_RULES
    }
}
