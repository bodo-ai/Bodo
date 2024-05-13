package com.bodosql.calcite.adapter.bodo

import org.apache.calcite.plan.RelOptRule

class BodoPhysicalRules private constructor() {
    companion object {
        @JvmField
        val BODO_PHYSICAL_PROJECT_RULE: RelOptRule = BodoPhysicalProjectRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_FILTER_RULE: RelOptRule = BodoPhysicalFilterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_MIN_ROW_NUMBER_FILTER_RULE: RelOptRule = BodoPhysicalMinRowNumberFilterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_AGGREGATE_RULE: RelOptRule = BodoPhysicalAggregateRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_JOIN_RULE: RelOptRule = BodoPhysicalJoinRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_SORT_RULE: RelOptRule = BodoPhysicalSortRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_UNION_RULE: RelOptRule = BodoPhysicalUnionRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_INTERSECT_RULE: RelOptRule = BodoPhysicalIntersectRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_MINUS_RULE: RelOptRule = BodoPhysicalMinusRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_VALUES_RULE: RelOptRule = BodoPhysicalValuesRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_SAMPLE_RULE: RelOptRule = BodoPhysicalSampleRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_ROW_SAMPLE_RULE: RelOptRule = BodoPhysicalRowSampleRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_FLATTEN_RULE: RelOptRule = BodoPhysicalFlattenRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_WINDOW_RULE: RelOptRule = BodoPhysicalWindowRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_TABLE_FUNCTION_SCAN_RULE: RelOptRule = BodoPhysicalTableFunctionScanRule.DEFAULT_CONFIG.toRule()

        // TODO(jsternberg): The following rules aren't necessarily correct.
        // These ones should probably be part of a different section of the codebase
        // as they're physical interactions with outside sources. The current code
        // treats these as logical operations rather than physical operations though
        // so just going to leave them as-is for the sake of transitioning.
        @JvmField
        val BODO_PHYSICAL_TABLE_MODIFY_RULE: RelOptRule = BodoPhysicalTableModifyRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_TABLE_CREATE_RULE: RelOptRule = BodoPhysicalTableCreateRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_JOIN_REBALANCE_OUTPUT_RULE: RelOptRule = BodoPhysicalJoinRebalanceOutputRule.Config.DEFAULT_CONFIG.toRule()

        @JvmField
        val BODO_PHYSICAL_RULES: List<RelOptRule> =
            listOf(
                BODO_PHYSICAL_PROJECT_RULE,
                BODO_PHYSICAL_FILTER_RULE,
                BODO_PHYSICAL_AGGREGATE_RULE,
                BODO_PHYSICAL_JOIN_RULE,
                BODO_PHYSICAL_SORT_RULE,
                BODO_PHYSICAL_UNION_RULE,
                BODO_PHYSICAL_INTERSECT_RULE,
                BODO_PHYSICAL_MINUS_RULE,
                BODO_PHYSICAL_VALUES_RULE,
                BODO_PHYSICAL_SAMPLE_RULE,
                BODO_PHYSICAL_ROW_SAMPLE_RULE,
                BODO_PHYSICAL_FLATTEN_RULE,
                BODO_PHYSICAL_WINDOW_RULE,
                BODO_PHYSICAL_TABLE_MODIFY_RULE,
                BODO_PHYSICAL_TABLE_CREATE_RULE,
                BODO_PHYSICAL_TABLE_FUNCTION_SCAN_RULE,
                BODO_PHYSICAL_MIN_ROW_NUMBER_FILTER_RULE,
            )

        fun rules(): List<RelOptRule> = BODO_PHYSICAL_RULES
    }
}
