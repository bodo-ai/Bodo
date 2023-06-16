package com.bodosql.calcite.adapter.snowflake

import org.apache.calcite.plan.RelOptRule

class SnowflakeRules private constructor() {
    companion object {
        @JvmField
        val SNOWFLAKE_AGGREGATE_RULE: RelOptRule =
            SnowflakeAggregateRule.Config.DEFAULT.toRule()

        @JvmField
        val SNOWFLAKE_AGGREGATE_FILTER_RULE: RelOptRule =
            SnowflakeAggregateRule.Config.WITH_FILTER.toRule()

        @JvmField
        val SNOWFLAKE_LIMIT_RULE: RelOptRule =
            SnowflakeLimitRule.Config.DEFAULT.toRule()

        @JvmField
        val SNOWFLAKE_LIMIT_FILTER_RULE: RelOptRule =
            SnowflakeLimitRule.Config.WITH_FILTER.toRule()

        @JvmField
        val TO_PANDAS: RelOptRule =
            SnowflakeToPandasConverterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_RULES: List<RelOptRule> = listOf(
            SNOWFLAKE_AGGREGATE_RULE,
            SNOWFLAKE_AGGREGATE_FILTER_RULE,
            SNOWFLAKE_LIMIT_RULE,
            SNOWFLAKE_LIMIT_FILTER_RULE,
        )

        fun rules(): List<RelOptRule> = SNOWFLAKE_RULES
    }
}
