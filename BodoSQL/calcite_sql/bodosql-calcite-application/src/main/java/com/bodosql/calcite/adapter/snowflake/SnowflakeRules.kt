package com.bodosql.calcite.adapter.snowflake

import org.apache.calcite.plan.RelOptRule

class SnowflakeRules private constructor() {
    companion object {
        @JvmField
        val SNOWFLAKE_AGGREGATE_RULE: RelOptRule =
            SnowflakeAggregateRule.Config.DEFAULT_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_NESTED_AGGREGATE_RULE: RelOptRule =
            SnowflakeAggregateRule.Config.NESTED_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_STREAMING_AGGREGATE_RULE: RelOptRule =
            SnowflakeAggregateRule.Config.STREAMING_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_LIMIT_RULE: RelOptRule =
            SnowflakeLimitRule.Config.DEFAULT_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_NESTED_LIMIT_RULE: RelOptRule =
            SnowflakeLimitRule.Config.NESTED_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_STREAMING_LIMIT_RULE: RelOptRule =
            SnowflakeLimitRule.Config.STREAMING_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_FILTER_RULE: RelOptRule =
            SnowflakeFilterRule.Config.DEFAULT_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_NESTED_FILTER_RULE: RelOptRule =
            SnowflakeFilterRule.Config.NESTED_CONFIG.toRule()

        @JvmField
        val TO_PANDAS: RelOptRule =
            SnowflakeToPandasConverterRule.DEFAULT_CONFIG.toRule()

        @JvmField
        val SNOWFLAKE_RULES: List<RelOptRule> = listOf(
            SNOWFLAKE_AGGREGATE_RULE,
            SNOWFLAKE_NESTED_AGGREGATE_RULE,
            SNOWFLAKE_STREAMING_AGGREGATE_RULE,
            SNOWFLAKE_LIMIT_RULE,
            SNOWFLAKE_NESTED_LIMIT_RULE,
            SNOWFLAKE_STREAMING_LIMIT_RULE,
            SNOWFLAKE_FILTER_RULE,
            SNOWFLAKE_NESTED_FILTER_RULE,
        )

        fun rules(): List<RelOptRule> = SNOWFLAKE_RULES
    }
}
