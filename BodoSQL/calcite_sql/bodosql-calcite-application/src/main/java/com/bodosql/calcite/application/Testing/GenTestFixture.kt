package com.bodosql.calcite.application.testing

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.traits.BatchingProperty

abstract class GenTestFixture {
    abstract fun isIceberg(): Boolean

    abstract fun supportsTimestampTZ(): Boolean

    abstract fun getCatalog(): BodoSQLCatalog

    abstract fun getSchema(): BodoSqlSchema

    /**
     * @param sql query to be compiled
     * @param generateCode Controls if we are generating code or executing DDL. You can set this to false if you want to observe the actual execution of DDL statements.
     */
    @Throws(Exception::class)
    fun run(
        sql: String,
        generateCode: Boolean = true,
    ) {
        val generator =
            RelationalAlgebraGenerator(
                getCatalog(),
                getSchema(),
                RelationalAlgebraGenerator.STREAMING_PLANNER,
                0,
                1,
                BatchingProperty.defaultBatchSize,
                // Always hide credentials
                true,
                // Enable Iceberg for testing
                isIceberg(),
                // Enable TIMESTAMP_TZ for testing
                supportsTimestampTZ(),
                // Enable Join Runtime filters for Testing
                true,
                // Disable Streaming Sort
                false,
                // Maintain case sensitivity in the Snowflake style by default
                "SNOWFLAKE",
                false,
            )

        println("SQL query:")
        println(sql + "\n")
        if (generateCode) {
            val pair = generator.getPandasAndPlanString(sql, true)
            println("Optimized plan:")
            println(pair.sqlPlan + "\n")
            println("Generated code:")
            println(pair.pdCode + "\n")
        } else {
            println("DDL OUTPUT:")
            println(generator.executeDDL(sql))
        }
    }
}
