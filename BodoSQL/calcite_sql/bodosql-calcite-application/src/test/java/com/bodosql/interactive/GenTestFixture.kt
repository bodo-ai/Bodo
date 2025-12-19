package com.bodosql.calcite.interactive

import com.bodosql.calcite.application.PythonEntryPoint
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
                true,
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
                // Disable Streaming Sort Limit Offset
                // Maintain case sensitivity in the Snowflake style by default
                "SNOWFLAKE",
                false,
                true,
                null,
            )

        println("SQL query:")
        println(sql + "\n")
        if (generateCode) {
            val pair = PythonEntryPoint.getPandasAndPlanString(generator, sql, true, mutableListOf(), mutableMapOf())
            println("Optimized plan:")
            println(pair.plan + "\n")
            println("Generated code:")
            println(pair.code + "\n")
        } else {
            println("DDL OUTPUT:")
            println(PythonEntryPoint.executeDDL(generator, sql))
        }
    }
}
