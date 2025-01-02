package com.bodosql.interactive

import com.bodosql.calcite.adapter.bodo.bodoPhysicalProject
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.schema.LocalSchema
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType
import com.bodosql.calcite.table.BodoSQLColumnImpl
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.ColumnDataTypeInfo
import com.bodosql.calcite.table.LocalTable
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.RelOptUtil

/** Class for locally testing codegen.  */
object BodoGenTest {
    @Throws(Exception::class)
    @JvmStatic
    fun main(args: Array<String>) {
        val sql = "select * from table1"
        val plannerChoice = RelationalAlgebraGenerator.STREAMING_PLANNER
        val schema = LocalSchema("__BODOLOCAL__")
        var arr: ArrayList<*> = ArrayList<Any?>()
        val dataType: BodoSQLColumnDataType = BodoSQLColumnDataType.INT64
        val dataTypeInfo = ColumnDataTypeInfo(dataType, true)
        val column = BodoSQLColumnImpl("A", dataTypeInfo)
        arr.add(column)
        val column2 = BodoSQLColumnImpl("D", dataTypeInfo)
        arr.add(column2)
        val column3 = BodoSQLColumnImpl("C", dataTypeInfo)
        arr.add(column3)
        val table: BodoSqlTable =
            LocalTable(
                "TABLE1",
                schema.getFullPath(),
                arr,
                true,
                "table1",
                "TABLE_1 WRITE HERE (%s, %s)",
                false,
                "MEMORY",
                null,
                null,
            )
        schema.addTable(table)
        arr = ArrayList<Any?>()
        arr.add(column)
        val column4 = BodoSQLColumnImpl("B", dataTypeInfo)
        arr.add(column4)
        val column5 = BodoSQLColumnImpl("C", dataTypeInfo)
        arr.add(column5)
        val table2: BodoSqlTable =
            LocalTable(
                "TABLE2",
                schema.getFullPath(),
                arr,
                true,
                "table2",
                "TABLE_2 WRITE HERE (%s, %s)",
                false,
                "MEMORY",
                null,
                null,
            )
        schema.addTable(table2)
        val table3: BodoSqlTable =
            LocalTable(
                "TABLE3",
                schema.getFullPath(),
                arr,
                true,
                "table3",
                "TABLE_3 WRITE HERE (%s, %s)",
                false,
                "MEMORY",
                null,
                null,
            )
        schema.addTable(table3)
        val generator =
            RelationalAlgebraGenerator(
                schema,
                plannerChoice,
                0,
                1,
                BatchingProperty.defaultBatchSize,
                true, // Always hide credentials
                true, // Enable Iceberg for testing
                true, // Enable TIMESTAMP_TZ for testing
                true, // Enable Join Runtime filters for Testing
                true, // Disable Streaming Sort for Testing
                false, // Disable Streaming Sort Limit Offset for Testing
                "SNOWFLAKE", // Maintain case sensitivity in the Snowflake style by default
                false, // Only cache identical nodes
                true, // Generate a prefetch call at the beginning of SQL queries
            )
        val paramTypes = java.util.List.of<ColumnDataTypeInfo>(ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false))
        val namedParamTypes =
            java.util.Map.of<String, ColumnDataTypeInfo>(
                "a",
                ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false),
                "c",
                ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false),
            )
        println("SQL query:")
        println(
            """
            $sql
            
            """.trimIndent(),
        )
        val optimizedPlanStr = getRelationalAlgebraString(generator, sql, paramTypes, namedParamTypes)
        println("Optimized plan:")
        println(
            """
            $optimizedPlanStr
            
            """.trimIndent(),
        )
        val pandasStr = generator.getPandasString(sql, paramTypes, namedParamTypes)
        println("Generated code:")
        println(
            """
            $pandasStr
            
            """.trimIndent(),
        )
        println("Lowered globals:")
        println(
            """
            ${generator.loweredGlobalVariables}
            
            """.trimIndent(),
        )
    }

    private fun getRelationalAlgebraString(
        generator: RelationalAlgebraGenerator,
        sql: String,
        paramTypes: List<ColumnDataTypeInfo>,
        namedParamTypes: Map<String, ColumnDataTypeInfo>,
    ): String =
        try {
            val root = generator.getRelationalAlgebra(sql, paramTypes, namedParamTypes)
            RelOptUtil.toString(root.left.bodoPhysicalProject())
        } catch (e: Exception) {
            throw RuntimeException(e)
        }
}
