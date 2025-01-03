package com.bodosql.interactive

import com.bodosql.calcite.adapter.bodo.bodoPhysicalProject
import com.bodosql.calcite.application.PythonEntryPoint.Companion.getPandasString
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.schema.LocalSchema
import com.bodosql.calcite.table.BodoSQLColumn
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
        val isStreaming = true
        val schema = LocalSchema("__BODOLOCAL__")
        var cols: ArrayList<BodoSQLColumn> = ArrayList()
        val dataType: BodoSQLColumnDataType = BodoSQLColumnDataType.INT64
        val dataTypeInfo = ColumnDataTypeInfo(dataType, true)
        val column = BodoSQLColumnImpl("A", dataTypeInfo)
        cols.add(column)
        val column2 = BodoSQLColumnImpl("D", dataTypeInfo)
        cols.add(column2)
        val column3 = BodoSQLColumnImpl("C", dataTypeInfo)
        cols.add(column3)
        val table: BodoSqlTable =
            LocalTable(
                "TABLE1",
                schema.fullPath,
                cols,
                true,
                "table1",
                "TABLE_1 WRITE HERE (%s, %s)",
                false,
                "MEMORY",
                null,
                mapOf(),
            )
        schema.addTable(table)
        var cols2: ArrayList<BodoSQLColumn> = ArrayList()
        cols2.add(column)
        val column4 = BodoSQLColumnImpl("B", dataTypeInfo)
        cols2.add(column4)
        val column5 = BodoSQLColumnImpl("C", dataTypeInfo)
        cols2.add(column5)
        val table2: BodoSqlTable =
            LocalTable(
                "TABLE2",
                schema.fullPath,
                cols2,
                true,
                "table2",
                "TABLE_2 WRITE HERE (%s, %s)",
                false,
                "MEMORY",
                null,
                mapOf(),
            )
        schema.addTable(table2)
        val table3: BodoSqlTable =
            LocalTable(
                "TABLE3",
                schema.fullPath,
                cols2,
                true,
                "table3",
                "TABLE_3 WRITE HERE (%s, %s)",
                false,
                "MEMORY",
                null,
                mapOf(),
            )
        schema.addTable(table3)
        val generator =
            RelationalAlgebraGenerator(
                schema,
                isStreaming,
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
        val paramTypes = MutableList(1) { ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false) }
        val namedParamTypes =
            java.util.Map.of(
                "a",
                ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false),
                "c",
                ColumnDataTypeInfo(BodoSQLColumnDataType.INT64, false),
            )
        println("SQL query:")
        println(sql)
        val optimizedPlanStr = getRelationalAlgebraString(generator, sql, paramTypes, namedParamTypes)
        println("Optimized plan:")
        println(optimizedPlanStr)
        val pandasStr = getPandasString(generator, sql, paramTypes, namedParamTypes)
        println("Generated code:")
        println(pandasStr)
        println("Lowered globals:")
        println(generator.loweredGlobalVariables)
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
