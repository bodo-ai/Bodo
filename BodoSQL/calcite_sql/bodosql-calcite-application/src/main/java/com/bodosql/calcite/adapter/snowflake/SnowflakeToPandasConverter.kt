package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.*
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.*
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.rel2sql.RelToSqlConverter
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFieldImpl
import org.apache.calcite.rel.type.RelRecordType

class SnowflakeToPandasConverter(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode) :
    ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits, input), PandasRel {


    override fun copy(traitSet: RelTraitSet, inputs: List<RelNode>): RelNode {
        return SnowflakeToPandasConverter(cluster, traitSet, sole(inputs))
    }

    /**
     * Even if SnowflakeToPandasConverter is a PandasRel, it is still
     * a single process operation which the other ranks wait on and doesn't
     * benefit from parallelism.
     */
    override fun splitCount(numRanks: Int): Int = 1

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost? {
        // Determine the average row size which determines how much data is returned.
        // If this data isn't available, just assume something like 4 bytes for each column.
        val averageRowSize = mq.getAverageRowSize(this)
            ?: (4.0 * getRowType().fieldCount)

        // Complete data size using the row count.
        val rows = mq.getRowCount(this)
        val dataSize = averageRowSize * rows

        // Determine parallelism or default to no parallelism.
        val parallelism = mq.splitCount(this) ?: 1

        // IO and memory are related to the number of bytes returned.
        val io = dataSize / parallelism
        return planner.makeCost(rows = rows, io = io, mem = io)
    }

    /**
     * While snowflake has its own casing convention, the pandas code will
     * convert these names to another casing to match with existing conventions
     * within sqlalchemy.
     *
     * We modify that casing here by changing the field names of the derived type.
     */
    override fun deriveRowType(): RelDataType {
        return RelRecordType(super.deriveRowType().fieldList.map { field ->
            val name = if (field.name.equals(field.name.uppercase())) {
                field.name.lowercase()
            } else field.name
            RelDataTypeFieldImpl(name, field.index, field.type)
        })
    }

    override fun emit(
        visitor: PandasCodeGenVisitor,
        builder: Module.Builder,
        inputs: () -> List<Dataframe>
    ): Dataframe {
        if (isStreaming()) {
            return generateStreamingDataFrame(visitor)
        } else {
            return generateNonStreamingDataFrame(visitor, builder)
        }
    }

    /**
     * Generate the required read expression for processing the P
     */
    private fun generateReadExpr(visitor: PandasCodeGenVisitor): Expr.Call {
        if (input is SnowflakeTableScan) {
            // The input is a table.
            return sqlReadTable(input as SnowflakeTableScan, visitor)
        } else {
            return readSql(visitor)
        }
    }

    /**
     * Generate the code required to read a table. This path is necessary because the Snowflake
     * API allows for more accurate sampling when operating directly on a table.
     */
    private fun sqlReadTable(tableScan: SnowflakeTableScan, visitor: PandasCodeGenVisitor): Expr.Call {
        val relInput = input as SnowflakeRel
        val catalogTable = tableScan.catalogTable
        val tableName = catalogTable.name
        val schemaName = catalogTable.schema.name
        val args = listOf(
            Expr.StringLiteral(tableName),
            Expr.StringLiteral(relInput.generatePythonConnStr(schemaName))
        )
        return Expr.Call("pd.read_sql", args, getNamedArgs(visitor, true))
    }

    /**
     * Generate a read expression for SQL operations that will be pushed into Snowflake. The currently
     * supported operations consist of Aggregates and filters.
     */
    private fun readSql(visitor: PandasCodeGenVisitor): Expr.Call {
        val relInput = input as SnowflakeRel
        // Use the snowflake dialect for generating the sql string.
        val rel2sql = RelToSqlConverter(SnowflakeSqlDialect.DEFAULT)
        val sql = rel2sql.visitRoot(input)
            .asSelect()
            .toSqlString { c ->
                c.withClauseStartsLine(false)
                    .withDialect(SnowflakeSqlDialect.DEFAULT)
            }
            .toString()
        val args = listOf(
            Expr.StringLiteral(sql),
            // We don't use a schema name because we've already fully qualified
            // all table references and it's better if this doesn't have any
            // potentially unexpected behavior.
            Expr.StringLiteral(relInput.generatePythonConnStr(""))
        )
        return Expr.Call("pd.read_sql", args, getNamedArgs(visitor, false))
    }

    private fun getNamedArgs(visitor: PandasCodeGenVisitor, isTable: Boolean): List<Pair<String, Expr>> {
        return listOf(
            "_bodo_is_table_input" to Expr.BooleanLiteral(isTable),
            "_bodo_chunksize" to getStreamingBatchArg(visitor),
        )
    }

    /**
     * Generate the argument that will be passed for '_bodo_chunksize' in a read_sql call. If
     * we are not streaming Python expects None.
     */
    private fun getStreamingBatchArg(visitor: PandasCodeGenVisitor) : Expr {
        if (isStreaming()) {
            return visitor.BATCH_SIZE
        } else {
            return Expr.None
        }
    }

    /**
     * Generate the DataFrame for streaming code. The code leverages the code generation process
     * from initStreamingIoLoop and does not yet support RelNode caching.
     */
    private fun generateStreamingDataFrame(visitor: PandasCodeGenVisitor): Dataframe {
        val readExpr = generateReadExpr(visitor)
        val columnNames = this.getRowType().fieldNames
        val initOutput: Variable =
            visitor.initStreamingIoLoop(readExpr, columnNames)
        return Dataframe(initOutput.name, this)
    }

    /**
     * Generate the DataFrame for the non-streaming code.
     */
    private fun generateNonStreamingDataFrame(visitor: PandasCodeGenVisitor, builder: Module.Builder): Dataframe {
        val readExpr = generateReadExpr(visitor)
        val df = builder.genDataframe(this)
        builder.add(Op.Assign(df.variable, readExpr))
        return df
    }

    /**
     * Is this used in a streaming operation
     */
    private fun isStreaming(): Boolean {
        return traitSet.contains(BatchingProperty.STREAMING)
    }
}
