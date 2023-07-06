package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.Dataframe
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.plan.makeCost
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

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "reading table"
    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails() = (if (input is SnowflakeTableScan) {
        getTableName()
    } else {
        getSnowflakeSQL()
    })!!

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

    override fun emit(implementor: PandasRel.Implementor): Dataframe =
        if (isStreaming()) {
            implementor.buildStreamingNoTimer { ctx -> generateStreamingDataFrame(ctx) }
        } else {
            implementor.build { ctx -> generateNonStreamingDataFrame(ctx)}
        }

    /**
     * Generate the required read expression for processing the P
     */
    private fun generateReadExpr(ctx: PandasRel.BuildContext): Expr.Call {
        if (input is SnowflakeTableScan) {
            // The input is a table.
            return sqlReadTable(input as SnowflakeTableScan, ctx)
        } else {
            return readSql(ctx)
        }
    }

    private fun getTableName()  = (input as SnowflakeTableScan).catalogTable.name
    private fun getSchemaName()  = (input as SnowflakeTableScan).catalogTable.schema.name

    /**
     * Generate the code required to read a table. This path is necessary because the Snowflake
     * API allows for more accurate sampling when operating directly on a table.
     */
    private fun sqlReadTable(tableScan: SnowflakeTableScan, ctx: PandasRel.BuildContext): Expr.Call {
        val tableName = getTableName()
        val schemaName = getSchemaName()
        val relInput = input as SnowflakeRel
        val args = listOf(
            Expr.StringLiteral(tableName),
            Expr.StringLiteral(relInput.generatePythonConnStr(schemaName))
        )
        return Expr.Call("pd.read_sql", args, getNamedArgs(ctx, true))
    }

    private fun getSnowflakeSQL(): String {
        // Use the snowflake dialect for generating the sql string.
        val rel2sql = RelToSqlConverter(SnowflakeSqlDialect.DEFAULT)
        return rel2sql.visitRoot(input)
            .asSelect()
            .toSqlString { c ->
                c.withClauseStartsLine(false)
                    .withDialect(SnowflakeSqlDialect.DEFAULT)
            }
            .toString()
    }

    /**
     * Generate a read expression for SQL operations that will be pushed into Snowflake. The currently
     * supported operations consist of Aggregates and filters.
     */
    private fun readSql(ctx: PandasRel.BuildContext): Expr.Call {
        val sql = getSnowflakeSQL()
        val relInput = input as SnowflakeRel
        val args = listOf(
            Expr.StringLiteral(sql),
            // We don't use a schema name because we've already fully qualified
            // all table references and it's better if this doesn't have any
            // potentially unexpected behavior.
            Expr.StringLiteral(relInput.generatePythonConnStr(""))
        )
        return Expr.Call("pd.read_sql", args, getNamedArgs(ctx, false))
    }

    private fun getNamedArgs(ctx: PandasRel.BuildContext, isTable: Boolean): List<Pair<String, Expr>> {
        return listOf(
            "_bodo_is_table_input" to Expr.BooleanLiteral(isTable),
            "_bodo_chunksize" to getStreamingBatchArg(ctx),
        )
    }

    /**
     * Generate the argument that will be passed for '_bodo_chunksize' in a read_sql call. If
     * we are not streaming Python expects None.
     */
    private fun getStreamingBatchArg(ctx: PandasRel.BuildContext): Expr =
        if (isStreaming()) {
            Expr.IntegerLiteral(ctx.streamingOptions().chunkSize)
        } else {
            Expr.None
        }

    /**
     * Generate the DataFrame for streaming code. The code leverages the code generation process
     * from initStreamingIoLoop and does not yet support RelNode caching.
     */
    private fun generateStreamingDataFrame(ctx: PandasRel.BuildContext): Dataframe {
        val readExpr = generateReadExpr(ctx)
        // Note: Using getRowType() instead of rowType because Calcite
        // lazily initializes the data.
        val initOutput: Variable =
            ctx.initStreamingIoLoop(readExpr, getRowType())
        return Dataframe(initOutput.name, this)
    }

    /**
     * Generate the DataFrame for the non-streaming code.
     */
    private fun generateNonStreamingDataFrame(ctx: PandasRel.BuildContext): Dataframe {
        val readExpr = generateReadExpr(ctx)
        return ctx.returns(readExpr)
    }
}
