package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Expr.StringLiteral
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.ConventionTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.rel2sql.BodoRelToSqlConverter
import org.apache.calcite.rex.RexInputRef

class SnowflakeToPandasConverter(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode) :
    ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits.replace(PandasRel.CONVENTION), input), PandasRel {

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

    override fun nodeDetails() = (
        if (isWholeTableRead()) {
            getTableName(input as SnowflakeRel)
        } else {
            getSnowflakeSQL()
        }
        )!!

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
    }

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()
        currentPipeline.addInitialization(Op.Assign(readerVar, generateReadExpr(ctx)))
        return readerVar
    }

    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.io.arrow_reader.arrow_reader_del", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

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

    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            implementor.createStreamingPipeline()
            implementor.buildStreaming(
                { ctx -> initStateVariable(ctx) },
                { ctx, stateVar -> generateStreamingTable(ctx, stateVar) },
                { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
            )
        } else {
            implementor.build { ctx -> generateNonStreamingTable(ctx) }
        }

    /**
     * Generate the required read expression for processing the P
     */
    private fun generateReadExpr(ctx: PandasRel.BuildContext): Expr.Call {
        val readExpr = when {
            (isWholeTableRead()) -> sqlReadTable(input as SnowflakeTableScan, ctx)
            else -> readSql(ctx)
        }
        return readExpr
    }

    private fun getTableName(input: SnowflakeRel) = input.getCatalogTable().name
    private fun getSchemaName(input: SnowflakeRel) = input.getCatalogTable().fullPath[1]

    private fun getDatabaseName(input: SnowflakeRel) = input.getCatalogTable().fullPath[0]

    /**
     * Generate the code required to read a table. This path is necessary because the Snowflake
     * API allows for more accurate sampling when operating directly on a table.
     */
    private fun sqlReadTable(tableScan: SnowflakeTableScan, ctx: PandasRel.BuildContext): Expr.Call {
        val tableName = getTableName(tableScan)
        val schemaName = getSchemaName(tableScan)
        val databaseName = getDatabaseName(tableScan)
        val relInput = input as SnowflakeRel
        val args = listOf(
            StringLiteral(tableName),
            StringLiteral(relInput.generatePythonConnStr(databaseName, schemaName)),
        )
        return Expr.Call("pd.read_sql", args, getNamedArgs(ctx, true, Expr.None, Expr.None))
    }

    private fun getSnowflakeSQL(): String {
        // Use the snowflake dialect for generating the sql string.
        val rel2sql = BodoRelToSqlConverter(BodoSnowflakeSqlDialect.DEFAULT)
        return rel2sql.visitRoot(input)
            .asSelect()
            .toSqlString { c ->
                c.withClauseStartsLine(false)
                    .withDialect(BodoSnowflakeSqlDialect.DEFAULT)
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

        val passTableInfo = canUseOptimizedReadSqlPath(relInput)

        val bodoTableNameExpr = if (passTableInfo) {
            // Store the original indices to allow handling renaming.
            val catalogTable = relInput.getCatalogTable()
            // Get the fully qualified name.
            StringLiteral(catalogTable.qualifiedName)
        } else {
            Expr.None
        }

        val originalIndices = if (passTableInfo) {
            // TODO: Replace with a global variable?
            Expr.Tuple(getOriginalColumnIndices(relInput).map { i -> Expr.IntegerLiteral(i) })
        } else {
            Expr.None
        }
        val args = listOf(
            StringLiteral(sql),
            // We don't use a schema name because we've already fully qualified
            // all table references, and it's better if this doesn't have any
            // potentially unexpected behavior.
            StringLiteral(relInput.generatePythonConnStr("", "")),
        )
        return Expr.Call("pd.read_sql", args, getNamedArgs(ctx, false, bodoTableNameExpr, originalIndices))
    }

    /**
     * Helper function to remap each column to its original index.
     * This function assumes that canUseOptimizedReadSqlPath has already
     * been called and evaluates to True.
     */
    private fun getOriginalColumnIndices(node: SnowflakeRel): List<Int> {
        val onlyColumnSubsetVisitor = object : RelVisitor() {
            // Initialize all columns to be in the original location.
            var originalColumns: MutableList<Int> = (0..<node.getRowType().fieldCount).toMutableList()

            override fun visit(node: RelNode, ordinal: Int, parent: RelNode?) {
                if (node is SnowflakeProject) {
                    for (i in 0..<originalColumns.size) {
                        val project = node.projects[originalColumns[i]]
                        if (project !is RexInputRef) {
                            throw RuntimeException("getOriginalColumnIndices() requires only InputRefs")
                        }
                        // Remap to the original location.
                        originalColumns[i] = project.index
                    }
                    node.childrenAccept(this)
                } else if (node is SnowflakeAggregate) {
                    // Re-map columns to either the group or original column and
                    // then visit the children.
                    val groups = node.groupSet.toList()
                    if (node.aggCallList.isNotEmpty()) {
                        // Note: This is enforced in canUseOptimizedReadSqlPath
                        throw RuntimeException("getOriginalColumnIndices() only support select distinct")
                    }
                    for (i in 0..<originalColumns.size) {
                        val index = originalColumns[i]
                        originalColumns[i] = groups[index]
                    }
                    node.childrenAccept(this)
                } else if (node is SnowflakeFilter || node is SnowflakeSort) {
                    // Enable moving past filters to get the original table
                    // for filter and limit.
                    node.childrenAccept(this)
                } else if (node is SnowflakeTableScan) {
                    for (i in 0..<originalColumns.size) {
                        originalColumns[i] = node.getOriginalColumnIndex(originalColumns[i])
                    }
                } else {
                    return
                }
            }
        }

        onlyColumnSubsetVisitor.go(node)
        return onlyColumnSubsetVisitor.originalColumns.toList()
    }

    /**
     * Helper function to determine if we can use an optimized read_sql path.
     * currently, this is true if:
     *  1. The query has only 1 root table.
     *  2. The input query only selects a subset of the columns from that root table,
     *      and possibly a filter or limit.
     */
    private fun canUseOptimizedReadSqlPath(node: SnowflakeRel): Boolean {
        val onlyColumnSubsetVisitor = object : RelVisitor() {
            var seenInvalidNode: Boolean = false

            override fun visit(node: RelNode, ordinal: Int, parent: RelNode?) {
                if (seenInvalidNode) {
                    return
                }

                if (node is SnowflakeProject) {
                    // This is already enforced in the SnowflakeProject, but this will
                    // help protect against unexpected changes.
                    if (node.projects.any { x -> x !is RexInputRef }) {
                        seenInvalidNode = true
                    } else {
                        node.childrenAccept(this)
                    }
                } else if (node is SnowflakeFilter || node is SnowflakeSort) {
                    // Enable moving past filters to get the original table
                    // for filter and limit.
                    node.childrenAccept(this)
                } else if (node is SnowflakeAggregate) {
                    // Aggregates may change types except for distinct,
                    // so we don't want to point to the original table.
                    if (node.aggCallList.isNotEmpty()) {
                        seenInvalidNode = true
                    } else {
                        node.childrenAccept(this)
                    }
                } else if (node is SnowflakeTableScan || node is Values) {
                    // Found the root table.
                    return
                } else {
                    throw RuntimeException("canUseOptimizedReadSqlPath(): Unexpected SnowflakeRel encountered. Please explicitly support each SnowflakeRel")
                }
            }
        }

        onlyColumnSubsetVisitor.go(node)
        return !onlyColumnSubsetVisitor.seenInvalidNode
    }

    private fun getNamedArgs(ctx: PandasRel.BuildContext, isTable: Boolean, origTableExpr: Expr, origTableIndices: Expr): List<Pair<String, Expr>> {
        return listOf(
            "_bodo_is_table_input" to Expr.BooleanLiteral(isTable),
            "_bodo_orig_table_name" to origTableExpr,
            "_bodo_orig_table_indices" to origTableIndices,
            "_bodo_chunksize" to getStreamingBatchArg(ctx),
            "_bodo_read_as_table" to Expr.BooleanLiteral(true),
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
     * Generate the Table for the body of the streaming code.
     */
    private fun generateStreamingTable(ctx: PandasRel.BuildContext, stateVar: StateVariable): BodoEngineTable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val tableChunkVar = builder.symbolTable.genTableVar()
        val isLastVar = currentPipeline.getExitCond()
        val outputControl = builder.symbolTable.genOutputControlVar()
        currentPipeline.addOutputControl(outputControl)
        val readArrowNextCall = Expr.Call("bodo.io.arrow_reader.read_arrow_next", listOf(stateVar, outputControl))
        builder.add(Op.TupleAssign(listOf(tableChunkVar, isLastVar), readArrowNextCall))
        return BodoEngineTable(tableChunkVar.emit(), this)
    }

    /**
     * Generate the Table for the non-streaming code.
     */
    private fun generateNonStreamingTable(ctx: PandasRel.BuildContext): BodoEngineTable {
        val readExpr = generateReadExpr(ctx)
        return ctx.returns(readExpr)
    }

    /**
     * Is the Snowflake read for a whole table without any column pruning,
     * filtering, or other snowflake nodes.
     */
    private fun isWholeTableRead(): Boolean {
        return input is SnowflakeTableScan && !(input as SnowflakeTableScan).prunesColumns()
    }
}
