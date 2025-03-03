package com.bodosql.calcite.adapter.snowflake

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.application.PythonLoggers
import com.bodosql.calcite.application.operatorTables.CastingOperatorTable
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Expr.StringLiteral
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.rel.core.RuntimeJoinFilterBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.ConventionTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelShuttleImpl
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.core.Values
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.rel2sql.BodoRelToSqlConverter
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexVisitorImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.Util

class SnowflakeToBodoPhysicalConverter(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
) : ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits.replace(BodoPhysicalRel.CONVENTION), input),
    BodoPhysicalRel {
    init {
        // Initialize the type to avoid errors with Kotlin suggesting to access
        // the protected field directly.
        rowType = getRowType()
    }

    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): SnowflakeToBodoPhysicalConverter = SnowflakeToBodoPhysicalConverter(cluster, traitSet, sole(inputs))

    /**
     * Even if SnowflakeToBodoPhysicalConverter is a BodoPhysicalRel, it is still
     * a single process operation which the other ranks wait on and doesn't
     * benefit from parallelism.
     */
    override fun splitCount(numRanks: Int): Int = 1

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "reading table"

    override fun loggingTitle() = "IO TIMING"

    override fun nodeDetails() =
        (
            if (isSimpleWholeTableRead()) {
                getTableName(input as SnowflakeRel)
            } else {
                getSnowflakeSQL()
            }
        )!!

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty =
        ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()
        currentPipeline.addInitialization(Op.Assign(readerVar, generateReadExpr(ctx)))
        return readerVar
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.io.arrow_reader.arrow_reader_del", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost? {
        // Determine the average row size which determines how much data is returned.
        // If this data isn't available, just assume something like 4 bytes for each column.
        val averageRowSize =
            mq.getAverageRowSize(this)
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

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            val runtimeJoinFilters = extractRuntimeJoinFilters()
            val readStage =
                OutputtingStageEmission(
                    { ctx, stateVar, _ ->
                        generateStreamingTable(ctx, stateVar)
                    },
                    reportOutTableSize = false,
                )
            val stages = mutableListOf(readStage)
            if (runtimeJoinFilters.isNotEmpty()) {
                val wrappedStage =
                    OutputtingStageEmission({ ctx, stateVar, inputTable ->
                        RuntimeJoinFilterBase.wrapResultInRuntimeJoinFilters(this, ctx, extractRuntimeJoinFilters(), inputTable!!)
                    }, reportOutTableSize = true)
                stages.add(wrappedStage)
            }
            val pipeline =
                OutputtingPipelineEmission(
                    stages,
                    true,
                    null,
                )
            val operatorEmission =
                OperatorEmission(
                    { ctx -> initStateVariable(ctx) },
                    { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                    listOf(),
                    pipeline,
                    timeStateInitialization = true,
                )
            implementor.buildStreaming(operatorEmission)!!
        } else {
            (implementor::build)(listOf()) { ctx, _ -> generateNonStreamingTable(ctx) }
        }

    /**
     * Generate the code required to read from Snowflake.
     */
    private fun generateReadExpr(ctx: BodoPhysicalRel.BuildContext): Expr.Call {
        val readExpr =
            when {
                (isSimpleWholeTableRead()) -> sqlReadTable(input as SnowflakeTableScan, ctx)
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
    private fun sqlReadTable(
        tableScan: SnowflakeTableScan,
        ctx: BodoPhysicalRel.BuildContext,
    ): Expr.Call {
        val tableName = getTableName(tableScan)
        val schemaName = getSchemaName(tableScan)
        val databaseName = getDatabaseName(tableScan)
        val relInput = input as SnowflakeRel
        val args =
            listOf(
                StringLiteral(tableName),
                relInput.generatePythonConnStr(ImmutableList.of(databaseName, schemaName)),
            )
        return Expr.Call("pd.read_sql", args, getNamedArgs(ctx, true, Expr.None, Expr.None))
    }

    private class TTZStringRelVisitor : RelVisitor() {
        // Visitor that scans for TimestampTZ -> String casts/conversions
        private inner class TTZStringDetector : RexVisitorImpl<Void?>(true) {
            override fun visitCall(call: RexCall): Void? {
                val op = call.operator
                if (op === CastingOperatorTable.TO_CHAR || op === CastingOperatorTable.TO_VARCHAR) {
                    val type = call.getOperands()[0].type
                    // if type is TimestampTZ then return true
                    if (type.sqlTypeName == SqlTypeName.TIMESTAMP_TZ) {
                        throw Util.FoundOne.NULL
                    }
                } else if (op.kind === SqlKind.CAST) {
                    // If this is a cast from TimestampTZ to char, return true
                    val fromType = call.getOperands()[0].type
                    val toType = call.getType()
                    if (fromType.sqlTypeName == SqlTypeName.TIMESTAMP_TZ && SqlTypeFamily.STRING.contains(toType)) {
                        throw Util.FoundOne.NULL
                    }
                }

                return super.visitCall(call)
            }
        }

        override fun visit(
            node: RelNode,
            ordinal: Int,
            parent: RelNode?,
        ) {
            when (node) {
                is Filter -> {
                    node.condition.accept<Void>((::TTZStringDetector)())
                }

                is Project -> {
                    for (project in node.projects) {
                        project.accept<Void>((::TTZStringDetector)())
                    }
                }
            }
            node.childrenAccept(this)
        }
    }

    // In order to read TimestampTZ columns, we set the session parameter that controls the default string format of TimestampTZ.
    // If the current set of pushed filters include a TimestampTZ->String cast, then emit a warning.
    private fun warnIfPushedFilterContainsTimestampTZCast() {
        try {
            val visitor = TTZStringRelVisitor()
            visitor.visit(input, 0, null)
        } catch (e: Util.FoundOne) {
            PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.warning(
                "Detected pushed TimestampTZ filter that casts to string. " +
                    "This can cause issues if the filter relies on a custom session value",
            )
        }
    }

    private var withoutRuntimeJoin: RelNode? = null

    /**
     * Return a copy of the RelNode subtree where runtime join filter are skipped
     */
    private fun skipRuntimeJoinFilters(): RelNode {
        if (withoutRuntimeJoin == null) {
            val runtimeJoinFilterSkipper =
                object : RelShuttleImpl() {
                    override fun visit(other: RelNode): RelNode {
                        return if (other is SnowflakeRuntimeJoinFilter) {
                            return visit(other.input)
                        } else {
                            super.visit(other)
                        }
                    }
                }
            withoutRuntimeJoin = input.accept(runtimeJoinFilterSkipper)
        }
        return withoutRuntimeJoin!!
    }

    private var extractedRtjfList: List<SnowflakeRuntimeJoinFilter>? = null

    /**
     * Returns a list of the RelNode from the subtree that are runtime join filters.
     */
    private fun extractRuntimeJoinFilters(): List<SnowflakeRuntimeJoinFilter> {
        if (extractedRtjfList == null) {
            val runtimeJoinFilterExtract =
                object : RelVisitor() {
                    val runtimeJoinFilters: MutableList<SnowflakeRuntimeJoinFilter> = mutableListOf()

                    override fun visit(
                        node: RelNode,
                        ordinal: Int,
                        parent: RelNode?,
                    ) {
                        if (node is SnowflakeRuntimeJoinFilter) {
                            runtimeJoinFilters.add(node)
                        }
                        super.visit(node, ordinal, parent)
                    }
                }
            runtimeJoinFilterExtract.go(input)
            extractedRtjfList = runtimeJoinFilterExtract.runtimeJoinFilters.toList()
        }
        return extractedRtjfList!!
    }

    private fun getSnowflakeSQL(): String {
        warnIfPushedFilterContainsTimestampTZCast()

        // Use the snowflake dialect for generating the sql string.
        val rel2sql = BodoRelToSqlConverter(BodoSnowflakeSqlDialect.DEFAULT)
        return rel2sql
            .visitRoot(skipRuntimeJoinFilters())
            .asSelect()
            .toSqlString { c ->
                c
                    .withClauseStartsLine(false)
                    .withDialect(BodoSnowflakeSqlDialect.DEFAULT)
            }.toString()
    }

    /**
     * Generate a read expression for SQL operations that will be pushed into Snowflake. The currently
     * supported operations consist of Aggregates and filters.
     */
    private fun readSql(ctx: BodoPhysicalRel.BuildContext): Expr.Call {
        // For now, pull out any Runtime Join Filters inside the converter
        // node and run them after the read. Will deal with pushing
        // them into the IO node later.
        val sql = getSnowflakeSQL()
        val relInput = (skipRuntimeJoinFilters()) as SnowflakeRel

        val passTableInfo = canUseOptimizedReadSqlPath(relInput)

        val bodoTableNameExpr =
            if (passTableInfo) {
                // Store the original indices to allow handling renaming.
                val catalogTable = relInput.getCatalogTable()
                // Get the fully qualified name.
                StringLiteral(catalogTable.getQualifiedName())
            } else {
                Expr.None
            }

        val originalIndices =
            if (passTableInfo) {
                // TODO: Replace with a global variable?
                Expr.Tuple(getOriginalColumnIndices(relInput).map { i -> Expr.IntegerLiteral(i) })
            } else {
                Expr.None
            }
        val args =
            listOf(
                StringLiteral(sql),
                // We don't use a schema name because we've already fully qualified
                // all table references, and it's better if this doesn't have any
                // potentially unexpected behavior.
                relInput.generatePythonConnStr(ImmutableList.of("", "")),
            )
        return Expr.Call("pd.read_sql", args, getNamedArgs(ctx, false, bodoTableNameExpr, originalIndices))
    }

    /**
     * Helper function to remap each column to its original index.
     * This function assumes that canUseOptimizedReadSqlPath has already
     * been called and evaluates to True.
     */
    private fun getOriginalColumnIndices(node: SnowflakeRel): List<Int> {
        val onlyColumnSubsetVisitor =
            object : RelVisitor() {
                // Initialize all columns to be in the original location.
                var originalColumns: MutableList<Int> = (0 until node.getRowType().fieldCount).toMutableList()

                override fun visit(
                    node: RelNode,
                    ordinal: Int,
                    parent: RelNode?,
                ) {
                    if (node is SnowflakeProject) {
                        for (i in 0 until originalColumns.size) {
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
                        for (i in 0 until originalColumns.size) {
                            val index = originalColumns[i]
                            originalColumns[i] = groups[index]
                        }
                        node.childrenAccept(this)
                    } else if (node is SnowflakeFilter || node is SnowflakeSort) {
                        // Enable moving past filters to get the original table
                        // for filter and limit.
                        node.childrenAccept(this)
                    } else if (node is SnowflakeTableScan) {
                        for (i in 0 until originalColumns.size) {
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
        val onlyColumnSubsetVisitor =
            object : RelVisitor() {
                var seenInvalidNode: Boolean = false

                override fun visit(
                    node: RelNode,
                    ordinal: Int,
                    parent: RelNode?,
                ) {
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
                        throw RuntimeException(
                            "canUseOptimizedReadSqlPath(): Unexpected SnowflakeRel encountered. " +
                                "Please explicitly support each SnowflakeRel",
                        )
                    }
                }
            }

        onlyColumnSubsetVisitor.go(node)
        return !onlyColumnSubsetVisitor.seenInvalidNode
    }

    private fun getNamedArgs(
        ctx: BodoPhysicalRel.BuildContext,
        isTable: Boolean,
        origTableExpr: Expr,
        origTableIndices: Expr,
    ): List<Pair<String, Expr>> =
        listOf(
            "_bodo_is_table_input" to Expr.BooleanLiteral(isTable),
            "_bodo_orig_table_name" to origTableExpr,
            "_bodo_orig_table_indices" to origTableIndices,
            "_bodo_chunksize" to getStreamingBatchArg(ctx),
            "_bodo_read_as_table" to Expr.BooleanLiteral(true),
            "_bodo_sql_op_id" to ctx.operatorID().toExpr(),
            "_bodo_runtime_join_filters" to RuntimeJoinFilterBase.getRuntimeJoinFilterTuple(ctx, extractRuntimeJoinFilters()),
        )

    /**
     * Generate the argument that will be passed for '_bodo_chunksize' in a read_sql call. If
     * we are not streaming Python expects None.
     */
    private fun getStreamingBatchArg(ctx: BodoPhysicalRel.BuildContext): Expr =
        if (isStreaming()) {
            Expr.IntegerLiteral(ctx.streamingOptions().chunkSize)
        } else {
            Expr.None
        }

    /**
     * Generate the Table for the body of the streaming code.
     */
    private fun generateStreamingTable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ): BodoEngineTable {
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
    private fun generateNonStreamingTable(ctx: BodoPhysicalRel.BuildContext): BodoEngineTable {
        val readExpr = generateReadExpr(ctx)
        return RuntimeJoinFilterBase.wrapResultInRuntimeJoinFilters(this, ctx, extractRuntimeJoinFilters(), readExpr)
    }

    /**
     * Is the Snowflake read for a whole table without any TimestampTZ columns, column pruning,
     * filtering, or other snowflake nodes.
     */
    private fun isSimpleWholeTableRead(): Boolean {
        val doesNotContainTimestampTz =
            input.rowType.fieldList.all {
                it.type.sqlTypeName != SqlTypeName.TIMESTAMP_TZ
            }
        return input is SnowflakeTableScan && !(input as SnowflakeTableScan).prunesColumns() && doesNotContainTimestampTz
    }
}
