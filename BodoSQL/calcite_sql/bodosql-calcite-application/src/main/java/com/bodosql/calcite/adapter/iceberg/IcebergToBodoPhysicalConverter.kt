package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.timers.SingleBatchRelNodeTimer
import com.bodosql.calcite.catalog.SnowflakeCatalog
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
import com.bodosql.calcite.rel.metadata.BodoRelMetadataQuery
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.ConventionTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.convert.ConverterImpl
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.type.SqlTypeFamily
import java.lang.RuntimeException
import java.math.BigDecimal

class IcebergToBodoPhysicalConverter(
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
    ): RelNode = IcebergToBodoPhysicalConverter(cluster, traitSet, sole(inputs))

    /**
     * Even if IcebergToBodoPhysicalConverter is a BodoPhysicalRel, it is still
     * a single process operation which the other ranks wait on and doesn't
     * benefit from parallelism.
     */
    override fun splitCount(numRanks: Int): Int = 1

    override fun getTimerType() = SingleBatchRelNodeTimer.OperationType.IO_BATCH

    override fun operationDescriptor() = "reading table"

    override fun loggingTitle() = "ICEBERG TIMING"

    // TODO: What to do with this function?
    override fun nodeDetails(): String = getTableName(input as IcebergRel)

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        // TODO: Can simplify now?
        return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
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

    // ----------------------------------- Codegen Helpers -----------------------------------
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            val runtimeJoinFilters: List<IcebergRuntimeJoinFilter> = flattenIcebergTree().runtimeJoinFilters
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
                        RuntimeJoinFilterBase.wrapResultInRuntimeJoinFilters(
                            this,
                            ctx,
                            flattenIcebergTree().runtimeJoinFilters,
                            inputTable!!,
                        )
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

    /**
     * Generate the code required to read from Iceberg.
     */
    private fun generateReadExpr(ctx: BodoPhysicalRel.BuildContext): Expr.Call {
        val relInput = input as IcebergRel
        val flattenedInfo = flattenIcebergTree()
        val cols = flattenedInfo.colNames
        val filters = flattenedInfo.filters
        val tableScanNode = flattenedInfo.scan
        val limit = flattenedInfo.limit

        // Store table name and base connection string in builder for prefetch gen
        // Note: Only when reading from a Snowflake-managed Iceberg table (using Snowflake catalog)
        if (ctx.streamingOptions().prefetchSFIceberg) {
            if (tableScanNode.getCatalogTable().getCatalog() is SnowflakeCatalog) {
                var staticConExpr = relInput.generatePythonConnStr(ImmutableList.of("", ""))
                staticConExpr =
                    if (staticConExpr is StringLiteral) StringLiteral("iceberg+${staticConExpr.arg}") else staticConExpr
                ctx.builder().addSfIcebergTablePath(staticConExpr, tableScanNode.getCatalogTable().getQualifiedName())
            }
        }

        val columnsArg = Expr.List(cols.map { v -> StringLiteral(v) })
        val filtersArg =
            if (filters.isEmpty()) {
                IcebergFilterVisitor.default()
            } else {
                val filterVisitor = IcebergFilterVisitor(tableScanNode, ctx)
                val pieces = filters.map { f -> f.accept(filterVisitor) }
                if (pieces.size == 1) {
                    pieces[0]
                } else {
                    IcebergFilterVisitor.op("AND", pieces)
                }
            }

        val schemaPath = getSchemaPath(relInput)
        var conExpr = relInput.generatePythonConnStr(schemaPath)
        conExpr = if (conExpr is StringLiteral) StringLiteral("iceberg+" + conExpr.arg) else conExpr

        val args =
            listOf(
                StringLiteral(getTableName(relInput)),
                // TODO: Replace with an implementation for the IcebergCatalog
                conExpr,
                StringLiteral(schemaPath.joinToString(separator = ".")),
            )
        // Add extra keyword argument specifying which columns should be dictionary encoded
        // based on their NDV values. If a string column is not in this list, then Bodo
        // will do an approx NDV check at compile time to determine whether it should be
        // dictionary encoded.
        val dictArgs: MutableList<String> = mutableListOf()
        val rowCount = cluster.metadataQuery.getRowCount(relInput)
        relInput.getRowType().fieldList.mapIndexed { idx, field ->
            if (getRowType().fieldList[idx].type.family == SqlTypeFamily.CHARACTER) {
                val distinctRowCount =
                    (cluster.metadataQuery as BodoRelMetadataQuery).getColumnDistinctCount(relInput, idx)
                // A column is added if it is a string column whose distinct count is less than
                // a certain ratio of the total row count, and is also less than the batch size.
                if (rowCount != null &&
                    distinctRowCount != null &&
                    distinctRowCount / rowCount <= RelationalAlgebraGenerator.READ_DICT_THRESHOLD &&
                    distinctRowCount < RelationalAlgebraGenerator.streamingBatchSize
                ) {
                    dictArgs.add(field.name)
                }
            }
        }
        val dictExpr =
            if (dictArgs.isEmpty()) {
                Expr.None
            } else {
                Expr.List(dictArgs.map { Expr.StringLiteral(it) })
            }
        val namedArgs =
            listOf(
                "_bodo_chunksize" to getStreamingBatchArg(ctx),
                "_bodo_read_as_table" to Expr.BooleanLiteral(true),
                "_bodo_columns" to columnsArg,
                "_bodo_filter" to filtersArg,
                "_bodo_limit" to limit,
                "_bodo_sql_op_id" to ctx.operatorID().toExpr(),
                "_bodo_read_as_dict" to dictExpr,
                "_bodo_runtime_join_filters" to
                    RuntimeJoinFilterBase.getRuntimeJoinFilterTuple(
                        ctx,
                        flattenIcebergTree().runtimeJoinFilters,
                    ),
            )

        return Expr.Call("pd.read_sql_table", args, namedArgs)
    }

    private data class FlattenedIcebergInfo(
        val colNames: List<String>,
        val filters: List<RexNode>,
        val scan: IcebergTableScan,
        val limit: Expr,
        val runtimeJoinFilters: List<IcebergRuntimeJoinFilter>,
    )

    private var cachedIcebergFlattenedInfo: FlattenedIcebergInfo? = null

    private fun flattenIcebergTree(): FlattenedIcebergInfo {
        if (cachedIcebergFlattenedInfo == null) {
            val node = input as IcebergRel
            val visitor =
                object : RelVisitor() {
                    // Initialize all columns to be in the original location.
                    var colMap: MutableList<Int> = (0 until node.rowType.fieldCount).toMutableList()
                    var filters: MutableList<RexNode> = mutableListOf()
                    var runtimeJoinFilters: MutableList<IcebergRuntimeJoinFilter> = mutableListOf()
                    var baseScan: IcebergTableScan? = null
                    var limit: BigDecimal? = null

                    override fun visit(
                        node: RelNode,
                        ordinal: Int,
                        parent: RelNode?,
                    ) {
                        when (node) {
                            // Enable moving past filters to get the original table
                            is IcebergFilter -> {
                                filters.add(node.condition)
                                node.childrenAccept(this)
                            }

                            is IcebergSort -> {
                                val nodeVal: BigDecimal =
                                    (node.fetch!! as RexLiteral).getValueAs(BigDecimal::class.java)!!
                                if (this.limit == null) {
                                    this.limit = nodeVal
                                } else {
                                    this.limit =
                                        minOf(
                                            this.limit!!,
                                            nodeVal,
                                        )
                                }
                                node.childrenAccept(this)
                            }

                            is IcebergProject -> {
                                val newColMap = mutableListOf<Int>()
                                // Projects may reorder columns, so we need to update the column mapping.
                                for (i in 0 until colMap.size) {
                                    val project = node.projects[colMap[i]]
                                    if (project !is RexInputRef) {
                                        throw RuntimeException("getOriginalColumnIndices() requires only InputRefs")
                                    }
                                    newColMap.add(project.index)
                                }
                                colMap = newColMap
                                node.childrenAccept(this)
                            }

                            is IcebergRuntimeJoinFilter -> {
                                runtimeJoinFilters.add(node)
                                node.childrenAccept(this)
                            }

                            is IcebergTableScan -> {
                                baseScan = node
                                for (value in colMap) {
                                    if (value >= node.deriveRowType().fieldNames.size) {
                                        throw RuntimeException("IcebergProjection Invalid")
                                    }
                                }
                            }
                        }
                    }
                }
            visitor.go(node)
            val actualLimit =
                if (visitor.limit == null) {
                    Expr.None
                } else {
                    Expr.DecimalLiteral(visitor.limit!!)
                }
            val baseScan = visitor.baseScan!!
            val origColNames = baseScan.deriveRowType().fieldNames
            val runtimeJoinFilters = visitor.runtimeJoinFilters
            cachedIcebergFlattenedInfo =
                FlattenedIcebergInfo(
                    visitor.colMap.mapIndexed { _, v -> origColNames[v] }.toList(),
                    visitor.filters,
                    baseScan,
                    actualLimit,
                    runtimeJoinFilters,
                )
        }
        return cachedIcebergFlattenedInfo!!
    }

    private fun getTableName(input: IcebergRel) = input.getCatalogTable().name

    private fun getSchemaPath(input: IcebergRel) = input.getCatalogTable().parentFullPath

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
        return RuntimeJoinFilterBase.wrapResultInRuntimeJoinFilters(this, ctx, flattenIcebergTree().runtimeJoinFilters, readExpr)
    }
}
