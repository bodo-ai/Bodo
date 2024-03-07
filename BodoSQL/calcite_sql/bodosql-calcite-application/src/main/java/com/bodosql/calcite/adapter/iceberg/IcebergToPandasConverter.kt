package com.bodosql.calcite.adapter.iceberg

import com.bodosql.calcite.adapter.pandas.PandasRel
import com.bodosql.calcite.application.BodoSQLCodegenException
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
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import java.lang.RuntimeException
import java.math.BigDecimal

class IcebergToPandasConverter(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode) :
    ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits.replace(PandasRel.CONVENTION), input), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): RelNode {
        return IcebergToPandasConverter(cluster, traitSet, sole(inputs))
    }

    /**
     * Even if IcebergToPandasConverter is a PandasRel, it is still
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

    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val currentPipeline = builder.getCurrentStreamingPipeline()
        val readerVar = builder.symbolTable.genStateVar()
        currentPipeline.addInitialization(Op.Assign(readerVar, generateReadExpr(ctx)))
        return readerVar
    }

    override fun deleteStateVariable(
        ctx: PandasRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val currentPipeline = ctx.builder().getCurrentStreamingPipeline()
        val deleteState = Op.Stmt(Expr.Call("bodo.io.arrow_reader.arrow_reader_del", listOf(stateVar)))
        currentPipeline.addTermination(deleteState)
    }

    /**
     * Generate the code required to read from Iceberg.
     */
    private fun generateReadExpr(ctx: PandasRel.BuildContext): Expr.Call {
        val relInput = input as IcebergRel
        val flattenedInfo = flattenIcebergTree(relInput)
        val cols = flattenedInfo.colNames
        val filters = flattenedInfo.filters
        val tableScanNode = flattenedInfo.scan
        val limit = flattenedInfo.limit
        val columnsArg = Expr.List(cols.map { v -> StringLiteral(v) })
        val typeSystem =
            ctx.builder().typeSystem
                ?: throw BodoSQLCodegenException("Iceberg Codegen requires a typeSystem in Module.Builder")
        val filterVisitor = IcebergFilterVisitor(tableScanNode, typeSystem)
        val filtersArg =
            if (filters.isEmpty()) {
                Expr.None
            } else {
                Expr.List(
                    listOf(
                        Expr.List(
                            filters.map {
                                    f ->
                                f.accept(filterVisitor)
                            }.flatten().map { (a, b, c) -> Expr.Tuple(StringLiteral(a), StringLiteral(b), c) },
                        ),
                    ),
                )
            }

        val schemaPath = getSchemaPath(relInput)

        val args =
            listOf(
                StringLiteral(getTableName(relInput)),
                // TODO: Replace with an implementation for the IcebergCatalog
                StringLiteral("iceberg+" + relInput.generatePythonConnStr(schemaPath)),
                StringLiteral(schemaPath.joinToString(separator = ".")),
            )
        val namedArgs =
            listOf(
                "_bodo_chunksize" to getStreamingBatchArg(ctx),
                "_bodo_read_as_table" to Expr.BooleanLiteral(true),
                "_bodo_columns" to columnsArg,
                "_bodo_filter" to filtersArg,
                "_bodo_limit" to limit,
            )

        return Expr.Call("pd.read_sql_table", args, namedArgs)
    }

    private data class FlattenedIcebergInfo(
        val colNames: List<String>,
        val filters: List<RexNode>,
        val scan: IcebergTableScan,
        val limit: Expr,
    )

    private fun flattenIcebergTree(node: IcebergRel): FlattenedIcebergInfo {
        val visitor =
            object : RelVisitor() {
                // Initialize all columns to be in the original location.
                var colMap: MutableList<Int> = (0..<node.getRowType().fieldCount).toMutableList()
                var filters: MutableList<RexNode> = mutableListOf()
                var baseScan: IcebergTableScan? = null
                var limit: Expr = Expr.None

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
                            limit = Expr.DecimalLiteral((node.fetch!! as RexLiteral).getValueAs(BigDecimal::class.java)!!)
                            node.childrenAccept(this)
                        }
                        is IcebergProject -> {
                            for (i in 0..<colMap.size) {
                                val project = node.projects[colMap[i]]
                                if (project !is RexInputRef) {
                                    throw RuntimeException("getOriginalColumnIndices() requires only InputRefs")
                                }
                            }
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
        val baseScan = visitor.baseScan!!
        val origColNames = baseScan.deriveRowType().fieldNames
        return FlattenedIcebergInfo(
            visitor.colMap.mapIndexed { _, v -> origColNames[v] }.toList(),
            visitor.filters,
            baseScan,
            visitor.limit,
        )
    }

    private fun getTableName(input: IcebergRel) = input.getCatalogTable().name

    private fun getSchemaPath(input: IcebergRel) = input.getCatalogTable().parentFullPath

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
    private fun generateStreamingTable(
        ctx: PandasRel.BuildContext,
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
    private fun generateNonStreamingTable(ctx: PandasRel.BuildContext): BodoEngineTable {
        val readExpr = generateReadExpr(ctx)
        return ctx.returns(readExpr)
    }
}
