package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.operatorTables.AggOperatorTable.BITAND_AGG
import com.bodosql.calcite.application.operatorTables.AggOperatorTable.BITOR_AGG
import com.bodosql.calcite.application.operatorTables.AggOperatorTable.BITXOR_AGG
import com.bodosql.calcite.application.operatorTables.AggOperatorTable.BOOLAND_AGG
import com.bodosql.calcite.application.operatorTables.AggOperatorTable.BOOLOR_AGG
import com.bodosql.calcite.application.operatorTables.AggOperatorTable.COUNT_IF
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.codeGeneration.TerminatingPipelineEmission
import com.bodosql.calcite.codeGeneration.TerminatingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.OperatorType
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.StreamingPipelineFrame
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.core.WindowBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.BatchingPropertyTraitDef
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelFieldCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelVisitor
import org.apache.calcite.rel.core.Window
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.ImmutableBitSet
import org.apache.calcite.util.Util
import kotlin.math.ceil

class BodoPhysicalWindow(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    hints: List<RelHint>,
    input: RelNode,
    constants: List<RexLiteral>,
    rowType: RelDataType,
    groups: List<Group>,
    inputsToKeep: ImmutableBitSet,
) :
    WindowBase(
            cluster,
            traitSet.replace(BodoPhysicalRel.CONVENTION),
            hints,
            input,
            constants,
            rowType,
            groups,
            inputsToKeep,
        ),
        BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): BodoPhysicalWindow {
        return BodoPhysicalWindow(cluster, traitSet, hints, inputs[0], constants, rowType, groups, inputsToKeep)
    }

    override fun copy(constants: MutableList<RexLiteral>): BodoPhysicalWindow {
        return BodoPhysicalWindow(cluster, traitSet, hints, input, constants, rowType, groups, inputsToKeep)
    }

    /**
     * Mapping of sql window operators to bodo window func names
     * Should include all functions where the sql name is not equal to the bodo name.
     */
    private val sqlToBodoWindowFuncName =
        mapOf(
            "AVG" to "mean",
        )

    /**
     * Maps operator name to a Boolean list that has length equal to the number of arguments. A "true" value at index
     * `i` indicates that the `i`th index is a scalar, otherwise it is a column. Everything not included in this
     * mapping is assumed to have no inputs (e.g. rank related functions, size)
     */
    val argsMap =
        mapOf(
            "NTILE" to listOf(true),
            "SUM" to listOf(false),
            "AVG" to listOf(false),
            "MAX" to listOf(false),
            "MIN" to listOf(false),
            "COUNT" to listOf(false),
            "COUNT_IF" to listOf(false),
            "BOOLOR_AGG" to listOf(false),
            "BOOLAND_AGG" to listOf(false),
            "BITAND_AGG" to listOf(false),
            "BITOR_AGG" to listOf(false),
            "BITXOR_AGG" to listOf(false),
        )

    /**
     *  Returns whether the Window node can be converted to streaming window codegen. If not, the Window node must
     *  be converted back into a Project node. Currently has the following requirements:
     *
     *  <ul>
     *      <li>Streaming must be enabled</li>
     *      <li>The window must only contain a single cohort</li>
     *      <li>All window aggfunc calls are from the limited list of support</li>
     *      <li>There is a single group with a single aggfunc</li>
     *  </ul>
     */
    fun supportsStreamingWindow(): Boolean {
        return traitSet.isEnabled(BatchingPropertyTraitDef.INSTANCE) && groups.size == 1 &&
            groups.all {
                    group ->
                isSupportedHashGroup(group) || isSupportedSortGroup(group) || isSupportedBlankWindowGroup(group)
            }
    }

    /**
     * Returns whether a window cohort is supported for the
     * Bodo execution codepath that either hashes on the partition.
     * Requires all aggfuncs in the group to be supported in the hash codepath,
     * and for there to be at least one partition column.
     */
    fun isSupportedHashGroup(group: Window.Group): Boolean {
        return group.keys.cardinality() >= 1 &&
            group.aggCalls.size == 1 &&
            group.aggCalls.all {
                    aggCall ->
                !aggCall.distinct && !aggCall.ignoreNulls
                when (aggCall.operator.kind) {
                    SqlKind.ROW_NUMBER,
                    SqlKind.RANK,
                    SqlKind.DENSE_RANK,
                    SqlKind.PERCENT_RANK,
                    SqlKind.CUME_DIST,
                    SqlKind.NTILE,
                    -> true
                    else -> false
                }
            }
    }

    // The list of types that are allowed in the streaming sort-based implementation of
    // selection functions (min, max, etc.)
    private val supportedSelectionTypes =
        listOf(
            SqlTypeName.BOOLEAN,
            SqlTypeName.TINYINT,
            SqlTypeName.SMALLINT,
            SqlTypeName.INTEGER,
            SqlTypeName.BIGINT,
            SqlTypeName.DECIMAL,
            SqlTypeName.FLOAT,
            SqlTypeName.REAL,
            SqlTypeName.DOUBLE,
            SqlTypeName.DATE,
            SqlTypeName.TIME,
            SqlTypeName.TIMESTAMP,
            SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE,
        )

    /**
     * Returns whether a window cohort is supported for the
     * Bodo execution codepath that sorts on the partition+orderby columns.
     * Requires all aggfuncs in the group to be supported in the sort codepath,
     * and for there to be at least one partition or orderby column.
     */
    fun isSupportedSortGroup(group: Window.Group): Boolean {
        return (group.keys.cardinality() >= 1 || group.orderKeys.keys.isNotEmpty()) &&
            group.aggCalls.size == 1 &&
            group.aggCalls.all {
                    aggCall ->
                !aggCall.distinct && !aggCall.ignoreNulls &&
                    when (aggCall.operator.kind) {
                        SqlKind.ROW_NUMBER,
                        SqlKind.RANK,
                        SqlKind.DENSE_RANK,
                        SqlKind.PERCENT_RANK,
                        SqlKind.CUME_DIST,
                        -> true
                        SqlKind.COUNTIF,
                        SqlKind.SUM,
                        SqlKind.COUNT,
                        -> group.lowerBound.isUnbounded && group.upperBound.isUnbounded
                        SqlKind.MIN,
                        SqlKind.MAX,
                        ->
                            group.lowerBound.isUnbounded &&
                                group.upperBound.isUnbounded &&
                                supportedSelectionTypes.contains(aggCall.operands[0].type.sqlTypeName)
                        SqlKind.BIT_AND, SqlKind.BIT_OR, SqlKind.BIT_XOR,
                        SqlKind.OTHER, SqlKind.OTHER_FUNCTION,
                        ->
                            when (aggCall.operator.name) {
                                COUNT_IF.name,
                                BOOLAND_AGG.name,
                                BOOLOR_AGG.name,
                                BITAND_AGG.name,
                                BITOR_AGG.name,
                                BITXOR_AGG.name,
                                COUNT_IF.name,
                                -> true
                                else -> false
                            }
                        else -> false
                    }
            }
    }

    /**
     * Returns whether a window cohort is supported for the
     * Bodo execution codepath without partition or order
     * columns.
     */
    fun isSupportedBlankWindowGroup(group: Group): Boolean {
        return group.keys.cardinality() == 0 &&
            group.orderKeys.keys.size == 0 &&
            group.aggCalls.size == 1 &&
            group.aggCalls.all {
                    aggCall ->
                !aggCall.distinct && !aggCall.ignoreNulls
                when (aggCall.operator.kind) {
                    SqlKind.MIN,
                    SqlKind.MAX,
                    SqlKind.SUM,
                    SqlKind.AVG,
                    SqlKind.COUNT,
                    -> true
                    else -> false
                }
            }
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        // The first pipeline accumulates all the rows
        val stage =
            TerminatingStageEmission { ctx, stateVar, table ->
                val inputVar = table!!
                val builder = ctx.builder()
                val pipeline: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
                val batchExitCond = pipeline.getExitCond()
                val consumeCall =
                    Expr.Call(
                        "bodo.libs.stream_window.window_build_consume_batch",
                        listOf(
                            stateVar,
                            inputVar,
                            batchExitCond,
                        ),
                    )
                val newExitCond: Variable = builder.symbolTable.genFinishedStreamingFlag()
                val inputRequest: Variable = builder.symbolTable.genInputRequestVar()
                val exitAssign = Op.TupleAssign(listOf(newExitCond, inputRequest), consumeCall)
                builder.add(exitAssign)
                pipeline.endSection(newExitCond)
                pipeline.addInputRequest(inputRequest)
                builder.forceEndOperatorAtCurPipeline(ctx.operatorID(), pipeline)
                null
            }
        val terminatingPipeline = TerminatingPipelineEmission(listOf(), stage, false, input)
        // The final pipeline generates the output from the state.
        val outputStage =
            OutputtingStageEmission(
                {
                        ctx, stateVar, _ ->
                    val builder = ctx.builder()
                    val pipeline = builder.getCurrentStreamingPipeline()
                    val outputControl: Variable = builder.symbolTable.genOutputControlVar()
                    pipeline.addOutputControl(outputControl)
                    val outputCall =
                        Expr.Call(
                            "bodo.libs.stream_window.window_produce_output_batch",
                            listOf(stateVar, outputControl),
                        )
                    val outTable: Variable = builder.symbolTable.genTableVar()
                    val finishedFlag = pipeline.getExitCond()
                    val outputAssign = Op.TupleAssign(listOf(outTable, finishedFlag), outputCall)
                    builder.add(outputAssign)
                    ctx.returns(outTable)
                },
                reportOutTableSize = true,
            )
        val outputPipeline = OutputtingPipelineEmission(listOf(outputStage), true, null)
        val operatorEmission =
            OperatorEmission(
                { ctx -> initStateVariable(ctx) },
                { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                listOf(terminatingPipeline),
                outputPipeline,
                timeStateInitialization = false,
            )
        return implementor.buildStreaming(operatorEmission)!!
    }

    /**
     * If non-streaming, use the regular path from projection-style window functions.
     */
    private fun emitSingleBatch(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        return convertToProject().emit(implementor)
    }

    /**
     * Iteratively traverses upward through the subtree starting at
     * the current RelNode to detect whether there is a nested loop
     * join in the same pipeline as the window node (and the window
     * is on the probe side).
     */
    private fun pipelineAncestorNestedJoin(parentMapping: Map<RelNode, List<RelNode>>): Boolean {
        var currentNode: RelNode = this
        while (true) {
            // Stop if we have hit the top of the main tree
            // or a cached subtree (0 ancestors, or multiple)
            val parents = parentMapping[currentNode]
            if (parents == null || parents.size > 1) {
                return false
            }
            val currentParent = parents[0]
            when (currentParent) {
                is BodoPhysicalJoin -> {
                    // If we hit a join, return true if it is a nested join
                    // & we are on the left side (probe), and false otherwise
                    return currentParent.analyzeCondition().leftKeys.isEmpty() && currentParent.getInput(0) == currentNode
                }
                is BodoPhysicalFilter,
                is BodoPhysicalRuntimeJoinFilter,
                is BodoPhysicalProject,
                is BodoPhysicalFlatten,
                -> {
                    // On pass-through nodes, keep searching upward
                    // through the parent nodes.
                    currentNode = currentParent
                }
                else -> {
                    // On all other nodes, return false since they
                    // are pipeline breakers.
                    return false
                }
            }
        }
    }

    /**
     * Recursively traverses downward through the subtree starting at
     * the current RelNode to detect whether there is a nested loop
     * join in the same pipeline as the window node.
     */
    private fun pipelineDescendantNestedJoin(): Boolean {
        return try {
            object : RelVisitor() {
                override fun visit(
                    node: RelNode,
                    ordinal: Int,
                    parent: RelNode?,
                ) {
                    when (node) {
                        is BodoPhysicalJoin -> {
                            // If we hit a join, throw the indicator exception
                            // but only if it is a nested loop join.
                            if (node.analyzeCondition().leftKeys.isEmpty()) {
                                throw Util.FoundOne.NULL
                            }
                        }
                        is BodoPhysicalFilter,
                        is BodoPhysicalRuntimeJoinFilter,
                        is BodoPhysicalProject,
                        is BodoPhysicalFlatten,
                        -> {
                            // On pass-through nodes, keep recursively
                            // searching through the child nodesz
                            super.visit(node, ordinal, parent)
                        }
                    }
                    // For all other nodes, we stop recursing since
                    // we have hit the end of the pipeline without finding
                    // a nested loop join
                }
            }.go(this.getInput())
            false
        } catch (e: Util.FoundOne) {
            true
        }
    }

    /**
     * Helper for getting the bodo name from an AggCall.
     */
    private fun getOperatorName(aggCall: Window.RexWinAggCall): String {
        val sqlName = aggCall.operator.name
        // COUNT is a special case where if it has zero arguments it is actually "size"
        if (sqlName == "COUNT" && aggCall.operands.size == 0) {
            return "size"
        } else if (sqlName in sqlToBodoWindowFuncName) {
            return sqlToBodoWindowFuncName[sqlName]!!
        }
        return sqlName.lowercase()
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val builder = ctx.builder()
        val stateVar = builder.symbolTable.genStateVar()
        val partitionKeys: MutableList<Expr> = mutableListOf()
        val orderKeys: MutableList<Expr> = mutableListOf()
        val orderAsc: MutableList<Expr> = mutableListOf()
        val orderNullPos: MutableList<Expr> = mutableListOf()
        val funcIndices: MutableList<Expr> = mutableListOf()
        val funcNames: MutableList<Expr> = mutableListOf()
        val funcScalarArgs: MutableList<Expr> = mutableListOf()
        assert(this.groups.size == 1)
        val group = this.groups[0]
        group.keys.forEach {
            partitionKeys.add(Expr.IntegerLiteral(it))
        }
        group.orderKeys.fieldCollations.forEach {
            orderKeys.add(Expr.IntegerLiteral(it.fieldIndex))
            orderAsc.add(Expr.BooleanLiteral(!it.direction.isDescending))
            orderNullPos.add(Expr.BooleanLiteral(it.nullDirection == RelFieldCollation.NullDirection.LAST))
        }
        group.aggCalls.forEach {
                aggCall ->
            val operatorName = getOperatorName(aggCall)
            funcNames.add(Expr.StringLiteral(operatorName))
            val funcColumnArgs: MutableList<Expr> = mutableListOf()
            val scalarArgs: MutableList<Expr> = mutableListOf()

            // For each idx'd operand lookup argsMap[funcName][idx]
            // If it is True -> then it is a scalar arg look up the constant in this.constants[op - numCols]
            // append to scalar args. Otherwise, the input refering to a column, add the input ref to columnArgs
            var constantIdx = 0 // index into this.constants
            val argsList = argsMap[aggCall.operator.name] ?: listOf()
            if (aggCall.operands.size < argsList.size && aggCall.kind != SqlKind.COUNT) {
                throw BodoSQLCodegenException("Too few arguments for window function: $aggCall (expected {${argsList.size}})")
            }
            aggCall.operands.forEachIndexed {
                    it, op ->
                if (it >= argsList.size) {
                    throw BodoSQLCodegenException("Too many arguments for window function: $aggCall (expected at most {${argsList.size}})")
                }
                val isScalar = argsList[it]
                if (isScalar) {
                    val constantArg = this.constants[constantIdx]
                    scalarArgs.add(ctx.rexTranslator(null).visitLiteral(constantArg))
                    constantIdx++
                } else if (op is RexInputRef) {
                    funcColumnArgs.add(Expr.IntegerLiteral(op.index))
                } else {
                    // Defensive check since op should always be a RexInputRef by the
                    // design of the Window node.
                    throw Exception("BodoPhysicalWindowNode: unsupported input to window function '$op'")
                }
            }
            funcIndices.add(Expr.Tuple(funcColumnArgs))
            funcScalarArgs.add(Expr.Tuple(scalarArgs))
        }

        val keptInputsArray = inputsToKeep.toList().map { Expr.IntegerLiteral(it) }
        val partitionGlobal = ctx.lowerAsMetaType(Expr.Tuple(partitionKeys))
        val orderGlobal = ctx.lowerAsMetaType(Expr.Tuple(orderKeys))
        val ascGlobal = ctx.lowerAsMetaType(Expr.Tuple(orderAsc))
        val nullPosGlobal = ctx.lowerAsMetaType(Expr.Tuple(orderNullPos))
        val funcNamesGlobal = ctx.lowerAsMetaType(Expr.Tuple(funcNames))
        val keptInputsGlobal = ctx.lowerAsMetaType(Expr.Tuple(keptInputsArray))
        val funcIndicesGlobal = ctx.lowerAsMetaType(Expr.Tuple(funcIndices))
        val funcScalarArgsGlobal = ctx.lowerAsMetaType(Expr.Tuple(funcScalarArgs))
        val allowWorkStealing =
            !(
                pipelineAncestorNestedJoin(ctx.fetchParentMappings()) ||
                    pipelineDescendantNestedJoin()
            )
        val stateCall =
            Expr.Call(
                "bodo.libs.stream_window.init_window_state",
                listOf(
                    ctx.operatorID().toExpr(),
                    partitionGlobal,
                    orderGlobal,
                    ascGlobal,
                    nullPosGlobal,
                    funcNamesGlobal,
                    funcIndicesGlobal,
                    keptInputsGlobal,
                    Expr.BooleanLiteral(allowWorkStealing),
                    Expr.IntegerLiteral(input.rowType.fieldCount),
                    funcScalarArgsGlobal,
                ),
            )
        val windowInit = Op.Assign(stateVar, stateCall)
        val initialPipeline = builder.getCurrentStreamingPipeline()
        val mq: RelMetadataQuery = cluster.metadataQuery
        initialPipeline.initializeStreamingState(
            ctx.operatorID(),
            windowInit,
            OperatorType.WINDOW,
            estimateBuildMemory(mq),
        )
        return stateVar
    }

    /**
     * Get window function build memory estimate for memory budget comptroller.
     * TODO: when refactoring the cost model, re-use the code elsewhere where possible.
     */
    private fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        // Window has the same high level memory consumption logic as aggregation
        val distinctRows = mq.getRowCount(this)
        val averageBuildRowSize = mq.getAverageRowSize(this) ?: 8.0
        return ceil(distinctRows * averageBuildRowSize).toInt()
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val builder = ctx.builder()
        val finalPipeline = builder.getCurrentStreamingPipeline()
        val deleteState =
            Op.Stmt(Expr.Call("bodo.libs.stream_window.delete_window_state", listOf(stateVar)))
        finalPipeline.addTermination(deleteState)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.streamingIfPossibleProperty(rowType)
    }

    /**
     * Converts a BodoPhysicalWindow back to a BodoPhysicalProject for codegen purposes.
     */
    fun convertToProject(): BodoPhysicalProject {
        val projTerms = convertToProjExprs()
        return BodoPhysicalProject.create(input, projTerms, this.rowType.fieldNames)
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            hints: List<RelHint>,
            input: RelNode,
            constants: List<RexLiteral>,
            rowType: RelDataType,
            groups: List<Group>,
            inputsToKeep: ImmutableBitSet,
        ): BodoPhysicalWindow {
            return BodoPhysicalWindow(cluster, input.traitSet, hints, input, constants, rowType, groups, inputsToKeep)
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            hints: List<RelHint>,
            input: RelNode,
            constants: List<RexLiteral>,
            rowType: RelDataType,
            groups: List<Group>,
        ): BodoPhysicalWindow {
            return create(cluster, hints, input, constants, rowType, groups, ImmutableBitSet.range(input.rowType.fieldCount))
        }
    }
}
