package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen
import com.bodosql.calcite.application.BodoSQLCodegenException
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.codeGeneration.TerminatingPipelineEmission
import com.bodosql.calcite.codeGeneration.TerminatingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op
import com.bodosql.calcite.ir.Op.Assign
import com.bodosql.calcite.ir.Op.Stmt
import com.bodosql.calcite.ir.OperatorType
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.StreamingPipelineFrame
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.core.SortBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelCollation
import org.apache.calcite.rel.RelFieldCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.type.SqlTypeName
import java.util.Locale

class BodoPhysicalSort(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    collation: RelCollation,
    offset: RexNode?,
    fetch: RexNode?,
) : SortBase(cluster, traitSet.replace(BodoPhysicalRel.CONVENTION), input, collation, offset, fetch),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        collation: RelCollation,
        offset: RexNode?,
        fetch: RexNode?,
    ): BodoPhysicalSort = BodoPhysicalSort(cluster, traitSet, input, collation, offset, fetch)

    private fun getLimitAndOffsetStrs(
        ctx: BodoPhysicalRel.BuildContext,
        inTable: BodoEngineTable?,
    ): Pair<String, String> {
        var limitStr = ""
        var offsetStr = ""

        val translator = ctx.rexTranslator(inTable)

        val fetchNode = fetch
        if (fetchNode != null) {
            // Check type for fetch. If its not an integer it shouldn't be a legal limit.
            // This is handled by the parser for all situations except namedParams
            // TODO: Determine how to move this into Calcite
            val typeName = fetchNode.type.sqlTypeName
            if (typeName != SqlTypeName.TINYINT &&
                typeName != SqlTypeName.SMALLINT &&
                typeName != SqlTypeName.INTEGER &&
                typeName != SqlTypeName.BIGINT
            ) {
                throw BodoSQLCodegenException(
                    String.format(Locale.ROOT, "Limit value must be an integer, value is of type: %s", fetchNode.type.toString()),
                )
            }

            // fetch is either a named Parameter or a literal from parsing.
            // We visit the node to resolve the name.
            limitStr = fetchNode.accept(translator).emit()
        }

        val offsetNode = offset
        if (offsetNode != null) {
            // Check type for fetch. If its not an integer it shouldn't be a legal offset.
            // This is handled by the parser for all situations except namedParams
            // TODO: Determine how to move this into Calcite
            val typeName = offsetNode.type.sqlTypeName
            if (typeName != SqlTypeName.TINYINT &&
                typeName != SqlTypeName.SMALLINT &&
                typeName != SqlTypeName.INTEGER &&
                typeName != SqlTypeName.BIGINT
            ) {
                throw BodoSQLCodegenException(
                    String.format(Locale.ROOT, "Offset value must be an integer, value is of type: %s", offsetNode.type.toString()),
                )
            }

            // Offset is either a named Parameter or a literal from parsing.
            // We visit the node to resolve the name.
            offsetStr = offsetNode.accept(translator).emit()
        }

        return Pair(limitStr, offsetStr)
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable =
        if (isStreaming()) {
            // The first pipeline accumulates all the rows
            val stage =
                TerminatingStageEmission { ctx, stateVar, table ->
                    val inputVar = table!!
                    val builder = ctx.builder()
                    val pipeline: StreamingPipelineFrame = builder.getCurrentStreamingPipeline()
                    val batchExitCond = pipeline.getExitCond()
                    val consumeCall =
                        Expr.Call(
                            "bodo.libs.streaming.sort.sort_build_consume_batch",
                            listOf(
                                stateVar,
                                inputVar,
                                batchExitCond,
                            ),
                        )
                    val newExitCond: Variable = builder.symbolTable.genFinishedStreamingFlag()
                    val exitAssign = Assign(newExitCond, consumeCall)
                    builder.add(exitAssign)
                    pipeline.endSection(newExitCond)
                    builder.forceEndOperatorAtCurPipeline(ctx.operatorID(), pipeline)
                    null
                }
            val terminatingPipeline = TerminatingPipelineEmission(listOf(), stage, false, input)
            // The final pipeline generates the output from the state.
            val outputStage =
                OutputtingStageEmission(
                    { ctx, stateVar, _ ->
                        val builder = ctx.builder()
                        val pipeline = builder.getCurrentStreamingPipeline()
                        val outputControl: Variable = builder.symbolTable.genOutputControlVar()
                        pipeline.addOutputControl(outputControl)
                        // TODO supply limit/offset to produce_output_batch and handle LIMIT in streaming
                        val outputCall =
                            Expr.Call(
                                "bodo.libs.streaming.sort.produce_output_batch",
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

            implementor.buildStreaming(operatorEmission)!!
        } else {
            (implementor::build)(listOf(this.input)) { ctx, inputs ->
                val inTable = inputs[0]
                val colNames = input.rowType.fieldNames

                // handle case for queries with "ORDER BY" clause
                val sortOrders: kotlin.collections.List<RelFieldCollation> = this.getCollation().fieldCollations

                // handle case for queries with "LIMIT" clause
                val limitAndOffsetStrs = getLimitAndOffsetStrs(ctx, inTable)
                val limitStr = limitAndOffsetStrs.first
                val offsetStr = limitAndOffsetStrs.second

                val inVar = ctx.convertTableToDf(inTable)
                val sortExpr = SortCodeGen.generateSortCode(inVar, colNames, sortOrders, limitStr, offsetStr)
                val sortVar: Variable = ctx.builder().symbolTable.genDfVar()
                ctx.builder().add(Assign(sortVar, sortExpr))

                ctx.convertDfToTable(sortVar, this)
            }
        }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        val colNames = input.rowType.fieldNames
        val sortOrders: kotlin.collections.List<RelFieldCollation> = this.getCollation().fieldCollations
        val (byList, ascendingList, naPositionList) = SortCodeGen.generateSortParameters(colNames, sortOrders)
        val batchPipeline: StreamingPipelineFrame = ctx.builder().getCurrentStreamingPipeline()
        val limitAndOffsetStrs = getLimitAndOffsetStrs(ctx, null)
        val limitStr = limitAndOffsetStrs.first
        val offsetStr = limitAndOffsetStrs.second
        val typeSystem = cluster.typeFactory.typeSystem
        val limit =
            if ((limitStr == "" && offsetStr == "") ||
                (typeSystem is BodoSQLRelDataTypeSystem && !typeSystem.enableStreamingSortLimitOffset)
            ) {
                Expr.IntegerLiteral(
                    -1,
                )
            } else {
                if (limitStr == "") {
                    Expr.IntegerLiteral(0)
                } else {
                    Expr.Raw(limitStr)
                }
            }
        val offset =
            if ((limitStr == "" && offsetStr == "") ||
                (typeSystem is BodoSQLRelDataTypeSystem && !typeSystem.enableStreamingSortLimitOffset)
            ) {
                Expr.IntegerLiteral(
                    -1,
                )
            } else {
                if (offsetStr == "") {
                    Expr.IntegerLiteral(0)
                } else {
                    Expr.Raw(offsetStr)
                }
            }

        val sortStateVar: StateVariable = ctx.builder().symbolTable.genStateVar()
        val stateCall =
            Expr.Call(
                "bodo.libs.streaming.sort.init_stream_sort_state",
                listOf(
                    ctx.operatorID().toExpr(),
                    limit,
                    offset,
                    byList,
                    ascendingList,
                    naPositionList,
                    Expr.Tuple(
                        colNames.map { it ->
                            Expr.StringLiteral(it)
                        },
                    ),
                ),
            )
        val sortInit = Assign(sortStateVar, stateCall)
        // TODO(aneesh) provide a better memory estimate
        batchPipeline.initializeStreamingState(ctx.operatorID(), sortInit, OperatorType.SORT, SORT_MEMORY_ESTIMATE)

        return sortStateVar
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        val finalizePipeline = ctx.builder().getCurrentStreamingPipeline()
        // Append the code to delete the state
        val deleteState = Stmt(Expr.Call("bodo.libs.streaming.sort.delete_stream_sort_state", listOf(stateVar)))
        finalizePipeline.addTermination(deleteState)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        val typeSystem = cluster.typeFactory.typeSystem
        if (typeSystem is BodoSQLRelDataTypeSystem) {
            if (typeSystem.enableStreamingSort && this.getCollation().fieldCollations.size > 0) {
                return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
            }
        }
        return ExpectedBatchingProperty.alwaysSingleBatchProperty()
    }

    // Return fetch for py4j use in C++ backend code
    fun getFetch(): RexNode? = fetch

    // Return offset for py4j use in C++ backend code
    fun getOffset(): RexNode? = offset

    companion object {
        fun create(
            child: RelNode,
            collation: RelCollation,
            offset: RexNode?,
            fetch: RexNode?,
        ): BodoPhysicalSort {
            val cluster = child.cluster
            val traitSet = cluster.traitSet().replace(collation)
            return BodoPhysicalSort(cluster, traitSet, child, collation, offset, fetch)
        }

        // Return nullDirection for py4j use in C++ backend code
        fun isNullsFirst(collation: RelFieldCollation): Boolean = (collation.nullDirection == RelFieldCollation.NullDirection.FIRST)

        // Start with a static budget. Sort then expands its budget dynamically during the Finalize step.
        const val SORT_MEMORY_ESTIMATE = 256 * 1024 * 1024
    }
}
