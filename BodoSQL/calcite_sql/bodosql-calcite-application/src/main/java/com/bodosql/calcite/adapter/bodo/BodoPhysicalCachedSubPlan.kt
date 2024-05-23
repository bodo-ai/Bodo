package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Op.Assign
import com.bodosql.calcite.ir.Op.Stmt
import com.bodosql.calcite.ir.Op.TupleAssign
import com.bodosql.calcite.ir.OperatorType
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.rel.core.CachedPlanInfo
import com.bodosql.calcite.rel.core.CachedSubPlanBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode

class BodoPhysicalCachedSubPlan private constructor(
    cachedPlan: CachedPlanInfo,
    cacheID: Int,
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
) :
    CachedSubPlanBase(
            cachedPlan,
            cacheID,
            cluster,
            traitSet.replace(BodoPhysicalRel.CONVENTION),
        ),
        BodoPhysicalRel {
        override fun copy(
            traitSet: RelTraitSet,
            inputs: List<RelNode>,
        ): BodoPhysicalCachedSubPlan {
            assert(inputs.isEmpty()) { "BodoPhysicalCachedSubPlan should not have any inputs" }
            return BodoPhysicalCachedSubPlan(cachedPlan, cacheID, cluster, traitSet)
        }

        /**
         * Emits the code necessary for implementing this relational operator.
         *
         * @param implementor implementation handler.
         * @return the variable that represents this relational expression.
         */
        override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
            val relationalOperatorCache = implementor.getRelationalOperatorCache()
            val isCached = relationalOperatorCache.isNodeCached(cacheID)
            return if (isStreaming()) {
                val stage =
                    OutputtingStageEmission(
                        { ctx, _, table ->
                            val builder = ctx.builder()
                            val frame = builder.getCurrentStreamingPipeline()
                            if (isCached) {
                                val (stateVar, operatorID) = relationalOperatorCache.getStateInfo(cacheID)
                                val newTableVar = builder.symbolTable.genTableVar()
                                frame.add(
                                    TupleAssign(
                                        listOf(newTableVar, frame.getExitCond()),
                                        Expr.Call("bodo.libs.table_builder.table_builder_pop_chunk", listOf(stateVar)),
                                    ),
                                )
                                frame.deleteStreamingState(
                                    operatorID,
                                    Stmt(Expr.Call("bodo.libs.table_builder.delete_table_builder_state", listOf(stateVar))),
                                )
                                BodoEngineTable(newTableVar.emit(), this)
                            } else {
                                val inputVar = table!!
                                // Mark the node as seen.
                                relationalOperatorCache.cacheTable(cacheID, inputVar)
                                val numConsumers = (cachedPlan.getNumConsumers() - 1)
                                val stateVars =
                                    (0 until numConsumers).map {
                                        Pair(builder.symbolTable.genStateVar(), builder.newOperatorID(this))
                                    }
                                stateVars.map {
                                    val (stateVar, operatorID) = it
                                    frame.initializeStreamingState(
                                        operatorID,
                                        Assign(
                                            stateVar,
                                            Expr.Call(
                                                "bodo.libs.table_builder.init_table_builder_state",
                                                listOf(operatorID.toExpr()),
                                                listOf(Pair("use_chunked_builder", Expr.True)),
                                            ),
                                        ),
                                        // ChunkedTableBuilder doesn't need pinned memory budget
                                        OperatorType.ACCUMULATE_TABLE,
                                        0,
                                    )
                                    frame.add(Stmt(Expr.Call("bodo.libs.table_builder.table_builder_append", listOf(stateVar, inputVar))))
                                }
                                relationalOperatorCache.setStateInfo(cacheID, stateVars)
                                inputVar
                            }
                        },
                        reportOutTableSize = isCached,
                    )
                val pipeline =
                    OutputtingPipelineEmission(
                        listOf(stage),
                        // The logic to either generate the cache content the first time or
                        // start a new pipeline for loading from cache is handled here.
                        isCached,
                        if (isCached) null else cachedPlan.plan,
                    )
                val operatorEmission =
                    OperatorEmission(
                        { ctx -> initStateVariable(ctx) },
                        { ctx, stateVar -> deleteStateVariable(ctx, stateVar) },
                        listOf(),
                        pipeline,
                        timeStateInitialization = false,
                    )
                implementor.buildStreaming(operatorEmission)!!
            } else {
                implementor.build { ctx ->
                    if (isCached) {
                        relationalOperatorCache.getCachedTable(cacheID)
                    } else {
                        val table = ctx.visitChild(cachedPlan.plan, 0)
                        relationalOperatorCache.cacheTable(cacheID, table)
                        table
                    }
                }
            }
        }

        /**
         * Function to create the initial state for a streaming pipeline.
         * This should be called from emit.
         */
        override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
            return UnusedStateVariable
        }

        /**
         * Function to delete the initial state for a streaming pipeline.
         * This should be called from emit.
         */
        override fun deleteStateVariable(
            ctx: BodoPhysicalRel.BuildContext,
            stateVar: StateVariable,
        ) {
            // Do Nothing
        }

        override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
            return ExpectedBatchingProperty.streamingIfPossibleProperty(getRowType())
        }

        companion object {
            @JvmStatic
            fun create(
                cachedPlan: CachedPlanInfo,
                cacheID: Int,
            ): BodoPhysicalCachedSubPlan {
                val rootNode = cachedPlan.plan
                return BodoPhysicalCachedSubPlan(cachedPlan, cacheID, rootNode.cluster, rootNode.traitSet)
            }
        }
    }
