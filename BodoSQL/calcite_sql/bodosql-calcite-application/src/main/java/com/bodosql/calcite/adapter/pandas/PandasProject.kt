package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.adapter.common.ProjectUtils.Companion.generateDataFrame
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.plan.makeCost
import com.bodosql.calcite.rel.core.ProjectBase
import com.bodosql.calcite.traits.BatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.validate.SqlValidatorUtil

class PandasProject(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    projects: List<RexNode>,
    rowType: RelDataType,
) : ProjectBase(cluster, traitSet.replace(PandasRel.CONVENTION), ImmutableList.of(), input, projects, rowType), PandasRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        projects: List<RexNode>,
        rowType: RelDataType,
    ): PandasProject {
        return PandasProject(cluster, traitSet, input, projects, rowType)
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     * This is needed because the current implementation just uses the
     * PandasToBodoPhysicalConverter as a barrier and generates all the
     * code in its inner nodes.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        return if (isStreaming()) {
            val stage =
                OutputtingStageEmission(
                    { ctx, stateVar, table ->
                        val inputVar = table!!
                        val localRefs = mutableListOf<Variable>()
                        val translator = ctx.streamingRexTranslator(inputVar, localRefs, stateVar)
                        generateDataFrame(ctx, inputVar, translator, projects, localRefs, projects, input)
                    },
                    reportOutTableSize = false,
                )
            val pipeline =
                OutputtingPipelineEmission(
                    listOf(stage),
                    false,
                    input,
                )
            val operatorEmission =
                OperatorEmission(
                    { UnusedStateVariable },
                    { _, _ -> },
                    listOf(),
                    pipeline,
                    timeStateInitialization = false,
                )
            implementor.buildStreaming(operatorEmission)!!
        } else {
            implementor.build { ctx ->
                val inputVar = ctx.visitChild(input, 0)
                val localRefs = mutableListOf<Variable>()
                val translator = ctx.rexTranslator(inputVar, localRefs)
                generateDataFrame(ctx, inputVar, translator, projects, localRefs, projects, input)
            }
        }
    }

    override fun computeSelfCost(
        planner: RelOptPlanner,
        mq: RelMetadataQuery,
    ): RelOptCost {
        val rows = mq.getRowCount(this)
        return planner.makeCost(cpu = 0.0, mem = 0.0, rows = rows)
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return inputBatchingProperty
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            projects: List<RexNode>,
            rowType: RelDataType,
        ): PandasProject {
            return PandasProject(cluster, traitSet, input, projects, rowType)
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            traitSet: RelTraitSet,
            input: RelNode,
            projects: List<RexNode>,
            fieldNames: List<String?>?,
        ): PandasProject {
            val rowType =
                RexUtil.createStructType(
                    cluster.typeFactory,
                    projects,
                    fieldNames,
                    SqlValidatorUtil.F_SUGGESTER,
                )
            return create(cluster, traitSet, input, projects, rowType)
        }
    }
}
