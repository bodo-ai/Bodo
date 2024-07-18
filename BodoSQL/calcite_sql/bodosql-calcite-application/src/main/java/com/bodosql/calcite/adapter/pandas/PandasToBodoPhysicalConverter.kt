package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.adapter.bodo.BodoPhysicalRel
import com.bodosql.calcite.codeGeneration.OperatorEmission
import com.bodosql.calcite.codeGeneration.OutputtingPipelineEmission
import com.bodosql.calcite.codeGeneration.OutputtingStageEmission
import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.ir.UnusedStateVariable
import com.bodosql.calcite.traits.BatchingProperty
import org.apache.calcite.plan.ConventionTraitDef
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.convert.ConverterImpl

class PandasToBodoPhysicalConverter(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode) :
    ConverterImpl(cluster, ConventionTraitDef.INSTANCE, traits.replace(BodoPhysicalRel.CONVENTION), input),
    BodoPhysicalRel {
    init {
        // Initialize the type to avoid errors with Kotlin suggesting to access
        // the protected field directly.
        rowType = getRowType()
    }

    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): PandasToBodoPhysicalConverter {
        return PandasToBodoPhysicalConverter(cluster, traitSet, sole(inputs))
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return inputBatchingProperty
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     * Right now this just has the actual Pandas nodes generate code.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        return if (isStreaming()) {
            // This node is just pass through.
            val stage =
                OutputtingStageEmission(
                    { _, _, table -> table!! },
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
            (implementor::build)(listOf()) { ctx, _ -> ctx.visitChild(input, 0) }
        }
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }
}
