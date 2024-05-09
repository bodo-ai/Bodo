package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.WindowBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.hint.RelHint
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.util.ImmutableBitSet

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

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
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

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.alwaysSingleBatchProperty()
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
