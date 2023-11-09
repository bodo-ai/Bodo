package com.bodosql.calcite.adapter.pandas

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.FlattenBase
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.util.ImmutableBitSet

class PandasFlatten(cluster: RelOptCluster, traits: RelTraitSet, input: RelNode, call: RexCall, callType: RelDataType, usedColOutputs: ImmutableBitSet, repeatColumns: ImmutableBitSet) : FlattenBase(cluster, traits.replace(PandasRel.CONVENTION), input, call, callType, usedColOutputs, repeatColumns), PandasRel {

    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        call: RexCall,
        callType: RelDataType,
        usedColOutputs: ImmutableBitSet,
        repeatColumns: ImmutableBitSet,
    ): PandasFlatten {
        return PandasFlatten(cluster, traitSet, input, call, callType, usedColOutputs, repeatColumns)
    }

    /**
     * Emits the code necessary for implementing this relational operator.
     *
     * @param implementor implementation handler.
     * @return the variable that represents this relational expression.
     */
    override fun emit(implementor: PandasRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    /**
     * Function to create the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun initStateVariable(ctx: PandasRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    /**
     * Function to delete the initial state for a streaming pipeline.
     * This should be called from emit.
     */
    override fun deleteStateVariable(ctx: PandasRel.BuildContext, stateVar: StateVariable) {
        TODO("Not yet implemented")
    }

    companion object {
        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            call: RexCall,
            callType: RelDataType,
        ): PandasFlatten {
            return create(cluster, input, call, callType, ImmutableBitSet.range(callType.fieldCount), ImmutableBitSet.of())
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            call: RexCall,
            callType: RelDataType,
            usedColOutputs: ImmutableBitSet,
        ): PandasFlatten {
            return create(cluster, input, call, callType, usedColOutputs, ImmutableBitSet.of())
        }

        @JvmStatic
        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            call: RexCall,
            callType: RelDataType,
            usedColOutputs: ImmutableBitSet,
            repeatColumns: ImmutableBitSet,
        ): PandasFlatten {
            return PandasFlatten(cluster, cluster.traitSet(), input, call, callType, usedColOutputs, repeatColumns)
        }
    }
}
