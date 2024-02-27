package com.bodosql.calcite.rel.logical

import com.bodosql.calcite.rel.core.FlattenBase
import org.apache.calcite.plan.Convention
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexCall
import org.apache.calcite.util.ImmutableBitSet

class BodoLogicalFlatten(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    input: RelNode,
    call: RexCall,
    callType: RelDataType,
    usedColOutputs: ImmutableBitSet,
    repeatColumns: ImmutableBitSet,
) : FlattenBase(
        cluster,
        traits,
        input,
        call,
        callType,
        usedColOutputs,
        repeatColumns,
    ) {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        call: RexCall,
        callType: RelDataType,
        usedColOutputs: ImmutableBitSet,
        repeatColumns: ImmutableBitSet,
    ): BodoLogicalFlatten {
        return BodoLogicalFlatten(cluster, traitSet, input, call, callType, usedColOutputs, repeatColumns)
    }

    companion object {
        @JvmStatic
        fun create(
            input: RelNode,
            call: RexCall,
            callType: RelDataType,
        ): BodoLogicalFlatten {
            return create(input, call, callType, ImmutableBitSet.range(callType.fieldCount), ImmutableBitSet.of())
        }

        @JvmStatic
        fun create(
            input: RelNode,
            call: RexCall,
            callType: RelDataType,
            usedColOutputs: ImmutableBitSet,
        ): BodoLogicalFlatten {
            return create(input, call, callType, usedColOutputs, ImmutableBitSet.of())
        }

        @JvmStatic
        fun create(
            input: RelNode,
            call: RexCall,
            callType: RelDataType,
            usedColOutputs: ImmutableBitSet,
            repeatColumns: ImmutableBitSet,
        ): BodoLogicalFlatten {
            val cluster = input.cluster
            val traitSet = cluster.traitSet().replace(Convention.NONE)
            return BodoLogicalFlatten(cluster, traitSet, input, call, callType, usedColOutputs, repeatColumns)
        }
    }
}
