package com.bodosql.calcite.adapter.bodo

import com.bodosql.calcite.ir.BodoEngineTable
import com.bodosql.calcite.ir.StateVariable
import com.bodosql.calcite.rel.core.AggregateBase
import com.bodosql.calcite.traits.BatchingProperty
import com.bodosql.calcite.traits.ExpectedBatchingProperty
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.AggregateCall
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.ImmutableBitSet
import kotlin.math.ceil

class BodoPhysicalAggregate(
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
    input: RelNode,
    groupSet: ImmutableBitSet,
    groupSets: List<ImmutableBitSet>?,
    aggCalls: List<AggregateCall>,
) : AggregateBase(
        cluster,
        traitSet.replace(BodoPhysicalRel.CONVENTION),
        ImmutableList.of(),
        input,
        groupSet,
        groupSets,
        aggCalls,
    ),
    BodoPhysicalRel {
    override fun copy(
        traitSet: RelTraitSet,
        input: RelNode,
        groupSet: ImmutableBitSet,
        groupSets: List<ImmutableBitSet>?,
        aggCalls: List<AggregateCall>,
    ): BodoPhysicalAggregate {
        return BodoPhysicalAggregate(cluster, traitSet, input, groupSet, groupSets, aggCalls)
    }

    override fun emit(implementor: BodoPhysicalRel.Implementor): BodoEngineTable {
        TODO("Not yet implemented")
    }

    override fun initStateVariable(ctx: BodoPhysicalRel.BuildContext): StateVariable {
        TODO("Not yet implemented")
    }

    override fun deleteStateVariable(
        ctx: BodoPhysicalRel.BuildContext,
        stateVar: StateVariable,
    ) {
        TODO("Not yet implemented")
    }

    override fun expectedOutputBatchingProperty(inputBatchingProperty: BatchingProperty): BatchingProperty {
        return ExpectedBatchingProperty.aggregateProperty(groupSets, aggCalls, getRowType())
    }

    /**
     * Get group by build memory estimate for memory budget comptroller
     */
    fun estimateBuildMemory(mq: RelMetadataQuery): Int {
        // See if streaming group by will use accumulate or aggregate code path
        var isStreamAccumulate = false
        for (aggCall in aggCalls) {
            val kind = aggCall.aggregation.getKind()
            val name = aggCall.aggregation.name
            // Should match accumulate function check in C++:
            // https://github.com/Bodo-inc/Bodo/blob/3c902f01b0aa0748793b00554304d8a051f511aa/bodo/libs/_stream_groupby.cpp#L1101
            if (name == "LISTAGG" || kind == SqlKind.MEDIAN || kind == SqlKind.MODE ||
                (kind == SqlKind.COUNT && aggCall.argList.isNotEmpty() && aggCall.isDistinct)
            ) {
                isStreamAccumulate = true
                break
            }
        }

        // Get the set of group by key indices to skip in type check below
        val keySet = mutableSetOf<Int>()
        for (i in 0..<groupSet.size()) {
            if (groupSet[i]) {
                keySet.add(i)
            }
        }

        // Streaming groupby uses accumulate path when running values are string or nested data types
        // https://github.com/Bodo-inc/Bodo/blob/da5696256a1fc44d41a62354de8f492bf0e7f148/bodo/libs/_stream_groupby.cpp#L1182
        for (i in 0..<this.getRowType().fieldList.size) {
            // NOTE: this.rowType will access the actual attribute here which could be null.
            // We are in a subclass and the attributed is protected, so we have access and Kotlin won't use the getter.
            if (keySet.contains(i)) {
                continue
            }
            val field = this.getRowType().fieldList[i]
            val typeName = field.type.sqlTypeName
            if (typeName.equals(SqlTypeName.VARCHAR) || typeName.equals(SqlTypeName.VARBINARY) ||
                typeName.equals(SqlTypeName.CHAR) || typeName.equals(SqlTypeName.BINARY) ||
                typeName.equals(SqlTypeName.ARRAY) || typeName.equals(SqlTypeName.MAP) ||
                // OTHER is VARIANT type
                typeName.equals(SqlTypeName.OTHER)
            ) {
                isStreamAccumulate = true
                break
            }
        }

        // Accumulate code path needs all input in memory
        if (isStreamAccumulate) {
            val buildRows = mq.getRowCount(this.getInput())
            val averageBuildRowSize = mq.getAverageRowSize(this.getInput()) ?: 8.0
            // multiply by 3 to account for extra memory needed in update call at the end
            return ceil(3 * buildRows * averageBuildRowSize).toInt()
        } else {
            // Use output row count for aggregate code path
            val distinctRows = mq.getRowCount(this)
            val averageBuildRowSize = mq.getAverageRowSize(this) ?: 8.0
            return ceil(distinctRows * averageBuildRowSize).toInt()
        }
    }

    companion object {
        fun create(
            cluster: RelOptCluster,
            input: RelNode,
            groupSet: ImmutableBitSet,
            groupSets: List<ImmutableBitSet>,
            aggCalls: List<AggregateCall>,
        ): BodoPhysicalAggregate {
            return BodoPhysicalAggregate(cluster, cluster.traitSet(), input, groupSet, groupSets, aggCalls)
        }
    }
}
