package com.bodosql.calcite.rel.core

import com.bodosql.calcite.plan.Cost
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptCost
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.RelFieldCollation
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexOver
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet

open class MinRowNumberFilterBase(
    cluster: RelOptCluster,
    traits: RelTraitSet,
    child: RelNode,
    condition: RexNode,
    val inputsToKeep: ImmutableBitSet,
) : FilterBase(cluster, traits, child, condition) {
    val partitionColSet: ImmutableBitSet
    val orderColSet: ImmutableBitSet
    val ascendingList: List<Boolean>
    val nullPosList: List<Boolean>
    init {
        if (condition is RexOver && condition.kind == SqlKind.MIN_ROW_NUMBER_FILTER) {
            partitionColSet = ImmutableBitSet.of(
                condition.window.partitionKeys.map {
                    if (it is RexInputRef) {
                        it.index
                    } else {
                        throw Exception("Malformed MinRowNumberFilter condition: $condition")
                    }
                },
            )
            orderColSet = ImmutableBitSet.of(
                condition.window.orderKeys.map {
                    val lhs = it.left
                    if (lhs is RexInputRef) {
                        (lhs as RexInputRef).index
                    } else {
                        throw Exception("Malformed MinRowNumberFilter condition: $condition")
                    }
                },
            )
            ascendingList = condition.window.orderKeys.map {
                it.direction == RelFieldCollation.Direction.ASCENDING
            }
            nullPosList = condition.window.orderKeys.map {
                it.nullDirection == RelFieldCollation.NullDirection.LAST
            }
        } else {
            throw Exception("Malformed MinRowNumberFilter condition: $condition")
        }
    }

    override fun deriveRowType(): RelDataType {
        val inputFields = input.rowType.fieldList
        val newInputFields = inputFields.filterIndexed { idx, _ -> inputsToKeep.contains(idx) }
        return cluster.typeFactory.createStructType(newInputFields)
    }

    override fun explainTerms(pw: RelWriter): RelWriter? {
        return pw.item("input", getInput()).item("partition", partitionColSet)
            .item("order", orderColSet)
            .item("ascending", ascendingList)
            .item("nullLast", nullPosList)
            .item("inputsToKeep", inputsToKeep)
    }

    override fun computeSelfCost(planner: RelOptPlanner, mq: RelMetadataQuery): RelOptCost {
        // The cost relates to the aggregating of rows which depends on the number of
        // distinct values of the partition columns, but also scales with the size
        // of the pass-through columns.
        val rowSize = mq.getAverageRowSize(this) ?: 0.0
        val outputRows = mq.getRowCount(this)
        val memoryCost = rowSize.times(outputRows)
        return Cost(rows = outputRows, mem = memoryCost)
    }
}
