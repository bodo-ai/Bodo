package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.bodo.BodoCostEstimator
import com.bodosql.calcite.adapter.bodo.BodoPhysicalMinRowNumberFilter
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdSize
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType

class BodoRelMdSize : RelMdSize() {
    /**
     * The default for this is to report that it doesn't know the size.
     *
     * Change the default to just return the average type value size
     * based on the column type.
     */
    override fun averageColumnSizes(
        rel: RelNode,
        mq: RelMetadataQuery,
    ): List<Double?> =
        rel.rowType.fieldList.map { rowType ->
            averageTypeValueSize(rowType.type)
        }

    fun averageColumnSizes(
        rel: BodoPhysicalMinRowNumberFilter,
        mq: RelMetadataQuery,
    ): List<Double?> =
        rel.rowType.fieldList.map { rowType ->
            averageTypeValueSize(rowType.type)
        }

    /**
     * Ensure we use the same calculation throughout the planner by reusing BodoCostEstimator.averageTypeValueSize.
     */
    override fun averageTypeValueSize(type: RelDataType): Double = BodoCostEstimator.averageTypeValueSize(type)
}
