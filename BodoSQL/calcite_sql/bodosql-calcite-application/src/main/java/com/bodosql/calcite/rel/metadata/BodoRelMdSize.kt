package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.pandas.PandasCostEstimator
import com.bodosql.calcite.adapter.pandas.PandasMinRowNumberFilter
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
    override fun averageColumnSizes(rel: RelNode, mq: RelMetadataQuery): List<Double?> =
        rel.rowType.fieldList.map { rowType ->
            averageTypeValueSize(rowType.type)
        }

    fun averageColumnSizes(rel: PandasMinRowNumberFilter, mq: RelMetadataQuery): List<Double?> =
        rel.rowType.fieldList.map { rowType ->
            averageTypeValueSize(rowType.type)
        }

    /**
     * Ensure we use the same calculation throughout the planner by reusing PandasCostEstimator.averageTypeValueSize.
     */
    override fun averageTypeValueSize(type: RelDataType): Double = PandasCostEstimator.averageTypeValueSize(type)
}
