package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.RelMdSize
import org.apache.calcite.rel.metadata.RelMetadataQuery

class PandasRelMdSize : RelMdSize() {
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
}
