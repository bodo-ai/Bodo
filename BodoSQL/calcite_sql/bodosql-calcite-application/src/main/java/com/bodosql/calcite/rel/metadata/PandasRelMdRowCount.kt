package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.metadata.RelMdRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexLiteral

class PandasRelMdRowCount : RelMdRowCount() {
    /**
     * This is a copy of RelMdRowCount but it also handles named parameters.
     */
    override fun getRowCount(rel: Sort, mq: RelMetadataQuery): Double? {
        var rowCount = mq.getRowCount(rel.input) ?: return null
        if (rel.offset is RexDynamicParam || rel.offset is RexCall) {
            return rowCount
        }
        val offset = if (rel.offset == null) {
            0
        } else {
            RexLiteral.intValue(rel.offset)
        }

        rowCount = (rowCount - offset).coerceAtLeast(0.0)
        if (rel.fetch != null) {
            if (rel.fetch is RexDynamicParam || rel.fetch is RexCall) {
                return rowCount
            }
            val limit = RexLiteral.intValue(rel.fetch)
            if (limit < rowCount) {
                return limit.toDouble()
            }
        }
        return rowCount
    }
}
