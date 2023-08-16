package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.snowflake.SnowflakeFilter
import com.bodosql.calcite.adapter.snowflake.SnowflakeRel
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Sort
import org.apache.calcite.rel.metadata.RelMdRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexDynamicParam
import org.apache.calcite.rex.RexLiteral

class BodoRelMdRowCount : RelMdRowCount() {

    // TODO: extend the rest of the functions in RelMdRowCount, to cover all possible future Snowflake rels,
    // and/or move the snowflakeRel handling to a separate row count handler that runs before this one.
    // https://bodo.atlassian.net/browse/BSE-878
    override fun getRowCount(rel: RelNode, mq: RelMetadataQuery?): Double? {
        return if (rel is SnowflakeRel) {
            // If the rel is a Snowflake Rel, get the row count directly from snowflake,
            // if appropriate.
            // Note that the way that this is currently set up,
            // snowflake table scans and filters are the only two Rels for which this is enabled.
            // Both will both independently
            // submit metadata queries to get their row count
            // For now, we're considering this ok, but if/when we increase the number of RelNodes allowed to independently
            // submit metadata queries, the number of unused metadata queries will also increase
            // Therefore, should we ever extend this, we should look into a more robust way
            // of limiting the number what metadata queries we submit to snowflake
            (rel as SnowflakeRel).tryGetExpectedRowCountFromSFQuery() ?: super.getRowCount(rel, mq)
        } else {
            // Otherwise, just use the default
            super.getRowCount(rel, mq)
        }
    }

    /**
     * Note this is the aggregate implementation on version 1.35 of Calcite.
     * TODO: Upgrade and remove.
     */
    override fun getRowCount(rel: Aggregate, mq: RelMetadataQuery): Double? {
        val groupKey = rel.groupSet
        if (groupKey.isEmpty) {
            // Aggregate with no GROUP BY always returns 1 row (even on empty table).
            return 1.0
        }
        // rowCount is the cardinality of the group by columns
        var distinctRowCount = mq.getDistinctRowCount(rel.input, groupKey, null)
        if (distinctRowCount == null) {
            distinctRowCount = mq.getRowCount(rel.input) / 10
        }

        // Grouping sets multiply
        distinctRowCount *= rel.getGroupSets().size.toDouble()
        return distinctRowCount
    }

    fun getRowCount(rel: SnowflakeFilter, mq: RelMetadataQuery?): Double? {
        return rel.tryGetExpectedRowCountFromSFQuery() ?: super.getRowCount(rel, mq)
    }

    /**
     * This is a copy of RelMdRowCount, but it also handles named parameters.
     */
    override fun getRowCount(rel: Sort, mq: RelMetadataQuery): Double? {
        var rowCount: Double = mq.getRowCount(rel.input) ?: return null

        // This is the only change, allowing RexCall (the node type
        // that holds named parameters
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
