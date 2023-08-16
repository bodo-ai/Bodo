package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.SingleRel
import org.apache.calcite.rel.metadata.RelMdDistinctRowCount
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode
import org.apache.calcite.util.ImmutableBitSet

class BodoRelMdDistinctRowCount : RelMdDistinctRowCount() {
    fun getDistinctRowCount(
        rel: SingleRel,
        mq: RelMetadataQuery,
        groupKey: ImmutableBitSet,
        predicate: RexNode?,
    ): Double? {
        return mq.getDistinctRowCount(rel.input, groupKey, predicate)
    }
}
