package com.bodosql.calcite.rel.metadata

import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.metadata.RelMdSelectivity
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

class BodoRelMdSelectivity : RelMdSelectivity() {
    override fun getSelectivity(
        rel: Join,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? {
        return if (rel.isSemiJoin) {
            super.getSelectivity(rel, mq, predicate)
        } else {
            // TODO(njriasan): FIXME to reference our custom metadata query
            // for column estimated distinct counts
            getSelectivity(rel as RelNode, mq, predicate)
        }
    }

    fun getSelectivity(
        rel: RelSubset,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? = getSelectivity(rel.getBestOrOriginal(), mq, predicate)
}
