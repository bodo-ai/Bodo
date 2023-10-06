package com.bodosql.calcite.rel.metadata

import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.metadata.RelMdSelectivity
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexNode

class BodoRelMdSelectivity : RelMdSelectivity() {
    fun getSelectivity(
        rel: RelSubset,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? = getSelectivity(rel.getBestOrOriginal(), mq, predicate)
}
