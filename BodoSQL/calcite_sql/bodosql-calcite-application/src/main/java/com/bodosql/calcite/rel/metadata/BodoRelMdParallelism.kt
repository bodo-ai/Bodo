package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.common.TimerSupportedRel
import org.apache.calcite.rel.metadata.RelMdParallelism
import org.apache.calcite.rel.metadata.RelMetadataQuery

class BodoRelMdParallelism(private val ranks: Int) : RelMdParallelism() {
    fun splitCount(
        rel: TimerSupportedRel,
        mq: RelMetadataQuery,
    ): Int = rel.splitCount(ranks) ?: ranks
}
