package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.pandas.PandasRel
import org.apache.calcite.rel.metadata.RelMdParallelism
import org.apache.calcite.rel.metadata.RelMetadataQuery

class BodoRelMdParallelism(private val ranks: Int) : RelMdParallelism() {
    fun splitCount(
        rel: PandasRel,
        mq: RelMetadataQuery,
    ): Int = rel.splitCount(ranks) ?: ranks
}
