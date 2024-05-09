package com.bodosql.calcite.rel.core

import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelTraitSet
import org.apache.calcite.rel.AbstractRelNode
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.RelRoot
import org.apache.calcite.rel.RelWriter
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rel.type.RelDataType

/**
 * Base class for a RelNode that represents a cached section of a plan.
 */
open class CachedSubPlanBase protected constructor(
    val cachedPlan: RelRoot,
    val cacheID: Int,
    cluster: RelOptCluster,
    traitSet: RelTraitSet,
) : AbstractRelNode(cluster, traitSet) {
    init {
        // Initialize the type to avoid errors with Kotlin suggesting to access
        // the protected field directly.
        rowType = getRowType()
    }

    override fun copy(
        traitSet: RelTraitSet,
        inputs: List<RelNode>,
    ): RelNode {
        assert(inputs.isEmpty()) { "CachedSubPlanBase should not have any inputs" }
        return CachedSubPlanBase(cachedPlan, cacheID, cluster, traitSet)
    }

    override fun explainTerms(pw: RelWriter): RelWriter {
        return super.explainTerms(pw)
            .item("CacheID", cacheID)
    }

    override fun deriveRowType(): RelDataType {
        return cachedPlan.validatedRowType
    }

    override fun estimateRowCount(mq: RelMetadataQuery): Double {
        return cachedPlan.rel.estimateRowCount(mq)
    }

    // TODO: Add cost calculation. This is tricky because the cost should be amortized
    // across the uses.
}
