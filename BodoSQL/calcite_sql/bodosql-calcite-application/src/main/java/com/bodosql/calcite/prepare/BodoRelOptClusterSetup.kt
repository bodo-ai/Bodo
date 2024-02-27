package com.bodosql.calcite.prepare

import com.bodosql.calcite.rel.metadata.BodoRelMetadataQuery
import org.apache.calcite.plan.RelOptCluster
import org.apache.calcite.plan.RelOptPlanner
import org.apache.calcite.rex.RexBuilder

/**
 * Bodo Implementation for RelOptCluster.create that includes any required configuration
 * for Metadata.
 */
class BodoRelOptClusterSetup {
    companion object {
        @JvmStatic
        fun create(
            planner: RelOptPlanner,
            rexBuilder: RexBuilder,
        ): RelOptCluster {
            val cluster = RelOptCluster.create(planner, rexBuilder)
            cluster.metadataQuerySupplier = BodoRelMetadataQuery.getSupplier()
            return cluster
        }
    }
}
