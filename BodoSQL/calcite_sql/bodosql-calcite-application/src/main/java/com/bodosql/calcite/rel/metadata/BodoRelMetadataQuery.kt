package com.bodosql.calcite.rel.metadata

import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.MetadataHandlerProvider
import org.apache.calcite.rel.metadata.RelMdUtil
import org.apache.calcite.rel.metadata.RelMetadataQuery

/**
 * Bodo class for extending RelMetadataQuery. Any built in calcite statistics are
 * handled through the call to the parent class. For any additional Metadata queries
 * you should define a new handler, which should be a private variable and then introduce
 * a new method to access it. This class design is based on the MyRelMetadataQuery
 * test inside Calcite.
 *
 * This is provided at the creation of a RelOptCluster. For consistency purposes we
 * specify this file at BodoRelOptClusterSetup.create and nowhere else. We require
 * BodoRelOptClusterSetup.create be used for any cluster creation inside this project.
 *
 * Also inspired by the custom RelMetadataQuery in Dremio:
 *https://github.com/dremio/dremio-oss/blob/be47367c523e4eded35df4e8fff725f53160558e/sabot/kernel/src/main/java/com/dremio/exec/planner/cost/DremioRelMetadataQuery.java#L32
 */
class BodoRelMetadataQuery(
    provider: MetadataHandlerProvider,
) : RelMetadataQuery(provider) {
    private var columnDistinctCountHandler: ColumnDistinctCount.Handler = handler(ColumnDistinctCount.Handler::class.java)

    fun getColumnDistinctCount(
        r: RelNode,
        column: Int,
    ): Double? {
        // Note: This pattern is copied from Calcite. I'm not sure what situations require
        // revise, but I copied the pattern to be conservative.
        while (true) {
            try {
                val result = columnDistinctCountHandler.getColumnDistinctCount(r, this, column)
                // Ensure that 1 <= result <= rowCount
                val rowCount = getRowCount(r)
                return if (rowCount != null && result != null) {
                    RelMdUtil.validateResult(maxOf(1.0, minOf(rowCount, result)))
                } else {
                    null
                }
            } catch (e: MetadataHandlerProvider.NoHandler) {
                columnDistinctCountHandler = revise(ColumnDistinctCount.Handler::class.java)
            }
        }
    }

    companion object {
        /**
         * Ensures only one instance of BodoRelMetadataQuery is created for statistics
         */
        @JvmStatic
        fun getSupplier(): BodoRelMetadataQuerySupplier {
            // TODO(njriasan): Add details to make this different from the default.
            val provider: MetadataHandlerProvider = BodoRelMetadataHandlerProvider(BodoRelMetadataProvider())
            return BodoRelMetadataQuerySupplier(BodoRelMetadataQuery(provider))
        }
    }
}
