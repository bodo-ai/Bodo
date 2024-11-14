package com.bodosql.calcite.rel.metadata

import org.apache.calcite.linq4j.tree.Types
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.metadata.Metadata
import org.apache.calcite.rel.metadata.MetadataDef
import org.apache.calcite.rel.metadata.MetadataHandler
import org.apache.calcite.rel.metadata.RelMetadataQuery

// ~ Inner classes and interfaces -------------------------------------------
interface ColumnDistinctCount : Metadata {
    fun getColumnDistinctCount(column: Int): Double?

    /** Handler API.  */
    interface Handler : MetadataHandler<ColumnDistinctCount?> {
        /**
         * <p>Interface for getting the approximate distinct count of a single column.
         * This can either be based on static planner information or eventually a query
         * submission to snowflake with approx_count_distinct
         *
         * For example the final snowflake query could be:
         * <code>select approx_count_distinct(A) from myTable</code>
         *
         * @return This function returns either an estimate based on the relationship between
         * operators or null if a reasonable estimate could not be made.
         */
        fun getColumnDistinctCount(
            rel: RelNode,
            mq: RelMetadataQuery,
            column: Int,
        ): Double?

        override fun getDef(): MetadataDef<ColumnDistinctCount?> = DEF
    }

    companion object {
        val METHOD =
            Types.lookupMethod(
                ColumnDistinctCount::class.java,
                "getColumnDistinctCount",
                Int::class.java,
            )

        val DEF = MetadataDef.of(ColumnDistinctCount::class.java, Handler::class.java, METHOD)
    }
}
