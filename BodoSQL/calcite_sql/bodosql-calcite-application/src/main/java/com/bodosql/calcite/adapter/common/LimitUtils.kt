package com.bodosql.calcite.adapter.common

import org.apache.calcite.plan.RelOptRuleCall
import org.apache.calcite.rel.core.Sort

class LimitUtils private constructor() {
    companion object {
        /**
         * Extract the nodes for standard Sort conversion rules.
         */
        @JvmStatic
        fun <E> extractSortNodes(call: RelOptRuleCall): Pair<Sort, E> {
            return when (call.rels.size) {
                // Inputs are:
                // Sort ->
                //     XXXToPandasConverter ->
                //         XXXRel
                3 -> Pair(call.rel(0), call.rel(2))
                // Inputs are:
                // Sort ->
                //     XXXRel
                else -> Pair(call.rel(0), call.rel(1))
            }
        }

        /**
         * Determine if a sort is only a limit and doesn't require any actual sorting.
         */
        @JvmStatic
        fun isOnlyLimit(sort: Sort): Boolean {
            // We push down sorts that only contain fetch and/or offset
            return (sort.offset != null || sort.fetch != null) && sort.getCollation().fieldCollations.isEmpty()
        }
    }
}
