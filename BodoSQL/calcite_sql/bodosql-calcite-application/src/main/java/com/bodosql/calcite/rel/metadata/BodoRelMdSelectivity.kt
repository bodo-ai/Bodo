package com.bodosql.calcite.rel.metadata

import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.metadata.RelMdSelectivity
import org.apache.calcite.rel.metadata.RelMdUtil
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexNode
import org.apache.calcite.rex.RexUtil
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.util.ImmutableBitSet

class BodoRelMdSelectivity : RelMdSelectivity() {
    fun getSelectivity(
        rel: RelSubset,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double = getSelectivity(rel.bestOrOriginal, mq, predicate)

    override fun getSelectivity(
        rel: Aggregate,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? {
        val notPushable: List<RexNode?> = ArrayList()
        val pushable: List<RexNode?> = ArrayList()
        RelOptUtil.splitFilters(
            rel.groupSet,
            predicate,
            pushable,
            notPushable,
        )
        val rexBuilder = rel.cluster.rexBuilder
        val childPredicate = RexUtil.composeConjunction(rexBuilder, pushable, true)
        val selectivity = mq.getSelectivity(rel.input, childPredicate)
        return if (selectivity == null) {
            null
        } else {
            val predicate = RexUtil.composeConjunction(rexBuilder, notPushable, true)
            selectivity * guessSelectivity(predicate)
        }
    }

    override fun getSelectivity(
        rel: Project,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double? {
        val notPushable: List<RexNode?> = ArrayList()
        val pushable: List<RexNode?> = ArrayList()
        RelOptUtil.splitFilters(
            ImmutableBitSet.range(rel.rowType.fieldCount),
            predicate,
            pushable,
            notPushable,
        )
        val rexBuilder = rel.cluster.rexBuilder
        val childPredicate = RexUtil.composeConjunction(rexBuilder, pushable, true)
        val modifiedPredicate: RexNode? =
            if (childPredicate == null) {
                null
            } else {
                RelOptUtil.pushPastProject(childPredicate, rel)
            }
        val selectivity = mq.getSelectivity(rel.input, modifiedPredicate)
        return if (selectivity == null) {
            null
        } else {
            val predicate = RexUtil.composeConjunction(rexBuilder, notPushable, true)
            selectivity * guessSelectivity(predicate)
        }
    }

    // Catch-all rule when none of the others apply.
    override fun getSelectivity(
        rel: RelNode?,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double = guessSelectivity(predicate)

    companion object {
        /**
         * Estimates the selectivity of a predicate. Replaces RelMdUtil.guessSelectivity.
         */
        @JvmStatic
        fun guessSelectivity(predicate: RexNode?): Double {
            var sel = 1.0
            if (predicate == null || predicate.isAlwaysTrue) {
                return sel
            }

            var artificialSel = 1.0

            for (pred in RelOptUtil.conjunctions(predicate)) {
                if (pred.kind == SqlKind.IS_NOT_NULL) {
                    sel *= .99
                } else if (pred.kind == SqlKind.IS_NULL) {
                    sel *= .01
                } else if (pred is RexCall &&
                    (
                        pred.operator
                            === RelMdUtil.ARTIFICIAL_SELECTIVITY_FUNC
                    )
                ) {
                    artificialSel *= RelMdUtil.getSelectivityValue(pred)
                } else if (pred.isA(SqlKind.EQUALS)) {
                    sel *= .15
                } else if (pred.isA(SqlKind.COMPARISON)) {
                    sel *= .5
                } else {
                    sel *= .25
                }
            }

            return sel * artificialSel
        }
    }
}
