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
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
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
            selectivity * estimateSelectivity(rel, mq, predicate)
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
            selectivity * estimateSelectivity(rel, mq, predicate)
        }
    }

    // Catch-all rule when none of the others apply.
    override fun getSelectivity(
        rel: RelNode?,
        mq: RelMetadataQuery,
        predicate: RexNode?,
    ): Double = estimateSelectivity(rel, mq, predicate)

    companion object {
        /**
         * Smallest selectivity we will derive from a column's NDV. Without histograms an
         * NDV-derived estimate assumes uniformity, so a very high NDV (e.g. a key column)
         * would otherwise produce a selectivity close to 0 and make any plan that filters
         * on it look free.
         */
        private const val MIN_NDV_SELECTIVITY = 1e-4

        /**
         *
         * Same as [guessSelectivity], except that an equality between a column of
         * [rel] and a literal uses 1/NDV when the column's distinct count is known,
         * instead of the fixed 0.15 guess. See `n_name = 'SAUDI ARABIA'` in TPC-H Q21 as an example.
         * [predicate] must be expressed over [rel]'s output columns.
         */
        @JvmStatic
        fun estimateSelectivity(
            rel: RelNode?,
            mq: RelMetadataQuery,
            predicate: RexNode?,
        ): Double {
            if (rel == null) {
                return guessSelectivity(predicate)
            }
            if (predicate == null || predicate.isAlwaysTrue) {
                return 1.0
            }
            var sel = 1.0
            for (pred in RelOptUtil.conjunctions(predicate)) {
                // Use NDVs for selectivity when available
                sel *= ndvEqualitySelectivity(rel, mq, pred) ?: guessSelectivity(pred)
            }
            return sel
        }

        /**
         * Selectivity of `<column> = <literal>` (or the reverse) derived from the column's
         * distinct count on [rel], or null if this is not such a predicate or the distinct
         * count is unknown.
         */
        private fun ndvEqualitySelectivity(
            rel: RelNode,
            mq: RelMetadataQuery,
            pred: RexNode,
        ): Double? {
            if (!pred.isA(SqlKind.EQUALS) || pred !is RexCall) {
                return null
            }
            val left = RexUtil.removeCast(pred.operands[0])
            val right = RexUtil.removeCast(pred.operands[1])
            val ref =
                when {
                    left is RexInputRef && right is RexLiteral -> left
                    right is RexInputRef && left is RexLiteral -> right
                    else -> return null
                }
            val ndv = mq.getDistinctRowCount(rel, ImmutableBitSet.of(ref.index), null) ?: return null
            if (ndv.isNaN() || ndv <= 1.0) {
                // A single distinct value: the predicate either matches everything or
                // nothing, and we have no way to tell which. Keep the old guess.
                return null
            }
            return (1.0 / ndv).coerceIn(MIN_NDV_SELECTIVITY, 1.0)
        }

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
