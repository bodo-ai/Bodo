package com.bodosql.calcite.rel.metadata

import org.apache.calcite.plan.RelOptUtil
import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Join
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
import org.apache.calcite.sql.type.SqlTypeUtil
import org.apache.calcite.util.ImmutableBitSet
import kotlin.math.max

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

        /** String search functions handled by [substringSelectivity]. */
        private val SUBSTRING_SEARCH_FUNCTIONS = setOf("CONTAINS", "STARTSWITH", "ENDSWITH")

        /** Selectivity of a substring search (CONTAINS/STARTSWITH/ENDSWITH) on a free-text column (almost all distinct values). */
        private const val SUBSTRING_SELECTIVITY = 0.05

        /** Selectivity of a substring search on a categorical column (few distinct values), where matches select whole categories. */
        private const val CATEGORICAL_SUBSTRING_SELECTIVITY = 0.2

        /** Columns with fewer distinct values than this are treated as categorical. */
        private const val CATEGORICAL_MAX_NDV = 1000.0

        /**
         * Selectivity of a substring search when the column's NDV is unknown.
         */
        private const val UNKNOWN_NDV_SUBSTRING_SELECTIVITY = 0.15

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
                sel *= ndvEqualitySelectivity(rel, mq, pred)
                    ?: substringSelectivity(rel, mq, pred)
                    ?: guessSelectivity(pred)
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
         * Selectivity of `CONTAINS/STARTSWITH/ENDSWITH(<col>, ...)` (what `LIKE '%abc%'`, `'abc%'`
         * and `'%abc'` simplify to) or their negation, using the column's NDV on [rel] (when known).
         * Returns null if [pred] is not a substring search for a literal pattern (e.g. a
         * column-to-column join condition like `STARTSWITH(a.x, b.y)`).
         */
        private fun substringSelectivity(
            rel: RelNode?,
            mq: RelMetadataQuery?,
            pred: RexNode,
        ): Double? {
            val search = if (pred.kind == SqlKind.NOT && pred is RexCall) pred.operands[0] else pred
            if (search !is RexCall ||
                search.operands.size != 2 ||
                search.operator.name.uppercase() !in SUBSTRING_SEARCH_FUNCTIONS ||
                !SqlTypeUtil.inCharFamily(search.operands[0].type) ||
                RexUtil.removeCast(search.operands[1]) !is RexLiteral
            ) {
                return null
            }
            val column = RexUtil.removeCast(search.operands[0])
            val ndv =
                if (rel != null && mq != null && column is RexInputRef) {
                    columnDistinctCount(rel, mq, column.index)
                } else {
                    null
                }
            val sel =
                when {
                    ndv == null || ndv.isNaN() -> UNKNOWN_NDV_SUBSTRING_SELECTIVITY
                    // Never below the 1/NDV estimate for an equality on the same column.
                    ndv < CATEGORICAL_MAX_NDV -> max(CATEGORICAL_SUBSTRING_SELECTIVITY, 1.0 / max(ndv, 1.0))
                    else -> SUBSTRING_SELECTIVITY
                }
            return if (search !== pred) 1.0 - sel else sel
        }

        /**
         * NDV of column [index] of [rel]. For a Join, asks the input that owns the column instead:
         * the Join's own NDV depends on its row count, which depends on the selectivity of its
         * condition, so asking the Join causes a CyclicMetadataException.
         */
        private fun columnDistinctCount(
            rel: RelNode,
            mq: RelMetadataQuery,
            index: Int,
        ): Double? {
            if (rel is Join) {
                val leftCount = rel.left.rowType.fieldCount
                return if (index < leftCount) {
                    columnDistinctCount(rel.left, mq, index)
                } else {
                    columnDistinctCount(rel.right, mq, index - leftCount)
                }
            }
            return mq.getDistinctRowCount(rel, ImmutableBitSet.of(index), null)
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
                val substringSel = substringSelectivity(null, null, pred)
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
                } else if (substringSel != null) {
                    sel *= substringSel
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
