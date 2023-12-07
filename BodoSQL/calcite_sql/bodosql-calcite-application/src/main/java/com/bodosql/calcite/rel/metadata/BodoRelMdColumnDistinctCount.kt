package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter
import com.bodosql.calcite.application.operatorTables.StringOperatorTable
import com.bodosql.calcite.application.utils.IsScalar
import com.bodosql.calcite.rel.core.Flatten
import org.apache.calcite.plan.volcano.RelSubset
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.SingleRel
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Join
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.core.Union
import org.apache.calcite.rel.metadata.MetadataDef
import org.apache.calcite.rel.metadata.MetadataHandler
import org.apache.calcite.rel.metadata.ReflectiveRelMetadataProvider
import org.apache.calcite.rel.metadata.RelMetadataQuery
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexLiteral
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.`fun`.SqlStdOperatorTable
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.ImmutableBitSet

class BodoRelMdColumnDistinctCount : MetadataHandler<ColumnDistinctCount> {
    override fun getDef(): MetadataDef<ColumnDistinctCount> {
        return ColumnDistinctCount.DEF
    }

    /** Catch-all implementation for
     * [ColumnDistinctCount.getColumnDistinctCount],
     * invoked using reflection.
     *
     * By default, we only return information is we can ensure the column is unique.
     *
     * @see ColumnDistinctCount
     */
    fun getColumnDistinctCount(rel: RelNode, mq: RelMetadataQuery, column: Int): Double? {
        val isUnique = mq.areColumnsUnique(rel, ImmutableBitSet.of(column))
        return if (isUnique != null && isUnique) {
            mq.getRowCount(rel)
        } else {
            null
        }
    }

    fun getColumnDistinctCount(subset: RelSubset, mq: RelMetadataQuery, column: Int): Double? {
        return (mq as BodoRelMetadataQuery).getColumnDistinctCount(subset.getBestOrOriginal(), column)
    }

    fun getColumnDistinctCount(rel: Union, mq: RelMetadataQuery, column: Int): Double? {
        val distinctCount = getColumnDistinctCount(rel as RelNode, mq, column)
        return if (distinctCount != null) {
            distinctCount
        } else {
            // Assume the worst case that all groups overlap, so we must take the max of any input.
            rel.inputs.map { r -> (mq as BodoRelMetadataQuery).getColumnDistinctCount(r, column) }.reduce { a, b ->
                if (a == null || b == null) {
                    null
                } else {
                    kotlin.math.max(a, b)
                }
            }
        }
    }

    fun getColumnDistinctCount(rel: Filter, mq: RelMetadataQuery, column: Int): Double? {
        val distinctCount = getColumnDistinctCount(rel as RelNode, mq, column)
        return if (distinctCount != null) {
            distinctCount
        } else {
            // For default filters assume the ratio remains the same after filtering.
            val distinctInput = (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, column)
            val ratio = mq.getRowCount(rel) / mq.getRowCount(rel.input)
            return distinctInput?.let { maxOf(distinctInput.times(ratio), 1.0) }
        }
    }

    /**
     * Attempts to infer the distinctiveness of a call to CAST based on the distinctiveness
     * of its input.
     *
     * @param rel The original projection containing this rex node
     * @param rex The value being casted
     * @param mq The metadata query handler
     * @param targetType The type that rex gets casted to
     * @return The number of distinct rows produced by rex when casted, or null if we cannot infer it.
     */
    private fun inferCastDistinctiveness(rel: Project, rex: RexNode, mq: RelMetadataQuery, targetType: SqlTypeName): Double? {
        // For certain types, the output always matches the input's distinctiveness
        return when (targetType) {
            SqlTypeName.TIMESTAMP,
            SqlTypeName.TINYINT,
            SqlTypeName.SMALLINT,
            SqlTypeName.INTEGER,
            SqlTypeName.BIGINT,
            SqlTypeName.DECIMAL,
            SqlTypeName.FLOAT,
            SqlTypeName.REAL,
            SqlTypeName.DOUBLE,
            ->
                inferRexDistinctness(rel, rex, mq)
            else -> null
        }
    }

    /**
     * Infer the distinctiveness for concat. Currently, we only support the case where
     * all literals are being appended to at most 1 column containing compute. We do not attempt to
     * make any estimations as to how concatenating multiple columns impacts uniqueness.
     *
     * [BSE-2213] Investigate adding distinctness propagation for more BodoSQL functions.
     *
     * @param rel The original projection containing this rex node
     * @param operands The arguments being passed to the concat function.
     * @param mq The metadata query handler
     * @return The number of distinct rows produced by rex when concatenated, or null if we cannot infer it.
     */
    private fun inferConcatDistinctiveness(rel: Project, operands: List<RexNode>, mq: RelMetadataQuery): Double? {
        var index = -1
        for ((i, operand) in operands.withIndex()) {
            if (operand !is RexLiteral) {
                if (index != -1) {
                    // Multiple columns encountered. Return NULL.
                    return null
                }
                index = i
            }
        }
        // If its all literals there is a singular unique value.
        if (index == -1) {
            return 1.0
        }
        return inferRexDistinctness(rel, operands[index], mq)
    }

    /**
     * Attempts to infer the distinctiveness of a RexNode by inferring the distinctiveness
     * of the Input Ref(s) involved and either passing it through unmodified, or transforming
     * it somehow if it goes through a function call.
     *
     * @param rel The original projection containing this rex node
     * @param rex The rex node whose distinctiveness information we are trying to infer
     * @param mq The metadata query handler
     * @return The number of distinct rows produced by rex, or null if we cannot infer it.
     */
    private fun inferRexDistinctness(rel: Project, rex: RexNode, mq: RelMetadataQuery): Double? {
        // Base case: known scalar values have only 1 distinct value
        if (rex.accept(IsScalar())) { return 1.0 }
        // Base case: once an InputRef is reached, its distinctiveness information is calculated
        // so that it can be propagated upward.
        if (rex is RexInputRef) {
            return (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, rex.index)
        }
        if (rex is RexCall) {
            if (rex.kind == SqlKind.CAST) {
                return inferCastDistinctiveness(rel, rex.operands[0], mq, rex.type.sqlTypeName)
            } else if (rex.kind == SqlKind.OTHER_FUNCTION) {
                val concatFunctions = listOf(StringOperatorTable.CONCAT.name, StringOperatorTable.CONCAT_WS.name)
                if (concatFunctions.contains(rex.operator.name)) {
                    return inferConcatDistinctiveness(rel, rex.operands, mq)
                }
            } else if (rex.kind == SqlKind.OTHER) {
                if (rex.operator.name == SqlStdOperatorTable.CONCAT.name) {
                    return inferConcatDistinctiveness(rel, rex.operands, mq)
                }
            }
        }
        return null
    }

    fun getColumnDistinctCount(rel: Project, mq: RelMetadataQuery, column: Int): Double? {
        val distinctCount = getColumnDistinctCount(rel as RelNode, mq, column)
        return if (distinctCount != null) {
            distinctCount
        } else {
            val columnNode = rel.projects[column]
            return inferRexDistinctness(rel, columnNode, mq)
        }
    }

    fun getColumnDistinctCount(rel: SingleRel, mq: RelMetadataQuery, column: Int): Double? {
        return (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, column)
    }

    fun getColumnDistinctCount(rel: Join, mq: RelMetadataQuery, column: Int): Double? {
        val distinctCount = getColumnDistinctCount(rel as RelNode, mq, column)
        return if (distinctCount != null) {
            distinctCount
        } else {
            val leftCount = rel.left.getRowType().fieldCount
            val isLeftInput = column < leftCount
            // For join assume an unchanged ratio and fetch the inputs.
            val input = if (isLeftInput) {
                rel.left
            } else {
                rel.right
            }
            val inputColumn = if (isLeftInput) {
                column
            } else {
                column - leftCount
            }
            // 1.0 if the join can create nulls in this column, otherwise 0.0.
            val extraValue =
                if ((isLeftInput && rel.joinType.generatesNullsOnLeft()) ||
                    (!isLeftInput && rel.joinType.generatesNullsOnRight())
                ) { 1.0 } else { 0.0 }

            val distinctInput = (mq as BodoRelMetadataQuery).getColumnDistinctCount(input, inputColumn)
            val expectedRowCount = mq.getRowCount(rel)
            // If we have an outer join you cannot decrease the number of distinct values
            // unless you add NULL.
            return if ((isLeftInput && rel.joinType.generatesNullsOnRight()) || (!isLeftInput && rel.joinType.generatesNullsOnLeft())) {
                // Note: Add a sanity check that we never exceed the expected row count.
                distinctInput?.let { minOf(distinctInput + extraValue, expectedRowCount) }
            } else {
                // Assume the ratio remains the same after filtering with the caveat that the number
                // of distinct rows cannot increase as a result of joining, except for one new value
                // that could be introduced as the result of creating nulls.
                val ratio = minOf(expectedRowCount / mq.getRowCount(input), 1.0)
                distinctInput?.let { maxOf(distinctInput.times(ratio), 1.0) + extraValue }
            }
        }
    }

    fun getColumnDistinctCount(rel: Aggregate, mq: RelMetadataQuery, column: Int): Double? {
        val distinctCount = getColumnDistinctCount(rel as RelNode, mq, column)
        return if (distinctCount != null) {
            distinctCount
        } else {
            val groupSetList = rel.groupSet.asList()
            return if (groupSetList.size == 0) {
                1.0
            } else if (column >= rel.groupSet.asList().size && rel.aggCallList[column - rel.groupSet.asList().size].aggregation.kind == SqlKind.LITERAL_AGG) {
                // A LITERAL_AGG is always a single value.
                return 1.0
            } else if (rel.groupSets.size != 1 || column >= rel.groupSet.asList().size) {
                return null
            } else {
                // The number of distinct rows in any grouping key the same as the input
                val inputColumn = rel.groupSet.asList()[column]
                val distinctInput = (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, inputColumn)
                val outRows = mq.getRowCount(rel)
                return distinctInput?.let { minOf(it, outRows) }
            }
        }
    }

    fun getColumnDistinctCount(rel: SnowflakeToPandasConverter, mq: RelMetadataQuery, column: Int): Double? {
        return (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, column)
    }

    fun getColumnDistinctCount(rel: SnowflakeTableScan, mq: RelMetadataQuery, column: Int): Double? {
        val trueCol = rel.keptColumns.nth(column)
        return rel.getCatalogTable().getColumnDistinctCount(trueCol)
    }

    fun getColumnDistinctCount(rel: Flatten, mq: RelMetadataQuery, column: Int): Double? {
        val nonNullEstimate = if (rel.rowType.fieldList.get(column).type.isNullable()) { 0.9 } else { 1.0 }
        val offset = rel.usedColOutputs.cardinality()
        return if (column >= offset) {
            (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, rel.repeatColumns.nth(column - offset))?.times(nonNullEstimate)
        } else {
            null
        }
    }

    companion object {
        val SOURCE = ReflectiveRelMetadataProvider.reflectiveSource(
            BodoRelMdColumnDistinctCount(),
            ColumnDistinctCount.Handler::class.java,
        )
    }
}
