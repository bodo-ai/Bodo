package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.adapter.snowflake.SnowflakeToPandasConverter
import com.bodosql.calcite.application.utils.IsScalar
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
import org.apache.calcite.rex.RexInputRef
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

    fun getColumnDistinctCount(rel: Project, mq: RelMetadataQuery, column: Int): Double? {
        val distinctCount = getColumnDistinctCount(rel as RelNode, mq, column)
        return if (distinctCount != null) {
            distinctCount
        } else {
            // For projects check based on RexNode type.
            val columnNode = rel.projects[column]
            return if (columnNode.accept(IsScalar())) {
                1.0
            } else if (columnNode is RexInputRef) {
                // Input refs are identical to the input.
                (mq as BodoRelMetadataQuery).getColumnDistinctCount(rel.input, columnNode.index)
            } else {
                null
            }
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
                if ((isLeftInput && rel.getJoinType().generatesNullsOnLeft()) ||
                    (!isLeftInput && rel.getJoinType().generatesNullsOnRight())
                ) { 1.0 } else { 0.0 }
            // Assume the ratio remains the same after filtering with the caveat that the number
            // of distinct rows cannot increase as a result of joining, except for one new value
            // that could be introduced as the result of creating nulls.
            val distinctInput = (mq as BodoRelMetadataQuery).getColumnDistinctCount(input, inputColumn)
            val ratio = minOf(mq.getRowCount(rel) / mq.getRowCount(input), 1.0)
            return distinctInput?.let { maxOf(distinctInput.times(ratio), 1.0) + extraValue }
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

    companion object {
        val SOURCE = ReflectiveRelMetadataProvider.reflectiveSource(
            BodoRelMdColumnDistinctCount(),
            ColumnDistinctCount.Handler::class.java,
        )
    }
}
