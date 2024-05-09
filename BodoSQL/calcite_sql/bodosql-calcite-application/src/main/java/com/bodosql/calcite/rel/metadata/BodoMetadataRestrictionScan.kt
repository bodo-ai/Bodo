package com.bodosql.calcite.rel.metadata

import com.bodosql.calcite.adapter.iceberg.IcebergTableScan
import com.bodosql.calcite.adapter.iceberg.IcebergToBodoPhysicalConverter
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan
import com.bodosql.calcite.adapter.snowflake.SnowflakeToBodoPhysicalConverter
import com.bodosql.calcite.rel.core.Flatten
import com.bodosql.calcite.rel.core.WindowBase
import com.bodosql.calcite.table.CatalogTable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.core.Aggregate
import org.apache.calcite.rel.core.Filter
import org.apache.calcite.rel.core.Project
import org.apache.calcite.rel.core.SetOp
import org.apache.calcite.rel.rules.MultiJoin
import org.apache.calcite.rex.RexCall
import org.apache.calcite.rex.RexInputRef
import org.apache.calcite.rex.RexNode
import org.apache.calcite.sql.SqlKind
import java.util.Locale

/**
 * Pre-scans a query plan to determine which columns should be allowed to have their
 * approximate distinct counts requested from Snowflake. This includes columns that
 * are used as join conditions, but also any columns used as grouping keys or in
 * a SELECT DISTINCT if the output later becomes an input to a join, since the number
 * of rows produced affects the cost of a join.
 *
 * Applying this restriction allows us to prevent the planner from requesting metadata
 * from Snowflake every time we see an aggregation or a SELECT DISTINCT, since there is
 * no point in approximating their row counts.
 *
 * [BSE-1572] TODO: investigate potential usage of RelMdExpressionLineage for this purpose.
 * [BSE-1607] TODO: unify getEquiJoinConditions with the actual logic we use in the
 *            join row counts calculations to reduce our technical debt & ensure that
 *            updating one automatically updates the other.
 * [BSE-1608] TODO: investigate moving the storage of which columns can be requested into
 *            the tables themselves, and potentially have all of the metadata requests come
 *            at the end of the scan so they can be done in batches.
 * [BSE-1609] TODO: altering the conditions under which an aggregation is allowed to request
 *            the metadata for its keys so that stranger join cases don't result in the
 *            aggregation not realizing that it is a descendant of a join.
 */
class BodoMetadataRestrictionScan {
    companion object {
        // Stores all the table columns whose approx distinctness count is allowed
        // to be requested based on the most recent call to scanForRequestableMetadata.
        // Columns stored as strings in the format "schema_name.table_name.column_name".
        private val columnsAllowedToRequest: MutableSet<String> = mutableSetOf()

        /**
         * Resets the set of columns that are allowed to be requested, for testing purposes.
         */
        fun resetRestrictionScan() {
            columnsAllowedToRequest.clear()
        }

        /**
         * Pre-scan the query plan to determine which columns of which tables will need their
         * metadata scanned in order to estimate the row counts of joins. This includes columns
         * used for equi-joins, and also grouping keys when the output of the aggregation node
         * later becomes the input to a join.
         *
         * @param node The root node of the query plan
         */
        fun scanForRequestableMetadata(node: RelNode) {
            columnsAllowedToRequest.clear()
            findColumnsThatCanBeRequested(node, setOf())
        }

        /**
         * Determines whether a metadata request for the approximate number of distinct columns
         * should be allowed to proceed. This is based solely on the most recent call to
         * scanForRequestableMetadata.
         *
         * TODO: in future, investigate if we can embed this information in the scan nodes
         * instead of this class' companion object.
         *
         * @param tableName The schema/table names of the table being requested
         * @param columnName The name of the column being requested
         * @return Whether the column is allowed to have its metadata requested
         */
        fun canRequestColumnDistinctiveness(
            tableName: List<String>,
            columnName: String,
        ): Boolean {
            return columnsAllowedToRequest.contains("${tableName.joinToString(".")}.$columnName".uppercase(Locale.ROOT))
        }

        /**
         * Takes in a RexNode and determines whether it corresponds to a column
         * from the child RelNode of the RelNode containing this RexNode. For example,
         * if we have the following projection node:
         *
         * Project(A=$1, B=+($0, $2), C=CONCAT('%', LEFT($5, 10), '%'))
         *
         * If we called getColumnForDistinctness on the first term it would return 1
         * since column 0 of the current RelNode can have its metadata inferred by
         * knowing the metadata of column 1. If we called it on the second term it
         * would return null since there is no one column we can use to infer the
         * metadata of +($0, $2). If we called it on the third term it would return
         * 5 since the distinctness of the entire term could potentially be inferred
         * just the knowing the distinctness of column $5.
         *
         * @param node The RexNode being searched for a column from the input RelNode
         * that should have its metadata requested.
         * @return The column index if there is one, otherwise null.
         */
        private fun getColumnForDistinctness(node: RexNode): Int? {
            if (node is RexInputRef) return node.index
            if (node is RexCall) {
                val operandDistinctnessCols = node.operands.map { getColumnForDistinctness(it) }
                val nonNullCols = operandDistinctnessCols.filterNotNull()
                if (nonNullCols.size == 1) return nonNullCols[0]
            }
            return null
        }

        /**
         * Takes in a list of conditions for a join and extracts the column indices
         * for columns used as equi-join conditions.
         *
         * @param conditions The list of conditions for the join
         * @return The list of column indices that correspond to equi-join conditions
         */
        private fun getEquiJoinConditions(conditions: List<RexNode?>): List<Int> {
            val equiJoins: MutableList<Int> = mutableListOf()
            conditions.forEach {
                it?.let {
                    // If the condition is an AND clause, recursively repeat the
                    // process on all of its operands.
                    if (it is RexCall && it.kind == SqlKind.AND) {
                        equiJoins.addAll(getEquiJoinConditions(it.operands))
                    } else if (it is RexCall && it.kind == SqlKind.EQUALS) {
                        // Otherwise, if the condition is an equality condition,
                        // add the corresponding columns to the list (if they
                        // are inferable from a single column).
                        it.operands.forEach { operand ->
                            getColumnForDistinctness(operand)?.let {
                                    col ->
                                equiJoins.add(col)
                            }
                        }
                    }
                }
            }
            return equiJoins
        }

        /**
         * Recursively scans through the query plan tree to find
         * all columns that will need their metadata requested to find
         * the row counts for any join ancestors of the current RelNode.
         *
         * @param node The current RelNode being analyzed
         * @param cols The set of column indices that must have their
         *
         */
        fun findColumnsThatCanBeRequested(
            node: RelNode,
            cols: Set<Int>,
        ) {
            if (node is SnowflakeTableScan || node is IcebergTableScan) {
                // Once we have reached a scan, add every column from cols to
                // the set of columns that are pre-cleared for metadata requests.
                val table: CatalogTable =
                    if (node is SnowflakeTableScan) {
                        node.getCatalogTable()
                    } else {
                        (node as IcebergTableScan).getCatalogTable()
                    }
                val names = node.getRowType().fieldNames
                cols.forEach {
                    val columnName = names[it]
                    val tablePath = table.fullPath.joinToString(".")
                    columnsAllowedToRequest.add("$tablePath.$columnName".uppercase(Locale.ROOT))
                }
            } else if (node is SetOp || node is Filter ||
                node is SnowflakeToBodoPhysicalConverter || node is IcebergToBodoPhysicalConverter
            ) {
                // For these types of RelNodes, forward the scan onto all the children un-modified.
                node.inputs.forEach { findColumnsThatCanBeRequested(it, cols) }
            } else if (node is Aggregate) {
                // When reaching an aggregate, the metadata of all of its group keys is needed if
                // its output row count is used to infer a join size, which we know based on
                // whether the col set is non-empty.
                val newCols: MutableSet<Int> = mutableSetOf()
                if (cols.isNotEmpty()) {
                    node.groupSet.forEach { newCols.add(it) }
                }
                findColumnsThatCanBeRequested(node.inputs[0], newCols)
            } else if (node is Project) {
                // For a projection, map each column from cols that we want to request
                // to the corresponding column from the child RelNode.
                val newCols: MutableSet<Int> = mutableSetOf()
                if (cols.isNotEmpty()) {
                    cols.forEach {
                        val proj = node.projects[it]
                        getColumnForDistinctness(proj)?.let { col ->
                            newCols.add(col)
                        }
                    }
                }
                findColumnsThatCanBeRequested(node.inputs[0], newCols)
            } else if (node is WindowBase) {
                // For a window, forward all distinctness requests that do not
                // come from a window argument.
                val nPassThroughCols = node.inputsToKeep.cardinality()
                val newCols = cols.filter { it < nPassThroughCols }.toSet()
                findColumnsThatCanBeRequested(node.inputs[0], newCols)
            } else if (node is MultiJoin) {
                /**
                 * Suppose we have a multi join with 3 children:
                 *
                 * MultiJoin(condition=[AND(=($0, $2), =($2, $7))])
                 *      Node A: has 2 columns
                 *      Node B: has 3 columns
                 *      Node C: has 4 columns
                 *
                 * If findColumnsThatCanBeRequested was called on this
                 * node with columns {0, 1, 3, 6, 8}:
                 *
                 * - We would want the distinctness info from columns
                 *   0, 1, 3, 6 and 8 for the ancestors, but also from
                 *   columns 0, 2 and 7 because they are equi-join conditions.
                 * - Columns 0 and 1 are from Node A
                 * - Columns 2, 3 and 4 are from Node B
                 * - Columns 5, 6, 7 and 8 are from Node C
                 *
                 * Therefore, we would send the propagate the following requests
                 * downward to the child nodes:
                 *      - A: 0 and 1
                 *      - B: 0 and 1 (remapped from 2 and 3)
                 *      - C: 1, 2 and 3 (remapped from 6, 7 and 8)
                 *
                 */
                val inputs = node.inputs
                val sizes = inputs.map { it.rowType.fieldCount }
                // Find the cutoffs for the column indices corresponding to each of
                // the inputs to a MultiJoin. E.g. if a MultiJoin has 3 inputs with
                // 4, 11 and 5 columns, then cumeSizes = <0, 4, 15, 20>
                val cumeSizes =
                    sizes.fold(listOf(0)) { x, y ->
                        if (x.isEmpty()) {
                            listOf(y)
                        } else {
                            x + (x.last() + y)
                        }
                    }
                // Find all the columns that will need their metadata requested in order
                // to infer the output row count of this join
                val equiJoinCols =
                    getEquiJoinConditions(node.outerJoinConditions) + getEquiJoinConditions(listOf(node.joinFilter))
                val targetCols = equiJoinCols + cols
                // For each input, recursively repeat the procedure on each element of cols or
                // equiJoinCols that is in the relevant range of column indices for the input.
                inputs.forEachIndexed { idx, rel ->
                    val lo = cumeSizes[idx]
                    val hi = cumeSizes[idx + 1]
                    val subsetOfCols = targetCols.filter { it in lo until hi }.map { it - lo }
                    findColumnsThatCanBeRequested(rel, subsetOfCols.toSet())
                }
            } else if (node is Flatten) {
                // For a flatten, map each column from cols that we want to request
                // to the corresponding input of the flatten node, omitting any columns
                // that are produced by the flatten node itself.
                val newCols: MutableSet<Int> = mutableSetOf()
                val offset = node.usedColOutputs.cardinality()
                val usedInputs = node.repeatColumns.toList()
                usedInputs.mapIndexed {
                        idx, value ->
                    if (cols.contains(idx + offset)) {
                        newCols.add(value)
                    }
                }
                findColumnsThatCanBeRequested(node.inputs[0], newCols)
            } else {
                // For any other type of RelNode, clear the set of columns to be requested. We still
                // make the recursive call in case the descendants contain joins, which will re-add
                // columns to the set.
                node.inputs.forEach { findColumnsThatCanBeRequested(it, mutableSetOf()) }
            }
        }
    }
}
