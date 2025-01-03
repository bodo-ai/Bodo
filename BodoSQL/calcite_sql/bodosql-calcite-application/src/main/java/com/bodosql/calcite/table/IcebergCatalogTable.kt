package com.bodosql.calcite.table

import com.bodosql.calcite.adapter.iceberg.IcebergTableScan
import com.bodosql.calcite.application.PythonLoggers
import com.bodosql.calcite.application.write.IcebergWriteTarget
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.catalog.IcebergCatalog
import com.bodosql.calcite.ddl.DDLExecutor
import com.bodosql.calcite.ddl.IcebergDDLExecutor
import com.bodosql.calcite.ir.Variable
import com.google.common.base.Supplier
import com.google.common.base.Suppliers
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.schema.Statistic
import org.apache.iceberg.catalog.Catalog
import org.apache.iceberg.catalog.SupportsNamespaces
import java.util.Locale

class IcebergCatalogTable<T>(
    name: String,
    schemaPath: ImmutableList<String>,
    columns: List<BodoSQLColumn>,
    private val catalog: IcebergCatalog<T>,
) : CatalogTable(
        name,
        schemaPath,
        columns,
        catalog,
    ) where T : Catalog, T : SupportsNamespaces {
    // Hold the statistics for this table.
    private val statistic: Statistic = StatisticImpl()

    /** Interface to get the Iceberg Catalog.  */
    override fun getCatalog(): IcebergCatalog<T> = catalog

    override fun toRel(
        toRelContext: RelOptTable.ToRelContext,
        relOptTable: RelOptTable,
    ): RelNode {
        val baseRelNode: RelNode = IcebergTableScan.create(toRelContext.cluster, relOptTable, this)
        if (canSafelyInlineView()) {
            val viewDefinition = getViewDefinitionString()
            if (viewDefinition != null) {
                return tryInlineView(toRelContext, viewDefinition, baseRelNode)
            } else {
                val message =
                    String.format(
                        Locale.ROOT,
                        "Unable to inline view %s because we cannot determine its definition.",
                        getQualifiedName(),
                    )
                PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(message)
            }
        } else if (isAccessibleView()) {
            if (isSecureView()) {
                val message =
                    String.format(
                        Locale.ROOT,
                        "Unable to inline view %s because it is a secure view.",
                        getQualifiedName(),
                    )
                PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(message)
            } else if (isMaterializedView()) {
                val message =
                    String.format(
                        Locale.ROOT,
                        "Unable to inline view %s because it is a materialized view.",
                        getQualifiedName(),
                    )
                PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(message)
            }
        }
        return baseRelNode
    }

    override fun getStatistic(): Statistic = statistic

    /**
     * Get the insert into write target for a particular table.
     * This should always return an IcebergWriteTarget.
     * @param columnNamesGlobal The global variable containing the column names. This should
     *                          be possible to remove in the future since we append to a table.
     * @return The IcebergWriteTarget.
     */
    override fun getInsertIntoWriteTarget(columnNamesGlobal: Variable): WriteTarget =
        IcebergWriteTarget(
            name,
            parentFullPath,
            WriteTarget.IfExistsBehavior.APPEND,
            columnNamesGlobal,
            generatePythonConnStr(parentFullPath),
        )

    override fun getDDLExecutor(): DDLExecutor = IcebergDDLExecutor(catalog.getIcebergConnection())

    private val columnDistinctCount =
        com.bodosql.calcite.application.utils.Memoizer.memoize<Int, Double?> { column: Int ->
            this.estimateColumnDistinctCount(
                column,
            )
        }

    private fun estimateColumnDistinctCount(column: Int): Double? {
        // Currently, the metadata restriction scan does not explicitly ban requesting approximate
        // NDV values via the metadata if they are available, but in future should be used to prevent
        // unnecessary sampling as a fallback when NDV is not immediately available. If we do this,
        // then such a check should be added in this location if estimateIcebergTableColumnDistinctCount
        // fails to find an answer.
        return catalog.estimateIcebergTableColumnDistinctCount(parentFullPath, name, column)
    }

    /**
     * Determine the estimated approximate number of distinct values for the column. This value is
     * memoized.
     *
     * @return Estimated distinct count for this table.
     */
    override fun getColumnDistinctCount(column: Int): Double? = columnDistinctCount.apply(column)

    private inner class StatisticImpl : Statistic {
        private val rowCount: Supplier<Double?> = Suppliers.memoize { estimateRowCount() }

        /**
         * Retrieves the estimated row count for this table. This value is memoized.
         *
         * @return estimated row count for this table.
         */
        override fun getRowCount(): Double? = rowCount.get()

        /**
         * Retrieves the estimated row count for this table. It performs a query every time this is
         * invoked.
         *
         * @return estimated row count for this table.
         */
        private fun estimateRowCount(): Double? = catalog.estimateIcebergTableRowCount(parentFullPath, name)
    }
}
