package com.bodosql.calcite.table

import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan.Companion.create
import com.bodosql.calcite.application.PythonLoggers
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.write.SnowflakeNativeWriteTarget
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.catalog.SnowflakeCatalog
import com.bodosql.calcite.ddl.DDLExecutor
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.rel.metadata.BodoMetadataRestrictionScan.Companion.canRequestColumnDistinctiveness
import com.google.common.base.Supplier
import com.google.common.base.Suppliers
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.schema.Statistic
import org.apache.calcite.sql.util.SqlString
import java.util.Locale

/**
 * Implementation of CatalogTable for Snowflake Catalogs.
 *
 * Note: This class is open, so we can extend it in testing.
 *
 */
open class SnowflakeCatalogTable(
    name: String,
    schemaPath: ImmutableList<String>,
    columns: List<BodoSQLColumn>,
    private val catalog: SnowflakeCatalog,
) : CatalogTable(name, schemaPath, columns, catalog) {
    // Hold the statistics for this table.
    private val statistic: Statistic = StatisticImpl()

    /** Interface to get the Snowflake Catalog.  */
    override fun getCatalog(): SnowflakeCatalog = catalog

    /**
     *
     * Check within the catalog if we have read access.
     *
     * @return Do we have read access?
     */
    override fun canRead(): Boolean = this.catalog.canReadTable(fullPath)

    private val columnDistinctCount =
        com.bodosql.calcite.application.utils.Memoizer.memoize<Int, Double?> { column: Int ->
            this.estimateColumnDistinctCount(
                column,
            )
        }

    /**
     * Determine the estimated approximate number of distinct values for the column. This value is
     * memoized.
     *
     * @return Estimated distinct count for this table.
     */
    override fun getColumnDistinctCount(column: Int): Double? = columnDistinctCount.apply(column)

    /**
     * Estimate the distinct count for a column by submitting an APPROX_COUNT_DISTINCT call to
     * Snowflake. This value is capped to be no greater than the row count, which Snowflake can
     * violate.
     *
     * @param column The column to check.
     * @return The approximate distinct count. Returns null if there is a timeout.
     */
    private fun estimateColumnDistinctCount(column: Int): Double? {
        val columnName = columnNames[column].uppercase()
        // Do not allow the metadata to be requested if this column of this table was
        // not pre-cleared by the metadata scanning pass.
        if (!canRequestColumnDistinctiveness(fullPath, columnName)) {
            val message =
                String.format(
                    Locale.ROOT,
                    "Skipping attempt to fetch column '%s' from table '%s' due to metadata restrictions",
                    columnName,
                    getQualifiedName(),
                )
            PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER.warning(message)
            return null
        }
        // Avoid ever returning more than the row count. This can happen because
        // Snowflake returns an estimate.
        var distinctCount = catalog.estimateColumnDistinctCount(fullPath, columnName)

        // If the original query failed, try again with sampling
        if (distinctCount == null) {
            distinctCount =
                catalog.estimateColumnDistinctCountWithSampling(
                    fullPath,
                    columnName,
                    statistic.rowCount,
                )
        }

        // Important: We must use getStatistic() here to allow subclassing, which we use for
        // our mocking infrastructure.
        val maxCount = statistic.rowCount
        return if (distinctCount == null || maxCount == null) {
            null
        } else {
            java.lang.Double.min(distinctCount, maxCount)
        }
    }

    /**
     * Wrappers around submitRowCountQueryEstimateInternal that handles memoization. See
     * submitRowCountQueryEstimateInternal for documentation.
     */
    fun trySubmitIntegerMetadataQuerySnowflake(sql: SqlString): Long? = trySubmitLongMetadataQuerySnowflakeMemoizedFn.apply(sql)

    private val trySubmitLongMetadataQuerySnowflakeMemoizedFn =
        com.bodosql.calcite.application.utils.Memoizer.memoize<SqlString, Long?> { metadataSelectQueryString: SqlString ->
            this.trySubmitLongMetadataQuerySnowflakeInternal(
                metadataSelectQueryString,
            )
        }

    /**
     * Submits a Submits the specified query to Snowflake for evaluation, with a timeout. See
     * SnowflakeCatalog#trySubmitIntegerMetadataQuery for the full documentation.
     * @return A long result from the query or NULL.
     */
    private fun trySubmitLongMetadataQuerySnowflakeInternal(metadataSelectQueryString: SqlString): Long? =
        catalog.trySubmitLongMetadataQuery(metadataSelectQueryString)

    override fun toRel(
        toRelContext: RelOptTable.ToRelContext,
        relOptTable: RelOptTable,
    ): RelNode? {
        val baseRelNode: RelNode = create(toRelContext.cluster, relOptTable, this)
        // Check if this table is a view and if so attempt to inline it.
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

    fun isIcebergTable(): Boolean {
        // Note: This needs to be outside catalog.isIcebergTable so
        // the result isn't cached.
        if (!RelationalAlgebraGenerator.enableSnowflakeIcebergTables) {
            return false
        }
        return catalog.isIcebergTable(this.fullPath)
    }

    /**
     * Get the insert into write target for a particular table.
     * Ideally we would like to write native tables using a native
     * target and iceberg tables with Iceberg, but we don't have
     * Iceberg insert into support yet.
     * @param columnNamesGlobal The global variable containing the column names. This should
     *                          be possible to remove in the future since we append to a table.
     * @return The WriteTarget for the table.
     */
    override fun getInsertIntoWriteTarget(columnNamesGlobal: Variable): WriteTarget =
        SnowflakeNativeWriteTarget(
            name,
            parentFullPath,
            WriteTarget.IfExistsBehavior.APPEND,
            columnNamesGlobal,
            generatePythonConnStr(parentFullPath),
        )

    override fun getDDLExecutor(): DDLExecutor = catalog.ddlExecutor

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
        private fun estimateRowCount(): Double? = catalog.estimateRowCount(fullPath)
    }
}
