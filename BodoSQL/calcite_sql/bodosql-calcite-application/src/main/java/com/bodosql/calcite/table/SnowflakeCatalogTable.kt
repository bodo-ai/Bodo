package com.bodosql.calcite.table

import com.bodosql.calcite.adapter.pandas.logicalProject
import com.bodosql.calcite.adapter.snowflake.SnowflakeTableScan.Companion.create
import com.bodosql.calcite.application.PythonLoggers
import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.application.utils.checkTablePermissions.Companion.canRead
import com.bodosql.calcite.catalog.SnowflakeCatalogImpl
import com.bodosql.calcite.rel.metadata.BodoMetadataRestrictionScan.Companion.canRequestColumnDistinctiveness
import com.bodosql.calcite.schema.ExpandViewInput
import com.bodosql.calcite.schema.InlineViewMetadata
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.sql.util.SqlString
import java.util.*

/**
 * Implementation of CatalogTable for Snowflake Catalogs.
 *
 */
public open class SnowflakeCatalogTable(
    name: String,
    schemaPath: ImmutableList<String>,
    columns: List<BodoSQLColumn>,
    private val catalog: SnowflakeCatalogImpl,
) :
    CatalogTable(name, schemaPath, columns, catalog) {

    /** Interface to get the Snowflake Catalog.  */
    override fun getCatalog(): SnowflakeCatalogImpl {
        return catalog
    }

    /**
     * Check within the catalog if we have read access.
     *
     * @return Do we have read access?
     */
    override fun canRead(): Boolean {
        return this.catalog.canReadTable(fullPath)
    }

    private val columnDistinctCount = com.bodosql.calcite.application.utils.Memoizer.memoize<Int, Double?> { column: Int ->
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
    fun getColumnDistinctCount(column: Int): Double? {
        return columnDistinctCount.apply(column)
    }

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
            val message = String.format(
                Locale.ROOT,
                "Skipping attempt to fetch column '%s' from table '%s' due to metadata restrictions",
                columnName,
                qualifiedName,
            )
            PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER.warning(message)
            return null
        }
        // Avoid ever returning more than the row count. This can happen because
        // Snowflake returns an estimate.
        var distinctCount = catalog.estimateColumnDistinctCount(fullPath, columnName)

        // If the original query failed, try again with sampling
        if (distinctCount == null) {
            distinctCount = catalog.estimateColumnDistinctCountWithSampling(
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
    fun trySubmitIntegerMetadataQuerySnowflake(sql: SqlString): Long? {
        return trySubmitLongMetadataQuerySnowflakeMemoizedFn.apply(sql)
    }

    private val trySubmitLongMetadataQuerySnowflakeMemoizedFn = com.bodosql.calcite.application.utils.Memoizer.memoize<SqlString, Long?> { metadataSelectQueryString: SqlString ->
        this.trySubmitLongMetadataQuerySnowflakeInternal(
            metadataSelectQueryString,
        )
    }

    /**
     * Submits a Submits the specified query to Snowflake for evaluation, with a timeout. See
     * SnowflakeCatalogImpl#trySubmitIntegerMetadataQuery for the full documentation.
     * @return A long result from the query or NULL.
     */
    fun trySubmitLongMetadataQuerySnowflakeInternal(
        metadataSelectQueryString: SqlString,
    ): Long? {
        return catalog.trySubmitLongMetadataQuery(metadataSelectQueryString)
    }

    /**
     * Load the view metadata information from the catalog. If the table is not a view or no
     * information can be found this should return NULL. This should be used to implement
     * isAccessibleView(), canSafelyInlineView(), and getViewDefinitionString().
     *
     *
     * This is currently only support for Snowflake catalogs.
     *
     * @return The InlineViewMetadata loaded from the catalog or null if no information is available.
     */
    private fun tryGetViewMetadata(): InlineViewMetadata? {
        return catalog.tryGetViewMetadata(fullPath)
    }

    // Cache used for inlining views. We cannot use the Memoizer here because the
    // ToRelContext doesn't have .equals() properly defined. Here we know that
    // all uses of the same CatalogTable are safe to cache.
    private val inlineViewCache: MutableMap<ExpandViewInput, RelNode?> = HashMap()

    /**
     * Inline a view. If this inlining is not possible return Null.
     *
     * @param toRelContext The context used for expanding the view.
     * @param input The inputs used to call toRelContext.expandView(). This is grouped into one object
     * for caching purposes.
     * @return The RelNode after expanding the view or NULL.
     */
    private fun inlineViewImpl(
        toRelContext: RelOptTable.ToRelContext,
        input: ExpandViewInput,
    ): RelNode? {
        try {
            val root = toRelContext.expandView(
                input.outputType,
                input.viewDefinition,
                input.defaultPath,
                input.viewPath,
            )
            val rel = root.logicalProject()
            // Verify that we can read before inlining.
            if (canRead(rel)) {
                return rel
            }
        } catch (e: Exception) {
            // Log the failure
            val message = String.format(
                Locale.ROOT,
                """
              Unable to expand view %s with definition:
              %s. Error encountered when compiling view:
              %s
                """.trimIndent(),
                qualifiedName,
                input.viewDefinition,
                e.message,
            )
            PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.warning(message)
        }
        return null
    }

    /**
     * Try to inline a view. If the view cannot be inlined then return the baseRelNode instead.
     *
     * @param toRelContext The context used to expand a view.
     * @param viewDefinition The view definition.
     * @param baseRelNode The RelNode generated if inlining this view fails.
     * @return Either the new tree generated from inlining a view or the baseRelNode.
     */
    private fun tryInlineView(
        toRelContext: RelOptTable.ToRelContext,
        viewDefinition: String,
        baseRelNode: RelNode,
    ): RelNode? {
        val input = ExpandViewInput(
            baseRelNode.rowType,
            viewDefinition,
            // TODO: FIXME to avoid dependence on the catalog
            listOf(catalog.catalogName, parentFullPath[0]),
            listOf(catalog.catalogName, parentFullPath[0], name),
        )
        // Check the cache.
        val result: RelNode?
        if (inlineViewCache.containsKey(input)) {
            result = inlineViewCache[input]
        } else {
            result = inlineViewImpl(toRelContext, input)
            // Store in the cache
            inlineViewCache[input] = result
        }
        return if (result != null) {
            // Log that we inlined the view.
            val levelOneMessage = String.format(
                Locale.ROOT,
                "Successfully inlined view %s",
                qualifiedName,
            )
            PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(levelOneMessage)
            val levelTwoMessage = String.format(
                Locale.ROOT,
                "Replaced view %s with definition %s",
                qualifiedName,
                input.viewDefinition,
            )
            PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER.info(levelTwoMessage)
            result
        } else {
            baseRelNode
        }
    }

    override fun toRel(toRelContext: RelOptTable.ToRelContext, relOptTable: RelOptTable?): RelNode? {
        val baseRelNode: RelNode = create(toRelContext.cluster, relOptTable!!, this)
        // Check if this table is a view and if so attempt to inline it.
        if (RelationalAlgebraGenerator.tryInlineViews && canSafelyInlineView()) {
            val viewDefinition = getViewDefinitionString()
            if (viewDefinition != null) {
                return tryInlineView(toRelContext, viewDefinition, baseRelNode)
            } else {
                val message = String.format(
                    Locale.ROOT,
                    "Unable to inline view %s because we cannot determine its definition.",
                    qualifiedName,
                )
                PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(message)
            }
        } else if (isAccessibleView()) {
            if (isSecureView()) {
                val message = String.format(
                    Locale.ROOT,
                    "Unable to inline view %s because it is a secure view.",
                    qualifiedName,
                )
                PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(message)
            } else if (isMaterializedView()) {
                val message = String.format(
                    Locale.ROOT,
                    "Unable to inline view %s because it is a materialized view.",
                    qualifiedName,
                )
                PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(message)
            }
        }
        return baseRelNode
    }

    /**
     * Is this table definitely a view (meaning we can access its definition). If this returns False
     * we may have a view if we lack the permissions necessary to know it is a view.
     *
     * @return True if this is a view for which we can load metadata information.
     */
    private fun isAccessibleView(): Boolean {
        return tryGetViewMetadata() != null
    }

    /**
     * Is this table actually a materialized view.
     *
     * @return True if this table is definitely a materialized view.
     */
    private fun isMaterializedView(): Boolean {
        if (isAccessibleView()) {
            val metadata = tryGetViewMetadata()
            return metadata!!.isMaterialized
        }
        return false
    }

    /**
     * Is this table actually a secure view.
     *
     * @return True if this table is definitely a secure view.
     */
    private fun isSecureView(): Boolean {
        if (isAccessibleView()) {
            val metadata = tryGetViewMetadata()
            return metadata!!.unsafeToInline
        }
        return false
    }

    /**
     * Is this a view that can be safely inlined.
     *
     * @return Returns true if table is a view and the metadata indicates inlining is legal. If this
     * table is not a view this return false.
     */
    private fun canSafelyInlineView(): Boolean {
        return isAccessibleView() && !(isSecureView() || isMaterializedView())
    }

    /**
     * Get the SQL query definition used to define this table if it is a view.
     *
     * @return The string definition that was used to create the view. Returns null if the table is
     * not a view.
     */
    private fun getViewDefinitionString(): String? {
        return if (isAccessibleView()) {
            tryGetViewMetadata()!!.viewDefinition
        } else {
            null
        }
    }
}
