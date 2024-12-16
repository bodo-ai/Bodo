package com.bodosql.calcite.table

import com.bodosql.calcite.adapter.bodo.StreamingOptions
import com.bodosql.calcite.adapter.bodo.calciteLogicalProject
import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.application.PythonLoggers
import com.bodosql.calcite.application.utils.CheckTablePermissions
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.ddl.DDLExecutor
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.schema.ExpandViewInput
import com.bodosql.calcite.schema.InlineViewMetadata
import com.bodosql.calcite.table.ColumnDataTypeInfo.Companion.fromSqlType
import com.google.common.collect.ImmutableList
import org.apache.calcite.plan.RelOptTable
import org.apache.calcite.rel.RelNode
import org.apache.calcite.rel.type.RelDataTypeField
import org.apache.calcite.schema.Table
import org.apache.calcite.schema.TranslatableTable
import org.apache.calcite.util.BodoStatic
import java.util.Locale

/**
 *
 *
 * <h1>Stores a table with its corresponding columns</h1>
 *
 * @author bodo
 */
open class CatalogTable(
    name: String,
    schemaPath: ImmutableList<String>,
    columns: List<BodoSQLColumn>,
    // The catalog that holds this table's origin.
    private val catalog: BodoSQLCatalog,
) : BodoSqlTable(name, schemaPath, columns),
    TranslatableTable {
    /*
     * See the design described on Confluence:
     * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Table
     */

    /**
     * Return the fully qualified name. This should be of the form
     * "DATABASE_NAME"."SCHEMA_NAME"."TABLE_NAME"
     *
     * @return
     */
    fun getQualifiedName(): String {
        val quotedPath = ImmutableList.Builder<String>()
        for (elem in fullPath) {
            quotedPath.add(String.format(Locale.ROOT, "\"%s\"", elem))
        }

        return quotedPath.build().joinToString(".")
    }

    /** Interface to get the catalog for creating RelNodes. */
    open fun getCatalog(): BodoSQLCatalog = catalog

    /**
     * Can BodoSQL write to this table. By default, this is true but in the future this may be extended
     * to look at the permissions given in the catalog.
     *
     * @return Can BodoSQL write to this table.
     */
    override fun isWriteable(): Boolean {
        // TODO: Update with the ability to check permissions from the schema/catalog
        return true
    }

    /**
     * Generate the code needed to write the given variable to storage. This table type generates code
     * common to all tables in the catalog.
     *
     * @param varName Name of the variable to write.
     * @return The generated code to write the table.
     */
    override fun generateWriteCode(
        visitor: BodoCodeGenVisitor,
        varName: Variable,
    ): Expr = catalog.generateAppendWriteCode(visitor, varName, fullPath)

    /**
     * Generate the code needed to write the given variable to storage.
     *
     * @param varName Name of the variable to write.
     * @param extraArgs Extra arguments to pass to the Python API. They are assume to be escaped by
     *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
     * @return The generated code to write the table.
     */
    override fun generateWriteCode(
        visitor: BodoCodeGenVisitor,
        varName: Variable,
        extraArgs: String,
    ): Variable = throw UnsupportedOperationException("Catalog APIs do not support additional arguments")

    /**
     * Return the location from which the table is generated. The return value is always entirely
     * capitalized.
     *
     * @return The source DB location.
     */
    override fun getDBType(): String = catalog.dbType.uppercase(Locale.getDefault())

    /**
     * Generate the code needed to read the table. This table type generates code common to all tables
     * in the catalog.
     *
     * @param useStreaming Should we generate code to read the table as streaming (currently only
     *     supported for snowflake tables)
     * @param streamingOptions Streaming-related options including batch size
     * @return The generated code to read the table.
     */
    override fun generateReadCode(
        useStreaming: Boolean,
        streamingOptions: StreamingOptions,
    ): Expr = catalog.generateReadCode(fullPath, useStreaming, streamingOptions)

    /**
     * Generate the code needed to read the table. This function is called by specialized IO
     * implementations that require passing 1 or more additional arguments.
     *
     * @param extraArgs: Extra arguments to pass to the Python API. They are assume to be escaped by
     *     the calling function and are of the form "key1=value1, ..., keyN=valueN".
     * @return The generated code to read the table.
     */
    override fun generateReadCode(extraArgs: String): Expr =
        throw UnsupportedOperationException("Catalog APIs do not support additional arguments")

    override fun generateReadCastCode(varName: Variable): Expr {
        // Snowflake catalog uses _bodo_read_date_as_dt64=True to convert date columns to datetime64
        // without astype() calls in the IR which cause issues for limit pushdown.
        // see BE-4238
        return varName
    }

    /**
     * Generates the code necessary to submit the remote query to the catalog DB. This is not
     * supported for local tables.
     *
     * @param query Query to submit.
     * @return The generated code.
     */
    override fun generateRemoteQuery(query: String): Expr = catalog.generateRemoteQuery(query)

    override fun extend(extensionFields: List<RelDataTypeField>): Table {
        val name = this.name
        val extendedColumns = mutableListOf<BodoSQLColumn>()
        extendedColumns.addAll(this.columns)
        for (i in 0..extensionFields.size) {
            val curField = extensionFields[i]
            val fieldName = curField.name
            val colType = curField.type
            val newColType = fromSqlType(colType)
            val newCol = BodoSQLColumnImpl(fieldName, newColType)
            extendedColumns.add(newCol)
        }
        return CatalogTable(name, parentFullPath, extendedColumns, this.catalog)
    }

    /**
     * Returns if calling `generateReadCode()` for a table will result in an IO operation in the Bodo
     * generated code.
     *
     * @return Does the table require IO?
     */
    override fun readRequiresIO(): Boolean = true

    override fun toRel(
        toRelContext: RelOptTable.ToRelContext,
        relOptTable: RelOptTable,
    ): RelNode? =
        throw UnsupportedOperationException(
            "toRel() must be implemented by specific catalog table implementations",
        )

    /**
     * Generate a Python connection string to read from the given catalog at the provided path.
     *
     * @param schemaPath The path to the table not including the table name.
     * @return A string that can be passed to Python to read from the table.
     */
    fun generatePythonConnStr(schemaPath: ImmutableList<String>): Expr = catalog.generatePythonConnStr(schemaPath)

    /**
     * Get the insert into write target for a particular table. Most tables must maintain the same
     * table type as already exists for the table, so this will generally be implemented by
     * subclasses.
     *
     * <p>TODO: Remove as an API when the CatalogTable becomes an abstract class
     *
     * @param columnNamesGlobal The global variable containing the column names. This should be
     *     possible to remove in the future since we append to a table.
     * @return The WriteTarget for the table.
     */
    override fun getInsertIntoWriteTarget(columnNamesGlobal: Variable): WriteTarget =
        throw UnsupportedOperationException("Insert into is not supported for this table")

    /** Get the DDL Executor for the table. This is used to execute DDL commands on the table. */
    open fun getDDLExecutor(): DDLExecutor = throw UnsupportedOperationException("DDL operations are not supported for this table")

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
    private fun tryGetViewMetadata(): InlineViewMetadata? = catalog.tryGetViewMetadata(fullPath)

    /**
     * Is this table definitely a view (meaning we can access its definition). If this returns False
     * we may have a view if we lack the permissions necessary to know it is a view.
     *
     * @return True if this is a view for which we can load metadata information.
     */
    fun isAccessibleView(): Boolean = tryGetViewMetadata() != null

    /**
     * Is this table actually a materialized view.
     *
     * @return True if this table is definitely a materialized view.
     */
    fun isMaterializedView(): Boolean {
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
    fun isSecureView(): Boolean {
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
    fun canSafelyInlineView(): Boolean = isAccessibleView() && !(isSecureView() || isMaterializedView())

    /**
     * Get the SQL query definition used to define this table if it is a view.
     *
     * @return The string definition that was used to create the view. Returns null if the table is
     * not a view.
     */
    fun getViewDefinitionString(): String? =
        if (isAccessibleView()) {
            tryGetViewMetadata()!!.viewDefinition
        } else {
            null
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
            val root =
                toRelContext.expandView(
                    input.outputType,
                    input.viewDefinition,
                    input.defaultPath,
                    input.viewPath,
                )
            val rel = root.calciteLogicalProject()
            // Verify that we can read before inlining.
            if (CheckTablePermissions.canRead(rel)) {
                return rel
            } else {
                throw BodoStatic.BODO_SQL_RESOURCE.noReadPermissionExpandingView(getQualifiedName()).ex()
            }
        } catch (e: Exception) {
            // Log the failure
            val message =
                String.format(
                    Locale.ROOT,
                    """
                    Unable to expand view %s with definition:
                    %s. Error encountered when compiling view:
                    %s
                    """.trimIndent(),
                    getQualifiedName(),
                    input.viewDefinition,
                    e.message,
                )
            PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.warning(message)
        } catch (e: Error) {
            // Log the failure
            val message =
                String.format(
                    Locale.ROOT,
                    """
                    Unable to expand view %s with definition:
                    %s. Error encountered when compiling view:
                    %s
                    """.trimIndent(),
                    getQualifiedName(),
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
    fun tryInlineView(
        toRelContext: RelOptTable.ToRelContext,
        viewDefinition: String,
        baseRelNode: RelNode,
    ): RelNode {
        val input =
            ExpandViewInput(
                baseRelNode.rowType,
                viewDefinition,
                parentFullPath,
                fullPath,
            )
        // Check the cache. We can only use the cache if the clusters
        // are the same.
        val result: RelNode?
        if (inlineViewCache.containsKey(input) && inlineViewCache[input]?.cluster == toRelContext.cluster) {
            result = inlineViewCache[input]
        } else {
            result = inlineViewImpl(toRelContext, input)
            // Store in the cache
            inlineViewCache[input] = result
        }
        return if (result != null) {
            // Log that we inlined the view.
            val levelOneMessage =
                String.format(
                    Locale.ROOT,
                    "Successfully inlined view %s",
                    getQualifiedName(),
                )
            PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER.info(levelOneMessage)
            val levelTwoMessage =
                String.format(
                    Locale.ROOT,
                    "Replaced view %s with definition %s",
                    getQualifiedName(),
                    input.viewDefinition,
                )
            PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER.info(levelTwoMessage)
            result
        } else {
            baseRelNode
        }
    }
}
