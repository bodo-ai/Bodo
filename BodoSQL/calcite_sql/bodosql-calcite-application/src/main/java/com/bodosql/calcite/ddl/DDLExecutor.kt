package com.bodosql.calcite.ddl

import com.bodosql.calcite.schema.CatalogSchema
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.ddl.SqlCreateView
import org.apache.calcite.sql.validate.SqlValidator

class NamespaceAlreadyExistsException : Exception()

class NamespaceNotFoundException : Exception()

class ViewAlreadyExistsException : Exception()

class MissingObjectException(message: String) : Exception(message)

/**
 * General interface for executing DDL operations. Each distinct catalog table type
 * (e.g. Iceberg, Snowflake Native, etc.) should have its own implementation of this
 * interface. This allows for the DDL operations to be executed properly by directly
 * interacting with the connector.
 */
interface DDLExecutor {
    /**
     * Create a schema / namespace in the catalog. Note: We don't need ifNotExists
     * because we will do error handling for the existence of the schema in the caller
     */
    @Throws(NamespaceAlreadyExistsException::class)
    fun createSchema(schemaPath: ImmutableList<String>)

    /**
     * Drops a schema / namespace from the catalog. Note: We don't need ifExists because we
     * handle that case during error checking in the caller.
     */
    @Throws(NamespaceNotFoundException::class)
    fun dropSchema(
        defaultSchemaPath: ImmutableList<String>,
        schemaName: String,
    )

    /**
     * Drops a table from the catalog. Note: We don't need ifExists because we
     * have already checked for the existence of the table before calling this.
     * @param tablePath The path to the table to drop.
     * @param cascade Specifies whether the table can be dropped if foreign keys exist that reference the table.
     * @param purge Indicate whether (meta)data files are marked for permanent deletion.
     * @return The result of the operation.
     */
    fun dropTable(
        tablePath: ImmutableList<String>,
        cascade: Boolean,
        purge: Boolean,
    ): DDLExecutionResult

    /**
     * Renames a table `tablePath` to `renamePath`. If `ifExists` is true, then
     * even if the table does not exist, the operation will not fail (instead raising an error).
     *
     * Meant to be invoked by the `ALTER TABLE RENAME TO` command.
     * @TODO: May need to be renamed when adding support for additional DDL commands.
     *
     * @param tablePath The path to the table to rename.
     * @param renamePath The new name of the table.
     * @param ifExists If true, the operation will not fail if the table does not exist.
     */
    fun renameTable(
        tablePath: ImmutableList<String>,
        renamePath: ImmutableList<String>,
        ifExists: Boolean,
    ): DDLExecutionResult

    /**
     * Renames a view `viewPath` to `renamePath`. If `ifExists` is true, then
     * even if the view does not exist, the operation will not fail (instead raising an error).
     *
     * Meant to be invoked by the `ALTER VIEW RENAME TO` command.
     * @TODO: May need to be renamed when adding support for additional DDL commands.
     *
     * @param viewPath The path to the view to rename.
     * @param renamePath The new name of the view.
     * @param ifExists If true, the operation will not fail if the view does not exist.
     */
    fun renameView(
        viewPath: ImmutableList<String>,
        renamePath: ImmutableList<String>,
        ifExists: Boolean,
    ): DDLExecutionResult

    /**
     * Set a property on a table in the catalog. If the property already exists, it will be
     * overwritten. If the property does not exist, it will be created.
     *
     * @param tablePath The path to the table to set the property on.
     * @param propertyList The list of properties to set. Must be a SqlNodeList of SqlLiteral.
     * @param valueList The list of values to set. Must be a SqlNodeList of SqlLiteral.
     * @param ifExists If true, the operation will not fail if the table does not exist.
     *
     * @return The result of the operation.
     */
    fun setProperty(
        tablePath: ImmutableList<String>,
        propertyList: SqlNodeList,
        valueList: SqlNodeList,
        ifExists: Boolean,
    ): DDLExecutionResult

    /**
     * Unset (delete) a property on a table in the catalog.
     *
     * @param tablePath The path to the table to unset the properties on.
     * @param propertyList The list of properties to unset. Must be a SqlNodeList of SqlLiteral.
     * @param ifExists If true, the operation will not fail if the table does not exist.
     * @param ifPropertyExists If true, the operation will not fail if the property does not exist.
     *
     * @return The result of the operation.
     */
    fun unsetProperty(
        tablePath: ImmutableList<String>,
        propertyList: SqlNodeList,
        ifExists: Boolean,
        ifPropertyExists: Boolean,
    ): DDLExecutionResult

    /**
     * Add a column to a table in the catalog.
     *
     * @param tablePath The path to the table to add the column to.
     * @param ifExists If true, the operation will not fail if the table does not exist.
     * @param ifNotExists If true, the operation will not fail if the property does not exist.
     * @param addCol SqlNode representing column details to be added (name, type, etc)
     * @param validator Validator needed to derive type information from addCol SqlNode.
     * @return The result of the operation.
     */
    fun addColumn(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        ifNotExists: Boolean,
        addCol: SqlNode,
        validator: SqlValidator,
    ): DDLExecutionResult

    /**
     * Drops a column in a table in the catalog.
     *
     * @param tablePath The path to the table to add the column to.
     * @param ifExists If true, the operation will not fail if the table does not exist.
     * @param dropCols SqlNodeList representing columns to be dropped. (name, type, etc)
     *                 Should be list of CompoundIdentifiers.
     * @param ifColumnExists If true, the operation will not fail if the columns do not exist.
     * @return The result of the operation.
     */
    fun dropColumn(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        dropCols: SqlNodeList,
        ifColumnExists: Boolean,
    ): DDLExecutionResult

    /**
     * Describes a table in the catalog. We use a type factory to create the Bodo
     * type consistently across all catalogs.
     * @param tablePath The path to the table to describe.
     * @param typeFactory The type factory to use for creating the Bodo Type.
     * @return The result of the operation.
     */
    fun describeTable(
        tablePath: ImmutableList<String>,
        typeFactory: RelDataTypeFactory,
    ): DDLExecutionResult

    /**
     * Show objects in the database.
     * @param schemaPath The path to the schema to show objects from.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     */
    fun showObjects(schemaPath: ImmutableList<String>): DDLExecutionResult

    /**
     * Show schemas in the database.
     * @param dbPath The path to schema to show all sub-schemas from.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     */
    fun showSchemas(dbPath: ImmutableList<String>): DDLExecutionResult

    /**
     * Show tables in the database.
     * @param schemaPath The path to the schema to show tables from.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     */
    fun showTables(schemaPath: ImmutableList<String>): DDLExecutionResult

    /**
     * Show views in the database.
     * @param schemaPath The path to the schema to show views from.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     */
    fun showViews(schemaPath: ImmutableList<String>): DDLExecutionResult

    @Throws(ViewAlreadyExistsException::class)
    fun createOrReplaceView(
        viewPath: ImmutableList<String>,
        query: SqlCreateView,
        parentSchema: CatalogSchema,
        rowType: RelDataType,
    )

    /**
     * Describes a view in the catalog. We use a type factory to create the Bodo
     * type consistently across all catalogs.
     * @param viewPath The path to the table to describe.
     * @param typeFactory The type factory to use for creating the Bodo Type.
     * @return The result of the operation.
     */
    fun describeView(
        viewPath: ImmutableList<String>,
        typeFactory: RelDataTypeFactory,
    ): DDLExecutionResult

    /*
     * Drops a view from the catalog. Note: We don't need ifExists because we
     * have already checked for the existence of the table before calling this.
     * @param viewPath The path to the view to describe.
     * @return The result of the operation.
     */
    @Throws(NamespaceNotFoundException::class, MissingObjectException::class)
    fun dropView(viewPath: ImmutableList<String>)
}
