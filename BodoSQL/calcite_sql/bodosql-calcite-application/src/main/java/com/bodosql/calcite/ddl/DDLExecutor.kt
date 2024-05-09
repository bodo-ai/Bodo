package com.bodosql.calcite.ddl

import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.sql.ddl.SqlCreateView

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
     * @param cascade The cascade operation lag used by Snowflake. This is ignored
     * by other connectors.
     * @return The result of the operation.
     */
    fun dropTable(
        tablePath: ImmutableList<String>,
        cascade: Boolean,
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

    @Throws(ViewAlreadyExistsException::class)
    fun createView(
        viewPath: ImmutableList<String>,
        query: SqlCreateView,
    )
}
