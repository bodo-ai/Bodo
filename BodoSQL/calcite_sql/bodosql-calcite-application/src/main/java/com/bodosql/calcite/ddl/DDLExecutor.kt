package com.bodosql.calcite.ddl

import com.google.common.collect.ImmutableList

/**
 * General interface for executing DDL operations. Each distinct catalog table type
 * (e.g. Iceberg, Snowflake Native, etc.) should have its own implementation of this
 * interface. This allows for the DDL operations to be executed properly by directly
 * interacting with the connector.
 */
interface DDLExecutor {
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
}
