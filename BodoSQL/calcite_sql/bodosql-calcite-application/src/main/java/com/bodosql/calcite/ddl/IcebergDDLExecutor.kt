package com.bodosql.calcite.ddl

import com.bodosql.calcite.catalog.IcebergCatalog.Companion.tablePathToTableIdentifier
import com.google.common.collect.ImmutableList
import org.apache.calcite.util.Util
import org.apache.iceberg.BaseMetastoreCatalog

class IcebergDDLExecutor(private val icebergConnection: BaseMetastoreCatalog) : DDLExecutor {
    /**
     * Drops a table from the catalog.
     * @param tablePath The path to the table to drop.
     * @param cascade The cascade operation lag used by Snowflake. This is ignored
     * by other connectors.
     * @return The result of the operation.
     */
    override fun dropTable(
        tablePath: ImmutableList<String>,
        cascade: Boolean,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableIdentifier = tablePathToTableIdentifier(tablePath.subList(0, tablePath.size - 1), Util.last(tablePath))
        // TOOD: Should we set purge=True. This will delete the data/metadata files but prevents time travel.
        val result = icebergConnection.dropTable(tableIdentifier)
        if (!result) {
            throw RuntimeException("Unable to drop table $tableName. Please check that you have sufficient permissions.")
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("$tableName successfully dropped.")))
    }
}
