package com.bodosql.calcite.ddl

import com.bodosql.calcite.catalog.IcebergCatalog
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.tablePathToTableIdentifier
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataTypeFactory
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

    override fun describeTable(
        tablePath: ImmutableList<String>,
        typeFactory: RelDataTypeFactory,
    ): DDLExecutionResult {
        val names = listOf("NAME", "TYPE", "KIND", "NULL?", "DEFAULT", "PRIMARY_KEY", "UNIQUE_KEY")
        val columnValues = List(7) { ArrayList<String?>() }

        val tableName = tablePath[tablePath.size - 1]
        val tableIdentifier = tablePathToTableIdentifier(tablePath.subList(0, tablePath.size - 1), Util.last(tablePath))
        val table = icebergConnection.loadTable(tableIdentifier)
        val schema = table.schema()
        schema.columns().forEach {
            columnValues[0].add(it.name())
            val typeInfo = IcebergCatalog.icebergTypeToTypeInfo(it.type(), it.isOptional)
            val bodoType = typeInfo.convertToSqlType(typeFactory)
            columnValues[1].add(bodoType.toString())
            columnValues[2].add("COLUMN")
            columnValues[3].add(if (it.isOptional) "Y" else "N")
            // We don't support default values yet.
            columnValues[4].add(null)
            // Iceberg doesn't support primary or unique keys yet.
            columnValues[5].add("N")
            columnValues[6].add("N")
        }
        return DDLExecutionResult(names, columnValues)
    }
}
