package com.bodosql.calcite.ddl

import com.bodosql.calcite.catalog.IcebergCatalog
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.schemaPathToNamespace
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.tablePathToTableIdentifier
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.util.Util
import org.apache.iceberg.BaseMetastoreCatalog
import org.apache.iceberg.catalog.Namespace
import org.apache.iceberg.catalog.SupportsNamespaces
import org.apache.iceberg.exceptions.AlreadyExistsException
import org.apache.iceberg.exceptions.NamespaceNotEmptyException

class IcebergDDLExecutor<T>(private val icebergConnection: T) : DDLExecutor where T : BaseMetastoreCatalog, T : SupportsNamespaces {
    override fun createSchema(schemaPath: ImmutableList<String>) {
        val ns = schemaPathToNamespace(schemaPath)
        try {
            icebergConnection.createNamespace(ns)
            // Even though not explicitly set, all catalogs use this exception
        } catch (e: AlreadyExistsException) {
            throw NamespaceAlreadyExistsException()
        }
    }

    /**
     * Helper Function to recursively drop a schema, including its contents
     * @param ns Namespace path of schema to drop
     * @return True if the namespace existed, false otherwise
     */
    private fun dropSchema(ns: Namespace): Boolean {
        return try {
            icebergConnection.dropNamespace(ns)
        } catch (e: NamespaceNotEmptyException) {
            for (nsInner in icebergConnection.listNamespaces(ns)) {
                dropSchema(nsInner)
            }

            for (tableInner in icebergConnection.listTables(ns)) {
                icebergConnection.dropTable(tableInner)
            }

            // Should return true in this path, since the schema has contents
            icebergConnection.dropNamespace(ns)
        }
    }

    override fun dropSchema(
        defaultSchemaPath: ImmutableList<String>,
        schemaName: String,
    ) {
        val ns = schemaPathToNamespace(ImmutableList.copyOf(defaultSchemaPath + schemaName))
        if (!dropSchema(ns)) {
            throw NamespaceNotFoundException()
        }
    }

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
