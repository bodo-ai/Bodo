package com.bodosql.calcite.catalog

import com.bodosql.calcite.table.BodoSQLColumn
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType
import com.bodosql.calcite.table.BodoSQLColumnImpl
import com.bodosql.calcite.table.ColumnDataTypeInfo
import com.google.common.collect.ImmutableList
import org.apache.iceberg.BaseMetastoreCatalog
import org.apache.iceberg.Table
import org.apache.iceberg.catalog.Namespace
import org.apache.iceberg.catalog.TableIdentifier
import org.apache.iceberg.types.Type
import org.apache.iceberg.types.Types.NestedField
import java.lang.RuntimeException

/**
 * Base abstract class for an Iceberg catalog. This is any catalog that
 * has partial or full Iceberg Support that can use a catalog via the
 * IcebergConnector to load explicit Iceberg information. Conceptually
 * this is extending the BodoSQLCatalog interface by providing an interface
 * to the Iceberg connector and additional API calls for Iceberg tables.
 *
 * Each implementing class will be responsible for providing the appropriate
 * Iceberg connection, but the majority of APIs should just depend on the
 * base Iceberg information. Some of this information may be possible to
 * abstract behind our existing Iceberg connector, which will reduce
 * the requirements in each individual implementation, but for now we will
 * be explicitly calling the public Iceberg API during development.
 */
abstract class IcebergCatalog(private val icebergConnection: BaseMetastoreCatalog) : BodoSQLCatalog {
    /**
     * Convert a BodoSQL representation for a table, which is an immutable list of strings
     * for the schema and a string for the table name into a TableIdentifier, which is the
     * Iceberg representation.
     * @param schemaPath The schema path to the table.
     * @param tableName The name of the table.
     * @return An Iceberg usable table identifier.
     */
    private fun tablePathToTableIdentifier(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): TableIdentifier {
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        return TableIdentifier.of(namespace, tableName)
    }

    /**
     * Load an Iceberg table from the connector via its path information.
     * @param schemaPath The schema path to the table.
     * @param tableName The name of the table.
     * @return The Iceberg table object.
     */
    private fun loadIcebergTable(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): Table {
        val tableIdentifier = tablePathToTableIdentifier(schemaPath, tableName)
        return icebergConnection.loadTable(tableIdentifier)
    }

    /**
     * Load a table's metadata map if it exists for the current snapshot.
     */
    private fun loadTableMetadataMap(table: Table): Map<String, String> {
        return table.currentSnapshot()?.summary() ?: mapOf()
    }

    private fun icebergTypeToBodoSQLColumnDataType(type: Type): BodoSQLColumnDataType {
        return when (type.typeId()) {
            Type.TypeID.BOOLEAN -> BodoSQLColumnDataType.BOOL8
            Type.TypeID.BINARY -> BodoSQLColumnDataType.BINARY
            Type.TypeID.INTEGER -> BodoSQLColumnDataType.INT32
            Type.TypeID.LONG -> BodoSQLColumnDataType.INT64
            Type.TypeID.FLOAT -> BodoSQLColumnDataType.FLOAT32
            Type.TypeID.DOUBLE -> BodoSQLColumnDataType.FLOAT64
            // TODO: Support other types
            else -> throw RuntimeException("Unsupported Iceberg Type")
        }
    }

    /**
     * Convert an Iceberg File type to its corresponding ColumnDataTypeInfo
     * used for BodoSQL types.
     * @param field The Iceberg Field to type.
     * @return The BodoSQL ColumnDataTypeInfo
     */
    private fun icebergTypeToTypeInfo(field: NestedField): ColumnDataTypeInfo {
        val dataType = icebergTypeToBodoSQLColumnDataType(field.type())
        val isNullable = field.isOptional
        // TODO: Support nested types
        return ColumnDataTypeInfo(dataType, isNullable)
    }

    /**
     * Get the column information for a table as described by its path and table name. This information
     * is obtained by fetching the table definition from the Iceberg connector and then converting
     * from the Iceberg type to standard BodoSQL types.
     * @param schemaPath The schema path to the table.
     * @param tableName The name of the table.
     * @return A list of typed columns for the table.
     */
    fun getIcebergTableColumns(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): List<BodoSQLColumn> {
        val table = loadIcebergTable(schemaPath, tableName)
        val schema = table.schema()
        return schema.columns().map { BodoSQLColumnImpl(it.name(), it.name(), icebergTypeToTypeInfo(it)) }
    }

    /**
     * Estimate the row count for an Iceberg table.
     * @param schemaPath The schema path to the table.
     * @param tableName The name of the table.
     * @return The row count for the iceberg table. If the metadata doesn't exist
     * we return null.
     */
    fun estimateIcebergTableRowCount(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): Double? {
        val table = loadIcebergTable(schemaPath, tableName)
        val metadata = loadTableMetadataMap(table)
        return metadata["total-records"]?.toDouble()
    }
}
