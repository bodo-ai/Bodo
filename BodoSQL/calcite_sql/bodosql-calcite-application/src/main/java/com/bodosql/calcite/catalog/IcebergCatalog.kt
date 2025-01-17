package com.bodosql.calcite.catalog

import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.schema.InlineViewMetadata
import com.bodosql.calcite.table.BodoSQLColumn
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType
import com.bodosql.calcite.table.BodoSQLColumnImpl
import com.bodosql.calcite.table.ColumnDataTypeInfo
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.util.Util
import org.apache.iceberg.ManifestFiles
import org.apache.iceberg.Table
import org.apache.iceberg.catalog.Catalog
import org.apache.iceberg.catalog.Namespace
import org.apache.iceberg.catalog.SupportsNamespaces
import org.apache.iceberg.catalog.TableIdentifier
import org.apache.iceberg.catalog.ViewCatalog
import org.apache.iceberg.exceptions.NoSuchViewException
import org.apache.iceberg.types.Type
import org.apache.iceberg.types.Types.DecimalType
import org.apache.iceberg.types.Types.FixedType
import org.apache.iceberg.types.Types.ListType
import org.apache.iceberg.types.Types.MapType
import org.apache.iceberg.types.Types.StructType
import org.apache.iceberg.types.Types.TimestampType
import org.apache.iceberg.view.View

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
abstract class IcebergCatalog<T>(
    private val icebergConnection: T,
) : BodoSQLCatalog where T : Catalog, T : SupportsNamespaces {
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

    private fun loadIcebergView(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): View {
        if (icebergConnection !is ViewCatalog) {
            throw AssertionError("To call this function, icebergConnection must support views")
        }
        val tableIdentifier = tablePathToTableIdentifier(schemaPath, tableName)
        return icebergConnection.loadView(tableIdentifier)
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
        val schema =
            if (isView(schemaPath, tableName)) {
                val view = loadIcebergView(schemaPath, tableName)
                view.schema()
            } else {
                val table = loadIcebergTable(schemaPath, tableName)
                table.schema()
            }
        return schema.columns().map { BodoSQLColumnImpl(it.name(), it.name(), icebergTypeToTypeInfo(it.type(), it.isOptional)) }
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
        if (isView(schemaPath, tableName)) {
            return null
        }

        val table = loadIcebergTable(schemaPath, tableName)
        // If we can't find a snapshot then the table must be brand new and empty.
        val currentSnapshot = table.currentSnapshot() ?: return 0.0
        val summary = currentSnapshot.summary() ?: mapOf()
        val summaryRowCount = summary["total-records"]?.toDouble()
        return if (summaryRowCount == null) {
            val io = table.io()
            val manifests = currentSnapshot.allManifests(io)
            manifests
                .sumOf {
                    // Summary values from the manifest list to avoid checking each manifest file.
                    val addedRows = it.addedRowsCount()
                    val existingRows = it.existingRowsCount()
                    val deletedRows = it.deletedRowsCount()
                    if (addedRows == null || existingRows == null || deletedRows == null) {
                        // We don't have 1 or more of the optional values, we need to actually
                        // read the avro file and check the data_file field
                        val manifestContents = ManifestFiles.read(it, io)
                        val sign = if (manifestContents.isDeleteManifestReader) -1 else 1
                        manifestContents.sumOf { f -> sign * f.recordCount() }
                    } else {
                        (addedRows + existingRows) - deletedRows
                    }
                }.toDouble()
        } else {
            summaryRowCount
        }
    }

    /**
     * Estimate the number of distinct rows in a column of an Iceberg table.
     * @param schemaPath The schema path to the table.
     * @param tableName The name of the table.
     * @param colIdx The column index whose NDV is being approximated.
     * @return The approximate distinct count for the specified column of the table.
     * If the metadata doesn't exist we return null.
     */
    fun estimateIcebergTableColumnDistinctCount(
        schemaPath: ImmutableList<String>,
        tableName: String,
        colIdx: Int,
    ): Double? {
        if (isView(schemaPath, tableName)) {
            return null
        }

        val table = loadIcebergTable(schemaPath, tableName)
        val currentSnapshot = table.currentSnapshot() ?: return null
        val schema = table.schema()
        val fieldId = schema.columns()[colIdx].fieldId()
        // [BSE-3168] TODO: explore using the sequence number to check if
        // a file is "fresh enough."
        table.statisticsFiles().forEach { statFile ->
            if (statFile.snapshotId() == currentSnapshot.snapshotId()) {
                statFile.blobMetadata().forEach { blob ->
                    if ((blob.type() == "apache-datasketches-theta-v1") &&
                        (blob.fields().size == 1) &&
                        (blob.fields()[0] == fieldId)
                    ) {
                        return blob.properties()["ndv"]?.toDouble()
                    }
                }
            }
        }
        return null
    }

    override fun getAccountName(): String? = null

    private fun isView(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): Boolean {
        if (icebergConnection !is ViewCatalog) return false
        val tableIdentifier = tablePathToTableIdentifier(schemaPath, tableName)
        return icebergConnection.viewExists(tableIdentifier)
    }

    /**
     * Returns a schema found within the given parent path.
     *
     * @param schemaPath The parent schema path to check.
     * @param schemaName Name of the schema to fetch.
     * @return A schema object.
     */
    override fun getSchema(
        schemaPath: ImmutableList<String>,
        schemaName: String,
    ): CatalogSchema = CatalogSchema(schemaName, schemaPath.size + 1, schemaPath, this)

    override fun tryGetViewMetadata(names: MutableList<String>): InlineViewMetadata? =
        if (icebergConnection is ViewCatalog) {
            val id = tablePathToTableIdentifier(ImmutableList.copyOf(Util.skipLast(names)), Util.last(names))
            try {
                val view = icebergConnection.loadView(id)
                InlineViewMetadata(unsafeToInline = false, isMaterialized = false, view.sqlFor("bodo").sql())
            } catch (e: NoSuchViewException) {
                null
            }
        } else {
            null
        }

    fun getIcebergConnection(): T = icebergConnection

    override fun getDBType(): String = "ICEBERG"

    companion object {
        // All Iceberg Timestamp columns have precision 6
        private const val ICEBERG_DATETIME_PRECISION = 6

        // Iceberg UUID is defined as 16 bytes
        private const val ICEBERG_UUID_PRECISION = 16

        /**
         * Convert a BodoSQL schema path representation, which is an immutable list of
         * strings, into an Iceberg Namespace.
         */
        @JvmStatic
        fun schemaPathToNamespace(schemaPath: List<String>): Namespace = Namespace.of(*schemaPath.toTypedArray())

        /**
         * Convert a BodoSQL representation for a table, which is an immutable list of strings
         * for the schema and a string for the table name into a TableIdentifier, which is the
         * Iceberg representation.
         * @param schemaPath The schema path to the table.
         * @param tableName The name of the table.
         * @return An Iceberg usable table identifier.
         */
        @JvmStatic
        fun tablePathToTableIdentifier(
            schemaPath: List<String>,
            tableName: String,
        ): TableIdentifier {
            val namespace = schemaPathToNamespace(schemaPath)
            return TableIdentifier.of(namespace, tableName)
        }

        @JvmStatic
        private fun icebergTypeToBodoSQLColumnDataType(type: Type): BodoSQLColumnDataType =
            when (type.typeId()) {
                Type.TypeID.BOOLEAN -> BodoSQLColumnDataType.BOOL8
                Type.TypeID.INTEGER -> BodoSQLColumnDataType.INT32
                Type.TypeID.LONG -> BodoSQLColumnDataType.INT64
                Type.TypeID.FLOAT -> BodoSQLColumnDataType.FLOAT32
                Type.TypeID.DOUBLE -> BodoSQLColumnDataType.FLOAT64
                Type.TypeID.DATE -> BodoSQLColumnDataType.DATE
                Type.TypeID.TIME -> BodoSQLColumnDataType.TIME
                Type.TypeID.TIMESTAMP -> {
                    if ((type as TimestampType).shouldAdjustToUTC()) {
                        BodoSQLColumnDataType.TIMESTAMP_LTZ
                    } else {
                        BodoSQLColumnDataType.TIMESTAMP_NTZ
                    }
                }
                Type.TypeID.STRING -> BodoSQLColumnDataType.STRING
                Type.TypeID.UUID -> BodoSQLColumnDataType.FIXED_SIZE_STRING
                Type.TypeID.FIXED -> BodoSQLColumnDataType.FIXED_SIZE_BINARY
                Type.TypeID.BINARY -> BodoSQLColumnDataType.BINARY
                Type.TypeID.DECIMAL -> BodoSQLColumnDataType.DECIMAL
                Type.TypeID.LIST -> BodoSQLColumnDataType.ARRAY
                Type.TypeID.MAP -> BodoSQLColumnDataType.JSON_OBJECT
                Type.TypeID.STRUCT -> BodoSQLColumnDataType.STRUCT
                else -> throw RuntimeException("Unsupported Iceberg Type")
            }

        /**
         * Convert an Iceberg File type to its corresponding ColumnDataTypeInfo
         * used for BodoSQL types.
         * @param type The Iceberg type.
         * @param isNullable Is this nullable. This is needed for nested types.
         * @return The BodoSQL ColumnDataTypeInfo
         */
        @JvmStatic
        fun icebergTypeToTypeInfo(
            type: Type,
            isNullable: Boolean,
        ): ColumnDataTypeInfo {
            val dataType = icebergTypeToBodoSQLColumnDataType(type)
            val precision =
                when (type.typeId()) {
                    Type.TypeID.DECIMAL -> (type as DecimalType).precision()
                    Type.TypeID.UUID -> ICEBERG_UUID_PRECISION
                    Type.TypeID.FIXED -> (type as FixedType).length()
                    Type.TypeID.TIMESTAMP, Type.TypeID.TIME -> ICEBERG_DATETIME_PRECISION
                    else -> RelDataType.PRECISION_NOT_SPECIFIED
                }
            val scale =
                if (type is DecimalType) {
                    type.scale()
                } else {
                    RelDataType.SCALE_NOT_SPECIFIED
                }
            val children =
                if (type.isListType) {
                    listOf(icebergTypeToTypeInfo((type as ListType).elementType(), type.isElementOptional))
                } else if (type.isMapType) {
                    val typeAsMap = type as MapType
                    listOf(
                        icebergTypeToTypeInfo(typeAsMap.keyType(), false),
                        icebergTypeToTypeInfo(typeAsMap.valueType(), type.isValueOptional),
                    )
                } else if (type.isStructType) {
                    (type as StructType).fields().map { icebergTypeToTypeInfo(it.type(), it.isOptional) }
                } else {
                    listOf()
                }
            val fieldNames =
                if (type.isStructType) {
                    (type as StructType).fields().map { it.name() }
                } else {
                    listOf()
                }
            return ColumnDataTypeInfo(dataType, isNullable, precision, scale, children, fieldNames)
        }
    }
}
