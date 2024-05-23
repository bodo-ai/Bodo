package com.bodosql.calcite.ddl

import com.bodosql.calcite.adapter.snowflake.BodoSnowflakeSqlDialect
import com.bodosql.calcite.catalog.IcebergCatalog
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.schemaPathToNamespace
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.tablePathToTableIdentifier
import com.bodosql.calcite.schema.CatalogSchema
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.rel.type.RelDataTypeField
import org.apache.calcite.sql.ddl.SqlCreateView
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.util.Util
import org.apache.iceberg.Schema
import org.apache.iceberg.catalog.Catalog
import org.apache.iceberg.catalog.Namespace
import org.apache.iceberg.catalog.SupportsNamespaces
import org.apache.iceberg.catalog.TableIdentifier
import org.apache.iceberg.catalog.ViewCatalog
import org.apache.iceberg.exceptions.AlreadyExistsException
import org.apache.iceberg.exceptions.NamespaceNotEmptyException
import org.apache.iceberg.types.Type
import org.apache.iceberg.types.Types

class IcebergDDLExecutor<T>(private val icebergConnection: T) : DDLExecutor where T : Catalog, T : SupportsNamespaces {
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

    override fun showObjects(schemaPath: ImmutableList<String>): DDLExecutionResult {
        // TODO:
        val fieldNames =
            listOf("CREATED_ON", "NAME", "KIND", "SCHEMA_NAME")
        val columnValues = List(4) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        // LOOP over objects
        // Tables
        icebergConnection.listTables(namespace).forEach {
            columnValues[0].add(null)
            columnValues[1].add(it.name())
            columnValues[2].add("TABLE")
            columnValues[3].add(namespace.levels().joinToString("."))
        }
        // TODO: Views
        return DDLExecutionResult(fieldNames, columnValues)
    }

    override fun showSchemas(dbPath: ImmutableList<String>): DDLExecutionResult {
        val fieldNames =
            listOf("CREATED_ON", "NAME", "KIND", "SCHEMA_NAME")
        val columnValues = List(4) { ArrayList<String?>() }
        // LOOP over objects
        val namespace = Namespace.of(*dbPath.toTypedArray())
        icebergConnection.listNamespaces(namespace).forEach {
            columnValues[0].add(null)
            // get child schema name only
            columnValues[1].add(it.level(it.levels().size - 1))
            columnValues[2].add(null)
            // get full schema path
            columnValues[3].add(it.toString())
        }
        return DDLExecutionResult(fieldNames, columnValues)
    }

    fun relDataTypeToViewSchemaType(dataType: RelDataType): Type {
        return when {
            SqlTypeFamily.STRING.contains(dataType) -> Types.StringType.get()
            SqlTypeFamily.BINARY.contains(dataType) -> Types.BinaryType.get()
            SqlTypeFamily.TIMESTAMP.contains(dataType) -> {
                if (dataType.sqlTypeName == SqlTypeName.TIMESTAMP) {
                    Types.TimestampType.withoutZone()
                } else {
                    Types.TimestampType.withZone()
                }
            }
            SqlTypeFamily.TIME.contains(dataType) -> Types.TimeType.get()
            SqlTypeFamily.INTEGER.contains(dataType) -> Types.IntegerType.get()
            SqlTypeFamily.DECIMAL.contains(dataType) -> Types.DecimalType.of(dataType.precision, dataType.scale)
            SqlTypeFamily.NUMERIC.contains(dataType) -> Types.DoubleType.get()
            SqlTypeFamily.BOOLEAN.contains(dataType) -> Types.BooleanType.get()
            SqlTypeFamily.ARRAY.contains(dataType) -> Types.ListType.ofOptional(0, relDataTypeToViewSchemaType(dataType.componentType!!))
            SqlTypeFamily.MAP.contains(
                dataType,
            ) ->
                Types.MapType.ofOptional(
                    0,
                    0,
                    relDataTypeToViewSchemaType(dataType.keyType!!),
                    relDataTypeToViewSchemaType(dataType.valueType!!),
                )
            else -> throw Exception("Unsupported data type $dataType in Iceberg view")
        }
    }

    override fun createOrReplaceView(
        viewPath: ImmutableList<String>,
        query: SqlCreateView,
        parentSchema: CatalogSchema,
        rowType: RelDataType,
    ) {
        if (icebergConnection is ViewCatalog) {
            val ns = schemaPathToNamespace(parentSchema.fullPath)
            val id = TableIdentifier.of(ns, viewPath.last())
            val viewBuilder = icebergConnection.buildView(id)

            // Construct a schema for the view by converting the RowType of the query
            var schemaId = 0
            val schema =
                Schema(
                    rowType.fieldList.map<RelDataTypeField, Types.NestedField> {
                        Types.NestedField.required(schemaId++, it.name, relDataTypeToViewSchemaType(it.type))
                    },
                )
            viewBuilder.withSchema(schema)

            val sqlString = query.query.toSqlString(BodoSnowflakeSqlDialect.DEFAULT).getSql()
            viewBuilder.withQuery("bodo", sqlString)
            viewBuilder.withDefaultNamespace(ns)
            viewBuilder.createOrReplace()
        } else {
            throw RuntimeException("CREATE VIEW is unimplemented for the current catalog")
        }
    }

    override fun dropView(viewPath: ImmutableList<String>) {
        if (icebergConnection is ViewCatalog) {
            val viewName = viewPath[viewPath.size - 1]
            val tableIdentifier = tablePathToTableIdentifier(viewPath.subList(0, viewPath.size - 1), Util.last(viewPath))
            val result = icebergConnection.dropView(tableIdentifier)
            if (!result) {
                throw RuntimeException("Unable to drop view $viewName. Please check that you have sufficient permissions.")
            }
        } else {
            throw RuntimeException("DROP VIEW is unimplemented for the current catalog")
        }
    }
}
