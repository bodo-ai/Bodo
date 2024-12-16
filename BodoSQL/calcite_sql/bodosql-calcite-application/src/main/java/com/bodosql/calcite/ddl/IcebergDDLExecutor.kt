package com.bodosql.calcite.ddl

import com.bodosql.calcite.adapter.snowflake.BodoSnowflakeSqlDialect
import com.bodosql.calcite.catalog.IcebergCatalog
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.schemaPathToNamespace
import com.bodosql.calcite.catalog.IcebergCatalog.Companion.tablePathToTableIdentifier
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.sql.ddl.SqlSnowflakeColumnDeclaration
import com.google.common.collect.ImmutableList
import org.apache.calcite.rel.type.RelDataType
import org.apache.calcite.rel.type.RelDataTypeFactory
import org.apache.calcite.rel.type.RelDataTypeField
import org.apache.calcite.sql.SqlIdentifier
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.SqlNodeList
import org.apache.calcite.sql.ddl.SqlCreateView
import org.apache.calcite.sql.type.SqlTypeFamily
import org.apache.calcite.sql.type.SqlTypeName
import org.apache.calcite.sql.validate.SqlValidator
import org.apache.calcite.util.Util
import org.apache.iceberg.Schema
import org.apache.iceberg.catalog.Catalog
import org.apache.iceberg.catalog.Namespace
import org.apache.iceberg.catalog.SupportsNamespaces
import org.apache.iceberg.catalog.TableIdentifier
import org.apache.iceberg.catalog.ViewCatalog
import org.apache.iceberg.exceptions.AlreadyExistsException
import org.apache.iceberg.exceptions.NamespaceNotEmptyException
import org.apache.iceberg.exceptions.NoSuchTableException
import org.apache.iceberg.exceptions.NoSuchViewException
import org.apache.iceberg.exceptions.ValidationException
import org.apache.iceberg.types.Type
import org.apache.iceberg.types.Types

class IcebergDDLExecutor<T>(
    private val icebergConnection: T,
) : DDLExecutor where T : Catalog, T : SupportsNamespaces {
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
    private fun dropSchema(ns: Namespace): Boolean =
        try {
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
     * @param purge If purge is true, it will actually delete the data/metadata files which prevents time travel.
     * If purge is false, even though the table is dropped the underlying data/metadata may still exist.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return The result of the operation.
     */
    override fun dropTable(
        tablePath: ImmutableList<String>,
        cascade: Boolean,
        purge: Boolean,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableIdentifier = tablePathToTableIdentifier(tablePath.subList(0, tablePath.size - 1), Util.last(tablePath))
        val result = icebergConnection.dropTable(tableIdentifier, purge)
        if (!result) {
            throw RuntimeException("Unable to drop table $tableName. Please check that you have sufficient permissions.")
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("$tableName successfully dropped.")), returnTypes)
    }

    override fun describeTable(
        tablePath: ImmutableList<String>,
        typeFactory: RelDataTypeFactory,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val names =
            listOf(
                "NAME",
                "TYPE",
                "KIND",
                "NULL?",
                "DEFAULT",
                "PRIMARY_KEY",
                "UNIQUE_KEY",
                "CHECK",
                "EXPRESSION",
                "COMMENT",
                "POLICY NAME",
                "PRIVACY DOMAIN",
            )
        val columnValues = List(12) { ArrayList<String?>() }

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
            // Five new columns from Snowflake that is set to null in Iceberg
            for (idx in 7..11) columnValues[idx].add(null)
        }
        return DDLExecutionResult(names, columnValues, returnTypes)
    }

    /**
     * Emulates DESCRIBE SCHEMA for a specified namespace in Iceberg.
     * The method will list out all tables, and if applicable, views, in the
     * namespace provided by schemaPath.
     *
     * @param schemaPath The schema path to describe.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, KIND
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @throws NoSuchNamespaceException if namespace cannot be found
     */
    override fun describeSchema(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val fieldNames =
            listOf("CREATED_ON", "NAME", "KIND")
        val columnValues = List(fieldNames.size) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        // Loop over all objects in the schema,
        // and add their details to the result.
        // Tables
        icebergConnection.listTables(namespace).forEach {
            columnValues[0].add(null)
            columnValues[1].add(it.name())
            columnValues[2].add("TABLE")
        }
        // Views
        if (icebergConnection is ViewCatalog) {
            icebergConnection.listViews(namespace).forEach {
                // Iceberg tables do not store created timestamps, so we
                // always store them as null.
                columnValues[0].add(null)
                columnValues[1].add(it.name())
                columnValues[2].add("VIEW")
            }
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW TERSE OBJECTS for a specified namespace in Iceberg.
     *
     * @param schemaPath The schema path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws NoSuchNamespaceException if namespace cannot be found
     *
     * The method uses the .listTables(namespace) and .listViews(namespace) method of the respective catalog
     * to emulate SHOW TERSE OBJECTS.
     */
    override fun showTerseObjects(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val fieldNames =
            listOf("CREATED_ON", "NAME", "KIND", "SCHEMA_NAME")
        val columnValues = List(4) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        // Loop over all objects in the schema,
        // and add their details to the result.
        // Tables
        icebergConnection.listTables(namespace).forEach {
            columnValues[0].add(null)
            columnValues[1].add(it.name())
            columnValues[2].add("TABLE")
            columnValues[3].add(namespace.levels().joinToString("."))
        }
        // Views
        if (icebergConnection is ViewCatalog) {
            icebergConnection.listViews(namespace).forEach {
                columnValues[0].add(null)
                columnValues[1].add(it.name())
                columnValues[2].add("VIEW")
                columnValues[3].add(namespace.levels().joinToString("."))
            }
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW OBJECTS for a specified namespace in Iceberg.
     *
     * @param schemaPath The schema path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     * @throws NoSuchNamespaceException if namespace cannot be found
     *
     * The method uses the .listTables(namespace) and .listViews(namespace) method of the respective catalog
     * to emulate the table part of SHOW TERSE OBJECTS.
     */
    override fun showObjects(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        // Short helper to update the last element of a list.
        fun <T> MutableList<T>.setLast(value: T) {
            if (this.isNotEmpty()) {
                this[this.size - 1] = value
            }
        }
        val fieldNames =
            listOf(
                "CREATED_ON",
                "NAME",
                "SCHEMA_NAME",
                "KIND",
                "COMMENT",
                "CLUSTER_BY",
                "ROWS",
                "BYTES",
                "OWNER",
                "RETENTION_TIME",
                "OWNER_ROLE_TYPE",
            )
        val columnValues = List(fieldNames.size) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        // Loop over all objects in the schema,
        // and add their details to the result.
        // Tables
        icebergConnection.listTables(namespace).forEach {
            val table = icebergConnection.loadTable(it)
            // Iceberg tables do not store created timestamps, so we
            // always store them as null. All unsupported fields are
            // also null by default.
            for (i in fieldNames.indices) {
                columnValues[i].add(null)
            }
            // We only modify the supported fields.
            columnValues[fieldNames.indexOf("NAME")].setLast(it.name())
            columnValues[fieldNames.indexOf("KIND")].setLast("TABLE")
            columnValues[fieldNames.indexOf("SCHEMA_NAME")].setLast(namespace.levels().joinToString("."))
            columnValues[fieldNames.indexOf("COMMENT")].setLast(table.properties()["comment"])
        }
        // Views
        if (icebergConnection is ViewCatalog) {
            icebergConnection.listViews(namespace).forEach {
                val table = icebergConnection.loadView(it)
                // Iceberg views do not store created timestamps, so we
                // always store them as null. All unsupported fields are
                // also null by default.
                for (i in fieldNames.indices) {
                    columnValues[i].add(null)
                }
                // We only modify the supported fields.
                columnValues[fieldNames.indexOf("NAME")].setLast(it.name())
                columnValues[fieldNames.indexOf("KIND")].setLast("VIEW")
                columnValues[fieldNames.indexOf("SCHEMA_NAME")].setLast(namespace.levels().joinToString("."))
                columnValues[fieldNames.indexOf("COMMENT")].setLast(table.properties()["comment"])
            }
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW TERSE SCHEMAS for a specified namespace in Iceberg.
     *
     * @param dbPath The db path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws NoSuchNamespaceException if namespace cannot be found
     *
     * The method uses the .listNamespaces(namespace) method of the respective catalog
     * to emulate SHOW TERSE SCHEMAS.
     */
    override fun showTerseSchemas(
        dbPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
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

        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW SCHEMAS for a specified namespace in Iceberg.
     *
     * @param dbPath The db path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     * @throws NoSuchNamespaceException if namespace cannot be found
     *
     * The method uses the .listNamespaces(namespace) method of the respective catalog
     * to emulate SHOW SCHEMAS.
     */
    override fun showSchemas(
        dbPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        // Short helper to update the last element of a list.
        fun <T> MutableList<T>.setLast(value: T) {
            if (this.isNotEmpty()) {
                this[this.size - 1] = value
            }
        }
        val fieldNames =
            listOf(
                "CREATED_ON",
                "NAME",
                "IS_DEFAULT",
                "IS_CURRENT",
                "DATABASE_NAME",
                "OWNER",
                "COMMENT",
                "OPTIONS",
                "RETENTION_TIME",
                "OWNER_ROLE_TYPE",
            )
        val columnValues = List(fieldNames.size) { ArrayList<String?>() }
        // LOOP over objects
        val namespace = Namespace.of(*dbPath.toTypedArray())
        icebergConnection.listNamespaces(namespace).forEach {
            // Set all fields to null by default,
            // and only add supported fields.
            for (i in fieldNames.indices) {
                columnValues[i].add(null)
            }
            columnValues[fieldNames.indexOf("NAME")].setLast(it.level(it.levels().size - 1))
            columnValues[fieldNames.indexOf("DATABASE_NAME")].setLast(namespace.levels().joinToString("."))
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW TERSE TABLES for a specified namespace in Iceberg.
     *
     * @param schemaPath The schema path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws NoSuchNamespaceException if namespace cannot be found
     *
     * The method uses the .listTables(namespace) method of the respective catalog
     * to emulate SHOW TERSE TABLES.
     */
    override fun showTerseTables(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val fieldNames =
            listOf("CREATED_ON", "NAME", "KIND", "SCHEMA_NAME")
        val columnValues = List(4) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        // Identical to showObjects code, but only for tables.
        icebergConnection.listTables(namespace).forEach {
            columnValues[0].add(null)
            columnValues[1].add(it.name())
            columnValues[2].add("TABLE")
            columnValues[3].add(namespace.levels().joinToString("."))
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW TABLES for a specified namespace in Iceberg.
     *
     * @param schemaPath The schema path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     * @throws NoSuchNamespaceException if namespace cannot be found
     *
     * The method uses the .listTables(namespace) method of the respective catalog
     * to emulate SHOW TABLES.
     */
    override fun showTables(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        // Short helper to update the last element of a list.
        fun <T> MutableList<T>.setLast(value: T) {
            if (this.isNotEmpty()) {
                this[this.size - 1] = value
            }
        }
        val fieldNames =
            listOf(
                "CREATED_ON",
                "NAME",
                "SCHEMA_NAME",
                "KIND",
                "COMMENT",
                "CLUSTER_BY",
                "ROWS",
                "BYTES",
                "OWNER",
                "RETENTION_TIME",
                "AUTOMATIC_CLUSTERING",
                "CHANGE_TRACKING",
                "IS_EXTERNAL",
                "ENABLE_SCHEMA_EVOLUTION",
                "OWNER_ROLE_TYPE",
                "IS_EVENT",
                "IS_HYBRID",
                "IS_ICEBERG",
                "IS_IMMUTABLE",
            )
        val columnValues = List(fieldNames.size) { ArrayList<Any?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        icebergConnection.listTables(namespace).forEach {
            val table = icebergConnection.loadTable(it)
            // Iceberg tables do not store created timestamps, so we
            // always store them as null. All unsupported fields are
            // also null by default.
            for (i in fieldNames.indices) {
                columnValues[i].add(null)
            }
            // We only modify the supported fields.
            columnValues[fieldNames.indexOf("NAME")].setLast(it.name())
            columnValues[fieldNames.indexOf("KIND")].setLast("TABLE")
            columnValues[fieldNames.indexOf("SCHEMA_NAME")].setLast(namespace.levels().joinToString("."))
            columnValues[fieldNames.indexOf("COMMENT")].setLast(table.properties()["comment"])
            columnValues[fieldNames.indexOf("IS_ICEBERG")].setLast("Y")
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW TERSE VIEWS for a specified namespace in Iceberg.
     *
     * @param schemaPath The schema path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws NoSuchNamespaceException if namespace cannot be found
     * @throws RuntimeException if catalog does not support view operations
     *
     * The method uses the .listViews(namespace) method of the respective catalog
     * to emulate SHOW TERSE VIEWS, if the catalog supports the method.
     */
    override fun showTerseViews(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        if (icebergConnection !is ViewCatalog) {
            throw RuntimeException("SHOW VIEWS is unimplemented for the current catalog")
        }
        val fieldNames =
            listOf("CREATED_ON", "NAME", "KIND", "SCHEMA_NAME")
        val columnValues = List(4) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        icebergConnection.listViews(namespace).forEach {
            columnValues[0].add(null)
            columnValues[1].add(it.name())
            columnValues[2].add("VIEW")
            columnValues[3].add(namespace.levels().joinToString("."))
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW VIEWS for a specified namespace in Iceberg.
     *
     * @param schemaPath The schema path.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     * @throws NoSuchNamespaceException if namespace cannot be found
     * @throws RuntimeException if catalog does not support view operations
     *
     * The method uses the .listViews(namespace) method of the respective catalog
     * to emulate SHOW VIEWS, if the catalog supports the method.
     */
    override fun showViews(
        schemaPath: ImmutableList<String>,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        // Short helper to update the last element of a list.
        fun <T> MutableList<T>.setLast(value: T) {
            if (this.isNotEmpty()) {
                this[this.size - 1] = value
            }
        }
        if (icebergConnection !is ViewCatalog) {
            throw RuntimeException("SHOW VIEWS is unimplemented for the current catalog")
        }
        val fieldNames =
            listOf(
                "CREATED_ON",
                "NAME",
                "RESERVED",
                "SCHEMA_NAME",
                "COMMENT",
                "OWNER",
                "TEXT",
                "IS_SECURE",
                "IS_MATERIALIZED",
                "OWNER_ROLE_TYPE",
                "CHANGE_TRACKING",
            )
        val columnValues = List(fieldNames.size) { ArrayList<String?>() }
        val namespace = Namespace.of(*schemaPath.toTypedArray())
        icebergConnection.listViews(namespace).forEach {
            val view = icebergConnection.loadView(it)
            // Iceberg tables do not store created timestamps, so we
            // always store them as null. All unsupported fields are
            // also null by default.
            for (i in fieldNames.indices) {
                columnValues[i].add(null)
            }
            // We only modify the supported fields.
            columnValues[fieldNames.indexOf("NAME")].setLast(it.name())
            columnValues[fieldNames.indexOf("SCHEMA_NAME")].setLast(namespace.levels().joinToString("."))
            columnValues[fieldNames.indexOf("COMMENT")].setLast(view.properties()["comment"])
        }
        return DDLExecutionResult(fieldNames, columnValues, returnTypes)
    }

    /**
     * Emulates SHOW TBLPROPERTIES [(property)] for a specified table in Iceberg.
     *
     * @param tablePath The table path.
     * @param property The specific table property to display. If not specified, will be null.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws Exception if a property is specified but cannot be found
     *
     * The method uses the .listViews(namespace) method of the respective catalog
     * to emulate SHOW TERSE VIEWS, if the catalog supports the method.
     */
    override fun showTableProperties(
        tablePath: ImmutableList<String>,
        property: SqlLiteral?,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableIdentifier = tablePathToTableIdentifier(tablePath.subList(0, tablePath.size - 1), Util.last(tablePath))
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        val names: List<String>
        val columnValues: List<ArrayList<String?>>

        if (property != null) {
            names =
                listOf(
                    "VALUE",
                )
            columnValues = List(1) { ArrayList<String?>() }
            val key = property.toValue()
            val value =
                table.properties()[key]
                    ?: throw Exception("The property $key was not found.")
            columnValues[0].add(value)
        } else {
            names =
                listOf(
                    "KEY",
                    "VALUE",
                )
            columnValues = List(2) { ArrayList<String?>() }
            // Show all properties
            table.properties().forEach {
                columnValues[0].add(it.key)
                columnValues[1].add(it.value)
            }
        }
        return DDLExecutionResult(names, columnValues, returnTypes)
    }

    /**
     * Used to convert SQL relDataTypes into Iceberg-compatible types.
     *
     * @param dataType The RelDataType.
     * @return The appropriate Iceberg type corresponding to the dataType.
     */
    fun relDataTypeToIcebergType(dataType: RelDataType): Type =
        when {
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
            dataType.sqlTypeName == SqlTypeName.BIGINT -> Types.LongType.get()
            SqlTypeFamily.INTEGER.contains(dataType) -> Types.IntegerType.get()
            SqlTypeFamily.DECIMAL.contains(dataType) -> Types.DecimalType.of(dataType.precision, dataType.scale)
            dataType.sqlTypeName == SqlTypeName.DOUBLE -> Types.DoubleType.get()
            dataType.sqlTypeName == SqlTypeName.REAL -> Types.DoubleType.get()
            dataType.sqlTypeName == SqlTypeName.FLOAT -> Types.FloatType.get()
            SqlTypeFamily.NUMERIC.contains(dataType) -> Types.DecimalType.of(dataType.precision, dataType.scale)
            SqlTypeFamily.BOOLEAN.contains(dataType) -> Types.BooleanType.get()
            SqlTypeFamily.DATE.contains(dataType) -> Types.DateType.get()
            SqlTypeFamily.TIME.contains(dataType) -> Types.TimeType.get()
            SqlTypeFamily.ARRAY.contains(dataType) -> Types.ListType.ofOptional(0, relDataTypeToIcebergType(dataType.componentType!!))
            SqlTypeFamily.MAP.contains(
                dataType,
            ) ->
                Types.MapType.ofOptional(
                    0,
                    0,
                    relDataTypeToIcebergType(dataType.keyType!!),
                    relDataTypeToIcebergType(dataType.valueType!!),
                )
            else -> throw Exception("Unsupported data type $dataType in Iceberg view")
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
                        Types.NestedField.required(schemaId++, it.name, relDataTypeToIcebergType(it.type))
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

    override fun describeView(
        viewPath: ImmutableList<String>,
        typeFactory: RelDataTypeFactory,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        if (icebergConnection is ViewCatalog) {
            val names =
                listOf(
                    "NAME",
                    "TYPE",
                    "KIND",
                    "NULL?",
                    "DEFAULT",
                    "PRIMARY_KEY",
                    "UNIQUE_KEY",
                    "CHECK",
                    "EXPRESSION",
                    "COMMENT",
                    "POLICY NAME",
                    "PRIVACY DOMAIN",
                )
            val columnValues = List(12) { ArrayList<String?>() }
            val viewIdentifier = tablePathToTableIdentifier(viewPath.subList(0, viewPath.size - 1), Util.last(viewPath))
            val view = icebergConnection.loadView(viewIdentifier)
            val schema = view.schema()
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
                // Five new columns from Snowflake that is set to null in Iceberg
                for (idx in 7..11) columnValues[idx].add(null)
            }
            return DDLExecutionResult(names, columnValues, returnTypes)
        }
        throw RuntimeException("DESCRIBE VIEW is unimplemented for the current catalog")
    }

    /**
     * Renames a table in Iceberg using ALTER TABLE RENAME TO. Supports the IF EXISTS clause through
     * the ifExists parameter.
     *
     * @param tablePath The path of the table to rename.
     * @param renamePath The new path of the table.
     * @param ifExists Whether to use the IF EXISTS clause.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return The result of the operation.
     * @throws NoSuchTableException If the table does not exist and IF EXISTS clause not present.
     * @throws AlreadyExistsException If the renamed table already exists.
     * @throws Exception for any other exception thrown.
     */
    override fun renameTable(
        tablePath: ImmutableList<String>,
        renamePath: ImmutableList<String>,
        ifExists: Boolean,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val renameName = renamePath[renamePath.size - 1]
        val renameSchema =
            if (renamePath.size > 1) {
                renamePath.subList(0, renamePath.size - 1)
            } else {
                tableSchema // If a schema is not provided for the renamed table we assume it is the same schema as the original table.
            }
        // Now get identifiers
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val renameIdentifier = tablePathToTableIdentifier(renameSchema, renameName)
        return try {
            icebergConnection.renameTable(tableIdentifier, renameIdentifier)
            DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
        } catch (e: NoSuchTableException) {
            // To match snowflake's behavior, we see if it is a view and try renaming the view too.
            if (icebergConnection is ViewCatalog) {
                renameView(tablePath, renamePath, ifExists, returnTypes)
            } else if (ifExists) {
                // Need to return gracefully with an IF EXISTS clause.
                DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
            } else {
                throw e
            }
        } catch (e: AlreadyExistsException) {
            throw e
        } catch (e: Exception) {
            // Handle any other unexpected exceptions
            throw e
        }
    }

    /**
     * Renames a view in Iceberg using ALTER VIEW RENAME TO. Supports the IF EXISTS clause through
     * the ifExists parameter.
     *
     * @param viewPath The path of the view to rename.
     * @param renamePath The new path of the view.
     * @param ifExists Whether to use the IF EXISTS clause.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return The result of the operation.
     * @throws NoSuchViewException If the view does not exist and IF EXISTS clause not present.
     * @throws AlreadyExistsException If the renamed view already exists.
     * @throws RuntimeException If catalog does not support renaming views.
     * @throws Exception for any other exception thrown.
     */
    override fun renameView(
        viewPath: ImmutableList<String>,
        renamePath: ImmutableList<String>,
        ifExists: Boolean,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        if (icebergConnection is ViewCatalog) {
            val viewName = viewPath[viewPath.size - 1]
            val viewSchema = viewPath.subList(0, viewPath.size - 1)
            val renameName = renamePath[renamePath.size - 1]
            val renameSchema =
                if (renamePath.size > 1) {
                    renamePath.subList(0, renamePath.size - 1)
                } else {
                    viewSchema // If a schema is not provided for the renamed view we assume it is the same schema as the original view.
                }
            // Now get identifiers
            val viewIdentifier = tablePathToTableIdentifier(viewSchema, viewName)
            val renameIdentifier = tablePathToTableIdentifier(renameSchema, renameName)
            return try {
                icebergConnection.renameView(viewIdentifier, renameIdentifier)
                DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
            } catch (e: NoSuchViewException) {
                if (ifExists) {
                    // Need to return gracefully with an IF EXISTS clause.
                    DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
                } else {
                    throw e
                }
            } catch (e: AlreadyExistsException) {
                throw e
            } catch (e: Exception) {
                // Handle any other unexpected exceptions
                throw e
            }
        } else {
            throw RuntimeException("ALTER VIEW is unimplemented for the current catalog")
        }
    }

    /**
     * Drops a view from the catalog.
     * @param viewPath The path to the view to drop.
     * @return The result of the operation.
     */
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

    /**
     * Sets the properties of an Iceberg table identified by the given table path
     * using ALTER TABLE SET PROPERTIES and its variants (e.g. SET TBLPROPERTIES).
     *
     * @param tablePath The path of the table.
     * @param propertyList The list of properties to set. Must be a SqlNodeList of SqlLiteral.
     * @param valueList The list of values to set. Must be a SqlNodeList of SqlLiteral.
     * @param ifExists Flag indicating whether to set the properties only if the table exists.
     *                 If true, will not error even if the table does not exist.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return The result of the DDL execution.
     */

    override fun setProperty(
        tablePath: ImmutableList<String>,
        propertyList: SqlNodeList,
        valueList: SqlNodeList,
        ifExists: Boolean,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        var updater = table.updateProperties()

        for ((_property, _value) in propertyList.zip(valueList)) {
            val property = _property as SqlLiteral
            val value = _value as SqlLiteral
            val propertyStr = property.toValue()
            val valueStr = value.toValue()
            updater = updater.set(propertyStr, valueStr)
        }
        updater.commit()

        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }

    /**
     * Unsets (deletes) the properties of an Iceberg table identified by the given table path
     * using `ALTER TABLE UNSET PROPERTIES` and its variants (e.g. `SET TBLPROPERTIES`).
     *
     * @param tablePath The path of the table.
     * @param propertyList The list of properties to unset. Must be a SqlNodeList of SqlLitera.
     * @param ifExists Flag indicating whether to unset the properties only if the table exists.
     * @param ifPropertyExists Flag indicating whether to unset the properties only if the property exists
     *                         on the table. If this flag is not set to True, and a non-existent property is
     *                         attempted to be unset, a RuntimeException will be thrown.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return The result of the DDL execution.
     */

    override fun unsetProperty(
        tablePath: ImmutableList<String>,
        propertyList: SqlNodeList,
        ifExists: Boolean,
        ifPropertyExists: Boolean,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        var updater = table.updateProperties()
        for (_property in propertyList) {
            val property = _property as SqlLiteral
            val propertyStr = property.toValue()
            // If (IF EXISTS) for property not set, and the property doesn't exist, throw error.
            if (!ifPropertyExists && !table.properties().containsKey(propertyStr)) {
                throw RuntimeException("Property $propertyStr does not exist.")
            }
            updater = updater.remove(propertyStr)
        }
        updater.commit()

        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }

    /**
     * Adds a column of a given type for a specified table in Iceberg.
     * Effectively emulates `ALTER TABLE _ ADD COLUMN`.
     * Only supports simple column names for now.
     *
     * @param tablePath The table path to add the column to.
     * @param ifExists Do nothing if true and the table does not exist. (This is already dealt with in DDLResolverImpl so
     *                 has no effect here, but is included to match signature.)
     * @param ifNotExists Do nothing if true and the column to add already exists.
     * @param addCol SqlNode representing column details to be added (name, type, etc)
     * @param validator Validator needed to derive type information from addCol SqlNode.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     */
    override fun addColumn(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        ifNotExists: Boolean,
        addCol: SqlNode,
        validator: SqlValidator,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        // Table info
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        // Column info
        val column = addCol as SqlSnowflakeColumnDeclaration
        if (!column.name.isSimple) {
            throw RuntimeException("BodoSQL does not yet support nested columns and/or compound column names.")
        }
        val columnName = column.name.simple
        val columnType = relDataTypeToIcebergType(column.dataType.deriveType(validator))
        // Check if column exists
        for (c in table.schema().columns()) {
            val cName = c.name()
            if (columnName == cName && ifNotExists) {
                // Return early
                return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
            }
            // Otherwise, let the query throw its exception
        }
        // Execute query
        table.updateSchema().addColumn(columnName, columnType).commit()
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }

    /**
     * Drops columns for a specified table in Iceberg.
     * Effectively emulates `ALTER TABLE _ DROP COLUMN`.
     *
     * @param tablePath The table path of the column.
     * @param ifExists Do nothing if true and the table does not exist. (This is already dealt with in DDLResolverImpl so
     *                 has no effect here, but is included to match signature.)
     * @param dropCols SqlNodeList representing column names to be dropped.
     * @param ifColumnExists Do nothing if true and the columns do not exist.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     */
    override fun dropColumn(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        dropCols: SqlNodeList,
        ifColumnExists: Boolean,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        // Table info
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        var updater = table.updateSchema()
        // Execute queries
        for (col in dropCols) {
            if (col is SqlIdentifier) {
                // Build column name from compound identifier.
                // @NOTE: The column name passed into deleteColumn are period separated.
                // It can either be the case that they are column names with dots in them, or subfields.
                val colName = col.names.joinToString(separator = ".")
                try {
                    updater = updater.deleteColumn(colName)
                } catch (e: IllegalArgumentException) {
                    if (!ifColumnExists || e.message?.contains("Cannot delete missing column") != true) {
                        throw e
                    }
                }
            } else {
                throw RuntimeException("Unsupported syntax for DROP COLUMN.")
            }
        }
        updater.commit()
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }

    /**
     * Renames a column in a specified table in Iceberg.
     * Effectively emulates `ALTER TABLE _ RENAME COLUMN _ TO _`.
     *
     * @param tablePath The table path of the column.
     * @param ifExists Do nothing if true and the table does not exist. (This is already dealt with in DDLResolverImpl so
     *                 has no effect here, but is included to match signature.)
     * @param renameColOld SqlIdentifier signifying the column to rename.
     * @param renameColNew SqlIdentifier signifying what to rename renameColOld to.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     */
    override fun renameColumn(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        renameColOld: SqlIdentifier,
        renameColNew: SqlIdentifier,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        var updater = table.updateSchema()
        // Build column name from compound identifier.
        // @NOTE: The column name passed into renameColumn is period separated, and
        // identified with the same function as is used in deleteColumn.
        // It can either be the case that they are column names with dots in them, or subfields.
        val colName = renameColOld.names.joinToString(separator = ".")
        val renameColName = renameColNew.names.joinToString(separator = ".")
        try {
            updater = updater.renameColumn(colName, renameColName)
            // Commit changes
            updater.commit()
        } catch (e: ValidationException) {
            throw Exception("Column $renameColName already exists; cannot rename")
        }

        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }

    /**
     * Sets a comment on a column for a specified table in Iceberg.
     * Effectively emulates `ALTER TABLE _ ALTER COLUMN _ SET COMMENT _`.
     *
     * @param tablePath The table path of the column.
     * @param ifExists Do nothing if true and the table does not exist. (This is already dealt with in DDLResolverImpl so
     *                 has no effect here, but is included to match signature.)
     * @param column SqlIdentifier signifying the column to set the comment on.
     * @param comment SqlLiteral containing the string of the comment to set.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     */
    override fun alterColumnComment(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        column: SqlIdentifier,
        comment: SqlLiteral,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        // Build column name
        val colName = column.names.joinToString(separator = ".")
        try {
            var updater = table.updateSchema()
            updater = updater.updateColumnDoc(colName, comment.toValue())
            updater.commit()
        } catch (e: IllegalArgumentException) {
            throw Exception("Invalid column name or column does not exist: $colName")
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }

    /**
     * Changes a column to be nullable for a specified table in Iceberg.
     * Effectively emulates `ALTER TABLE _ ALTER COLUMN _ DROP NOT NULL`.
     *
     * @param tablePath The table path of the column.
     * @param ifExists Do nothing if true and the table does not exist. (This is already dealt with in DDLResolverImpl so
     *                 has no effect here, but is included to match signature.)
     * @param column SqlIdentifier signifying the column to change to nullable.
     * @param returnTypes The return types for the operation when generating the DDLExecutionResult.
     * @return DDLExecutionResult
     */
    override fun alterColumnDropNotNull(
        tablePath: ImmutableList<String>,
        ifExists: Boolean,
        column: SqlIdentifier,
        returnTypes: List<String>,
    ): DDLExecutionResult {
        val tableName = tablePath[tablePath.size - 1]
        val tableSchema = tablePath.subList(0, tablePath.size - 1)
        val tableIdentifier = tablePathToTableIdentifier(tableSchema, tableName)
        val table = icebergConnection.loadTable(tableIdentifier)
        // Metadata refresh
        table.refresh()
        // Build column name
        val colName = column.names.joinToString(separator = ".")
        try {
            var updater = table.updateSchema()
            updater = updater.makeColumnOptional(colName)
            updater.commit()
        } catch (e: IllegalArgumentException) {
            throw Exception("Invalid column name or column does not exist: $colName")
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
    }
}
