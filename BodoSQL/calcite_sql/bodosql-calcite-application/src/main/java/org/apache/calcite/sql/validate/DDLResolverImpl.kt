package org.apache.calcite.sql.validate

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.ddl.DDLExecutionResult
import com.bodosql.calcite.ddl.MissingObjectException
import com.bodosql.calcite.ddl.NamespaceAlreadyExistsException
import com.bodosql.calcite.ddl.NamespaceNotFoundException
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.sql.ddl.SqlDropTable
import com.bodosql.calcite.sql.ddl.SqlSnowflakeShowObjects
import com.bodosql.calcite.sql.ddl.SqlSnowflakeShowSchemas
import com.bodosql.calcite.sql.validate.BodoSqlValidator
import com.bodosql.calcite.sql.validate.DDLResolver
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.CatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.sql.SqlDescribeTable
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.ddl.SqlCreateSchema
import org.apache.calcite.sql.ddl.SqlDropSchema
import org.apache.calcite.sql.ddl.SqlCreateView
import org.apache.calcite.sql.ddl.SqlDropView
import org.apache.calcite.sql.ddl.SqlDdlNodes
import org.apache.calcite.util.Util
import java.util.function.Function

/**
 * Implementation class for the DDLResolver interface.
 * @param catalogReader The catalog reader to resolve table/column references
 * in DDL operations.
 * @param dmlValidator The validator to use for DML operations. This can
 * be used to validate sub-expressions if a DDL operation requires it.
 */
open class DDLResolverImpl(private val catalogReader: CalciteCatalogReader, private val getValidator: Function<List<String>, BodoSqlValidator>) :
    DDLResolver {
    private val scope: SqlValidatorScope = CatalogScope(EmptyScope(getValidator.apply(listOf())), ImmutableList.of("CATALOG"))

    private fun validateTable(table: BodoSqlTable, kind: SqlKind, tableName: String): CatalogTable {
        if (table !is CatalogTable) {
            throw RuntimeException("DLL Operation $kind is only supported on catalog tables. $tableName is not a catalog table.")
        }
        return table
    }

    private fun validateSchema(schema: BodoSqlSchema, kind: SqlKind, schemaName: String): CatalogSchema {
        if (schema !is CatalogSchema) {
            throw RuntimeException("DLL Operation $kind is only supported on catalog schemas. $schemaName is not a catalog schema.")
        }
        return schema
    }


    /**
     * Executes a DDL operation.
     */
    override fun executeDDL(node: SqlNode): DDLExecutionResult {
        assert (!RelationalAlgebraGenerator.isComputeKind(node.kind)) { "Node is not a DDL operation: $node" }
        return when (node.kind) {
            // No-ops, only return expected value
            SqlKind.BEGIN, SqlKind.COMMIT, SqlKind.ROLLBACK -> {
                DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")))
            }

            // Create Queries
            SqlKind.CREATE_SCHEMA -> executeCreateSchema(node as SqlCreateSchema)
            // Drop Queries
            SqlKind.DROP_SCHEMA -> executeDropSchema(node as SqlDropSchema)
            SqlKind.DROP_TABLE -> executeDropTable(node as SqlDropTable)

            SqlKind.DESCRIBE_TABLE -> {
                executeDescribeTable(node as SqlDescribeTable)
            }
            SqlKind.SHOW_OBJECTS-> {
                executeShowObjects(node as SqlSnowflakeShowObjects)
            }
            SqlKind.SHOW_SCHEMAS-> {
                executeShowSchemas(node as SqlSnowflakeShowSchemas)
            }
            SqlKind.CREATE_VIEW -> {
                executeCreateView(node as SqlCreateView)
            }
            SqlKind.DROP_VIEW -> {
                executeDropView(node as SqlDropView)
            }
            else -> {
                throw RuntimeException("Unsupported DDL operation: ${node.kind}")
            }
        }
    }

    private fun executeCreateSchema(node: SqlCreateSchema): DDLExecutionResult {
        val schemaPath = node.name.names
        val schemaPathStr = schemaPath.joinToString(separator = ".")
        val schemaName = schemaPath.last()
        val parentSchemaPath = Util.skipLast(schemaPath)
        try {
            val schema = deriveSchema(parentSchemaPath)
            val catalogSchema = validateSchema(schema, node.kind, schemaName)
            // Perform the actual create schema implementation
            catalogSchema.getDDLExecutor().createSchema(schemaPath)
        } catch (e: NamespaceAlreadyExistsException) {
            return if (node.ifNotExists) {
                DDLExecutionResult(listOf("STATUS"), listOf(listOf("'$schemaName' already exists, statement succeeded.")))
            } else {
                throw RuntimeException("Schema '$schemaPathStr' already exists.")
            }
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Schema '$schemaName' successfully created.")))
    }

    private fun executeDropSchema(node: SqlDropSchema): DDLExecutionResult {
        val schemaPath = node.name.names
        val schemaName = schemaPath.last()

        // We always need to validate the parent schema / database / namespace exists.
        val parentSchemaPath = Util.skipLast(schemaPath)
        val parentSchemaPathStr = parentSchemaPath.joinToString(separator = ".")

        val catalogSchema = try {
            val schema = deriveSchema(parentSchemaPath)
            validateSchema(schema, node.kind, parentSchemaPathStr)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Schema '$parentSchemaPathStr' does not exist or drop cannot be performed.")
        }

        try {
            catalogSchema.ddlExecutor.dropSchema(catalogSchema.fullPath, schemaName)
        } catch (e: NamespaceNotFoundException) {
            if (node.ifExists) {
                return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Schema '$schemaName' already dropped, statement succeeded.")))
            } else {
                // Already verified that the parent exists
                throw RuntimeException("Schema '$schemaName' does not exist or drop cannot be performed.")
            }
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Schema '$schemaName' successfully dropped.")))
    }

    private fun executeDropTable(node: SqlDropTable): DDLExecutionResult {
        val tablePath = node.name.names
        val tableName = tablePath.joinToString(separator = ".")
        try {
            val table = deriveTable(tablePath)
            val catalogTable = validateTable(table, node.kind, tableName)
            // Perform the actual drop table operation
            return catalogTable.getDDLExecutor().dropTable(catalogTable.fullPath, node.cascade)
        } catch (e: MissingObjectException) {
            if (node.ifExists) {
                // If drop table doesn't care if the table exists, we still need to validate the
                // schema exists.
                val schemaPath = Util.skipLast(tablePath)
                val schemaName = schemaPath.joinToString(separator = ".")
                try {
                    val schema = deriveSchema(Util.skipLast(tablePath))
                    validateSchema(schema, node.kind, schemaName)
                    return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Drop statement executed successfully ($tableName already dropped).")))
                } catch (e: MissingObjectException) {
                    throw RuntimeException("Schema '$schemaName' does not exist or not authorized.")
                }
            } else {
                throw RuntimeException("Table '$tableName' does not exist or not authorized to drop.")
            }
        }
    }

    private fun executeDescribeTable(node: SqlDescribeTable): DDLExecutionResult {
        val tablePath = node.table.names
        val tableName = tablePath.joinToString(separator = ".")
        try {
            val table = deriveTable(tablePath)
            val catalogTable = validateTable(table, node.kind, tableName)
            // Perform the actual describe table operation
            return catalogTable.getDDLExecutor().describeTable(catalogTable.fullPath, getValidator.apply(listOf()).typeFactory)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Table $tableName does not exist or not authorized to describe.")
        }
    }

    private fun executeShowObjects(node: SqlSnowflakeShowObjects): DDLExecutionResult {
        val schemaPath = node.schemaName.names
        val schemaName = schemaPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(schemaPath)
            val schemaCat = validateSchema(schema, node.kind, schemaName)
            return schemaCat.ddlExecutor.showObjects(schemaPath)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Schema $schemaName does not exist or not authorized.")
        }
    }

    private fun executeShowSchemas(node: SqlSnowflakeShowSchemas): DDLExecutionResult {
        val dbPath = node.dbName.names
        val dbName = dbPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(dbPath)
            val schemaCat = validateSchema(schema, node.kind, dbName)
            return schemaCat.ddlExecutor.showSchemas(dbPath)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Database $dbName does not exist or not authorized.")
        }
    }

    private fun executeCreateView(node: SqlCreateView): DDLExecutionResult {
        val schemaPath = node.name.names
        val schemaName = schemaPath.last()
        val parentSchemaPath = Util.skipLast(schemaPath)
        val schema = deriveSchema(parentSchemaPath)
        val catalogSchema = validateSchema(schema, node.kind, schemaName)

        // Validate the query in the context of the schema it will be created in (as opposed to the default schema -
        // this matters if both the default schema and the view's parent schema have tables with the same names, or if a
        // referenced table only exists in one of the schemas).
        // We also get the rowType so that schemas for the view can be constructed if needed.
        val validator = getValidator.apply(catalogSchema.fullPath)
        val validatedQuery = validator.validate(node.query)
        val rowDataType = validator.getValidatedNodeType(validatedQuery)

        // Construct a new SqlCreateView node with the validated query and pass it on to the catalog's executor
        val validatedCreate = SqlDdlNodes.createView(node.parserPosition, node.replace, node.name, node.columnList, validatedQuery)
        catalogSchema.ddlExecutor.createOrReplaceView(schemaPath, validatedCreate, catalogSchema, rowDataType)

        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("View '$schemaName' successfully created.")))
    }

    private fun executeDropView(node: SqlDropView): DDLExecutionResult {
        val viewPath = node.name.names
        val viewName = viewPath.joinToString(separator = ".")
        val ifexists: () -> DDLExecutionResult = {
            if (node.ifExists) {
                // If drop table doesn't care if the table exists, we still need to validate the
                // schema exists.
                val schemaPath = Util.skipLast(viewPath)
                val schemaName = schemaPath.joinToString(separator = ".")
                try {
                    val schema = deriveSchema(Util.skipLast(viewPath))
                    validateSchema(schema, node.kind, schemaName)
                    DDLExecutionResult(listOf("STATUS"), listOf(listOf("Drop statement executed successfully ($viewName already dropped).")))
                } catch (e: MissingObjectException) {
                    throw RuntimeException("Schema '$schemaName' does not exist or not authorized.")
                }
            } else {
                throw RuntimeException("View '$viewName' does not exist or not authorized to drop.")
            }
        }
        try {
            val view = deriveTable(viewPath)
            val catalogView = validateTable(view, node.kind, viewName)
            catalogView.getDDLExecutor().dropView(catalogView.fullPath)
        } catch (e: MissingObjectException) {
            ifexists()
        } catch (e: NamespaceNotFoundException) {
            ifexists()
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("View '$viewName' successfully dropped.")))
    }

    /**
     * Derive a table object from a given path.
     * @param tablePath The path to the table. This may be incomplete and need to be resolved.
     * @return A BodoSQL table object. If the table is not a BodoSQL table,
     * then we will raise an error.
     */
    private fun deriveTable(tablePath: List<String>): BodoSqlTable {
        val tableName = tablePath.joinToString(separator = ".")
        // Create an empty resolve to accumulate the results
        val resolved = SqlValidatorScope.ResolvedImpl()
        scope.resolveTable(
            tablePath,
            catalogReader.nameMatcher(),
            SqlValidatorScope.Path.EMPTY,
            resolved
        )
        if (resolved.count() != 1) {
            throw MissingObjectException("Unable to find table $tableName")
        }
        val namespace = resolved.only().namespace
        if (namespace !is TableNamespace) {
            throw RuntimeException("Table path does not resolve to a table: $tableName")
        }
        val relOptTable = namespace.table
        if (relOptTable !is RelOptTableImpl) {
            throw RuntimeException("Table is not a BodoSQL table: $tableName")
        }
        val bodoSqlTable = relOptTable.table()
        if (bodoSqlTable !is BodoSqlTable) {
            throw RuntimeException("Table is not a BodoSQL table: $tableName")
        }
        return bodoSqlTable
    }

    /**
     * Derive a schema object from a given path.
     * @param schemaPath The path to the schema. This may be incomplete and need to be resolved.
     * @return A BodoSQL schema object. If the schema is not a BodoSQL schema,
     * then we will raise an error.
     */
    private fun deriveSchema(schemaPath: List<String>): BodoSqlSchema {
        val schemaName = schemaPath.joinToString(separator = ".")
        // Create an empty resolve to accumulate the results
        val resolved = SqlValidatorScope.ResolvedImpl()
        scope.resolveSchema(
            schemaPath,
            catalogReader.nameMatcher(),
            SqlValidatorScope.Path.EMPTY,
            resolved
        )
        if (resolved.count() != 1) {
            throw MissingObjectException("Unable to find schema $schemaName")
        }
        val namespace = resolved.only().namespace
        if (namespace !is SchemaNamespace) {
            throw RuntimeException("Schema path does not resolve to a schema: $schemaName")
        }
        val schema = namespace.schema
        if (schema !is BodoSqlSchema) {
            throw RuntimeException("Schema is not a BodoSQL schema: $schemaName")
        }
        return schema
    }
}
