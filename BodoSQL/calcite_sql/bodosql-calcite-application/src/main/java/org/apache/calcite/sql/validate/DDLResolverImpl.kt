package org.apache.calcite.sql.validate

import com.bodosql.calcite.application.RelationalAlgebraGenerator
import com.bodosql.calcite.ddl.DDLExecutionResult
import com.bodosql.calcite.ddl.GenerateDDLTypes
import com.bodosql.calcite.ddl.MissingObjectException
import com.bodosql.calcite.ddl.NamespaceAlreadyExistsException
import com.bodosql.calcite.ddl.NamespaceNotFoundException
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.sql.ddl.SqlDropTable
import com.bodosql.calcite.sql.ddl.SqlDescribeView
import com.bodosql.calcite.sql.ddl.SqlAlterTable
import com.bodosql.calcite.sql.ddl.SqlAlterTableAlterColumn
import com.bodosql.calcite.sql.ddl.SqlAlterTableAlterColumnComment
import com.bodosql.calcite.sql.ddl.SqlAlterTableAlterColumnDropNotNull
import com.bodosql.calcite.sql.ddl.SqlAlterTableRenameTable
import com.bodosql.calcite.sql.ddl.SqlAlterTableSetProperty
import com.bodosql.calcite.sql.ddl.SqlAlterTableUnsetProperty
import com.bodosql.calcite.sql.ddl.SqlAlterTableAddCol
import com.bodosql.calcite.sql.ddl.SqlAlterTableDropCol
import com.bodosql.calcite.sql.ddl.SqlAlterTableRenameCol
import com.bodosql.calcite.sql.ddl.SqlAlterView
import com.bodosql.calcite.sql.ddl.SqlAlterViewRenameView
import com.bodosql.calcite.sql.ddl.SqlSnowflakeShowObjects
import com.bodosql.calcite.sql.ddl.SqlSnowflakeShowSchemas
import com.bodosql.calcite.sql.ddl.SqlShowTables
import com.bodosql.calcite.sql.ddl.SqlShowViews
import com.bodosql.calcite.sql.ddl.SqlShowTblproperties
import com.bodosql.calcite.sql.validate.BodoSqlValidator
import com.bodosql.calcite.sql.validate.DDLResolver
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.CatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.sql.SqlDescribeTable
import org.apache.calcite.sql.SqlDescribeSchema
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlLiteral
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.sql.ddl.SqlCreateSchema
import org.apache.calcite.sql.ddl.SqlDropSchema
import org.apache.calcite.sql.ddl.SqlCreateView
import org.apache.calcite.sql.ddl.SqlDropView
import org.apache.calcite.sql.ddl.SqlDdlNodes
import org.apache.calcite.util.Util
import java.util.function.Function
import org.apache.calcite.rel.type.RelDataType

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
    private val typeGenerator: GenerateDDLTypes = GenerateDDLTypes(catalogReader.typeFactory)

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
        val ddlType: RelDataType = typeGenerator.generateType(node)
        val returnTypes = DDLExecutionResult.toPandasDataTypeList(ddlType)

        return when (node.kind) {
            // No-ops, only return expected value
            SqlKind.BEGIN, SqlKind.COMMIT, SqlKind.ROLLBACK -> {
                DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes)
            }

            // Create Queries
            SqlKind.CREATE_SCHEMA -> executeCreateSchema(node as SqlCreateSchema, returnTypes)
            // Drop Queries
            SqlKind.DROP_SCHEMA -> executeDropSchema(node as SqlDropSchema, returnTypes)
            SqlKind.DROP_TABLE -> executeDropTable(node as SqlDropTable, returnTypes)

            // Alter Queries
            SqlKind.ALTER_TABLE -> executeAlterTable(node as SqlAlterTable, returnTypes)
            SqlKind.ALTER_VIEW -> executeAlterView(node as SqlAlterView, returnTypes)
            
            SqlKind.DESCRIBE_TABLE -> {
                executeDescribeTable(node as SqlDescribeTable, returnTypes)
            }
            SqlKind.DESCRIBE_SCHEMA -> {
                executeDescribeSchema(node as SqlDescribeSchema, returnTypes)
            }
            SqlKind.SHOW_OBJECTS-> {
                executeShowObjects(node as SqlSnowflakeShowObjects, returnTypes)
            }
            SqlKind.SHOW_SCHEMAS-> {
                executeShowSchemas(node as SqlSnowflakeShowSchemas, returnTypes)
            }
            SqlKind.SHOW_TABLES-> {
                executeShowTables(node as SqlShowTables, returnTypes)
            }
            SqlKind.SHOW_VIEWS-> {
                executeShowViews(node as SqlShowViews, returnTypes)
            }
            SqlKind.SHOW_TBLPROPERTIES-> {
                executeShowTblproperties(node as SqlShowTblproperties, returnTypes)
            }
            SqlKind.CREATE_VIEW -> {
                executeCreateView(node as SqlCreateView, returnTypes)
            }
            SqlKind.DESCRIBE_VIEW -> {
                executeDescribeView(node as SqlDescribeView, returnTypes)
            }

            SqlKind.DROP_VIEW -> {
                executeDropView(node as SqlDropView, returnTypes)
            }
            else -> {
                throw RuntimeException("Unsupported DDL operation: ${node.kind}")
            }
        }
    }

    /**
     * Executes an ALTER TABLE operation.
     * 
     * This function is called for any ALTER TABLE operation, but cases on
     * the specific type of ALTER TABLE operation to execute (implemented as
     * subclasses of the SqlAlterTable class).
     * 
     * Currently only supports RENAME TO operations.
     * 
     * @param node The ALTER TABLE node to execute.
     * @return The result of the operation.
     */
    private fun executeAlterTable(node: SqlAlterTable, returnTypes: List<String>): DDLExecutionResult{
        // Type check up front
        // When adding support for additional ALTER operations, expand this check.
        // NOTE: This check will not actually catch cases like ALTER TABLE CLUSTER BY, since
        // the SQL node for that is not implemented yet and thus the parser will just fail
        // to parse the query all together.
        if (node !is SqlAlterTableRenameTable &&
            node !is SqlAlterTableSetProperty &&
            node !is SqlAlterTableUnsetProperty &&
            node !is SqlAlterTableAddCol &&
            node !is SqlAlterTableDropCol &&
            node !is SqlAlterTableRenameCol &&
            node !is SqlAlterTableAlterColumn
            ) {
            throw RuntimeException("This DDL operation is currently unsupported.")
        }

        val tablePath = node.table.names
        val tableName = tablePath.joinToString(separator = ".")

        val table : BodoSqlTable
        val catalogTable : CatalogTable

        // Since there are multiple alter statements, we deal with the table/schema validation first,
        // which is common to all statements.
        try {
            table = deriveTable(tablePath)
            catalogTable = validateTable(table, node.kind, tableName)
        } catch (e: MissingObjectException) {
            if (node.ifExists){
                // When IF EXISTS is passed in and the node doesn't exist, we will validate the schema.
                // If valid schema we can return gracefully.
                // If the schema does not exist, we will raise an error.
                val schemaPath = Util.skipLast(tablePath)
                val schemaName = schemaPath.joinToString(separator = ".")
                try {
                    val schema = deriveSchema(schemaPath)
                    validateSchema(schema, node.kind, schemaName)
                    return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes);
                } catch (e: MissingObjectException){
                    throw RuntimeException("Schema '$schemaName' does not exist or not authorized.")
                }
            } else {
                // When IF EXISTS is not passed in, we will raise an error if the table does not exist.
                throw RuntimeException("Table '$tableName' does not exist or not authorized.")
            }
        }
        // After validation, we can then case on the specific type of ALTER TABLE.
        val validator = getValidator.apply(listOf())
        return when (node) {
            is SqlAlterTableRenameTable -> {
                catalogTable.getDDLExecutor().renameTable(catalogTable.fullPath, node.renameName.names, node.ifExists, returnTypes)
            }
            is SqlAlterTableSetProperty -> {
                catalogTable.getDDLExecutor().setProperty(catalogTable.fullPath, node.propertyList, node.valueList, node.ifExists, returnTypes)
            }
            is SqlAlterTableUnsetProperty -> {
                catalogTable.getDDLExecutor().unsetProperty(catalogTable.fullPath, node.propertyList, node.ifExists, node.ifPropertyExists, returnTypes)
            }
            is SqlAlterTableAddCol -> {
                catalogTable.getDDLExecutor().addColumn(catalogTable.fullPath,node.ifExists, node.ifNotExists, node.addCol, validator, returnTypes)
            }
            is SqlAlterTableDropCol -> {
                catalogTable.getDDLExecutor().dropColumn(catalogTable.fullPath,node.ifExists, node.dropCols, node.ifColumnExists, returnTypes)
            }
            is SqlAlterTableRenameCol -> {
                catalogTable.getDDLExecutor().renameColumn(catalogTable.fullPath,node.ifExists, node.renameColOld, node.renameColNew, returnTypes)
            }
            is SqlAlterTableAlterColumnComment -> {
                catalogTable.getDDLExecutor().alterColumnComment(catalogTable.fullPath,node.ifExists, node.column, node.comment as SqlLiteral, returnTypes)
            }
            is SqlAlterTableAlterColumnDropNotNull -> {
                catalogTable.getDDLExecutor().alterColumnDropNotNull(catalogTable.fullPath,node.ifExists, node.column, returnTypes)
            }
            else -> throw RuntimeException("This DDL operation is currently unsupported.") // Should not be here anyway, since type check is up front.
        }
    }

    /**
     * Executes an ALTER VIEW operation.
     * 
     * This function is called for any ALTER VIEW operation, but cases on
     * the specific type of ALTER VIEW operation to execute (implemented as
     * subclasses of the SqlAlterView class).
     * Note: The function is virtually identical to executeAlterTable, but is kept separate/
     * Refer to executeAlterTable for more detailed comments.
     * 
     * Currently only supports RENAME TO operations.
     * 
     * @param node The ALTER VIEW node to execute.
     * @return The result of the operation.
     */
    private fun executeAlterView(node: SqlAlterView, returnTypes: List<String>): DDLExecutionResult{
        // Type check up front
        // When adding support for additional ALTER operations, expand this check.
        if (node !is SqlAlterViewRenameView) {
            throw RuntimeException("This DDL operation is currently unsupported.")
        }

        val viewPath = node.view.names
        val viewName = viewPath.joinToString(separator = ".")
        val view : BodoSqlTable
        val catalogView : CatalogTable

        // Since there are multiple alter statements, we deal with the validation first.
        try {
            view = deriveTable(viewPath)
            catalogView = validateTable(view, node.kind, viewName)
        } catch (e: MissingObjectException) {
            if (node.ifExists){
                // We still need to validate the schema.
                val schemaPath = Util.skipLast(viewPath)
                val schemaName = schemaPath.joinToString(separator = ".")
                try {
                    val schema = deriveSchema(schemaPath)
                    validateSchema(schema, node.kind, schemaName)
                    return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Statement executed successfully.")), returnTypes);
                } catch (e: MissingObjectException){
                    throw RuntimeException("Schema '$schemaName' does not exist or not authorized.")
                }
            } else {
                throw RuntimeException("View '$viewName' does not exist or not authorized.")
            }
        }
        // After validation, we can then case on the type of ALTER.
        return when (node) {
            is SqlAlterViewRenameView -> {
                catalogView.getDDLExecutor().renameView(catalogView.fullPath, node.renameName.names, node.ifExists, returnTypes)
            }
            else -> throw RuntimeException("This DDL operation is currently unsupported.")
        }
    }

    private fun executeCreateSchema(node: SqlCreateSchema, returnTypes: List<String>): DDLExecutionResult {
        val schemaPath = node.name.names
        val schemaPathStr = schemaPath.joinToString(separator = ".")
        val schemaName = schemaPath.last()
        val parentSchemaPath = Util.skipLast(schemaPath)
        try {
            val schema = deriveSchema(parentSchemaPath)
            val catalogSchema = validateSchema(schema, node.kind, schemaName)
            // Perform the actual create schema implementation
            catalogSchema.ddlExecutor.createSchema(schemaPath)
        } catch (e: NamespaceAlreadyExistsException) {
            return if (node.ifNotExists) {
                DDLExecutionResult(listOf("STATUS"), listOf(listOf("'$schemaName' already exists, statement succeeded.")), returnTypes)
            } else {
                throw RuntimeException("Schema '$schemaPathStr' already exists.")
            }
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Schema '$schemaName' successfully created.")), returnTypes)
    }

    private fun executeDropSchema(node: SqlDropSchema, returnTypes: List<String>): DDLExecutionResult {
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
                return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Schema '$schemaName' already dropped, statement succeeded.")), returnTypes)
            } else {
                // Already verified that the parent exists
                throw RuntimeException("Schema '$schemaName' does not exist or drop cannot be performed.")
            }
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Schema '$schemaName' successfully dropped.")), returnTypes)
    }

    private fun executeDropTable(node: SqlDropTable, returnTypes: List<String>): DDLExecutionResult {
        val tablePath = node.name.names
        val tableName = tablePath.joinToString(separator = ".")
        try {
            val table = deriveTable(tablePath)
            val catalogTable = validateTable(table, node.kind, tableName)
            // Perform the actual drop table operation
            return catalogTable.getDDLExecutor().dropTable(catalogTable.fullPath, node.cascade, node.purge, returnTypes)
        } catch (e: MissingObjectException) {
            if (node.ifExists) {
                // If drop table doesn't care if the table exists, we still need to validate the
                // schema exists.
                val schemaPath = Util.skipLast(tablePath)
                val schemaName = schemaPath.joinToString(separator = ".")
                try {
                    val schema = deriveSchema(Util.skipLast(tablePath))
                    validateSchema(schema, node.kind, schemaName)
                    return DDLExecutionResult(listOf("STATUS"), listOf(listOf("Drop statement executed successfully ($tableName already dropped).")), returnTypes)
                } catch (e: MissingObjectException) {
                    throw RuntimeException("Schema '$schemaName' does not exist or not authorized.")
                }
            } else {
                throw RuntimeException("Table '$tableName' does not exist or not authorized to drop.")
            }
        }
    }

    private fun executeDescribeTable(node: SqlDescribeTable, returnTypes: List<String>): DDLExecutionResult {
        val tablePath = node.table.names
        val tableName = tablePath.joinToString(separator = ".")
        try {
            val table = deriveTable(tablePath)
            val catalogTable = validateTable(table, node.kind, tableName)
            // Perform the actual describe table operation
            return catalogTable.getDDLExecutor().describeTable(catalogTable.fullPath, getValidator.apply(listOf()).typeFactory, returnTypes)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Table $tableName does not exist or not authorized to describe.")
        }
    }

    /**
     * Executes the "DESCRIBE SCHEMA" command for a specified schema and returns the result.
     *
     * @param node The SqlSnowflakeShowObjects SqlNode
     * @return DDLExecutionResult containing the details of the schema.
     * @throws RuntimeException if the schema does not exist or the user is not authorized to access it.
     */
    private fun executeDescribeSchema(node: SqlDescribeSchema, returnTypes: List<String>): DDLExecutionResult {
        val schemaPath = node.schema.names
        val schemaName = schemaPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(schemaPath)
            val schemaCat = validateSchema(schema, node.kind, schemaName)
            return schemaCat.ddlExecutor.describeSchema(schemaPath, returnTypes)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Schema $schemaName does not exist or not authorized.")
        }
    }


    /**
     * Executes the "SHOW OBJECTS" command for a specified schema and returns the result.
     *
     * @param node The SqlSnowflakeShowObjects SqlNode
     * @return DDLExecutionResult containing the details of the objects in the specified schema.
     * @throws RuntimeException if the schema does not exist or the user is not authorized to access it.
     */
    private fun executeShowObjects(node: SqlSnowflakeShowObjects, returnTypes: List<String>): DDLExecutionResult {
        val schemaPath = node.schemaName.names
        val schemaName = schemaPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(schemaPath)
            val schemaCat = validateSchema(schema, node.kind, schemaName)
            return if (node.isTerse){
                schemaCat.ddlExecutor.showTerseObjects(schemaPath, returnTypes)
            } else {
                schemaCat.ddlExecutor.showObjects(schemaPath, returnTypes)
            }
        } catch (e: MissingObjectException) {
            throw RuntimeException("Schema $schemaName does not exist or not authorized.")
        }
    }

    /**
     * Executes the "SHOW SCHEMAS" command for a specified DB and returns the result.
     *
     * @param node The SqlSnowflakeShowSchemas SqlNode
     * @return DDLExecutionResult containing the details of the schemas in the specified DB.
     * @throws RuntimeException if the DB does not exist or the user is not authorized to access it.
     */
    private fun executeShowSchemas(node: SqlSnowflakeShowSchemas, returnTypes: List<String>): DDLExecutionResult {
        val dbPath = node.dbName.names
        val dbName = dbPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(dbPath)
            val schemaCat = validateSchema(schema, node.kind, dbName)
            return if (node.isTerse){
                schemaCat.ddlExecutor.showTerseSchemas(dbPath, returnTypes)
            } else {
                schemaCat.ddlExecutor.showSchemas(dbPath, returnTypes)
            }

        } catch (e: MissingObjectException) {
            throw RuntimeException("Database $dbName does not exist or not authorized.")
        }
    }

    /**
     * Executes the "SHOW TABLES" command for a specified schema and returns the result.
     *
     * @param node The SqlShowTables SqlNode
     * @return DDLExecutionResult containing the details of the tables in the specified schema.
     * @throws RuntimeException if the schema does not exist or the user is not authorized to access it.
     */
    private fun executeShowTables(node: SqlShowTables, returnTypes: List<String>): DDLExecutionResult {

        val schemaPath = node.schemaName.names
        val schemaName = schemaPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(schemaPath)
            val schemaCat = validateSchema(schema, node.kind, schemaName)
            return if (node.isTerse){
                schemaCat.ddlExecutor.showTerseTables(schemaPath, returnTypes)
            } else {
                schemaCat.ddlExecutor.showTables(schemaPath, returnTypes)
            }
        } catch (e: MissingObjectException) {
            throw RuntimeException("Schema $schemaName does not exist or not authorized.")
        }
    }

    /**
     * Executes the "SHOW VIEWS" command for a specified schema and returns the result.
     *
     * @param node The SqlShowViews SqlNode
     * @return DDLExecutionResult containing the details of the tables in the specified schema.
     * @throws RuntimeException if the schema does not exist or the user is not authorized to access it.
     */
    private fun executeShowViews(node: SqlShowViews, returnTypes: List<String>): DDLExecutionResult {
        val schemaPath = node.schemaName.names
        val schemaName = schemaPath.joinToString(separator = ".")
        try {
            val schema = deriveSchema(schemaPath)
            val schemaCat = validateSchema(schema, node.kind, schemaName)
            return if (node.isTerse){
                schemaCat.ddlExecutor.showTerseViews(schemaPath, returnTypes)
            } else {
                schemaCat.ddlExecutor.showViews(schemaPath, returnTypes)
            }
        } catch (e: MissingObjectException) {
            throw RuntimeException("Schema $schemaName does not exist or not authorized.")
        }
    }

    /**
     * Executes the "SHOW TBLPROPERTIES" command for a specified table and returns the result.
     *
     * @param node The SqlShowTblproperties SqlNode
     * @return DDLExecutionResult containing the properties of the table.
     * @throws RuntimeException if the table does not exist or the user is not authorized to access it.
     */
    private fun executeShowTblproperties(node: SqlShowTblproperties, returnTypes: List<String>): DDLExecutionResult {
        val tablePath = node.table.names
        val tableName = tablePath.joinToString(separator = ".")
        try {
            val table = deriveTable(tablePath)
            val catalogTable = validateTable(table, node.kind, tableName)
            // Perform the actual describe table operation
            return catalogTable.getDDLExecutor().showTableProperties(catalogTable.fullPath, node.property, returnTypes)
        } catch (e: MissingObjectException) {
            throw RuntimeException("Table $tableName does not exist or not authorized to show properties.")
        }
    }

    private fun executeCreateView(node: SqlCreateView, returnTypes: List<String>): DDLExecutionResult {
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

        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("View '$schemaName' successfully created.")), returnTypes)
    }

    private fun executeDescribeView(node: SqlDescribeView, returnTypes: List<String>): DDLExecutionResult {
        val viewPath = node.view.names
        val viewName = viewPath.joinToString(separator = ".")
        try {
            val view = deriveTable(viewPath)
            val catalogTable = validateTable(view, node.kind, viewName)
            return catalogTable.getDDLExecutor().describeView(catalogTable.fullPath, getValidator.apply(listOf()).typeFactory, returnTypes)
        } catch (e: MissingObjectException) {
            throw RuntimeException("View '$viewName' does not exist or not authorized to describe.")
        }
    }
    
    private fun executeDropView(node: SqlDropView, returnTypes: List<String>): DDLExecutionResult {
        val viewPath = node.name.names
        val viewName = viewPath.joinToString(separator = ".")
        val ifExists: () -> DDLExecutionResult = {
            if (node.ifExists) {
                // If drop table doesn't care if the table exists, we still need to validate the
                // schema exists.
                val schemaPath = Util.skipLast(viewPath)
                val schemaName = schemaPath.joinToString(separator = ".")
                try {
                    val schema = deriveSchema(Util.skipLast(viewPath))
                    validateSchema(schema, node.kind, schemaName)
                    DDLExecutionResult(listOf("STATUS"), listOf(listOf("Drop statement executed successfully ($viewName already dropped).")), returnTypes)
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
            return ifExists()
        } catch (e: NamespaceNotFoundException) {
            return ifExists()
        }
        return DDLExecutionResult(listOf("STATUS"), listOf(listOf("View '$viewName' successfully dropped.")), returnTypes)
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
        val singleResolve = resolved.only()
        // resolveTable will give the namespace of the schema if the table cannot
        // be found, although this is not documented.
        if (singleResolve.remainingNames.isNotEmpty()) {
            throw MissingObjectException("Unable to find table $tableName")
        }
        val namespace = singleResolve.namespace
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
