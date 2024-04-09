package org.apache.calcite.sql.validate

import com.bodosql.calcite.ddl.DDLExecutionResult
import com.bodosql.calcite.schema.BodoSqlSchema
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.sql.ddl.SqlDropTable
import com.bodosql.calcite.sql.validate.BodoSqlValidator
import com.bodosql.calcite.sql.validate.DDLResolver
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.CatalogTable
import com.google.common.collect.ImmutableList
import java.lang.Exception
import org.apache.calcite.prepare.CalciteCatalogReader
import org.apache.calcite.prepare.RelOptTableImpl
import org.apache.calcite.sql.SqlKind
import org.apache.calcite.sql.SqlNode
import org.apache.calcite.util.Util

/**
 * Implementation class for the DDLResolver interface.
 * @param catalogReader The catalog reader to resolve table/column references
 * in DDL operations.
 * @param dmlValidator The validator to use for DML operations. This can
 * be used to validate sub-expressions if a DDL operation requires it.
 */
open class DDLResolverImpl(private val catalogReader: CalciteCatalogReader, private val definitionValidator: BodoSqlValidator) :
    DDLResolver {
    private val scope: SqlValidatorScope = CatalogScope(EmptyScope(definitionValidator), ImmutableList.of("CATALOG"))

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
        assert (SqlKind.DDL.contains(node.kind)) { "Node is not a DDL operation: $node" }
        return when (node.kind) {
            SqlKind.DROP_TABLE -> {
                return executeDropTable(node as SqlDropTable)
            }
            else -> {
                throw RuntimeException("Unsupported DDL operation: ${node.kind}")
            }
        }
    }

    private fun executeDropTable(node: SqlDropTable): DDLExecutionResult {
        val tablePath = node.name.names
        val tableName = tablePath.joinToString(separator = ".")
        try {
            val table = deriveTable(tablePath)
            val catalogTable = validateTable(table, node.kind, tableName)
            // Perform the actual drop table operation
            return catalogTable.ddlExecutor.dropTable(catalogTable.fullPath, node.cascade)
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
                    throw RuntimeException("Schema $schemaName does not exist or not authorized.")
                }
            } else {
                throw RuntimeException("Table $tableName does not exist or not authorized to drop.")
            }
        }
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
     * Derive a table object from a given path.
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

    private class MissingObjectException(message: String) : Exception(message)

}
