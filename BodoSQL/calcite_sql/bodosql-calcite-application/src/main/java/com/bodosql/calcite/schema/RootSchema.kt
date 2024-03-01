package com.bodosql.calcite.schema

import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata
import com.bodosql.calcite.table.BodoSqlTable
import com.bodosql.calcite.table.CatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.DataContext
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.linq4j.tree.Expression
import org.apache.calcite.linq4j.tree.Expressions
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.schema.Table
import org.apache.calcite.sql.ddl.SqlCreateTable.CreateTableType
import org.apache.calcite.util.BuiltInMethod

/**
 * Implementation of the BodoSQL root schema with optional catalog support.
 * Currently, this schema doesn't have any lookup capabilities, but in the future it
 * should be able to query the catalog directly for top-level schema information.
 */
class RootSchema {
    /**
     * Implementation of a root schema with a catalog.
     */
    private class CatalogRootSchema(catalog: BodoSQLCatalog) : CatalogSchema(rootName, rootDepth, ImmutableList.of(), catalog) {
        // The root schema should not support loading tables.
        // In a followup PR we will refactor getTableNames()
        // and getTable() for the CatalogSchema to enable multiple
        // databases and at that time we can delete this code.
        override fun getTableNames(): Set<String> {
            return HashSet()
        }

        override fun getTable(name: String): CatalogTable? {
            return null
        }

        // The path for the root schema should be empty, not "".
        // This is useful in case we need to build a schema from its
        // parents.
        override fun getFullPath(): ImmutableList<String> {
            return ImmutableList.of()
        }

        // These are operations that are disabled.
        override fun createTablePath(tableName: String?): ImmutableList<String?>? {
            throw UnsupportedOperationException("Creating a table path is not supported from the root schema")
        }

        override fun generateWriteCode(
            visitor: PandasCodeGenVisitor,
            varName: Variable,
            tableName: String,
            ifExists: IfExistsBehavior,
            createTableType: CreateTableType,
            meta: SnowflakeCreateTableMetadata,
        ): Expr? {
            throw UnsupportedOperationException("Creating a table path is not supported from the root schema")
        }
    }

    /**
     * Implementation of a root schema without a catalog.
     */
    private class LocalRootSchema : LocalSchema(rootName, rootDepth) {
        // These are operations that are disabled
        override fun getTable(name: String): Table? {
            return null
        }

        override fun addTable(table: BodoSqlTable) {}

        override fun removeTable(tableName: String) {}

        override fun getTableNames(): Set<String> {
            return HashSet()
        }

        // Copied from Calcite root schema implementation.
        // This is just used for testing.
        override fun getExpression(
            parentSchema: SchemaPlus?,
            name: String,
        ): Expression? {
            return Expressions.call(
                DataContext.ROOT,
                BuiltInMethod.DATA_CONTEXT_GET_ROOT_SCHEMA.method,
            )
        }
    }

    companion object {
        @JvmStatic
        private val rootName = ""

        @JvmStatic
        private val rootDepth = 0

        @JvmStatic
        private val rootMap: HashMap<BodoSQLCatalog?, BodoSqlSchema> = HashMap()

        fun createRootSchema(catalog: BodoSQLCatalog?): SchemaPlus {
            // Create the schema, loading from cache if it already exists
            val schema =
                if (rootMap.contains(catalog)) {
                    rootMap[catalog]
                } else {
                    val newSchema =
                        if (catalog == null) {
                            LocalRootSchema()
                        } else {
                            CatalogRootSchema(catalog)
                        }
                    rootMap[catalog] = newSchema
                    newSchema
                }
            // Wrap in a CachingCalciteSchema
            val cachedSchema = CalciteSchema.createRootSchema(false, true, rootName, schema)
            return cachedSchema.plus()
        }
    }
}
