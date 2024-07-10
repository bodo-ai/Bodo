package com.bodosql.calcite.schema

import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.catalog.BodoSQLCatalog
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.CreateTableMetadata
import com.google.common.collect.ImmutableList
import org.apache.calcite.jdbc.CalciteSchema
import org.apache.calcite.schema.SchemaPlus
import org.apache.calcite.sql.ddl.SqlCreateTable.CreateTableType

/**
 * Implementation of the BodoSQL root schema with optional catalog support.
 * Currently, this schema doesn't have any lookup capabilities, but in the future it
 * should be able to query the catalog directly for top-level schema information.
 */
class CatalogRootSchema(catalog: BodoSQLCatalog?) : CatalogSchema(rootName, 0, ImmutableList.of(), catalog) {
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
        visitor: BodoCodeGenVisitor,
        varName: Variable?,
        tableName: String?,
        ifExists: WriteTarget.IfExistsBehavior,
        createTableType: CreateTableType?,
        meta: CreateTableMetadata,
    ): Expr? {
        throw UnsupportedOperationException("Creating a table path is not supported from the root schema")
    }

    companion object {
        @JvmStatic
        private val rootName = ""

        @JvmStatic
        private val rootMap: HashMap<BodoSQLCatalog?, CatalogRootSchema> = HashMap()

        fun createRootSchema(catalog: BodoSQLCatalog?): SchemaPlus {
            // Create the schema, loading from cache if it already exists
            val schema =
                if (rootMap.contains(catalog)) {
                    rootMap[catalog]
                } else {
                    val newSchema = CatalogRootSchema(catalog)
                    rootMap[catalog] = newSchema
                    newSchema
                }
            // Wrap in a CachingCalciteSchema
            val cachedSchema = CalciteSchema.createRootSchema(false, true, rootName, schema)
            return cachedSchema.plus()
        }
    }
}
