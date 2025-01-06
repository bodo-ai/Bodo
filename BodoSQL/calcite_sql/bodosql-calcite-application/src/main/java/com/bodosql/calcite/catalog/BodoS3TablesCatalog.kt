package com.bodosql.calcite.catalog

import com.bodosql.calcite.adapter.bodo.StreamingOptions
import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.CreateTableMetadata
import com.bodosql.calcite.table.CatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable
import software.amazon.s3tables.iceberg.S3TablesCatalog

class BodoS3TablesCatalog(
    private val warehouse: String,
) : IcebergCatalog<S3TablesCatalog>(createS3TablesCatalog(warehouse)) {
    override fun getTableNames(schemaPath: ImmutableList<String>?): MutableSet<String> {
        TODO("Not yet implemented")
    }

    override fun getTable(
        schemaPath: ImmutableList<String>?,
        tableName: String?,
    ): CatalogTable {
        TODO("Not yet implemented")
    }

    override fun getSchemaNames(schemaPath: ImmutableList<String>?): MutableSet<String> {
        TODO("Not yet implemented")
    }

    override fun getDefaultSchema(depth: Int): MutableList<String> {
        TODO("Not yet implemented")
    }

    override fun numDefaultSchemaLevels(): Int {
        TODO("Not yet implemented")
    }

    override fun generateAppendWriteCode(
        visitor: BodoCodeGenVisitor?,
        varName: Variable?,
        tableName: ImmutableList<String>?,
    ): Expr {
        TODO("Not yet implemented")
    }

    override fun generateWriteCode(
        visitor: BodoCodeGenVisitor?,
        varName: Variable?,
        tableName: ImmutableList<String>?,
        ifExists: WriteTarget.IfExistsBehavior?,
        createTableType: SqlCreateTable.CreateTableType?,
        meta: CreateTableMetadata?,
    ): Expr {
        TODO("Not yet implemented")
    }

    override fun generateReadCode(
        tableName: ImmutableList<String>?,
        useStreaming: Boolean,
        streamingOptions: StreamingOptions?,
    ): Expr {
        TODO("Not yet implemented")
    }

    override fun generateRemoteQuery(query: String?): Expr {
        TODO("Not yet implemented")
    }

    override fun schemaDepthMayContainTables(depth: Int): Boolean {
        TODO("Not yet implemented")
    }

    override fun schemaDepthMayContainSubSchemas(depth: Int): Boolean {
        TODO("Not yet implemented")
    }

    override fun generatePythonConnStr(schemaPath: ImmutableList<String>?): Expr {
        TODO("Not yet implemented")
    }

    override fun getCreateTableWriteTarget(
        schema: ImmutableList<String>?,
        tableName: String?,
        createTableType: SqlCreateTable.CreateTableType?,
        ifExistsBehavior: WriteTarget.IfExistsBehavior?,
        columnNamesGlobal: Variable?,
    ): WriteTarget {
        TODO("Not yet implemented")
    }

    companion object {
        /**
         * Create a RESTCatalog object from the given connection string.
         * @param connStr The connection string to the REST catalog.
         * @return The RESTCatalog object.
         */
        @JvmStatic
        private fun createS3TablesCatalog(warehouse: String): S3TablesCatalog {
            val catalog = S3TablesCatalog()
            catalog.initialize("S3TablesCatalog", mapOf(Pair("warehouse", warehouse)))
            return catalog
        }
    }
}
