package com.bodosql.calcite.catalog

import com.bodosql.calcite.adapter.bodo.StreamingOptions
import com.bodosql.calcite.application.BodoCodeGenVisitor
import com.bodosql.calcite.application.write.IcebergWriteTarget
import com.bodosql.calcite.application.write.WriteTarget
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.sql.ddl.CreateTableMetadata
import com.bodosql.calcite.table.CatalogTable
import com.bodosql.calcite.table.IcebergCatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable
import org.apache.iceberg.aws.glue.GlueCatalog

class BodoGlueCatalog(
    private val warehouse: String,
) : IcebergCatalog<GlueCatalog>(createGlueCatalog(warehouse)) {
    /**
     * Generates the code necessary to submit the remote query to the catalog DB.
     *
     * @param query Query to submit.
     * @return The generated code.
     */
    override fun generateRemoteQuery(query: String): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Returns if a schema with the given depth is allowed to contain tables. Glue only allows Tables to be present in "databases", which maps to schemas when using iceberg. Schemas cannot be nested.
     *
     * @param depth The number of parent schemas that would need to be visited to reach the root.
     * @return true if the depth is 1 and false otherwise.
     */
    override fun schemaDepthMayContainTables(depth: Int): Boolean = depth == 1

    /**
     * Returns if a schema with the given depth is allowed to contain subSchemas.
     * Glue catalogs do not support subSchemas, so this method always returns false for non-zero values.
     *
     * @param depth The number of parent schemas that would need to be visited to reach the root.
     * @return true if the depth is 0 and false otherwise.
     */
    override fun schemaDepthMayContainSubSchemas(depth: Int): Boolean = depth == 0

    /**
     * Generate a Python connection string used to read from or write to a Catalog in Bodo's SQL
     * Python code.
     *
     *
     * TODO(jsternberg): This method is needed for the XXXToPandasConverter nodes, but exposing
     * this is a bad idea and this class likely needs to be refactored in a way that the connection
     * information can be passed around more easily.
     *
     * @param schemaPath The schema component to define the connection not including the table name.
     * @return The connection string
     */
    override fun generatePythonConnStr(schemaPath: ImmutableList<String>): Expr =
        Expr.Call("bodosql.get_glue_connection", Expr.StringLiteral(warehouse))

    /**
     * Return the desired WriteTarget for a create table operation.
     * This catalog only supports Iceberg tables, so the WriteTarget will be an IcebergWriteTarget.
     *
     * @param schema The schemaPath to the table.
     * @param tableName The name of the type that will be created.
     * @param createTableType The createTable type. This is unused by the file system catalog.
     * @param ifExistsBehavior The createTable behavior for if there is already a table defined.
     * @param columnNamesGlobal Global Variable holding the output column names.
     * @return The selected WriteTarget.
     */
    override fun getCreateTableWriteTarget(
        schema: ImmutableList<String>,
        tableName: String,
        createTableType: SqlCreateTable.CreateTableType,
        ifExistsBehavior: WriteTarget.IfExistsBehavior,
        columnNamesGlobal: Variable,
    ): WriteTarget =
        IcebergWriteTarget(
            tableName,
            schema,
            ifExistsBehavior,
            columnNamesGlobal,
            generatePythonConnStr(schema),
        )

    /**
     * Returns a set of all table names with the given schema name.
     *
     * @param schemaPath The list of schemas to traverse before finding the table.
     * @return Set of table names.
     */
    override fun getTableNames(schemaPath: ImmutableList<String>): MutableSet<String> {
        val ns = schemaPathToNamespace(schemaPath)
        return getIcebergConnection().listTables(ns).map { it.name() }.toMutableSet()
    }

    /**
     * Returns a table with the given name and found in the given schema.
     *
     * @param schemaPath The list of schemas to traverse before finding the table.
     * @param tableName Name of the table.
     * @return The table object.
     */
    override fun getTable(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): CatalogTable {
        val columns = getIcebergTableColumns(schemaPath, tableName)
        return IcebergCatalogTable(tableName, schemaPath, columns, this)
    }

    /**
     * Get the available subSchema names for the given path.
     *
     * @param schemaPath The parent schema path to check.
     * @return Set of available schema names.
     */
    override fun getSchemaNames(schemaPath: ImmutableList<String>): MutableSet<String> {
        val ns = schemaPathToNamespace(schemaPath)
        return getIcebergConnection().listNamespaces(ns).map { it.level(it.length() - 1) }.toMutableSet()
    }

    /**
     * Return the list of implicit/default schemas for the given catalog, in the order that they
     * should be prioritized during table resolution. The provided depth gives the "level" at which to
     * provide the default. Each entry in the list is a schema name at that level, not the path to reach
     * that level.
     *
     * @param depth The depth at which to find the default.
     * @return List of default Schema for this catalog.
     */
    override fun getDefaultSchema(depth: Int): List<String> {
        if (depth == 0) {
            return listOf("default")
        }
        return listOf()
    }

    /**
     * Return the number of levels at which a default schema may be found.
     * Glue catalogs don't have subSchemas, so this method always returns 1.
     * @return The number of levels a default schema can be found.
     */
    override fun numDefaultSchemaLevels(): Int = 1

    /**
     * Generates the code necessary to produce an append write expression from the given catalog.
     *
     * @param visitor The PandasCodeGenVisitor used to lower globals.
     * @param varName Name of the variable to write.
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @return The generated code to produce the append write.
     */
    override fun generateAppendWriteCode(
        visitor: BodoCodeGenVisitor,
        varName: Variable,
        tableName: ImmutableList<String>,
    ): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Generates the code necessary to produce a write expression from the given catalog.
     *
     * @param visitor The PandasCodeGenVisitor used to lower globals.
     * @param varName Name of the variable to write.
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @param ifExists Behavior to perform if the table already exists
     * @param createTableType Type of table to create if it doesn't exist
     * @param meta Expression containing the metadata information for init table information.
     * @return The generated code to produce a write.
     */
    override fun generateWriteCode(
        visitor: BodoCodeGenVisitor,
        varName: Variable,
        tableName: ImmutableList<String>,
        ifExists: WriteTarget.IfExistsBehavior,
        createTableType: SqlCreateTable.CreateTableType,
        meta: CreateTableMetadata,
    ): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Generates the code necessary to produce a read expression from the given catalog.
     *
     * @param useStreaming Should we generate code to read the table as streaming (currently only
     * supported for snowflake tables)
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @param useStreaming Should we generate code to read the table as streaming (currently only
     * supported for snowflake tables)
     * @param streamingOptions The options to use if streaming is enabled.
     * @return The generated code to produce a read.
     */
    override fun generateReadCode(
        tableName: ImmutableList<String>,
        useStreaming: Boolean,
        streamingOptions: StreamingOptions,
    ): Expr {
        TODO("Not yet implemented")
    }

    companion object {
        /**
         * Create a RESTCatalog object from the given connection string.
         * @param connStr The connection string to the REST catalog.
         * @return The RESTCatalog object.
         */
        @JvmStatic
        private fun createGlueCatalog(warehouse: String): GlueCatalog {
            val catalog = GlueCatalog()
            catalog.initialize("glueCatalog", mapOf(Pair("warehouse", warehouse)))
            return catalog
        }
    }
}