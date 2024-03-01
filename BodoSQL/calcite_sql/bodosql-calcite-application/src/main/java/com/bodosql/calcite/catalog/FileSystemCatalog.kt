package com.bodosql.calcite.catalog

import com.bodosql.calcite.adapter.pandas.StreamingOptions
import com.bodosql.calcite.application.PandasCodeGenVisitor
import com.bodosql.calcite.ir.Expr
import com.bodosql.calcite.ir.Variable
import com.bodosql.calcite.schema.CatalogSchema
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata
import com.bodosql.calcite.table.CatalogTable
import com.bodosql.calcite.table.IcebergCatalogTable
import com.google.common.collect.ImmutableList
import org.apache.calcite.sql.ddl.SqlCreateTable
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.FileSystem
import org.apache.hadoop.fs.Path
import org.apache.iceberg.hadoop.HadoopCatalog
import org.apache.iceberg.hadoop.Util
import org.apache.iceberg.util.LocationUtil
import java.io.FileNotFoundException

/**
 * Implementation of a BodoSQLCatalog catalog for a filesystem catalog implementation.
 * Within the file system a schema is referred to by a directory and a table is either
 * a file or a directory with some "contents" that define the table type.
 *
 * This class can be used to implement any generic file system with a
 * connection string, but certain file systems may require additional dependencies
 * which may be more suitable for subclasses.
 *
 * Generating code for iceberg tables has been manually tested locally and on s3
 *
 * Note: Hardcoding HadoopCatalog will likely change in the future to be more abstract/robust.
 */
class FileSystemCatalog(connStr: String) : IcebergCatalog(createHadoopCatalog(connStr)) {
    private val fs = createFileSystem(connStr)

    /**
     * Convert a given schemaPath which is a list of components to the
     * location in the file system.
     * @param schemaPath The list of schemas to traverse.
     * @return A converted schemaPath to the file system representation
     * of the location.
     */
    private fun schemaPathToFilePath(schemaPath: ImmutableList<String>): Path {
        return if (schemaPath.size != 0) fs.resolvePath(Path(schemaPath.joinToString(separator = "/"))) else fs.workingDirectory
    }

    /**
     * Convert a given schemaPath and tableName, which is the "path" to a table,
     * location in the file system.
     * @param schemaPath The list of schemas to traverse.
     * @param tableName The name of the table.
     * @return A converted path to the file system representation
     * of the location.
     */
    private fun tableInfoToFilePath(
        schemaPath: ImmutableList<String>,
        tableName: String,
    ): Path {
        return Path(schemaPathToFilePath(schemaPath), Path(tableName))
    }

    private fun getDirectoryContents(path: Path): List<Path> {
        return fs.listStatus(path).map { it.path }
    }

    /**
     * Determine if the given path information refers to a schema. Within the
     * FileSystemCatalog a schema is any directory that doesn't map to a table.
     * @param path The file system path.
     * @return Is this path referring to a schema?
     */
    private fun isSchema(path: Path): Boolean {
        return !isTable(path) && fs.getFileStatus(path).isDirectory
    }

    /**
     * Determine if the given path information refers to a table. Within the
     * FileSystemCatalog a table is either a directory with some special
     * "indicator" contents or it's an individual file that's directly
     * supported. Files that are unknown or unsupported are not considered
     * tables.
     * @param path The file system path.
     * @return Is this path referring to a table?
     */
    private fun isTable(path: Path): Boolean {
        return isIcebergTable(path)
    }

    /**
     * Is the given path an Iceberg table. An iceberg table is defined
     * as a directory that contains a metadata directory in a
     * plain file system.
     * @param path The file system path.
     * @return Is this path referring to an Iceberg table?
     */
    private fun isIcebergTable(path: Path): Boolean {
        return if (!fs.getFileStatus(path).isDirectory) {
            false
        } else {
            val metadataPath = Path(path, Path("metadata"))
            val metadataStatus =
                try {
                    fs.getFileStatus(metadataPath)
                } catch (e: FileNotFoundException) {
                    null
                }
            return metadataStatus?.isDirectory ?: false
        }
    }

    /**
     * Returns a set of all table names with the given schema name.
     *
     * @param schemaPath The list of schemas to traverse before finding the table.
     * @return Set of table names.
     */
    override fun getTableNames(schemaPath: ImmutableList<String>): Set<String> {
        val fullPath = schemaPathToFilePath(schemaPath)
        val elements = getDirectoryContents(fullPath).filter { isTable(it) }.map { it.name }
        return elements.toSet()
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
        val fullPath = tableInfoToFilePath(schemaPath, tableName)
        if (!isIcebergTable(fullPath)) {
            throw RuntimeException("FileSystemCatalog Error: Only Iceberg tables are currently supported.")
        }
        val columns = getIcebergTableColumns(schemaPath, tableName)
        return IcebergCatalogTable(tableName, schemaPath, columns, this)
    }

    /**
     * Get the available subSchema names for the given path.
     *
     * @param schemaPath The parent schema path to check.
     * @return Set of available schema names.
     */
    override fun getSchemaNames(schemaPath: ImmutableList<String>): Set<String> {
        val fullPath = schemaPathToFilePath(schemaPath)
        val elements = getDirectoryContents(fullPath).filter { isSchema(it) }.map { it.name }
        return elements.toSet()
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
    ): CatalogSchema {
        return CatalogSchema(schemaName, schemaPath.size + 1, schemaPath, this)
    }

    /**
     * Return the list of implicit/default schemas for the given catalog, in the order that they
     * should be prioritized during table resolution. The provided depth gives the "level" at which to
     * provide the default.
     *
     * @param depth The depth at which to find the default.
     * @return List of default Schema for this catalog.
     */
    override fun getDefaultSchema(depth: Int): List<String> {
        return listOf()
    }

    /**
     * Generates the code necessary to produce an append write expression from the given catalog.
     *
     * @param varName Name of the variable to write.
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @return The generated code to produce the append write.
     */
    override fun generateAppendWriteCode(
        visitor: PandasCodeGenVisitor?,
        varName: Variable?,
        tableName: ImmutableList<String>?,
    ): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Generates the code necessary to produce a write expression from the given catalog.
     *
     * @param varName Name of the variable to write.
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @param ifExists Behavior to perform if the table already exists
     * @param createTableType Type of table to create if it doesn't exist
     * @return The generated code to produce a write.
     */
    override fun generateWriteCode(
        visitor: PandasCodeGenVisitor?,
        varName: Variable?,
        tableName: ImmutableList<String>?,
        ifExists: BodoSQLCatalog.ifExistsBehavior?,
        createTableType: SqlCreateTable.CreateTableType?,
        meta: SnowflakeCreateTableMetadata?,
    ): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Generates the code necessary to produce the streaming write initialization code from the given
     * catalog for an append operation.
     *
     * @param operatorID ID of operation to use for retrieving memory budget.
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @return The generated code to produce the write-initialization code
     */
    override fun generateStreamingAppendWriteInitCode(
        operatorID: Expr.IntegerLiteral?,
        tableName: ImmutableList<String>?,
    ): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Generates the code necessary to produce the streaming write initialization code from the given
     * catalog.
     *
     * @param operatorID ID of operation to use for retrieving memory budget.
     * @param tableName The path of schema used to reach the table from the root that includes the
     * table.
     * @param ifExists Behavior to perform if the table already exists
     * @param createTableType Type of table to create if it doesn't exist
     * @param colNamesGlobal Column names of table to write
     * @param icebergBase path for writing Iceberg table data (excluding volume bucket path)
     * @return The generated code to produce the write-initialization code
     */
    override fun generateStreamingWriteInitCode(
        operatorID: Expr.IntegerLiteral,
        tableName: ImmutableList<String>,
        ifExists: BodoSQLCatalog.ifExistsBehavior,
        createTableType: SqlCreateTable.CreateTableType,
        colNamesGlobal: Variable,
        icebergBase: String,
    ): Expr {
        TODO("Not yet implemented")
    }

    /**
     * Generates the code necessary to produce code to append tables to the streaming writer of the
     * given object.
     *
     * @param stateVarName Name of the variable of the write state
     * @param tableVarName Name of the variable storing the table to append
     * @param colNamesGlobal Column names of table to append
     * @param isLastVarName Name of the variable indicating the is_last flag
     * @param iterVarName Name of the variable storing the loop iteration
     * @param columnPrecisions Name of the metatype tuple storing the precision of each column.
     * @param meta Metadata of table to write (e.g. comments).
     * @param ifExists Behavior if table exists (e.g. replace).
     * @param createTableType type of table to create (e.g. transient).
     * @return The generated code to produce the write-appending code
     */
    override fun generateStreamingWriteAppendCode(
        visitor: PandasCodeGenVisitor?,
        stateVarName: Variable?,
        tableVarName: Variable?,
        colNamesGlobal: Variable?,
        isLastVarName: Variable?,
        iterVarName: Variable?,
        columnPrecisions: Expr?,
        meta: SnowflakeCreateTableMetadata?,
        ifExists: BodoSQLCatalog.ifExistsBehavior?,
        createTableType: SqlCreateTable.CreateTableType?,
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
        tableName: ImmutableList<String>?,
        useStreaming: Boolean,
        streamingOptions: StreamingOptions?,
    ): Expr {
        TODO("Not yet implemented")
    }

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
     * Return the db location to which this Catalog refers.
     *
     * @return The source DB location.
     */
    override fun getDBType(): String {
        TODO("Not yet implemented")
    }

    /**
     * Returns if a schema with the given depth is allowed to contain tables.
     * A file system catalog has no rules for where a table can be located.
     *
     * @param depth The number of parent schemas that would need to be visited to reach the root.
     * @return True
     */
    override fun schemaDepthMayContainTables(depth: Int): Boolean {
        return true
    }

    /**
     * Returns if a schema with the given depth is allowed to contain subSchemas.
     * A file system catalog has no rules for where a subschema can be located.
     *
     * @param depth The number of parent schemas that would need to be visited to reach the root.
     * @return True.
     */
    override fun schemaDepthMayContainSubSchemas(depth: Int): Boolean {
        return true
    }

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
    override fun generatePythonConnStr(schemaPath: ImmutableList<String>): String {
        return fs.workingDirectory.toString()
    }

    /**
     *  Create a FileSystem object from the given connection string.
     *  @param connStr The connection string to the file system.
     *  @return The file system object.
     */
    private fun createFileSystem(connStr: String): FileSystem {
        val updatedConnStr = updateConnStr(connStr)
        val conf = createConf()

        val fs =
            Util.getFs(
                Path(LocationUtil.stripTrailingSlash(updatedConnStr)),
                conf,
            )
        val rootPath = Path(updatedConnStr)
        fs.workingDirectory = rootPath
        if (!fs.getFileStatus(rootPath).isDirectory) {
            throw RuntimeException("FileSystemCatalog Error: Root Path provided is not a valid directory.")
        }
        return fs
    }

    companion object {
        /**
         * Create a HadoopCatalog object from the given connection string.
         * @param connStr The connection string to the file system.
         * @return The HadoopCatalog object.
         */
        @JvmStatic
        private fun createHadoopCatalog(connStr: String): HadoopCatalog {
            val updatedConnStr = updateConnStr(connStr)
            val conf = createConf()
            return HadoopCatalog(conf, updatedConnStr)
        }

        /**
         * Update the connection string to be compatible with Hadoop.
         * @param connStr The connection string to the file system.
         * @return The updated connection string.
         */
        @JvmStatic
        private fun updateConnStr(connStr: String): String {
            if (connStr.isEmpty()) {
                throw RuntimeException("FileSystemCatalog Error: Connection string must be provided.")
            }
            // Hadoop wants s3a:// instead of s3://
            return if (connStr.startsWith("s3://")) connStr.replace("s3://", "s3a://") else connStr
        }

        /**
         * Create a Configuration object additional AWS authentication providers enabled.
         * @return The Configuration object.
         */
        @JvmStatic
        private fun createConf(): Configuration {
            val conf = Configuration()
            // This is to add the Profile credential provider which for some reason isn't used by default
            conf.set(
                "fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider," +
                    "com.amazonaws.auth.EnvironmentVariableCredentialsProvider," +
                    "com.amazonaws.auth.profile.ProfileCredentialsProvider," +
                    "com.amazonaws.auth.InstanceProfileCredentialsProvider," +
                    "org.apache.hadoop.fs.s3a.AnonymousAWSCredentialsProvider",
            )
            return conf
        }
    }
}
