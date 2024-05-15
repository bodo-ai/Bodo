package com.bodosql.calcite.catalog;

import static com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior;

import com.bodosql.calcite.adapter.bodo.StreamingOptions;
import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.CatalogSchema;
import com.bodosql.calcite.schema.InlineViewMetadata;
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata;
import com.bodosql.calcite.table.CatalogTable;
import com.google.common.collect.ImmutableList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import org.apache.calcite.schema.Function;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * See the design described on Confluence:
 * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Catalog
 */
public interface BodoSQLCatalog {

  /**
   * Returns a set of all table names with the given schema name.
   *
   * @param schemaPath The list of schemas to traverse before finding the table.
   * @return Set of table names.
   */
  Set<String> getTableNames(ImmutableList<String> schemaPath);

  /**
   * Returns a table with the given name and found in the given schema.
   *
   * @param schemaPath The list of schemas to traverse before finding the table.
   * @param tableName Name of the table.
   * @return The table object.
   */
  CatalogTable getTable(ImmutableList<String> schemaPath, String tableName);

  /**
   * Get the available subSchema names for the given path.
   *
   * @param schemaPath The parent schema path to check.
   * @return Set of available schema names.
   */
  Set<String> getSchemaNames(ImmutableList<String> schemaPath);

  /**
   * Returns a schema found within the given parent path.
   *
   * @param schemaPath The parent schema path to check.
   * @param schemaName Name of the schema to fetch.
   * @return A schema object.
   */
  CatalogSchema getSchema(ImmutableList<String> schemaPath, String schemaName);

  /**
   * Return the list of implicit/default schemas for the given catalog, in the order that they
   * should be prioritized during table resolution. The provided depth gives the "level" at which to
   * provide the default.
   *
   * @param depth The depth at which to find the default.
   * @return List of default Schema for this catalog.
   */
  List<String> getDefaultSchema(int depth);

  /**
   * Return the number of levels at which a default schema may be found.
   *
   * @return The number of levels a default schema can be found.
   */
  int numDefaultSchemaLevels();

  /**
   * Returns a set of all function names with the given schema name.
   *
   * @param schemaPath The list of schemas to traverse before finding the function.
   * @return Set of function names.
   */
  default Set<String> getFunctionNames(ImmutableList<String> schemaPath) {
    return Set.of();
  }

  /**
   * Returns all functions defined in this catalog with a given name and schema path.
   *
   * @param schemaPath The list of schemas to traverse before finding the function.
   * @param funcName Name of functions with a given name.
   * @return Collection of all functions with that name.
   */
  default Collection<Function> getFunctions(ImmutableList<String> schemaPath, String funcName) {
    return List.of();
  }

  /**
   * Generates the code necessary to produce an append write expression from the given catalog.
   *
   * @param varName Name of the variable to write.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @return The generated code to produce the append write.
   */
  Expr generateAppendWriteCode(
      BodoCodeGenVisitor visitor, Variable varName, ImmutableList<String> tableName);

  /**
   * Generates the code necessary to produce a write expression from the given catalog.
   *
   * @param varName Name of the variable to write.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @param ifExists Behavior to perform if the table already exists
   * @param createTableType Type of table to create if it doesn't exist
   * @return The generated code to produce a write.
   */
  Expr generateWriteCode(
      BodoCodeGenVisitor visitor,
      Variable varName,
      ImmutableList<String> tableName,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType,
      SnowflakeCreateTableMetadata meta);

  /**
   * Generates the code necessary to produce a read expression from the given catalog.
   *
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The options to use if streaming is enabled.
   * @return The generated code to produce a read.
   */
  Expr generateReadCode(
      ImmutableList<String> tableName, boolean useStreaming, StreamingOptions streamingOptions);

  /**
   * Close any connections to the remote DataBase. If there are no connections this should be a
   * no-op.
   */
  default void closeConnections() {}

  /**
   * TODO: REMOVE
   *
   * <p>Generates the code necessary to submit the remote query to the catalog DB.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  Expr generateRemoteQuery(String query);

  /**
   * TODO: REMOVE
   *
   * <p>Return the db location to which this Catalog refers.
   *
   * @return The source DB location.
   */
  String getDBType();

  /**
   * Fetch the default timezone for this catalog. If the catalog doesn't influence the default
   * timezone it should return UTC.
   *
   * @return BodoTZInfo for the default timezone.
   */
  default BodoTZInfo getDefaultTimezone() {
    return BodoTZInfo.UTC;
  }

  /**
   * Fetch the WEEK_START session parameter for this catalog. If not specified it should return 0 as
   * default.
   *
   * @return Integer for the WEEK_START session parameter.
   */
  default int getWeekStart() {
    return 0;
  }

  /**
   * Fetch the WEEK_OF_YEAR_POLICY session parameter for this catalog. If not specified it should
   * return 0 as default.
   *
   * @return Integer for the WEEK_OF_YEAR_POLICY session parameter.
   */
  default int getWeekOfYearPolicy() {
    return 0;
  }

  /**
   * Returns if a schema with the given depth is allowed to contain tables.
   *
   * @param depth The number of parent schemas that would need to be visited to reach the root.
   * @return Can a schema at that depth contain tables.
   */
  boolean schemaDepthMayContainTables(int depth);

  /**
   * Returns if a schema with the given depth is allowed to contain subSchemas.
   *
   * @param depth The number of parent schemas that would need to be visited to reach the root.
   * @return Can a schema at that depth contain subSchemas.
   */
  boolean schemaDepthMayContainSubSchemas(int depth);

  /**
   * Returns if a schema with the given depth is allowed to contain functions.
   *
   * @param depth The number of parent schemas that would need to be visited to reach the root.
   * @return Can a schema at that depth contain functions.
   */
  default boolean schemaDepthMayContainFunctions(int depth) {
    return false;
  }

  /**
   * Generate a Python connection string used to read from or write to a Catalog in Bodo's SQL
   * Python code.
   *
   * <p>TODO(jsternberg): This method is needed for the XXXToBodoConverter nodes, but exposing this
   * is a bad idea and this class likely needs to be refactored in a way that the connection
   * information can be passed around more easily.
   *
   * @param schemaPath The schema component to define the connection not including the table name.
   * @return The connection string
   */
  String generatePythonConnStr(ImmutableList<String> schemaPath);

  /**
   * Return the desired WriteTarget for a create table operation. We provide both creation details
   * and the desired target as these may influence the WriteTarget or modify the selected
   * WriteTarget. For example, a catalog may have feature gaps for its preferred WriteTable which
   * may prompt selecting a different WriteTable.
   *
   * @param schema The schemaPath to the table.
   * @param tableName The name of the type that will be created.
   * @param createTableType The createTable type.
   * @param ifExistsBehavior The createTable behavior for if there is already a table defined.
   * @param columnNamesGlobal Global Variable holding the output column names.
   * @return The selected WriteTarget.
   */
  WriteTarget getCreateTableWriteTarget(
      ImmutableList<String> schema,
      String tableName,
      SqlCreateTable.CreateTableType createTableType,
      IfExistsBehavior ifExistsBehavior,
      Variable columnNamesGlobal);

  String getAccountName();

  /**
   * Load the view metadata information from the catalog. If the table is not a view or no
   * information can be found this should return NULL.
   *
   * @param names A list of two names starting with SCHEMA_NAME and ending with TABLE_NAME.
   * @return The InlineViewMetadata loaded from the catalog or null if no information is available.
   */
  @Nullable
  InlineViewMetadata tryGetViewMetadata(List<String> names);
}
