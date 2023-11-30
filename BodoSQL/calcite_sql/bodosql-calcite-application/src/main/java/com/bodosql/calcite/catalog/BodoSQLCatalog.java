package com.bodosql.calcite.catalog;

import com.bodosql.calcite.adapter.pandas.StreamingOptions;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.CatalogSchema;
import com.bodosql.calcite.table.CatalogTable;
import com.google.common.collect.ImmutableList;
import java.util.List;
import java.util.Set;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.checkerframework.checker.nullness.qual.Nullable;

public interface BodoSQLCatalog {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Catalog
   */

  /** Enum describing the write behavior when the table already exists. */
  enum ifExistsBehavior {
    REPLACE,
    APPEND,
    FAIL;

    public String asToSqlKwArgument() {
      switch (this) {
        case REPLACE:
          return "replace";
        case FAIL:
          return "fail";
        case APPEND:
          return "append";
      }
      throw new RuntimeException("Reached Unreachable code in toToSqlKwArgument");
    }
  }

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
   * Generates the code necessary to produce an append write expression from the given catalog.
   *
   * @param varName Name of the variable to write.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @return The generated code to produce the append write.
   */
  Expr generateAppendWriteCode(Variable varName, List<String> tableName);

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
      Variable varName,
      List<String> tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType);

  /**
   * Generates the code necessary to produce the streaming write initialization code from the given
   * catalog for an append operation.
   *
   * @param operatorID ID of operation to use for retrieving memory budget.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @return The generated code to produce the write-initialization code
   */
  Expr generateStreamingAppendWriteInitCode(Expr.IntegerLiteral operatorID, List<String> tableName);

  /**
   * Generates the code necessary to produce the streaming write initialization code from the given
   * catalog.
   *
   * @param operatorID ID of operation to use for retrieving memory budget.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @param ifExists Behavior to perform if the table already exists
   * @param createTableType Type of table to create if it doesn't exist
   * @return The generated code to produce the write-initialization code
   */
  Expr generateStreamingWriteInitCode(
      Expr.IntegerLiteral operatorID,
      List<String> tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType);

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
   * @return The generated code to produce the write-appending code
   */
  Expr generateStreamingWriteAppendCode(
      Variable stateVarName,
      Variable tableVarName,
      Variable colNamesGlobal,
      Variable isLastVarName,
      Variable iterVarName,
      Expr columnPrecisions);

  /**
   * Generates the code necessary to produce a read expression from the given catalog.
   *
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table.
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The options to use if streaming is enabled.
   * @return The generated code to produce a read.
   */
  Expr generateReadCode(
      List<String> tableName, boolean useStreaming, StreamingOptions streamingOptions);

  /**
   * Close any connections to the remote DataBase. If there are no connections this should be a
   * no-op.
   */
  void closeConnections();

  /**
   * Generates the code necessary to submit the remote query to the catalog DB.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  Expr generateRemoteQuery(String query);

  /**
   * Return the db location to which this Catalog refers.
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
  BodoTZInfo getDefaultTimezone();

  /**
   * Fetch the WEEK_START session parameter for this catalog. If not specified it should return 0 as
   * default.
   *
   * @return Integer for the WEEK_START session parameter.
   */
  @Nullable
  Integer getWeekStart();

  /**
   * Fetch the WEEK_OF_YEAR_POLICY session parameter for this catalog. If not specified it should
   * return 0 as default.
   *
   * @return Integer for the WEEK_OF_YEAR_POLICY session parameter.
   */
  @Nullable
  Integer getWeekOfYearPolicy();

  /**
   * Returns if a schema with the given depth is allowed to contain tables.
   *
   * @param depth The number of parent schemas that would need to be visited to reach the root.
   * @return Can a schema at that depth contain tables.
   */
  boolean schemaDepthMayContainsTables(int depth);

  /**
   * Returns if a schema with the given depth is allowed to contain subSchemas.
   *
   * @param depth The number of parent schemas that would need to be visited to reach the root.
   * @return Can a schema at that depth contain subSchemas.
   */
  boolean schemaDepthMayContainSubSchemas(int depth);
}
