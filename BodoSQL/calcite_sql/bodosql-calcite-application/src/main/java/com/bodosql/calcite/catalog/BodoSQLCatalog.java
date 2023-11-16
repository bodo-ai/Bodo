package com.bodosql.calcite.catalog;

import com.bodosql.calcite.adapter.pandas.StreamingOptions;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.List;
import java.util.Set;
import javax.annotation.Nullable;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.BodoTZInfo;

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
   * @param schemaName Name of the schema in the catalog.
   * @return Set of table names.
   */
  Set<String> getTableNames(String schemaName);

  /**
   * Returns a table with the given name and found in the given schema.
   *
   * @param schema BodoSQL schema containing the table.
   * @param tableName Name of the table.
   * @return The table object.
   */
  Table getTable(BodoSqlSchema schema, String tableName);

  /**
   * Get the top level schemas available for this catalog. Each individual catalog will decide what
   * the "top level" is.
   *
   * @return Set of available schema names.
   */
  Set<String> getSchemaNames();

  /**
   * Returns a top level schema with the given name in the catalog. Each individual catalog will
   * decide what the "top level" is.
   *
   * @param schemaName Name of the schema to fetch.
   * @return A schema object.
   */
  Schema getSchema(String schemaName);

  /**
   * Return the list of implicit/default schemas for the given catalog, in the order that they
   * should be prioritized during table resolution. We choose to implement this behavior at the
   * catalog level, as different catalogs may have different rules for how ambiguous table
   * identifiers are resolved.
   *
   * @return List of default Schema for this catalog.
   */
  List<String> getDefaultSchema();

  /**
   * Generates the code necessary to produce a write expression from the given catalog.
   *
   * @param varName Name of the variable to write.
   * @param schemaName Name of the schema to use when writing.
   * @param tableName Name of the table to use when writing.
   * @return The generated code to produce a write.
   */
  Expr generateWriteCode(
      Variable varName,
      String schemaName,
      String tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType);

  /**
   * Generates the code necessary to produce the streaming write initialization code from the given
   * catalog.
   *
   * @param operatorID ID of operation to use for retrieving memory budget.
   * @param schemaName Name of the schema to use when writing.
   * @param tableName Name of the table to use when writing.
   * @param ifExists Behavior to perform if the table already exists
   * @param createTableType Type of table to create if it doesn't exist
   * @return The generated code to produce the write-initialization code
   */
  Expr generateStreamingWriteInitCode(
      Expr.IntegerLiteral operatorID,
      String schemaName,
      String tableName,
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
   * @param schemaName Name of the schema to use when reading.
   * @param tableName Name of the table to use when reading.
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The options to use if streaming is enabled.
   * @return The generated code to produce a read.
   */
  Expr generateReadCode(
      String schemaName, String tableName, boolean useStreaming, StreamingOptions streamingOptions);

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
   * Return the top level name of the Catalog.
   *
   * @return The top level name of the Catalog.
   */
  String getCatalogName();
}
