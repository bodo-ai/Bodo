package com.bodosql.calcite.catalog;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.BodoSqlSchema;
import java.util.List;
import java.util.Set;
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
  public enum ifExistsBehavior {
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
   * Generates the code necessary to produce a read expression from the given catalog.
   *
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param schemaName Name of the schema to use when reading.
   * @param tableName Name of the table to use when reading.
   * @return The generated code to produce a read.
   */
  Expr generateReadCode(String schemaName, String tableName, boolean useStreaming);

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
  public String getDBType();

  /**
   * Fetch the default timezone for this catalog. If the catalog doesn't influence the default
   * timezone it should return UTC.
   *
   * @return BodoTZInfo for the default timezone.
   */
  BodoTZInfo getDefaultTimezone();

  /**
   * Return the top level name of the Catalog.
   *
   * @return The top level name of the Catalog.
   */
  public String getCatalogName();
}
