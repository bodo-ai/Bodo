package com.bodosql.calcite.catalog;

import com.bodosql.calcite.schema.BodoSqlSchema;
import java.sql.*;
import java.util.*;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;

public interface BodoSQLCatalog {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Catalog
   */

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
   * Return the list of default schema for a user. In the future we may opt to include a default
   * schema at each level, so we return a list of schema.
   *
   * @return List of any default Schema that exist.
   */
  List<BodoSqlSchema> getDefaultSchema();

  /**
   * Generates the code necessary to produce a write expression from the given catalog.
   *
   * @param varName Name of the variable to write.
   * @param schemaName Name of the schema to use when writing.
   * @param tableName Name of the table to use when writing.
   * @return The generated code to produce a write.
   */
  String generateWriteCode(String varName, String schemaName, String tableName);

  /**
   * Generates the code necessary to produce a read expression from the given catalog.
   *
   * @param schemaName Name of the schema to use when reading.
   * @param tableName Name of the table to use when reading.
   * @return The generated code to produce a read.
   */
  String generateReadCode(String schemaName, String tableName);

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
  String generateRemoteQuery(String query);
}
