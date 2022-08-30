package com.bodosql.calcite.catalog;

import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.CatalogTableImpl;
import java.sql.*;
import java.util.*;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;

public class SnowflakeCatalogImpl implements BodoSQLCatalog {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Catalog
   */
  private String connectionString;

  private String catalogName;
  // Account info contains the username and password information
  private Properties accountInfo;

  /**
   * Create the catalog and store the relevant account information.
   *
   * @param accountName User's snowflake account name.
   * @param catalogName Name of the catalog (or database in Snowflake terminology). In the future
   *     this may be removed/modified
   * @param accountInfo Properties object that contains the username and password information
   */
  public SnowflakeCatalogImpl(String accountName, String catalogName, Properties accountInfo) {
    this.connectionString =
        String.format("jdbc:snowflake://%s.snowflakecomputing.com/", accountName);
    this.catalogName = catalogName;
    this.accountInfo = accountInfo;
  }

  /**
   * Get a connection to snowflake via jdbc
   *
   * @return Connection to Snowflake
   * @throws SQLException
   */
  private Connection getConnection() throws SQLException {
    // TODO: Cache the same connection if possible
    return DriverManager.getConnection(connectionString, accountInfo);
  }

  /**
   * Get the DataBase metadata for a Snowflake connection
   *
   * @param conn Connection to Snowflake
   * @return DatabaseMetaData for Snowflake
   * @throws SQLException
   */
  private DatabaseMetaData getDataBaseMetaData(Connection conn) throws SQLException {
    // TODO: Cache the same metadata if possible
    return conn.getMetaData();
  }

  /**
   * Returns a set of all table names with the given schema name. This code connects to Snowflake
   * via JDBC and loads the list of table names using standard JDBC APIs.
   *
   * @param schemaName Name of the schema in the catalog.
   * @return Set of table names.
   */
  @Override
  public Set<String> getTableNames(String schemaName) {
    try {
      Connection conn = getConnection();
      DatabaseMetaData metaData = getDataBaseMetaData(conn);
      // Passing null for tableNamePattern should match all table names. Although
      // this is not in the public documentation. TABLE refers to the JDBC table
      // type.
      ResultSet tableInfo =
          metaData.getTables(catalogName, schemaName, null, new String[] {"TABLE"});
      HashSet<String> tableNames = new HashSet<>();
      while (tableInfo.next()) {
        // Table name is stored in column 3
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getTables
        tableNames.add(tableInfo.getString(3));
      }
      // TODO: Cache the same connection if possible
      conn.close();
      return tableNames;
    } catch (SQLException e) {
      throw new RuntimeException(
          String.format(
              "Unable to get table names for Schema '%s' from Snowflake account. Error message: %s",
              schemaName, e));
    }
  }

  /**
   * Returns a table with the given name if found in the given schema from Snowflake. This code
   * connects to Snowflake via JDBC and loads the table metadata using standard JDBC APIs.
   *
   * @param schema BodoSQL schema containing the table.
   * @param tableName Name of the table.
   * @return The table object.
   */
  @Override
  public Table getTable(BodoSqlSchema schema, String tableName) {
    try {
      Connection conn = getConnection();
      DatabaseMetaData metaData = getDataBaseMetaData(conn);
      // Passing null for columnNamePattern should match all columns. Although
      // this is not in the public documentation.
      ResultSet tableInfo = metaData.getColumns(catalogName, schema.getName(), tableName, null);
      List<BodoSQLColumn> columns = new ArrayList<>();
      while (tableInfo.next()) {
        // Column name is stored in column 4
        // Data type is stored in column 5
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getColumns
        String columnName = tableInfo.getString(4);
        int dataType = tableInfo.getInt(5);
        BodoSQLColumnDataType type =
            BodoSQLColumnDataType.fromJavaSqlType(JDBCType.valueOf(dataType));
        columns.add(new BodoSQLColumnImpl(columnName, type));
      }
      // TODO: Cache the same connection if possible
      conn.close();
      return new CatalogTableImpl(tableName, schema, columns);
    } catch (SQLException e) {
      throw new RuntimeException(
          String.format(
              "Unable to get table '%s' for Schema '%s' from Snowflake account. Error message: %s",
              tableName, schema.getName(), e));
    }
  }

  /**
   * Get the Snowflake "databases" (top level directories) available for this catalog. Currently we
   * require a single Database, so these are treated as the top level.
   *
   * @return Set of available schema names.
   */
  @Override
  public Set<String> getSchemaNames() {
    HashSet<String> schemaNames = new HashSet<>();
    try {
      Connection conn = getConnection();
      DatabaseMetaData metaData = getDataBaseMetaData(conn);
      ResultSet schemaInfo = metaData.getSchemas(catalogName, null);
      while (schemaInfo.next()) {
        // Schema name is stored in column 1
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getSchemas
        schemaNames.add(schemaInfo.getString(1));
      }
      // TODO: Cache the same connection if possible
      conn.close();
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
    return schemaNames;
  }

  /**
   * Returns a Snowflake "database" (top level directory) with the given name in the catalog. This
   * is unimplemented as we only support a single database at this time.
   *
   * @param schemaName Name of the schema to fetch.
   * @return A schema object.
   */
  @Override
  public Schema getSchema(String schemaName) {
    // TODO: Implement when we support having multiple Snowflake Databases at once.
    return null;
  }

  /**
   * Generates the code necessary to produce a write expression from Snowflake.
   *
   * @param varName Name of the variable to write.
   * @param schemaName Name of the schema to use when writing.
   * @param tableName Name of the table to use when writing.
   * @return The generated code to produce a write.
   */
  @Override
  public String generateWriteCode(String varName, String schemaName, String tableName) {
    // TODO: Implement in a followup PR
    return "";
  }

  /**
   * Generates the code necessary to produce a read expression from Snowflake.
   *
   * @param schemaName Name of the schema to use when reading.
   * @param tableName Name of the table to use when reading.
   * @return The generated code to produce a read.
   */
  @Override
  public String generateReadCode(String schemaName, String tableName) {
    // TODO: Implement in a followup PR
    return "";
  }
}
