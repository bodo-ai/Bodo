package com.bodosql.calcite.catalog.connection;

import com.bodosql.calcite.catalog.domain.*;
import com.bodosql.calcite.schema.BodoSqlTable;
import java.sql.*;
import java.util.*;
import org.apache.calcite.schema.Table;

public class SnowflakeCatalogImpl implements SnowflakeCatalog {
  private String connectionString;
  private String catalogName;
  // Account info contains the username and password information
  private Properties accountInfo;

  private CatalogDatabaseImpl db;

  public SnowflakeCatalogImpl(String accountName, String catalogName, Properties accountInfo) {
    this.connectionString =
        String.format("jdbc:snowflake://%s.snowflakecomputing.com/", accountName);
    this.catalogName = catalogName;
    this.accountInfo = accountInfo;
    // This db is used to conform with the API used for the other
    // Bodo information. It doesn't represent a real object and any
    // of the actual schema information is fetched in the other
    // methods. More testing is needed to determine if/how this is used
    // after the initial types are loaded.
    this.db = new CatalogDatabaseImpl("Snowflake");
  }

  /**
   * Get the CatalogDatabaseImpl for the SnowflakeCatalog
   *
   * @return CatalogDatabase
   */
  public CatalogDatabase getDb() {
    return this.db;
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

  @Override
  public Set<String> getTableNames(String schemaName) throws SQLException {
    Connection conn = getConnection();
    DatabaseMetaData metaData = getDataBaseMetaData(conn);
    // Passing null for tableNamePattern should match all table names. Although
    // this is not in the public documentation. TABLE refers to the JDBC table
    // type.
    ResultSet tableInfo = metaData.getTables(catalogName, schemaName, null, new String[] {"TABLE"});
    HashSet<String> tableNames = new HashSet<>();
    while (tableInfo.next()) {
      // Table name is stored in column 3
      // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getTables
      tableNames.add(tableInfo.getString(3));
    }
    // TODO: Cache the same connection if possible
    conn.close();
    return tableNames;
  }

  @Override
  public Table getTable(String schemaName, String tableName) throws SQLException {
    Connection conn = getConnection();
    DatabaseMetaData metaData = getDataBaseMetaData(conn);
    // Passing null for columnNamePattern should match all columns. Although
    // this is not in the public documentation.
    ResultSet tableInfo = metaData.getColumns(catalogName, schemaName, tableName, null);
    List<CatalogColumnImpl> columns = new ArrayList<>();
    int columnNum = 0;
    while (tableInfo.next()) {
      // Column name is stored in column 4
      // Data type is stored in column 5
      // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getColumns
      String columnName = tableInfo.getString(4);
      CatalogColumnDataType type =
          CatalogColumnDataType.fromJavaSqlType(JDBCType.valueOf(tableInfo.getInt(5)));
      columns.add(new CatalogColumnImpl(columnName, type, columnNum));
      columnNum += 1;
    }
    // TODO: Cache the same connection if possible
    conn.close();
    return new BodoSqlTable(new CatalogTableImpl(tableName, this.db, columns));
  }
}
