package com.bodosql.calcite.catalog;

import static java.lang.Math.min;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.CatalogTableImpl;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import kotlin.*;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.json.simple.JSONObject;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SnowflakeCatalogImpl implements BodoSQLCatalog {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Catalog
   */
  private String connectionString;

  private final String username;

  private final String password;

  private final String accountName;

  private final String catalogName;

  private final String warehouseName;

  // Account info contains the username and password information
  private final Properties accountInfo;

  // Combination of accountInfo and username + password.
  // These are separate because the Java and Python connection
  // strings pass different properties.
  private final Properties totalProperties;

  // Cached valued for the connection
  private Connection conn;

  // Cached value for the database metadata.
  private DatabaseMetaData dbMeta;

  // Logger for logging warnings.
  private static final Logger LOGGER = LoggerFactory.getLogger(SnowflakeCatalogImpl.class);

  // The maximum number of retries for connecting to snowflake before succeeding.
  private static final int maxRetries = 5;

  // The initial time to wait when retrying connections. This will be done with
  // an exponential backoff so this is just the base wait time.
  private static final int backoffMilliseconds = 100;

  // Maximum amount of time we are going to wait between retries
  private static final int maxBackoffMilliseconds = 2000;

  private @Nullable static BodoTZInfo sfTZInfo;

  /**
   * Create the catalog and store the relevant account information.
   *
   * @param username Snowflake username
   * @param password Snowflake password
   * @param accountName User's snowflake account name.
   * @param catalogName Name of the catalog (or database in Snowflake terminology). In the future
   *     this may be removed/modified
   * @param accountInfo Any extra properties to pass to Snowflake.
   */
  public SnowflakeCatalogImpl(
      String username,
      String password,
      String accountName,
      String catalogName,
      String warehouseName,
      Properties accountInfo) {
    this.username = username;
    this.password = password;
    this.accountName = accountName;
    this.warehouseName = warehouseName;
    this.connectionString =
        String.format("jdbc:snowflake://%s.snowflakecomputing.com/", accountName);
    this.catalogName = catalogName;
    this.accountInfo = accountInfo;
    this.totalProperties = new Properties();
    // Add the user and password to the properties for JDBC
    this.totalProperties.put("user", username);
    this.totalProperties.put("password", password);
    // Add the catalog name as the default database.
    this.totalProperties.put("db", catalogName);
    this.totalProperties.putAll(this.accountInfo);
    this.conn = null;
    this.dbMeta = null;
    // TODO(Nick): Ensure keys are in all caps at the Python level
    if (accountInfo.contains("TIMEZONE")) {
      // This is an optimization to avoid a trip to Snowflake.
      // Note: We assume the Timestamp information can never be loaded as an int.
      this.sfTZInfo = new BodoTZInfo(accountInfo.getProperty("TIMEZONE"), "str");
    }
  }

  /**
   * Get a connection to snowflake via jdbc
   *
   * @return Connection to Snowflake
   * @throws SQLException
   */
  private Connection getConnection() throws SQLException {
    if (conn == null) {
      // DriverManager need manual retries
      // https://stackoverflow.com/questions/6110975/connection-retry-using-jdbc
      int numRetries = 0;
      do {
        conn = DriverManager.getConnection(connectionString, totalProperties);
        if (conn == null) {
          int sleepMultiplier = 2 << numRetries;
          int sleepTime = min(sleepMultiplier * backoffMilliseconds, maxBackoffMilliseconds);
          LOGGER.warn(
              String.format(
                  "Failed to connect to Snowflake, retrying after %d milleseconds...", sleepTime));
          try {
            Thread.sleep(sleepTime);
          } catch (InterruptedException e) {
            throw new RuntimeException(
                "Backoff between Snowflake connection attempt interupted...", e);
          }
        }
        numRetries += 1;
      } while (conn == null && numRetries < maxRetries);
    }
    return conn;
  }

  /**
   * Fetch a Snowflake Session parameter from a user's Snowflake account. These may not have been
   * set by us and instead may be session defaults.
   *
   * @param param The name of the parameter to fetch.
   * @param shouldRetry If failing to load the Metadata should we retry with a fresh connection?
   * @return The value of the parameter as a String
   * @throws SQLException An exception occurs contacting snowflake.
   */
  private String getSnowflakeParameter(String param, boolean shouldRetry) throws SQLException {
    try {
      ResultSet paramInfo =
          executeSnowflakeQuery(String.format("Show parameters like '%s'", param));
      if (paramInfo.next()) {
        return paramInfo.getString(2);
      } else {
        throw new SQLException("Snowflake returned a empty table of Session parameters");
      }
    } catch (SQLException e) {
      if (shouldRetry) {
        closeConnections();
        return getSnowflakeParameter(param, false);
      } else {
        throw e;
      }
    }
  }

  /**
   * Get the Snowflake timezone session parameter and update the cached value.
   *
   * @param shouldRetry If failing to load the Metadata should we retry with a fresh connection?
   * @return THe value of the BodoTZInfo with the TIMEZONE from Snowflake.
   * @throws SQLException An exception occurs contacting snowflake.
   */
  private BodoTZInfo getSnowflakeTimezone(boolean shouldRetry) throws SQLException {
    if (sfTZInfo == null) {
      // Note: We assume the Timestamp information can never be loaded as an int.
      sfTZInfo = new BodoTZInfo(getSnowflakeParameter("TIMEZONE", shouldRetry), "str");
    }
    return sfTZInfo;
  }

  /**
   * Is the current connection going to be loaded from cache. This determines if we need to try any
   * operation that uses the connection.
   *
   * @return If the connection is not null.
   */
  private boolean isConnectionCached() {
    return conn != null;
  }

  /**
   * Get the DataBase metadata for the Snowflake connection
   *
   * @param shouldRetry If failing to load the Metadata should we retry with a fresh connection?
   * @return DatabaseMetaData for Snowflake
   * @throws SQLException
   */
  private DatabaseMetaData getDataBaseMetaData(boolean shouldRetry) throws SQLException {
    try {
      if (dbMeta == null) {
        dbMeta = getConnection().getMetaData();
      }
    } catch (SQLException e) {
      if (shouldRetry) {
        closeConnections();
        return getDataBaseMetaData(false);
      } else {
        throw e;
      }
    }
    return dbMeta;
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
    return getTableNamesImpl(schemaName, isConnectionCached());
  }

  /**
   * Implementation of getTableNames that enables retrying if a cached connection fails
   *
   * @param schemaName Name of the schema in the catalog.
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return
   */
  private Set<String> getTableNamesImpl(String schemaName, boolean shouldRetry) {
    try {
      DatabaseMetaData metaData = getDataBaseMetaData(shouldRetry);
      // Passing null for tableNamePattern should match all table names. Although
      // this is not in the public documentation. Passing null for types ensures we
      // allow all possible table types, including regular tables and views.
      ResultSet tableInfo = metaData.getTables(catalogName, schemaName, null, null);
      HashSet<String> tableNames = new HashSet<>();
      while (tableInfo.next()) {
        // Table name is stored in column 3
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getTables
        tableNames.add(tableInfo.getString(3));
      }
      return tableNames;
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get table names for Schema '%s' from Snowflake account. Error message: %s",
              schemaName, e);
      if (shouldRetry) {
        LOGGER.warn(errorMsg);
        closeConnections();
        return getTableNamesImpl(schemaName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
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
    return getTableImpl(schema, tableName, isConnectionCached());
  }

  /**
   * Implementation of getTable that enables retrying if a cached connection fails
   *
   * @param schema BodoSQL schema containing the table.
   * @param tableName Name of the table.
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return The table object.
   */
  private Table getTableImpl(BodoSqlSchema schema, String tableName, boolean shouldRetry) {
    try {
      // Fetch the timezone info.
      BodoTZInfo tzInfo = getSnowflakeTimezone(shouldRetry);
      // Table metadata needs to be derived from describe table because some types
      // aren't available via the JDBC connector. In particular Snowflake doesn't communicate
      // information about Variant, Array, or Object.
      ResultSet tableInfo =
          executeSnowflakeQuery(
              String.format("Describe table %s.%s.%s", catalogName, schema.getName(), tableName));
      List<BodoSQLColumn> columns = new ArrayList<>();
      while (tableInfo.next()) {
        // Column name is stored in column 1
        // Data type is stored in column 2
        // NULLABLE is stored in column 4.
        // https://docs.snowflake.com/en/sql-reference/sql/desc-table#examples
        // TODO: Can we leverage primary key and unique key? Snowflake
        // states they don't enforce them.
        // TODO: Can we leverage additional type information (e.g. max string size).
        String writeName = tableInfo.getString(1);
        String readName = writeName;
        if (readName.equals(readName.toUpperCase())) {
          readName = readName.toLowerCase();
        }
        String dataType = tableInfo.getString(2);
        // Parse the given type for the column type and precision information.
        Pair<BodoSQLColumnDataType, Integer> snowflakeTypeInfo =
            BodoSQLColumnDataType.fromSnowflakeTypeName(dataType);
        BodoSQLColumnDataType type = snowflakeTypeInfo.getFirst();
        int precision = snowflakeTypeInfo.getSecond();
        // The column is nullable unless we are certain it has no nulls.
        boolean nullable = tableInfo.getString(4).toUpperCase(Locale.ROOT).equals("Y");
        columns.add(new BodoSQLColumnImpl(readName, writeName, type, nullable, tzInfo, precision));
      }
      return new CatalogTableImpl(tableName, schema, columns);
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get table '%s' for Schema '%s' from Snowflake account. Error message: %s",
              tableName, schema.getName(), e);
      if (shouldRetry) {
        LOGGER.warn(errorMsg);
        closeConnections();
        return getTableImpl(schema, tableName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
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
    return getSchemaNamesImpl(isConnectionCached());
  }

  /**
   * Implementation of getSchemaNames that enables retrying if a cached connection fails
   *
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return Set of available schema names.
   */
  private Set<String> getSchemaNamesImpl(boolean shouldRetry) {
    HashSet<String> schemaNames = new HashSet<>();
    try {
      DatabaseMetaData metaData = getDataBaseMetaData(shouldRetry);
      ResultSet schemaInfo = metaData.getSchemas(catalogName, null);
      while (schemaInfo.next()) {
        // Schema name is stored in column 1
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getSchemas
        schemaNames.add(schemaInfo.getString(1));
      }
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get a list of schema names from the Snowflake account. Error message: %s",
              e);
      if (shouldRetry) {
        LOGGER.warn(errorMsg);
        closeConnections();
        return getSchemaNamesImpl(false);
      } else {
        throw new RuntimeException(errorMsg);
      }
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
   * Return the list of implicit/default schemas for the given catalog, in the order that they
   * should be prioritized during table resolution.
   *
   * <p>For a snowflake catalog, the order should be based on the list of schemas returned by
   * CURRENT_SCHEMAS(). IE, if CURRENT_SCHEMAS() = ["MYTESTDB.schema1", "MYTESTDB.schema2"...
   * "MYTESTDB.schemaN"]
   *
   * <p>We would attempt to resolve any non-qualified tables as:
   *
   * <p>MYTESTDB.schema1.(table_identifier) MYTESTDB.schema2.(table_identifier) ...
   * MYTESTDB.non_default_schema_n.(table_identifier)
   *
   * <p>(see https://docs.snowflake.com/en/sql-reference/name-resolution)
   *
   * @return List of default Schema to check when attempting to resolve a table in this catalog.
   */
  public List<String> getDefaultSchema() {
    return getDefaultSchemaImpl(isConnectionCached());
  }

  /**
   * Implementation of getDefaultSchema that enables retrying if a cached connection fails
   *
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return List of default Schema if they exist.
   */
  private List<String> getDefaultSchemaImpl(boolean shouldRetry) {
    List<String> defaultSchema = new ArrayList<>();
    try {
      Connection conn = getConnection();
      Statement stmt = conn.createStatement();

      // For Snowflake, the attempted order of resolution should be the
      // list of schemas provided by CURRENT_SCHEMAS()
      ResultSet schemaInfo = stmt.executeQuery("select current_schemas()");
      while (schemaInfo.next()) {
        // Output in column 1
        String schemaNames = schemaInfo.getString(1);
        if (schemaNames != null) {
          // schemaNames() returns a string in the form of '["BD.Scheam1", "DB.Schema2"...]'
          // Do some simple regex to extract the schema names.
          Pattern pattern = Pattern.compile("\"" + catalogName + ".([^\"]*)\"");
          Matcher matcher = pattern.matcher(schemaNames);

          while (matcher.find()) {
            defaultSchema.add(matcher.group(1));
          }
        }
      }
      // TODO: Cache the same connection if possible
      conn.close();
    } catch (SQLException e) {

      String errorMsg =
          String.format("Unable to load default schema from snowflake. Error message: %s", e);
      if (shouldRetry) {
        LOGGER.warn(errorMsg);
        closeConnections();
        return getDefaultSchemaImpl(false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
    return defaultSchema;
  }

  /**
   * Generate a Python connection string used to read from or write to Snowflake in Bodo's SQL
   * Python code.
   *
   * <p>TODO(jsternberg): This method is needed for the SnowflakeAggregateTableScan, but exposing
   * this is a bad idea and this class likely needs to be refactored in a way that the connection
   * information can be passed around more easily.
   *
   * @param schemaName The of the schema which must be inserted into the connection string.
   * @return The connection string
   */
  public String generatePythonConnStr(String schemaName) {
    // First create the basic connection string that must
    // always be included.
    StringBuilder connString = new StringBuilder();
    // Append the base url

    try {
      connString.append(
          String.format(
              "snowflake://%s:%s@%s/%s",
              URLEncoder.encode(this.username, "UTF-8"),
              URLEncoder.encode(this.password, "UTF-8"),
              URLEncoder.encode(this.accountName, "UTF-8"),
              URLEncoder.encode(this.catalogName, "UTF-8")));
      if (!schemaName.isEmpty()) {
        // Append a schema if it exists
        connString.append(String.format("/%s", URLEncoder.encode(schemaName, "UTF-8")));
      }
      connString.append(
          String.format("?warehouse=%s", URLEncoder.encode(this.warehouseName, "UTF-8")));
      // Add support for any additional optional properties
      if (!this.accountInfo.isEmpty()) {
        connString.append("&");
        JSONObject o1 = new JSONObject();
        for (Map.Entry<Object, Object> entry : this.accountInfo.entrySet()) {
          Object key = entry.getKey();
          Object value = entry.getValue();
          o1.put(key, value);
        }
        String encodedJSONString = URLEncoder.encode(o1.toJSONString(), "UTF-8");
        connString.append("session_parameters").append("=").append(encodedJSONString);
      }
    } catch (UnsupportedEncodingException e) {
      throw new BodoSQLCodegenException(
          "Internal Error: Unable to encode Python connection string. Error message: " + e);
    }

    return connString.toString();
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
  public String generateWriteCode(
      String varName,
      String schemaName,
      String tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType tableType) {
    return String.format(
        "%s.to_sql('%s', '%s', schema='%s', if_exists='%s', _bodo_create_table_type='%s',"
            + " index=False)",
        varName,
        tableName,
        generatePythonConnStr(schemaName),
        schemaName,
        ifExists.asToSqlKwArgument(),
        tableType.asStringKeyword());
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
    return String.format(
        "pd.read_sql('%s', '%s', _bodo_is_table_input=True)",
        tableName, generatePythonConnStr(schemaName));
  }

  /**
   * Close the connection to Snowflake and clear any internal variables. If there is no active
   * connection this is a no-op.
   */
  public void closeConnections() {
    if (conn != null) {
      try {
        conn.close();
      } catch (SQLException e) {
        // We ignore any exception from closing the connection string as
        // we should no longer need to connect to Snowflake. This could happen
        // for example if the connection already timed out.
        LOGGER.warn(
            String.format(
                "Exception encountered when trying to close the Snowflake connection: %s", e));
      }
    }
    dbMeta = null;
    conn = null;
  }

  /**
   * Generates the code necessary to submit the remote query to the catalog DB.
   *
   * @param query Query to submit.
   * @return The generated code.
   */
  @Override
  public String generateRemoteQuery(String query) {
    // For correctness we need to verify that Snowflake can support this
    // query in its entirely (no BodoSQL specific features). To do this
    // we run an explain, which won't execute the query.
    executeExplainQuery(query);
    // We need to include the default schema to ensure the query works as intended
    List<String> schemaList = getDefaultSchema();
    String schemaName = "";
    if (schemaList.size() > 0) {
      schemaName = schemaList.get(0);
    }
    return String.format("pd.read_sql('%s', '%s')", query, generatePythonConnStr(schemaName));
  }

  /**
   * Return the db location to which this Catalog refers.
   *
   * @return The source DB location.
   */
  @Override
  public String getDBType() {
    return "SNOWFLAKE";
  }

  /**
   * Fetch the default timezone for this catalog. If the catalog doesn't influence the default
   * timezone it should return UTC.
   *
   * @return BodoTZInfo for the default timezone.
   */
  @Override
  public BodoTZInfo getDefaultTimezone() {
    try {
      return getSnowflakeTimezone(true);
    } catch (SQLException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Verify that a query can be executed inside Snowflake by performing the EXPLAIN QUERY
   * functionality. This is done to provide a better error message than a random failure inside
   * Bodo.
   *
   * @param query Query to push into Snowflake.
   */
  private void executeExplainQuery(String query) {
    executeExplainQueryImpl(query, isConnectionCached());
  }

  /**
   * Implementation of executeExplainQuery that enables retrying if a cached connection fails.
   *
   * @param query Query to push into Snowflake.
   * @param shouldRetry Should we retry the connection if we see an exception?
   */
  private void executeExplainQueryImpl(String query, boolean shouldRetry) {
    try {
      executeSnowflakeQuery(String.format("Explain %s", query));
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Error encountered while trying verify a query to push into Snowflake.\n"
                  + "Query: \"\"\"%s\"\"\"\n"
                  + "Snowflake Error Message: %s",
              query, e.getMessage());
      if (shouldRetry) {
        LOGGER.warn(errorMsg);
        closeConnections();
        executeExplainQueryImpl(query, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
  }

  @Override
  public String getCatalogName() {
    return catalogName;
  }

  /**
   * Execute the given query inside Snowflake and return the result set.
   *
   * @param query The query to execute exactly.
   * @return The ResultSet returned by executing the query.
   */
  private ResultSet executeSnowflakeQuery(String query) throws SQLException {
    conn = getConnection();
    Statement stmt = conn.createStatement();
    return stmt.executeQuery(query);
  }
}
