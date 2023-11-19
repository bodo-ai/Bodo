package com.bodosql.calcite.catalog;

import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER;
import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_THREE_LOGGER;
import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER;
import static java.lang.Math.min;

import com.bodosql.calcite.adapter.pandas.StreamingOptions;
import com.bodosql.calcite.adapter.snowflake.BodoSnowflakeSqlDialect;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.utils.Memoizer;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.BodoSqlSchema;
import com.bodosql.calcite.schema.InlineViewMetadata;
import com.bodosql.calcite.sql.BodoSqlUtil;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.CatalogTableImpl;
import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.annotation.Nullable;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.schema.Schema;
import org.apache.calcite.schema.Table;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlNumericLiteral;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.util.SqlString;
import org.apache.calcite.util.Pair;
import org.jetbrains.annotations.NotNull;
import org.json.simple.JSONObject;

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

  // The maximum number of retries for connecting to snowflake before succeeding.
  private static final int maxRetries = 5;

  // The initial time to wait when retrying connections. This will be done with
  // an exponential backoff so this is just the base wait time.
  private static final int backoffMilliseconds = 100;

  // Maximum amount of time we are going to wait between retries
  private static final int maxBackoffMilliseconds = 2000;

  // Default value for WEEK_START
  private static final int defaultWeekStart = 0;

  // Default value for WEEK_OF_YEAR_POLICY
  private static final int defaultWeekOfYearPolicy = 0;

  private @Nullable static BodoTZInfo sfTZInfo;

  private @Nullable Integer weekStart;
  private @Nullable Integer weekOfYearPolicy;
  private @Nullable String currentDatabase;

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
    // Add the warehouse for any complex metadata queries.
    // TODO: Have the ability to have a smaller compilation warehouse.
    this.totalProperties.put("warehouse", warehouseName);
    this.totalProperties.putAll(this.accountInfo);
    this.conn = null;
    this.dbMeta = null;
    // TODO(Nick): Ensure keys are in all caps at the Python level
    if (accountInfo.contains("TIMEZONE")) {
      // This is an optimization to avoid a trip to Snowflake.
      // Note: We assume the Timestamp information can never be loaded as an int.
      this.sfTZInfo = new BodoTZInfo(accountInfo.getProperty("TIMEZONE"), "str");
    }

    this.weekStart = parseIntegerProperty(accountInfo, "WEEK_START", defaultWeekStart);
    this.weekOfYearPolicy =
        parseIntegerProperty(accountInfo, "WEEK_OF_YEAR_POLICY", defaultWeekOfYearPolicy);

    this.currentDatabase = catalogName;
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
          VERBOSE_LEVEL_TWO_LOGGER.info(
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
   * Parse a Snowflake Session parameter into an Integer.
   *
   * @param paramName The name of the parameter to fetch.
   * @param defaultValue The default value of the parameter if it is not found.
   * @return The value of the parameter as an Integer.
   * @throws RuntimeException An exception occurs parsing the parameter string to an Integer.
   */
  private Integer parseSnowflakeIntegerParam(String paramName, Integer defaultValue)
      throws RuntimeException {
    Integer result = defaultValue;
    String paramStr = accountInfo.getProperty(paramName);
    if (paramStr != null) {
      try {
        result = Integer.valueOf(paramStr);
      } catch (NumberFormatException e) {
        throw new RuntimeException("Unable to parse snowflake session variable " + paramName, e);
      }
    }
    return result;
  }

  private Integer parseIntegerProperty(
      Properties accountInfo, String propertyName, Integer defaultValue) throws RuntimeException {
    Integer result = defaultValue;
    String propertyValueStr = accountInfo.getProperty(propertyName);
    if (propertyValueStr != null) {
      try {
        result = Integer.valueOf(propertyValueStr);
      } catch (NumberFormatException e) {
        throw new RuntimeException("Unable to parse snowflake session variable " + propertyName, e);
      }
    }
    return result;
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
   * Get the Snowflake WEEK_START session parameter and update the cached value.
   *
   * @return The value of WEEK_START.
   * @throws SQLException An exception occurs contacting snowflake.
   */
  private Integer getSnowflakeWeekStart(boolean shouldRetry) throws SQLException {
    if (weekStart == null) {
      weekStart = parseSnowflakeIntegerParam("WEEK_START", defaultWeekStart);
    }
    return weekStart;
  }

  /**
   * Get the Snowflake WEEK_OF_YEAR_POLICY session parameter and update the cached value.
   *
   * @return The value of WEEK_OF_YEAR_POLICY.
   * @throws SQLException An exception occurs contacting snowflake.
   */
  private Integer getSnowflakeWeekOfYearPolicy(boolean shouldRetry) throws SQLException {
    if (weekOfYearPolicy == null) {
      weekOfYearPolicy = parseSnowflakeIntegerParam("WEEK_OF_YEAR_POLICY", defaultWeekOfYearPolicy);
    }
    return weekOfYearPolicy;
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
        VERBOSE_LEVEL_TWO_LOGGER.warning(errorMsg);
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

  public static class SnowflakeTypeInfo {
    public final BodoSQLColumnDataType columnDataType;
    public final BodoSQLColumnDataType elemType;
    public final int precision;

    SnowflakeTypeInfo(BodoSQLColumnDataType dtype, BodoSQLColumnDataType etype, int prec) {
      columnDataType = dtype;
      elemType = etype;
      precision = prec;
    }
  }

  /**
   * Parse the BodoSQL column data type obtained from a type returned by a "DESCRIBE TABLE" call.
   * This is done to obtain information about types that cannot be communicated via JDBC APIs.
   *
   * @param typeName The type name string returned for a column by Snowflake's describe table query.
   * @return A dataclass of the BodoSQLColumnDataType of the column itself, the elem type if it is
   *     an ARRAY containing nested data and precision for the type in question. If the type is not
   *     Time then the precision value is garbage and its value is ignored.
   */
  public static SnowflakeTypeInfo snowflakeTypeNameToTypeInfo(String typeName) {
    // Convert the type to all caps to simplify checking.
    typeName = typeName.toUpperCase(Locale.ROOT);
    final BodoSQLColumnDataType columnDataType;
    BodoSQLColumnDataType elemType = BodoSQLColumnDataType.EMPTY;
    int precision = 0;
    if (typeName.startsWith("NUMBER")) {
      // If we encounter a number type we need to parse it to determine the actual
      // type.
      // The type information is of the form NUMBER(PRECISION, SCALE)
      String internalFields = typeName.split("\\(|\\)")[1];
      String[] numericParts = internalFields.split(",");
      precision = Integer.valueOf(numericParts[0].trim());
      int scale = Integer.valueOf(numericParts[1].trim());
      if (scale > 0) {
        // If scale > 0 then we have a Float, Double, or Decimal Type
        // Currently we only support having DOUBLE inside SQL
        columnDataType = BodoSQLColumnDataType.FLOAT64;
      } else {
        // We have an integer type.
        if (precision < 3) {
          // If we use only 2 digits we know that this value fits in an int8
          columnDataType = BodoSQLColumnDataType.INT8;
        } else if (precision < 5) {
          // Using 4 digits always fits in an int16
          columnDataType = BodoSQLColumnDataType.INT16;
        } else if (precision < 10) {
          // Using 10 digits fits in an int32
          columnDataType = BodoSQLColumnDataType.INT32;
        } else {
          // Our max type is int64
          columnDataType = BodoSQLColumnDataType.INT64;
        }
      }
    } else if (typeName.equals("BOOLEAN")) {
      columnDataType = BodoSQLColumnDataType.BOOL8;
    } else if (typeName.equals("TEXT") || typeName.equals("STRING")) {
      columnDataType = BodoSQLColumnDataType.STRING;
      // TEXT/STRING using no defined limit.
      precision = RelDataType.PRECISION_NOT_SPECIFIED;
    } else if (typeName.startsWith("VARCHAR") || typeName.startsWith("CHAR")) {
      columnDataType = BodoSQLColumnDataType.STRING;
      // Load max string information if it exists
      precision = RelDataType.PRECISION_NOT_SPECIFIED;
      String[] typeFields = typeName.split("\\(|\\)");
      if (typeFields.length > 1) {
        // The precision is passed as VARCHAR(N)/CHAR(N)
        precision = Integer.valueOf(typeFields[1].trim());
        // If precision is the Snowflake max, convert it back to -1 to avoid
        // max character issues.
        if (precision == 16777216) {
          precision = RelDataType.PRECISION_NOT_SPECIFIED;
        }
      }
    } else if (typeName.equals("DATE")) {
      columnDataType = BodoSQLColumnDataType.DATE;
    } else if (typeName.startsWith("DATETIME")) {
      // TODO(njriasan): can DATETIME contain precision?
      // Most likely Snowflake should never return this type as
      // its a user alias
      columnDataType = BodoSQLColumnDataType.DATETIME;
      precision = 9;
    } else if (typeName.startsWith("TIMESTAMP_NTZ(")) {
      columnDataType = BodoSQLColumnDataType.DATETIME;
      // Determine the precision by parsing the type.
      String precisionString = typeName.split("\\(|\\)")[1];
      precision = Integer.valueOf(precisionString.trim());
    } else if (typeName.startsWith("TIMESTAMP_TZ(") || typeName.startsWith("TIMESTAMP_LTZ(")) {
      // TODO(njriasan): Remove TIMESTAMP_TZ
      columnDataType = BodoSQLColumnDataType.TZ_AWARE_TIMESTAMP;
      // Determine the precision by parsing the type.
      String precisionString = typeName.split("\\(|\\)")[1];
      precision = Integer.valueOf(precisionString.trim());
    } else if (typeName.startsWith("TIME(")) {
      columnDataType = BodoSQLColumnDataType.TIME;
      // Determine the precision by parsing the type.
      String precisionString = typeName.split("\\(|\\)")[1];
      precision = Integer.valueOf(precisionString.trim());
    } else if (typeName.startsWith("BINARY") || typeName.startsWith("VARBINARY")) {
      columnDataType = BodoSQLColumnDataType.BINARY;
    } else if (typeName.equals("VARIANT")) {
      columnDataType = BodoSQLColumnDataType.VARIANT;
    } else if (typeName.equals("OBJECT")) {
      // TODO: Replace with a map type if possible.
      columnDataType = BodoSQLColumnDataType.JSON_OBJECT;
    } else if (typeName.startsWith("ARRAY")) {
      // TODO: Replace with the true inner dtype rather than defaulting to variant
      columnDataType = BodoSQLColumnDataType.ARRAY;
      elemType = BodoSQLColumnDataType.VARIANT;
    } else if (typeName.startsWith("TIMESTAMP")) {
      // TODO: Leverage the snowflake session parameter.
      columnDataType = BodoSQLColumnDataType.DATETIME;
    } else if (typeName.startsWith("FLOAT")
        || typeName.startsWith("DOUBLE")
        || typeName.equals("REAL")
        || typeName.equals("DECIMAL")
        || typeName.equals("NUMERIC")) {
      // A snowflake bug outputs float for double and float, so we match double
      columnDataType = BodoSQLColumnDataType.FLOAT64;
    } else if (typeName.startsWith("INT")
        || typeName.equals("BIGINT")
        || typeName.equals("SMALLINT")
        || typeName.equals("TINYINT")
        || typeName.equals("BYTEINT")) {
      // Snowflake treats these all internally as suggestions so we must use INT64
      columnDataType = BodoSQLColumnDataType.INT64;
    } else {
      // Unsupported types (e.g. GEOGRAPHY and GEOMETRY) may be in the table but
      // unused,
      // so we don't fail here.
      columnDataType = BodoSQLColumnDataType.UNSUPPORTED;
    }
    SnowflakeTypeInfo typeInfo = new SnowflakeTypeInfo(columnDataType, elemType, precision);
    return typeInfo;
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
      String dotSeparatedTableName =
          String.format(Locale.ROOT, "%s.%s.%s", catalogName, schema.getName(), tableName);
      VERBOSE_LEVEL_THREE_LOGGER.info(
          String.format(Locale.ROOT, "Validating table: %s", dotSeparatedTableName));
      // Fetch the timezone info.
      BodoTZInfo tzInfo = getSnowflakeTimezone(shouldRetry);
      // Table metadata needs to be derived from describe table because some types
      // aren't available via the JDBC connector. In particular Snowflake doesn't
      // communicate
      // information about Variant, Array, or Object.
      ResultSet tableInfo =
          executeSnowflakeQuery(
              String.format(Locale.ROOT, "Describe table %s", dotSeparatedTableName));
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
        SnowflakeTypeInfo typeInfo = snowflakeTypeNameToTypeInfo(dataType);
        BodoSQLColumnDataType type = typeInfo.columnDataType;
        BodoSQLColumnDataType elemType = typeInfo.elemType;
        int precision = typeInfo.precision;
        // The column is nullable unless we are certain it has no nulls.
        boolean nullable = tableInfo.getString(4).toUpperCase(Locale.ROOT).equals("Y");
        columns.add(
            new BodoSQLColumnImpl(
                readName, writeName, type, elemType, nullable, tzInfo, precision));
      }
      return new CatalogTableImpl(tableName, schema, columns);
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get table '%s' for Schema '%s' from Snowflake account. Error message: %s",
              tableName, schema.getName(), e);
      if (shouldRetry) {
        VERBOSE_LEVEL_TWO_LOGGER.warning(errorMsg);
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
        VERBOSE_LEVEL_TWO_LOGGER.warning(errorMsg);
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
          // schemaNames() returns a string in the form of '["BD.Scheam1",
          // "DB.Schema2"...]'
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
        VERBOSE_LEVEL_TWO_LOGGER.warning(errorMsg);
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

    // Hide the username, password, and accountName for
    // sharing code without leaking credentials.
    String username;
    String password;
    String accountName;
    if (RelationalAlgebraGenerator.hideCredentials) {
      username = "USERNAME***";
      password = this.password.isEmpty() ? "" : "PASSWORD***";
      accountName = "ACCOUNT***";
    } else {
      username = this.username;
      password = this.password;
      accountName = this.accountName;
    }

    // Only add extra colon if password is given
    if (!password.isEmpty()) {
      password = String.format(":%s", URLEncoder.encode(password));
    }

    try {
      connString.append(
          String.format(
              "snowflake://%s%s@%s/%s",
              URLEncoder.encode(username, "UTF-8"),
              password,
              URLEncoder.encode(accountName, "UTF-8"),
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
  public Expr generateWriteCode(
      Variable varName,
      String schemaName,
      String tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType tableType) {
    return new Expr.Raw(
        String.format(
            "%s.to_sql('%s', '%s', schema='%s', if_exists='%s', _bodo_create_table_type='%s',"
                + " index=False)",
            varName.emit(),
            tableName,
            generatePythonConnStr(schemaName),
            schemaName,
            ifExists.asToSqlKwArgument(),
            tableType.asStringKeyword()));
  }

  @Override
  public Expr generateStreamingWriteInitCode(
      Expr.IntegerLiteral operatorID,
      String schemaName,
      String tableName,
      BodoSQLCatalog.ifExistsBehavior ifExists,
      SqlCreateTable.CreateTableType createTableType) {
    return new Expr.Call(
        "bodo.io.snowflake_write.snowflake_writer_init",
        operatorID,
        new Expr.StringLiteral(generatePythonConnStr(schemaName)),
        new Expr.StringLiteral(tableName),
        new Expr.StringLiteral(schemaName),
        new Expr.StringLiteral(ifExists.asToSqlKwArgument()),
        new Expr.StringLiteral(createTableType.asStringKeyword()));
  }

  @Override
  public Expr generateStreamingWriteAppendCode(
      Variable stateVarName,
      Variable tableVarName,
      Variable colNamesGlobal,
      Variable isLastVarName,
      Variable iterVarName,
      Expr columnPrecisions) {
    return new Expr.Call(
        "bodo.io.snowflake_write.snowflake_writer_append_table",
        List.of(
            stateVarName,
            tableVarName,
            colNamesGlobal,
            isLastVarName,
            iterVarName,
            columnPrecisions));
  }

  /**
   * Generates the code necessary to produce a read expression from Snowflake.
   *
   * @param schemaName Name of the schema to use when reading.
   * @param tableName Name of the table to use when reading.
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The options to use if streaming is enabled.
   * @return The generated code to produce a read.
   */
  @Override
  public Expr generateReadCode(
      String schemaName,
      String tableName,
      boolean useStreaming,
      StreamingOptions streamingOptions) {
    // TODO: Convert to use Expr.Call
    String streamingArg = "";
    if (useStreaming) {
      streamingArg = "_bodo_chunksize=" + streamingOptions.getChunkSize();
    }
    return new Expr.Raw(
        String.format(
            "pd.read_sql('%s', '%s', _bodo_is_table_input=True, _bodo_read_as_table=True, %s)",
            tableName, generatePythonConnStr(schemaName), streamingArg));
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
        VERBOSE_LEVEL_TWO_LOGGER.warning(
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
  public Expr generateRemoteQuery(String query) {
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
    return new Expr.Raw(
        String.format(
            "pd.read_sql('%s', '%s', _bodo_read_as_table=True)",
            query, generatePythonConnStr(schemaName)));
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

  @Override
  public @Nullable Integer getWeekStart() {
    if (this.weekStart == null) {
      try {
        this.weekStart = getSnowflakeWeekStart(true);
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }

    return this.weekStart;
  }

  @Override
  public @Nullable Integer getWeekOfYearPolicy() {
    if (this.weekOfYearPolicy == null) {
      try {
        this.weekOfYearPolicy = getSnowflakeWeekOfYearPolicy(true);
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }
    return this.weekOfYearPolicy;
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
        VERBOSE_LEVEL_TWO_LOGGER.warning(errorMsg);
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
   * Estimate the row count for a given fully qualified table.
   *
   * @param tableName qualified table name.
   * @return estimated row count.
   */
  public @Nullable Double estimateRowCount(List<String> tableName) {
    // While this function is named estimateRowCount, it presently returns
    // the exact row count because that information is readily available to us.
    // This method is still named estimateRowCount in case there ends up being
    // a more efficient query that only gives us an estimation so we don't have
    // to rename this function.
    //
    // So while this function returns an exact row count, the result should just
    // be used as an estimation rather than an exact number of rows.
    SqlSelect select = rowCountQuery(tableName);
    SqlString sql =
        select.toSqlString(
            (c) -> c.withClauseStartsLine(false).withDialect(BodoSnowflakeSqlDialect.DEFAULT));

    @Nullable Long snowflakeResult = trySubmitLongMetadataQuery(sql);
    @Nullable Double output = snowflakeResult == null ? null : (double) snowflakeResult;
    return output;
  }

  /**
   * Estimate the number of distinct entries of the given column for a given fully qualified table.
   *
   * @param tableName qualified table name.
   * @return estimated distinct count.
   */
  public @Nullable Double estimateColumnDistinctCount(List<String> tableName, String columnName) {
    // This function calls approx_count_distinct on the given column name.
    SqlSelect select = approxCountDistinctQuery(tableName, columnName);
    SqlString sql =
        select.toSqlString(
            (c) -> c.withClauseStartsLine(false).withDialect(BodoSnowflakeSqlDialect.DEFAULT));

    @Nullable Long snowflakeResult = trySubmitLongMetadataQuery(sql);
    @Nullable Double output = snowflakeResult == null ? null : (double) snowflakeResult;
    return output;
  }

  /**
   * Estimate the number of distinct entries of the given column for a given fully qualified table,
   * using sampling instead of querying the entire table.
   *
   * @param tableName qualified table name.
   * @param columnName which column to sample from the table.
   * @param rowCount how many rows does the full table have.
   * @return estimated distinct count.
   */
  public @Nullable Double estimateColumnDistinctCountWithSampling(
      List<String> tableName, String columnName, Double rowCount) {
    // Sampling is not supported for views. This is because Row based
    // sampling is generally too slow and views do not support system
    // sampling.
    boolean isView = tryGetViewMetadataFn.apply(tableName) != null;
    if (isView) {
      String dotTableName = String.join(".", tableName);
      String loggingMessage =
          String.format(
              Locale.ROOT, "Skipping attempt to sample from %s as it is a view", dotTableName);
      VERBOSE_LEVEL_TWO_LOGGER.info(loggingMessage);
      return null;
    }
    // This function calls approx_count_distinct on the given column name.
    SqlSelect select = approxCountDistinctSamplingQuery(tableName, columnName);
    SqlString sql =
        select.toSqlString(
            (c) -> c.withClauseStartsLine(false).withDialect(BodoSnowflakeSqlDialect.DEFAULT));

    @Nullable Pair<Long, Long> snowflakeResult = trySubmitDistinctSampleQuery(sql);
    // If the original query failed, try again with sampling
    if (snowflakeResult != null) {
      Long sampleSize = snowflakeResult.left;
      Long sampleDistinct = snowflakeResult.right;
      return inferDistinctCountFromSample(rowCount, (double) sampleSize, (double) sampleDistinct);
    } else {
      return null;
    }
  }

  /**
   * Approximation of the error function ERF. Copied from
   * https://introcs.cs.princeton.edu/java/21function/ErrorFunction.java.html
   */
  public static double erf(double z) {
    double t = 1.0 / (1.0 + 0.5 * Math.abs(z));
    double exponent = 0.17087277;
    exponent *= t;
    exponent += -0.82215223;
    exponent *= t;
    exponent += 1.48851587;
    exponent *= t;
    exponent += -1.13520398;
    exponent *= t;
    exponent += 0.27886807;
    exponent *= t;
    exponent += -0.18628806;
    exponent *= t;
    exponent += 0.09678418;
    exponent *= t;
    exponent += 0.37409196;
    exponent *= t;
    exponent += 1.00002368;
    exponent *= t;
    exponent += -1.26551223;
    exponent += -z * z;
    return Math.signum(z) * (1 - t * Math.exp(exponent));
  }

  /**
   * Returns the cumulative value of a normal distribution from negative infinity up to a certain
   * value.
   *
   * @param x The value to accumulate up to.
   * @param mean The mean of the distribution
   * @param var The variance of the distribution
   * @return The probability mass under the curve up to x.
   */
  public static double normalCdf(double x, double mean, double var) {
    return 0.5 * (1 + erf((x - mean) / (Math.sqrt(2) * var)));
  }

  /**
   * Calculates the approximate number of distinct rows based on a sample. See here for algorithm
   * design description:
   * https://bodo.atlassian.net/wiki/spaces/B/pages/1468235780/Snowflake+Metadata+Handling#Testing-Proposed-Algorithm-on-Sample-Data
   *
   * @param rowCount The number of rows in the full table.
   * @param sampleSize The number of rows in the sample.
   * @param sampleDistinct The number of distinct rows in the sample.
   * @return An approximation of the number of distinct rows in the original table
   */
  public static Double inferDistinctCountFromSample(
      Double rowCount, Double sampleSize, Double sampleDistinct) {
    double mean = sampleSize / sampleDistinct;
    double C0 = normalCdf(0.0, mean, mean);
    double C1 = normalCdf(1.0, mean, mean);
    double CD = normalCdf(sampleDistinct, mean, mean);
    double normalizedRatio = (C1 - C0) / (CD - C0);
    double appearsOnce = Math.floor(Math.max((sampleDistinct * normalizedRatio) - 1.0, 0.0));
    return sampleDistinct + appearsOnce * (rowCount - sampleSize) / sampleSize;
  }

  /**
   * Submits the specified query to Snowflake for evaluation. Expects the return to contain exactly
   * one value integer, which is returned by this function. Returns null in the event of a timeout,
   * which is 30 seconds by default. May cause undefined behavior if the supplied sql string does
   * not return exactly one integer value.
   *
   * @param sql The SQL query to submit to snowflake
   * @return The integer value returned by the sql query
   */
  public @Nullable Long trySubmitLongMetadataQuery(SqlString sql) {
    String sqlString = sql.getSql();
    // TODO(jsternberg): This class mostly doesn't handle connections correctly.
    // This should be inside of a try/resource block, but it will likely cause
    // this to fail because we're not utilizing connection pooling or reuse and
    // closing this is likely to result in more confusion.
    try {
      Connection conn = getConnection();
      try (PreparedStatement stmt = conn.prepareStatement(sqlString)) {
        // Value is in seconds
        String defaultTimeout = "30";
        stmt.setQueryTimeout(
            Integer.parseInt(
                System.getenv()
                    .getOrDefault(
                        "BODOSQL_METADATA_QUERY_TIMEOUT_TIME_IN_SECONDS", defaultTimeout)));
        try (ResultSet rs = stmt.executeQuery()) {
          // Only one row matters.
          if (rs.next()) {
            return rs.getLong(1);
          }
        }
      }
    } catch (SQLException ex) {
      // Error executing the query. Nothing we can do except
      // log the information.
      String msg =
          String.format(
              Locale.ROOT,
              "Failure in attempted metadata query\n%s\nReason given for failure\n%s",
              sql,
              ex.getMessage());
      VERBOSE_LEVEL_ONE_LOGGER.warning(msg);
    }
    return null;
  }

  /**
   * Submits the specified query to Snowflake for evaluation. Expects the return to contain exactly
   * two integers, which are returned by this function. Returns null in the event of a timeout,
   * which is 30 seconds by default. May cause undefined behavior if the supplied sql string does
   * not return exactly two integer values.
   *
   * @param sql The SQL query to submit to snowflake
   * @return The integer values returned by the sql query
   */
  public @Nullable Pair<Long, Long> trySubmitDistinctSampleQuery(SqlString sql) {
    String sqlString = sql.getSql();

    try {
      Connection conn = getConnection();
      try (PreparedStatement stmt = conn.prepareStatement(sqlString)) {
        // Value is in seconds
        String defaultTimeout = "30";
        stmt.setQueryTimeout(
            Integer.parseInt(
                System.getenv()
                    .getOrDefault(
                        "BODOSQL_METADATA_QUERY_TIMEOUT_TIME_IN_SECONDS", defaultTimeout)));
        try (ResultSet rs = stmt.executeQuery()) {
          // Only one row matters.
          if (rs.next()) {
            Long sampleSize = rs.getLong(1);
            Long distinctCount = rs.getLong(2);
            return new Pair(sampleSize, distinctCount);
          }
        }
      }
    } catch (SQLException ex) {
      // Error executing the query. Nothing we can do except
      // log the information.
      String msg =
          String.format(
              Locale.ROOT,
              "Failure in attempted distinctness metadata query\n%s\nReason given for failure\n%s",
              sql,
              ex.getMessage());
      VERBOSE_LEVEL_ONE_LOGGER.warning(msg);
    }
    return null;
  }

  /**
   * Creates a query to get the approximate distinct count of a certain column of a certain table
   * using sampling. The table must be a real table and not a view.
   *
   * @param tableName qualified table name.
   * @param columnName which column to sample from the table.
   * @return The sampling query.
   */
  private SqlSelect approxCountDistinctSamplingQuery(List<String> tableName, String columnName) {
    SqlNodeList selectList =
        SqlNodeList.of(
            SqlStdOperatorTable.COUNT.createCall(SqlParserPos.ZERO, SqlIdentifier.STAR),
            SqlStdOperatorTable.APPROX_COUNT_DISTINCT.createCall(
                SqlParserPos.ZERO, new SqlIdentifier(columnName, SqlParserPos.ZERO)));
    SqlNodeList from = SqlNodeList.of(new SqlIdentifier(tableName, SqlParserPos.ZERO));
    final String samplingRate = "1";
    SqlNumericLiteral samplePercentage =
        SqlLiteral.createExactNumeric(samplingRate, SqlParserPos.ZERO);
    SqlSampleSpec tableSampleSpec =
        BodoSqlUtil.createTableSample(SqlParserPos.ZERO, false, samplePercentage, false, false, -1);
    SqlLiteral tableSampleLiteral = SqlLiteral.createSample(tableSampleSpec, SqlParserPos.ZERO);
    from =
        SqlNodeList.of(
            SqlStdOperatorTable.TABLESAMPLE.createCall(
                SqlParserPos.ZERO, from, tableSampleLiteral));
    return new SqlSelect(
        SqlParserPos.ZERO,
        SqlNodeList.EMPTY,
        selectList,
        from,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }

  private SqlSelect rowCountQuery(List<String> tableName) {
    SqlNodeList selectList =
        SqlNodeList.of(SqlStdOperatorTable.COUNT.createCall(SqlParserPos.ZERO, SqlIdentifier.STAR));
    SqlNodeList from = SqlNodeList.of(new SqlIdentifier(tableName, SqlParserPos.ZERO));
    return new SqlSelect(
        SqlParserPos.ZERO,
        SqlNodeList.EMPTY,
        selectList,
        from,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }

  private SqlSelect approxCountDistinctQuery(List<String> tableName, String columnName) {
    SqlNodeList selectList =
        SqlNodeList.of(
            SqlStdOperatorTable.APPROX_COUNT_DISTINCT.createCall(
                SqlParserPos.ZERO, new SqlIdentifier(columnName, SqlParserPos.ZERO)));
    SqlNodeList from = SqlNodeList.of(new SqlIdentifier(tableName, SqlParserPos.ZERO));
    return new SqlSelect(
        SqlParserPos.ZERO,
        SqlNodeList.EMPTY,
        selectList,
        from,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null,
        null);
  }

  /**
   * Execute the given query inside Snowflake and return the result set with the given timeout.
   *
   * @param query The query to execute exactly.
   * @param timeout The timeout for the query in seconds.
   * @return The ResultSet returned by executing the query.
   */
  private ResultSet executeSnowflakeQuery(String query, int timeout) throws SQLException {
    conn = getConnection();
    Statement stmt = conn.createStatement();
    stmt.setQueryTimeout(timeout);
    return stmt.executeQuery(query);
  }

  /**
   * Execute the given query inside Snowflake and return the result set with no timeout.
   *
   * @param query The query to execute exactly.
   * @return The ResultSet returned by executing the query.
   */
  private ResultSet executeSnowflakeQuery(String query) throws SQLException {
    return executeSnowflakeQuery(query, 0);
  }

  /**
   * Load the view metadata information from the catalog. If the table is not a view or no
   * information can be found this should return NULL.
   *
   * @param names A list of two names starting with SCHEMA_NAME and ending with TABLE_NAME.
   * @return The InlineViewMetadata loaded from the catalog or null if no information is available.
   */
  public @Nullable InlineViewMetadata tryGetViewMetadata(List<String> names) {
    return tryGetViewMetadataFn.apply(names);
  }

  private final Function<List<String>, InlineViewMetadata> tryGetViewMetadataFn =
      Memoizer.memoize(this::tryGetViewMetadataImpl);

  /**
   * The actual implementation for tryGetViewMetadata including calls to Snowflake. We add an extra
   * layer of indirection to ensure this function is cached.
   */
  private @Nullable InlineViewMetadata tryGetViewMetadataImpl(List<String> names) {
    try {
      final String query =
          String.format(
              Locale.ROOT,
              "Show views like '%s' in schema %s.%s starts with '%s'",
              names.get(1),
              catalogName,
              names.get(0),
              names.get(1));
      ResultSet paramInfo = executeSnowflakeQuery(query);
      if (paramInfo.next()) {
        // See the return types: https://docs.snowflake.com/en/sql-reference/sql/show-views.
        // Note: This is 1 indexed
        String queryDefinition = paramInfo.getString(8);
        // is_secure is not safe to inline.
        String unsafeToInline = paramInfo.getString(9);
        String isMaterialized = paramInfo.getString(10);
        return new InlineViewMetadata(
            Boolean.valueOf(unsafeToInline), Boolean.valueOf(isMaterialized), queryDefinition);
      } else {
        return null;
      }
    } catch (SQLException e) {
      // If we cannot get view information just return NULL.
      return null;
    }
  }

  /**
   * Determine if we have read access to a table by trying to submit a read query to a table with
   * limit 0. This is necessary because you need admin permissions to directly check permissions and
   * nested permissions make lookups very difficult.
   *
   * @param names A list of two names starting with SCHEMA_NAME and ending with TABLE_NAME.
   * @return Do we have read support.
   */
  public @Nullable boolean canReadTable(List<String> names) {
    return canReadTableFn.apply(names);
  }

  private final Function<List<String>, Boolean> canReadTableFn =
      Memoizer.memoize(this::canReadTableImpl);

  /**
   * The actual implementation for canReadTable including calls to Snowflake. We add an extra layer
   * of indirection to ensure this function is cached.
   */
  private @NotNull boolean canReadTableImpl(List<String> names) {
    String dotTableName =
        String.format(Locale.ROOT, "%s.%s.%s", catalogName, names.get(0), names.get(1));
    try {
      final String query = String.format(Locale.ROOT, "Select * from %s limit 0", dotTableName);
      executeSnowflakeQuery(query, 5);
      return true;
    } catch (SQLException e) {
      // If the query errors just return and log.
      String errorMsg =
          String.format(
              "Unable to read table '%s' from Snowflake. Error message: %s",
              dotTableName, e.getMessage());
      VERBOSE_LEVEL_TWO_LOGGER.info(errorMsg);
      return false;
    }
  }
}
