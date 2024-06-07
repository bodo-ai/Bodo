package com.bodosql.calcite.catalog;

import static com.bodosql.calcite.adapter.snowflake.SnowflakeUtils.parseSnowflakeShowFunctionsArguments;
import static com.bodosql.calcite.adapter.snowflake.SnowflakeUtils.snowflakeYesNoToBoolean;
import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_ONE_LOGGER;
import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_THREE_LOGGER;
import static com.bodosql.calcite.application.PythonLoggers.VERBOSE_LEVEL_TWO_LOGGER;
import static com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable.EXTERNAL_TABLE_FILES_NAME;
import static java.lang.Math.min;

import com.bodosql.calcite.adapter.bodo.StreamingOptions;
import com.bodosql.calcite.adapter.snowflake.BodoSnowflakeSqlDialect;
import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLTypeSystems.BodoSQLRelDataTypeSystem;
import com.bodosql.calcite.application.RelationalAlgebraGenerator;
import com.bodosql.calcite.application.operatorTables.TableFunctionOperatorTable;
import com.bodosql.calcite.application.utils.Memoizer;
import com.bodosql.calcite.application.write.SnowflakeIcebergWriteTarget;
import com.bodosql.calcite.application.write.SnowflakeNativeWriteTarget;
import com.bodosql.calcite.application.write.WriteTarget;
import com.bodosql.calcite.application.write.WriteTarget.IfExistsBehavior;
import com.bodosql.calcite.ddl.DDLExecutionResult;
import com.bodosql.calcite.ddl.DDLExecutor;
import com.bodosql.calcite.ddl.NamespaceAlreadyExistsException;
import com.bodosql.calcite.ddl.NamespaceNotFoundException;
import com.bodosql.calcite.ddl.ViewAlreadyExistsException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.schema.CatalogSchema;
import com.bodosql.calcite.schema.InlineViewMetadata;
import com.bodosql.calcite.sql.BodoSqlUtil;
import com.bodosql.calcite.sql.ddl.SnowflakeCreateTableMetadata;
import com.bodosql.calcite.table.BodoSQLColumn;
import com.bodosql.calcite.table.BodoSQLColumn.BodoSQLColumnDataType;
import com.bodosql.calcite.table.BodoSQLColumnImpl;
import com.bodosql.calcite.table.ColumnDataTypeInfo;
import com.bodosql.calcite.table.SnowflakeCatalogTable;
import com.google.common.collect.ImmutableList;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLEncoder;
import java.sql.Connection;
import java.sql.DatabaseMetaData;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;
import java.sql.Statement;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.UUID;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SnowflakeUserDefinedFunction;
import org.apache.calcite.sql.SnowflakeUserDefinedTableFunction;
import org.apache.calcite.sql.SqlIdentifier;
import org.apache.calcite.sql.SqlLiteral;
import org.apache.calcite.sql.SqlNodeList;
import org.apache.calcite.sql.SqlSampleSpec;
import org.apache.calcite.sql.SqlSelect;
import org.apache.calcite.sql.ddl.SqlCreateTable;
import org.apache.calcite.sql.ddl.SqlCreateView;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoTZInfo;
import org.apache.calcite.sql.util.SqlString;
import org.apache.calcite.util.Pair;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.jetbrains.annotations.NotNull;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;

public class SnowflakeCatalog implements BodoSQLCatalog {
  /**
   * See the design described on Confluence: <a
   * href="https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Catalog">...</a>
   */
  private final String connectionString;

  private final String username;

  private final String password;

  private final String accountName;

  private final String warehouseName;

  // Account info contains the username and password information
  private final Properties accountInfo;

  // Snowflake external volume (e.g. S3/ADLS) for writing Iceberg data if available
  private final @Nullable String icebergVolume;

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
  private final @Nullable String currentDatabase;

  private final SnowflakeJDBCExecutor ddlExecutor;

  /**
   * Create the catalog and store the relevant account information.
   *
   * @param username Snowflake username
   * @param password Snowflake password
   * @param accountName User's snowflake account name.
   * @param defaultDatabaseName Name of the default database (catalog in Snowflake terminology). In
   *     the future this should become optional.
   * @param accountInfo Any extra properties to pass to Snowflake.
   * @param icebergVolume Snowflake object storage volume for writing Iceberg tables if available.
   */
  public SnowflakeCatalog(
      String username,
      String password,
      String accountName,
      @Nullable String defaultDatabaseName,
      String warehouseName,
      Properties accountInfo,
      @Nullable String icebergVolume) {
    this.username = username;
    this.password = password;
    this.accountName = accountName;
    this.warehouseName = warehouseName;
    this.connectionString =
        String.format("jdbc:snowflake://%s.snowflakecomputing.com/", accountName);
    this.accountInfo = accountInfo;
    this.icebergVolume = icebergVolume;
    this.totalProperties = new Properties();
    // Add the user and password to the properties for JDBC
    this.totalProperties.put("user", username);
    this.totalProperties.put("password", password);
    // Add the catalog name as the default database.
    this.totalProperties.put("db", defaultDatabaseName);
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

    this.currentDatabase = defaultDatabaseName;
    this.ddlExecutor = new SnowflakeJDBCExecutor();
  }

  /**
   * Get a connection to snowflake via jdbc
   *
   * @return Connection to Snowflake
   * @throws SQLException If database access error occurs
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
          VERBOSE_LEVEL_THREE_LOGGER.info(
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
        return paramInfo.getString("value");
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
  private Integer getSnowflakeWeekStart() throws SQLException {
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
  private Integer getSnowflakeWeekOfYearPolicy() throws SQLException {
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
   * @throws SQLException If database connection error occurs
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
   * @param schemaPath The list of tables to traverse before finding the schema.
   * @return Set of table names.
   */
  @Override
  public Set<String> getTableNames(ImmutableList<String> schemaPath) {
    if (schemaPath.size() != 2) {
      return Set.of();
    }
    return getTableNamesImpl(schemaPath.get(0), schemaPath.get(1), isConnectionCached());
  }

  /**
   * Implementation of getTableNames that enables retrying if a cached connection fails
   *
   * @param databaseName The name of the database to use to load the table.
   * @param schemaName The name of the schema to use to load the table.
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return List of table names
   */
  private Set<String> getTableNamesImpl(
      String databaseName, String schemaName, boolean shouldRetry) {
    try {
      DatabaseMetaData metaData = getDataBaseMetaData(shouldRetry);
      // Passing null for tableNamePattern should match all table names. Although
      // this is not in the public documentation. Passing null for types ensures we
      // allow all possible table types, including regular tables and views.
      ResultSet tableInfo = metaData.getTables(databaseName, schemaName, null, null);
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
              "Unable to get table names for Schema '%s'.'%s' from Snowflake account. Error"
                  + " message: %s",
              databaseName, schemaName, e);
      if (shouldRetry) {
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getTableNamesImpl(databaseName, schemaName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
  }

  /**
   * Returns a table with the given name if found in the given schema from Snowflake. This code
   * connects to Snowflake via JDBC and loads the table metadata using standard JDBC APIs.
   *
   * @param schemaPath The list of schemas to traverse before finding the table.
   * @param tableName Name of the table.
   * @return The table object.
   */
  @Override
  public SnowflakeCatalogTable getTable(ImmutableList<String> schemaPath, String tableName) {
    if (schemaPath.size() != 2) {
      return null;
    }
    return getTableImpl(schemaPath.get(0), schemaPath.get(1), tableName, isConnectionCached());
  }

  /**
   * Implementation of getTable that enables retrying if a cached connection fails
   *
   * @param databaseName The name of the database to use to load the table.
   * @param schemaName The name of the schema to use to load the table.
   * @param tableName Name of the table.
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return The table object.
   */
  private SnowflakeCatalogTable getTableImpl(
      String databaseName, String schemaName, String tableName, boolean shouldRetry) {
    try {
      String dotSeparatedTableName =
          String.format(
              Locale.ROOT,
              "%s.%s.%s",
              makeObjectNameQuoted(databaseName),
              makeObjectNameQuoted(schemaName),
              makeObjectNameQuoted(tableName));
      VERBOSE_LEVEL_THREE_LOGGER.info(
          String.format(Locale.ROOT, "Validating table: %s", dotSeparatedTableName));
      // Fetch the timezone info.
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
        String writeName = tableInfo.getString("name");
        String readName = writeName;
        String dataType = tableInfo.getString("type");
        // The column is nullable unless we are certain it has no nulls.
        boolean isNullable = snowflakeYesNoToBoolean(tableInfo.getString("null?"));
        // Parse the given type for the column type and precision information.
        ColumnDataTypeInfo typeInfo = snowflakeTypeNameToTypeInfo(dataType, isNullable);
        columns.add(new BodoSQLColumnImpl(readName, writeName, typeInfo));
      }
      return new SnowflakeCatalogTable(
          tableName, ImmutableList.of(databaseName, schemaName), columns, this);
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get table '%s' for Schema '%s.%s' from Snowflake account. Error message:"
                  + " %s",
              tableName, databaseName, schemaName, e);
      if (shouldRetry) {
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getTableImpl(databaseName, schemaName, tableName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
  }

  /**
   * Parse the BodoSQL column data type obtained from a type returned by a "DESCRIBE TABLE" call.
   * This is done to obtain information about types that cannot be communicated via JDBC APIs.
   *
   * @param typeName The type name string returned for a column by Snowflake's describe table query.
   * @param isNullable Is the element nullable?v
   * @return A dataclass of the BodoSQLColumnDataType of the column itself, with support for nested
   *     data.
   */
  public static ColumnDataTypeInfo snowflakeTypeNameToTypeInfo(
      String typeName, boolean isNullable) {
    // Convert the type to all caps to simplify checking.
    typeName = typeName.toUpperCase(Locale.ROOT);
    final BodoSQLColumnDataType columnDataType;
    int precision = 0;
    if (typeName.startsWith("NULL")) {
      // This may not ever be reached but is added for completeness.
      columnDataType = BodoSQLColumnDataType.NULL;
      return new ColumnDataTypeInfo(columnDataType, true);
    } else if (typeName.startsWith("NUMBER")) {
      // If we encounter a number type we need to parse it to determine the actual
      // type.
      // The type information is of the form NUMBER(PRECISION, SCALE).
      // For UDFs it may just be NUMBER, in case we use the default of 38, 0.
      final int scale;
      if (typeName.contains("(")) {
        String internalFields = typeName.split("\\(|\\)")[1];
        String[] numericParts = internalFields.split(",");
        precision = Integer.parseInt(numericParts[0].trim());
        scale = Integer.parseInt(numericParts[1].trim());
      } else {
        precision = 38;
        scale = 0;
      }
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
      // Note: We don't use the precision/scale yet, but we should when we add decimal support.
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision, scale);
    } else if (typeName.equals("BOOLEAN")) {
      columnDataType = BodoSQLColumnDataType.BOOL8;
      return new ColumnDataTypeInfo(columnDataType, isNullable);
    } else if (typeName.equals("TEXT") || typeName.equals("STRING")) {
      columnDataType = BodoSQLColumnDataType.STRING;
      // TEXT/STRING using no defined limit.
      precision = RelDataType.PRECISION_NOT_SPECIFIED;
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.startsWith("VARCHAR") || typeName.startsWith("CHAR")) {
      columnDataType = BodoSQLColumnDataType.STRING;
      // Load max string information if it exists
      precision = RelDataType.PRECISION_NOT_SPECIFIED;
      // Snowflake columns should contain precision, but UDFs may not.
      if (typeName.contains("(")) {
        String[] typeFields = typeName.split("\\(|\\)");
        if (typeFields.length > 1) {
          // The precision is passed as VARCHAR(N)/CHAR(N)
          precision = Integer.parseInt(typeFields[1].trim());
        }
      }
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.equals("DATE")) {
      columnDataType = BodoSQLColumnDataType.DATE;
      return new ColumnDataTypeInfo(columnDataType, isNullable);
    } else if (typeName.startsWith("DATETIME")) {
      // TODO(njriasan): can DATETIME contain precision?
      // Most likely Snowflake should never return this type as
      // its a user alias
      columnDataType = BodoSQLColumnDataType.TIMESTAMP_NTZ;
      precision = BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION;
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.startsWith("TIMESTAMP_NTZ")) {
      columnDataType = BodoSQLColumnDataType.TIMESTAMP_NTZ;
      // Snowflake table types should contain precision, but UDFs may not.
      if (typeName.contains("(")) {
        // Determine the precision by parsing the type.
        String precisionString = typeName.split("\\(|\\)")[1];
        precision = Integer.parseInt(precisionString.trim());
      } else {
        precision = BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION;
      }
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.startsWith("TIMESTAMP_TZ") || typeName.startsWith("TIMESTAMP_LTZ")) {
      columnDataType =
          (typeName.startsWith("TIMESTAMP_TZ") && RelationalAlgebraGenerator.enableTimestampTz)
              ? BodoSQLColumnDataType.TIMESTAMP_TZ
              : BodoSQLColumnDataType.TIMESTAMP_LTZ;
      // Snowflake table types should contain precision, but UDFs may not.
      if (typeName.contains("(")) {
        // Determine the precision by parsing the type.
        String precisionString = typeName.split("\\(|\\)")[1];
        precision = Integer.parseInt(precisionString.trim());
      } else {
        precision = BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION;
      }
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.startsWith("TIME")) {
      columnDataType = BodoSQLColumnDataType.TIME;
      // Snowflake table types should contain precision, but UDFs may not.
      if (typeName.contains("(")) {
        // Determine the precision by parsing the type.
        String precisionString = typeName.split("\\(|\\)")[1];
        precision = Integer.parseInt(precisionString.trim());
      } else {
        precision = BodoSQLRelDataTypeSystem.MAX_DATETIME_PRECISION;
      }
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.startsWith("BINARY") || typeName.startsWith("VARBINARY")) {
      columnDataType = BodoSQLColumnDataType.BINARY;
      precision = RelDataType.PRECISION_NOT_SPECIFIED;
      // Snowflake columns should contain precision, but UDFs may not.
      if (typeName.contains("(")) {
        String[] typeFields = typeName.split("\\(|\\)");
        if (typeFields.length > 1) {
          // The precision is passed as VARCHAR(N)/CHAR(N)
          precision = Integer.parseInt(typeFields[1].trim());
        }
      }
      return new ColumnDataTypeInfo(columnDataType, isNullable, precision);
    } else if (typeName.equals("VARIANT")) {
      columnDataType = BodoSQLColumnDataType.VARIANT;
      return new ColumnDataTypeInfo(columnDataType, isNullable);
    } else if (typeName.equals("OBJECT")) {
      // TODO: Replace with a map type if possible.
      columnDataType = BodoSQLColumnDataType.JSON_OBJECT;
      // Assume keys and values are always nullable
      ColumnDataTypeInfo key = new ColumnDataTypeInfo(BodoSQLColumnDataType.STRING, true);
      ColumnDataTypeInfo value = new ColumnDataTypeInfo(BodoSQLColumnDataType.VARIANT, true);
      return new ColumnDataTypeInfo(columnDataType, isNullable, key, value);
    } else if (typeName.startsWith("ARRAY")) {
      columnDataType = BodoSQLColumnDataType.ARRAY;
      // Assume inner elements are always nullable
      ColumnDataTypeInfo child = new ColumnDataTypeInfo(BodoSQLColumnDataType.VARIANT, true);
      return new ColumnDataTypeInfo(columnDataType, isNullable, child);
    } else if (typeName.startsWith("FLOAT")
        || typeName.startsWith("DOUBLE")
        || typeName.equals("REAL")
        || typeName.equals("DECIMAL")
        || typeName.equals("NUMERIC")) {
      // A snowflake bug outputs float for double and float, so we match double
      columnDataType = BodoSQLColumnDataType.FLOAT64;
      return new ColumnDataTypeInfo(columnDataType, isNullable);
    } else if (typeName.startsWith("INT")
        || typeName.equals("BIGINT")
        || typeName.equals("SMALLINT")
        || typeName.equals("TINYINT")
        || typeName.equals("BYTEINT")) {
      // Snowflake treats these all internally as suggestions, so we must use INT64
      columnDataType = BodoSQLColumnDataType.INT64;
      return new ColumnDataTypeInfo(columnDataType, isNullable);
    } else {
      // Unsupported types (e.g. GEOGRAPHY and GEOMETRY) may be in the table but
      // unused,
      // so we don't fail here.
      columnDataType = BodoSQLColumnDataType.UNSUPPORTED;
      return new ColumnDataTypeInfo(columnDataType, isNullable);
    }
  }

  /**
   * Get the available subSchema names for the given path. Currently, we only support fetching the
   * schemas found the default database.
   *
   * @param schemaPath The parent schema path to check.
   * @return Set of available schema names.
   */
  @Override
  public Set<String> getSchemaNames(ImmutableList<String> schemaPath) {
    if (schemaPath.size() == 0) {
      return getDatabaseNamesImpl(isConnectionCached());
    } else if (schemaPath.size() == 1) {
      return getSchemaNamesImpl(schemaPath.get(0), isConnectionCached());
    } else {
      return Set.of();
    }
  }

  /**
   * Implementation of getSchemaNames that enables retrying if a cached connection fails
   *
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return Set of available schema names.
   */
  private Set<String> getDatabaseNamesImpl(boolean shouldRetry) {
    HashSet<String> databaseNames = new HashSet<>();
    try {
      DatabaseMetaData metaData = getDataBaseMetaData(shouldRetry);
      ResultSet schemaInfo = metaData.getCatalogs();
      while (schemaInfo.next()) {
        // Schema name is stored in column 1
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getCatalogs
        databaseNames.add(schemaInfo.getString(1));
      }
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get a list of DataBase names from the Snowflake account. Error message:"
                  + " %s",
              e);
      if (shouldRetry) {
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getDatabaseNamesImpl(false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
    return databaseNames;
  }

  /**
   * Implementation of getSchemaNames that enables retrying if a cached connection fails
   *
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return Set of available schema names.
   */
  private Set<String> getSchemaNamesImpl(String databaseName, boolean shouldRetry) {
    HashSet<String> schemaNames = new HashSet<>();
    try {
      DatabaseMetaData metaData = getDataBaseMetaData(shouldRetry);
      ResultSet schemaInfo = metaData.getSchemas(databaseName, null);
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
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getSchemaNamesImpl(databaseName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
    return schemaNames;
  }

  /**
   * Returns a schema found within the given parent path.
   *
   * @param schemaPath The parent schema path to check.
   * @param schemaName Name of the schema to fetch.
   * @return A schema object.
   */
  @Override
  public CatalogSchema getSchema(ImmutableList<String> schemaPath, String schemaName) {
    // Snowflake schema path can't be greater than length 1.
    if (schemaPath.size() > 1) {
      return null;
    }
    return new CatalogSchema(schemaName, schemaPath.size() + 1, schemaPath, this);
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
   * <p>(see <a href="https://docs.snowflake.com/en/sql-reference/name-resolution">...</a>)
   *
   * @param depth The depth at which to find the default.
   * @return List of default Schema to check when attempting to resolve a table in this catalog.
   */
  public List<String> getDefaultSchema(int depth) {
    if (depth == 0) {
      return List.of(currentDatabase);
    } else if (depth == 1) {
      return getDefaultSchemaImpl(isConnectionCached());
    } else {
      throw new RuntimeException(
          String.format(Locale.ROOT, "Unsupported depth for default Snowflake schema: %d.", depth));
    }
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
          // TODO: Support passing the default database directly.
          Pattern pattern = Pattern.compile("\"" + currentDatabase + ".([^\"]*)\"");
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
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getDefaultSchemaImpl(false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
    return defaultSchema;
  }

  /**
   * Return the number of levels at which a default schema may be found. Snowflake only has 2
   * levels, so this is always 2.
   *
   * @return The number of levels a default schema can be found.
   */
  @Override
  public int numDefaultSchemaLevels() {
    return 2;
  }

  /**
   * Returns a set of all function names with the given schema name.
   *
   * @param schemaPath The list of schemas to traverse before finding the function.
   * @return Set of function names.
   */
  public Set<String> getFunctionNames(ImmutableList<String> schemaPath) {
    if (schemaPath.size() != 2) {
      return Set.of();
    }
    return getFunctionNamesImpl(schemaPath.get(0), schemaPath.get(1), isConnectionCached());
  }

  // Adds supported builtin UDF/UDTF function names from Snowflake, like EXTERNAL_TABLE_FILES
  protected void addBuiltinFunctionNames(
      String databaseName, String schemaName, HashSet<String> functionNames) {
    if (schemaName.equals("INFORMATION_SCHEMA")) {
      functionNames.add(EXTERNAL_TABLE_FILES_NAME);
    }
  }

  // Adds supported builtin UDF/UDTF functions from Snowflake, like EXTERNAL_TABLE_FILES
  protected void addBuiltinFunctions(
      String databaseName, String schemaName, List<org.apache.calcite.schema.Function> functions) {
    if (schemaName.equals("INFORMATION_SCHEMA")) {
      functions.add(
          TableFunctionOperatorTable.makeExternalTableFiles(
              this, List.of(databaseName, schemaName)));
    }
  }

  /**
   * Implementation of getFunctionNames that enables retrying if a cached connection fails
   *
   * @param databaseName The name of the database to use to load the table.
   * @param schemaName The name of the schema to use to load the table.
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return Set of function names.
   */
  private Set<String> getFunctionNamesImpl(
      String databaseName, String schemaName, boolean shouldRetry) {
    try {
      DatabaseMetaData metaData = getDataBaseMetaData(shouldRetry);
      // Passing null for tableNamePattern should match all table names. Although
      // this is not in the public documentation. Passing null for types ensures we
      // allow all possible table types, including regular tables and views.
      ResultSet tableInfo = metaData.getFunctions(databaseName, schemaName, null);
      HashSet<String> tableNames = new HashSet<>();
      while (tableInfo.next()) {
        // Table name is stored in column 3
        // https://docs.oracle.com/javase/8/docs/api/java/sql/DatabaseMetaData.html#getFunctions
        tableNames.add(tableInfo.getString(3));
      }
      // Add builtin Snowflake UDF/UDFT names
      addBuiltinFunctionNames(databaseName, schemaName, tableNames);
      return tableNames;
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get functions names for Schema '%s'.'%s' from Snowflake account. Error"
                  + " message: %s",
              databaseName, schemaName, e);
      if (shouldRetry) {
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getFunctionNamesImpl(databaseName, schemaName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
  }

  /**
   * Returns all functions defined in this catalog with a given name and schema path.
   *
   * @param schemaPath The list of schemas to traverse before finding the function.
   * @param funcName Name of functions with a given name.
   * @return Collection of all functions with that name.
   */
  public Collection<org.apache.calcite.schema.Function> getFunctions(
      ImmutableList<String> schemaPath, String funcName) {
    if (schemaPath.size() != 2) {
      return List.of();
    }
    return getFunctionsImpl(schemaPath.get(0), schemaPath.get(1), funcName, isConnectionCached());
  }

  /**
   * Implementation of getFunctionNames that enables retrying if a cached connection fails. This
   * submits a show functions like call to Snowflake and uses the information to construct Calcite
   * function objects.
   *
   * @param databaseName The name of the database to use to load the table.
   * @param schemaName The name of the schema to use to load the table.
   * @param functionName The name of the function to load.
   * @param shouldRetry Should we retry the connection if we see an exception?
   * @return Collection of all functions with that name.
   */
  public Collection<org.apache.calcite.schema.Function> getFunctionsImpl(
      String databaseName, String schemaName, String functionName, boolean shouldRetry) {
    List<org.apache.calcite.schema.Function> functions = new ArrayList<>();
    try {
      String query =
          String.format(
              Locale.ROOT,
              "show user functions like '%s' in schema %s.%s",
              functionName,
              makeObjectNameQuoted(databaseName),
              makeObjectNameQuoted(schemaName));
      ResultSet results = executeSnowflakeQuery(query, 5);
      while (results.next()) {
        // See the return values here:
        // https://docs.snowflake.com/en/sql-reference/sql/show-functions
        java.sql.Timestamp createdOn = results.getTimestamp("created_on");
        // Function args + return value, not names
        String arguments = results.getString("arguments");
        // Is this a table function?
        Boolean isTable = snowflakeYesNoToBoolean(results.getString("is_table_function"));
        // Is this a secure function?
        Boolean isSecure = snowflakeYesNoToBoolean(results.getString("is_secure"));
        // Is this an external function?
        Boolean isExternal = snowflakeYesNoToBoolean(results.getString("is_external_function"));
        // What is this function's language?
        String language = results.getString("language");
        // Is this function memoizable?
        Boolean isMemoizable = snowflakeYesNoToBoolean(results.getString("is_memoizable"));
        // Generate a call to describe function to get the information needed to validate the
        // function.
        // This is necessary because the arguments column doesn't contain names.
        ImmutableList<String> functionPath =
            ImmutableList.of(databaseName, schemaName, functionName);
        DescribeFunctionOutput functionInfo = describeFunctionImpl(functionPath, arguments);
        String args = functionInfo.signature;
        int numOptional = functionInfo.numOptional;
        String returns = functionInfo.returns;
        String body = functionInfo.body;
        final org.apache.calcite.schema.Function function;
        if (isTable) {
          function =
              SnowflakeUserDefinedTableFunction.create(
                  functionPath,
                  args,
                  numOptional,
                  returns,
                  body,
                  isSecure,
                  isExternal,
                  language,
                  isMemoizable,
                  createdOn);
        } else {
          function =
              SnowflakeUserDefinedFunction.create(
                  functionPath,
                  args,
                  numOptional,
                  returns,
                  body,
                  isSecure,
                  isExternal,
                  language,
                  isMemoizable,
                  createdOn);
        }
        functions.add(function);
      }
      // Add builtin Snowflake UDFs/UDFTs
      addBuiltinFunctions(databaseName, schemaName, functions);
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Unable to get functions names for Schema '%s'.'%s' from Snowflake account. Error"
                  + " message: %s",
              databaseName, schemaName, e);
      if (shouldRetry) {
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        return getFunctionsImpl(databaseName, schemaName, functionName, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
    return functions;
  }

  /**
   * Fetch the Arguments, Return Type information, and body from a Snowflake UDF definition.
   *
   * @param functionPath The path to the function, including the name.
   * @param showFunctionArguments The arguments loaded from show functions.
   * @return The arguments, number of optional values, return type, and body information.
   */
  private DescribeFunctionOutput describeFunctionImpl(
      ImmutableList<String> functionPath, String showFunctionArguments) {
    kotlin.Pair<String, Integer> parsedInput =
        parseSnowflakeShowFunctionsArguments(
            showFunctionArguments, makeObjectNameQuoted(functionPath.get(2)));
    String describeFunctionInput = parsedInput.getFirst();
    int numOptional = parsedInput.getSecond();
    String describeQuery =
        String.format(
            Locale.ROOT,
            "describe function %s.%s.%s",
            makeObjectNameQuoted(functionPath.get(0)),
            makeObjectNameQuoted(functionPath.get(1)),
            describeFunctionInput);
    try {
      ResultSet describeResults = executeSnowflakeQuery(describeQuery, 5);
      // There may be several rows per languages. We determine the rows we want based
      // on the property.
      // Documentation: https://docs.snowflake.com/en/sql-reference/sql/desc-function.
      // Note: JDBC connections are 1-indexed
      String signature = null;
      String returns = null;
      String body = null;
      while (describeResults.next()) {
        String property = describeResults.getString("property").toUpperCase(Locale.ROOT);
        String value = describeResults.getString("value");
        if (property.equals("SIGNATURE")) {
          signature = value;
        } else if (property.equals("RETURNS")) {
          returns = value;
        } else if (property.equals("BODY")) {
          body = value;
        }
      }
      // Signature and returns must be provided. If a query is externally defined there
      // might not be a body.
      if (signature == null || returns == null) {
        String errorMsg =
            String.format("Unexpected results returned when processing query: %s", describeQuery);
        throw new RuntimeException(errorMsg);
      }
      return new DescribeFunctionOutput(signature, numOptional, returns, body);
    } catch (SQLException e) {
      String errorMsg =
          String.format(
              "Error encountered when running describe function query: %s. Error found: %s",
              describeQuery, e);
      throw new RuntimeException(errorMsg);
    }
  }

  private class DescribeFunctionOutput {

    private final String signature;
    private final int numOptional;
    private final String returns;
    private final @Nullable String body;

    DescribeFunctionOutput(
        String signature, int numOptional, String returns, @Nullable String body) {
      this.signature = signature;
      this.numOptional = numOptional;
      this.returns = returns;
      this.body = body;
    }
  }

  /**
   * Generate a Python connection string used to read from or write to Snowflake in Bodo's SQL
   * Python code.
   *
   * <p>TODO(jsternberg): This method is needed for the SnowflakeToBodoPhysicalConverter nodes, but
   * exposing this is a bad idea and this class likely needs to be refactored in a way that the
   * connection information can be passed around more easily.
   *
   * @param schemaPath The schema component to define the connection.
   * @return An Expr representing the connection string.
   */
  @Override
  public Expr generatePythonConnStr(ImmutableList<String> schemaPath) {
    if (schemaPath.size() != 2) {
      throw new BodoSQLCodegenException(
          "Internal Error: Snowflake requires exactly one database and one schema.");
    }
    String databaseName = schemaPath.get(0);
    String schemaName = schemaPath.get(1);
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
              "snowflake://%s%s@%s",
              URLEncoder.encode(username, "UTF-8"),
              password,
              URLEncoder.encode(accountName, "UTF-8")));
      if (!databaseName.isEmpty()) {
        // Append a schema if it exists
        connString.append(String.format("/%s", URLEncoder.encode(databaseName, "UTF-8")));
      }
      if (!schemaName.isEmpty()) {
        if (databaseName.isEmpty()) {
          throw new RuntimeException(
              "Internal Error: If database name is empty the schema name must also be empty");
        }
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

    return new Expr.StringLiteral(connString.toString());
  }

  /**
   * Generates the code necessary to produce a write expression from Snowflake.
   *
   * @param varName Name of the variable to write.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table. This should be of the form SCHEMA_NAME, TABLE_NAME.
   * @return The generated code to produce a write.
   */
  @Override
  public Expr generateAppendWriteCode(
      BodoCodeGenVisitor visitor, Variable varName, ImmutableList<String> tableName) {
    return generateWriteCode(
        visitor,
        varName,
        tableName,
        IfExistsBehavior.APPEND,
        SqlCreateTable.CreateTableType.DEFAULT,
        new SnowflakeCreateTableMetadata());
  }

  /**
   * Generates the code necessary to produce a write expression from Snowflake.
   *
   * @param varName Name of the variable to write.
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table. This should be of the form SCHEMA_NAME, TABLE_NAME.
   * @return The generated code to produce a write.
   */
  @Override
  public Expr generateWriteCode(
      BodoCodeGenVisitor visitor,
      Variable varName,
      ImmutableList<String> tableName,
      IfExistsBehavior ifExists,
      SqlCreateTable.CreateTableType tableType,
      SnowflakeCreateTableMetadata meta) {
    List<Expr> args = new ArrayList<>();
    List<kotlin.Pair<String, Expr>> kwargs = new ArrayList<>();
    args.add(new Expr.StringLiteral(tableName.get(2)));
    args.add(generatePythonConnStr(tableName.subList(0, 2)));
    kwargs.add(new kotlin.Pair<>("schema", new Expr.StringLiteral(tableName.get(1))));
    kwargs.add(
        new kotlin.Pair<>("if_exists", new Expr.StringLiteral(ifExists.asToSqlKwArgument())));
    kwargs.add(
        new kotlin.Pair<>(
            "_bodo_create_table_type", new Expr.StringLiteral(tableType.asStringKeyword())));
    // CTAS metadata not used for non-streaming writes
    kwargs.add(new kotlin.Pair<>("index", new Expr.BooleanLiteral(false)));
    return new Expr.Method(varName, "to_sql", args, kwargs);
  }

  /**
   * Check if using Iceberg write is possible (Iceberg volume is provided, table type is default,
   * and we can get Iceberg base URL from Snowflake volume)
   *
   * @param ifExists Behavior if table exists (e.g. replace).
   * @param createTableType type of table to create (e.g. transient).
   * @return Iceberg write flag
   */
  private boolean useIcebergWrite(
      IfExistsBehavior ifExists, SqlCreateTable.CreateTableType createTableType) {
    if (icebergVolume == null || !RelationalAlgebraGenerator.enableSnowflakeIcebergTables) {
      return false;
    }
    if ((ifExists == IfExistsBehavior.APPEND)
        || (createTableType != SqlCreateTable.CreateTableType.DEFAULT)) {
      String reason =
          (ifExists == IfExistsBehavior.APPEND) ? "table inserts." : "non-default tables.";
      VERBOSE_LEVEL_ONE_LOGGER.warning("Iceberg write not supported for " + reason);
      return false;
    }

    if (getIcebergBaseURL(icebergVolume) == null) {
      VERBOSE_LEVEL_ONE_LOGGER.warning("Cannot get Iceberg base URL");
      return false;
    }
    return true;
  }

  /**
   * Get object storage base path for writing Iceberg tables from Snowflake external volume (e.g.
   * s3://bucket_name/).
   *
   * @return storage path
   */
  public String getIcebergBaseURL(String icebergVolume_) {
    return getIcebergBaseURLFn.apply(icebergVolume_);
  }

  private final Function<String, String> getIcebergBaseURLFn =
      Memoizer.memoize(this::getIcebergBaseURLImpl);

  private String getIcebergBaseURLImpl(String icebergVolume_) {
    String storage_base_url = null;
    // TODO[BSE-2666]: Add robust error checking
    try {
      ResultSet describeResults =
          executeSnowflakeQuery("describe external volume " + icebergVolume_, 5);
      while (describeResults.next()) {
        String parent_property = describeResults.getString("parent_property");
        String property = describeResults.getString("property");
        String property_type = describeResults.getString("property_type");
        String property_value = describeResults.getString("property_value");
        if (parent_property.equals("STORAGE_LOCATIONS")
            && property.startsWith("STORAGE_LOCATION")
            && property_type.equals("String")) {
          JSONObject p_json = (JSONObject) new JSONParser().parse(property_value);
          storage_base_url = (String) p_json.get("STORAGE_BASE_URL");
          if (!storage_base_url.endsWith("/")) {
            storage_base_url += "/";
          }
        }
      }
    } catch (Exception e) {
      VERBOSE_LEVEL_TWO_LOGGER.warning(
          "Getting Snowflake Iceberg volume base URL failed: " + e.getMessage());
    }
    assert storage_base_url != null;
    if (storage_base_url.startsWith("azure://")) {
      try {
        URI uri = new URI(storage_base_url);
        // The path should be: /CONTAINER_NAME/EVERTHING_ELSE
        String[] paths = uri.getPath().split("/", 3);
        String host = uri.getHost().replace(".blob.core.windows.net", ".dfs.core.windows.net");
        storage_base_url = String.format(Locale.ROOT, "abfss://%s@%s/%s", paths[1], host, paths[2]);
      } catch (URISyntaxException e) {
        VERBOSE_LEVEL_TWO_LOGGER.warning(
            "Getting Snowflake Iceberg volume base URL failed: " + e.getMessage());
        return null;
      }
    }
    return storage_base_url;
  }

  /**
   * Generates the code necessary to produce a read expression from Snowflake.
   *
   * @param tableName The path of schema used to reach the table from the root that includes the
   *     table. This should be of the form SCHEMA_NAME, TABLE_NAME.
   * @param useStreaming Should we generate code to read the table as streaming (currently only
   *     supported for snowflake tables)
   * @param streamingOptions The options to use if streaming is enabled.
   * @return The generated code to produce a read.
   */
  @Override
  public Expr generateReadCode(
      ImmutableList<String> tableName, boolean useStreaming, StreamingOptions streamingOptions) {
    ArrayList<kotlin.Pair<String, Expr>> kwargs = new ArrayList<>();
    String streamingArg = "";
    if (useStreaming) {
      kwargs.add(
          new kotlin.Pair<>(
              "_bodo_chunksize", new Expr.IntegerLiteral(streamingOptions.getChunkSize())));
    }
    kwargs.add(new kotlin.Pair<>("_bodo_is_table_input", new Expr.BooleanLiteral(true)));
    kwargs.add(new kotlin.Pair<>("_bodo_read_as_table", new Expr.BooleanLiteral(true)));
    List<Expr> args =
        List.of(
            new Expr.StringLiteral(tableName.get(2)),
            generatePythonConnStr(tableName.subList(0, 2)));
    return new Expr.Call("pd.read_sql", args, kwargs);
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
        VERBOSE_LEVEL_THREE_LOGGER.warning(
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
    // For correctness, we need to verify that Snowflake can support this
    // query in its entirely (no BodoSQL specific features). To do this
    // we run an explain, which won't execute the query.
    executeExplainQuery(query);
    // We need to include the default schema to ensure the query works as intended
    List<String> schemaList = getDefaultSchema(1);
    String schemaName = "";
    if (schemaList.size() > 0) {
      schemaName = schemaList.get(0);
    }

    List<kotlin.Pair<String, Expr>> kwargs =
        List.of(new kotlin.Pair<>("_bodo_read_as_table", new Expr.BooleanLiteral(true)));
    return new Expr.Call(
        "pd.read_sql",
        List.of(
            new Expr.StringLiteral(query),
            generatePythonConnStr(ImmutableList.of(currentDatabase, schemaName))),
        kwargs);
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
  public int getWeekStart() {
    if (this.weekStart == null) {
      try {
        this.weekStart = getSnowflakeWeekStart();
      } catch (SQLException e) {
        throw new RuntimeException(e);
      }
    }

    return this.weekStart;
  }

  @Override
  public int getWeekOfYearPolicy() {
    if (this.weekOfYearPolicy == null) {
      try {
        this.weekOfYearPolicy = getSnowflakeWeekOfYearPolicy();
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
        VERBOSE_LEVEL_THREE_LOGGER.warning(errorMsg);
        closeConnections();
        executeExplainQueryImpl(query, false);
      } else {
        throw new RuntimeException(errorMsg);
      }
    }
  }

  @Override
  public boolean schemaDepthMayContainTables(int depth) {
    // Snowflake is always DATABASE.SCHEMA.TABLE, so the
    // schema must be two hops from the root.
    return depth == 2;
  }

  @Override
  public boolean schemaDepthMayContainSubSchemas(int depth) {
    // Snowflake is always DATABASE.SCHEMA.TABLE, so the
    // schema must be less than two hops from the root.
    return depth < 2;
  }

  @Override
  public boolean schemaDepthMayContainFunctions(int depth) {
    // Snowflake is always DATABASE.SCHEMA.FUNCTION, so the
    // schema must be two hops from the root.
    return depth == 2;
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
      String dotTableName =
          String.join(
              ".", tableName.stream().map(this::makeObjectNameQuoted).collect(Collectors.toList()));
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
   * Approximation of the error function ERF. Copied from <a
   * href="https://introcs.cs.princeton.edu/java/21function/ErrorFunction.java.html">...</a>
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
   * design description: <a
   * href="https://bodo.atlassian.net/wiki/spaces/B/pages/1468235780/Snowflake+Metadata+Handling#Testing-Proposed-Algorithm-on-Sample-Data">...</a>
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
            return new Pair<>(sampleSize, distinctCount);
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
    BigDecimal samplePercentage = BigDecimal.ONE;
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
              names.get(2),
              makeObjectNameQuoted(names.get(0)),
              makeObjectNameQuoted(names.get(1)),
              names.get(2));
      ResultSet paramInfo = executeSnowflakeQuery(query);
      if (paramInfo.next()) {
        // See the return types: https://docs.snowflake.com/en/sql-reference/sql/show-views.
        // Note: This is 1 indexed
        String queryDefinition = paramInfo.getString("text");
        // is_secure is not safe to inline.
        String unsafeToInline = paramInfo.getString("is_secure");
        String isMaterialized = paramInfo.getString("is_materialized");
        return new InlineViewMetadata(
            Boolean.parseBoolean(unsafeToInline),
            Boolean.parseBoolean(isMaterialized),
            queryDefinition);
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
   * @return True if we have read support, false otherwise.
   */
  public boolean canReadTable(List<String> names) {
    return canReadTableFn.apply(names);
  }

  private final Function<List<String>, Boolean> canReadTableFn =
      Memoizer.memoize(this::canReadTableImpl);

  /**
   * The actual implementation for canReadTable including calls to Snowflake. We add an extra layer
   * of indirection to ensure this function is cached.
   */
  private boolean canReadTableImpl(List<String> names) {
    String dotTableName =
        String.join(
            ".", names.stream().map(this::makeObjectNameQuoted).collect(Collectors.toList()));
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

  /** Determine if the provided Snowflake table is actually an Iceberg table inside Snowflake. */
  public boolean isIcebergTable(List<String> names) {
    return isIcebergTableFn.apply(names);
  }

  private final Function<List<String>, Boolean> isIcebergTableFn =
      Memoizer.memoize(this::isIcebergTableImpl);

  /**
   * The actual implementation for isIcebergTable including calls to Snowflake. We add an extra
   * layer of indirection to ensure this function is cached.
   */
  private boolean isIcebergTableImpl(List<String> names) {
    String qualifiedName =
        String.join(
            ".", names.stream().map(this::makeObjectNameQuoted).collect(Collectors.toList()));
    try {
      final String query = String.format(Locale.ROOT, "DESCRIBE ICEBERG TABLE %s;", qualifiedName);
      executeSnowflakeQuery(query, 5);

      // If the query errors just return and log.
      String successMsg = String.format("Can read table '%s' as an Iceberg table", qualifiedName);
      VERBOSE_LEVEL_TWO_LOGGER.info(successMsg);
      return true;
    } catch (SQLException e) {
      return false;
    }
  }

  public String genIcebergTableUUID() {
    return UUID.randomUUID().toString();
  }

  /**
   * Return the desired WriteTarget for a create table operation. We prioritize Iceberg writes if
   * Iceberg is enabled and, we have sufficient feature support.
   *
   * @param schema The schemaPath to the table.
   * @param tableName The name of the type that will be created.
   * @param createTableType The createTable type.
   * @param ifExistsBehavior The createTable behavior for if there is already a table defined.
   * @param columnNamesGlobal Global Variable holding the output column names.
   * @return The selected WriteTarget.
   */
  public WriteTarget getCreateTableWriteTarget(
      ImmutableList<String> schema,
      String tableName,
      SqlCreateTable.CreateTableType createTableType,
      IfExistsBehavior ifExistsBehavior,
      Variable columnNamesGlobal) {
    Expr snowflakeConnectionString = generatePythonConnStr(schema);
    if (useIcebergWrite(ifExistsBehavior, createTableType)) {
      return new SnowflakeIcebergWriteTarget(
          tableName,
          schema,
          ifExistsBehavior,
          columnNamesGlobal,
          new Expr.StringLiteral(getIcebergBaseURL(icebergVolume)),
          icebergVolume,
          snowflakeConnectionString,
          genIcebergTableUUID());
    } else {
      return new SnowflakeNativeWriteTarget(
          tableName, schema, ifExistsBehavior, columnNamesGlobal, generatePythonConnStr(schema));
    }
  }

  public @Nullable String getAccountName() {
    return accountName;
  }

  public SnowflakeJDBCExecutor getDdlExecutor() {
    return ddlExecutor;
  }

  /**
   * Interface for executing DDL commands on Snowflake via the JDBC connection. This is meant to
   * give a cleaner abstraction for DDL commands.
   */
  public class SnowflakeJDBCExecutor implements DDLExecutor {
    private SnowflakeJDBCExecutor() {}

    private @NotNull String generateSnowflakeObjectString(@NotNull ImmutableList<String> path) {
      return path.stream()
          .map(SnowflakeCatalog.this::makeObjectNameQuoted)
          .collect(Collectors.joining("."));
    }

    @Override
    public void createSchema(@NotNull ImmutableList<String> schemaPath)
        throws NamespaceAlreadyExistsException {
      String schemaFullStr = generateSnowflakeObjectString(schemaPath);
      String query = String.format(Locale.ROOT, "CREATE SCHEMA %s", schemaFullStr);
      try {
        executeSnowflakeQuery(query);
      } catch (SQLException e) {
        if (e.getMessage().contains("already exists.")) {
          throw new NamespaceAlreadyExistsException();
        }

        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to create Snowflake schema %s. Error: %s",
                schemaFullStr,
                e.getMessage()));
      }
    }

    @Override
    public void dropSchema(
        @NotNull ImmutableList<String> defaultSchemaPath, @NotNull String schemaName)
        throws NamespaceNotFoundException {
      assert (defaultSchemaPath.size() == 1 || defaultSchemaPath.size() == 2);
      ImmutableList<String> schemaPath = ImmutableList.of(defaultSchemaPath.get(0), schemaName);
      String schemaFullStr = generateSnowflakeObjectString(schemaPath);
      String query = String.format(Locale.ROOT, "DROP SCHEMA %s", schemaFullStr);

      try {
        executeSnowflakeQuery(query);
      } catch (SQLException e) {
        if (e.getMessage().contains("does not exist")) {
          throw new NamespaceNotFoundException();
        }

        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to drop Snowflake schema %s. Error: %s",
                schemaFullStr,
                e.getMessage()));
      }
    }

    @NotNull
    @Override
    public DDLExecutionResult dropTable(@NotNull ImmutableList<String> tablePath, boolean cascade) {
      String tableName = generateSnowflakeObjectString(tablePath);
      String query =
          String.format(
              Locale.ROOT, "DROP TABLE %s %s", tableName, cascade ? " CASCADE" : "RESTRICT");
      try {
        ResultSet output = executeSnowflakeQuery(query);
        List<List<String>> columnValues = new ArrayList<>();
        columnValues.add(new ArrayList<>());
        while (output.next()) {
          columnValues.get(0).add(output.getString(1));
        }
        return new DDLExecutionResult(List.of("STATUS"), columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to drop Snowflake table %s. Error: %s",
                tableName,
                e.getMessage()));
      }
    }

    /**
     * Renames a table in Snowflake using ALTER TABLE RENAME TO. Supports the IF EXISTS clause
     * through the ifExists parameter.
     *
     * @param tablePath The path of the table to rename.
     * @param renamePath The new path of the table.
     * @param ifExists Whether to use the IF EXISTS clause.
     * @return The result of the operation.
     * @throws RuntimeException If the operation fails.
     */
    @NotNull
    @Override
    public DDLExecutionResult renameTable(
        @NotNull ImmutableList<String> tablePath,
        @NotNull ImmutableList<String> renamePath,
        boolean ifExists) {
      // Convert paths into Snowflake object strings
      String tableName = generateSnowflakeObjectString(tablePath);
      String renameName = generateSnowflakeObjectString(renamePath);
      // SQL query
      String query =
          String.format(
              Locale.ROOT,
              "ALTER TABLE %s %s RENAME TO %s",
              ifExists ? "IF EXISTS" : "",
              tableName,
              renameName);
      try {
        // Execute query
        ResultSet output = executeSnowflakeQuery(query);
        // Build result from output
        List<List<String>> columnValues = new ArrayList<>();
        columnValues.add(new ArrayList<>());
        while (output.next()) {
          columnValues.get(0).add(output.getString(1));
        }
        return new DDLExecutionResult(List.of("STATUS"), columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to rename Snowflake table %s. Error: %s",
                tableName,
                e.getMessage()));
      }
    }

    /**
     * Renames a view in Snowflake using ALTER VIEW RENAME TO. Supports the IF EXISTS clause through
     * the ifExists parameter.
     *
     * @param viewPath The path of the view to rename.
     * @param renamePath The new path of the view.
     * @param ifExists Whether to use the IF EXISTS clause.
     * @return The result of the operation.
     * @throws RuntimeException If the operation fails.
     */
    @NotNull
    @Override
    public DDLExecutionResult renameView(
        @NotNull ImmutableList<String> viewPath,
        @NotNull ImmutableList<String> renamePath,
        boolean ifExists) {
      // Convert paths into Snowflake object strings
      String viewName = generateSnowflakeObjectString(viewPath);
      String renameName = generateSnowflakeObjectString(renamePath);
      // SQL query
      String query =
          String.format(
              Locale.ROOT,
              "ALTER VIEW %s %s RENAME TO %s",
              ifExists ? "IF EXISTS" : "",
              viewName,
              renameName);
      try {
        // Execute query
        ResultSet output = executeSnowflakeQuery(query);
        // Build result from output
        List<List<String>> columnValues = new ArrayList<>();
        columnValues.add(new ArrayList<>());
        while (output.next()) {
          columnValues.get(0).add(output.getString(1));
        }
        return new DDLExecutionResult(List.of("STATUS"), columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to rename Snowflake view %s. Error: %s",
                viewName,
                e.getMessage()));
      }
    }

    @NotNull
    @Override
    public DDLExecutionResult describeTable(
        @NotNull ImmutableList<String> tablePath, @NotNull RelDataTypeFactory typeFactory) {
      String tableName = generateSnowflakeObjectString(tablePath);
      String query = String.format(Locale.ROOT, "DESCRIBE TABLE %s", tableName);
      List<List<String>> columnValues = new ArrayList<>();
      List<String> columnNames =
          List.of(
              "NAME",
              "TYPE",
              "KIND",
              "NULL?",
              "DEFAULT",
              "PRIMARY_KEY",
              "UNIQUE_KEY",
              "CHECK",
              "EXPRESSION",
              "COMMENT",
              "POLICY NAME",
              "PRIVACY DOMAIN");
      for (int i = 0; i < columnNames.size(); i++) {
        columnValues.add(new ArrayList<>());
      }
      try {
        ResultSet output = executeSnowflakeQuery(query);
        while (output.next()) {
          // We keep the columns, but we do custom processing
          // for types to unify our output.
          int typeColIdx = output.findColumn("type"); // Second column as of 2024-05-06
          String columnType = output.getString(typeColIdx);
          // Note We don't care about nullability
          ColumnDataTypeInfo bodoColumnTypeInfo = snowflakeTypeNameToTypeInfo(columnType, true);
          RelDataType type = bodoColumnTypeInfo.convertToSqlType(typeFactory);
          columnValues.get(1).add(type.toString());
          columnValues.get(0).add(output.getString("name")); // First column as of 2024-05-06
          // Other columns are just copied over.
          for (int i = 2; i < columnNames.size(); i++) {
            columnValues.get(i).add(output.getString(i + 1));
          }
        }
        return new DDLExecutionResult(columnNames, columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to describe Snowflake table %s. Error: %s",
                tableName,
                e.getMessage()));
      }
    }

    /**
     * Emulates SHOW TERSE OBJECTS for a specified schema in Snowflake.
     *
     * @param schemaPath The schema path.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws RuntimeException on error The method executes a Snowflake query SHOW OBJECTS to show
     *     objects within a specified schema, and processes the result set. It builds up and returns
     *     a DDLExecutionResult object which contains the column names and their respective values.
     *     Note that the database name and schema name columns of the original query are combined
     *     due to compatibility with Iceberg, which uses namespaces instead of DB/schemas.
     */
    @NotNull
    @Override
    public DDLExecutionResult showObjects(@NotNull ImmutableList<String> schemaPath) {

      String schemaName = generateSnowflakeObjectString(schemaPath);
      String query = String.format(Locale.ROOT, "SHOW OBJECTS IN %s", schemaName);
      List<List<String>> columnValues = new ArrayList();
      List<String> columnNames = List.of("CREATED_ON", "NAME", "SCHEMA_NAME", "KIND");
      for (int i = 0; i < columnNames.size(); i++) {
        columnValues.add(new ArrayList<>());
      }
      try {
        ResultSet output = executeSnowflakeQuery(query);
        while (output.next()) {
          columnValues.get(0).add(output.getString("created_on"));
          columnValues.get(1).add(output.getString("name"));
          String db_schema =
              output.getString("database_name") + "." + output.getString("schema_name");
          columnValues.get(2).add(db_schema);
          columnValues.get(3).add(output.getString("kind"));
        }
        return new DDLExecutionResult(columnNames, columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to show objects in %s. Error: %s",
                schemaName,
                e.getMessage()));
      }
    }

    /**
     * Emulates SHOW TERSE TABLES for a specified schema in Snowflake.
     *
     * @param schemaPath The schema path.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws RuntimeException on error The method executes a Snowflake query SHOW TABLES to show
     *     tables within a specified schema, and processes the result set. It builds up and returns
     *     a DDLExecutionResult object which contains the column names and their respective values.
     *     Note that the database name and schema name columns of the original query are combined
     *     due to compatibility with Iceberg, which uses namespaces instead of DB/schemas.
     */
    @NotNull
    @Override
    public DDLExecutionResult showTables(@NotNull ImmutableList<String> schemaPath) {
      String schemaName = generateSnowflakeObjectString(schemaPath);
      String query = String.format(Locale.ROOT, "SHOW TABLES IN %s", schemaName);
      List<List<String>> columnValues = new ArrayList();
      List<String> columnNames = List.of("CREATED_ON", "NAME", "SCHEMA_NAME", "KIND");
      for (int i = 0; i < columnNames.size(); i++) {
        columnValues.add(new ArrayList<>());
      }
      try {
        ResultSet output = executeSnowflakeQuery(query);
        while (output.next()) {
          columnValues.get(0).add(output.getString("created_on"));
          columnValues.get(1).add(output.getString("name"));
          String db_schema =
              output.getString("database_name") + "." + output.getString("schema_name");
          columnValues.get(2).add(db_schema);
          columnValues.get(3).add(output.getString("kind"));
        }
        return new DDLExecutionResult(columnNames, columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT, "Unable to show tables in %s. Error: %s", schemaName, e.getMessage()));
      }
    }

    /**
     * Emulates SHOW TERSE SCHEMAS for a specified db in Snowflake.
     *
     * @param dbPath The db path.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws RuntimeException on error The method executes a Snowflake query SHOW SCHEMAS to show
     *     schemas within a specified schema, and processes the result set. It builds up and returns
     *     a DDLExecutionResult object which contains the column names and their respective values.
     *     Note that the database name column of the original query is renamed to SCHEMA_NAME due to
     *     compatibility with Iceberg, which uses namespaces instead of DB/schemas.
     */
    @NotNull
    @Override
    public DDLExecutionResult showSchemas(@NotNull ImmutableList<String> dbPath) {

      String dbName = generateSnowflakeObjectString(dbPath);
      String query = String.format(Locale.ROOT, "SHOW SCHEMAS IN %s", dbName);
      List<List<String>> columnValues = new ArrayList();
      List<String> columnNames = List.of("CREATED_ON", "NAME", "SCHEMA_NAME", "KIND");
      for (int i = 0; i < columnNames.size(); i++) {
        columnValues.add(new ArrayList<>());
      }
      try {
        ResultSet output = executeSnowflakeQuery(query);
        while (output.next()) {
          columnValues.get(0).add(output.getString("created_on"));
          String schemaName = output.getString("name");
          columnValues.get(1).add(schemaName);
          String dbSchemaName = output.getString("database_name");
          columnValues.get(2).add(dbSchemaName);
          // kind. This is not shown with `show schemas`
          // but with `show terse schemas` it appears and its value is null
          columnValues.get(3).add(null);
        }
        return new DDLExecutionResult(columnNames, columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT, "Unable to show schemas in %s. Error: %s", dbName, e.getMessage()));
      }
    }

    /**
     * Emulates SHOW TERSE VIEWS for a specified schema in Snowflake.
     *
     * @param schemaPath The schema path.
     * @return DDLExecutionResult containing columns CREATED_ON, NAME, SCHEMA_NAME, KIND
     * @throws RuntimeException on error
     *     <p>The method executes a Snowflake query SHOW TABLES to show tables within a specified
     *     schema, and processes the result set. It builds up and returns a DDLExecutionResult
     *     object which contains the column names and their respective values. Note that the
     *     database name and schema name columns of the original query are combined due to
     *     compatibility with Iceberg, which uses namespaces instead of DB/schemas.
     */
    @NotNull
    @Override
    public DDLExecutionResult showViews(@NotNull ImmutableList<String> schemaPath) {

      String schemaName = generateSnowflakeObjectString(schemaPath);
      // NOTE: Normal SHOW VIEWS does not provide a "kind" column.
      String query = String.format(Locale.ROOT, "SHOW TERSE VIEWS IN %s", schemaName);
      List<List<String>> columnValues = new ArrayList();
      List<String> columnNames = List.of("CREATED_ON", "NAME", "SCHEMA_NAME", "KIND");
      for (int i = 0; i < columnNames.size(); i++) {
        columnValues.add(new ArrayList<>());
      }
      try {
        ResultSet output = executeSnowflakeQuery(query);
        while (output.next()) {
          columnValues.get(0).add(output.getString("created_on"));
          columnValues.get(1).add(output.getString("name"));
          String db_schema =
              output.getString("database_name") + "." + output.getString("schema_name");
          columnValues.get(2).add(db_schema);
          columnValues.get(3).add(output.getString("kind"));
        }
        return new DDLExecutionResult(columnNames, columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT, "Unable to show views in %s. Error: %s", schemaName, e.getMessage()));
      }
    }

    @Override
    public void createOrReplaceView(
        @NotNull ImmutableList<String> viewPath,
        @NotNull SqlCreateView query,
        @NotNull CatalogSchema parentSchema,
        @NotNull RelDataType rowType)
        throws ViewAlreadyExistsException {
      String queryStr = query.toSqlString(BodoSnowflakeSqlDialect.DEFAULT).getSql();
      try {
        executeSnowflakeQuery(queryStr);
      } catch (SQLException e) {
        if (e.getMessage().contains("already exists.")) {
          throw new ViewAlreadyExistsException();
        }

        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to create Snowflake view from query %s. Error: %s",
                queryStr,
                e.getMessage()));
      }
    }

    /**
     * Describe a view with call to Snowflake API. Signals error whenever Snowflake signals an error
     * or when the view does not exist. The columns are up-to-date as of 2024-05-22
     *
     * @param viewPath The path to the view to describe.
     * @param typeFactory The type factory to use for creating the Bodo Type.
     */
    @NotNull
    @Override
    public DDLExecutionResult describeView(
        @NotNull ImmutableList<String> viewPath, @NotNull RelDataTypeFactory typeFactory) {
      String viewName = generateSnowflakeObjectString(viewPath);
      String query = String.format(Locale.ROOT, "DESCRIBE VIEW %s", viewName);
      List<List<String>> columnValues = new ArrayList<>();
      List<String> columnNames =
          List.of(
              "NAME",
              "TYPE",
              "KIND",
              "NULL?",
              "DEFAULT",
              "PRIMARY_KEY",
              "UNIQUE_KEY",
              "CHECK",
              "EXPRESSION",
              "COMMENT",
              "POLICY NAME",
              "PRIVACY DOMAIN");
      for (int i = 0; i < columnNames.size(); i++) {
        columnValues.add(new ArrayList<>());
      }
      try {
        ResultSet output = executeSnowflakeQuery(query);
        while (output.next()) {
          // We keep the columns, but we do custom processing
          // for types to unify our output.
          int typeColIdx = output.findColumn("type"); // Second column as of 2024-05-06
          String columnType = output.getString(typeColIdx);
          // Note We don't care about nullability
          ColumnDataTypeInfo bodoColumnTypeInfo = snowflakeTypeNameToTypeInfo(columnType, true);
          RelDataType type = bodoColumnTypeInfo.convertToSqlType(typeFactory);
          columnValues.get(1).add(type.toString());
          columnValues.get(0).add(output.getString("name")); // First column as of 2024-05-06
          // Other columns are just copied over.
          for (int i = 2; i < columnNames.size(); i++) {
            columnValues.get(i).add(output.getString(i + 1));
          }
        }
        return new DDLExecutionResult(columnNames, columnValues);
      } catch (SQLException e) {
        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to describe Snowflake view %s. Error: %s",
                viewName,
                e.getMessage()));
      }
    }

    /*
     * Drop a view with call to Snowflake API. Signals error whenever Snowflake signals an error.
     * The "if_exists" flag is handled in DDLExecutor.
     *
     * @param viewPath Global Variable holding the output column names.
     */
    @Override
    public void dropView(@NotNull ImmutableList<String> viewPath)
        throws NamespaceNotFoundException {
      String viewName = generateSnowflakeObjectString(viewPath);
      String query = String.format(Locale.ROOT, "DROP VIEW %s", viewName);
      try {
        executeSnowflakeQuery(query);
      } catch (SQLException e) {
        if (e.getMessage().contains("does not exist")) {
          throw new NamespaceNotFoundException();
        }

        throw new RuntimeException(
            String.format(
                Locale.ROOT,
                "Unable to drop Snowflake view from query %s. Error: %s",
                viewName,
                e.getMessage()));
      }
    }

    @Override
    @NotNull
    public DDLExecutionResult setProperty(
        @NotNull ImmutableList<String> tablePath,
        @NotNull SqlNodeList propertyList,
        @NotNull SqlNodeList valueList,
        boolean ifExists) {
      throw new RuntimeException("SET PROPERTY/TAG is not supported for Snowflake.");
    }

    @Override
    @NotNull
    public DDLExecutionResult unsetProperty(
        @NotNull ImmutableList<String> tablePath,
        @NotNull SqlNodeList propertyList,
        boolean ifExists,
        boolean ifPropertyExists) {
      throw new RuntimeException("UNSET PROPERTY/TAG is not supported for Snowflake.");
    }
  }

  // HELPER FUNCTIONS
  private String makeObjectNameQuoted(String name) {
    return "\"" + name + "\"";
  }
}
