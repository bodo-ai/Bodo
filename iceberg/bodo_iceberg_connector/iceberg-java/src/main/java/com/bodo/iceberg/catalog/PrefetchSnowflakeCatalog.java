package com.bodo.iceberg.catalog;

import static java.lang.Math.min;

import java.net.URISyntaxException;
import java.sql.*;
import java.util.*;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import net.snowflake.client.core.QueryStatus;
import net.snowflake.client.jdbc.SnowflakeConnection;
import net.snowflake.client.jdbc.SnowflakePreparedStatement;
import net.snowflake.client.jdbc.SnowflakeResultSet;
import org.apache.hc.core5.http.NameValuePair;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.BaseTable;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.CatalogUtil;
import org.apache.iceberg.StaticTableOperations;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.snowflake.SnowflakeCatalog;
import org.json.JSONObject;

public class PrefetchSnowflakeCatalog extends SnowflakeCatalog {
  private static final int MAX_RETRIES = 5;
  private static final int BACKOFF_MS = 100;
  private static final int MAX_BACKOFF_MS = 2000;

  private final String connStr;
  private Connection conn;
  private final Map<String, String> tablePathToQueryID;
  private boolean loadTableVerbose = false;

  // Copies from SnowflakeCatalog
  private Object conf;
  private Map<String, String> catalogProperties;

  public PrefetchSnowflakeCatalog(String connStr) {
    this.connStr = connStr;
    this.conn = null;
    this.tablePathToQueryID = new HashMap<>();
  }

  public PrefetchSnowflakeCatalog(String connStr, List<String> tablePaths, int verboseLevel)
      throws SQLException, URISyntaxException {
    this(connStr);
    // Should print verbose info during loadTable
    this.loadTableVerbose = (verboseLevel > 0);
    this.prefetchMetadataPaths(tablePaths);
  }

  private Connection getConnection() throws SQLException, URISyntaxException {
    var props = new Properties();
    var uriBuilder = new URIBuilder(connStr);
    Map<String, String> params =
        uriBuilder.getQueryParams().stream()
            .collect(Collectors.toMap(NameValuePair::getName, NameValuePair::getValue));
    props.putAll(params);
    uriBuilder = uriBuilder.removeQuery();
    uriBuilder.setScheme("jdbc:snowflake");

    // DriverManager need manual retries
    // https://stackoverflow.com/questions/6110975/connection-retry-using-jdbc
    int numRetries = 0;
    do {
      Logger logger = Logger.getLogger("net.snowflake.client.jdbc");
      logger.setLevel(Level.OFF);
      DriverManager.setLogWriter(null);
      this.conn = DriverManager.getConnection(uriBuilder.build().toString(), props);
      if (this.conn == null) {
        int sleepMultiplier = 2 << numRetries;
        int sleepTime = min(sleepMultiplier * BACKOFF_MS, MAX_BACKOFF_MS);
        try {
          Thread.sleep(sleepTime);
        } catch (InterruptedException e) {
          throw new RuntimeException(
              "Backoff between Snowflake connection attempt interupted...", e);
        }
      }
      numRetries += 1;
    } while (this.conn == null && numRetries < MAX_RETRIES);

    return this.conn;
  }

  /**
   * Start the Snowflake query to get the metadata paths for a list of Snowflake-managed Iceberg
   * tables.
   *
   * <p>The query is not expected to finish until the table is needed for initialization / read.
   *
   * @param tablePaths A list of tablePaths to prefetch
   */
  public void prefetchMetadataPaths(List<String> tablePaths)
      throws SQLException, URISyntaxException {
    // Start prefetch
    var conn = this.getConnection();

    String sql_command = "SELECT SYSTEM$GET_ICEBERG_TABLE_INFORMATION(?) AS METADATA";
    PreparedStatement stmt = conn.prepareStatement(sql_command);

    for (var tablePath : tablePaths) {
      stmt.setString(1, tablePath);
      try (ResultSet resultSet =
          stmt.unwrap(SnowflakePreparedStatement.class).executeAsyncQuery()) {
        this.tablePathToQueryID.put(
            tablePath, resultSet.unwrap(SnowflakeResultSet.class).getQueryID());
      }
    }
  }

  /**
   * Helper function to wait for the Snowflake prefetch query to finish and extract the metadata
   * path from the output JSON
   *
   * @param queryID Snowflake query identifier of query to wait for
   * @return Metadata location from query result
   */
  private String waitPrefetch(String queryID)
      throws InterruptedException, SQLException, URISyntaxException {
    var conn = this.getConnection();
    try (ResultSet resultSet = conn.unwrap(SnowflakeConnection.class).createResultSet(queryID)) {
      // Wait for query to finish (not running)
      QueryStatus queryStatus = resultSet.unwrap(SnowflakeResultSet.class).getStatus();
      while (queryStatus == QueryStatus.RUNNING || queryStatus == QueryStatus.RESUMING_WAREHOUSE) {
        Thread.sleep(BACKOFF_MS);
        queryStatus = resultSet.unwrap(SnowflakeResultSet.class).getStatus();
      }

      // Based on
      // https://docs.snowflake.com/en/developer-guide/jdbc/jdbc-using#examples-of-asynchronous-queries
      if (queryStatus == QueryStatus.FAILED_WITH_ERROR) {
        throw new RuntimeException(
            String.format(
                "SnowflakePrefetch::waitPrefetch - Prefetch query failed: (%d) %s",
                queryStatus.getErrorCode(), queryStatus.getErrorMessage()));
      } else if (queryStatus != QueryStatus.SUCCESS) {
        throw new RuntimeException(
            String.format(
                "SnowflakePrefetch::waitPrefetch - Prefetch query errored unexpectedly: %s",
                queryStatus));
      }

      if (!resultSet.next()) {
        throw new RuntimeException(
            "SnowflakePrefetch::waitPrefetch - Prefetch query didn't return any results");
      }

      return (new JSONObject(resultSet.getString(1))).getString("metadataLocation");
    }
  }

  /**
   * Convert a TableIdentifier to a full qualified table path, where every path part is wrapped in
   * quotes and delimited using `.` (periods)
   */
  private static String idToQualifiedName(TableIdentifier id) {
    List<String> components = new ArrayList<String>(Arrays.asList(id.namespace().levels()));
    components.add(id.name());
    // Construct tablePath by wrapping components in quotes
    return components.stream()
        .map(x -> String.format("\"%s\"", x))
        .collect(Collectors.joining("."));
  }

  /**
   * Load and construct the Snowflake-managed Iceberg table using the prefetched data.
   *
   * <p>This will wait for the metadata path query to finish, load the metadata from storage, and
   * construct a read-only table object from it.
   *
   * @param id The identifier of the table to load
   * @return A read-only Iceberg table object representing loaded table
   */
  public Table loadTable(TableIdentifier id) {
    String tablePath = idToQualifiedName(id);

    // Redirect if not prefetched
    if (!this.tablePathToQueryID.containsKey(tablePath)) {
      if (this.loadTableVerbose) {
        System.err.println(
            "BODO VERBOSE: PrefetchSnowflakeCatalog::loadTable: Loading `"
                + tablePath
                + "` regularly");
      }
      return super.loadTable(id);
    }

    if (this.loadTableVerbose) {
      System.err.println(
          "BODO VERBOSE: PrefetchSnowflakeCatalog::loadTable: Loading `"
              + tablePath
              + "` from prefetch");
    }

    // Get and wait for the query ID to get metadata location
    String metadataLoc;
    try {
      metadataLoc = waitPrefetch(this.tablePathToQueryID.get(tablePath));
    } catch (InterruptedException | SQLException | URISyntaxException e) {
      throw new RuntimeException(e);
    }

    // Construct table from just location
    String fileIOImpl = "org.apache.iceberg.io.ResolvingFileIO";
    if (this.catalogProperties.containsKey(CatalogProperties.FILE_IO_IMPL)) {
      fileIOImpl = this.catalogProperties.get(CatalogProperties.FILE_IO_IMPL);
    }

    FileIO fileIO = CatalogUtil.loadFileIO(fileIOImpl, this.properties(), this.conf);
    var op = new StaticTableOperations(metadataLoc, fileIO);
    var table = new BaseTable(op, id.name());
    table.refresh();
    return table;
  }

  // Overrides to have access to configuration properties
  public void setConf(Object conf) {
    super.setConf(conf);
    this.conf = conf;
  }

  public void initialize(String name, Map<String, String> properties) {
    super.initialize(name, properties);
    this.catalogProperties = properties;
  }
}
