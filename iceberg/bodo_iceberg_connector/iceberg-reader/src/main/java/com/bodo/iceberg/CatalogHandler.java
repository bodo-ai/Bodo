package com.bodo.iceberg;

import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.hadoop.conf.Configuration;
import org.apache.hc.core5.http.NameValuePair;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.hadoop.HadoopTables;
import org.apache.iceberg.hive.HiveCatalog;
import org.apache.iceberg.nessie.NessieCatalog;

/** connects to Iceberg catalog using connection string and loads the Iceberg table */
public class CatalogHandler {

  private String connStr;
  private String dbName;
  private String tableName;

  public CatalogHandler(String connStr, String dbName, String tableName) {
    this.connStr = connStr;
    this.dbName = dbName;
    this.tableName = tableName;
  }

  public Table loadTable() throws URISyntaxException {

    // extract parameters from URI
    URIBuilder uri_builder = new URIBuilder(this.connStr);

    // get catalog type and parameters
    Map<String, String> params =
        uri_builder.getQueryParams().stream()
            .collect(Collectors.toMap(NameValuePair::getName, NameValuePair::getValue));
    String catalogType = params.getOrDefault("type", "hive");
    params.remove("type");

    // catalog URI (without parameters)
    String uri_str = uri_builder.removeQuery().build().toString();

    // local file system case (no scheme or file://)
    if (uri_builder.getScheme() == null || uri_builder.getScheme().equals("file")) {
      HadoopTables hadoopTables = new HadoopTables();
      // Set CWD for opening the metadata files later.
      System.setProperty("user.dir", uri_str);
      return hadoopTables.load(uri_str + "/" + this.dbName + "/" + this.tableName);
    }

    // create general catalog parameters
    Configuration conf = new Configuration();
    Map<String, String> properties = new HashMap<>();
    properties.put(CatalogProperties.URI, uri_str);
    Namespace db_namespace = Namespace.of(this.dbName);
    TableIdentifier name = TableIdentifier.of(db_namespace, this.tableName);
    params.forEach((k, v) -> properties.put(k, v));

    // Hive metastore case
    if (catalogType.equals("hive")) {
      HiveCatalog catalog = new HiveCatalog();
      catalog.setConf(conf);
      // TODO[BE-2833]: explore using CachingCatalog
      catalog.initialize("hive_catalog", properties);
      return catalog.loadTable(name);
    }

    // Nessie metastore case
    if (catalogType.equals("nessie")) {
      NessieCatalog catalog = new NessieCatalog();
      catalog.setConf(conf);
      // Nessie connector requires warehouse to be set, even though it may not be necessary
      // https://github.com/apache/iceberg/blob/82db4fa9ef6d958f5b0f9742c16e30d28338a584/nessie/src/main/java/org/apache/iceberg/nessie/NessieCatalog.java#L134
      properties.put(CatalogProperties.WAREHOUSE_LOCATION, params.getOrDefault("warehouse", ""));
      catalog.initialize("nessie_catalog", properties);
      return catalog.loadTable(name);
    }

    throw new UnsupportedOperationException(
        String.format("Iceberg catalog type '%s' not supported yet.", catalogType));
  }
}
