package com.bodo.iceberg.catalog;

import java.net.URISyntaxException;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.hadoop.conf.Configuration;
import org.apache.hc.core5.http.NameValuePair;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.CachingCatalog;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;

/** Iceberg Catalog Connector and Communicator */
public class CatalogHandler {

  private final TableIdentifier id;
  private final Catalog catalog;

  public CatalogHandler(String connStr, String dbName, String tableName) throws URISyntaxException {
    Namespace dbNamespace = Namespace.of(dbName);
    this.id = TableIdentifier.of(dbNamespace, tableName);

    // Extract Parameters from URI
    URIBuilder uriBuilder = new URIBuilder(connStr);
    Map<String, String> params =
        uriBuilder.getQueryParams().stream()
            .collect(Collectors.toMap(NameValuePair::getName, NameValuePair::getValue));
    String catalogType = params.remove("type");

    // Catalog URI (without parameters)
    String uriStr = uriBuilder.removeQuery().build().toString();
    params.put(CatalogProperties.URI, uriStr);

    // Create Configuration
    Configuration conf = new Configuration();

    // Create Catalog
    Catalog catalog;

    if (NessieBuilder.isConnStr(catalogType)) catalog = NessieBuilder.create(conf, params);
    else if (ThriftBuilder.isConnStr(uriBuilder, catalogType))
      catalog = ThriftBuilder.create(conf, params);
    else if (GlueBuilder.isConnStr(uriBuilder, catalogType))
      catalog = GlueBuilder.create(conf, params);
    else if (HadoopBuilder.isConnStr(uriBuilder, catalogType))
      catalog =
          HadoopBuilder.create(
              uriBuilder.removeQuery().setScheme("").build().toString(), conf, params);
    else {
      if (catalogType == null)
        throw new UnsupportedOperationException(
            String.format(
                "Cannot detect Iceberg catalog type from connection string '%s'", connStr));
      else
        throw new UnsupportedOperationException(
            String.format("Iceberg catalog type '%s' not supported yet.", catalogType));
    }

    // CachingCatalog is a simple map between table names and their `Table` object
    // Does not modify how the Table object works in any way
    // Can be invalidated by CREATE / REPLACE ops or a timer (if time passed in as arg)
    // Benefits: Potentially some speed up in collecting repeated data like schema
    // Downsides: Does not refresh if another program modifies the Catalog
    this.catalog = CachingCatalog.wrap(catalog);
  }

  public Table loadTable() {
    return catalog.loadTable(id);
  }
}
