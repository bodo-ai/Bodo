package com.bodo.iceberg.catalog;

import java.net.URISyntaxException;
import java.util.Map;
import java.util.stream.Collectors;
import org.apache.hadoop.conf.Configuration;
import org.apache.hc.core5.http.NameValuePair;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.CachingCatalog;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;

/** Iceberg Catalog Connector and Communicator */
public class CatalogCreator {
  public static Catalog create(String connStr, String catalogType) throws URISyntaxException {
    // Extract Parameters from URI
    // TODO: Just get from Python
    URIBuilder uriBuilder = new URIBuilder(connStr);
    Map<String, String> params =
        uriBuilder.getQueryParams().stream()
            .collect(Collectors.toMap(NameValuePair::getName, NameValuePair::getValue));
    params.remove("type");

    // Create Configuration
    // Additional parameters like Iceberg-specific ones should be ignored
    // Since the conf is reused by multiple objects, like Hive and Hadoop ones
    // TODO: Spark does something similar, but I believe they do some filtering. What is it?
    Configuration conf = new Configuration();
    for (Map.Entry<String, String> entry : params.entrySet()) {
      conf.set(entry.getKey(), entry.getValue());
    }

    // Catalog URI (without parameters)
    String uriStr = uriBuilder.removeQuery().build().toString();
    params.put(CatalogProperties.URI, uriStr);

    // Create Catalog
    final Catalog catalog;

    switch (catalogType) {
      case "nessie":
        catalog = NessieBuilder.create(conf, params);
        break;
      case "hive":
        catalog = ThriftBuilder.create(conf, params);
        break;
      case "glue":
        catalog = GlueBuilder.create(conf, params);
        break;
      case "hadoop":
        catalog =
            HadoopBuilder.create(
                uriBuilder.removeQuery().setScheme("").build().toString(), conf, params);
        break;
      case "hadoop-s3":
        catalog = HadoopBuilder.create(uriBuilder.removeQuery().build().toString(), conf, params);
        break;
      case "snowflake":
        catalog = SnowflakeBuilder.create(conf, params);
        break;
      default:
        throw new UnsupportedOperationException("Should never occur. Captured in Python");
    }

    // CachingCatalog is a simple map between table names and their `Table` object
    // Does not modify how the Table object works in any way
    // Can be invalidated by CREATE / REPLACE ops or a timer (if time passed in as
    // arg)
    // Benefits: Potentially some speed up in collecting repeated metadata like
    // schema
    // Downsides: Does not refresh if another program modifies the Catalog
    return CachingCatalog.wrap(catalog);
  }
}
