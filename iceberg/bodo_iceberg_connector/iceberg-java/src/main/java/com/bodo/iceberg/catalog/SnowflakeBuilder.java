package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.jdbc.JdbcCatalog;

public class SnowflakeBuilder {
  public static void initialize(
      PrefetchSnowflakeCatalog catalog, Configuration conf, Map<String, String> properties) {
    catalog.setConf(conf);
    // Snowflake uses a JDBC connection, so we need to convert the URI to a JDBC URI
    properties.put(
        CatalogProperties.URI,
        properties.get(CatalogProperties.URI).replace("snowflake://", "jdbc:snowflake://"));
    // Snowflake uses a JDBC connection, so we need to add the JDBC properties
    properties.put(JdbcCatalog.PROPERTY_PREFIX + "user", properties.get("user"));
    properties.put(JdbcCatalog.PROPERTY_PREFIX + "password", properties.get("password"));
    if (properties.containsKey("role")) {
      properties.put(JdbcCatalog.PROPERTY_PREFIX + "role", properties.get("role"));
    }
    // Provide our own File IO that converts wasb:// to abfs:/
    String fileIOImpl = "org.apache.iceberg.io.BodoResolvingFileIO";
    properties.put(CatalogProperties.FILE_IO_IMPL, fileIOImpl);
    catalog.initialize("snowflake_catalog", properties);
  }

  public static Catalog create(Configuration conf, Map<String, String> properties) {
    PrefetchSnowflakeCatalog catalog =
        new PrefetchSnowflakeCatalog(properties.get(CatalogProperties.URI));
    SnowflakeBuilder.initialize(catalog, conf, properties);
    return catalog;
  }
}
