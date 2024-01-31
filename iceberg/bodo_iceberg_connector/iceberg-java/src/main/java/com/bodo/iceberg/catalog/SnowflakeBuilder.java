package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.jdbc.JdbcCatalog;
import org.apache.iceberg.snowflake.SnowflakeCatalog;

class SnowflakeBuilder {
  public static Catalog create(Configuration conf, Map<String, String> properties) {
    SnowflakeCatalog catalog = new SnowflakeCatalog();
    catalog.setConf(conf);
    // Snowflake uses a JDBC connection, so we need to convert the URI to a JDBC URI
    properties.put(
        CatalogProperties.URI,
        properties.get(CatalogProperties.URI).replace("snowflake://", "jdbc:snowflake://"));
    // Snowflake uses a JDBC connection, so we need to add the JDBC properties
    properties.put(JdbcCatalog.PROPERTY_PREFIX + "user", properties.get("user"));
    properties.put(JdbcCatalog.PROPERTY_PREFIX + "password", properties.get("password"));

    catalog.initialize("snowflake_catalog", properties);
    return catalog;
  }
}
