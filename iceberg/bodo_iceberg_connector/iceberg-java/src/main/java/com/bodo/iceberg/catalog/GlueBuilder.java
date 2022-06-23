package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.aws.glue.GlueCatalog;
import org.apache.iceberg.catalog.Catalog;

public class GlueBuilder {
  public static boolean isConnStr(final URIBuilder uriBuilder, final String catalogType) {
    return catalogType == null ? (uriBuilder.getPath().equals("glue")) : catalogType.equals("glue");
  }

  public static Catalog create(Configuration conf, Map<String, String> properties) {
    // Glue catalog requires the WAREHOUSE_LOCATION property to be set
    // to non-empty even though it may not be necessary
    if (properties.containsKey("warehouse")) {
      properties.put(CatalogProperties.WAREHOUSE_LOCATION, properties.remove("warehouse"));
    } else {
      properties.put(CatalogProperties.WAREHOUSE_LOCATION, "s3");
    }

    GlueCatalog catalog = new GlueCatalog();
    catalog.setConf(conf);

    catalog.initialize("glue_catalog", properties);
    return catalog;
  }
}
