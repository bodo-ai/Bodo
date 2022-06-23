package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.hadoop.HadoopCatalog;

public class HadoopBuilder {

  public static boolean isConnStr(final URIBuilder uriBuilder, final String catalogType) {
    return catalogType == null
        ? (uriBuilder.getScheme() == null || uriBuilder.getScheme().equals("file"))
        : catalogType.equals("hadoop");
  }

  public static Catalog create(
      String warehouseLoc, Configuration conf, Map<String, String> properties) {
    HadoopCatalog catalog = new HadoopCatalog();

    catalog.setConf(conf);
    properties.remove(CatalogProperties.URI);
    properties.put(CatalogProperties.WAREHOUSE_LOCATION, warehouseLoc);

    catalog.initialize("hadoop_catalog", properties);
    return catalog;
  }
}
