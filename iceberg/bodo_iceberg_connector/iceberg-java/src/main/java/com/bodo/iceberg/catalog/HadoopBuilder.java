package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.hadoop.HadoopCatalog;

public class HadoopBuilder {
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
