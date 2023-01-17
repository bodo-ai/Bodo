package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.hive.HiveCatalog;

public class ThriftBuilder {
  public static Catalog create(Configuration conf, Map<String, String> properties) {
    HiveCatalog catalog = new HiveCatalog();
    catalog.setConf(conf);

    // Can only read from catalogs stored in a S3 warehouse via S3FileIO
    if (properties.containsKey(CatalogProperties.WAREHOUSE_LOCATION)) {
      String warehouseLoc = properties.get(CatalogProperties.WAREHOUSE_LOCATION);
      if (warehouseLoc.startsWith("s3://") || warehouseLoc.startsWith("s3a://"))
        properties.put(CatalogProperties.FILE_IO_IMPL, "org.apache.iceberg.aws.s3.S3FileIO");
    }

    catalog.initialize("hive_catalog", properties);
    return catalog;
  }
}
