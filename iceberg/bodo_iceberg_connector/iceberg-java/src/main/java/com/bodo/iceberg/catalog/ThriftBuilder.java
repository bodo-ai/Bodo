package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.hive.HiveCatalog;

public class ThriftBuilder {
  public static Catalog create(Configuration conf, Map<String, String> properties) {
    HiveCatalog catalog = new HiveCatalog();
    catalog.setConf(conf);

    catalog.initialize("hive_catalog", properties);
    return catalog;
  }
}
