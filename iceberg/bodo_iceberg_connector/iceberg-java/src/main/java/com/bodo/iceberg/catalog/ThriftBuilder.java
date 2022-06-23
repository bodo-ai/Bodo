package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.hc.core5.net.URIBuilder;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.hive.HiveCatalog;

public class ThriftBuilder {

  public static boolean isConnStr(final URIBuilder uriBuilder, final String catalogType) {
    return catalogType == null
        ? (uriBuilder.getScheme() != null && uriBuilder.getScheme().equals("thrift"))
        : catalogType.equals("hive");
  }

  public static Catalog create(Configuration conf, Map<String, String> properties) {
    HiveCatalog catalog = new HiveCatalog();
    catalog.setConf(conf);

    catalog.initialize("hive_catalog", properties);
    return catalog;
  }
}
