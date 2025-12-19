package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.rest.RESTCatalog;

public class RESTBuilder {
  public static Catalog create(Configuration conf, Map<String, String> properties) {
    RESTCatalog catalog = new RESTCatalog();
    catalog.setConf(conf);
    catalog.initialize("REST_catalog", properties);
    return catalog;
  }
}
