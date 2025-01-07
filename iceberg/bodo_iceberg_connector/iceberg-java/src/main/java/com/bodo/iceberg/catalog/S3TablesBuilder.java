package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.iceberg.catalog.Catalog;
import software.amazon.s3tables.iceberg.S3TablesCatalog;

public class S3TablesBuilder {
  public static Catalog create(String connStr) {
    S3TablesCatalog catalog = new S3TablesCatalog();
    catalog.initialize("S3Tables_catalog", Map.of("warehouse", connStr));
    return catalog;
  }
}
