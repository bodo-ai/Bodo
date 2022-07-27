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

    // Hadoop Catalog only accepts Hadoop-compatible directory paths to read from
    // S3 is technically not a hadoop-compatible file system
    // S3A is variant of S3's file system that is hadoop-compatible
    // Thus, if we want to read a S3-based directory catalog, we need to ensure that
    // S3A is being used as the filesystem.
    // TODO: If we switch to S3FileIO instead of HadoopFileIO, this is not necessary
    // TODO: In Iceberg 0.14.0, this conversion is apparently done automatically in HadoopFileIO
    if (warehouseLoc.startsWith("s3://"))
      warehouseLoc = warehouseLoc.replaceFirst("s3://", "s3a://");
    properties.put(CatalogProperties.WAREHOUSE_LOCATION, warehouseLoc);

    catalog.initialize("hadoop_catalog", properties);
    return catalog;
  }
}
