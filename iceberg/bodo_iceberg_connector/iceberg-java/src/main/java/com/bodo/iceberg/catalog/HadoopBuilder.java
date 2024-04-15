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

    // Hadoop Catalogs use both a FileIO and a Hadoop Filesystem object to access Iceberg tables
    // FileIO lacks operations needed for all Iceberg functionality (namely `ls`, `rename`, etc.)
    // Hadoop Filesystem is used for these situations, but Hadoop expects S3 paths to start with
    // `s3a://`
    // Thus, we still need to convert `s3://` to `s3a://`
    if (warehouseLoc.startsWith("s3")) {
      warehouseLoc = warehouseLoc.replaceFirst("s3://", "s3a://");
    }

    if (warehouseLoc.startsWith("s3a")) {
      // We still want to use S3FileIO if we can though
      properties.put(CatalogProperties.FILE_IO_IMPL, "org.apache.iceberg.aws.s3.S3FileIO");
    }
    properties.put(CatalogProperties.WAREHOUSE_LOCATION, warehouseLoc);

    catalog.initialize("hadoop_catalog", properties);
    return catalog;
  }
}
