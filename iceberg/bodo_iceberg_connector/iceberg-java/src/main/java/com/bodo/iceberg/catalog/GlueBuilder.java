package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.aws.glue.GlueCatalog;
import org.apache.iceberg.catalog.Catalog;

public class GlueBuilder {
  public static Catalog create(Configuration conf, Map<String, String> properties) {
    // By this point, CatalogProperties.URI == "glue" should always be true
    // GlueCatalog may be using this parameter now or in the future
    // So its safer to clear it out instead of leaving a junk value
    properties.remove(CatalogProperties.URI);

    // Glue catalog requires the WAREHOUSE_LOCATION property to be set
    // to non-empty even though it may not be necessary
    if (properties.containsKey(CatalogProperties.WAREHOUSE_LOCATION)) {
      // Can only read from catalogs stored in a S3 warehouse via S3FileIO
      String warehouseLoc = properties.get(CatalogProperties.WAREHOUSE_LOCATION);
      if (warehouseLoc.startsWith("s3://") || warehouseLoc.startsWith("s3a://"))
        properties.put(CatalogProperties.FILE_IO_IMPL, "org.apache.iceberg.aws.s3.S3FileIO");

    } else {
      properties.put(CatalogProperties.WAREHOUSE_LOCATION, "s3");
    }

    GlueCatalog catalog = new GlueCatalog();
    catalog.setConf(conf);

    // As of Iceberg-AWS 1.0.0, GlueCatalog always use S3FileIO by default
    // since it is expected (and probably only allowed) that Glue tables are stored in S3
    catalog.initialize("glue_catalog", properties);
    return catalog;
  }
}
