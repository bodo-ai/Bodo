package com.bodo.iceberg.catalog;

import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.nessie.NessieCatalog;

public class NessieBuilder {
  public static Catalog create(Configuration conf, Map<String, String> properties) {
    // Nessie catalog requires the WAREHOUSE_LOCATION property to be set
    // (even if it's an empty string) even though it may not be necessary
    // https://github.com/apache/iceberg/blob/82db4fa9ef6d958f5b0f9742c16e30d28338a584/nessie/src/main/java/org/apache/iceberg/nessie/NessieCatalog.java#L134
    if (properties.containsKey("warehouse")) {
      String warehouseLoc = properties.remove("warehouse");
      // See HadoopBuilder for explaination
      if (warehouseLoc.startsWith("s3://"))
        warehouseLoc = warehouseLoc.replaceFirst("s3://", "s3a://");
      properties.put(CatalogProperties.WAREHOUSE_LOCATION, warehouseLoc);
    } else {
      properties.put(CatalogProperties.WAREHOUSE_LOCATION, "");
    }

    NessieCatalog catalog = new NessieCatalog();
    catalog.setConf(conf);

    catalog.initialize("nessie_catalog", properties);
    return catalog;
  }
}
