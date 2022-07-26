package com.bodo.iceberg;

import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.*;
import org.apache.iceberg.aws.glue.GlueCatalog;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;

public class IcebergReadEntryPoint {
  IcebergReadEntryPoint() {}

  public static void main(String[] args) throws URISyntaxException {
    //    GatewayServer gatewayServer = new GatewayServer(new IcebergReadEntryPoint());
    //    gatewayServer.start();

    Map<String, String> properties = new HashMap<>();
    GlueCatalog catalog = new GlueCatalog();

    catalog.setConf(new Configuration());
    properties.put(CatalogProperties.WAREHOUSE_LOCATION, "s3");
    catalog.initialize("glue_catalog", properties);

    Namespace nm = Namespace.of("db1");
    TableIdentifier id = TableIdentifier.of(nm, "table1");
    //
    //    Map<String, String> table_props = new HashMap<>();
    //    table_props.put(TableProperties.FORMAT_VERSION, "2");
    //
    //    Schema schema =
    //        new Schema(
    //            Types.NestedField.required(1, "A", Types.IntegerType.get()),
    //            Types.NestedField.required(2, "B", Types.IntegerType.get()));

    Table table = catalog.loadTable(id);
    System.out.println(table.schema());
    System.out.println(table.location());
  }
}
