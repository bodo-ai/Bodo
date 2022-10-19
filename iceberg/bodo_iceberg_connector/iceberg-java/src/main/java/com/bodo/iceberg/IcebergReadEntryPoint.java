package com.bodo.iceberg;

import java.net.URISyntaxException;
import java.util.HashMap;
import java.util.Map;
import org.apache.hadoop.conf.Configuration;
import org.apache.iceberg.CatalogProperties;
import org.apache.iceberg.Table;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.hive.HiveCatalog;

public class IcebergReadEntryPoint {
  public static void main(String[] args) throws URISyntaxException {
    //    GatewayServer gatewayServer = new GatewayServer(new IcebergReadEntryPoint());
    //    gatewayServer.start();
    System.out.println("Here?");

    Map<String, String> properties = new HashMap<>();
    HiveCatalog catalog = new HiveCatalog();
    System.out.println("Create Catalog");

    catalog.setConf(new Configuration());
    properties.put(CatalogProperties.WAREHOUSE_LOCATION, "s3a://bodo-iceberg-test2/");
    properties.put(CatalogProperties.URI, "thrift://localhost:9083");
    catalog.initialize("hive_catalog", properties);
    System.out.println("Initialize Catalog");

    Namespace nm = Namespace.of("db4");
    TableIdentifier id = TableIdentifier.of(nm, "table");
    System.out.println("Working up to here");
    Table table = catalog.loadTable(id);
    System.out.println(table.schema());
    System.out.println(table.location());
  }
}
