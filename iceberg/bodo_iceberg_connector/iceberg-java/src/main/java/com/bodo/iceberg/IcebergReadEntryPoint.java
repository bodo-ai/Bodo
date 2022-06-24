package com.bodo.iceberg;

import java.net.URISyntaxException;
import py4j.GatewayServer;

public class IcebergReadEntryPoint {

  public IcebergReadEntryPoint() {}

  public static BodoIcebergReader getBodoIcebergReader(
      String warehouse_loc, String db_name, String tableName) throws URISyntaxException {
    return new BodoIcebergReader(warehouse_loc, db_name, tableName);
  }

  public static void main(String[] args) {
    GatewayServer gatewayServer = new GatewayServer(new IcebergReadEntryPoint());
    gatewayServer.start();
  }
}
