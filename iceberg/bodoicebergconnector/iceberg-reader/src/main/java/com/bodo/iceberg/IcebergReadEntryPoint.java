package com.bodo.iceberg;

import py4j.GatewayServer;

public class IcebergReadEntryPoint {

  public IcebergReadEntryPoint() {}

  public BodoIcebergReader getBodoIcebergReader(
      String warehouse_loc, String db_name, String tableName) {
    return new BodoIcebergReader(warehouse_loc, db_name, tableName);
  }

  public static void main(String[] args) {
    if (args.length != 1) {
      throw new RuntimeException("Java Iceberg API requires exactly 1 argument, the port number");
    }
    String portString = args[0];
    int port = 0;
    try {
      port = Integer.valueOf(portString);
    } catch (NumberFormatException e) {
      throw new RuntimeException("Java Iceberg API port number");
    }
    GatewayServer gatewayServer = new GatewayServer(new IcebergReadEntryPoint(), port);
    gatewayServer.start();
    System.out.println("Gateway Server Started");
  }
}
