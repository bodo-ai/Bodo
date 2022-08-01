package com.bodosql.calcite.application;

import py4j.GatewayServer;

public class BodoSQLpy4jEntryPoint {

  public BodoSQLpy4jEntryPoint() {}

  public static void main(String[] args) {
    GatewayServer gatewayServer = new GatewayServer(new BodoSQLpy4jEntryPoint());
    gatewayServer.start();
  }
}
