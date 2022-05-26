package com.bodo.iceberg;

public class IcebergReadEntryPoint {

  public IcebergReadEntryPoint() {}

  public static BodoIcebergReader getBodoIcebergReader(
      String warehouse_loc, String db_name, String tableName) {
    return new BodoIcebergReader(warehouse_loc, db_name, tableName);
  }

  public static void main(String[] args) {
    BodoIcebergReader reader = getBodoIcebergReader("thrift://localhost:9083", "db", "table2");
    //    BodoIcebergReader reader =
    // getBodoIcebergReader("/Users/ehsan/dev/bodo/iceberg/test-dataset-creation/", "iceberg_db",
    // "simple_numeric_table");
    System.out.println(reader.getIcebergSchema().toString());
  }
}
