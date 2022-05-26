package com.bodo.iceberg;

public class IcebergTester {

  public IcebergTester() {}

  public BodoIcebergReader getBodoIcebergReader(
      String warehouse_loc, String db_name, String tableName) {
    return new BodoIcebergReader(warehouse_loc, db_name, tableName);
  }

  public static void main(String[] args) {
    // Replace me with your path
    String WAREHOUSE_LOC =
        "/Users/nicholasriasanovsky/Documents/bodo/bodo-iceberg-expl/iceberg-expl";
    String DB_NAME = "iceberg_db";
    String TABLE_NAME = "test_table";
    BodoIcebergReader iceberg_reader = new BodoIcebergReader(WAREHOUSE_LOC, DB_NAME, TABLE_NAME);
    System.out.println(iceberg_reader.getIcebergSchema());
    System.out.println(iceberg_reader.getParquetInfo(null));
  }
}
