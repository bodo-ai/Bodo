package com.bodo.iceberg;

import java.net.URISyntaxException;

public class IcebergReadEntryPoint {

  public IcebergReadEntryPoint() {}

  public static BodoIcebergReader getBodoIcebergReader(
      String warehouse_loc, String db_name, String tableName) throws URISyntaxException {
    return new BodoIcebergReader(warehouse_loc, db_name, tableName);
  }

  public static void main(String[] args) throws URISyntaxException {
    BodoIcebergReader reader =
        getBodoIcebergReader(
            "https://nessie.dremio.cloud/v1/projects/50824e14-fd95-434c-a0cb-cc988e57969f?type=nessie&authentication.type=BEARER&authentication.token=...",
            "",
            "arc_test3");
    //    BodoIcebergReader reader = getBodoIcebergReader("thrift://localhost:9083", "db",
    // "table2");
    //    BodoIcebergReader reader =
    //     getBodoIcebergReader("file:///Users/ehsan/dev/bodo/iceberg/test-dataset-creation/",
    //         "iceberg_db",
    //             "simple_numeric_table");
    System.out.println(reader.getIcebergSchema().toString());
    System.out.println(reader.getParquetInfo(null).toString());
  }
}
