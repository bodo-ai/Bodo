package com.bodo.iceberg;

import java.util.ArrayList;
import java.util.List;
import org.apache.iceberg.*;
import org.apache.iceberg.hadoop.HadoopTables;

public class IcebergRead {

  public IcebergRead() {}

  public List<String> getParquetPaths(String warehouse_loc, String db_name, String tableName) {

    HadoopTables hadoopTables = new HadoopTables();

    System.setProperty("user.dir", warehouse_loc);
    Table table = hadoopTables.load(warehouse_loc + "/" + db_name + "/" + tableName);

    System.out.println("Table Name: " + table.name());

    TableScan scan = table.newScan();
    Schema schema = scan.schema();

    System.out.println("Schema: " + schema.toString());

    //        Iterable<CombinedScanTask> tasks = scan.planTasks();

    List<String> parquetPaths = new ArrayList<>();

    Iterable<FileScanTask> files = scan.planFiles();

    files.forEach(
        (file) -> {
          parquetPaths.add(file.file().path().toString());
          //            System.out.println(file.toString());
          //            System.out.println(file.length());
          //            System.out.println(file.start());
        });

    return parquetPaths;
  }
}
