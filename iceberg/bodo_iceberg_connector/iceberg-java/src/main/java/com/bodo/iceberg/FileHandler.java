package com.bodo.iceberg;

import com.bodo.iceberg.filters.FilterExpr;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.iceberg.*;
import org.apache.iceberg.avro.Avro;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.types.Types;
import org.json.JSONObject;

public class FileHandler {
  static Types.NestedField STATUS =
      Types.NestedField.required(0, "status", Types.IntegerType.get());
  static Schema SKELETON_SCHEMA = new Schema(STATUS);

  private static Map<String, Long> genFileToSchemaID(Table table) {
    var res = new HashMap<String, Long>();

    var manifestFiles = table.currentSnapshot().allManifests(table.io());
    for (var file : manifestFiles) {
      var reader = ManifestFiles.read(file, table.io());

      var metadata = Avro.read(reader.file()).project(SKELETON_SCHEMA).build().getMetadata();
      JSONObject obj = new JSONObject(metadata.get("schema"));
      long schemaID = obj.getLong("schema-id");

      for (var line : reader) {
        res.put(line.path().toString(), schemaID);
      }
    }

    return res;
  }

  /** Returns a list of parquet files that construct the given Iceberg table. */
  public static Triple<
          List<BodoParquetInfo>, Map<Integer, org.apache.arrow.vector.types.pojo.Schema>, Long>
      getParquetInfo(Table table, FilterExpr filters) throws IOException {
    var t0 = System.nanoTime();
    var fileToSchemaID = genFileToSchemaID(table);
    var fileToSchemaIDNs = System.nanoTime() - t0;

    Expression filter = filters.toExpr();
    TableScan scan = table.newScan().filter(filter);
    List<BodoParquetInfo> parquetPaths = new ArrayList<>();

    try (CloseableIterable<FileScanTask> fileTasks = scan.planFiles()) {
      for (FileScanTask fileTask : fileTasks) {
        // Set to null by default to save memory while we don't support deletes.
        List<String> deletes = null;
        // Check for any delete files.
        List<DeleteFile> deleteFiles = fileTask.deletes();
        if (!deleteFiles.isEmpty()) {
          deletes = new LinkedList<>();
          for (DeleteFile deleteFile : deleteFiles) {
            deletes.add(deleteFile.path().toString());
          }
        }

        long schemaID = fileToSchemaID.get(fileTask.file().path().toString());
        parquetPaths.add(new BodoParquetInfo(fileTask, schemaID, deletes));
      }
    }

    var arrowSchemas =
        table.schemas().entrySet().stream()
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey, (entry) -> BodoArrowSchemaUtil.convert(entry.getValue())));
    return new Triple<>(parquetPaths, arrowSchemas, fileToSchemaIDNs);
  }
}
