package com.bodo.iceberg;

import static com.bodo.iceberg.FilterExpr.filtersToExpr;

import com.bodo.iceberg.catalog.CatalogCreator;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.*;
import org.apache.iceberg.*;
import org.apache.iceberg.catalog.Catalog;
import org.apache.iceberg.catalog.Namespace;
import org.apache.iceberg.catalog.TableIdentifier;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;

public class BodoIcebergHandler {
  /**
   * Java Class used to map Bodo's required read and write operations to a corresponding Iceberg
   * table. This is meant to provide 1 instance per Table and Bodo is responsible for closing it.
   */
  private final Catalog catalog;

  private final TableIdentifier id;

  public BodoIcebergHandler(String connStr, String catalogType, String dbName, String tableName)
      throws URISyntaxException {
    this.catalog = CatalogCreator.create(connStr, catalogType);
    Namespace dbNamespace = Namespace.of(dbName);
    id = TableIdentifier.of(dbNamespace, tableName);
  }

  /**
   * Get Information About Table
   *
   * @return Information about Table needed by Bodo
   */
  public TableInfo getTableInfo(boolean error) {
    if (!catalog.tableExists(id) && !error) {
      return null;
    }

    // Note that repeated calls to loadTable are cheap due to CachingCatalog
    return new TableInfo(catalog.loadTable(id));
  }

  /** Returns a list of parquet files that construct the given Iceberg table. */
  public List<BodoParquetInfo> getParquetInfo(LinkedList<Object> filters) throws IOException {
    Expression filter = filtersToExpr(filters);
    TableScan scan = catalog.loadTable(id).newScan().filter(filter);
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

        parquetPaths.add(
            new BodoParquetInfo(
                fileTask.file().path().toString(), fileTask.start(), fileTask.length(), deletes));
      }
    }

    return parquetPaths;
  }

  /** Create a new table in the DB. */
  public void createOrReplaceTable(
      List<String> fileNames,
      List<Long> fileSizes,
      List<Long> fileRecords,
      Schema schema,
      boolean replace) {
    Map<String, String> properties = new HashMap<>();
    properties.put(TableProperties.FORMAT_VERSION, "2");

    // TODO: Support passing in new partition spec and sort order as well
    Transaction txn;
    if (replace)
      txn =
          catalog.newReplaceTableTransaction(
              id, schema, PartitionSpec.unpartitioned(), properties, false);
    else
      txn =
          catalog.newCreateTableTransaction(id, schema, PartitionSpec.unpartitioned(), properties);

    List<DataFileInfo> fileInfos = DataFileInfo.fromLists(fileNames, fileSizes, fileRecords);
    this.addData(txn.newAppend(), PartitionSpec.unpartitioned(), SortOrder.unsorted(), fileInfos);
    txn.commitTransaction();
  }

  /** Appends data into preexisting table */
  public void appendTable(
      List<String> fileNames, List<Long> fileSizes, List<Long> fileRecords, int schemaID) {
    List<DataFileInfo> fileInfos = DataFileInfo.fromLists(fileNames, fileSizes, fileRecords);

    Table table = catalog.loadTable(id);
    if (table.schema().schemaId() != schemaID)
      throw new IllegalStateException("Iceberg Table has updated its schema");

    this.addData(table.newAppend(), table.spec(), table.sortOrder(), fileInfos);
  }

  /** Insert data files into the table */
  public void addData(
      AppendFiles append,
      PartitionSpec currSpec,
      SortOrder currOrder,
      List<DataFileInfo> fileInfos) {
    boolean isPartitionedPaths = currSpec.isPartitioned();

    for (DataFileInfo info : fileInfos) {
      DataFile dataFile = info.toDataFile(currSpec, currOrder, isPartitionedPaths);
      append.appendFile(dataFile);
    }

    append.commit();
  }
}
