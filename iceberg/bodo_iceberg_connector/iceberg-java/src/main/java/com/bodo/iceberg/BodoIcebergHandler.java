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

    // Snowflake uses dot separated strings for DB and schema names
    // Iceberg uses Namespaces with multiple levels to represent this
    Namespace dbNamespace = Namespace.of(dbName.split("\\."));
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
  public void createOrReplaceTable(String fileInfoJson, Schema schema, boolean replace) {
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

    List<DataFileInfo> fileInfos = DataFileInfo.fromJson(fileInfoJson);
    this.addData(txn.newAppend(), PartitionSpec.unpartitioned(), SortOrder.unsorted(), fileInfos);
    txn.commitTransaction();
  }

  /** Appends Rows into a Pre-existing Table */
  public void appendTable(String fileInfoJson, int schemaID) {
    // Remove the Table instance associated with `id` from the cache
    // So that the next load gets the current instance from the underlying catalog
    catalog.invalidateTable(id);
    Table table = catalog.loadTable(id);
    if (table.schema().schemaId() != schemaID)
      throw new IllegalStateException("Iceberg Table has updated its schema");

    List<DataFileInfo> fileInfos = DataFileInfo.fromJson(fileInfoJson);
    this.addData(table.newAppend(), table.spec(), table.sortOrder(), fileInfos);
  }

  /** Merge Rows into Pre-existing Table by Copy-on-Write Rules */
  public void mergeCOWTable(List<String> oldFileNames, String newFileInfoJson, long snapshotID) {

    // Remove the Table instance associated with `id` from the cache
    // So that the next load gets the current instance from the underlying catalog
    catalog.invalidateTable(id);
    Table table = catalog.loadTable(id);
    if (table.currentSnapshot().snapshotId() != snapshotID)
      throw new IllegalStateException(
          "Iceberg Table has been updated since reading. Can not complete MERGE INTO");

    List<DataFileInfo> fileInfos = DataFileInfo.fromJson(newFileInfoJson);

    this.overwriteData(
        table.newTransaction(), table.spec(), table.sortOrder(), oldFileNames, fileInfos);
  }

  /** Insert data files into the table */
  public void addData(
      AppendFiles action, PartitionSpec spec, SortOrder order, List<DataFileInfo> fileInfos) {
    boolean isPartitionedPaths = spec.isPartitioned();

    for (DataFileInfo info : fileInfos) {
      DataFile dataFile = info.toDataFile(spec, order, isPartitionedPaths);
      action.appendFile(dataFile);
    }

    action.commit();
  }

  /** Overwrite Data Files with New Modified Versions */
  public void overwriteData(
      Transaction transaction,
      PartitionSpec spec,
      SortOrder order,
      List<String> oldFileNames,
      List<DataFileInfo> newFiles) {
    boolean isPartitionedPaths = spec.isPartitioned();

    // Data Files should be uniquely identified by path only. Other values should not matter
    DeleteFiles delAction = transaction.newDelete();
    for (String oldFileName : oldFileNames) delAction.deleteFile(oldFileName);
    delAction.commit();

    AppendFiles action = transaction.newAppend();
    for (DataFileInfo newFile : newFiles)
      action.appendFile(newFile.toDataFile(spec, order, isPartitionedPaths));
    action.commit();

    transaction.commitTransaction();
  }

  /**
   * Fetch the snapshot id for a table. Returns -1 for a newly created table without any snapshots
   */
  public long getSnapshotId() {
    var snapshot = catalog.loadTable(id).currentSnapshot();
    // When the table has just been created
    if (snapshot == null) return -1;
    return snapshot.snapshotId();
  }
}
