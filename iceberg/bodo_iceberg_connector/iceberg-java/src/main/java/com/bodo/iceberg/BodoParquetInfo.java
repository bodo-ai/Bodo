package com.bodo.iceberg;

import java.util.LinkedList;
import java.util.List;
import org.apache.iceberg.FileScanTask;

public class BodoParquetInfo {
  /** Class that holds the minimal Parquet info needed by Bodo */
  private final String filepath;

  private final long rowCount;
  private final long schemaID;
  private final List<String> deleteFiles;

  BodoParquetInfo(FileScanTask task, long schemaID, List<String> deleteFiles) {
    this.filepath = task.file().path().toString();
    this.rowCount = task.file().recordCount();
    this.schemaID = schemaID;
    this.deleteFiles = deleteFiles;
  }

  public String getFilepath() {
    return filepath;
  }

  public long getRowCount() {
    return rowCount;
  }

  public long getSchemaID() {
    return schemaID;
  }

  public List<String> getDeleteFiles() {
    if (this.deleteFiles == null) {
      return new LinkedList<>();
    }
    return deleteFiles;
  }

  public boolean hasDeleteFile() {
    return deleteFiles != null && !deleteFiles.isEmpty();
  }

  public String toString() {
    return String.format(
        "(Filepath: %s, Row Count: %d, Schema ID: %d)",
        getFilepath(), getRowCount(), getSchemaID());
  }
}
