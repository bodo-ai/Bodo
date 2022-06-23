package com.bodo.iceberg;

import java.util.LinkedList;
import java.util.List;

public class BodoParquetInfo {
  /** Class that holds the minimal Parquet info needed by Bodo */
  private final String filepath;

  private final long start;
  private final long length;
  private final List<String> deleteFiles;

  BodoParquetInfo(String filepath, long start, long length, List<String> deleteFiles) {
    this.filepath = filepath;
    this.start = start;
    this.length = length;
    this.deleteFiles = deleteFiles;
  }

  public String getFilepath() {
    return filepath;
  }

  public long getStart() {
    return start;
  }

  public long getLength() {
    return length;
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
        "(Filepath:%s, Start Offset:%d,  Length:%d)", getFilepath(), getStart(), getLength());
  }
}
