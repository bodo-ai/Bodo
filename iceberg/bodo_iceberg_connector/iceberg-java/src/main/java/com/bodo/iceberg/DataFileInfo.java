package com.bodo.iceberg;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.iceberg.*;

public class DataFileInfo {

  private final String path;
  // file size in bytes
  private final long size;
  private final long recordCount;

  DataFileInfo(String path, long size, long recordCount) {
    this.path = path;
    this.size = size;
    this.recordCount = recordCount;
  }

  /** Construct a list of DataFileInfo instances from lists of its components */
  public static List<DataFileInfo> fromLists(
      List<String> paths, List<Long> sizes, List<Long> counts) {
    if (paths.size() != sizes.size() || sizes.size() != counts.size())
      throw new IllegalArgumentException("List Arguments Must All be the Same Size");

    return IntStream.range(0, paths.size())
        .mapToObj(i -> new DataFileInfo(paths.get(i), sizes.get(i), counts.get(i)))
        .collect(Collectors.toList());
  }

  public String getPath() {
    return path;
  }

  public long getSize() {
    return size;
  }

  public long getRecordCount() {
    return recordCount;
  }

  public String toString() {
    return String.format(
        "(Path: %s, Size: %d, Record Count: %d)", getPath(), getSize(), getRecordCount());
  }

  /** Construct an Iceberg DataFile from DataFileInfo instance */
  public DataFile toDataFile(PartitionSpec spec) {
    return DataFiles.builder(spec)
        .withPath(getPath())
        .withFileSizeInBytes(getSize())
        .withRecordCount(getRecordCount())
        .withSortOrder(null)
        .build();
  }
}
