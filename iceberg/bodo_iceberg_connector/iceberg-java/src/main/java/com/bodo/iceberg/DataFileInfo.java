package com.bodo.iceberg;

import java.math.BigDecimal;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.avro.util.Utf8;
import org.apache.iceberg.*;
import org.apache.iceberg.expressions.Literal;
import org.apache.iceberg.relocated.com.google.common.base.Preconditions;
import org.apache.iceberg.types.Type;
import org.apache.iceberg.types.Types;

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
  public DataFile toDataFile(PartitionSpec spec, SortOrder order, boolean isPartitionedPath) {
    DataFiles.Builder builder =
        DataFiles.builder(spec)
            .withPath(getPath())
            .withFileSizeInBytes(getSize())
            .withRecordCount(getRecordCount())
            .withSortOrder(order);

    if (isPartitionedPath) {
      // Parse the given path to get the substring of partition folder names.
      // NOTE that this assumes that the files were written using the Hive
      // partitioning scheme (`/<key>=<value>/`).
      // E.g.: ./iceberg_db/simple_numeric_table/data/B_trunc=1/A_bucket=2/data1.pq ->
      // B_trunc=1/A_bucket=2
      // Note that method of collecting partition data only works with hive-style partitioning
      // TODO: Better by passing in a prefix path to toDataFile (i.e. local relative or full
      // S3 path)
      String partitionFolders =
          getPath().substring(getPath().lastIndexOf("data/") + 5, getPath().lastIndexOf("/"));

      String[] partitions = partitionFolders.split("/", -1);
      Preconditions.checkArgument(partitions.length == spec.fields().size());
      PartitionData data = new PartitionData(spec);

      // Copied from @see org.apache.iceberg.PartitionData
      // https://github.com/apache/iceberg/blob/dbb8a404f6632a55acb36e949f0e7b84b643cede/core/src/main/java/org/apache/iceberg/PartitionData.java
      for (int i = 0; i < partitions.length; i += 1) {
        PartitionField field = spec.fields().get(i);
        String[] parts = partitions[i].split("=", 2);
        Preconditions.checkArgument(
            parts.length == 2 && parts[0] != null && field.name().equals(parts[0]),
            "Invalid partition: %s",
            partitions[i]);

        data.set(i, fromPartitionString(data.getTransform(i), data.getType(i), parts[1]));
      }

      builder.withPartition(data);
    }

    return builder.build();
  }

  /**
   * Parses Partition String Value to Actual Value Based on Type and Transformation. Copied
   * from @see org.apache.iceberg.PartitionData and modified for special null partition handling. <a
   * href="https://github.com/apache/iceberg/blob/dbb8a404f6632a55acb36e949f0e7b84b643cede/core/src/main/java/org/apache/iceberg/PartitionData.java">Github
   * Permalink</a>
   */
  static Object fromPartitionString(String transformName, Type type, String asString) {
    if (asString == null || "__HIVE_DEFAULT_PARTITION__".equals(asString)) return null;

    // Bodo change: Added check for null partition from original code
    if (asString.equals("null")) return null;

    switch (type.typeId()) {
      case BOOLEAN:
        return Boolean.valueOf(asString);
      case INTEGER:
        return Integer.valueOf(asString);
      case LONG:
      case TIMESTAMP:
        return Long.valueOf(asString);
      case FLOAT:
        return Float.valueOf(asString);
      case DOUBLE:
        return Double.valueOf(asString);
      case STRING:
        return asString;
      case UUID:
        return UUID.fromString(asString);
      case FIXED:
        Types.FixedType fixed = (Types.FixedType) type;
        return Arrays.copyOf(asString.getBytes(StandardCharsets.UTF_8), fixed.length());
      case BINARY:
        return asString.getBytes(StandardCharsets.UTF_8);
      case DECIMAL:
        return new BigDecimal(asString);
      case DATE:
        return Literal.of(asString).to(Types.DateType.get()).value();
      default:
        throw new UnsupportedOperationException(
            "Unsupported type for fromPartitionString: " + type);
    }
  }

  /**
   * Partition Data Class to Pass Partition Value of a File to Icebergs API. Primarily Copied
   * from @see org.apache.iceberg.PartitionData class which is private, so unusable. <a
   * href="https://github.com/apache/iceberg/blob/dbb8a404f6632a55acb36e949f0e7b84b643cede/core/src/main/java/org/apache/iceberg/PartitionData.java">Github
   * Permalink</a>
   */
  static class PartitionData implements StructLike {
    private final Types.StructType partitionType;
    private final PartitionSpec spec;
    private final int size;
    Object[] data;

    PartitionData(PartitionSpec spec) {
      this.spec = spec;
      this.partitionType = spec.partitionType();
      this.size = spec.fields().size();
      this.data = new Object[size];
    }

    public Type getType(int pos) {
      return partitionType.fields().get(pos).type();
    }

    public String getTransform(int pos) {
      return spec.fields().get(pos).transform().toString();
    }

    @Override
    public int size() {
      return size;
    }

    @Override
    public <T> T get(int pos, Class<T> javaClass) {
      Object value = get(pos);
      if (value == null || javaClass.isInstance(value)) {
        return javaClass.cast(value);
      }

      throw new IllegalArgumentException(
          String.format("Wrong class, %s, for object: %s", javaClass.getName(), value));
    }

    public Object get(int pos) {
      if (pos >= size) {
        return null;
      }

      if (data[pos] instanceof byte[]) {
        return ByteBuffer.wrap((byte[]) data[pos]);
      }

      return data[pos];
    }

    @Override
    public <T> void set(int pos, T value) {
      if (value instanceof Utf8) {
        // Utf8 is not Serializable
        data[pos] = value.toString();
      } else if (value instanceof ByteBuffer) {
        // ByteBuffer is not Serializable
        ByteBuffer buffer = (ByteBuffer) value;
        byte[] bytes = new byte[buffer.remaining()];
        buffer.duplicate().get(bytes);
        data[pos] = bytes;
      } else {
        data[pos] = value;
      }
    }
  }
}
