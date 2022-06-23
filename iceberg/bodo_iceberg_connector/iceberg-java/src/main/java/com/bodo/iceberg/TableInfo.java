package com.bodo.iceberg;

import java.util.List;
import org.apache.iceberg.PartitionField;
import org.apache.iceberg.Schema;
import org.apache.iceberg.SortField;
import org.apache.iceberg.Table;

public class TableInfo {
  /** Class that holds the minimal Table info needed by Bodo */
  private final Schema currSchema;

  private final List<SortField> sortFields;
  private final List<PartitionField> partitionFields;

  TableInfo(Table table) {
    currSchema = table.schema();
    sortFields = table.sortOrder().fields();
    partitionFields = table.spec().fields();
  }

  public Schema getCurrSchema() {
    return currSchema;
  }

  public List<SortField> getSortFields() {
    return sortFields;
  }

  public List<PartitionField> getPartitionFields() {
    return partitionFields;
  }
}
