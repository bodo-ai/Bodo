package com.bodo.iceberg;

import java.util.List;
import org.apache.iceberg.*;
import org.apache.iceberg.arrow.ArrowSchemaUtil;

public class TableInfo {
  /** Class that holds the minimal Table info needed by Bodo */
  private final int schemaID;

  private final Schema icebergSchema;
  private final String loc;
  private final List<SortField> sortFields;
  private final List<PartitionField> partitionFields;

  TableInfo(Table table) {
    schemaID = table.schema().schemaId();
    icebergSchema = table.schema();
    sortFields = table.sortOrder().fields();
    partitionFields = table.spec().fields();
    loc = table.location();
  }

  public int getSchemaID() {
    return schemaID;
  }

  public Schema getIcebergSchema() {
    return icebergSchema;
  }

  public String getIcebergSchemaEncoding() {
    return SchemaParser.toJson(icebergSchema);
  }

  public org.apache.arrow.vector.types.pojo.Schema getArrowSchema() {
    return ArrowSchemaUtil.convert(icebergSchema);
  }

  public List<SortField> getSortFields() {
    return sortFields;
  }

  public List<PartitionField> getPartitionFields() {
    return partitionFields;
  }

  public String getLoc() {
    return loc;
  }
}
