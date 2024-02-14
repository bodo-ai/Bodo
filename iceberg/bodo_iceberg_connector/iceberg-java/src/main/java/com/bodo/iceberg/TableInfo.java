package com.bodo.iceberg;

import java.util.List;
import java.util.Map;
import org.apache.iceberg.*;

public class TableInfo {
  /** Class that holds the minimal Table info needed by Bodo */
  private final int schemaID;

  private final Schema icebergSchema;
  private final String loc;
  private final List<SortField> sortFields;
  private final List<PartitionField> partitionFields;
  private final Map<String, String> properties;

  TableInfo(Table table) {
    schemaID = table.schema().schemaId();
    icebergSchema = table.schema();
    sortFields = table.sortOrder().fields();
    partitionFields = table.spec().fields();
    loc = table.location();
    properties = table.properties();
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
    return BodoArrowSchemaUtil.convert(icebergSchema);
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

  public Map<String, String> getProperties() {
    return properties;
  }
}
