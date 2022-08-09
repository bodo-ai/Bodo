package com.bodosql.calcite.catalog.domain;

import static java.sql.Types.BIGINT;

import java.sql.JDBCType;

public enum CatalogColumnDataType {
  // See cudf/types.hpp type_id enum
  EMPTY(0, "EMPTY"), // / < Always null with no underlying data
  INT8(1, "INT8"), // / < 1 byte signed integer
  INT16(2, "INT16"), // / < 2 byte signed integer
  INT32(3, "INT32"), // / < 4 byte signed integer
  INT64(4, "INT64"), // / < 8 byte signed integer
  UINT8(5, "UINT8"), // /< 1 byte unsigned integer
  UINT16(6, "UINT16"), // /< 2 byte unsigned integer
  UINT32(7, "UINT32"), // /< 4 byte unsigned integer
  UINT64(8, "UINT64"), // /< 8 byte unsigned integer
  FLOAT32(9, "FLOAT32"), // /< 4 byte floating point
  FLOAT64(10, "FLOAT64"), // /< 8 byte floating point
  BOOL8(11, "BOOL8"), // /< Boolean using one byte per value, 0 == false, else true
  DATE(12, "DATE"), // /< equivalent to datetime.date value
  DATETIME(13, "DATETIME"), // /< equivalent to datetime64[ns] value
  TIMEDELTA(14, "TIMEDELTA"), // /< equivalent to timedelta64[ns] value
  DATEOFFSET(15, "DATEOFFSET"), // /< equivalent to pd.DateOffset value
  STRING(16, "STRING"), // /< String elements
  BINARY(17, "BINARY"), // /< Binary (byte) array
  // `NUM_TYPE_IDS` must be last!
  NUM_TYPE_IDS(18, "NUM_TYPE_IDS"); // /< Total number of type ids

  private final int type_id;
  private final String type_id_name;

  private CatalogColumnDataType(int type_id, String type_id_name) {
    this.type_id = type_id;
    this.type_id_name = type_id_name;
  }

  public final int getTypeId() {
    return this.type_id;
  }

  public final String getTypeIdName() {
    return this.type_id_name;
  }

  public static CatalogColumnDataType fromTypeId(int type_id) {
    for (CatalogColumnDataType verbosity : CatalogColumnDataType.values()) {
      if (verbosity.getTypeId() == type_id) return verbosity;
    }

    return EMPTY;
  }

  public static CatalogColumnDataType fromString(final String typeIDName) {
    CatalogColumnDataType dataType = null;
    switch (typeIDName) {
      case "EMPTY":
        return EMPTY;
      case "INT8":
        return INT8;
      case "INT16":
        return INT16;
      case "INT32":
        return INT32;
      case "INT64":
        return INT64;
      case "UINT8":
        return UINT8;
      case "UINT16":
        return UINT16;
      case "UINT32":
        return UINT32;
      case "UINT64":
        return UINT64;
      case "FLOAT32":
        return FLOAT32;
      case "FLOAT64":
        return FLOAT64;
      case "BOOL8":
        return BOOL8;
      case "DATE":
        return DATE;
      case "DATETIME":
        return DATETIME;
      case "TIMEDELTA":
        return TIMEDELTA;
      case "STRING":
        return STRING;
      case "BINARY":
        return BINARY;
      case "NUM_TYPE_IDS":
        return NUM_TYPE_IDS;
      default:
        throw new RuntimeException(String.format("Unsupported SQL Type: %s", typeIDName));
    }
  }

  public static CatalogColumnDataType fromJavaSqlType(final JDBCType typID) {
    switch (typID) {
      case BIGINT:
        return INT64;
      case BINARY:
      case LONGVARBINARY:
      case VARBINARY:
        return BINARY;
      case BOOLEAN:
        return BOOL8;
      case CHAR:
      case LONGVARCHAR:
      case LONGNVARCHAR:
      case NCHAR:
      case VARCHAR:
        return STRING;
      case DATE:
        return DATE;
      case DECIMAL:
      case DOUBLE:
      case NUMERIC:
        return FLOAT64;
      case FLOAT:
        return FLOAT32;
      case INTEGER:
        return INT32;
      case SMALLINT:
        return INT16;
      case TIMESTAMP:
        return DATETIME;
      case TINYINT:
        return INT8;
      default:
        throw new RuntimeException(String.format("Unsupported Java SQL Type: %s", typID.getName()));
    }
  }
}
