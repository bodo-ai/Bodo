package com.bodosql.calcite.table;

import com.bodosql.calcite.ir.Variable;
import com.bodosql.calcite.rel.type.BodoRelDataTypeFactory;
import java.util.List;
import java.util.Locale;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.rel.type.StructKind;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public interface BodoSQLColumn {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Column
   */
  String getColumnName();

  /** Name to use for the column when writing to the original DB. * */
  String getWriteColumnName();

  /** Data Type Information. * */
  ColumnDataTypeInfo getDataTypeInfo();

  /**
   * Does reading this column type need to be cast to another Bodo type to match the generated Java
   * type.
   */
  boolean requiresReadCast();

  /**
   * Generate the expression to cast this column to its BodoSQL type with a read.
   *
   * @param varName Name of the table to use.
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its BodoSQL
   *     supported type with a read.
   */
  String getReadCastExpr(Variable varName);

  /** Logger * */
  Logger LOGGER = LoggerFactory.getLogger(BodoSQLColumn.class);

  enum BodoSQLColumnDataType {
    // See SqlTypeEnum in context.py
    NULL(0, "NULL"), // / < Always null with no underlying data
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
    DECIMAL(11, "DECIMAL"), // Decimal Type
    BOOL8(12, "BOOL8"), // /< Boolean using one byte per value, 0 == false, else true
    DATE(13, "DATE"), // /< equivalent to datetime.date value
    TIME(14, "TIME"), // /< equivalent to bodo.types.Time value
    TIMESTAMP_NTZ(15, "TIMESTAMP_NTZ"), // /< equivalent to datetime64[ns] value or pd.Timestamp
    TIMESTAMP_LTZ(16, "TIMESTAMP_LTZ"), // /< equivalent to a Timestamp with a timezone
    TIMESTAMP_TZ(17, "TIMESTAMP_TZ"), // /< equivalent to Timestamp with an offset
    TIMEDELTA(18, "TIMEDELTA"), // /< equivalent to timedelta64[ns] value
    DATEOFFSET(19, "DATEOFFSET"), // /< equivalent to pd.DateOffset value
    STRING(20, "STRING"), // /< String elements
    BINARY(21, "BINARY"), // /< Binary (byte) array
    CATEGORICAL(22, "CATEGORICAL"),
    ARRAY(23, "ARRAY"),
    JSON_OBJECT(24, "JSON_OBJECT"),
    STRUCT(25, "STRUCT"),
    VARIANT(26, "VARIANT"),
    FIXED_SIZE_STRING(27, "FIXED_SIZE_STRING"),
    FIXED_SIZE_BINARY(28, "FIXED_SIZE_BINARY"),
    UNSUPPORTED(29, "UNSUPPORTED"), // Unknown type we may be able to prune
    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS(30, "NUM_TYPE_IDS"); // /< Total number of type ids

    private final int type_id;
    private final String type_id_name;

    BodoSQLColumnDataType(int type_id, String type_id_name) {
      this.type_id = type_id;
      this.type_id_name = type_id_name;
    }

    public final int getTypeId() {
      return this.type_id;
    }

    public final String getTypeIdName() {
      return this.type_id_name;
    }

    public static BodoSQLColumnDataType fromTypeId(int typeId) {
      for (BodoSQLColumnDataType verbosity : BodoSQLColumnDataType.values()) {
        if (verbosity.getTypeId() == typeId) return verbosity;
      }
      throw new RuntimeException(String.format(Locale.ROOT, "Unknown type id: %d", typeId));
    }

    public RelDataType convertToSqlType(
        RelDataTypeFactory typeFactory,
        boolean nullable,
        int precision,
        int scale,
        List<RelDataType> children,
        List<String> fieldNames) {
      RelDataType temp;
      switch (this) {
        case NULL:
          temp = typeFactory.createSqlType(SqlTypeName.NULL);
          // Ensure NULL sets nullable to true.
          nullable = true;
          break;
        case INT8:
        case UINT8:
          temp = typeFactory.createSqlType(SqlTypeName.TINYINT);
          break;
        case INT16:
        case UINT16:
          temp = typeFactory.createSqlType(SqlTypeName.SMALLINT);
          break;
        case INT32:
        case UINT32:
          temp = typeFactory.createSqlType(SqlTypeName.INTEGER);
          break;
        case INT64:
        case UINT64:
          temp = typeFactory.createSqlType(SqlTypeName.BIGINT);
          break;
        case FLOAT32:
          temp = typeFactory.createSqlType(SqlTypeName.FLOAT);
          break;
        case FLOAT64:
          temp = typeFactory.createSqlType(SqlTypeName.DOUBLE);
          break;
        case DECIMAL:
          temp = typeFactory.createSqlType(SqlTypeName.DECIMAL, precision, scale);
          break;
        case BOOL8:
          temp = typeFactory.createSqlType(SqlTypeName.BOOLEAN);
          break;
        case DATE:
          temp = typeFactory.createSqlType(SqlTypeName.DATE);
          break;
        case TIME:
          temp = typeFactory.createSqlType(SqlTypeName.TIME, precision);
          break;
        case TIMESTAMP_NTZ:
          temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP, precision);
          break;
        case TIMESTAMP_TZ:
          temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP_TZ, precision);
          break;
        case TIMESTAMP_LTZ:
          temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE, precision);
          break;
        case TIMEDELTA:
          // TODO: Figure out SqlParserPos. Probably not relevant
          // if we aren't using timedelta directly for literals.
          // TODO: Determine how to hanlde Timeunits properly. Default
          // seems to only allow Year-Month and Day-Second.
          temp =
              typeFactory.createSqlIntervalType(
                  new SqlIntervalQualifier(TimeUnit.DAY, TimeUnit.SECOND, SqlParserPos.ZERO));
          break;
        case DATEOFFSET:
          // TODO: Figure out SqlParserPos. Probably not relevant
          // if we aren't using timedelta directly for literals.
          temp =
              typeFactory.createSqlIntervalType(
                  new SqlIntervalQualifier(TimeUnit.YEAR, TimeUnit.MONTH, SqlParserPos.ZERO));
          break;
        case VARIANT:
          temp = BodoRelDataTypeFactory.createVariantSqlType(typeFactory);
          break;
        case JSON_OBJECT:
          temp = typeFactory.createMapType(children.get(0), children.get(1));
          break;
        case ARRAY:
          temp = typeFactory.createArrayType(children.get(0), -1);
          break;
        case STRUCT:
          temp = typeFactory.createStructType(StructKind.FULLY_QUALIFIED, children, fieldNames);
          break;
        case STRING:
          temp = typeFactory.createSqlType(SqlTypeName.VARCHAR, precision);
          break;
        case FIXED_SIZE_STRING:
          temp = typeFactory.createSqlType(SqlTypeName.CHAR, precision);
          break;
        case BINARY:
          temp = typeFactory.createSqlType(SqlTypeName.VARBINARY, precision);
          break;
        case FIXED_SIZE_BINARY:
          temp = typeFactory.createSqlType(SqlTypeName.BINARY, precision);
          break;
        default:
          // If a type is not supported default to unknown
          temp = typeFactory.createSqlType(SqlTypeName.UNKNOWN);
      }
      return typeFactory.createTypeWithNullability(temp, nullable);
    }

    public boolean requiresReadCast() {
      switch (this) {
        case CATEGORICAL:
          return true;
        default:
          return false;
      }
    }

    /**
     * @return A string that represents a nullable version of this type.
     */
    public String getTypeString() {
      switch (this) {
        case NULL:
          return "None";
        case INT8:
          return "Int8";
        case INT16:
          return "Int16";
        case INT32:
          return "Int32";
        case INT64:
          return "Int64";
        case UINT8:
          return "UInt8";
        case UINT16:
          return "Uint16";
        case UINT32:
          return "UInt32";
        case UINT64:
          return "UInt64";
        case FLOAT32:
          return "Float32";
        case FLOAT64:
          return "Float64";
        case DECIMAL:
          return "Decimal";
        case BOOL8:
          return "boolean";
        case STRING:
          return "str";
        case FIXED_SIZE_STRING:
          return "FIXED_STRING(n)";
        case BINARY:
          return "Binary";
        case FIXED_SIZE_BINARY:
          return "FIXED_BINARY(n)";
        case DATE:
          return "Date";
        case TIME:
          return "Time";
        case TIMESTAMP_NTZ:
          return "datetime64[ns]";
        case TIMEDELTA:
          return "timedelta64[ns]";
        case VARIANT:
          return "VARIANT";
        default:
          throw new RuntimeException(
              String.format("Cast to type %s not supported.", this.getTypeIdName()));
      }
    }
  }
}
