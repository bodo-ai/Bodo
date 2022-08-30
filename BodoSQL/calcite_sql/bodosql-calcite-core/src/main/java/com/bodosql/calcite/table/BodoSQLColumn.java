package com.bodosql.calcite.table;

import java.sql.JDBCType;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.SqlTypeName;

public interface BodoSQLColumn {
  /**
   * See the design described on Confluence:
   * https://bodo.atlassian.net/wiki/spaces/BodoSQL/pages/1130299393/Java+Table+and+Schema+Typing#Column
   */
  String getColumnName();

  BodoSQLColumnDataType getColumnDataType();

  RelDataType convertToSqlType(RelDataTypeFactory typeFactory);

  /**
   * Does this column type need to be cast to another Bodo type to match the generated Java type.
   */
  boolean requiresCast();

  enum BodoSQLColumnDataType {
    // See _numba_to_sql_param_type_map in context.py
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

    public static BodoSQLColumnDataType fromTypeId(int type_id) {
      for (BodoSQLColumnDataType verbosity : BodoSQLColumnDataType.values()) {
        if (verbosity.getTypeId() == type_id) return verbosity;
      }

      return EMPTY;
    }

    public static BodoSQLColumnDataType fromJavaSqlType(final JDBCType typID) {
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
          throw new RuntimeException(
              String.format("Unsupported Java SQL Type: %s", typID.getName()));
      }
    }

    public RelDataType convertToSqlType(RelDataTypeFactory typeFactory) {
      RelDataType temp;
      switch (this) {
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
        case BOOL8:
          temp = typeFactory.createSqlType(SqlTypeName.BOOLEAN);
          break;
        case DATE:
          temp = typeFactory.createSqlType(SqlTypeName.DATE);
          break;
        case DATETIME:
          temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
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
        case STRING:
          temp = typeFactory.createSqlType(SqlTypeName.VARCHAR);
          break;
        case BINARY:
          temp = typeFactory.createSqlType(SqlTypeName.VARBINARY);
          break;
        default:
          temp = null;
      }
      // TODO: Raise an exception if temp is NULL
      return temp;
    }

    public boolean requiresCast() {
      switch (this) {
        case DATE:
          return true;
        default:
          return false;
      }
    }
  }
}
