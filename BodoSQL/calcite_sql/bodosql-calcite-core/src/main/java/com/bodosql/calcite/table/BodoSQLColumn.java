package com.bodosql.calcite.table;

import java.sql.JDBCType;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
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

  BodoSQLColumnDataType getColumnDataType();

  RelDataType convertToSqlType(RelDataTypeFactory typeFactory);

  /**
   * Does reading this column type need to be cast to another Bodo type to match the generated Java
   * type.
   */
  boolean requiresReadCast();

  /** Does write this column type need to be cast back to the original table type. */
  boolean requiresWriteCast();

  /**
   * Generate the expression to cast this column to its BodoSQL type with a read.
   *
   * @param varName Name of the table to use.
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its BodoSQL
   *     supported type with a read.
   */
  String getReadCastExpr(String varName);

  /**
   * Generate the expression to cast this column to its BodoSQL type with a write.
   *
   * @param varName Name of the table to use.
   * @return The string passed to __bodosql_replace_columns_dummy to cast this column to its
   *     original data type with a write.
   */
  String getWriteCastExpr(String varName);

  /** Logger * */
  Logger LOGGER = LoggerFactory.getLogger(BodoSQLColumn.class);

  enum BodoSQLColumnDataType {
    // See SqlTypeEnum in context.py
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
    TIME(13, "TIME"), // /< equivalent to bodo.Time value
    DATETIME(14, "DATETIME"), // /< equivalent to datetime64[ns] value
    TIMEDELTA(15, "TIMEDELTA"), // /< equivalent to timedelta64[ns] value
    DATEOFFSET(16, "DATEOFFSET"), // /< equivalent to pd.DateOffset value
    STRING(17, "STRING"), // /< String elements
    BINARY(18, "BINARY"), // /< Binary (byte) array
    CATEGORICAL(19, "CATEGORICAL"),
    UNSUPPORTED(20, "UNSUPPORTED"), // Unknown type we may be able to prune
    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS(21, "NUM_TYPE_IDS"); // /< Total number of type ids

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
        case TIMESTAMP_WITH_TIMEZONE:
          // TODO: Define a separate type for containing timezones
          return DATETIME;
        case TINYINT:
          return INT8;
        default:
          // We may be able to prune the column so we just output a warning.
          // TODO: Ensure these warnings are visible to users. This probably
          // needs a larger refactoring on the Java side.
          LOGGER.warn(String.format("Unsupported Java SQL Type: %s", typID.getName()));
          return UNSUPPORTED;
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
          // NOTE: BodoSQL doesn't support the date type yet, so all
          // date values are converted to timestamp.
          temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
          break;
        case TIME:
          temp = typeFactory.createSqlType(SqlTypeName.TIME);
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
          // If a type is not supported default to unknown
          temp = typeFactory.createSqlType(SqlTypeName.UNKNOWN);
      }
      return temp;
    }

    public boolean requiresReadCast() {
      switch (this) {
        case CATEGORICAL:
        case DATE:
          return true;
        default:
          return false;
      }
    }

    public boolean requiresWriteCast() {
      switch (this) {
        case DATE:
          return true;
        default:
          return false;
      }
    }

    /** @return The type used to cast an individual type to the supported BodoSQL type. */
    public BodoSQLColumnDataType getCastType() {
      if (this == DATE) {
        return DATETIME;
      }
      return this;
    }

    /** @return A string that represents a nullable version of this type. */
    public String getTypeString() {
      switch (this) {
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
          return "float32";
        case FLOAT64:
          return "float64";
        case BOOL8:
          return "boolean";
        case STRING:
          return "str";
        case DATETIME:
          return "datetime64[ns]";
        case TIMEDELTA:
          return "timedelta64[ns]";
        default:
          throw new RuntimeException(
              String.format("Cast to type %s not supported.", this.getTypeIdName()));
      }
    }
  }
}
