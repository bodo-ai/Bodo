package com.bodosql.calcite.table;

import com.bodosql.calcite.ir.Variable;
import java.sql.JDBCType;
import java.util.Locale;
import kotlin.Pair;
import org.apache.calcite.avatica.util.TimeUnit;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rel.type.RelDataTypeFactory;
import org.apache.calcite.sql.SqlIntervalQualifier;
import org.apache.calcite.sql.parser.SqlParserPos;
import org.apache.calcite.sql.type.BodoTZInfo;
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

  boolean isNullable();

  BodoTZInfo getTZInfo();

  int getPrecision();

  RelDataType convertToSqlType(
      RelDataTypeFactory typeFactory, boolean nullable, BodoTZInfo tzInfo, int precision);

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
    TZ_AWARE_TIMESTAMP(15, "TZ_AWARE_TIMESTAMP"), // /< equivalent to Timestamp with tz info
    TIMEDELTA(16, "TIMEDELTA"), // /< equivalent to timedelta64[ns] value
    DATEOFFSET(17, "DATEOFFSET"), // /< equivalent to pd.DateOffset value
    STRING(18, "STRING"), // /< String elements
    BINARY(19, "BINARY"), // /< Binary (byte) array
    CATEGORICAL(20, "CATEGORICAL"),
    ARRAY(21, "ARRAY"),
    JSON_OBJECT(22, "JSON_OBJECT"),
    VARIANT(23, "VARIANT"),
    UNSUPPORTED(24, "UNSUPPORTED"), // Unknown type we may be able to prune
    // `NUM_TYPE_IDS` must be last!
    NUM_TYPE_IDS(25, "NUM_TYPE_IDS"); // /< Total number of type ids

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
          // TODO: Define a separate type for containing timezones
          return DATETIME;
        case TIMESTAMP_WITH_TIMEZONE:
          return TZ_AWARE_TIMESTAMP;
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

    public static BodoSQLColumnDataType fromSqlType(RelDataType relDataType) {
      SqlTypeName typeName = relDataType.getSqlTypeName();
      BodoSQLColumnDataType outType;
      switch (typeName) {
        case TINYINT:
          outType = BodoSQLColumnDataType.INT8;
          break;
        case SMALLINT:
          outType = BodoSQLColumnDataType.INT16;
          break;
        case INTEGER:
          outType = BodoSQLColumnDataType.INT32;
          break;
        case BIGINT:
          outType = BodoSQLColumnDataType.INT64;
          break;
        case FLOAT:
          outType = BodoSQLColumnDataType.FLOAT32;
          break;
        case REAL:
        case DOUBLE:
        case DECIMAL:
          outType = BodoSQLColumnDataType.FLOAT64;
          break;
        case DATE:
          outType = BodoSQLColumnDataType.DATE;
          break;
        case CHAR:
        case VARCHAR:
          outType = BodoSQLColumnDataType.STRING;
          break;
        case TIMESTAMP:
          outType = BodoSQLColumnDataType.DATETIME;
          break;
        case BOOLEAN:
          outType = BodoSQLColumnDataType.BOOL8;
          break;
        case INTERVAL_DAY_HOUR:
        case INTERVAL_DAY_MINUTE:
        case INTERVAL_DAY_SECOND:
        case INTERVAL_HOUR_MINUTE:
        case INTERVAL_HOUR_SECOND:
        case INTERVAL_MINUTE_SECOND:
        case INTERVAL_HOUR:
        case INTERVAL_MINUTE:
        case INTERVAL_SECOND:
        case INTERVAL_DAY:
        case INTERVAL_YEAR:
        case INTERVAL_MONTH:
        case INTERVAL_YEAR_MONTH:
          outType = BodoSQLColumnDataType.TIMEDELTA;
          break;
        default:
          throw new RuntimeException(
              "Internal Error: Calcite Plan Produced an Unsupported relDataType"
                  + "for table extension Type");
      }
      return outType;
    }

    /**
     * Parse the BodoSQL column data type obtained from a type returned by a "DESCRIBE TABLE" call.
     * This is done to obtain information about types that cannot be communicated via JDBC APIs.
     *
     * @param typeName The type name string returned for a column by Snowflake's describe table
     *     query.
     * @return A pair of the BodoSQLColumnDataType and precision for the type in question. If the
     *     type is not Time then the precision value is garbage and its value is ignored.
     */
    public static Pair<BodoSQLColumnDataType, Integer> fromSnowflakeTypeName(String typeName) {
      // Convert the type to all caps to simplify checking.
      typeName = typeName.toUpperCase(Locale.ROOT);
      final BodoSQLColumnDataType columnDataType;
      int precision = 0;
      if (typeName.startsWith("NUMBER")) {
        // If we encounter a number type we need to parse it to determine the actual type.
        // The type information is of the form NUMBER(PRECISION, SCALE)
        String internalFields = typeName.split("\\(|\\)")[1];
        String[] numericParts = internalFields.split(",");
        precision = Integer.valueOf(numericParts[0].trim());
        int scale = Integer.valueOf(numericParts[1].trim());
        if (scale > 0) {
          // If scale > 0 then we have a Float, Double, or Decimal Type
          // Currently we only support having DOUBLE inside SQL
          columnDataType = FLOAT64;
        } else {
          // We have an integer type.
          if (precision < 3) {
            // If we use only 2 digits we know that this value fits in an int8
            columnDataType = INT8;
          } else if (precision < 5) {
            // Using 4 digits always fits in an int16
            columnDataType = INT16;
          } else if (precision < 10) {
            // Using 10 digits fits in an int32
            columnDataType = INT32;
          } else {
            // Our max type is int64
            columnDataType = INT64;
          }
        }
      } else if (typeName.equals("BOOLEAN")) {
        columnDataType = BOOL8;
      } else if (typeName.startsWith("VARCHAR")
          || typeName.startsWith("CHAR")
          || typeName.equals("TEXT")
          || typeName.equals("STRING")) {
        // TODO: Load max string information
        columnDataType = STRING;
      } else if (typeName.equals("DATE")) {
        columnDataType = DATE;
      } else if (typeName.startsWith("DATETIME") || typeName.startsWith("TIMESTAMP_NTZ")) {
        columnDataType = DATETIME;
      } else if (typeName.startsWith("TIMESTAMP_TZ") || typeName.startsWith("TIMESTAMP_LTZ")) {
        columnDataType = TZ_AWARE_TIMESTAMP;
      } else if (typeName.startsWith("TIME(")) {
        columnDataType = TIME;
        // Determine the precision by parsing the type.
        String precisionString = typeName.split("\\(|\\)")[1];
        precision = Integer.valueOf(precisionString.trim());
      } else if (typeName.startsWith("BINARY") || typeName.startsWith("VARBINARY")) {
        columnDataType = BINARY;
      } else if (typeName.equals("VARIANT")) {
        columnDataType = VARIANT;
      } else if (typeName.equals("OBJECT")) {
        // TODO: Replace with a map type if possible.
        columnDataType = JSON_OBJECT;
      } else if (typeName.startsWith("ARRAY")) {
        // TODO: Replace with array containing dtype info.
        columnDataType = ARRAY;
      } else if (typeName.startsWith("TIMESTAMP")) {
        // TODO: Leverage the snowflake session parameter.
        columnDataType = DATETIME;
      } else if (typeName.startsWith("FLOAT")
          || typeName.startsWith("DOUBLE")
          || typeName.equals("REAL")
          || typeName.equals("DECIMAL")
          || typeName.equals("NUMERIC")) {
        // A snowflake bug outputs float for double and float, so we match double
        columnDataType = FLOAT64;
      } else if (typeName.startsWith("INT")
          || typeName.equals("BIGINT")
          || typeName.equals("SMALLINT")
          || typeName.equals("TINYINT")
          || typeName.equals("BYTEINT")) {
        // Snowflake treats these all internally as suggestions so we must use INT64
        columnDataType = INT64;
      } else {
        // Unsupported types (e.g. GEOGRAPHY and GEOMETRY) may be in the table but unused,
        // so we don't fail here.
        columnDataType = UNSUPPORTED;
      }
      return new Pair<>(columnDataType, precision);
    }

    public RelDataType convertToSqlType(
        RelDataTypeFactory typeFactory, boolean nullable, BodoTZInfo tzInfo, int precision) {
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
        case TIME:
          temp = typeFactory.createSqlType(SqlTypeName.TIME, precision);
          break;
        case DATETIME:
          temp = typeFactory.createSqlType(SqlTypeName.TIMESTAMP);
          break;
        case TZ_AWARE_TIMESTAMP:
          assert tzInfo != null;
          temp = typeFactory.createTZAwareSqlType(tzInfo);
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
        case VARIANT:
        case JSON_OBJECT:
        case ARRAY:
          // Currently VARIANT, ARRAY, and JSON_OBJECT aren't yet supported
          // inside BodoSQL, so we treat them as strings
          // (which is how the snowflake connector will load them)
          temp = typeFactory.createSqlType(SqlTypeName.VARCHAR);
          break;
        case BINARY:
          temp = typeFactory.createSqlType(SqlTypeName.VARBINARY);
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
          return "Float32";
        case FLOAT64:
          return "Float64";
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
