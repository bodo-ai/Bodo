package com.bodosql.calcite.application.Utils;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import org.apache.calcite.sql.type.SqlTypeName;

public class BodoArrayHelpers {

  /**
   * Takes a sql type, and a string length, and returns a string representing the appropriate
   * nullable array, allocated using bodo helper functions. Will upcast the type to the maximum bit
   * width.
   *
   * @param len String length expression
   * @param typ SqlType
   * @return A string representation of the allocated bodo array with the specified length
   */
  public static String sqlTypeToNullableBodoArray(String len, SqlTypeName typ) {
    switch (typ) {
      case BOOLEAN:
        return String.format("bodo.libs.bool_arr_ext.alloc_bool_array(%s)", len);
      case TINYINT:
      case SMALLINT:
      case INTEGER:
      case BIGINT:
        return String.format("bodo.libs.int_arr_ext.alloc_int_array(%s, bodo.int64)", len);
      case FLOAT:
      case DOUBLE:
      case DECIMAL:
        return String.format("np.empty(%s, dtype=np.float64)", len);
      case DATE:
      case TIMESTAMP:
        return String.format("np.empty(%s, dtype=\"datetime64[ns]\")", len);
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
        return String.format("np.empty(%s, dtype=\"timedelta64[ns]\")", len);
      case CHAR:
      case VARCHAR:
        return String.format("bodo.libs.str_arr_ext.pre_alloc_string_array(%s, -1)", len);
      case BINARY:
      case VARBINARY:
        return String.format("bodo.libs.str_arr_ext.pre_alloc_binary_array(%s, -1)", len);
      default:
        throw new BodoSQLCodegenException(
            "Error, type: " + typ.toString() + " not supported for Window Aggregation functions");
    }
  }

  /**
   * Takes a sql type, and returns a string representing the appropriate type when converting a
   * scalar to this array type that can be used as a global.
   *
   * @param typ SqlType
   * @return The string representing the global.
   */
  public static String sqlTypeToBodoScalarArrayType(SqlTypeName typ) {
    switch (typ) {
      case BOOLEAN:
        return "numba.core.types.Array(numba.core.types.bool_, 1, 'C')";
      case TINYINT:
        return "numba.core.types.Array(numba.core.types.int8, 1, 'C')";
      case SMALLINT:
        return "numba.core.types.Array(numba.core.types.int16, 1, 'C')";
      case INTEGER:
        return "numba.core.types.Array(numba.core.types.int32, 1, 'C')";
      case BIGINT:
        return "numba.core.types.Array(numba.core.types.int64, 1, 'C')";
      case FLOAT:
        return "numba.core.types.Array(numba.core.types.float32, 1, 'C')";
      case DOUBLE:
      case DECIMAL:
        return "numba.core.types.Array(numba.core.types.float64, 1, 'C')";
      case DATE:
      case TIMESTAMP:
        return "numba.core.types.Array(bodo.datetime64ns, 1, 'C')";
      case TIME:
        return "numba.core.types.Array(bodo.Time, 1, 'C')";
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
        return "numba.core.types.Array(bodo.timedelta64ns, 1, 'C')";
      case CHAR:
      case VARCHAR:
        return "bodo.dict_str_arr_type";
      case BINARY:
      case VARBINARY:
        return "bodo.binary_array_type";
      default:
        throw new BodoSQLCodegenException(
            "Error, type: " + typ.toString() + " not supported for scalars");
    }
  }
}
