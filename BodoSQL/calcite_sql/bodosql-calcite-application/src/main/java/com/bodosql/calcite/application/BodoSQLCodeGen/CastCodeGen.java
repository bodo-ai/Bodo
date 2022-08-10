package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.sqlTypenameToPandasTypename;

import org.apache.calcite.sql.type.SqlTypeName;

/** Class that returns the generated code for Cast calls after all inputs have been visited. */
public class CastCodeGen {
  /**
   * Function that return the necessary generated code for Cast call.
   *
   * @param arg The arg expr.
   * @param typeName The SQL type to cast to.
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the Cast call.
   */
  public static String generateCastCode(String arg, SqlTypeName typeName, boolean outputScalar) {
    StringBuilder codeBuilder = new StringBuilder();
    String dtype = sqlTypenameToPandasTypename(typeName, outputScalar, false);
    if (outputScalar) {
      codeBuilder.append(dtype).append("(").append(arg).append(")");
    } else {
      codeBuilder.append(arg).append(".astype(").append(dtype).append(", _bodo_nan_to_str=False)");
    }
    // Date needs special handling to truncate timestamp. We always round down.
    // TODO: Remove once we support Date type natively
    if (typeName == SqlTypeName.DATE) {
      if (outputScalar) {
        codeBuilder.append(".floor(freq=\"D\")");
      } else {
        codeBuilder.append(".dt.floor(freq=\"D\")");
      }
    }
    return codeBuilder.toString();
  }

  /**
   * Function that returns the generated name for a Cast call.
   *
   * @param name The name for the arg.
   * @return The name generated that matches the Cast call.
   */
  public static String generateCastName(String name, SqlTypeName typeName) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("CAST(").append(name).append(" AS ").append(typeName.toString()).append(")");
    return nameBuilder.toString();
  }
}
