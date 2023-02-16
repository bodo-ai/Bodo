package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.DatetimeFnCodeGen.generateDateTruncCode;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.Utils.Utils.sqlTypenameToPandasTypename;
import static org.apache.calcite.sql.type.SqlTypeName.DATE;
import static org.apache.calcite.sql.type.SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE;

import com.bodosql.calcite.application.*;
import org.apache.calcite.rel.type.*;
import org.apache.calcite.sql.type.*;

/** Class that returns the generated code for Cast calls after all inputs have been visited. */
public class CastCodeGen {
  /**
   * Function that return the necessary generated code for Cast call.
   *
   * @param arg The arg expr.
   * @param inputType The original input data type that needs to be cast.
   * @param outputType The output data type that is the target of the case.
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the Cast call.
   */
  public static String generateCastCode(
      String arg, RelDataType inputType, RelDataType outputType, boolean outputScalar) {
    StringBuilder codeBuilder = new StringBuilder();
    SqlTypeName inputTypeName = inputType.getSqlTypeName();
    SqlTypeName outputTypeName = outputType.getSqlTypeName();
    switch (outputTypeName) {
      case CHAR:
      case VARCHAR:
        codeBuilder.append("bodo.libs.bodosql_array_kernels.to_char(").append(arg).append(")");
        break;
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
        String tzStr = ((TZAwareSqlType) outputType).getTZInfo().getPyZone();
        // TZ-Aware data needs special handling
        switch (inputTypeName) {
          case TIMESTAMP:
          case DATE:
            // Both date and timestamp use the same kernel because the input
            // data is a tz-naive timestamp
            codeBuilder
                .append("bodo.libs.bodosql_array_kernels.cast_tz_naive_to_tz_aware(")
                .append(arg)
                .append(", ")
                .append(tzStr)
                .append(")");
            break;
          case CHAR:
          case VARCHAR:
            // Strings cast as a normal timestamp but with tz-information.
            codeBuilder
                .append("bodo.libs.bodosql_array_kernels.cast_str_to_tz_aware(")
                .append(arg)
                .append(", ")
                .append(tzStr)
                .append(")");
            break;
          default:
            throw new BodoSQLCodegenException(
                String.format("Unsupported cast: %s to %s", inputTypeName, outputTypeName));
        }
        break;
      case DATE:
      case TIMESTAMP:
        // If we cast from tz-aware to naive there is special handling. Otherwise, we
        // fall back to the default case.
        if (inputTypeName == TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
          // Should we normalize the Timestamp to just the date contents?
          String secondArg = outputTypeName == DATE ? "True" : "False";
          codeBuilder
              .append("bodo.libs.bodosql_array_kernels.cast_tz_aware_to_tz_naive(")
              .append(arg)
              .append(", ")
              .append(secondArg)
              .append(")");
          break;
        }
      default:
        StringBuilder asTypeBuilder = new StringBuilder();
        SqlTypeName typeName = outputType.getSqlTypeName();
        String dtype = sqlTypenameToPandasTypename(typeName, outputScalar);
        if (outputScalar) {
          asTypeBuilder.append(dtype).append("(").append(arg).append(")");
        } else {
          // TODO: replace Series.astype/dt with array operation
          asTypeBuilder
              .append("bodo.hiframes.pd_series_ext.get_series_data(")
              .append("pd.Series(")
              .append(arg)
              .append(").astype(")
              .append(dtype)
              .append(", _bodo_nan_to_str=False))");
        }
        // Date needs special handling to truncate timestamp. We always round down.
        // TODO: Remove once we support Date type natively
        if (typeName == SqlTypeName.DATE) {
          // Generate a dummy visitor so we can reuse DATE_TRUNC code.
          RexNodeVisitorInfo dayVisitor = new RexNodeVisitorInfo("day");
          codeBuilder.append(
              generateDateTruncCode(
                      dayVisitor.getExprCode(), new RexNodeVisitorInfo(asTypeBuilder.toString()))
                  .getExprCode());
        } else {
          codeBuilder.append(asTypeBuilder);
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
