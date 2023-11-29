package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.Utils.sqlTypenameToPandasTypename;
import static com.bodosql.calcite.ir.ExprKt.BodoSQLKernel;
import static org.apache.calcite.sql.type.SqlTypeName.TIMESTAMP_WITH_LOCAL_TIME_ZONE;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.False;
import com.bodosql.calcite.ir.Expr.None;
import com.bodosql.calcite.ir.Expr.True;
import java.util.ArrayList;
import java.util.List;
import kotlin.Pair;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.sql.type.TZAwareSqlType;
import org.apache.calcite.sql.type.VariantSqlType;

/** Class that returns the generated code for Cast calls after all inputs have been visited. */
public class CastCodeGen {
  /**
   * Function that return the necessary generated code for Cast call.
   *
   * @param arg The arg expr.
   * @param inputType The original input data type that needs to be cast.
   * @param outputType The output data type that is the target of the case.
   * @param outputScalar Should the output generate scalar code.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The code generated that matches the Cast call.
   */
  public static Expr generateCastCode(
      Expr arg,
      RelDataType inputType,
      RelDataType outputType,
      boolean outputScalar,
      List<Pair<String, Expr>> streamingNamedArgs) {
    SqlTypeName inputTypeName = inputType.getSqlTypeName();
    SqlTypeName outputTypeName = outputType.getSqlTypeName();
    String fnName;
    // Create the args. Some function paths may have multiple args.
    List<Expr> args = new ArrayList<>();
    args.add(arg);
    boolean appendStreamingArgs = false;
    switch (outputTypeName) {
      case CHAR:
      case VARCHAR:
        fnName = "bodo.libs.bodosql_array_kernels.to_char";
        break;
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
        Expr tzExpr = ((TZAwareSqlType) outputType).getTZInfo().getZoneExpr();
        args.add(tzExpr);
        // TZ-Aware data needs special handling
        switch (inputTypeName) {
          case TIMESTAMP:
            fnName = "bodo.libs.bodosql_array_kernels.cast_tz_naive_to_tz_aware";
            break;
          case DATE:
            fnName = "bodo.libs.bodosql_array_kernels.cast_date_to_tz_aware";
            break;
          case CHAR:
          case VARCHAR:
          case NULL:
            fnName = "bodo.libs.bodosql_array_kernels.cast_str_to_tz_aware";
            appendStreamingArgs = true;
            break;
          default:
            throw new BodoSQLCodegenException(
                java.lang.String.format(
                    "Unsupported cast: %s to %s", inputTypeName, outputTypeName));
        }
        break;
      case DATE:
        fnName = "bodo.libs.bodosql_array_kernels.to_date";
        args.add(Expr.None.INSTANCE);
        appendStreamingArgs = true;
        break;
      case TIMESTAMP:
        // If we cast from tz-aware to naive there is special handling. Otherwise, we
        // fall back to the default case.
        if (inputTypeName == TIMESTAMP_WITH_LOCAL_TIME_ZONE) {
          fnName = "bodo.libs.bodosql_array_kernels.cast_tz_aware_to_tz_naive";
          break;
        }
      default:
        if (outputType instanceof VariantSqlType) {
          return arg;
        }
        StringBuilder asTypeBuilder = new StringBuilder();
        SqlTypeName typeName = outputType.getSqlTypeName();
        String dtype = sqlTypenameToPandasTypename(typeName, outputScalar);
        if (outputScalar) {
          fnName = dtype;
          asTypeBuilder.append(dtype).append("(").append(arg).append(")");
        } else {
          // TODO(njriasan): replace Series.astype/dt with array operation
          fnName = "bodo.hiframes.pd_series_ext.get_series_data";
          // Replace the arg with a call to pd.Series and an astype.
          Expr newArg =
              new Expr.Method(
                  new Expr.Call("pd.Series", arg),
                  "astype",
                  List.of(new Expr.Raw(dtype)),
                  List.of(new Pair<>("_bodo_nan_to_str", False.INSTANCE)));
          args.set(0, newArg);
        }
    }
    return new Expr.Call(fnName, args, appendStreamingArgs ? streamingNamedArgs : List.of());
  }

  /**
   * Function that return the necessary generated code for a tryCast call.
   *
   * @param arg The arg expr.
   * @param outputType The output data type that is the target of the case.
   * @param streamingNamedArgs The additional arguments used for streaming. This is an empty list if
   *     we aren't in a streaming context.
   * @return The code generated that matches the tryCast call.
   */
  public static Expr generateTryCastCode(
      Expr arg, RelDataType outputType, List<Pair<String, Expr>> streamingNamedArgs) {
    SqlTypeName outputTypeName = outputType.getSqlTypeName();
    String fnName;
    List<Expr> args = new ArrayList<>();
    args.add(arg);
    switch (outputTypeName) {
      case CHAR:
      case VARCHAR:
        // Casting String -> String is a NO-OP
        return arg;
      case BOOLEAN:
        fnName = "try_to_boolean";
        break;
      case DATE:
        fnName = "try_to_date";
        args.add(None.INSTANCE);
        break;
      case DECIMAL:
      case INTEGER:
        fnName = "try_to_number";
        // Add defaults
        args.add(new Expr.IntegerLiteral(38));
        args.add(new Expr.IntegerLiteral(0));
        break;
      case DOUBLE:
      case FLOAT:
        fnName = "try_to_double";
        args.add(None.INSTANCE);
        break;
      case TIME:
        fnName = "to_time";
        args.add(None.INSTANCE);
        args.add(True.INSTANCE);
        break;
      case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
        fnName = "try_to_timestamp";
        args.add(None.INSTANCE);
        Expr tzExpr = ((TZAwareSqlType) outputType).getTZInfo().getZoneExpr();
        args.add(tzExpr);
        args.add(new Expr.IntegerLiteral(0));
        break;
      case TIMESTAMP:
        fnName = "try_to_timestamp";
        args.add(None.INSTANCE);
        args.add(None.INSTANCE);
        args.add(new Expr.IntegerLiteral(0));
        break;
      default:
        throw new BodoSQLCodegenException(
            String.format("%s is not supported by TRY_CAST.", outputTypeName));
    }
    return BodoSQLKernel(fnName, args, streamingNamedArgs);
  }
}
