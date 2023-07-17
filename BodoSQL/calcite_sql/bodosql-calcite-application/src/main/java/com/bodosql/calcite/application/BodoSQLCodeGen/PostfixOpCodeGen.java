package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.*;
import org.apache.calcite.sql.*;

/**
 * Class that returns the generated code for Postfix Operators after all inputs have been visited.
 */
public class PostfixOpCodeGen {
  /**
   * Function that return the necessary generated code for a Postfix Operator call.
   *
   * @param arg The arg expr.
   * @param postfixOp The postfix operator.
   * @return The code generated that matches the Postfix Operator call.
   */
  public static String generatePostfixOpCode(String arg, SqlOperator postfixOp) {
    StringBuilder codeBuilder = new StringBuilder();
    SqlKind kind = postfixOp.getKind();
    switch (kind) {
      case IS_NULL:
        codeBuilder.append("pd.isna(").append(arg).append(")");
        break;
      case IS_NOT_NULL:
        codeBuilder.append("pd.notna(").append(arg).append(")");
        break;
      case IS_NOT_FALSE:
      case IS_NOT_TRUE:
      case IS_TRUE:
      case IS_FALSE:
        // fn_name will be one of is_not_false, is_not_true,
        // is_true, or is_false.
        String fn_name = kind.toString().toLowerCase(Locale.ROOT);
        codeBuilder
            .append("bodo.libs.bodosql_array_kernels.")
            .append(fn_name)
            .append("(")
            .append(arg)
            .append(")");
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported Postfix Operator");
    }

    return codeBuilder.toString();
  }
}
