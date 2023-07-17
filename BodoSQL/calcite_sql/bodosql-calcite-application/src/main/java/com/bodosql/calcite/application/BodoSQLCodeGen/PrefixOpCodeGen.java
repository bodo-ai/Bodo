package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import org.apache.calcite.sql.SqlOperator;

/**
 * Class that returns the generated code for Prefix Operators after all inputs have been visited.
 */
public class PrefixOpCodeGen {
  /**
   * Function that return the necessary generated code for a Prefix Operator call.
   *
   * @param arg The arg expr.
   * @param prefixOp The prefix operator.
   * @return The code generated that matches the Prefix Operator call.
   */
  public static String generatePrefixOpCode(String arg, SqlOperator prefixOp) {
    StringBuilder codeBuilder = new StringBuilder();
    switch (prefixOp.getKind()) {
      case NOT:
        codeBuilder.append("bodo.libs.bodosql_array_kernels.boolnot(").append(arg).append(")");
        break;
      case MINUS_PREFIX:
        codeBuilder.append("bodo.libs.bodosql_array_kernels.negate(").append(arg).append(")");
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported Prefix Operator");
    }

    return codeBuilder.toString();
  }
}
