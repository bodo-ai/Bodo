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
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the Prefix Operator call.
   */
  public static String generatePrefixOpCode(
      String arg, SqlOperator prefixOp, boolean outputScalar) {
    StringBuilder codeBuilder = new StringBuilder();
    switch (prefixOp.getKind()) {
      case NOT:
        if (outputScalar) {
          codeBuilder
              .append("bodosql.libs.generated_lib.sql_null_checking_not( ")
              .append(arg)
              .append(")");
        } else {

          codeBuilder.append("~(").append(arg).append(")");
        }
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

  /**
   * Function that returns the generated name for a Prefix Operator call.
   *
   * @param name The name for the arg.
   * @param postfixOp The postfix operator.
   * @return The name generated that matches the Prefix Operator call.
   */
  public static String generatePrefixOpName(String name, SqlOperator postfixOp) {
    StringBuilder nameBuilder = new StringBuilder();
    switch (postfixOp.getKind()) {
      case NOT:
        nameBuilder.append("NOT(").append(name).append(")");
        break;
      case MINUS_PREFIX:
        nameBuilder.append("-(").append(name).append(")");
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported Prefix Operator");
    }
    return nameBuilder.toString();
  }
}
