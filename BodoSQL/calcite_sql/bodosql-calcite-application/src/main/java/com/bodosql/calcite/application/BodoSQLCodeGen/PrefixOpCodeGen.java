package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.List;
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
  public static Expr generatePrefixOpCode(Expr arg, SqlOperator prefixOp) {
    String fnName;
    switch (prefixOp.getKind()) {
      case NOT:
        fnName = "boolnot";
        break;
      case MINUS_PREFIX:
        fnName = "negate";
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported Prefix Operator");
    }

    return ExprKt.bodoSQLKernel(fnName, List.of(arg), List.of());
  }
}
