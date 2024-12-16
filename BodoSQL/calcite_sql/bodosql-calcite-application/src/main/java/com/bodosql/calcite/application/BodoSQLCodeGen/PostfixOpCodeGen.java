package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import java.util.Locale;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.SqlOperator;

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
  public static Expr generatePostfixOpCode(Expr arg, SqlOperator postfixOp) {
    SqlKind kind = postfixOp.getKind();
    String fnName;
    switch (kind) {
      case IS_NULL:
        fnName = "pd.isna";
        break;
      case IS_NOT_NULL:
        fnName = "pd.notna";
        break;
      case IS_NOT_FALSE:
      case IS_NOT_TRUE:
      case IS_TRUE:
      case IS_FALSE:
        // fn_name will be one of is_not_false, is_not_true,
        // is_true, or is_false.
        fnName = "bodosql.kernels." + kind.toString().toLowerCase(Locale.ROOT);
        break;
      default:
        throw new BodoSQLCodegenException(
            "Internal Error: Calcite Plan Produced an Unsupported Postfix Operator");
    }

    return new Expr.Call(fnName, arg);
  }
}
