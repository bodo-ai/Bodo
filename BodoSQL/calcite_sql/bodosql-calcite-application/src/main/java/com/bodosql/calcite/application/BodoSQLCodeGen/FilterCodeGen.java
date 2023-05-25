package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import com.bodosql.calcite.ir.Variable;
import java.util.List;

/**
 * Class that returns the generated code for Filter expressions after all inputs have been visited.
 */
public class FilterCodeGen {

  /**
   * Function that return the necessary generated code for a Filter expression.
   *
   * @param inVar The input variable/dateframe.
   * @param filterCode The code for the filter expression.
   * @param filterExprType The expr type for the filter expression, scalar or column.
   * @return The code generated for the Filter expression.
   */
  public static Expr generateFilterCode(
      Variable inVar, Expr filterCode, BodoSQLExprType.ExprType filterExprType) {
    // If we have a scalar we need to convert the filter to an array

    Expr index;
    if (filterExprType == BodoSQLExprType.ExprType.SCALAR) {
      Expr.Len lenVar = new Expr.Len(inVar);
      Expr isTrueKernel = ExprKt.BodoSQLKernel("is_true", List.of(filterCode), List.of());

      Expr.Attribute outputType = new Expr.Attribute(new Expr.Raw("bodo"), "boolean_array_type");

      index =
          new Expr.Call("bodo.utils.utils.full_type", List.of(lenVar, isTrueKernel, outputType));
    } else {
      index = filterCode;
    }
    // Note: Pandas/Bodo treats NA values in a boolean array of a getitem operation
    // as False.
    // Note the index may not match Spark/be reset in the final output. This is an
    // expected
    // difference because the
    // index cannot influence the SQL queries. A user can call reset_index on the
    // output if they
    // would like.
    Expr filter = new Expr.GetItem(inVar, index);
    return filter;
  }
}
