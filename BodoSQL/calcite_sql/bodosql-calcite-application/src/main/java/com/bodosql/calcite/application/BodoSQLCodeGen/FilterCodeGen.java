package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;

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
    StringBuilder filterBuilder = new StringBuilder();
    // TODO: Parameterize indent?
    filterBuilder.append(inVar.emit()).append("[(");
    // If we have a scalar we need to convert the filter to an array
    if (filterExprType == BodoSQLExprType.ExprType.SCALAR) {
      filterBuilder
          .append("bodo.utils.utils.full_type(len(")
          .append(inVar.emit())
          .append("), bodo.libs.bodosql_array_kernels.is_true(")
          .append(filterCode.emit())
          .append("), bodo.boolean_array_type)");
    } else {
      filterBuilder.append(filterCode.emit());
    }
    // Note: Pandas/Bodo treats NA values in a boolean array of a getitem operation as False.
    filterBuilder.append(")]");
    // Note the index may not match Spark/be reset in the final output. This is an expected
    // difference because the
    // index cannot influence the SQL queries. A user can call reset_index on the output if they
    // would like.
    return new Expr.Raw(filterBuilder.toString());
  }
}
