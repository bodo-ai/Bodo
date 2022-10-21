package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;

import com.bodosql.calcite.application.BodoSQLExprType;

/**
 * Class that returns the generated code for Filter expressions after all inputs have been visited.
 */
public class FilterCodeGen {

  /**
   * Function that return the necessary generated code for a Filter expression.
   *
   * @param inVar The input variable/dateframe.
   * @param outVar The output variable.
   * @param filterCode The code for the filter expression.
   * @param filterExprType The expr type for the filter expression, scalar or column.
   * @return The code generated for the Filter expression.
   */
  public static String generateFilterCode(
      String inVar, String outVar, String filterCode, BodoSQLExprType.ExprType filterExprType) {
    StringBuilder filterBuilder = new StringBuilder();
    // TODO: Parameterize indent?
    final String indent = getBodoIndent();
    filterBuilder.append(indent).append(outVar).append(" = ").append(inVar).append("[(");
    // If we have a scalar we need to convert the filter to an array
    if (filterExprType == BodoSQLExprType.ExprType.SCALAR) {
      filterBuilder
          .append("np.full(len(")
          .append(inVar)
          .append("), np.bool_(")
          .append(filterCode)
          .append("), np.bool_)");
    } else {
      filterBuilder.append(filterCode);
    }
    // Note: Pandas/Bodo treats NA values in a boolean array of a getitem operation as False.
    filterBuilder.append(")]\n");
    // Note the index may not match Spark/be reset in the final output. This is an expected
    // difference because the
    // index cannot influence the SQL queries. A user can call reset_index on the output if they
    // would like.
    return filterBuilder.toString();
  }
}
