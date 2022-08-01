package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

public class WindowedAggregationArgument {
  /**
   * A simple wrapper class for the input to an aggregation function, that keeps track of the type
   * of the input argument type (literal or column reference).
   */
  private final boolean isDfCol;

  private final String exprString;

  protected WindowedAggregationArgument(String exprString, boolean isDfCol) {
    this.isDfCol = isDfCol;
    this.exprString = exprString;
  }

  public static com.bodosql.calcite.application.BodoSQLCodeGen.WindowedAggregationArgument
      fromColumnName(String colName) {
    return new com.bodosql.calcite.application.BodoSQLCodeGen.WindowedAggregationArgument(
        colName, true);
  }

  public static com.bodosql.calcite.application.BodoSQLCodeGen.WindowedAggregationArgument
      fromLiteralExpr(String expr) {
    return new com.bodosql.calcite.application.BodoSQLCodeGen.WindowedAggregationArgument(
        expr, false);
  }

  public String toFormatedExprString(String input_df_name) {
    if (isDfCol) {
      return input_df_name + "[" + makeQuoted(exprString) + "]";
    } else {
      return exprString;
    }
  }

  public String getExprString() {
    return exprString;
  }

  public boolean isDfCol() {
    return isDfCol;
  }
}
