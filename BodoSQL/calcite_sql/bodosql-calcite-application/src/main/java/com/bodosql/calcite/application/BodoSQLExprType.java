package com.bodosql.calcite.application;

public class BodoSQLExprType {
  public enum ExprType {
    /**
     * Enum that tracks the value returned by each operation. This is used to produce different code
     * based upon the type of input.
     */
    COLUMN,
    SCALAR,
    // Not currently used, but included to handle correctness.
    DATAFRAME,
    // TODO: Add more types like array.
  }

  /**
   * Helper function used to determine the output expr type for operations that perform an
   * elementwise operation when executed between a scalar and a column.
   *
   * @param e1 ExprType of LHS
   * @param e2 ExprType of RHS
   * @return ExprType of the output after performing the operation.
   */
  public static ExprType meet_elementwise_op(ExprType e1, ExprType e2) {
    if ((e1 == ExprType.COLUMN) || (e2 == ExprType.COLUMN)) {
      return ExprType.COLUMN;
    }
    return ExprType.SCALAR;
  }
}
