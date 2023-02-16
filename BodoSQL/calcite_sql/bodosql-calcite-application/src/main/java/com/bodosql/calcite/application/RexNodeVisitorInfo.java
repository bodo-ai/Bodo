package com.bodosql.calcite.application;

/** data class containing the return values from visiting a RexNode. */
public class RexNodeVisitorInfo {
  private String exprCode;
  // index of column in input table for RexInputRef case, -1 otherwise
  private int index;

  public RexNodeVisitorInfo(String exprCode) {
    this.exprCode = exprCode;
    this.index = -1;
  }

  public RexNodeVisitorInfo(String exprCode, int index) {
    this.exprCode = exprCode;
    this.index = index;
  }

  public String getExprCode() {
    return exprCode;
  }

  public int getIndex() {
    return index;
  }
}
