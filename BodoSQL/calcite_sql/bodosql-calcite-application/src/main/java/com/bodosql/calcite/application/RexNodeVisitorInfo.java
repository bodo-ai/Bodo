package com.bodosql.calcite.application;

/** data class containing the return values from visiting a RexNode. */
public class RexNodeVisitorInfo {
  private String name;
  private String exprCode;
  // index of column in input table for RexInputRef case, -1 otherwise
  private int index;

  public RexNodeVisitorInfo(String name, String exprCode) {
    this.name = name;
    this.exprCode = exprCode;
    this.index = -1;
  }

  public RexNodeVisitorInfo(String name, String exprCode, int index) {
    this.name = name;
    this.exprCode = exprCode;
    this.index = index;
  }

  // Name is currently not used in the output. It was used to replace calcite internal
  // names with more reasonable expressions like SUM(A), but it has issues with conflicting
  // names (and may produce complex names that aren't helpful).
  public String getName() {
    return name;
  }

  public String getExprCode() {
    return exprCode;
  }

  public int getIndex() {
    return index;
  }
}
