package com.bodosql.calcite.application;

/**
 * Defines a simple struct to store the Pandas code and SQL plan. This is used by APIs that return
 * both the plan and the generated code. e.g. getPandasAndPlanString
 */
public final class CodePlanPair {
  // The Pandas code
  private final String code;
  // The SQL plan
  private final String plan;

  public CodePlanPair(String code, String plan) {
    this.code = code;
    this.plan = plan;
  }

  public String getCode() {
    return code;
  }

  public String getPlan() {
    return plan;
  }
}
