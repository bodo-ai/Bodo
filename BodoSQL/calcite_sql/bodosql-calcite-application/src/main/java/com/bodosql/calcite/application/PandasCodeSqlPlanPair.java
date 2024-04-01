package com.bodosql.calcite.application;

/**
 * Defines a simple struct to store the Pandas code and SQL plan. This is used by APIs that return
 * both the plan and the generated code. e.g. getPandasAndPlanString
 */
public final class PandasCodeSqlPlanPair {
  // The Pandas code
  private final String pdCode;
  // The SQL plan
  private final String sqlPlan;

  public PandasCodeSqlPlanPair(String pdCode, String sqlPlan) {
    this.pdCode = pdCode;
    this.sqlPlan = sqlPlan;
  }

  public String getPdCode() {
    return this.pdCode;
  }

  public String getSqlPlan() {
    return this.sqlPlan;
  }
}
