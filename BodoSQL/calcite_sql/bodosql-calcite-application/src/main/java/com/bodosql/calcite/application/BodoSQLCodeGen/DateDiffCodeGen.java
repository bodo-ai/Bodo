package com.bodosql.calcite.application.BodoSQLCodeGen;

public class DateDiffCodeGen {

  /**
   * Function that return the necessary generated code for a DateDiff Function Call.
   *
   * @param arg0 The first arg expr.
   * @param arg0Scalar Should arg0 generate scalar code.
   * @param arg1 The second arg expr.
   * @param arg1Scalar Should arg1 generate scalar code.
   * @param isSingleRow Does this datediff take place within an apply
   * @return The code generated that matches the DateDiff expression.
   */
  public static String generateDateDiffCode(
      String arg0, boolean arg0Scalar, String arg1, boolean arg1Scalar, boolean isSingleRow) {
    // TODO: needs null checking, as null timestamps can be None
    StringBuilder diffExpr = new StringBuilder();
    boolean allArgsScalar = arg0Scalar && arg1Scalar || isSingleRow;
    if (allArgsScalar) {
      diffExpr.append("bodosql.libs.generated_lib.sql_null_checking_subtraction(");
      diffExpr.append(generateFloorCall(arg0, true));
      diffExpr.append(", ");
      diffExpr.append(generateFloorCall(arg1, true));
      diffExpr.append(")");
    } else {
      diffExpr.append("(");
      diffExpr.append(generateFloorCall(arg0, arg0Scalar));
      diffExpr.append(" - ");
      diffExpr.append(generateFloorCall(arg1, arg1Scalar));
      diffExpr.append(")");
    }

    return generateDaysCall(diffExpr.toString(), allArgsScalar);
  }

  /**
   * Function that returns the generated floor code for various exprTypes.
   *
   * @param arg The code for the expr that will have the floor operation called on it.
   * @param isScalar Generate scalar code.
   * @return The code generated given the arg and exprType.
   */
  public static String generateFloorCall(String arg, boolean isScalar) {
    StringBuilder floorBuilder = new StringBuilder();
    if (isScalar) {
      floorBuilder
          .append("bodosql.libs.generated_lib.sql_null_checking_timestamp_dayfloor(")
          .append(arg)
          .append(")");
    } else {
      floorBuilder.append(arg).append(".dt.floor(freq='D')");
    }
    return floorBuilder.toString();
  }

  /**
   * Function that returns the generated days code for various exprTypes.
   *
   * @param isScalar Generate scalar code.
   * @return The code generated for the given exprType.
   */
  public static String generateDaysCall(String expr, boolean isScalar) {
    // TODO: needs null checking, as null timestamps can be None
    if (isScalar) {
      return "bodosql.libs.generated_lib.sql_null_checking_pd_timedelta_days(" + expr + ")";
    } else {
      return expr + ".dt.days";
    }
  }

  /**
   * Function that returns the generated name for a DateDiff Function Call.
   *
   * @param arg0Name The first arg's name.
   * @param arg1Name The second arg's name.
   * @return The name generated that matches the DateDiff expression.
   */
  public static String generateDateDiffName(String arg0Name, String arg1Name) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("DATEDIFF(").append(arg0Name).append(", ").append(arg1Name).append(")");
    return nameBuilder.toString();
  }
}
