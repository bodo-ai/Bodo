package com.bodosql.calcite.application.BodoSQLCodeGen;

/**
 * Class that returns the generated code for a DateSub expression after all inputs have been
 * visited.
 */
public class DateSubCodeGen {
  /**
   * Function that return the necessary generated code for a DateSub Function Call.
   *
   * @param arg0 The first arg expr.
   * @param arg1 The second arg expr.
   * @param generateScalarCode Should scalar code be generated
   * @return The code generated that matches the DateSub expression.
   */
  public static String generateDateSubCode(String arg0, String arg1, boolean generateScalarCode) {
    // TODO: needs null checking, as null timestamps can be None
    StringBuilder subBuilder = new StringBuilder();
    if (generateScalarCode) {
      subBuilder
          .append("bodosql.libs.generated_lib.sql_null_checking_subtraction(")
          .append(arg0)
          .append(", ")
          .append(arg1)
          .append(")");
    } else {
      subBuilder.append("(").append(arg0).append(" - ").append(arg1).append(")");
    }
    return subBuilder.toString();
  }

  /**
   * Function that returns the generated name for a DateSub Function Call.
   *
   * @param arg0Name The first arg's name.
   * @param arg1Name The second arg's name.
   * @return The name generated that matches the DateSub expression.
   */
  public static String generateDateSubName(String arg0Name, String arg1Name) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("DATE_SUB(").append(arg0Name).append(", ").append(arg1Name).append(")");
    return nameBuilder.toString();
  }
}
