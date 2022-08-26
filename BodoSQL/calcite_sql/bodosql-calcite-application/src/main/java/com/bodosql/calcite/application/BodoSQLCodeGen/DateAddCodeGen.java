package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.CastCodeGen.generateCastCode;

import org.apache.calcite.sql.type.SqlTypeName;

/**
 * Class that returns the generated code for a DateAdd expression after all inputs have been
 * visited.
 */
public class DateAddCodeGen {
  /**
   * Function that return the necessary generated code for a DateAdd Function Call.
   *
   * @param arg0 The first arg expr.
   * @param arg1 The second arg expr.
   * @param generateScalarCode Should scalar code be generated
   * @param strNeedsCast Is arg0 a string that needs casting.
   * @return The code generated that matches the DateAdd expression.
   */
  public static String generateDateAddCode(
      String arg0, String arg1, boolean generateScalarCode, boolean strNeedsCast) {
    // Note: Null handling is supported by Bodo/Pandas behavior
    // TODO: Only in the case that timestamp NULLS == NaN
    StringBuilder addBuilder = new StringBuilder();
    if (strNeedsCast) {
      arg0 = generateCastCode(arg0, SqlTypeName.TIMESTAMP, generateScalarCode);
    }
    if (generateScalarCode) {
      addBuilder
          .append("bodosql.libs.generated_lib.sql_null_checking_addition(")
          .append(arg0)
          .append(", ")
          .append(arg1)
          .append(")");
    } else {
      addBuilder.append("(pd.Series(").append(arg0).append(") + ").append(arg1).append(").values");
    }

    return addBuilder.toString();
  }

  /**
   * Function that returns the generated name for a DateAdd Function Call.
   *
   * @param arg0Name The first arg's name.
   * @param arg1Name The second arg's name.
   * @return The name generated that matches the DateAdd expression.
   */
  public static String generateDateAddName(String arg0Name, String arg1Name) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder.append("DATE_ADD(").append(arg0Name).append(", ").append(arg1Name).append(")");
    return nameBuilder.toString();
  }
}
