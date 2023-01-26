package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

/**
 * Class that returns the generated code for a like expression after all inputs have been visited.
 */
public class LikeCodeGen {
  /**
   * Function that return the necessary generated code for a like operator. This generates different
   * code depending on if the SQLRegex is a string known at compile time or an expression.
   *
   * @param opName The name of the expression being executed.
   * @param expr The expression that needs to be compared to the regex.
   * @param SQLRegex SQL Regular Expression. This is either a literal string or an expression that
   *     will be evaluated at runtime. In either case the SQLRegex must be converted to a Python
   *     regular expression.
   * @param SQLEscape Character used to escape wildcards in like and ilike. This value is ignored by
   *     the pattern functions.
   * @return The code generated that matches the Like expression.
   */
  public static String generateLikeCode(
      String opName, String expr, String SQLRegex, String SQLEscape) {
    StringBuilder likeParts = new StringBuilder();
    boolean patternRegex = (opName.equals("REGEXP") || opName.equals("RLIKE"));
    boolean caseInsensitive = (opName.equals("ILIKE"));
    if (patternRegex) {
      // Note we ignore SQLEscape because it is not a valid argument to these functions.
      likeParts.append("bodo.libs.bodosql_array_kernels.regexp_like(").append(expr);
      likeParts.append(", ").append(SQLRegex);
      likeParts.append(", ").append(makeQuoted("")).append(")");
    } else {
      likeParts
          .append("bodo.libs.bodosql_array_kernels.like_kernel(")
          .append(expr)
          .append(", ")
          .append(SQLRegex)
          .append(", ")
          .append(SQLEscape)
          .append(", ");
      if (caseInsensitive) {
        likeParts.append("True");
      } else {
        likeParts.append("False");
      }
      likeParts.append(")");
    }
    return likeParts.toString();
  }

  /**
   * Function that returns the generated name for a Like Operation.
   *
   * @param argName The string Expression's name.
   * @param SQLRegex A name for a SQL regular expression.
   * @return The name generated that matches the Like Operation.
   */
  public static String generateLikeName(String opName, String argName, String SQLRegex) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder
        .append(opName)
        .append("(")
        .append(argName)
        .append(", ")
        .append("(")
        .append(SQLRegex)
        .append("))");
    return nameBuilder.toString();
  }
}
