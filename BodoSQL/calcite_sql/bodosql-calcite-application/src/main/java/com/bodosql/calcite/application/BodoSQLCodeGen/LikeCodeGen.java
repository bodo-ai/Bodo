package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.SQLToPython.RegexHelpers.*;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

/**
 * Class that returns the generated code for a like expression after all inputs have been visited.
 */
public class LikeCodeGen {
  /**
   * Function that return the necessary generated code for a like operator. This generates different
   * code depending on if the SQLRegex is a string known at compile time or an expression.
   *
   * @param expr The expression that needs to be compared to the regex.
   * @param SQLRegex SQL Regular Expression. This is either a literal string or an expression that
   *     will be evaluated at runtime. In either case the SQLRegex must be converted to a Python
   *     regular expression.
   * @param isLiteral Is the SQL expression known at compile time. If so we can optimize the code
   *     generated.
   * @param outputScalar Should the output generate scalar code.
   * @return The code generated that matches the Like expression.
   */
  public static String generateLikeCode(
      String expr,
      String SQLRegex,
      boolean isLiteral,
      boolean patternIsRegex,
      boolean outputScalar) {
    StringBuilder likeParts = new StringBuilder();
    // TODO: can the SQL Regex be null? I assume if it is, we can say it's not allowed
    if (isLiteral) {
      /* If the SQL Regex doesn't contain a SQL wildcard we can convert the
      value to an equality check. */
      int numWildcards = getNumSQLWildcards(SQLRegex);
      if (numWildcards == 0) {
        // Handles RLIKE and REGEXP here
        if (patternIsRegex) {
          likeParts.append("bodo.libs.bodosql_array_kernels.regexp_like(").append(expr);
          likeParts.append(", ").append(SQLRegex);
          likeParts.append(", ").append(makeQuoted("")).append(")");
        } else {
          likeParts.append(expr).append(" == ").append(SQLRegex);
        }
      } else {
        int prevLen = SQLRegex.length();
        String trimmedRegex = trimPercentWildcard(SQLRegex);
        // Determine if the expression started and ended with %
        boolean startPercent = SQLRegex.charAt(1) == '%';
        boolean endPercent = SQLRegex.charAt(prevLen - 2) == '%';
        /* If all the wildcards were at the front or end, generate special code without regular expressions. */
        if (prevLen == (trimmedRegex.length() + numWildcards)) {
          // If both start and end had %, use contains with regex=False
          if (startPercent && endPercent) {
            if (outputScalar) {
              likeParts
                  .append("bodosql.libs.generated_lib.sql_null_checking_in(")
                  .append(trimmedRegex)
                  .append(", ")
                  .append(expr)
                  .append(")");
            } else {
              likeParts
                  .append("pd.Series(")
                  .append(expr)
                  .append(").str.contains(")
                  .append(trimmedRegex)
                  .append(", regex=False).values");
            }
          } else if (startPercent) {
            // If only start, use endswith
            if (outputScalar) {
              // Scalar doesn't have an optimized expr
              likeParts
                  .append("bodosql.libs.generated_lib.sql_null_checking_re_match(")
                  .append(convertSQLRegexToPython(trimmedRegex, startPercent, endPercent))
                  .append(", ")
                  .append(expr)
                  .append(")");
            } else {
              likeParts
                  .append("pd.Series(")
                  .append(expr)
                  .append(").str.endswith(")
                  .append(trimmedRegex)
                  .append(").values");
            }
          } else {
            if (outputScalar) {
              // Scalar doesn't have an optimized expr
              likeParts
                  .append("bodosql.libs.generated_lib.sql_null_checking_re_match(")
                  .append(convertSQLRegexToPython(trimmedRegex, startPercent, endPercent))
                  .append(", ")
                  .append(expr)
                  .append(")");
            } else {
              likeParts
                  .append("pd.Series(")
                  .append(expr)
                  .append(").str.startswith(")
                  .append(trimmedRegex)
                  .append(").values");
            }
          }
        } else {
          if (outputScalar) {
            likeParts
                .append("bodosql.libs.generated_lib.sql_null_checking_re_match(")
                .append(convertSQLRegexToPython(trimmedRegex, startPercent, endPercent))
                .append(", ")
                .append(expr)
                .append(")");
          } else {
            likeParts
                .append("pd.Series(")
                .append(expr)
                .append(").str.contains(")
                .append(convertSQLRegexToPython(trimmedRegex, startPercent, endPercent))
                .append(").values");
          }
        }
      }
    } else {
      if (outputScalar) {
        likeParts
            .append(
                "bodosql.libs.generated_lib.sql_null_checking_re_match(bodosql.libs.generated_lib.sql_null_checking_sql_to_python(")
            .append(SQLRegex)
            .append("), ")
            .append(expr)
            .append(")");
      } else {
        likeParts
            .append("pd.Series(")
            .append(expr)
            .append(").str.contains(bodosql.libs.generated_lib.sql_null_checking_sql_to_python(")
            .append(SQLRegex)
            .append(")).values");
      }
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
  public static String generateLikeName(String argName, String SQLRegex) {
    StringBuilder nameBuilder = new StringBuilder();
    nameBuilder
        .append("LIKE(")
        .append(argName)
        .append(", ")
        .append("(")
        .append(SQLRegex)
        .append("))");
    return nameBuilder.toString();
  }
}
