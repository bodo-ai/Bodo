package com.bodosql.calcite.application.SQLToPython;

import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;
import static java.util.Map.entry;

import java.util.Map;

/**
 * Class that contains a series of static helper functions for parsing SQL regular expressions into
 * Python regular expressions.
 */
public class RegexHelpers {
  // Python special character map
  static Map<Character, String> escapedChars =
      Map.ofEntries(
          entry('(', "\\("),
          entry(')', "\\)"),
          entry('[', "\\["),
          entry(']', "\\]"),
          entry('{', "\\{"),
          entry('}', "\\}"),
          entry('?', "\\?"),
          entry('*', "\\*"),
          entry('+', "\\+"),
          entry('-', "\\-"),
          entry('|', "\\|"),
          entry('^', "\\^"),
          entry('$', "\\$"),
          entry('\\', "\\\\"),
          entry('.', "\\."),
          entry('&', "\\&"),
          entry('~', "\\~"),
          entry('#', "\\#"),
          entry(' ', "\\ "),
          entry('\t', "\\\t"),
          entry('\n', "\\\n"),
          entry('\r', "\\\r"),
          // Java doesn't support \v as a special character.
          // Use the ascii value 11
          entry('\u000B', "\\u000B"),
          entry('\f', "\\\f"));
  // SQL Wildcard Map
  static Map<Character, String> sqlWildcardMap = Map.ofEntries(entry('%', ".*"), entry('_', "."));

  /**
   * Function that determines the number of SQL wildcards in a provided pattern.
   *
   * @param SQLRegex String containing a SQL regular expression.
   * @return The number of wildcard characters in the regular expression.
   */
  public static int getNumSQLWildcards(String SQLRegex) {
    int count = 0;
    for (int i = 0; i < SQLRegex.length(); i++) {
      char c = SQLRegex.charAt(i);
      if ((c == '%') || (c == '_')) {
        count++;
      }
    }
    return count;
  }

  /**
   * Function that takes a SQL String and trims any leading or trailing %. All string literals start
   * and end with ', so we skip the first and last value.
   *
   * @param SQLRegex The original SQL pattern that start and ends with '
   * @return The SQL pattern with any % at the front or end removed.
   */
  public static String trimPercentWildcard(String SQLRegex) {
    int startIndex = 1;
    int endIndex = SQLRegex.length() - 1;
    while ((startIndex < SQLRegex.length()) && SQLRegex.charAt(startIndex) == '%') {
      startIndex++;
    }
    while ((endIndex > startIndex) && SQLRegex.charAt(endIndex - 1) == '%') {
      endIndex--;
    }
    // Protect against a possible Index out of bounds exception
    // This code shouldn't be necessary.
    if (startIndex > endIndex) {
      return makeQuoted("");
    }
    return makeQuoted(SQLRegex.substring(startIndex, endIndex));
  }

  /**
   * Function that converts a SQL Regex to a Python regex. It implements Python's re.escape as part
   * of the implementation. Relevant source code:
   * https://github.com/python/cpython/blob/adef445dc34685648bd0ea1c125df2ef143912ed/Lib/re.py#L245
   *
   * @param SQLRegex A SQL regular expression that main contain characters that acts as wildcards in
   *     Python. This expression begins and ends with ", which we skip
   * @param startPercent Did the original expr begin with a %
   * @param endPercent Did the original expr end with a %
   * @return The regular expression with all Python wildcards escaped.
   */
  public static String convertSQLRegexToPython(
      String SQLRegex, boolean startPercent, boolean endPercent) {
    // Final Python Regex
    StringBuilder pythonRegex = new StringBuilder();
    pythonRegex.append('"');
    // If the expr didn't start with %, add a ^
    if (!startPercent) {
      pythonRegex.append('^');
    }
    int startIndex = 1;
    for (int endIndex = 1; endIndex < SQLRegex.length() - 1; endIndex++) {
      char c = SQLRegex.charAt(endIndex);
      if (sqlWildcardMap.containsKey(c) || escapedChars.containsKey(c)) {
        // If the last character wasn't a special character, append a section of the string
        pythonRegex.append(SQLRegex, startIndex, endIndex);
        if (sqlWildcardMap.containsKey(c)) {
          pythonRegex.append(sqlWildcardMap.get(c));
        } else {
          pythonRegex.append(escapedChars.get(c));
        }
        // Skip the current character
        startIndex = endIndex + 1;
      }
    }
    /* If startIndex is not the end of the string, append the
    last section. */
    if (startIndex < SQLRegex.length()) {
      pythonRegex.append(SQLRegex, startIndex, SQLRegex.length() - 1);
    }
    // If the expr didn't end with %, add a $
    if (!endPercent) {
      pythonRegex.append('$');
    }
    pythonRegex.append('"');
    return pythonRegex.toString();
  }
}
