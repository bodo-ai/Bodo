package com.bodosql.calcite.application.SQLToPython;

import static java.util.Map.entry;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

/**
 * Class that contains a series of static helper functions for parsing SQL format strings into
 * Python/Pandas format strings.
 */
public class FormatHelpers {
  /* Reference for valid SQL format strings https://www.w3schools.com/SQl/func_mysql_str_to_date.asp. */
  /* Reference for Pandas format strings https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior. */

  // Map of characters following % in SQL to their Pandas replacement.
  // We include keys that don't change to avoid escaping them.
  static Map<Character, Character> escapedChars =
      Map.ofEntries(
          entry('a', 'a'),
          entry('b', 'b'),
          entry('d', 'd'),
          entry('H', 'H'),
          entry('h', 'I'),
          entry('I', 'I'),
          entry('i', 'M'),
          entry('j', 'j'),
          entry('M', 'B'),
          entry('m', 'm'),
          entry('p', 'p'),
          entry('S', 'S'),
          entry('s', 'S'),
          entry('T', 'X'),
          entry('U', 'U'),
          entry('u', 'W'),
          entry('W', 'A'),
          entry('w', 'w'),
          entry('Y', 'Y'),
          entry('y', 'y'));

  // Set of characters following % that don't have a direct Pandas
  // conversion and therefore aren't supported yet.
  static Set<Character> unsupportedChars =
      new HashSet<>(Arrays.asList('c', 'D', 'e', 'f', 'k', 'l', 'r', 'V', 'v', 'X', 'x'));

  /**
   * Function that converts a String literal SQL format string into a Python format string that can
   * be interpretted properly by pd.ToDatetime.
   *
   * @param SQLFormatStr The format string with valid SQL syntax.
   * @return An equivalent format string that can be used by Pandas.
   */
  public static String SQLFormatToPandasToDatetimeFormat(String SQLFormatStr) {
    StringBuilder pythonFormatStr = new StringBuilder();
    int startIndex = 0;
    int endIndex = 0;
    while (endIndex < SQLFormatStr.length()) {
      char c = SQLFormatStr.charAt(endIndex);
      // If the current character is a %, we may need to replace a character.
      if (c == '%') {
        // If the last character is % we don't have a valid SQL format string.
        if ((endIndex + 1) >= SQLFormatStr.length()) {
          throw new BodoSQLCodegenException(
              "STR_TO_DATE contains an invalid format string: " + SQLFormatStr);
        } else {
          char nextChar = SQLFormatStr.charAt(endIndex + 1);
          // If the char is in escapedChars, then we need to replace
          // the format string.
          if (escapedChars.containsKey(nextChar)) {
            // Append any previous portion + '%' character.
            pythonFormatStr.append(SQLFormatStr, startIndex, endIndex + 1);
            pythonFormatStr.append(escapedChars.get(nextChar));
            endIndex += 2;
            startIndex = endIndex;
            // If the char is in unsupportedChars, then the format string is legal,
            // but BodoSQL doesn't support it.
          } else if (unsupportedChars.contains(nextChar)) {
            throw new BodoSQLCodegenException(
                "STR_TO_DATE contains an unsupported escape character %"
                    + nextChar
                    + " in format string "
                    + SQLFormatStr);
            // Otherwise we don't have a valid format string
          } else {
            throw new BodoSQLCodegenException(
                "STR_TO_DATE contains an invalid format string: " + SQLFormatStr);
          }
        }
      } else {
        endIndex++;
      }
    }
    /* If startIndex is not the end of the string, append the last section. */
    if (startIndex < SQLFormatStr.length()) {
      pythonFormatStr.append(SQLFormatStr, startIndex, SQLFormatStr.length());
    }
    return pythonFormatStr.toString();
  }
}
