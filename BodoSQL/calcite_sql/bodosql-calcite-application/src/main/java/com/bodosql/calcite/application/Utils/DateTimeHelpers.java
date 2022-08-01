package com.bodosql.calcite.application.Utils;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.*;

public class DateTimeHelpers {

  // Hashmap of valid mysql formatters, to their python equivalents
  static HashMap<Character, String> dtFormatMap;
  // hashset of valid mysql formatters, that are currently unsupported
  static HashSet<Character> unsupportedFormatters;
  // hashset of valid mysql formatters, that are currently unsupported
  static HashSet<Character> pythonFormatters;

  static {
    dtFormatMap = new HashMap<>();
    unsupportedFormatters = new HashSet<>();
    // minutes, 00 to 59
    dtFormatMap.put('i', "%M");
    // Month, full name
    dtFormatMap.put('M', "%B");
    // time in format (hh:mm:ss AM/PM)
    dtFormatMap.put('r', "%X %p");
    // seconds 00 to 59
    dtFormatMap.put('s', "%S");
    // time in format (hh:mm:ss)
    dtFormatMap.put('T', "%X");
    // week where monday is the first day of the week
    dtFormatMap.put('u', "%W");
    // Abbreviated weekday name (sun-sat)
    dtFormatMap.put('a', "%a");
    // Abbreviated month name (jan-dec)
    dtFormatMap.put('b', "%b");
    // ms, left padded with 0's, (000000 to 999999)
    dtFormatMap.put('f', "%f");
    // Hour, 00 to 23
    dtFormatMap.put('H', "%H");
    // DOY, left padded with 0's
    dtFormatMap.put('j', "%j");
    // month as numeric, 00 to 12
    dtFormatMap.put('m', "%m");
    // AM or PM
    dtFormatMap.put('p', "%p");
    // day of month as numeric value (01 to 31)
    dtFormatMap.put('d', "%d");
    // Year as a 4 digit value
    dtFormatMap.put('Y', "%Y");
    // Year as a 2 digit value (0 padded)
    dtFormatMap.put('y', "%y");
    // week where sunday is the first day of the week
    dtFormatMap.put('U', "%U");
    // seconds 00 to 59
    dtFormatMap.put('S', "%S");
    // percent literal in mysql should turn into a python percent literal
    dtFormatMap.put('%', "%%");

    // TODO: These format strings don't have a 1 to one equivalent in python
    // week where sunday is the first day of the week used with %x?
    unsupportedFormatters.add('V');
    // week where monday is the first day of the week used with %x?
    unsupportedFormatters.add('v');
    // Year for the week where Sunday is the first day of the week. Used with %V
    unsupportedFormatters.add('X');
    // Year for the week where Monday is the first day of the week. Used with %v
    unsupportedFormatters.add('x');
    // Hour, 00 to 12
    unsupportedFormatters.add('h');
    unsupportedFormatters.add('i');
    // Hour, 0 to 23
    unsupportedFormatters.add('k');
    // Hour, 1 to 23
    unsupportedFormatters.add('l');
    // numeric month name (1, 2, 3), not 0 padded
    unsupportedFormatters.add('c');
    // day of month followed by suffix (1st, 2nd..)
    unsupportedFormatters.add('D');
    // day of month as numeric value (0 to 31)
    unsupportedFormatters.add('e');

    // Set of all valid python formatters
    // found here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    // pythonFormatters.addAll(Arrays.asList('a', 'A', 'w', 'd', 'b', 'B', 'm', 'y', 'Y', 'H', 'I',
    // 'p', 'M', 'S', 'f', 'z', 'j', 'U', 'W', 'c', 'x', 'X', 'G', 'u', 'V'));
  }

  public static String convertMySQLFormatStringToPython(String mySQLFormatStr) {
    StringBuilder pythonFormatStr = new StringBuilder();

    for (int curIndex = 0; curIndex < mySQLFormatStr.length(); curIndex++) {
      char curChar = mySQLFormatStr.charAt(curIndex);
      // need -2 to insure that ...%" => ...%".
      if (curChar == '%' && curIndex != mySQLFormatStr.length() - 2) {
        curIndex++;
        char formatChar = mySQLFormatStr.charAt(curIndex);

        if (dtFormatMap.containsKey(formatChar)) {
          pythonFormatStr.append(dtFormatMap.get(formatChar));
        } else if (unsupportedFormatters.contains(formatChar)) {
          throw new BodoSQLCodegenException(
              "Error, formatting character: " + formatChar + "Not supported");
        } else {
          // if it's not a formatting char, we omit the % symbol
          pythonFormatStr.append(formatChar);
        }
      } else {
        pythonFormatStr.append(curChar);
      }
    }
    return pythonFormatStr.toString();
  }

  /**
   * helper function to determine if a string expression represents a string literal
   *
   * @param s the string expression
   * @return Does the expression represent a string literal
   */
  public static boolean isStringLiteral(String s) {
    return s.charAt(0) == '"' && s.charAt(s.length() - 1) == '"';
  }

  /**
   * Helper function to extract the underlying value for a String Literal.
   *
   * @param s the string expression
   * @return Does the expression represent a string literal
   */
  public static String getStringLiteralValue(String s) {
    return s.substring(1, s.length() - 1);
  }
}
