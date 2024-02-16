package com.bodosql.calcite.application.operatorTables;

import java.util.Locale;

// Class that contains code shared by several datetime function defintions
// (Currently just the DateTimePart Enum)
public class DatetimeFnUtils {

  // Enum to describe symbols used by Functions that use date/time part symbols (Ex. DateAdd and
  // TimeAdd)
  public enum DateTimePart {
    // Time,
    HOUR("HOUR"),
    MINUTE("MINUTE"),
    SECOND("SECOND"),
    MILLISECOND("MILLISECOND"),
    MICROSECOND("MICROSECOND"),
    NANOSECOND("NANOSECOND"),
    EPOCH_SECOND("EPOCH_SECOND"),
    EPOCH_MILLISECOND("EPOCH_MILLISECOND"),
    EPOCH_MICROSECOND("EPOCH_MICROSECOND"),
    EPOCH_NANOSECOND("EPOCH_NANOSECOND"),
    TIMEZONE_HOUR("TIMEZONE_HOUR"),
    TIMEZONE_MINUTE("TIMEZONE_MINUTE"),

    // Date
    YEAR("YEAR"),
    MONTH("MONTH"),
    DAY("DAY"),
    DAYOFWEEK("DAYOFWEEK"),
    DAYOFWEEKISO("DAYOFWEEKISO"),
    DAYOFYEAR("DAYOFYEAR"),
    WEEK("WEEK"),
    WEEKISO("WEEKISO"),
    QUARTER("QUARTER"),
    YEAROFWEEK("YEAROFWEEK"),
    YEAROFWEEKISO("YEAROFWEEKISO"),
    ;

    private final String label;

    private DateTimePart(String label) {
      this.label = label;
    }

    public static DateTimePart FromString(String inString) {
      switch (inString.toUpperCase(Locale.ROOT)) {
        case "HOUR":
        case "H":
        case "HH":
        case "HR":
        case "HOURS":
        case "HRS":
          return DateTimePart.HOUR;
        case "MINUTE":
        case "M":
        case "MI":
        case "MIN":
        case "MINUTES":
        case "MINS":
          return DateTimePart.MINUTE;
        case "SECOND":
        case "S":
        case "SEC":
        case "SECS":
        case "SECONDS":
          return DateTimePart.SECOND;
        case "MS":
        case "MSEC":
        case "MILLISECOND":
        case "MILLISECONDS":
          return DateTimePart.MILLISECOND;
        case "US":
        case "USEC":
        case "MICROSECOND":
        case "MICROSECONDS":
          return DateTimePart.MICROSECOND;
        case "NANOSECOND":
        case "NS":
        case "NSEC":
        case "NANOSEC":
        case "NSECOND":
        case "NANOSECONDS":
        case "NANOSECS":
        case "NSECONDS":
          return DateTimePart.NANOSECOND;
        case "EPOCH_SECOND":
        case "EPOCH":
        case "EPOCH_SECONDS":
          return DateTimePart.EPOCH_SECOND;
        case "EPOCH_MILLISECOND":
        case "EPOCH_MILLISECONDS":
          return DateTimePart.EPOCH_MILLISECOND;
        case "EPOCH_MICROSECOND":
        case "EPOCH_MICROSECONDS":
          return DateTimePart.EPOCH_MICROSECOND;
        case "EPOCH_NANOSECOND":
        case "EPOCH_NANOSECONDS":
          return DateTimePart.EPOCH_NANOSECOND;
        case "TIMEZONE_HOUR":
        case "TZH":
          return DateTimePart.TIMEZONE_HOUR;
        case "TIMEZONE_MINUTE":
        case "TZM":
          return DateTimePart.TIMEZONE_MINUTE;
          // DATE START
        case "YEAR":
        case "Y":
        case "YY":
        case "YYY":
        case "YYYY":
        case "YR":
        case "YEARS":
        case "YRS":
          return DateTimePart.YEAR;
        case "MONTH":
        case "MM":
        case "MON":
        case "MONS":
        case "MONTHS":
          return DateTimePart.MONTH;
        case "DAY":
        case "D":
        case "DD":
        case "DAYS":
        case "DAYOFMONTH":
          return DateTimePart.DAY;
        case "DAYOFWEEK":
        case "WEEKDAY":
        case "DOW":
        case "DW":
          return DateTimePart.DAYOFWEEK;
        case "DAYOFWEEKISO":
        case "WEEKDAY_ISO":
        case "DOW_ISO":
        case "DW_ISO":
          return DateTimePart.DAYOFWEEKISO;
        case "DAYOFYEAR":
        case "YEARDAY":
        case "DOY":
        case "DY":
          return DateTimePart.DAYOFYEAR;
        case "WEEK":
        case "W":
        case "WK":
        case "WEEKOFYEAR":
        case "WOY":
        case "WY":
          return DateTimePart.WEEK;
        case "WEEKISO":
        case "WEEK_ISO":
        case "WEEKOFYEARISO":
        case "WEEKOFYEAR_ISO":
          return DateTimePart.WEEKISO;
        case "QUARTER":
        case "Q":
        case "QTR":
        case "QTRS":
        case "QUARTERS":
          return DateTimePart.QUARTER;
        case "YEAROFWEEK":
          return DateTimePart.YEAROFWEEK;
        case "YEAROFWEEKISO":
          return DateTimePart.YEAROFWEEKISO;
        default:
          throw new RuntimeException(
              "Unable to construct Symbol from the following string: " + inString);
      }
    }

    public String ToString() {
      return this.label;
    }
  }
}
