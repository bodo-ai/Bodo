package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.utils.Utils.makeQuoted;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.ExprKt;
import java.util.ArrayList;
import java.util.List;

/** Class that returns the generated code for Extract after all inputs have been visited. */
public class ExtractCodeGen {

  // List of units that are unsupported for TIME inputs
  public static List<String> dayPlusUnits;

  static {
    dayPlusUnits = new ArrayList<>();
    dayPlusUnits.add("YEAR");
    dayPlusUnits.add("QUARTER");
    dayPlusUnits.add("MONTH");
    dayPlusUnits.add("WEEK");
    dayPlusUnits.add("WEEKOFYEAR");
    dayPlusUnits.add("WEEKISO");
    dayPlusUnits.add("DAY");
    dayPlusUnits.add("DAYOFMONTH");
    dayPlusUnits.add("DOY");
    dayPlusUnits.add("DAYOFYEAR");
    dayPlusUnits.add("DOW");
    dayPlusUnits.add("DAYOFWEEK");
    dayPlusUnits.add("DAYOFWEEKISO");
  }

  // List of units that are only supported for datetime inputs, not date or time.
  public static List<String> timestampOnlyUnits;

  static {
    timestampOnlyUnits = new ArrayList<>();
    timestampOnlyUnits.add("TIMEZONE_HOUR");
    timestampOnlyUnits.add("TIMEZONE_MINUTE");
    timestampOnlyUnits.add("EPOCH_SECOND");
    timestampOnlyUnits.add("EPOCH_MILLISECOND");
    timestampOnlyUnits.add("EPOCH_MICROSECOND");
    timestampOnlyUnits.add("EPOCH_NANOSECOND");
  }

  /**
   * Function that return the necessary generated code for an Extract call.
   *
   * @param datetimeVal The arg expr for selecting which datetime field to extract. This must be a
   *     constant string.
   * @param column The column arg expr.
   * @param isTime Is the input TIME data?
   * @return The code generated that matches the Extract expression.
   */
  public static Expr generateExtractCode(
      String datetimeVal,
      Expr column,
      boolean isTime,
      boolean isDate,
      Integer weekStart,
      Integer weekOfYearPolicy) {
    if ((isDate || isTime) && timestampOnlyUnits.contains(datetimeVal)) {
      throw new BodoSQLCodegenException(
          "To extract unit " + datetimeVal + " requires TIMESTAMP values");
    }
    if (isTime && dayPlusUnits.contains(datetimeVal)) {
      throw new BodoSQLCodegenException("Cannot extract unit " + datetimeVal + " from TIME values");
    }
    if (isDate && !dayPlusUnits.contains(datetimeVal)) {
      throw new BodoSQLCodegenException("Cannot extract unit " + datetimeVal + " from DATE values");
    }
    String kernelName;
    List<Expr> args = new ArrayList<>();
    args.add(column);
    switch (datetimeVal) {
      case "NANOSECOND":
      case "MICROSECOND":
      case "MILLISECOND":
      case "SECOND":
      case "MINUTE":
      case "HOUR":
      case "MONTH":
      case "QUARTER":
      case "YEAR":
        kernelName = "get_" + datetimeVal.toLowerCase();
        break;
      case "DAY":
      case "DAYOFMONTH":
        kernelName = "dayofmonth";
        break;
      case "DAYOFWEEKISO":
        kernelName = "dayofweekiso";
        break;
      case "DOY":
      case "DAYOFYEAR":
        kernelName = "dayofyear";
        break;
      case "DOW":
      case "DAYOFWEEK":
        kernelName = "dayofweek";
        args.add(new Expr.IntegerLiteral(weekStart));
        break;
      case "WEEK":
        kernelName = "week";
        args.add(new Expr.IntegerLiteral(weekStart));
        args.add(new Expr.IntegerLiteral(weekOfYearPolicy));
        break;
      case "WEEKOFYEAR":
        kernelName = "weekofyear";
        args.add(new Expr.IntegerLiteral(weekStart));
        args.add(new Expr.IntegerLiteral(weekOfYearPolicy));
        break;
      case "WEEKISO":
        kernelName = "get_weekofyear";
        break;
      case "TIMEZONE_HOUR":
        kernelName = "get_timezone_offset";
        args.add(new Expr.StringLiteral("hr"));
        break;
      case "TIMEZONE_MINUTE":
        kernelName = "get_timezone_offset";
        args.add(new Expr.StringLiteral("min"));
        break;
      case "EPOCH_SECOND":
        kernelName = "get_epoch";
        args.add(new Expr.StringLiteral("s"));
        break;
      case "EPOCH_MILLISECOND":
        kernelName = "get_epoch";
        args.add(new Expr.StringLiteral("ms"));
        break;
      case "EPOCH_MICROSECOND":
        kernelName = "get_epoch";
        args.add(new Expr.StringLiteral("us"));
        break;
      case "EPOCH_NANOSECOND":
        kernelName = "get_epoch";
        args.add(new Expr.StringLiteral("ns"));
        break;
      default:
        throw new BodoSQLCodegenException(
            "ERROR, datetime value: " + datetimeVal + " not supported inside of extract");
    }
    return ExprKt.bodoSQLKernel(kernelName, args, List.of());
  }

  /**
   * Returns the Expr for DATE_PART by mapping the string literals to the same code gen as EXTRACT
   *
   * @param operandsInfo The information about the arguments to the call
   * @param isTime Is the input TIME data?
   * @return The name generated that matches the Extract expression.
   */
  public static Expr generateDatePart(
      List<Expr> operandsInfo,
      boolean isTime,
      boolean isDate,
      int weekStart,
      int weekOfYearPolicy) {
    String unit = makeQuoted(operandsInfo.get(0).emit().toLowerCase());
    switch (unit) {
      case "\"year\"":
      case "\"y\"":
      case "\"yy\"":
      case "\"yyy\"":
      case "\"yyyy\"":
      case "\"yr\"":
      case "\"years\"":
      case "\"yrs\"":
        unit = "YEAR";
        break;

      case "\"month\"":
      case "\"mm\"":
      case "\"mon\"":
      case "\"mons\"":
      case "\"months\"":
        unit = "MONTH";
        break;

      case "\"day\"":
      case "\"d\"":
      case "\"dd\"":
      case "\"days\"":
      case "\"dayofmonth\"":
        unit = "DAY";
        break;

      case "\"dayofweek\"":
      case "\"weekday\"":
      case "\"dow\"":
      case "\"dw\"":
        unit = "DAYOFWEEK";
        break;

      case "\"dayofyear\"":
      case "\"yearday\"":
      case "\"doy\"":
      case "\"dy\"":
        unit = "DAYOFYEAR";
        break;

      case "\"week\"":
      case "\"w\"":
      case "\"wk\"":
      case "\"weekofyear\"":
      case "\"woy\"":
      case "\"wy\"":
        unit = "WEEK";
        break;

      case "\"weekiso\"":
      case "\"week_iso\"":
      case "\"weekofyeariso\"":
      case "\"weekofyear_iso\"":
        unit = "WEEKISO";
        break;

      case "\"quarter\"":
      case "\"q\"":
      case "\"qtr\"":
      case "\"qtrs\"":
      case "\"quarters\"":
        unit = "QUARTER";
        break;

      case "\"hour\"":
      case "\"h\"":
      case "\"hh\"":
      case "\"hr\"":
      case "\"hours\"":
      case "\"hrs\"":
        unit = "HOUR";
        break;

      case "\"minute\"":
      case "\"m\"":
      case "\"mi\"":
      case "\"min\"":
      case "\"minutes\"":
      case "\"mins\"":
        unit = "MINUTE";
        break;

      case "\"second\"":
      case "\"s\"":
      case "\"sec\"":
      case "\"seconds\"":
      case "\"secs\"":
        unit = "SECOND";
        break;

      case "\"nanosecond\"":
      case "\"ns\"":
      case "\"nsec\"":
      case "\"nanosec\"":
      case "\"nsecond\"":
      case "\"nanoseconds\"":
      case "\"nanosecs\"":
      case "\"nseconds\"":
        unit = "NANOSECOND";
        break;

      default:
        throw new BodoSQLCodegenException(
            "Unsupported DATE_PART unit: " + operandsInfo.get(0).emit());
    }
    return generateExtractCode(
        unit, operandsInfo.get(1), isTime, isDate, weekStart, weekOfYearPolicy);
  }
}
