package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.DecimalLiteral;
import com.google.common.collect.Range;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import kotlin.Pair;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.NlsString;
import org.apache.calcite.util.Sarg;
import org.apache.calcite.util.TimeString;
import org.apache.calcite.util.TimestampString;
import org.apache.calcite.util.TimestampWithTimeZoneString;

/** Class that returns the generated code for Literal Code after all inputs have been visited. */
public class LiteralCodeGen {

  /**
   * Extract the total number of nanoseconds in a Time or Timestamp node that represent the total
   * amount of sub-millisecond time. e.g. given a node containg a time value of 01:02:03.004005006,
   * this method would output 5006.
   *
   * @param valueType The type to use to get the value of the node. Must be either TimeString or
   *     TimestampString
   * @param node
   * @return int
   */
  private static <T> int getSubMillisecondTimeComponent(Class<T> valueType, RexLiteral node) {
    assert valueType == TimeString.class
        || valueType == TimestampString.class
        || valueType == TimestampWithTimeZoneString.class;
    // Neither TimeString nor TimestampString provide a way to access the sub-milliseconds
    // components of their value, so we need to parse the underlying string.

    int total_nanoseconds = 0;
    String rawTimeString;
    if (valueType == TimestampWithTimeZoneString.class) {
      rawTimeString =
          Objects.requireNonNull(
                  node.getValueAs(TimestampWithTimeZoneString.class).getLocalTimestampString())
              .toString();
    } else {
      rawTimeString = Objects.requireNonNull(node.getValueAs(valueType)).toString();
    }
    // Extract the sub-second component
    String[] parts = rawTimeString.split("\\.");
    if (parts.length == 2) {
      String subSeconds = parts[1];
      if (subSeconds.length() > 3) {
        // If there's anything sub nanosecond, trim it
        StringBuilder nsecString =
            new StringBuilder(subSeconds.substring(3, Math.min(9, subSeconds.length())));
        while (nsecString.length() < 6) {
          // Pad the string with 0s so that it represents the number of nanoseconds present
          nsecString.append("0");
        }
        total_nanoseconds = Integer.parseInt(nsecString.toString());
      }
    }
    return total_nanoseconds;
  }

  /**
   * Function that return the necessary generated code for Literals.
   *
   * @param node The RexLiteral node.
   * @param visitor The BodoCodeGenVisitor class. This is used to lower certain values as globals
   *     (only in the case that isSingleRow is false, we cannot lower globals within case
   *     statements)
   * @return The code generated that matches the Literal.
   */
  public static Expr generateLiteralCode(RexLiteral node, BodoCodeGenVisitor visitor) {
    SqlTypeName typeName = node.getType().getSqlTypeName();
    // TODO: Add more types here
    switch (node.getTypeName()) {
      case NULL:
        return Expr.None.INSTANCE;
      case SARG:
        List<Expr> literalList = new ArrayList<>();

        Sarg sargVal = (Sarg) node.getValue();
        Iterator<Range> iter = sargVal.rangeSet.asRanges().iterator();
        // We require the range to contain at least one value.
        // In the event that it contains no values,
        // we expect that calcite would optimize it out
        // as an always true/false operation
        assert iter.hasNext() : "Internal error: We expect sargVal to contain at least one range";
        while (iter.hasNext()) {
          Range curRange = iter.next();
          // Assert that each element of the range is scalar.
          assert curRange.hasLowerBound()
                  && curRange.hasUpperBound()
                  && curRange.upperEndpoint() == curRange.lowerEndpoint()
              : "Internal error: Attempted to convert a non-discrete sarg into a literal value.";
          Expr expr = sargValToPyLiteral(curRange.lowerEndpoint());
          literalList.add(expr);
        }
        // initialize the array, and lower it as a global

        // Note, currently, setting the dtype of this array directly can cause
        // issues in typing. So, we just let Bodo infer the type of the lowered array.
        Expr arrayExpr = new Expr.Call("pd.array", List.of(new Expr.List(literalList)));
        return visitor.lowerAsGlobal(arrayExpr);

      default:
        // TODO: investigate if this is the correct default value
        switch (typeName) {
          case TINYINT:
            return new Expr.Raw("np.int8(" + node.getValue().toString() + ")");
          case SMALLINT:
            return new Expr.Raw("np.int16(" + node.getValue().toString() + ")");
          case INTEGER:
            return new Expr.Raw("np.int32(" + node.getValue().toString() + ")");
          case BIGINT:
            return new Expr.Raw("np.int64(" + node.getValue().toString() + ")");
          case FLOAT:
            return new Expr.Raw("np.float32(" + node.getValue().toString() + ")");
          case REAL:
          case DOUBLE:
            return new Expr.Raw("np.float64(" + node.getValue().toString() + ")");
          case DECIMAL:
            if (node.getType().getScale() > 0) {
              return new Expr.Raw("np.float64(" + node.getValue().toString() + ")");
            } else {
              return new Expr.Raw("np.int64(" + node.getValue().toString() + ")");
            }
          // TODO: Determine why this case exists
          case SYMBOL:
            return new Expr.Raw(node.getValue().toString());
          case DATE:
            // TODO:[BE-4593]Stop calcite from converting DATE literal
            {
              GregorianCalendar calendar = (GregorianCalendar) node.getValue();
              int year = calendar.get(Calendar.YEAR);
              // Month is 0-indexed in GregorianCalendar
              int month = calendar.get(Calendar.MONTH) + 1;
              int day = calendar.get(Calendar.DAY_OF_MONTH);
              return new Expr.Raw(String.format("datetime.date(%d, %d, %d)", year, month, day));
            }
          case CHAR:
          case VARCHAR:
            // extract value without specific sql type info.
            return new Expr.StringLiteral(node.getValue2().toString());
          case BINARY:
          case VARBINARY:
            // extract value without specific sql type info.
            return new Expr.BinaryLiteral(node.getValue2().toString());
          case TIMESTAMP:
            {
              GregorianCalendar calendar = (GregorianCalendar) node.getValue();
              long nanoseconds = calendar.getTimeInMillis() * 1000 * 1000;
              nanoseconds += getSubMillisecondTimeComponent(TimestampString.class, node);
              return new Expr.Raw(String.format("pd.Timestamp(%d)", nanoseconds));
            }
          case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
            {
              // TZ-Aware uses a timestamp string. We can't use the integer version
              // as the value is in UTC time. We could in the future convert this to
              // the integer code path + tz_localize if that's faster, but it's probably
              // better to optimize pd.Timestamp.
              TimestampString timestampString = node.getValueAs(TimestampString.class);
              Expr argString = new Expr.StringLiteral(timestampString.toString());
              Expr tzInfo = visitor.genDefaultTZ().getZoneExpr();
              return new Expr.Call(
                  "pd.Timestamp", List.of(argString), List.of(new Pair<>("tz", tzInfo)));
            }
          case TIMESTAMP_TZ:
            {
              TimestampWithTimeZoneString timestampString =
                  node.getValueAs(TimestampWithTimeZoneString.class);
              TimestampString localString = timestampString.getLocalTimestampString();
              int minuteOffset = timestampString.getTimeZone().getRawOffset() / 60_000;
              Expr minuteOffsetExpr = new Expr.IntegerLiteral(minuteOffset);
              long nsEpoch = localString.getMillisSinceEpoch() * 1000 * 1000;
              nsEpoch += getSubMillisecondTimeComponent(TimestampWithTimeZoneString.class, node);
              Expr tsExpr = new Expr.Raw(String.format("pd.Timestamp(%d)", nsEpoch));
              return new Expr.Call(
                  "bodo.hiframes.timestamptz_ext.init_timestamptz_from_local",
                  List.of(tsExpr, minuteOffsetExpr));
            }
          case TIME:
            {
              int totalMilliseconds = node.getValueAs(Integer.class);
              int hour = (totalMilliseconds / (1000 * 60 * 60)) % 24;
              int minute = (totalMilliseconds / (1000 * 60)) % 60;
              int second = (totalMilliseconds / 1000) % 60;
              int millisecond = totalMilliseconds % 1000;

              int total_nanoseconds = getSubMillisecondTimeComponent(TimeString.class, node);
              int microsecond = total_nanoseconds / 1000;
              int nanosecond = total_nanoseconds % 1000;

              return new Expr.Raw(
                  String.format(
                      "bodo.types.Time(%d, %d, %d, %d, %d, %d, %d)",
                      hour,
                      minute,
                      second,
                      millisecond,
                      microsecond,
                      nanosecond,
                      node.getType().getPrecision()));
            }
          case BOOLEAN:
            String boolName = node.toString();
            return new Expr.Raw(boolName.substring(0, 1).toUpperCase() + boolName.substring(1));
          /* according to https://calcite.apache.org/javadocAggregate/org/apache/calcite/rex/RexLiteral.html,
          INTERVAL_YEAR/YEAR_MONTH/MONTH are measured in months, and everything else is measured in miliseconds
           */
          case INTERVAL_DAY_HOUR:
          case INTERVAL_DAY_MINUTE:
          case INTERVAL_DAY_SECOND:
          case INTERVAL_HOUR_MINUTE:
          case INTERVAL_HOUR_SECOND:
          case INTERVAL_MINUTE_SECOND:
          case INTERVAL_HOUR:
          case INTERVAL_MINUTE:
          case INTERVAL_SECOND:
          case INTERVAL_DAY:
            // Value is given in milliseconds in these cases
            BigDecimal millis = node.getValueAs(BigDecimal.class);
            long nanoseconds = millis.scaleByPowerOfTen(6).longValue();
            return new Expr.Raw("pd.Timedelta(" + nanoseconds + ")");
          case INTERVAL_YEAR:
          case INTERVAL_MONTH:
          case INTERVAL_YEAR_MONTH:
            // value is given in months
            // May later refactor this code to create DateOffsets, for now
            // causes an error
            String months = node.getValue().toString();
            return new Expr.Raw("pd.DateOffset(months=" + months + ")");
          default:
            throw new BodoSQLCodegenException(
                "Internal Error: Calcite Plan Produced an Unsupported Literal Type");
        }
    }
  }

  /**
   * Helper function that converts a scalar value obtained from a Sarg into a python friendly
   * literal.
   *
   * <p>May throw a BodoSQLCodegenException in the case that the scalar value is of an unsupported
   * type. Currently, the only supported types are NlsString and BigDecimal, see
   * https://bodo.atlassian.net/browse/BE-4046
   *
   * @param scalarVal The Sarg limit value to be converted to a python literal
   * @return A string that represents the value as a python literal
   * @param <C> The type of said literal.
   */
  public static <C extends Comparable<C>> Expr sargValToPyLiteral(C scalarVal) {
    // TODO: extend this to other types (timestamp, etc) which may need to do some
    // extra parsing/conversion

    if (scalarVal instanceof NlsString) {
      NlsString as_Nls = ((NlsString) scalarVal);
      // Call get value to get the underlying string without the surrounding 's
      // indicating a SQL literal. Expr.StringLiteral will handle string requirements.
      return new Expr.StringLiteral(as_Nls.getValue());
    } else if (scalarVal instanceof BigDecimal) {
      return new DecimalLiteral((BigDecimal) scalarVal);
    } else {
      throw new BodoSQLCodegenException(
          "Internal error: Sarg limit value cannot be converted to python literal.");
    }
  }
}
