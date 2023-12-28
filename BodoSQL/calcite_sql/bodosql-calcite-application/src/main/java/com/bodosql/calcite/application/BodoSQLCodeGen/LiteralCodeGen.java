package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Expr.DecimalLiteral;
import com.google.common.collect.Range;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.GregorianCalendar;
import java.util.Iterator;
import java.util.List;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.NlsString;
import org.apache.calcite.util.Sarg;

/** Class that returns the generated code for Literal Code after all inputs have been visited. */
public class LiteralCodeGen {
  /**
   * Function that return the necessary generated code for Literals.
   *
   * @param node The RexLiteral node.
   * @param isSingleRow flag for if table references refer to a single row or the whole table. This
   *     is used for determining if an expr returns a scalar or a column. Only CASE statements set
   *     this to True currently.
   * @param visitor The PandasCodeGenVisitor class. This is used to lower certain values as globals
   *     (only in the case that isSingleRow is false, we cannot lower globals within case
   *     statements)
   * @return The code generated that matches the Literal.
   */
  public static Expr generateLiteralCode(
      RexLiteral node, boolean isSingleRow, PandasCodeGenVisitor visitor) {
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

        if (isSingleRow) {
          // note that we can't lower this as a global, if we are inside a case statement,
          // because bodosql_case_placeholder doesn't have the same global state as
          // the rest of the main generated code, and calling pd.array directly in bodo jit code
          // causes issues.
          // We should never reach this case since we disallow it in visitInternalOp,
          // but we throw an error here just in case.
          throw new BodoSQLCodegenException(
              "Internal Error: Attempted to generate a Sarg literal within a case statement.");
        } else {
          return visitor.lowerAsGlobal(arrayExpr);
        }

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
          case DECIMAL:
            return new Expr.Raw("np.float64(" + node.getValue().toString() + ")");
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
              // TODO: How do we represent microseconds and nanoseconds?
              long nanoseconds = calendar.getTimeInMillis() * 1000 * 1000;
              return new Expr.Raw(String.format("pd.Timestamp(%d)", nanoseconds));
            }
          case TIME:
            {
              // TODO: How do we represent microseconds and nanoseconds?
              int totalMilliseconds = node.getValueAs(Integer.class);
              int hour = (totalMilliseconds / (1000 * 60 * 60)) % 24;
              int minute = (totalMilliseconds / (1000 * 60)) % 60;
              int second = (totalMilliseconds / 1000) % 60;
              int millisecond = totalMilliseconds % 1000;
              return new Expr.Raw(
                  String.format(
                      "bodo.Time(%d, %d, %d, %d, 0, 0, %d)",
                      hour, minute, second, millisecond, node.getType().getPrecision()));
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
