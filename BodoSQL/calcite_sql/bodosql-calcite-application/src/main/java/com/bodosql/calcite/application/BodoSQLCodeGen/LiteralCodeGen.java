package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.PandasCodeGenVisitor;
import com.google.common.collect.Range;
import java.math.BigDecimal;
import java.util.*;
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
   * @param node isSingleRow flag for if table references refer to a single row or the whole table.
   *     This is used for determining if an expr returns a scalar or a column. Only CASE statements
   *     set this to True currently.
   * @param visitor The PandasCodeGenVisitor class. This is used to lower certain values as globals
   *     (only in the case that isSingleRow is false, we cannot lower globals within case
   *     statements)
   * @return The code generated that matches the Literal.
   */
  public static String generateLiteralCode(
      RexLiteral node, boolean isSingleRow, PandasCodeGenVisitor visitor) {
    StringBuilder codeBuilder = new StringBuilder();
    SqlTypeName typeName = node.getType().getSqlTypeName();
    String out = "";
    // TODO: Add more types here
    switch (node.getTypeName()) {
      case NULL:
        switch (typeName) {
          case VARCHAR:
          case CHAR:
          case VARBINARY:
          case BINARY:
          case BOOLEAN:
          case TINYINT:
          case SMALLINT:
          case INTEGER:
          case BIGINT:
          case DATE:
          case TIMESTAMP:
          case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
          case TIME:
            // TODO: Support all remaining interval types.
          case INTERVAL_WEEK:
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
          case INTERVAL_YEAR:
          case INTERVAL_MONTH:
          case INTERVAL_YEAR_MONTH:
          case NULL:
          case FLOAT:
          case REAL:
          case DOUBLE:
          case DECIMAL:
            codeBuilder.append("None");
            break;
          default:
            throw new BodoSQLCodegenException(
                "Internal Error: Calcite Plan Produced an Unsupported Null Literal Type: "
                    + typeName);
        }
        break;
      case SARG:
        StringBuilder literalList = new StringBuilder("[");

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
          String expr = sargValToPyLiteral(curRange.lowerEndpoint());
          literalList.append(expr).append(", ");
        }
        literalList.append("]");
        // initialize the array, and lower it as a global

        // Note, currently, setting the dtype of this array directly can cause
        // issues in typing. So, we just let Bodo infer the type of the lowered array.
        String arrayExpr = "pd.array(" + literalList.toString() + ")";

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
          String globalVal = visitor.lowerAsGlobal(arrayExpr);
          codeBuilder.append(globalVal);
        }
        break;

      default:
        // TODO: investigate if this is the correct default value
        switch (typeName) {
          case TINYINT:
            codeBuilder.append("np.int8(" + node.getValue().toString() + ")");
            break;
          case SMALLINT:
            codeBuilder.append("np.int16(" + node.getValue().toString() + ")");
            break;
          case INTEGER:
            codeBuilder.append("np.int32(" + node.getValue().toString() + ")");
            break;
          case BIGINT:
            codeBuilder.append("np.int64(" + node.getValue().toString() + ")");
            break;
          case FLOAT:
            codeBuilder.append("np.float32(" + node.getValue().toString() + ")");
            break;
          case REAL:
          case DOUBLE:
          case DECIMAL:
            codeBuilder.append("np.float64(" + node.getValue().toString() + ")");
            break;
            // TODO: Determine why this case exists
          case SYMBOL:
            codeBuilder.append(node.getValue().toString());
            break;
          case DATE:
            // TODO:[BE-4593]Stop calcite from converting DATE literal
            {
              GregorianCalendar calendar = (GregorianCalendar) node.getValue();
              int year = calendar.get(Calendar.YEAR);
              // Month is 0-indexed in GregorianCalendar
              int month = calendar.get(Calendar.MONTH) + 1;
              int day = calendar.get(Calendar.DAY_OF_MONTH);
              codeBuilder.append(String.format("datetime.date(%d, %d, %d)", year, month, day));
              break;
            }
          case CHAR:
          case VARCHAR:
            codeBuilder.append(
                makeQuoted(
                    escapePythonQuotes(
                        node.getValue2()
                            .toString()))); // extract value without specific sql type info.
            break;
          case TIMESTAMP:
            {
              GregorianCalendar calendar = (GregorianCalendar) node.getValue();
              // TODO: How do we represent microseconds and nanoseconds?
              long nanoseconds = calendar.getTimeInMillis() * 1000 * 1000;
              codeBuilder.append(String.format("pd.Timestamp(%d)", nanoseconds));
              break;
            }
          case BOOLEAN:
            String boolName = node.toString();
            codeBuilder.append(boolName.substring(0, 1).toUpperCase() + boolName.substring(1));
            break;
            /* according to https://calcite.apache.org/javadocAggregate/org/apache/calcite/rex/RexLiteral.html,
            INTERVAL_YEAR/YEAR_MONTH/MONTH are measured in months, and everything else is measured in miliseconds
             */
          case INTERVAL_WEEK:
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
            String milliseconds = node.getValue().toString();
            codeBuilder.append("pd.Timedelta(milliseconds=" + milliseconds + ")");
            break;
          case INTERVAL_YEAR:
          case INTERVAL_MONTH:
          case INTERVAL_YEAR_MONTH:
            // value is given in months
            // May later refactor this code to create DateOffsets, for now
            // causes an error
            String months = node.getValue().toString();
            codeBuilder.append("pd.DateOffset(months=" + months + ")");
            break;
          default:
            throw new BodoSQLCodegenException(
                "Internal Error: Calcite Plan Produced an Unsupported Literal Type");
        }
    }
    return codeBuilder.toString();
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
  public static <C extends Comparable<C>> String sargValToPyLiteral(C scalarVal) {
    // TODO: extend this to other types (timestamp, etc) which may need to do some
    // extra parsing/conversion

    if (scalarVal instanceof NlsString) {
      NlsString as_Nls = ((NlsString) scalarVal);
      // Simply calling toString creates a string with charset information attached
      // IE UTF-8:"hello world". This is needed to convert to a python literal string
      // TODO: check if this handles dealing with non-ascii strings
      return escapePythonQuotes(as_Nls.asSql(false, false));
    } else if (scalarVal instanceof BigDecimal) {
      return scalarVal.toString();
    } else {
      throw new BodoSQLCodegenException(
          "Internal error: Sarg limit value cannot be converted to python literal.");
    }
  }
}
