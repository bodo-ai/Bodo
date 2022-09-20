package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.escapePythonQuotes;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.sql.type.SqlTypeName;

/** Class that returns the generated code for Literal Code after all inputs have been visited. */
public class LiteralCodeGen {
  /**
   * Function that return the necessary generated code for Literals.
   *
   * @param node The RexLiteral node.
   * @return The code generated that matches the Literal.
   */
  public static String generateLiteralCode(RexLiteral node) {
    StringBuilder codeBuilder = new StringBuilder();
    SqlTypeName typeName = node.getType().getSqlTypeName();
    String out = "";
    // TODO: Add more types here
    if (node.getTypeName() == SqlTypeName.NULL) {
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
        case TIME:
          // TODO: Support all remaining interval types.
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
          codeBuilder.append("None");
          break;
        case FLOAT:
        case REAL:
        case DOUBLE:
        case DECIMAL:
          codeBuilder.append("np.nan");
          break;
        default:
          throw new BodoSQLCodegenException(
              "Internal Error: Calcite Plan Produced an Unsupported Null Literal Type: "
                  + typeName);
      }
    } else {
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
          // TODO: Parse string into fields to avoid objmode in Bodo
          codeBuilder.append("pd.Timestamp(" + makeQuoted(node.toString()) + ")");
          break;
        case CHAR:
        case VARCHAR:
          codeBuilder.append(
              makeQuoted(
                  escapePythonQuotes(
                      node.getValue2()
                          .toString()))); // extract value without specific sql type info.
          break;
        case TIMESTAMP:
          // TODO: Parse string into fields to avoid objmode in Bodo
          // Note this can possibly be done via node.getValue()

          // Currently, node.toString will contain the precision of the TIMESTAMP type.
          // We workaround this by simply splitting the string when needed.
          // TODO: have some more robust code here using getValue
          String tsString = node.toString();
          int precision_string_length = 13;
          if (tsString.contains("TIMESTAMP")) {
            tsString = tsString.substring(0, tsString.length() - precision_string_length);
          }
          codeBuilder.append("pd.Timestamp(" + makeQuoted(tsString) + ")");
          break;
        case BOOLEAN:
          String boolName = node.toString();
          codeBuilder.append(boolName.substring(0, 1).toUpperCase() + boolName.substring(1));
          break;
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
}
