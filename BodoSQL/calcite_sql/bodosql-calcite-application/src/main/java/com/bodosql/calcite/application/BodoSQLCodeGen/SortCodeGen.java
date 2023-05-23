package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.ir.*;
import java.util.List;
import org.apache.calcite.rel.RelFieldCollation;

/**
 * Class that returns the generated code for Sort expressions after all inputs have been visited.
 */
public class SortCodeGen {

  /**
   * Function that return the necessary generated code for a Sort expression.
   *
   * @param inVar The input variable.
   * @param colNames The names of columns that may be sorted.
   * @param sortOrders The directions of each sort (Ascending/Descending).
   * @param limitStr String corresponding to limit param in limit queries. If there is no limit this
   *     is an empty string.
   * @param offsetStr String corresponding to offset param in limit queries. If there is no
   *     limit/offset this is an empty string.
   * @return The code generated for the Sort expression.
   */
  public static Expr generateSortCode(
      Variable inVar,
      List<String> colNames,
      List<RelFieldCollation> sortOrders,
      String limitStr,
      String offsetStr) {
    // StringBuilder for the final expr
    StringBuilder sortString = new StringBuilder();
    sortString.append(inVar.emit());

    // Sort handles both limit and sort_values (possibly both).
    // If the sortOrders is empty then we are not sorting
    if (!sortOrders.isEmpty()) {
      // StringBuilder for the ascending section
      StringBuilder orderString = new StringBuilder();
      StringBuilder naPositionString = new StringBuilder();
      sortString.append(".sort_values(by=[");
      orderString.append("ascending=[");
      naPositionString.append("na_position=[");
      for (RelFieldCollation order : sortOrders) {
        int index = order.getFieldIndex();
        sortString.append(makeQuoted(colNames.get(index))).append(", ");
        orderString.append(getAscendingExpr(order.getDirection()).emit()).append(", ");
        naPositionString
            .append(getNAPositionStringLiteral(order.nullDirection).emit())
            .append(", ");
      }
      orderString.append("]");
      naPositionString.append("]");
      sortString
          .append("], ")
          .append(orderString)
          .append(", ")
          .append(naPositionString)
          .append(")");
    }
    if (!offsetStr.equals("")) {
      // If offsetStr is not empty, we are taking a limit with an offset and need df.loc
      sortString
          .append(".iloc[")
          .append(offsetStr)
          .append(": ")
          .append(offsetStr)
          .append(" + ")
          .append(limitStr)
          .append(", :]");
    } else if (!limitStr.equals("")) {
      // If limitStr is not empty but offset is, we are taking a limit with no offset and can use
      // head().
      // TODO: Determine if we should use iloc here.
      sortString.append(".head(").append(limitStr).append(")");
    }

    return new Expr.Raw(sortString.toString());
  }

  /**
   * Get the boolean corresponding to the direction of ORDER BY
   *
   * @param direction whether Order by is ascending or descending
   * @return true if ascending order false otherwise.
   */
  public static Expr.BooleanLiteral getAscendingExpr(RelFieldCollation.Direction direction) {
    boolean val = false;
    if (direction == RelFieldCollation.Direction.ASCENDING) {
      val = true;
    }
    return new Expr.BooleanLiteral(val);
  }

  /**
   * Get the string corresponding to the NA Position
   *
   * @param direction whether NA position is first or last
   * @return String "first" if first and "last" if last.
   */
  public static Expr.StringLiteral getNAPositionStringLiteral(
      RelFieldCollation.NullDirection direction) {
    String result = "last";
    if (direction == RelFieldCollation.NullDirection.FIRST) {
      result = "first";
    }
    return new Expr.StringLiteral(result);
  }
}
