package com.bodosql.calcite.application.BodoSQLCodeGen;

import com.bodosql.calcite.ir.*;
import java.util.ArrayList;
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
    List<Expr> byList = new ArrayList<>();
    List<Expr> ascendingList = new ArrayList<>();
    List<Expr.StringLiteral> naPositionList = new ArrayList<>();

    if (!sortOrders.isEmpty()) {
      for (RelFieldCollation order : sortOrders) {
        int index = order.getFieldIndex();
        naPositionList.add(getNAPositionStringLiteral(order.nullDirection));
        byList.add(new Expr.StringLiteral(colNames.get(index)));
        ascendingList.add(getAscendingExpr(order.getDirection()));
      }
    }
    Expr sortExpr =
        new Expr.SortValues(
            inVar,
            new Expr.List(byList),
            new Expr.List(ascendingList),
            new Expr.List(naPositionList));

    if (!offsetStr.isEmpty()) {
      Expr sliceStart = new Expr.Raw(offsetStr);
      Expr sliceEnd = new Expr.Raw(offsetStr + " + " + limitStr);
      Expr limitSlice = new Expr.Slice(List.of(sliceStart, sliceEnd));

      sortExpr = new Expr.GetItem(new Expr.Attribute(sortExpr, "iloc"), limitSlice);
    } else if (!limitStr.isEmpty()) {
      sortExpr = new Expr.Method(sortExpr, "head", List.of(new Expr.Raw(limitStr)), List.of());
    }
    return sortExpr;
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
