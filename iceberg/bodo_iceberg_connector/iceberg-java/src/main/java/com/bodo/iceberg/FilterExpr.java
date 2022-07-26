package com.bodo.iceberg;

import java.util.ArrayList;
import java.util.LinkedList;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;
import org.apache.iceberg.expressions.Literal;

public class FilterExpr {
  /**
   * Parses the filters passed to java, which is a list of operators and filter components and
   * converts them to proper iceberg scalars. Each individual filter consists of column name,
   * OpEnum, IcebergLiteral.
   *
   * <p>In addition, filters are joined by other OpEnum values AND/OR. We don't have to worry about
   * operator precedence because the form is always ORing AND expressions.
   */
  public static Expression filtersToExpr(LinkedList<Object> filters) {
    if (filters == null || filters.isEmpty()) {
      // If there are no predicates pass the true predicate
      return Expressions.alwaysTrue();
    }
    // We now process expressions in ANDs.
    LinkedList<Expression> expressions = new LinkedList<>();
    while (!filters.isEmpty()) {
      expressions.push(andFiltersToExpr(filters));
    }
    if (expressions.size() == 1) {
      return expressions.pop();
    }
    Expression currentExpr = expressions.removeFirst();
    while (!expressions.isEmpty()) {
      currentExpr = Expressions.or(currentExpr, expressions.removeFirst());
    }
    return currentExpr;
  }

  public static Expression andFiltersToExpr(LinkedList<Object> filters) {
    Object andStart = filters.removeFirst();
    assert andStart.equals(OpEnum.AND_START);
    LinkedList<Expression> expressions = new LinkedList<>();
    while (!filters.getFirst().equals(OpEnum.AND_END)) {
      expressions.push(singleFilterToExpr(filters));
    }
    // Remove the ANDEND
    filters.removeFirst();
    if (expressions.size() == 0) {
      // If this is just start-end return TRUE.
      return Expressions.alwaysTrue();
    } else if (expressions.size() == 1) {
      return expressions.pop();
    }
    Expression currentExpr = expressions.removeFirst();
    while (!expressions.isEmpty()) {
      currentExpr = Expressions.and(currentExpr, expressions.removeFirst());
    }
    return currentExpr;
  }

  public static Expression singleFilterToExpr(LinkedList<Object> filters) {
    // First value is always a field ID.
    String name = (String) filters.removeFirst();
    // Get the op.
    OpEnum op = (OpEnum) filters.removeFirst();

    Expression.Operation icebergOp = Expression.Operation.TRUE;
    // Only used by in/not in
    ArrayList<Object> lit_list = null;
    switch (op) {
      case EQ:
        icebergOp = Expression.Operation.EQ;
        break;
      case NE:
        icebergOp = Expression.Operation.NOT_EQ;
        break;
      case LT:
        icebergOp = Expression.Operation.LT;
        break;
      case GT:
        icebergOp = Expression.Operation.GT;
        break;
      case GE:
        icebergOp = Expression.Operation.GT_EQ;
        break;
      case LE:
        icebergOp = Expression.Operation.LT_EQ;
        break;
      case STARTS_WITH:
        icebergOp = Expression.Operation.STARTS_WITH;
        break;
      case NOT_STARTS_WITH:
        icebergOp = Expression.Operation.NOT_STARTS_WITH;
        break;
      case IN:
        icebergOp = Expression.Operation.IN;
        // NOTE: Iceberg takes regular Java lists in this case, not Literal lists.
        // see predicate(Expression.Operation op, java.lang.String name,
        //               java.lang.Iterable<T> values)
        // https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        lit_list = (ArrayList<Object>) filters.removeFirst();
        return Expressions.predicate(icebergOp, name, lit_list);
      case NOT_IN:
        icebergOp = Expression.Operation.NOT_IN;
        // NOTE: Iceberg takes regular Java lists in this case, not Literal lists.
        // see predicate(Expression.Operation op, java.lang.String name,
        //               java.lang.Iterable<T> values)
        // https://iceberg.apache.org/javadoc/0.13.1/index.html?org/apache/iceberg/types/package-summary.html
        lit_list = (ArrayList<Object>) filters.removeFirst();
        return Expressions.predicate(icebergOp, name, lit_list);
      case IS_NULL:
        icebergOp = Expression.Operation.IS_NULL;
        // remove "NULL" from list
        filters.removeFirst();
        return Expressions.predicate(icebergOp, name);
      case NOT_NULL:
        icebergOp = Expression.Operation.NOT_NULL;
        // remove "NULL" from list
        filters.removeFirst();
        return Expressions.predicate(icebergOp, name);
      default:
        // We should never reach this case.
        assert false;
    }
    // Get the literal
    Literal<Object> lit = (Literal<Object>) filters.removeFirst();
    return Expressions.predicate(icebergOp, name, lit);
  }
}
