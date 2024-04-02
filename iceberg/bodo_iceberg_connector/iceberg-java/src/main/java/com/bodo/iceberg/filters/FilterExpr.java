package com.bodo.iceberg.filters;

import java.util.*;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.expressions.Expressions;

public class FilterExpr extends Filter {
  private final String op;
  private final List<Filter> args;

  public FilterExpr(String op, List<Filter> args) {
    this.op = op;
    this.args = args;
  }

  String asCol(Filter f) {
    if (f instanceof ColumnRef) {
      return ((ColumnRef) f).name;
    } else {
      throw new IllegalArgumentException("Arg is not Column");
    }
  }

  Object asConst(Filter f) {
    if (f instanceof Const) {
      return ((Const) f).value.value();
    } else {
      throw new IllegalArgumentException("Arg is not a Const with Literal");
    }
  }

  ArrayList<Object> asConstArr(Filter f) {
    if (f instanceof ArrayConst) {
      ArrayList<Object> outVal = new ArrayList<>();
      for (Object o : ((ArrayConst) f).value) {
        if (o instanceof Const) {
          outVal.add(((Const) o).value.value());
        } else {
          throw new IllegalArgumentException("Value in ArrayConst is not a Const: " + o.getClass());
        }
      }
      return outVal;
    } else {
      throw new IllegalArgumentException("Arg is not an ArrayConst");
    }
  }

  String asConstStr(Filter f) {
    if (!(f instanceof Const)) {
      throw new IllegalArgumentException("Arg is not Literal");
    }

    Object val = ((Const) f).value.value();
    if (!(val instanceof String)) {
      throw new IllegalArgumentException("Literal content is not string");
    }

    return ((String) val);
  }

  public Expression toExpr() {
    switch (op) {
      case "==":
        return Expressions.equal(asCol(args.get(0)), asConst(args.get(1)));
      case "!=":
        return Expressions.notEqual(asCol(args.get(0)), asConst(args.get(1)));
      case ">":
        return Expressions.greaterThan(asCol(args.get(0)), asConst(args.get(1)));
      case ">=":
        return Expressions.greaterThanOrEqual(asCol(args.get(0)), asConst(args.get(1)));
      case "<":
        return Expressions.lessThan(asCol(args.get(0)), asConst(args.get(1)));
      case "<=":
        return Expressions.lessThanOrEqual(asCol(args.get(0)), asConst(args.get(1)));

      case "IS_NULL":
        return Expressions.isNull(asCol(args.get(0)));
      case "IS_NOT_NULL":
        return Expressions.notNull(asCol(args.get(0)));

      case "STARTS_WITH":
        return Expressions.startsWith(asCol(args.get(0)), asConstStr(args.get(1)));
      case "NOT_STARTS_WITH":
        return Expressions.notStartsWith(asCol(args.get(0)), asConstStr(args.get(1)));

      case "IN":
        return Expressions.in(asCol(args.get(0)), asConstArr(args.get(1)));
      case "NOT_IN":
        return Expressions.notIn(asCol(args.get(0)), asConstArr(args.get(1)));

      case "AND":
        Optional<Expression> newArgs =
            args.stream().map(a -> ((FilterExpr) a).toExpr()).reduce(Expressions::and);
        assert newArgs.isPresent();
        return newArgs.get();
      case "OR":
        Optional<Expression> orArgs =
            args.stream().map(a -> ((FilterExpr) a).toExpr()).reduce(Expressions::or);
        assert orArgs.isPresent();
        return orArgs.get();
      case "NOT":
        return Expressions.not(((FilterExpr) args.get(0)).toExpr());

      case "ALWAYS_TRUE":
        return Expressions.alwaysTrue();
      case "ALWAYS_FALSE":
        return Expressions.alwaysFalse();
    }

    throw new IllegalArgumentException("Invalid Filter Expression Name: '" + op + "'");
  }
}
