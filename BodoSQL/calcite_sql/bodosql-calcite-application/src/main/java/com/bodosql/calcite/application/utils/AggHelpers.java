package com.bodosql.calcite.application.utils;

import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.Pair;

/** Class of helper functions used to process an Agg RelNode. */
public class AggHelpers {

  // Set of Functions for which the name of the output column is simply
  // fn_name(input)
  static HashMap<SqlKind, String> simpleColnameMap;

  static {
    simpleColnameMap = new HashMap<>();
    simpleColnameMap.put(SqlKind.SUM, "sum");
    simpleColnameMap.put(SqlKind.MIN, "min");
    simpleColnameMap.put(SqlKind.MAX, "max");
    simpleColnameMap.put(SqlKind.VAR_SAMP, "var_samp");
    simpleColnameMap.put(SqlKind.AVG, "avg");
    simpleColnameMap.put(SqlKind.MEDIAN, "median");
    simpleColnameMap.put(SqlKind.STDDEV_SAMP, "std_samp");
    simpleColnameMap.put(SqlKind.STDDEV_POP, "std");
    simpleColnameMap.put(SqlKind.VAR_POP, "var");
  }

  /**
   * Helper function to generated dummy names for a column. This is used by NamedAggregation when a
   * generated columnName is not valid Python.
   *
   * @param colNum The column number that needs a dummy name.
   * @returns A dummy column name for given colNum
   */
  public static String getDummyColName(int colNum) {
    return Utils.getDummyColNameBase() + colNum;
  }

  /**
   * Determine if an aggCallList contains any filters. If filters are included we need to generated
   * groupby.apply instead of groupby.agg.
   *
   * @param aggCallList List of aggregations
   * @return Does any column include a filter
   */
  public static boolean aggContainsFilter(List<AggregateCall> aggCallList) {
    for (int i = 0; i < aggCallList.size(); i++) {
      AggregateCall a = aggCallList.get(i);
      if (a.filterArg != -1) {
        return true;
      }
    }
    return false;
  }

  /**
   * Return the name of the function/method used by count.
   *
   * @param a The aggregate count call
   * @param isGroupbyCall Is the call being directly used in a groupby.
   * @return A pair of value, the name of the function/method and if the call is a method call.
   */
  public static Pair<String, Boolean> getCountCall(AggregateCall a, boolean isGroupbyCall) {
    if (a.getArgList().isEmpty()) {
      if (isGroupbyCall) {
        return new Pair<>("size", true);
      } else {
        return new Pair<>("len", false);
      }
    } else if (a.isDistinct()) {
      return new Pair<>("nunique", true);
    } else {
      return new Pair<>("count", true);
    }
  }

  /**
   * Function that generate the code to create the proper group by.
   *
   * @param inVar DataFrame variable on which to perform the groupby
   * @param inputColumnNames All possible input columns
   * @param groups The groups for the group by.
   * @return The code generated for the group by.
   */
  public static Expr generateGroupByCall(
      Variable inVar, List<String> inputColumnNames, List<Integer> groups) {
    List<Expr.StringLiteral> groupList = new ArrayList<>();
    for (int i : groups) {
      String columnName = inputColumnNames.get(i);
      groupList.add(new Expr.StringLiteral(columnName));
    }

    return new Expr.Groupby(inVar, new Expr.List(groupList), false, false);
  }
}
