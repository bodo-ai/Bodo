package com.bodosql.calcite.application.Utils;

import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.BodoSQLCodegenException;
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
    simpleColnameMap.put(SqlKind.STDDEV_SAMP, "std_samp");
    simpleColnameMap.put(SqlKind.STDDEV_POP, "std");
    simpleColnameMap.put(SqlKind.VAR_POP, "var");
    simpleColnameMap.put(SqlKind.OTHER_FUNCTION, "sum");
  }

  /**
   * function to get the renamed aggregation
   *
   * @param kind Kind of aggregation
   * @param isDistinct Is the Function called on Distinct data. Currently only used for count, but
   *     is provided for all AggregateCalls.
   * @param fieldName name of the field being aggregated
   * @return string corresponding to aggregation
   */
  public static String renameAgg(SqlKind kind, boolean isDistinct, String fieldName) {
    String aggColName;
    if (simpleColnameMap.containsKey(kind)) {
      aggColName = simpleColnameMap.get(kind);
    } else {
      switch (kind) {
        case COUNT:
          if (isDistinct) {
            aggColName = "count(distinct " + fieldName + ")";
          } else {
            aggColName = "count(" + fieldName + ")";
          }
          break;
        default:
          throw new BodoSQLCodegenException(
              "Unsupported Aggregate Function, " + kind.toString() + " specified in query.");
      }
    }

    return aggColName;
  }

  /**
   * @param colExpr the string column expression
   * @param kind The SqlKind of the aggregation
   * @param name The name of the aggregation
   * @return The string expression of the aggregated column
   */
  public static String getColumnAggCall(String colExpr, SqlKind kind, String name) {
    switch (kind) {
      case MAX:
        return colExpr + ".max()";
      case MIN:
        return colExpr + ".min()";
      case SUM0:
        return colExpr + ".sum()";
      case COUNT:
        return colExpr + ".count()";
      case AVG:
        return colExpr + ".mean()";
      case STDDEV_SAMP:
        return colExpr + ".std()";
      case STDDEV_POP:
        return colExpr + ".std(ddof=0)";
      case VAR_SAMP:
        return colExpr + ".var()";
      case VAR_POP:
        return colExpr + ".var(ddof=0)";
        // Currently, the empty slice case is being handled in the calling
        // function, generateWindowedAggFn
      case LAST_VALUE:
        return colExpr + ".iloc[-1]";
      case ANY_VALUE:
      case FIRST_VALUE:
        return colExpr + ".iloc[0]";
      case OTHER_FUNCTION:
        switch (name) {
          case "COUNT_IF":
            return colExpr + ".sum()";
        }
      default:
        throw new BodoSQLCodegenException(
            "Error, column aggregation function " + kind.toString() + " not supported");
    }
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
   * Function to enclose string in square brackets
   *
   * @param s string to be enclosed
   * @return single enclosed string
   */
  public static String encloseInBrackets(String s) {
    return '[' + s + ']';
  }

  /**
   * Determine if an aggCallList contains any filters. If filters are included we need to generated
   * groupby.apply instead of groupby.agg.
   *
   * @param AggList List of aggregations
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
   * @param inputColumnNames All possible input columns
   * @param groups The groups for the group by.
   * @return The code generated for the group by.
   */
  public static String generateGroupByCall(List<String> inputColumnNames, List<Integer> groups) {
    StringBuilder groupbyBuilder = new StringBuilder();
    List<String> groupList = new ArrayList<>();
    for (int i : groups) {
      String columnName = inputColumnNames.get(i);
      groupList.add(makeQuoted(columnName));
    }
    // Generate the Group By section
    groupbyBuilder
        .append(".groupby(")
        .append(groupList)
        .append(", as_index = False, dropna=False)");
    return groupbyBuilder.toString();
  }
}
