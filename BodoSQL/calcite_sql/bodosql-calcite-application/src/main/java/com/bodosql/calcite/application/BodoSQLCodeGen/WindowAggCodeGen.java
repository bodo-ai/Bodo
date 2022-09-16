package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.AggHelpers.getColumnAggCall;
import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToNullableBodoArray;
import static com.bodosql.calcite.application.Utils.Utils.assertWithErrMsg;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.Pair;

public class WindowAggCodeGen {

  // We define several variable names statically, for greater clarity when generating the function
  // text

  // The name of the column used for performing the reverse sort
  // This column should always be present in the input dataframe,
  // Though it will be pruned fairly early on if it is not needed.
  public static final String reverseSortColumName = "ORIG_POSITION_COL";

  // The variable name for the input to the apply function
  private static final String argumentDfName = "argument_df";

  // The variable name which stores the argument dataframe's original index
  private static final String argumentDfOriginalIndex = "argument_df_orig_index";

  // The variable name which stores the length of the argument dataframe
  private static final String argumentDfLen = "argument_df_len";

  private static final String indent = getBodoIndent();

  static HashMap<SqlKind, String> windowOptimizedKernels;

  static {
    windowOptimizedKernels = new HashMap<SqlKind, String>();
    windowOptimizedKernels.put(SqlKind.SUM, "bodo.libs.bodosql_array_kernels.windowed_sum");
    windowOptimizedKernels.put(SqlKind.SUM0, "bodo.libs.bodosql_array_kernels.windowed_sum");
    windowOptimizedKernels.put(SqlKind.COUNT, "bodo.libs.bodosql_array_kernels.windowed_count");
    windowOptimizedKernels.put(SqlKind.AVG, "bodo.libs.bodosql_array_kernels.windowed_avg");
    windowOptimizedKernels.put(SqlKind.MEDIAN, "bodo.libs.bodosql_array_kernels.windowed_median");
  }

  /**
   * Generates a function definition to be used in a groupby apply to perform a SQL Lead/Lag
   * aggregation.
   *
   * <p>This function should only ever be called from generateWindowedAggFn, as
   * generateWindowedAggFn takes care some code generation common to both Lead/Lag aggregations, and
   * the other aggregations (Namely, the definitions for argumentDfOriginalIndex and argumentDfLen)
   *
   * @param funcText The existing funcText, supplied by the first part of generateWindowedAggFn
   * @param argsListList the List of arguments to each of the aggregations being performed
   * @param expectedOutputColumns The list of string column names in which to store the results of
   *     the aggregations
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @param isLead Is the aggregation a LEAD, as opposed to a lag.
   * @return the generated function text
   */
  private static String generateLeadLagAggFn(
      StringBuilder funcText,
      final List<List<WindowedAggregationArgument>> argsListList,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final boolean isLead) {

    // Currently, we don't support aggregation fusion for lead and lag,
    // so we expect to only handle 1 lead/lag call in this function
    // TODO: Support aggregation fusion for LEAD/LAG, see BS-615
    assertWithErrMsg(
        argsListList.size() == 1,
        "generateLeadLagAggFn was supplied with more then one aggregation");

    List<WindowedAggregationArgument> curArgsList = argsListList.get(0);
    int num_arguments = curArgsList.size();
    // Lead/Lag expects one required argument, a column, and two optional arguments, an offset, and
    // a
    // default value
    assertWithErrMsg(
        1 <= num_arguments && num_arguments <= 3,
        "Lead/Lag expects between 1 and 3 arguments, instead got: " + curArgsList.size());

    WindowedAggregationArgument aggColArg = curArgsList.get(0);

    assertWithErrMsg(aggColArg.isDfCol(), "Lead/Lag's first argument must be a column");

    String aggColName = aggColArg.getExprString();

    // Default shift amount is 1
    String shiftAmount = "1";
    String fillValue = "";

    if (num_arguments >= 2) {
      WindowedAggregationArgument shiftAmountArg = curArgsList.get(1);
      assertWithErrMsg(
          !shiftAmountArg.isDfCol(),
          "Lead/Lag expects the offset to be a scalar literal, if it is provided. Got: "
              + curArgsList.toString());

      shiftAmount = shiftAmountArg.getExprString();

      // Add the default fill value (if it's present)
      if (num_arguments == 3) {
        WindowedAggregationArgument fillValueArg = curArgsList.get(2);
        // I don't know if this is handled within Calcite or not, so throwing it as a Bodo error
        if (fillValueArg.isDfCol()) {
          throw new BodoSQLCodegenException(
              "Error! Only scalar fill value is supported for LEAD/LAG");
        }
        fillValue = fillValueArg.getExprString();
      }
    }

    // Sort the input dataframe, if needed.
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    if (isLead) {
      shiftAmount = "(-" + shiftAmount + ")";
    }
    String aggColRef = "sorted_df[" + makeQuoted(aggColName) + "]";
    funcText
        .append(indent)
        .append(indent)
        .append(aggColRef + " = " + aggColRef + ".shift(" + shiftAmount);

    if (!fillValue.equals("")) {
      funcText.append(", fill_value=").append(fillValue);
    }

    funcText.append(")\n");

    funcText
        .append(indent)
        .append(indent)
        .append("arr = sorted_df[" + makeQuoted(aggColName) + "]\n");

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFunctextAndOutputDfName =
        reverseSortLocalDfIfNeeded(
            arraysToSort, "sorted_df", expectedOutputColumns, !sortByCols.equals(""));

    funcText.append(additionalFunctextAndOutputDfName.getKey());
    String outputDfName = additionalFunctextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that generate the dataframe to be returned by the groupby apply lambda
   * function. This helper function also reverse sorts the array containing the returned data, if
   * needs be.
   *
   * @param arrsToSort the list arrays that needs to be sorted/returned
   * @param sorted_df_name the name of the dataframe to be sorted/returned. Must contain
   *     reverseSortColumName if a reverse sort is needed.
   * @param expectedOutputColNames the list of string column names in which to store each of the
   *     arrays in the returned dataframe
   * @param needsReverseSort Does the above array need to be sorted before return. We need to sort
   *     the returned array if we had to sort the input data.
   * @return returns a string that contains the input columns stored in the specified output column
   *     names, reverse sorted if needed.
   */
  private static Pair<String, String> reverseSortLocalDfIfNeeded(
      final List<String> arrsToSort,
      final String sorted_df_name,
      final List<String> expectedOutputColNames,
      final boolean needsReverseSort) {

    // The number of arrays to sort should be equivalent to the number of expected output columns
    assert arrsToSort.size() == expectedOutputColNames.size();

    // TODO: if arrsToSort.size() == 1, we could return an array instead of a dataframe.
    // see BS-616

    StringBuilder funcText = new StringBuilder();
    List<String> outputCols = new ArrayList<>();
    // If we didn't have to do a sort on the input data, we can just return the arrays wrapped in a
    // dataframe.
    if (!needsReverseSort) {
      funcText.append(indent).append(indent).append("retval = pd.DataFrame({");
      for (int i = 0; i < arrsToSort.size(); i++) {
        outputCols.add(expectedOutputColNames.get(i));
        funcText.append(
            makeQuoted(expectedOutputColNames.get(i)) + ": " + arrsToSort.get(i) + ", ");
      }

      funcText.append("}, index = " + argumentDfOriginalIndex + ")\n");
    } else {
      // If we did need to do a sort on the dataframe, we need to sort the output column(s) before
      // returning them.
      funcText.append(indent).append(indent).append("_tmp_sorted_df = pd.DataFrame({");
      for (int i = 0; i < arrsToSort.size(); i++) {
        String curArr = arrsToSort.get(i);
        funcText.append(makeQuoted(expectedOutputColNames.get(i)) + ": " + curArr + ", ");
      }

      funcText
          .append(
              makeQuoted(reverseSortColumName)
                  + ": "
                  + sorted_df_name
                  + "["
                  + makeQuoted(reverseSortColumName)
                  + "]})")
          .append(".sort_values(by=[")
          .append(makeQuoted(reverseSortColumName))
          .append("], ascending=[True])\n");

      funcText.append(indent).append(indent).append("retval = pd.DataFrame({");

      for (int i = 0; i < arrsToSort.size(); i++) {
        funcText.append(
            makeQuoted(expectedOutputColNames.get(i))
                + ": _tmp_sorted_df["
                + makeQuoted(expectedOutputColNames.get(i))
                + "], ");
      }

      funcText.append("}, index = " + argumentDfOriginalIndex + ")\n");
    }

    return new Pair<>(funcText.toString(), "retval");
  }

  /**
   * Takes an input dataframe, performs a sort on it (if needed), and stores it to the specified
   * output variable.
   *
   * @param input_df_name The name of the dataframe to sort.
   * @param output_sorted_df_name The variable name where the output dataframe will be stored.
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   */
  private static String sortLocalDfIfNeeded(
      final String input_df_name,
      final String output_sorted_df_name,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList) {

    // TODO: performance upgrade
    // Currently we appending a column (ORIG_COL_POSITION) that keeps track of the original
    // positions.
    // Then, in the group by, we sort each of the partitioned dataframes on each rank, before
    // returning the sorted
    // dataframe it might be faster to do one sort on the entire data, instead of a sort on each of
    // the
    // partitioned dataframes.

    assert !sortByCols.equals("");

    StringBuilder sortText = new StringBuilder();
    if (!sortByCols.equals("")) {
      sortText
          .append(indent)
          .append(indent)
          .append(output_sorted_df_name + " = " + input_df_name + ".sort_values(by=")
          .append(sortByCols)
          .append(", ascending=")
          .append(ascendingList)
          .append(", na_position=")
          .append(NAPositionList)
          .append(")\n");
    } else {
      sortText
          .append(indent)
          .append(indent)
          .append(output_sorted_df_name + " = " + input_df_name + "\n");
    }

    return sortText.toString();
  }

  /**
   * Generates a function definition with the specified name, to be used in a groupby apply to
   * perform a SQL windowed aggregation.
   *
   * @param fn_name The name of the Window Function
   * @param sortByCols A string representing a list of string column names, to be used by
   *     df.sort_values, or empty string, if no sorting is necessary, or, in the case of RANK, will
   *     be the columns to be ranked in question
   * @param ascendingList A string representing a list of string boolean, to be used by
   *     df.sort_values, or empty string, if no sorting is necessary
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @param sortByList The list of string columns names, to be used when the each column name string
   *     is needed (e.g. generateRankFns uses them to fetch the appropriate columns).
   * @param agg The kind of the windowed aggregation to perform
   * @param aggName The string name of the windowed aggregation to perform
   * @param typs List of types for the output column, 1 per window function.
   * @param upper_bounded Does this window have an upper bound?
   * @param upper_bound_expr String expression that represents the "shift" amount for the window
   *     upper_bound
   * @param lower_bounded Does this window have a lower bound?
   * @param lower_bound_expr String expression that represents the "shift" amount for the window
   *     lower_bound
   * @param zeroExpr String that matches a window expression when the value is 0. This is included
   *     to enable passing types in the window exprs.
   * @param argsListList the List of arguments to each of the aggregations being performed
   * @return The generated function text, and a list of output column names, where the indexes of
   *     the output column list correspond to the indexes for the input list for each aggregation's
   *     arguments.
   */
  // For example:
  //     argsListList = [A, B] (agg = LEAD)
  //
  //     (simplified example func_text, this is the 0th return value in the tuple)
  //     def impl(df): ...
  //        out_df["AGG_OUTPUT_1"] = RESULT_OF_LEAD_AGG_ON_A
  //        out_df["AGG_OUTPUT_2"] = RESULT_OF_LEAD_AGG_ON_B
  //        return out_df
  //
  //     (1th return value in the tuple)
  //     outputColsList = ["AGG_OUTPUT_1", "AGG_OUTPUT_2"]]```

  public static Pair<String, List<String>> generateWindowedAggFn(
      final String fn_name,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final List<String> sortByList,
      final SqlKind agg,
      final String aggName,
      final List<SqlTypeName> typs,
      final boolean upper_bounded,
      final String upper_bound_expr,
      final boolean lower_bounded,
      final String lower_bound_expr,
      final String zeroExpr,
      final List<List<WindowedAggregationArgument>> argsListList) {

    // Before doing anything else, filter the partition columns out of the input dataframe. This is
    // done to enable Bodo to know that these columns are unused within this function
    // Add the aggregation columns

    List<String> returnedDfOutputCols = new ArrayList<>();
    StringBuilder kept_cols = new StringBuilder();

    /**
     * The columns that we need to keep are as follows: If we have a column on which we are
     * performing an aggregation (IE, MAX(A)), we need to keep that column If we have any columns by
     * which we need to sort, we need to keep those columns, and the ORIG_POSITION_COL which is
     * needed for the reverse sort.
     */

    // First, add the aggregation columns to the list of kept columns.
    for (int i = 0; i < argsListList.size(); i++) {

      // if we have a column argument, it is always the 0-th argument
      // TODO: update this when we support window functions where this is not the case
      if (argsListList.get(i).size() > 0 && argsListList.get(i).get(0).isDfCol()) {
        String colName = argsListList.get(i).get(0).getExprString();
        kept_cols.append(makeQuoted(colName)).append(", ");
      }
    }

    StringBuilder funcText = new StringBuilder();
    funcText.append(indent).append("def ").append(fn_name).append("(" + argumentDfName + "):\n");

    // Add sortbycols, removing the enclosing brackets, and add the original position column,
    // which is needed for the reverse sort.
    if (!sortByCols.equals("")) {
      kept_cols
          .append(makeQuoted(reverseSortColumName) + ", ")
          // sortbycols is passed as a string that looks like "['A', 'B', 'C']", so this substring
          // just removes the outer brackets so that we can add the columns to the new list needed
          // for
          // the call to loc
          .append(sortByCols.substring(1, sortByCols.length() - 1));
    }

    // Drop unneeded columns.
    funcText.append(indent).append(indent);
    if (!kept_cols.toString().equals("")) {
      funcText.append(
          argumentDfName + " = " + argumentDfName + ".loc[:, [" + kept_cols.toString() + "]]\n");
    } else {
      // In the case that kept_cols is none, we have to do the slicing with iloc, as numba has a
      // typing error with the
      // empty column list
      funcText.append(argumentDfName + " = " + argumentDfName + ".iloc[:, :0]\n");
    }

    // Next, initialize the list of expected output column names
    // We expect a number of output columns equal to the number of aggregations we are performing,
    // which is equal to the length of argsListList
    for (int i = 0; i < argsListList.size(); i++) {
      returnedDfOutputCols.add("AGG_OUTPUT_" + i);
    }

    // In the majority of cases (some exceptions like ROW NUMBER) we need to keep track of the
    // original index of the
    // input dataframe. This is needed due to some niche
    // Pandas behavior where the rows of the output dataframe from this function are returned to
    // their original
    // locations in the dataframe that is the output of the overall group by apply if the index of
    // the output dataframe is the same as the input dataframe.
    // Otherwise, the output of the overall group by apply will be multi-indexed.
    //

    // For example:
    //    @bodo.jit
    //    def example():
    //
    //    df = pd.DataFrame({"A": [1,2,3] * 2, "B": [1,2,3,4,5,6]})
    //
    //    def apply_impl1(sub_df):
    //    return pd.DataFrame({"C": [1,2]})
    //
    //
    //    def apply_impl2(sub_df):
    //    return pd.DataFrame({"C": [1,2]}, index=sub_df.index)
    //
    //
    //    print(df.groupby("A").apply(apply_impl1))
    //    print(df.groupby("A").apply(apply_impl2))

    // output:
    //    C
    //            A
    //    1 0  1
    //    1  2
    //    2 0  1
    //    1  2
    //    3 0  1
    //    1  2
    //
    //    C
    //            A
    //    1 0  1
    //    2 1  1
    //    3 2  1
    //    1 3  2
    //    2 4  2
    //    3 5  2

    // We could generate this variable definition on a case by case basis, but it simplifies codegen
    // to always append it
    // at the beginning of the func_text
    funcText
        .append(indent)
        .append(indent)
        .append(argumentDfOriginalIndex)
        .append(" = ")
        .append(argumentDfName)
        .append(".index\n");

    // There are also several locations where we need the length of the input dataframe. While we
    // could omit this
    // definition for certain aggregations, it simplifies codegen if we always include it at the
    // start
    // of the function definition
    funcText
        .append(indent)
        .append(indent)
        .append(argumentDfLen)
        .append(" = len(")
        .append(argumentDfName)
        .append(")\n");

    if (agg == SqlKind.RANK
        || agg == SqlKind.DENSE_RANK
        || agg == SqlKind.PERCENT_RANK
        || agg == SqlKind.CUME_DIST) {

      return new Pair<>(
          generateRankFns(
              funcText,
              argsListList,
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList,
              sortByList,
              agg),
          returnedDfOutputCols);
    } else if (agg == SqlKind.ROW_NUMBER) {
      // We have a separate function for row_number, that just uses np.arange()
      return new Pair<>(
          handleRowNumberWindowAgg(
              funcText,
              sortByCols,
              ascendingList,
              NAPositionList,
              argsListList,
              returnedDfOutputCols),
          returnedDfOutputCols);
    } else if (aggName == "CONDITIONAL_TRUE_EVENT") {
      // Similarly, we have a helper function that we can use for CONDITIONAL_TRUE_EVENT
      if (argsListList.size() != 1) {
        throw new BodoSQLCodegenException("CONDITIONAL_TRUE_EVENT should have exactly 1 argument");
      }
      WindowedAggregationArgument arg0 = argsListList.get(0).get(0);
      assert arg0.isDfCol();
      String arg0ColName = arg0.getExprString();
      return new Pair<>(
          generateTrueEventFn(
              funcText,
              arg0ColName,
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList),
          returnedDfOutputCols);
    } else if (agg == SqlKind.NTILE) {
      // Similarly, we have a helper function that we can use for NTILE
      return new Pair<>(
          generateNtileFn(
              funcText,
              argsListList,
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList),
          returnedDfOutputCols);
    } else if (aggName == "CONDITIONAL_CHANGE_EVENT") {
      // Similarly, we have a helper function that we can use for CONDITIONAL_CHANGE_EVENT
      assert argsListList.size() == 1;
      WindowedAggregationArgument arg0 = argsListList.get(0).get(0);
      assert arg0.isDfCol();
      String arg0ColName = arg0.getExprString();
      return new Pair<>(
          generateChangeEventFn(
              funcText,
              arg0ColName,
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList),
          returnedDfOutputCols);
    } else if (agg == SqlKind.COUNT && argsListList.get(0).size() == 0) {
      // For COUNT(*), we can manually calculate the length of the window, and
      // avoid actually taking slices of the input dataframe.
      return new Pair<>(
          generateCountStarFn(
              funcText,
              argsListList,
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList,
              upper_bounded,
              upper_bound_expr,
              lower_bounded,
              lower_bound_expr),
          returnedDfOutputCols);
    } else if (windowOptimizedKernels.containsKey(agg)) {
      // These functions have special window-optimized kernels
      assert argsListList.size() == 1;
      WindowedAggregationArgument arg0 = argsListList.get(0).get(0);
      assert arg0.isDfCol();
      String arg0ColName = arg0.getExprString();
      String lower = lower_bound_expr;
      String upper = upper_bound_expr;
      if (lower == "UNUSABLE_LOWER_BOUND") {
        lower = "-" + argumentDfLen;
      }
      if (upper == "UNUSABLE_UPPER_BOUND") {
        upper = argumentDfLen;
      }
      return new Pair<>(
          generateWindowOptimizedFn(
              funcText,
              arg0ColName,
              windowOptimizedKernels.get(agg),
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList,
              upper,
              lower),
          returnedDfOutputCols);
    } else if (agg == SqlKind.LAG || agg == SqlKind.LEAD) {
      // LAG/LEAD require special handling
      return new Pair<>(
          generateLeadLagAggFn(
              funcText,
              argsListList,
              returnedDfOutputCols,
              sortByCols,
              ascendingList,
              NAPositionList,
              (agg == SqlKind.LEAD)),
          returnedDfOutputCols);
    }

    // Perform the sort on the input dataframe (if needed) and store the resulting dataframe
    // in a variable named "sorted_df"
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    // If we have FIRST_VALUE with upper_bound=0 and unbounded_lower
    // then we have an array of just the first element.
    // If we have LAST_VALUE with lower_bound=0 and unbounded_upper
    // then we have an array of just the last element.
    // This enables a more efficient copy procedure.
    boolean optimized_copy =
        (agg == SqlKind.LAST_VALUE && !upper_bounded && lower_bound_expr.equals(zeroExpr))
            || (agg == SqlKind.FIRST_VALUE && !lower_bounded && upper_bound_expr.equals(zeroExpr));

    // Output columns used for both paths
    List<String> colsToAddToOutputDf = new ArrayList<>();
    // Variable to track the reverse shuffle
    boolean needsSort;

    // determine the type of the output
    if (optimized_copy) {
      for (int i = 0; i < argsListList.size(); i++) {
        SqlTypeName typeName = typs.get(i);
        String target_idx =
            agg == SqlKind.FIRST_VALUE ? "0" : String.format("len(target_arr%d) - 1", i);
        // How to fill NAs for this type
        StringBuilder na_arr_call = new StringBuilder();
        if (typeName == SqlTypeName.CHAR || typeName == SqlTypeName.VARCHAR) {
          // We generate a dummy array for gen_na_str_array_lens because we have an optimized path
          // when
          // length is 0.
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "arr%d = bodo.libs.str_arr_ext.gen_na_str_array_lens(%s, 0, np.empty(1,"
                          + " np.int64))\n",
                      i, argumentDfLen));
        } else if (typeName == SqlTypeName.BINARY || typeName == SqlTypeName.VARBINARY) {
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "arr%d = bodo.libs.str_arr_ext.pre_alloc_binary_array(%s, 0)\n",
                      i, argumentDfLen));
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(String.format("for j in range(len(arr%d)):\n", i));
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(String.format("bodo.libs.array_kernels.setna(arr%d, j)\n", i));
        } else {
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "arr%d = %s\n", i, sqlTypeToNullableBodoArray(argumentDfLen, typeName)));
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(String.format("for j in range(len(arr%d)):\n", i));
          na_arr_call
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(String.format("bodo.libs.array_kernels.setna(arr%d, j)\n", i));
        }
        // Generate the non-na path
        StringBuilder fill_op = new StringBuilder();
        if (typeName == SqlTypeName.BINARY || typeName == SqlTypeName.VARBINARY) {
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(String.format("val%d = target_arr%d[%s]\n", i, i, target_idx));
        } else if (typeName != SqlTypeName.CHAR && typeName != SqlTypeName.VARCHAR) {
          // Strings have an optimized path that don't require any intermediate allocations
          // so we don't generate a val.
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "val%d = bodo.utils.conversion.unbox_if_timestamp(target_arr%d[%s])\n",
                      i, i, target_idx));
        }
        if (typeName == SqlTypeName.CHAR || typeName == SqlTypeName.VARCHAR) {
          // We generate a dummy array for gen_na_str_array_lens because we have an optimized path
          // when
          // length is 0.
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "arr%d = bodo.libs.str_arr_ext.pre_alloc_string_array(%s,"
                          + " bodo.libs.str_arr_ext.get_str_arr_item_length(target_arr%d, %s) *"
                          + " %s)\n",
                      i, argumentDfLen, i, target_idx, argumentDfLen));
        } else if (typeName == SqlTypeName.BINARY || typeName == SqlTypeName.VARBINARY) {
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "arr%d = bodo.libs.str_arr_ext.pre_alloc_binary_array(%s, len(val%d) * %s)\n",
                      i, argumentDfLen, i, argumentDfLen));
        } else {
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "arr%d = %s\n", i, sqlTypeToNullableBodoArray(argumentDfLen, typeName)));
        }
        fill_op
            .append(indent)
            .append(indent)
            .append(indent)
            .append(String.format("for j in range(len(arr%d)):\n", i));
        if (typeName == SqlTypeName.CHAR || typeName == SqlTypeName.VARCHAR) {
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "bodo.libs.str_arr_ext.get_str_arr_item_copy(arr%d, j, target_arr%d, %s)\n",
                      i, i, target_idx));
        } else {
          fill_op
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(String.format("arr%d[j] = val%d\n", i, i, target_idx));
        }

        String curAggColName = argsListList.get(i).get(0).getExprString();
        // Get the target array.
        // TODO: Use get_dataframe_data. The sort columns alter the offsets.
        funcText
            .append(indent)
            .append(indent)
            .append(
                String.format(
                    "target_arr%d = bodo.hiframes.pd_series_ext.get_series_data(sorted_df[%s])\n",
                    i, makeQuoted(curAggColName)));
        funcText
            .append(indent)
            .append(indent)
            .append(
                String.format(
                    "if bodo.libs.array_kernels.isna(target_arr%d, %s):\n", i, target_idx));
        funcText.append(na_arr_call);
        funcText.append(indent).append(indent).append("else:\n");
        funcText.append(fill_op);
        colsToAddToOutputDf.add("arr" + i);
      }
      needsSort = false;

    } else {
      switch (aggName) {
        case "AVG":
        case "STDDEV_POP":
        case "STDDEV_SAMP":
        case "VAR_POP":
        case "VAR_SAMP":
        case "VARIANCE_SAMP":
        case "VARIANCE_POP":
          // TODO: These are the windowed aggregation functions that were previously decomposed
          // It seems that they currently just type check to the type of the input rows.
          // Ideally, I want to fix this, but for right now, I can do this as a workaround.
          // UPDATE: Technically, these typing of the aggregation functions are correct by ansi SQL.
          // It seems this workaround may last longer then intended.
          for (int i = 0; i < typs.size(); i++) {
            funcText
                .append(indent)
                .append(indent)
                .append("arr" + i + " = np.empty(" + argumentDfLen + ", dtype=np.float64)\n");
            colsToAddToOutputDf.add("arr" + i);
          }
          break;

        case "MAX":
        case "MIN":
          for (SqlTypeName typ : typs) {
            if (SqlTypeName.STRING_TYPES.contains(typ)) {
              throw new BodoSQLCodegenException(
                  "Windowed aggregation function "
                      + agg.toString()
                      + " not supported for SQL type "
                      + typ.toString());
            }
          }

        default:
          for (int i = 0; i < typs.size(); i++) {
            funcText
                .append(indent)
                .append(indent)
                .append(
                    "arr"
                        + i
                        + " = "
                        + sqlTypeToNullableBodoArray(argumentDfLen, typs.get(i))
                        + "\n");
            colsToAddToOutputDf.add("arr" + i);
          }
      }

      // TODO: in the case we have both upper/lower unbound, we can just do the calculation once,
      // and
      // create a numpy array with only those values
      funcText.append(indent).append(indent).append("for i in range(" + argumentDfLen + "):\n");

      // create cur_lower/upper_bounds variables which will be used for slicing.
      // We can omit creating bounds in the unbounded case, and use the empty string
      if (lower_bounded) {
        funcText
            .append(indent)
            .append(indent)
            .append(indent)
            .append("cur_lower_bound = max(0, i + " + lower_bound_expr + ")\n");
      }
      if (upper_bounded) {
        funcText
            .append(indent)
            .append(indent)
            .append(indent)
            .append("cur_upper_bound = max(0, i + " + upper_bound_expr + " + 1)\n");
      }

      funcText.append(indent).append(indent).append(indent).append("output_index = i\n");

      if (agg == SqlKind.LAST_VALUE || agg == SqlKind.FIRST_VALUE || agg == SqlKind.NTH_VALUE) {
        // For LAST/FIRST/NTH value, we can simply perform a get item on the input series

        // The value to be passed into ILOC, depending on if we are selecting the first/last value
        String ilocVal;

        // TODO: As of right now, in the case of unbounded, I'm simply setting the bounds
        // Lowest/highest bounds possible, since it simplifies both selecting the index for get
        // item,
        // and checking if the current window is non empty
        // in the future, I can move some of these checks from runtime to compile time.
        if (!lower_bounded) {
          funcText.append(indent).append(indent).append(indent).append("cur_lower_bound = 0\n");
        }
        if (!upper_bounded) {
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append("cur_upper_bound = " + argumentDfLen + "\n");
        }

        if (agg == SqlKind.LAST_VALUE) {
          ilocVal = "min(" + argumentDfLen + " - 1, cur_upper_bound - 1)";
        } else if (agg == SqlKind.FIRST_VALUE) {
          ilocVal = "cur_lower_bound";
        } else {
          assert agg == SqlKind.NTH_VALUE;
          // By mysql Nth Value, n is one indexed, and must be >= 1.
          // Mysql also allows a FROM first/last argument.
          // calcite has no restriction on the value of N, and does not allow a FROM first/last
          // argument.
          // for right now, I'm simply going to set N equal to 1 in the case that it is <= 1.
          // and defaulting to FROM FIRST

          assert !argsListList.get(0).get(0).isDfCol();
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              // TODO: Support aggregation fusion for last/NTH value, see BS-612
              .append("n = max(1, " + argsListList.get(0).get(1).getExprString() + ")\n");
          ilocVal = "cur_lower_bound + n - 1";
        }

        funcText
            .append(indent)
            .append(indent)
            .append(indent)
            .append(
                "if (cur_lower_bound >= cur_upper_bound or cur_lower_bound >= " + argumentDfLen);
        // In the case of Nth value, we have to add a check to see if the window is large enough
        if (agg == SqlKind.NTH_VALUE) {
          funcText.append(" or (cur_upper_bound - cur_lower_bound) < n");
        }
        funcText.append("):\n");
        for (int i = 0; i < argsListList.size(); i++) {
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append("bodo.libs.array_kernels.setna(arr" + i + ", output_index)\n");
        }

        funcText.append(indent).append(indent).append(indent).append("else:\n");
        for (int i = 0; i < argsListList.size(); i++) {
          // Last Value, First Value, and NTH value all expect a column arg0
          WindowedAggregationArgument curAggArg0 = argsListList.get(i).get(0);
          assert curAggArg0.isDfCol();
          String curAggColName = curAggArg0.getExprString();
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format(
                      "inarr_%d = bodo.hiframes.pd_series_ext.get_series_data(sorted_df[%s])\n",
                      i, makeQuoted(curAggColName)));
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  String.format("if bodo.libs.array_kernels.isna(inarr_%d, %s):\n", i, ilocVal));
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append("bodo.libs.array_kernels.setna(arr" + i + ", output_index)\n");
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append("continue\n");
          funcText
              .append(indent)
              .append(indent)
              .append(indent)
              .append(indent)
              .append(
                  "arr"
                      + i
                      + "[output_index] = bodo.utils.conversion.unbox_if_timestamp(inarr_"
                      + i
                      + "["
                      + ilocVal
                      + "])\n");
        }

      } else {
        // standard case, need to take a slice of the input, and perform
        // an aggregation on that slice
        // TODO: Support aggregation fusion for generic row fns. See BS-611
        assert argsListList.size() == 1;

        // For all aggregations handled in this clause arg0 should be a column
        WindowedAggregationArgument arg0 = argsListList.get(0).get(0);
        assert arg0.isDfCol();
        String arg0ColName = arg0.getExprString();

        // TODO: Find some way to avoid use slicing, as it will make a copy every time.
        funcText
            .append(indent)
            .append(indent)
            .append(indent)
            .append("cur_slice = sorted_df[")
            .append(makeQuoted(arg0ColName))
            .append("].iloc[");
        // TODO: if we ever want to allow non-scalar aggregation bounds, this will need to be
        // updated
        if (lower_bounded) {
          funcText.append("cur_lower_bound");
        }
        funcText.append(":");
        if (upper_bounded) {
          funcText.append("cur_upper_bound");
        }
        funcText.append("]\n");
        String columnAggCall = getColumnAggCall("cur_slice", agg, aggName);
        // currently, several aggregation functions have errors when called on empty slices,See
        // BE-1124
        // In the event that the other aggregation functions are fixed, we will still need
        // to perform check for LAST_VALUE and FIRST_VALUE
        if (agg != SqlKind.COUNT) {
          funcText.append(indent).append(indent).append(indent).append("if len(cur_slice) == 0:\n");
          for (int i = 0; i < argsListList.size(); i++) {
            funcText
                .append(indent)
                .append(indent)
                .append(indent)
                .append(indent)
                .append("bodo.libs.array_kernels.setna(arr" + i + ", output_index)\n");
          }

          funcText.append(indent).append(indent).append(indent).append("else:\n");
          funcText.append(indent).append(indent).append(indent).append(indent);
        } else {
          funcText.append(indent).append(indent).append(indent);
        }
        // Need to do unboxing to put pd timedelta/timestamp types into array
        // Needed for min/max
        for (int i = 0; i < argsListList.size(); i++) {
          funcText.append(
              "arr"
                  + i
                  + "[output_index] = bodo.utils.conversion.unbox_if_timestamp("
                  + columnAggCall
                  + ")\n");
        }
      }

      // Since the index of the output dataframe is the same as the index of the input dataframe,
      // The output of this dataframe will be a dataframe with the array values reverse shuffled
      // to their original positions in the input dataframe
      needsSort = !sortByCols.equals("");
    }

    Pair<String, String> out =
        reverseSortLocalDfIfNeeded(
            colsToAddToOutputDf, "sorted_df", returnedDfOutputCols, needsSort);
    String new_func_text = out.getKey();
    funcText.append(new_func_text);
    String dfName = out.getValue();
    funcText.append(indent).append(indent).append("return ").append(dfName).append("\n");

    return new Pair<>(funcText.toString(), returnedDfOutputCols);
  }

  /**
   * Helper function that handles the ROW_NUMBER window aggregation. Should only be called from
   * generateWindowedAggFn, after performing the column filtering, and the definitions for
   * argumentDfOriginalIndex, and argumentDfLen.
   *
   * @param funcText The current functext. Must contain only the function declaration.
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @param argsListList the List of arguments to each of the aggregations being performed
   * @param expectedOutputColumns the List of string column names at which to store the outputs of
   *     the aggregation
   * @return A string func_text that performs the aggregations, and stores the output in dataframe
   *     with the columns specified in expectedOutputColumns
   */
  public static String handleRowNumberWindowAgg(
      StringBuilder funcText,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final List<List<WindowedAggregationArgument>> argsListList,
      final List<String> expectedOutputColumns) {

    // We currently do not support aggregation fusion for ROW_NUMBER
    // TODO: Support aggregation fusion for ROW_NUMBER, BS-613
    assert argsListList.size() == 1;

    // ROW_NUMBER takes no arguments
    for (int i = 0; i < argsListList.size(); i++) {
      assert argsListList.get(i).size() == 0;
    }

    if (!sortByCols.equals("")) {
      funcText
          .append(indent)
          .append(indent)
          // reuse the input dataframe
          .append(argumentDfName)
          .append("[\"OUTPUT_COL\"] = np.arange(1, " + argumentDfLen + " + 1)\n");
      // Perform the sort
      funcText.append(
          sortLocalDfIfNeeded(
              argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));
      // Extract the value
      funcText.append(indent).append(indent).append("arr = sorted_df[\"OUTPUT_COL\"]\n");
    } else {
      funcText
          .append(indent)
          .append(indent)
          .append("arr = np.arange(1, " + argumentDfLen + " + 1)\n");
    }

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(arraysToSort, "sorted_df", expectedOutputColumns, false);

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that handles the CONDITIONAL_TRUE_EVENT window aggregation. Should only be
   * called from generateWindowedAggFn, after performing the column filtering, and the definitions
   * for argumentDfOriginalIndex, and argumentDfLen.
   *
   * <p>Note: if a window frame was provided, this function ignores it because this window function
   * always operates on the entire partition
   *
   * @param funcText The current func text. Must contain only the function declaration.
   * @param colName the name of the column whose change events are being tracked
   * @param expectedOutputColumns the list of string column names at which to store the output
   *     columns
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @return The completed funcText, which returns an output dataframe with the aggregations stored
   *     in the column names provided in expectedOutputColumns
   */
  private static String generateTrueEventFn(
      StringBuilder funcText,
      final String colName,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList) {

    if (sortByCols == "") {
      throw new BodoSQLCodegenException("CONDITIONAL_TRUE_EVENT requires an ORDER_BY clause");
    }

    // Perform the sort on the input dataframe (if needed) and store the resulting dataframe
    // in a variable named "sorted_df"
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    funcText
        .append(indent)
        .append(indent)
        .append("arr = sorted_df[" + makeQuoted(colName) + "].astype('uint8').cumsum()\n");

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(arraysToSort, "sorted_df", expectedOutputColumns, true);

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that handles the NTILE window aggregation. Should only be called from
   * generateWindowedAggFn, after performing the column filtering, and the definitions for
   * argumentDfOriginalIndex, and argumentDfLen.
   *
   * @param funcText The current func text. Must contain only the function declaration.
   * @param argsListList the list of arguments to each aggregation call
   * @param expectedOutputColumns the list of string column names at which to store the output
   *     columns
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @return The completed funcText, which returns an output dataframe with the aggregations stored
   *     in the column names provided in expectedOutputColumns
   */
  private static String generateNtileFn(
      StringBuilder funcText,
      final List<List<WindowedAggregationArgument>> argsListList,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList) {

    // Number of expected output columns should be equal to the number of functions
    assert argsListList.size() == expectedOutputColumns.size();

    // TODO: Support aggregation fusion for NTILE BS-614
    assert argsListList.size() == 1;

    // NTILE expects one argument (number of bins)
    for (int i = 0; i < argsListList.size(); i++) {
      assert argsListList.get(i).size() == 1;
    }

    WindowedAggregationArgument numBinsArg = argsListList.get(0).get(0);

    // numBins should be a literal
    assert !numBinsArg.isDfCol();

    String numBins = numBinsArg.getExprString();

    // By mysql, if the number of columns does not divide cleanly into the number of bins,
    // Then the earlier bins get the extra elements. This is achieved though the
    // use of a helper function, that generates the array containing the bin numbers
    // for each row

    if (!sortByCols.equals("")) {
      funcText
          .append(indent)
          .append(indent)
          // reuse the input dataframe
          .append(
              argumentDfName
                  + "[\"OUTPUT_COL\"] = bodosql.libs.ntile_helper.ntile_helper("
                  + argumentDfLen
                  + ", "
                  + numBins
                  + ")\n");
      // Perform the sort
      funcText.append(
          sortLocalDfIfNeeded(
              argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));
      funcText.append(indent).append(indent).append("arr = sorted_df[\"OUTPUT_COL\"]\n");
    } else {
      funcText
          .append(indent)
          .append(indent)
          .append(
              "arr = bodosql.libs.ntile_helper.ntile_helper("
                  + argumentDfLen
                  + ", "
                  + numBins
                  + ")\n");
    }

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(arraysToSort, "sorted_df", expectedOutputColumns, false);

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that handles the CONDITIONAL_CHANGE_EVENT window aggregation. Should only be
   * called from generateWindowedAggFn, after performing the column filtering, and the definitions
   * for argumentDfOriginalIndex, and argumentDfLen.
   *
   * <p>Note: if a window frame was provided, this function ignores it because this window function
   * always operates on the entire partition
   *
   * @param funcText The current func text. Must contain only the function declaration.
   * @param colName the name of the column whose change events are being tracked
   * @param expectedOutputColumns the list of string column names at which to store the output
   *     columns
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @return The completed funcText, which returns an output dataframe with the aggregations stored
   *     in the column names provided in expectedOutputColumns
   */
  private static String generateChangeEventFn(
      StringBuilder funcText,
      final String colName,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList) {

    // Perform the sort on the input dataframe (if needed) and store the resulting dataframe
    // in a variable named "sorted_df"
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    funcText
        .append(indent)
        .append(indent)
        .append(
            "arr = bodo.libs.bodosql_array_kernels.change_event("
                + "sorted_df["
                + makeQuoted(colName)
                + "])\n");

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(arraysToSort, "sorted_df", expectedOutputColumns, true);

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that handles window frmae-optimized window aggregation. Should only be called
   * from generateWindowedAggFn, after performing the column filtering, and the definitions for
   * argumentDfOriginalIndex, and argumentDfLen.
   *
   * @param funcText The current func text. Must contain only the function declaration.
   * @param colName the name of the column whose change events are being tracked
   * @param expectedOutputColumns the list of string column names at which to store the output
   *     columns
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @return The completed funcText, which returns an output dataframe with the aggregations stored
   *     in the column names provided in expectedOutputColumns
   */
  private static String generateWindowOptimizedFn(
      StringBuilder funcText,
      final String colName,
      final String kernelName,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final String upper_bound,
      final String lower_bound) {

    // Perform the sort on the input dataframe (if needed) and store the resulting dataframe
    // in a variable named "sorted_df"
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    funcText
        .append(indent)
        .append(indent)
        .append(
            "arr = "
                + kernelName
                + "("
                + "sorted_df["
                + makeQuoted(colName)
                + "], "
                + lower_bound
                + ", "
                + upper_bound
                + ")\n");

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(
            arraysToSort, "sorted_df", expectedOutputColumns, !sortByCols.equals(""));

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that handles the COUNT(*) window aggregation. Should only be called from
   * generateWindowedAggFn, after performing the column filtering, and the definitions for
   * argumentDfOriginalIndex, and argumentDfLen.
   *
   * @param funcText The current functext. Must contain only the function declaration.
   * @param argsListList the list of arguments to each aggregation call
   * @param expectedOutputColumns the list of string column names at which to store the output
   *     columns
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @param upper_bounded Does this window have an upper bound?
   * @param upper_bound_expr String expression that represents the "shift" amount for the window
   *     upper_bound
   * @param lower_bounded Does this window have a lower bound?
   * @param lower_bound_expr String expression that represents the "shift" amount for the window
   *     lower_bound
   * @return The func_text which performs the aggregation and stores the outputs into the specified
   *     output columns
   */
  private static String generateCountStarFn(
      StringBuilder funcText,
      final List<List<WindowedAggregationArgument>> argsListList,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final boolean upper_bounded,
      final String upper_bound_expr,
      final boolean lower_bounded,
      final String lower_bound_expr) {

    // Number of expected output columns should be equal to the number of functions
    assert argsListList.size() == expectedOutputColumns.size();

    // TODO: Support aggregation fusion for NTILE BS-614
    assert argsListList.size() == 1;

    // COUNT(*) expects no arguments
    for (int i = 0; i < argsListList.size(); i++) {
      assert argsListList.get(i).size() == 0;
    }

    funcText
        .append(indent)
        .append(indent)
        .append("arr = np.empty(" + argumentDfLen + ", np.int32)\n");

    funcText.append(indent).append(indent).append("for i in range(" + argumentDfLen + "):\n");

    if (lower_bounded) {
      funcText
          .append(indent)
          .append(indent)
          .append(indent)
          .append("lower_bound = max(0, i - ")
          .append(lower_bound_expr)
          .append(")\n");
    } else {
      funcText.append(indent).append(indent).append(indent).append("lower_bound = 0\n");
    }
    if (upper_bounded) {
      funcText
          .append(indent)
          .append(indent)
          .append(indent)
          .append("upper_bound = min(" + argumentDfLen + "-1, i + ")
          .append(upper_bound_expr)
          .append(")\n");
    } else {
      funcText
          .append(indent)
          .append(indent)
          .append(indent)
          .append("upper_bound = " + argumentDfLen + "-1\n");
    }

    funcText
        .append(indent)
        .append(indent)
        .append(indent)
        .append("arr[i] = max(0, (upper_bound - lower_bound) + 1)\n");

    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("arr");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(
            arraysToSort, "sorted_df", expectedOutputColumns, !sortByCols.equals(""));

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }

  /**
   * Helper function that handles the RANK window aggregations. Should only be called from
   * generateWindowedAggFn, after performing the column filtering, and the definitions for
   * argumentDfOriginalIndex, and argumentDfLen.
   *
   * @param funcText The current func text. Must contain only the function declaration.
   * @param argsListList the list of arguments to each aggregation call
   * @param expectedOutputColumns the list of string column names at which to store the output
   *     columns
   * @param sortByCols The string representing the list of string column names by which to sort
   * @param ascendingList The string representing the list of boolean values, which determining if
   *     the columns in sortByCols will be sorted ascending or descending
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @param sortByList Does this window have an upper bound?
   * @param agg the SqlKind for the aggregation being performed. Must be one of RANK, CUME_DIST, or
   *     DENSE_RANK
   * @return The func_text which performs the rank aggregations and stores the outputs into the
   *     specified output columns
   */
  private static String generateRankFns(
      StringBuilder funcText,
      final List<List<WindowedAggregationArgument>> argsListList,
      final List<String> expectedOutputColumns,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final List<String> sortByList,
      final SqlKind agg) {

    // TODO: support aggregation fusion for the RANK functions
    assert argsListList.size() == 0;

    // Rank takes no arguments
    for (int i = 0; i < argsListList.size(); i++) {
      assert argsListList.get(i).size() == 0;
    }

    String methodStr;
    if (agg == SqlKind.DENSE_RANK) {
      methodStr = "dense";
    } else if (agg == SqlKind.CUME_DIST) {
      methodStr = "max";
    } else {
      methodStr = "min";
    }
    String pctStr = agg == SqlKind.CUME_DIST ? "True" : "False";
    // Window functions must contain an order by clause (no case for sortByCols == "")
    if (sortByCols == "") {
      throw new BodoSQLCodegenException(
          "The OVER clause of ranking window functions must include an ORDER BY clause.");
    }
    // Perform the sort
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    funcText
        .append(indent)
        .append(indent)
        .append("sorted_df[\"OUTPUT_COL\"] = bodo.libs.bodosql_array_kernels.rank_sql((");

    for (String col : sortByList) {
      funcText.append(
          String.format("bodo.hiframes.pd_series_ext.get_series_data(sorted_df[%s]), ", col));
    }
    funcText.append(String.format("), method=\"%s\", pct=%s)", methodStr, pctStr));
    if (agg == SqlKind.PERCENT_RANK) {
      funcText.append(" - 1\n");
      funcText.append(indent).append(indent).append("if " + argumentDfLen + " == 1:\n");
      funcText
          .append(indent)
          .append(indent)
          .append(indent)
          .append("sorted_df[\"OUTPUT_COL\"] = 0.0\n");
      funcText.append(indent).append(indent).append("else:\n");
      funcText
          .append(indent)
          .append(indent)
          .append(indent)
          .append(
              "sorted_df[\"OUTPUT_COL\"] = sorted_df[\"OUTPUT_COL\"] / ("
                  + argumentDfLen
                  + " - 1)\n");
    } else {
      funcText.append("\n");
    }

    // Extract the value
    List<String> arraysToSort = new ArrayList<>();
    arraysToSort.add("sorted_df[\"OUTPUT_COL\"]");

    Pair<String, String> additionalFuncTextAndOutputDfName =
        reverseSortLocalDfIfNeeded(arraysToSort, "sorted_df", expectedOutputColumns, true);

    funcText.append(additionalFuncTextAndOutputDfName.getKey());
    String outputDfName = additionalFuncTextAndOutputDfName.getValue();

    funcText.append(indent).append(indent).append("return " + outputDfName + "\n");

    return funcText.toString();
  }
}
