package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.ProjectCodeGen.generateProjectedDataframe;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.getAscendingExpr;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.getNAPositionStringLiteral;
import static com.bodosql.calcite.application.Utils.BodoArrayHelpers.sqlTypeToNullableBodoArray;
import static com.bodosql.calcite.application.Utils.Utils.addIndent;
import static com.bodosql.calcite.application.Utils.Utils.assertWithErrMsg;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;

import com.bodosql.calcite.adapter.pandas.RexToPandasTranslator;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.BodoSQLExprType;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Module;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import com.google.common.collect.ImmutableList;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexFieldCollation;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexOver;
import org.apache.calcite.rex.RexWindow;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.util.Pair;

/*
 * For explanations of how/why this file has been refactored recently:
 * https://bodo.atlassian.net/wiki/spaces/B/pages/1164836876/Fusing+window+function+calls+with+the+same+window
 */

public class WindowAggCodeGen {

  // We define several variable names statically, for greater clarity when generating the function
  // text

  // Variable names for the dummy strings to use when there is no upper/lower bound
  public static final String unboundedLowerBound = "UNUSABLE_LOWER_BOUND";
  public static final String unboundedUpperBound = "UNUSABLE_UPPER_BOUND";

  // The name of the column used for performing the sort reversion
  // This column should always be present in the input DataFrame,
  // Though it will be pruned fairly early on if it is not needed.
  public static final String revertSortColumnName = "ORIG_POSITION_COL";

  // The variable name for the input to the apply function
  private static final String argumentDfName = "argument_df";

  // The variable name which stores the argument DataFrame's original index
  private static final String argumentDfOriginalIndex = "argument_df_orig_index";

  // The variable name which stores the length of the argument dataframe
  private static final String partitionLength = "argument_df_len";

  private static final String indent = getBodoIndent();

  // Which function names correspond to a kernel made with gen_windowed
  static List<String> windowOptimizedKernels = new ArrayList<String>();

  // Same as windowOptimizedKernels but for 2-argument window functions
  public static List<String> twoArgWindowOptimizedKernels = new ArrayList<String>();

  // A mapping of window function names to a single function that can be
  // used to directly compute the result of the window function of interest
  static HashMap<String, String> windowCodeExpressions = new HashMap<String, String>();

  // A mapping of window function names to a method that can be
  // used to directly compute the result of the window function of interest
  static HashMap<String, String> windowFunctions = new HashMap<String, String>();

  static HashSet<String> optimizedEngineKernelFunctions = new HashSet<String>();

  // Note: $SUM0 is included in addition to SUM because of some Calcite quirks
  static {
    optimizedEngineKernelFunctions.add("ROW_NUMBER");
    optimizedEngineKernelFunctions.add("MIN_ROW_NUMBER_FILTER");
    optimizedEngineKernelFunctions.add("RANK");
    optimizedEngineKernelFunctions.add("DENSE_RANK");
    optimizedEngineKernelFunctions.add("PERCENT_RANK");
    optimizedEngineKernelFunctions.add("CUME_DIST");
    optimizedEngineKernelFunctions.add("NTILE");
    optimizedEngineKernelFunctions.add("CONDITIONAL_TRUE_EVENT");
    optimizedEngineKernelFunctions.add("CONDITIONAL_CHANGE_EVENT");

    // Window functions that have a sliding-window kernel
    windowOptimizedKernels.add("SUM");
    windowOptimizedKernels.add("$SUM0");
    windowOptimizedKernels.add("COUNT");
    windowOptimizedKernels.add("COUNT(*)");
    windowOptimizedKernels.add("COUNT_IF");
    windowOptimizedKernels.add("AVG");
    windowOptimizedKernels.add("MEDIAN");
    windowOptimizedKernels.add("MODE");
    windowOptimizedKernels.add("RATIO_TO_REPORT");
    windowOptimizedKernels.add("VARIANCE");
    windowOptimizedKernels.add("VARIANCE_SAMP");
    windowOptimizedKernels.add("VARIANCE_POP");
    windowOptimizedKernels.add("VAR_SAMP");
    windowOptimizedKernels.add("VAR_POP");
    windowOptimizedKernels.add("STDDEV");
    windowOptimizedKernels.add("STDDEV_SAMP");
    windowOptimizedKernels.add("STDDEV_POP");
    windowOptimizedKernels.add("SKEW");
    windowOptimizedKernels.add("KURTOSIS");
    windowOptimizedKernels.add("BITOR_AGG");
    windowOptimizedKernels.add("BITAND_AGG");
    windowOptimizedKernels.add("BITXOR_AGG");
    windowOptimizedKernels.add("BOOLOR_AGG");
    windowOptimizedKernels.add("BOOLAND_AGG");
    windowOptimizedKernels.add("BOOLXOR_AGG");
    windowOptimizedKernels.add("MIN");
    windowOptimizedKernels.add("MAX");

    // Window functions that have a two-argument sliding-window kernel
    twoArgWindowOptimizedKernels.add("COVAR_SAMP");
    twoArgWindowOptimizedKernels.add("COVAR_POP");
    twoArgWindowOptimizedKernels.add("CORR");

    // Window functions that have a dedicated kernel
    windowCodeExpressions.put("SUM", "windowed_sum");
    windowCodeExpressions.put("$SUM0", "windowed_sum");
    windowCodeExpressions.put("COUNT", "windowed_count");
    windowCodeExpressions.put("COUNT(*)", "windowed_count_star");
    windowCodeExpressions.put("COUNT_IF", "windowed_count_if");
    windowCodeExpressions.put("AVG", "windowed_avg");
    windowCodeExpressions.put("MEDIAN", "windowed_median");
    windowCodeExpressions.put("MODE", "windowed_mode");
    windowCodeExpressions.put("RATIO_TO_REPORT", "windowed_ratio_to_report");
    windowCodeExpressions.put("COVAR_SAMP", "windowed_covar_samp");
    windowCodeExpressions.put("COVAR_POP", "windowed_covar_pop");
    windowCodeExpressions.put("CORR", "windowed_corr");
    windowCodeExpressions.put("CONDITIONAL_CHANGE_EVENT", "change_event");
    windowCodeExpressions.put("STDDEV", "windowed_stddev_samp");
    windowCodeExpressions.put("STDDEV_SAMP", "windowed_stddev_samp");
    windowCodeExpressions.put("STDDEV_POP", "windowed_stddev_pop");
    windowCodeExpressions.put("SKEW", "windowed_skew");
    windowCodeExpressions.put("KURTOSIS", "windowed_kurtosis");
    windowCodeExpressions.put("VARIANCE", "windowed_var_samp");
    windowCodeExpressions.put("VARIANCE_SAMP", "windowed_var_samp");
    windowCodeExpressions.put("VAR_SAMP", "windowed_var_samp");
    windowCodeExpressions.put("VARIANCE_POP", "windowed_var_pop");
    windowCodeExpressions.put("VAR_POP", "windowed_var_pop");
    windowCodeExpressions.put("BITOR_AGG", "windowed_bitor_agg");
    windowCodeExpressions.put("BITAND_AGG", "windowed_bitand_agg");
    windowCodeExpressions.put("BITXOR_AGG", "windowed_bitxor_agg");
    windowCodeExpressions.put("BOOLOR_AGG", "windowed_boolor");
    windowCodeExpressions.put("BOOLAND_AGG", "windowed_booland");
    windowCodeExpressions.put("BOOLXOR_AGG", "windowed_boolxor");
    windowCodeExpressions.put("MIN", "windowed_min");
    windowCodeExpressions.put("MAX", "windowed_max");
  }

  /**
   * Generates a function definition with the specified name, to be used in a groupby apply to
   * perform one or more SQL windowed aggregations. All aggregations handled by a single call are
   * assumed to have the same PARTITION BY / ORDER BY clauses, but everything else (function,
   * arguments, window frame) can vary from call to call.
   *
   * @param fnVar The variable for Window Function used in df.apply.
   * @param sortByCols A string representing a list of string column names, to be used by
   *     df.sort_values, or empty string, if no sorting is necessary, or, in the case of RANK, will
   *     be the columns to be ranked in question
   * @param ascendingList A string representing a list of string boolean, to be used by
   *     df.sort_values, or empty string, if no sorting is necessary
   * @param NAPositionList The string representing the list of string values, which determine null
   *     ordering for each column being sorted. This is empty if no sorting is necessary.
   * @param sortByList The list of string columns names, to be used when the each column name string
   *     is needed (e.g. generateRankFns uses them to fetch the appropriate columns).
   * @param aggs The kinds of the windowed aggregations to perform
   * @param aggNames The string names of the windowed aggregations to perform
   * @param typs List of types for the output column, 1 per window function.
   * @param upperBoundedFlags List of whether each call has an upper bound
   * @param upperBoundExprs String expressions that represent the "shift" amount for each window's
   *     upper bound
   * @param lowerBoundedFlags List of whether each call has an lower bound
   * @param lowerBoundExprs String expressions that represent the "shift" amount for each window's
   *     lower bound
   * @param zeroExpr String that matches a window expression when the value is 0. This is included
   *     to enable passing types in the window exprs.
   * @param argsListList the List of arguments to each of the aggregations being performed
   * @return The generated function text, and a list of output column names, where the indexes of
   *     the output column list correspond to the indexes for the input list for each aggregation's
   *     arguments.
   */
  public static Pair<String, List<String>> generateWindowedAggFn(
      final Variable fnVar,
      final String sortByCols,
      final String ascendingList,
      final String NAPositionList,
      final List<String> sortByList,
      final List<SqlKind> aggs,
      final List<String> aggNames,
      final List<RelDataType> typs,
      final List<Boolean> lowerBoundedFlags,
      final List<Boolean> upperBoundedFlags,
      final List<String> lowerBoundExprs,
      final List<String> upperBoundExprs,
      final String zeroExpr,
      final List<List<WindowedAggregationArgument>> argsListList,
      final List<Boolean> isRespectNulls) {

    // Buffer where we will store the text for the closure
    StringBuilder funcText = new StringBuilder();

    // List where we will store the names of the arrays that contain the result
    // for each window function call
    List<String> colsToAddToOutputDf = new ArrayList<>();

    // Whether or not these window function calls require a sort
    boolean hasOrder = !sortByCols.equals("");

    /* First, initialize the list of expected output column names. We expect a number
     * of output columns equal to the number of aggregations we are performing,
     * which is equal to the length of argsListList
     */
    List<String> returnedDfOutputCols = new ArrayList<>();
    for (int i = 0; i < argsListList.size(); i++) {
      returnedDfOutputCols.add("AGG_OUTPUT_" + i);
    }

    /* Add the definition line for the closure. Looks as follows:
     *
     * def closure_name(argument_df):
     */
    addIndent(funcText, 1);
    funcText.append("def ").append(fnVar.getName()).append("(" + argumentDfName + "):\n");

    /* Before doing anything else, filter the partition columns out of the input DataFrame. This is
     * done to enable Bodo to know that these columns are unused within this function
     * and the aggregation columns. Looks as follows:
     *
     * argument_df = argument_df.iloc[:, ["AGG_OP_0", "AGG_OP_1", "ORIG_POSITION_COL", "ASC_COL_0"]]
     */
    pruneColumns(funcText, aggNames, argsListList, hasOrder, sortByCols);

    /* In the majority of cases (some exceptions like ROW NUMBER) we need to keep
     * track of the original index of the input DataFrame. This is needed due to
     * some niche Pandas behavior where the rows of the output DataFrame from this
     * function are returned to their original locations in the DataFrame that is
     * the output of the overall groupby apply if the index of the output DataFrame
     * is the same as the input DataFrame. Otherwise, the output of the overall
     * groupby apply will be multi-indexed. Looks as follows:
     *
     * argument_df_orig_index = argument_df.index
     */
    addIndent(funcText, 2);
    funcText
        .append(argumentDfOriginalIndex)
        .append(" = ")
        .append(argumentDfName)
        .append(".index\n");

    /* There are also several locations where we need the length of the input
     * DataFrame. While we could omit this definition for certain aggregations,
     * it simplifies codegen if we always include it at the start of the function
     * definition. Looks as follows:
     *
     * argument_df_len = len(argument_df)
     */
    addIndent(funcText, 2);
    funcText.append(partitionLength).append(" = len(").append(argumentDfName).append(")\n");

    /* Sort the entries in partition by the ORDER BY columns. Looks as follows:
     *
     * sorted_df = argument_df.sort_values(by=["ASC_COL_0", ], ascending=[True, ], na_position=["last", ])
     */
    funcText.append(
        sortLocalDfIfNeeded(
            argumentDfName, "sorted_df", sortByCols, ascendingList, NAPositionList));

    /* Keep track of whether or not we will need to revert the sort at the very
     * end. Each time we perform an aggregation, we set this to true UNLESS there
     * are no sort columns or the aggregation is an optimized first_value/last_value.
     *
     * We do not revert the data sorting unless we encounter a call to a window
     * function that requires the output data to be in the same order as the
     * input data, such as SUM.
     */
    Boolean needsToRevertSort = false;

    // Loop over each aggregation in the closure and map it to the correct
    // helper logic
    for (int i = 0; i < aggs.size(); i++) {
      SqlKind agg = aggs.get(i);
      String aggName = aggNames.get(i);
      List<WindowedAggregationArgument> argsList = argsListList.get(i);

      // If COUNT(*), the aggregate will be COUNT with no arguments.
      // This is treated separately from the normal count.
      if (aggName.equals("COUNT") && argsList.size() == 0) {
        aggName = "COUNT(*)";
      }

      // Create easy-to-use string variables for the upper and lower bounds by
      // mapping UNBOUNDED XXX to offsets that match the size of the current
      // partition.
      Boolean upperBounded = upperBoundedFlags.get(i);
      Boolean lowerBounded = lowerBoundedFlags.get(i);
      String lower = lowerBoundExprs.get(i);
      String upper = upperBoundExprs.get(i);
      if (lower.equals(unboundedLowerBound)) {
        lower = "-" + partitionLength;
      }
      if (upper.equals(unboundedUpperBound)) {
        upper = partitionLength;
      }

      // Case on the aggregation and redirect to the appropriate code-generating
      // helper function (COUNT(*) should have already been handled separately)
      switch (aggName) {

          /* Generate the code for a RANK call. Looks as follows:
           *
           * arr0  = bodo.libs.bodosql_array_kernels.rank_sql((bodo.hiframes.pd_series_ext.get_series_data(sorted_df["ASC_COL_0"]), ), method="min", pct=False)
           */
        case "RANK":
        case "DENSE_RANK":
        case "PERCENT_RANK":
        case "CUME_DIST":
          if (sortByCols == "") {
            throw new BodoSQLCodegenException(
                "The OVER clause of ranking window functions must include an ORDER BY clause.");
          }
          needsToRevertSort |= hasOrder;
          generateRankFns(funcText, argsList, sortByList, agg, colsToAddToOutputDf, i);
          break;

          /* Generate the code for a ROW_NUMBER call. Looks as follows:
           *
           * arr0 = np.arange(1, argument_df_len + 1)
           */
        case "ROW_NUMBER":
          needsToRevertSort |= hasOrder;
          generateRowNumberFn(funcText, argsList, colsToAddToOutputDf, i);
          break;

          /* Generate the code for a NTILE call. Looks as follows:
           *
           * arr0 = bodosql.libs.ntile_helper.ntile_helper(argument_df_len, np.int32(10))
           */
        case "NTILE":
          needsToRevertSort |= hasOrder;
          generateNtileFn(funcText, argsList, colsToAddToOutputDf, i);
          break;

          /* Generate the code for a LEAD/LAG call. Looks as follows:
           *
           * arr0 = sorted_df["AGG_OP_0"].shift(1)
           */
        case "LAG":
        case "LEAD":
          needsToRevertSort |= hasOrder;
          generateLeadLagAggFn(
              funcText,
              argsList,
              agg == SqlKind.LEAD,
              isRespectNulls.get(i),
              colsToAddToOutputDf,
              i);
          break;

          /* Generate the code for a CONDITIONAL_TRUE_EVENT call. Looks as follows:
           *
           * arr0 = sorted_df["AGG_OP_0"].astype('uint32').cumsum()
           */
        case "CONDITIONAL_TRUE_EVENT":
          needsToRevertSort |= hasOrder;
          generateTrueEventFn(funcText, argsList, colsToAddToOutputDf, i);
          break;

          // TODO [BE-3948]: try to fuse these loops
        case "FIRST_VALUE":
        case "LAST_VALUE":
        case "ANY_VALUE":
        case "NTH_VALUE":
          if (!isRespectNulls.get(i)) {
            String errMsg = "IGNORE_NULLS not yet supported for " + aggName;
            throw new BodoSQLCodegenException(errMsg);
          }
          /* If doing FIRST_VALUE/ANY_VALUE on a prefix window, or LAST_VALUE
           * on a suffix window, the optimized version can be used which
           * does NOT require the sorting to be reverted at the end.
           * Looks as follows:
           *
           * target_arr0 = bodo.hiframes.pd_series_ext.get_series_data(sorted_df["AGG_OP_0"])
           * if bodo.libs.array_kernels.isna(target_arr0, 0):
           *   arr0 = np.empty(argument_df_len, dtype="datetime64[ns]")
           *   for j in range(len(arr0)):
           *     bodo.libs.array_kernels.setna(arr0, j)
           * else:
           *   val0 = target_arr0[0]
           *   arr0 = np.empty(argument_df_len, dtype="datetime64[ns]")
           *   for j in range(len(arr0)):
           *     arr0[j] = val0
           */
          if (((aggName == "FIRST_VALUE" || aggName == "ANY_VALUE")
                  && !lowerBounded
                  && upper.equals(zeroExpr))
              || (aggName == "LAST_VALUE" && !upperBounded && lower.equals(zeroExpr))) {
            generateOptimizedFirstLast(
                funcText,
                typs,
                partitionLength,
                lowerBounded,
                upperBounded,
                lower,
                upper,
                aggName,
                argsList,
                colsToAddToOutputDf,
                i);

            /* Otherwise, generate the FIRST_VALUE/LAST_VALUE/ANY_VALUE/NTH_VALUE
             * normally using a loop. Looks as follows:
             *
             * n = max(1, np.int32(3))
             * arr0 = np.empty(argument_df_len, dtype="datetime64[ns]")
             * in_arr0 = bodo.hiframes.pd_series.ext.get_series_data(sorted_df["AGG_OP_0"])
             * for i in range(argument_df_len):
             *   cur_lower_bound = 0
             *   cur_upper_bound = max(0, i + np.int64(0) + 1)
             *   cur_idx = cur_lower_bound + n - 1
             *   if cur_lower_bound >= cur_upper_bound or cur_lower_bound + n - 1 >= argument_df_len or (cur_upper_bound - cur_lower_bound) < n or bodo.libs.array_kernels.isna(in_arr0, cur_idx):
             *     bodo.libs.array_kernels.setna(arr0, i)
             *   else:
             *     arr0[i] = in_arr0[cur_idx]
             */
          } else {
            needsToRevertSort |= hasOrder;
            generateFirstLastNth(
                funcText,
                typs,
                partitionLength,
                lowerBounded,
                upperBounded,
                lower,
                upper,
                aggName,
                argsList,
                colsToAddToOutputDf,
                i);
          }
          break;

          /* All the remaining window functions have a dedicated kernel. Looks
           * as follows:
           *
           * arr0 = bodo.libs.bodosql_array_kernels.windowed_mode(sorted_df["AGG_OP_0"], -argument_df_len, np.int64(0))
           */
        default:
          needsToRevertSort |= hasOrder;
          if (!windowCodeExpressions.containsKey(aggName)) {
            throw new BodoSQLCodegenException("Unrecognized window function: " + aggName);
          }
          generateSimpleWindowFnCode(
              funcText, argsList, aggName, lower, upper, colsToAddToOutputDf, i);
      }
    }

    /* If the sort-reverting flag is true, revert the sort. Afterwards, shove
     * everything into the output DataFrame. Looks as follows:
     *
     * _tmp_sorted_df = pd.DataFrame({"AGG_OUTPUT_0": arr0, "ORIG_POSITION_COL": sorted_df["ORIG_POSITION_COL"]}).sort_values(by=["ORIG_POSITION_COL"], ascending=[True])
     * retval = pd.DataFrame({"AGG_OUTPUT_0": _tmp_sorted_df["AGG_OUTPUT_0"], }, index = argument_df_orig_index)
     * return retval
     */
    Pair<String, String> out =
        revertSortLocalDfIfNeeded(
            colsToAddToOutputDf, "sorted_df", returnedDfOutputCols, needsToRevertSort);
    String new_func_text = out.getKey();
    funcText.append(new_func_text);
    String dfName = out.getValue();
    addIndent(funcText, 2);
    funcText.append("return ").append(dfName).append("\n");

    return new Pair<>(funcText.toString(), returnedDfOutputCols);
  }

  /**
   * Generates the code for FIRST_VALUE / LAST_VALUE / ANY_VALUE / NTH_VALUE in non-optimized cases.
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param typs Types of the input columns
   * @param partitionLength String representation of the length of the partition
   * @param lowerBounded Is there a lower bound?
   * @param upperBounded Is there an upper bound?
   * @param lowerBound String representation of the lower bound
   * @param upperBound String representation of the upper bound
   * @param aggName Which window funciton (out of FIRST/LAST/ANY/NTH) is being called
   * @param argsList List of all inputs to the current window function call
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateFirstLastNth(
      StringBuilder funcText,
      final List<RelDataType> typs,
      final String partitionLength,
      final boolean lowerBounded,
      final boolean upperBounded,
      final String lowerBound,
      final String upperBound,
      final String aggName,
      final List<WindowedAggregationArgument> argsList,
      final List<String> arraysToSort,
      final int i) {

    String arrIdx;

    // Which index is the target value exctracted from
    if (aggName == "FIRST_VALUE" || aggName == "ANY_VALUE") {
      assertWithErrMsg(argsList.size() >= 1, aggName + " requires 1 argument");
      arrIdx = "cur_lower_bound";
    } else if (aggName == "LAST_VALUE") {
      assertWithErrMsg(argsList.size() >= 1, aggName + " requires 1 argument");
      arrIdx = "cur_upper_bound - 1";
    } else {
      assertWithErrMsg(argsList.size() >= 1, aggName + " requires 2 arguments");
      arrIdx = "cur_lower_bound + n - 1";
      addIndent(funcText, 2);
      funcText.append("n = max(1, " + argsList.get(1).getExprString() + ")\n");
    }

    assertWithErrMsg(argsList.get(0).isDfCol(), aggName + " requires a column input");

    String outputArrayName = "arr" + String.valueOf(i);
    String inputArrayName = "in_arr" + String.valueOf(i);
    WindowedAggregationArgument arg0 = argsList.get(0);
    String arg0ColName = arg0.getExprString();

    // Build the array to store the output
    addIndent(funcText, 2);
    funcText.append(outputArrayName).append(" = ");
    funcText.append(sqlTypeToNullableBodoArray(partitionLength, typs.get(i)) + "\n");

    // Load the input array
    addIndent(funcText, 2);
    funcText
        .append(inputArrayName)
        .append(" = bodo.hiframes.pd_series_ext.get_series_data(sorted_df[")
        .append(makeQuoted(arg0ColName))
        .append("])\n");

    // Loop over each entry and calculate the current upper/lower bound of the window
    // (handling defaults appropriately)
    addIndent(funcText, 2);
    funcText.append("for i in range(" + partitionLength + "):\n");
    if (lowerBounded) {
      addIndent(funcText, 3);
      funcText.append(
          "cur_lower_bound = min(max(0, i + " + lowerBound + "), " + partitionLength + ")\n");
    } else {
      addIndent(funcText, 3);
      funcText.append("cur_lower_bound = 0\n");
    }
    if (upperBounded) {
      addIndent(funcText, 3);
      funcText.append(
          "cur_upper_bound = min(max(0, i + " + upperBound + " + 1), " + partitionLength + ")\n");
    } else {
      addIndent(funcText, 3);
      funcText.append("cur_upper_bound = " + partitionLength + "\n");
    }
    // Create the current index as its own variable.
    addIndent(funcText, 3);
    funcText.append("cur_idx = " + arrIdx + "\n");

    // Add the condition to check whether the window frame is out
    // of bounds
    addIndent(funcText, 3);
    funcText.append("if (cur_lower_bound >= cur_upper_bound)");

    // For NTH value, also make sure that the window is wide enough to have
    // at least n values inside it
    if (aggName == "NTH_VALUE") {
      funcText.append(" or ((cur_upper_bound - cur_lower_bound) < n)");
    }
    // If the input is already NA the output should be NA.
    funcText
        .append(" or bodo.libs.array_kernels.isna(")
        .append(inputArrayName)
        .append(", cur_idx)");
    funcText.append(":\n");

    // If its out of bounds (or too small), then the output is NULL
    addIndent(funcText, 4);
    funcText.append("bodo.libs.array_kernels.setna(" + outputArrayName + ", i)\n");

    // Otherwise, extract the corresponding location of the input array
    // and place it in index i of the output array
    addIndent(funcText, 3);
    funcText.append("else:\n");
    addIndent(funcText, 4);
    funcText.append(outputArrayName + "[i] = ").append(inputArrayName).append("[cur_idx]\n");

    // Register the output array
    arraysToSort.add(outputArrayName);
  }

  /**
   * Generates the code for FIRST_VALUE / LAST_VALUE/ ANY_VALUE in optimized cases (i.e. the prefix
   * frame for first/any or the suffix frame for last)
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param typs Types of the input columns
   * @param partitionLength String representation of the length of the partition
   * @param lowerBounded Is there a lower bound?
   * @param upperBounded Is there an upper bound?
   * @param lowerBound String representation of the lower bound
   * @param upperBound String representation of the upper bound
   * @param aggName Which window funciton (out of FIRST/LAST/ANY/NTH) is being called
   * @param argsList List of all inputs to the current window function call
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateOptimizedFirstLast(
      StringBuilder funcText,
      final List<RelDataType> typs,
      final String partitionLength,
      final boolean lowerBounded,
      final boolean upperBounded,
      final String lowerBound,
      final String upperBound,
      final String aggName,
      final List<WindowedAggregationArgument> argsList,
      final List<String> arraysToSort,
      final int i) {

    assertWithErrMsg(argsList.size() == 1, aggName + " requires 1 input");
    assertWithErrMsg(argsList.get(0).isDfCol(), aggName + " requires a column input");

    // Get the type and its name.
    RelDataType typ = typs.get(i);
    SqlTypeName type_name = typs.get(i).getSqlTypeName();
    String outputArrayName = "arr" + String.valueOf(i);
    WindowedAggregationArgument arg0 = argsList.get(0);
    String arg0ColName = arg0.getExprString();

    // Keep track of which value is copied to the rest of the array
    String target_idx;
    if (aggName == "LAST_VALUE") {
      target_idx = partitionLength + " - 1";
    } else {
      target_idx = "0";
    }

    StringBuilder na_arr_call = new StringBuilder();
    StringBuilder fill_op = new StringBuilder();

    // Generate the code to fill the entire array with nulls
    if (SqlTypeName.CHAR_TYPES.contains(type_name)) {
      // We generate a dummy array for gen_na_str_array_lens because we have an optimized path
      // when length is 0.
      addIndent(na_arr_call, 3);
      na_arr_call.append(
          String.format(
              "arr%d = bodo.libs.str_arr_ext.gen_na_str_array_lens(%s, 0, np.empty(1,"
                  + " np.int64))\n",
              i, partitionLength));
    } else {
      if (SqlTypeName.BINARY_TYPES.contains(type_name)) {
        addIndent(na_arr_call, 3);
        na_arr_call.append(
            String.format(
                "arr%d = bodo.libs.str_arr_ext.pre_alloc_binary_array(%s, 0)\n",
                i, partitionLength));
      } else {
        addIndent(na_arr_call, 3);
        na_arr_call.append(
            String.format("arr%d = %s\n", i, sqlTypeToNullableBodoArray(partitionLength, typ)));
      }
      addIndent(na_arr_call, 3);
      na_arr_call.append(String.format("for j in range(len(arr%d)):\n", i));
      addIndent(na_arr_call, 4);
      na_arr_call.append(String.format("bodo.libs.array_kernels.setna(arr%d, j)\n", i));
    }

    // Generate the code to copy a single value when it is not null
    if (SqlTypeName.BINARY_TYPES.contains(type_name)) {
      addIndent(fill_op, 3);
      fill_op.append(String.format("val%d = target_arr%d[%s]\n", i, i, target_idx));
    } else if (type_name != SqlTypeName.CHAR && type_name != SqlTypeName.VARCHAR) {
      // Strings have an optimized path that don't require any intermediate allocations
      // so we don't generate a val.
      addIndent(fill_op, 3);
      fill_op.append(String.format("val%d = target_arr%d[%s]\n", i, i, target_idx));
    }
    if (SqlTypeName.CHAR_TYPES.contains(type_name)) {
      // We generate a dummy array for gen_na_str_array_lens because we have an optimized path
      // when length is 0.
      addIndent(fill_op, 3);
      fill_op.append(
          String.format(
              "arr%d = bodo.libs.str_arr_ext.pre_alloc_string_array(%s,"
                  + " bodo.libs.str_arr_ext.get_str_arr_item_length(target_arr%d, %s) *"
                  + " %s)\n",
              i, partitionLength, i, target_idx, partitionLength));
    } else if (SqlTypeName.BINARY_TYPES.contains(type_name)) {
      addIndent(fill_op, 3);
      fill_op.append(
          String.format(
              "arr%d = bodo.libs.str_arr_ext.pre_alloc_binary_array(%s, len(val%d) * %s)\n",
              i, partitionLength, i, partitionLength));
    } else {
      addIndent(fill_op, 3);
      fill_op.append(
          String.format("arr%d = %s\n", i, sqlTypeToNullableBodoArray(partitionLength, typ)));
    }
    addIndent(fill_op, 3);
    fill_op.append(String.format("for j in range(len(arr%d)):\n", i));
    if (SqlTypeName.CHAR_TYPES.contains(type_name)) {
      addIndent(fill_op, 4);
      fill_op.append(
          String.format(
              "bodo.libs.str_arr_ext.get_str_arr_item_copy(arr%d, j, target_arr%d, %s)\n",
              i, i, target_idx));
    } else {
      addIndent(fill_op, 4);
      fill_op.append(String.format("arr%d[j] = val%d\n", i, i));
    }

    // Generate the logic that extracts the first/last value, checks if
    // its null, and executes one of the two branches generated above
    addIndent(funcText, 2);
    funcText.append(
        String.format(
            "target_arr%d = bodo.hiframes.pd_series_ext.get_series_data(sorted_df[%s])\n",
            i, makeQuoted(arg0ColName)));
    addIndent(funcText, 2);
    funcText.append(
        String.format("if bodo.libs.array_kernels.isna(target_arr%d, %s):\n", i, target_idx));
    funcText.append(na_arr_call);
    addIndent(funcText, 2);
    funcText.append("else:\n");
    funcText.append(fill_op);

    arraysToSort.add(outputArrayName);
  }

  /**
   * Helper function that handles window aggregations implemented with loops and methods.
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param typs Types of the input columns
   * @param partitionLength String representation of the length of the partition
   * @param lowerBoundedFlags For each window function call: is there a lower bound?
   * @param upperBoundedFlags For each window function call: Is there an upper bound?
   * @param lowerBounds String representation of the lower bound
   * @param upperBounds String representation of the upper bound
   * @param aggNames List of the names of the window functions being called
   * @param argsLists List of lists of arguments to each window function call
   * @param arraysToSort List that the output array name should be added to
   * @param aggOutputs List of names of arrays to store each window function output
   * @param zeroExpr String representation of 0
   *     <p>TODO [BE-3949]: allow slice-reusing
   */
  private static void generateLoopBasedWindowFn(
      StringBuilder funcText,
      final List<RelDataType> typs,
      final String partitionLength,
      final List<Boolean> lowerBoundedFlags,
      final List<Boolean> upperBoundedFlags,
      final List<String> lowerBounds,
      final List<String> upperBounds,
      final List<String> arraysToSort,
      final List<String> aggNames,
      final List<List<WindowedAggregationArgument>> argsLists,
      final List<String> aggOutputs,
      final String zeroExpr) {

    // Generate all of the arrays that will store the outputs
    for (int i = 0; i < aggNames.size(); i++) {

      addIndent(funcText, 2);
      funcText.append(aggOutputs.get(i)).append(" = ");
      switch (aggNames.get(i)) {

          // The functions that always have a float output
        case "KURTOSIS":
        case "SKEW":
          funcText.append(
              "bodo.libs.float_arr_ext.alloc_float_array(" + partitionLength + ", bodo.float64)\n");
          break;

          // The functions that always have a (positive) integer output
        case "COUNT(*)":
          funcText.append(
              "bodo.libs.int_arr_ext.alloc_int_array(" + partitionLength + ", bodo.uint32)\n");
          break;

          // The functions that always have output int64
        case "BITOR_AGG":
        case "BITAND_AGG":
        case "BITXOR_AGG":
          funcText.append(
              "bodo.libs.int_arr_ext.alloc_int_array(" + partitionLength + ", bodo.int64)\n");
          break;

          // Everything else has the same dtype as the input array
        default:
          funcText.append(sqlTypeToNullableBodoArray(partitionLength, typs.get(i)) + "\n");
      }
    }

    // This hashset contains the indices of the aggregations
    // which are performed on a constant window, IE
    // "Between Unbounded Preceding and Unbounded Following"
    // These aggregations take a special optimized path,
    // since we only have to compute the result once.
    HashSet<Integer> constant_slices = new HashSet<Integer>();
    for (int i = 0; i < aggNames.size(); i++) {
      Boolean lowerBounded = lowerBoundedFlags.get(i);
      Boolean upperBounded = upperBoundedFlags.get(i);
      if (lowerBounded || upperBounded) {
        continue;
      }
      assertWithErrMsg(argsLists.get(i).size() == 1, aggNames.get(i) + " requires 1 input");
      assertWithErrMsg(
          argsLists.get(i).get(0).isDfCol(), aggNames.get(i) + " requires a column input");

      constant_slices.add(i);

      // Set the entire array to zero if there are no non-null elements.
      addIndent(funcText, 2);
      funcText
          .append("if sorted_df[")
          .append(makeQuoted(argsLists.get(i).get(0).getExprString()))
          .append("].count() == 0:\n");
      addIndent(funcText, 3);
      funcText.append("for i in range(").append(partitionLength).append("):\n");
      addIndent(funcText, 4);
      funcText.append("bodo.libs.array_kernels.setna(").append(aggOutputs.get(i)).append(", i)\n");
      addIndent(funcText, 2);
      funcText.append("else:\n");
      addIndent(funcText, 1);

      // Set the entire array equal to the result of calling the aggregation
      // method on the entire input column
      addIndent(funcText, 2);
      String columnAggCall =
          String.format(
              windowFunctions.get(aggNames.get(i)),
              "sorted_df[" + makeQuoted(argsLists.get(i).get(0).getExprString()) + "]");
      funcText
          .append(aggOutputs.get(i))
          .append("[:] = bodo.utils.conversion.unbox_if_tz_naive_timestamp(")
          .append(columnAggCall)
          .append(")\n");
    }

    // Now loop across the rows and perform the slice-based aggregation for each
    // window function column that was not already handled above.
    // If there exist no such slice-based aggregations to handle, immediately return
    if (aggNames.size() == constant_slices.size()) {
      return;
    }

    addIndent(funcText, 2);
    funcText.append("for i in range(" + partitionLength + "):\n");

    for (int i = 0; i < aggNames.size(); i++) {
      if (constant_slices.contains(i)) {
        continue;
      }
      Boolean lowerBounded = lowerBoundedFlags.get(i);
      Boolean upperBounded = upperBoundedFlags.get(i);
      String lowerBound = lowerBounds.get(i);
      String upperBound = upperBounds.get(i);

      // Create cur_lower/upperBounds variables which will be used for slicing.
      // We can omit creating bounds in the unbounded case, and use the empty string
      if (!lowerBounded) {
        addIndent(funcText, 3);
        funcText.append("cur_lower_bound = 0\n");
      } else {
        addIndent(funcText, 3);
        funcText.append("cur_lower_bound = max(0, i + " + lowerBound + ")\n");
      }
      if (!upperBounded) {
        addIndent(funcText, 3);
        funcText.append("cur_upper_bound = " + partitionLength + "\n");
      } else {
        addIndent(funcText, 3);
        funcText.append("cur_upper_bound = max(0, i + " + upperBound + " + 1)\n");
      }

      // Handle count separately by taking the difference between the upper
      // bound and the lower bound
      if (aggNames.get(i) == "COUNT(*)") {

        addIndent(funcText, 3);
        funcText
            .append(aggOutputs.get(i))
            .append("[i] = max(0, cur_upper_bound - cur_lower_bound)\n");
        continue;
      }

      assertWithErrMsg(argsLists.get(i).size() == 1, aggNames.get(i) + " requires 1 input");
      assertWithErrMsg(
          argsLists.get(i).get(0).isDfCol(), aggNames.get(i) + " requires a column input");

      WindowedAggregationArgument arg0 = argsLists.get(i).get(0);
      String arg0ColName = arg0.getExprString();

      String sliceName = "slice" + String.valueOf(i);
      // TODO: port over as many of these functions as possible to sliding
      // window kernels since calculating the slice each time is expensive
      addIndent(funcText, 3);
      funcText
          .append(sliceName + " = sorted_df[")
          .append(makeQuoted(arg0ColName))
          .append("].iloc[");

      // TODO: if we ever want to allow non-scalar aggregation bounds,
      // this will need to be updated

      // Add the slice bounds. I.e., if we have a lower bound and an upper
      // bound: xxx.iloc[cur_lower_bound:cur_upper_bound]
      if (lowerBounded) {
        funcText.append("cur_lower_bound");
      }
      funcText.append(":");
      if (upperBounded) {
        funcText.append("cur_upper_bound");
      }
      funcText.append("]\n");

      // For all slice-based functions, if there is not at least 1 non-null entry in the slice, set
      // the output to NULL
      addIndent(funcText, 3);
      funcText.append("if " + sliceName + ".count() == 0:\n");
      addIndent(funcText, 4);
      funcText.append("bodo.libs.array_kernels.setna(" + aggOutputs.get(i) + ", i)\n");
      addIndent(funcText, 3);
      funcText.append("else:\n");
      addIndent(funcText, 1);

      // Call the Pandas method on the slice and store the output
      String columnAggCall = String.format(windowFunctions.get(aggNames.get(i)), sliceName);
      addIndent(funcText, 3);
      funcText
          .append(aggOutputs.get(i))
          .append("[i] = ")
          .append("bodo.utils.conversion.unbox_if_tz_naive_timestamp(")
          .append(columnAggCall)
          .append(")\n");
    }
  }

  /**
   * Generates the code for a LEAD/LAG computation
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param argsList List of all inputs to the current window function call
   * @param isLead Is this a call to LEAD or LAG
   * @param isRespectNulls Should nulls be respected or ignored
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateLeadLagAggFn(
      StringBuilder funcText,
      final List<WindowedAggregationArgument> argsList,
      final boolean isLead,
      final boolean isRespectNulls,
      final List<String> arraysToSort,
      final int i) {

    // Lead/Lag expects one required argument (a column) and two optional arguments
    // (an offset and a default value)
    int num_arguments = argsList.size();
    assertWithErrMsg(
        1 <= num_arguments && num_arguments <= 3,
        "Lead/Lag expects between 1 and 3 arguments, instead got: " + argsList.size());
    WindowedAggregationArgument aggColArg = argsList.get(0);
    assertWithErrMsg(aggColArg.isDfCol(), "Lead/Lag's first argument must be a column");

    Expr.StringLiteral aggColName = new Expr.StringLiteral(aggColArg.getExprString());

    // Default shift amount is 1
    Expr shiftAmount = new Expr.IntegerLiteral(1);
    Expr fillValue = Expr.None.INSTANCE;

    // If there are 2 (or 3) arguments, use the 2nd argument to extract the shift amount
    if (num_arguments >= 2) {
      WindowedAggregationArgument shiftAmountArg = argsList.get(1);
      assertWithErrMsg(
          !shiftAmountArg.isDfCol(),
          "Lead/Lag expects the offset to be a scalar literal, if it is provided. Got: "
              + argsList.toString());

      // TODO: Remove Raw
      shiftAmount = new Expr.Raw(shiftAmountArg.getExprString());

      // Add the default fill value (if it's present)
      if (num_arguments == 3) {
        WindowedAggregationArgument fillValueArg = argsList.get(2);
        // I don't know if this is handled within Calcite or not, so throwing it as a Bodo error
        if (fillValueArg.isDfCol()) {
          throw new BodoSQLCodegenException(
              "Error! Only scalar fill value is supported for LEAD/LAG");
        }
        // TODO: Remove Raw
        fillValue = new Expr.Raw(fillValueArg.getExprString());
      }
    }

    // If using lead, flip the sign
    if (isLead) {
      shiftAmount = new Expr.Unary("-", shiftAmount);
    }

    // TODO: Refactor all variable generation to use the temporary variables.
    // This is only safe because we are inside a closure and adhere to our own conventions
    Variable outputArray = new Variable("arr" + i);
    Expr.GetItem aggColRef = new Expr.GetItem(new Expr.Raw("sorted_df"), aggColName);

    if (isRespectNulls) {
      // If we are respecting nulls then we can call shift directly
      Expr.Method functionCall =
          new Expr.Method(
              aggColRef,
              "shift",
              List.of(shiftAmount),
              List.of(new kotlin.Pair<>("fill_value", fillValue)));
      // TODO: Switch to using the builder.
      addIndent(funcText, 2);
      funcText.append(outputArray.emit()).append(" = ").append(functionCall.emit()).append("\n");
    } else {
      // See the design for IGNORE NULLS here:
      // https://bodo.atlassian.net/wiki/spaces/~62c43badfa577c57c3b685b2/pages/1322745956/Ignore+Nulls+in+LEAD+LAG+design
      Expr.Call kernelCall =
          new Expr.Call(
              "bodo.libs.bodosql_array_kernels.null_ignoring_shift",
              List.of(aggColRef, shiftAmount, fillValue));
      addIndent(funcText, 2);
      funcText.append(outputArray.emit()).append(" = ").append(kernelCall.emit()).append("\n");
    }
    arraysToSort.add(outputArray.emit());
  }

  /**
   * Generates the code for a ROW_NUMBER() computation
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param argsList List of all inputs to the current window function call
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  public static void generateRowNumberFn(
      StringBuilder funcText,
      final List<WindowedAggregationArgument> argsList,
      final List<String> arraysToSort,
      final int i) {

    assertWithErrMsg(argsList.size() == 0, "ROW_NUMBER takes in no arguments");

    String arrName = "arr" + String.valueOf(i);

    // Generate the row_numbers.
    addIndent(funcText, 2);
    funcText.append(arrName).append(" = np.arange(1, " + partitionLength + " + 1)\n");

    arraysToSort.add(arrName);
  }

  /**
   * Helper function that handles the CONDITIONAL_TRUE_EVENT window aggregation.
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param argsList List of all inputs to the current window function call
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateTrueEventFn(
      StringBuilder funcText,
      final List<WindowedAggregationArgument> argsList,
      final List<String> arraysToSort,
      final int i) {

    assertWithErrMsg(argsList.size() == 1, "CONDITIONAL_TRUE_EVENT requires 1 input");
    assertWithErrMsg(argsList.get(0).isDfCol(), "CONDITIONAL_TRUE_EVENT requires a column input");

    String colName = argsList.get(0).getExprString();

    String arrName = "arr" + String.valueOf(i);

    // Generate the row_numbers.
    addIndent(funcText, 2);
    funcText
        .append(arrName)
        .append(" = sorted_df[" + makeQuoted(colName) + "].astype('uint32').cumsum()\n");

    arraysToSort.add(arrName);
  }

  /**
   * Helper function that handles the RANK window aggregations (except for ROW_NUMBER and NTILE)
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param argsList List of all inputs to the current window function call
   * @param sortByList Which columns were being used to sort by
   * @param agg Which rank funciton is being called
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateRankFns(
      StringBuilder funcText,
      final List<WindowedAggregationArgument> argsList,
      final List<String> sortByList,
      final SqlKind agg,
      final List<String> arraysToSort,
      final int i) {

    assertWithErrMsg(argsList.size() == 0, "RANK takes in no arguments");

    String arrName = "arr" + String.valueOf(i);

    // Adjust the method arguments based on which RANK function is being called
    String methodStr;
    if (agg == SqlKind.DENSE_RANK) {
      methodStr = "dense";
    } else if (agg == SqlKind.CUME_DIST) {
      methodStr = "max";
    } else {
      methodStr = "min";
    }
    String pctStr = agg == SqlKind.CUME_DIST ? "True" : "False";

    addIndent(funcText, 2);
    funcText.append(arrName).append("  = bodo.libs.bodosql_array_kernels.rank_sql((");

    // Add each sorting column to the tuple of arguments
    for (String col : sortByList) {
      funcText.append(
          String.format("bodo.hiframes.pd_series_ext.get_series_data(sorted_df[%s]), ", col));
    }
    funcText.append(String.format("), method=\"%s\", pct=%s)", methodStr, pctStr));

    /* If PERCENT_RANK, augment the calculation as follows:
     *
     * arr0 = [rank calculation] - 1
     * if argumentDfLen == 1:
     *   arr0[:] = 0.0
     * else:
     *   arr0 /= (argumentDfLen - 1)
     *
     */
    if (agg == SqlKind.PERCENT_RANK) {
      funcText.append(" - 1\n");
      addIndent(funcText, 2);
      funcText.append("if " + partitionLength + " == 1:\n");
      addIndent(funcText, 3);
      funcText.append(arrName).append("[:] = 0.0\n");
      addIndent(funcText, 2);
      funcText.append("else:\n");
      addIndent(funcText, 3);
      funcText.append(arrName).append(" /= (").append(partitionLength).append(" - 1)\n");
    } else {
      funcText.append("\n");
    }

    arraysToSort.add(arrName);
  }

  /**
   * Helper function that handles the NTILE window aggregation.
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param argsList List of all inputs to the current window function call
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateNtileFn(
      StringBuilder funcText,
      final List<WindowedAggregationArgument> argsList,
      final List<String> arraysToSort,
      final int i) {

    assertWithErrMsg(argsList.size() == 1, "NTILE requires 1 input");
    assertWithErrMsg(!argsList.get(0).isDfCol(), "NTILE requires a scalar input");

    // Extract how many bins to divide the output into
    WindowedAggregationArgument numBinsArg = argsList.get(0);
    String numBins = numBinsArg.getExprString();
    String arrName = "arr" + String.valueOf(i);

    // Use the helper funciton to generate the corresponding output array
    addIndent(funcText, 2);
    funcText
        .append(arrName)
        .append(" = bodosql.libs.ntile_helper.ntile_helper(")
        .append(partitionLength)
        .append(", ")
        .append(numBins)
        .append(")\n");

    arraysToSort.add(arrName);
  }

  /**
   * Helper function that handles window aggregations that have dedicated kernels.
   *
   * @param funcText String buffer where the output closure's codegen is being stored
   * @param argsList List of all inputs to the current window function call
   * @param fnName The name of the kernel. Must be in windowCodeExpressions.
   * @param lowerBound String representation of the lower bound of the window frame
   * @param upperBound String representation of the upper bound of the window frame
   * @param arraysToSort List that the output array name should be added to
   * @param i Which window aggregation (within the closure) are we dealing with
   */
  private static void generateSimpleWindowFnCode(
      StringBuilder funcText,
      final List<WindowedAggregationArgument> argsList,
      final String fnName,
      final String lowerBound,
      final String upperBound,
      final List<String> arraysToSort,
      final int i) {

    Boolean twoArgWindowOptimized = twoArgWindowOptimizedKernels.contains(fnName);
    Boolean windowOptimized = twoArgWindowOptimized || windowOptimizedKernels.contains(fnName);
    // This should be checked by the calling function
    String kernelName = windowCodeExpressions.get(fnName);

    if (twoArgWindowOptimized) {
      if (argsList.size() != 2) {
        throw new BodoSQLCodegenException(fnName + " requires 2 column inputs");
      }
    }
    String arrName = "arr" + String.valueOf(i);

    // Add the kernel call
    addIndent(funcText, 2);
    funcText
        .append(arrName)
        .append(" = bodo.libs.bodosql_array_kernels.")
        .append(kernelName)
        .append("(");

    if (argsList.size() == 0) {
      // COUNT(*) has no arguments and inserts the length of the window.
      funcText.append("argument_df_len");
    } else {
      for (int j = 0; j < argsList.size(); j++) {
        if (j != 0) {
          funcText.append(", ");
        }
        funcText
            .append("sorted_df[")
            .append(makeQuoted(argsList.get(j).getExprString()))
            .append("]");
      }
    }

    if (windowOptimized) {
      funcText.append(", ").append(lowerBound).append(", ").append(upperBound);
    }

    funcText.append(")\n");

    arraysToSort.add(arrName);
  }

  /**
   * Helper function that generates the DataFrame to be returned by the groupby apply lambda
   * function. This helper function also revert the sorts of the array containing the returned data,
   * if needs be.
   *
   * @param arrsToSort the list arrays that needs to be sorted/returned
   * @param sorted_df_name the name of the DataFrame to be sorted/returned. Must contain
   *     revertSortColumnName if a sort reversion is needed.
   * @param expectedOutputColNames the list of string column names in which to store each of the
   *     arrays in the returned DataFrame
   * @param needsRevertSort Does the above array need to be revert sorted before return. We need to
   *     revert the sort of the returned array if we had to sort the input data.
   * @return returns a string that contains the input columns stored in the specified output column
   *     names, with the sort reverted (if necessary).
   */
  private static Pair<String, String> revertSortLocalDfIfNeeded(
      final List<String> arrsToSort,
      final String sorted_df_name,
      final List<String> expectedOutputColNames,
      final boolean needsRevertSort) {

    assertWithErrMsg(
        arrsToSort.size() == expectedOutputColNames.size(),
        "The number of arrays to sort should be equivalent to the number of expected output"
            + " columns");

    // TODO: if arrsToSort.size() == 1, we could return an array instead of a DataFrame.
    // see BS-616

    StringBuilder funcText = new StringBuilder();
    List<String> outputCols = new ArrayList<>();
    // If we didn't have to do a sort on the input data, we can just return the arrays wrapped in a
    // DataFrame.
    if (!needsRevertSort) {
      addIndent(funcText, 2);
      funcText.append("retval = pd.DataFrame({");
      for (int i = 0; i < arrsToSort.size(); i++) {
        outputCols.add(expectedOutputColNames.get(i));
        funcText.append(
            makeQuoted(expectedOutputColNames.get(i)) + ": " + arrsToSort.get(i) + ", ");
      }

      funcText.append("}, index = " + argumentDfOriginalIndex + ")\n");
    } else {
      // If we did need to do a sort on the DataFrame, we need to revert the sort
      // on the output column(s) before returning them.
      addIndent(funcText, 2);
      funcText.append("_tmp_sorted_df = pd.DataFrame({");
      for (int i = 0; i < arrsToSort.size(); i++) {
        String curArr = arrsToSort.get(i);
        funcText.append(makeQuoted(expectedOutputColNames.get(i)) + ": " + curArr + ", ");
      }

      funcText
          .append(
              makeQuoted(revertSortColumnName)
                  + ": "
                  + sorted_df_name
                  + "["
                  + makeQuoted(revertSortColumnName)
                  + "]})")
          .append(".sort_values(by=[")
          .append(makeQuoted(revertSortColumnName))
          .append("], ascending=[True])\n");

      addIndent(funcText, 2);
      funcText.append("retval = pd.DataFrame({");

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
   * Generates the code to prune the columns of the input DataFrame
   *
   * @param funcText the string builder where we store the closure
   * @param aggNames the list of function names
   * @param argsListList the list of arguments to each window function
   * @param hasOrder whether or not there is an ORDER BY clause
   * @param sortByCols the columns we need to sort by, as a string
   */
  public static void pruneColumns(
      final StringBuilder funcText,
      final List<String> aggNames,
      final List<List<WindowedAggregationArgument>> argsListList,
      final boolean hasOrder,
      final String sortByCols) {

    /**
     * The columns that we need to keep are as follows: If we have a column on which we are
     * performing an aggregation (IE, MAX(A)), we need to keep that column If we have any columns by
     * which we need to sort, we need to keep those columns, and the ORIG_POSITION_COL which is
     * needed for the sort reversion.
     */
    StringBuilder kept_cols = new StringBuilder();

    // First, add the aggregation columns to the list of kept columns.
    for (int i = 0; i < argsListList.size(); i++) {

      // If we have a column argument, it is always the 0-th argument except for
      // the functions in twoArgWindowOptimizedKernels
      String colName;
      if (twoArgWindowOptimizedKernels.contains(aggNames.get(i))) {
        colName = argsListList.get(i).get(0).getExprString();
        kept_cols.append(makeQuoted(colName)).append(", ");
        colName = argsListList.get(i).get(1).getExprString();
        kept_cols.append(makeQuoted(colName)).append(", ");
      } else if (argsListList.get(i).size() > 0 && argsListList.get(i).get(0).isDfCol()) {
        colName = argsListList.get(i).get(0).getExprString();
        kept_cols.append(makeQuoted(colName)).append(", ");
      }
    }

    // Add sortbycols, removing the enclosing brackets, and add the original position column,
    // which is needed for the sort reversion.
    if (hasOrder) {
      kept_cols
          .append(makeQuoted(revertSortColumnName) + ", ")
          // sortbycols is passed as a string that looks like "['A', 'B', 'C']", so this substring
          // just removes the outer brackets so that we can add the columns to the new list needed
          // for the call to loc
          .append(sortByCols.substring(1, sortByCols.length() - 1));
    }

    // Drop unneeded columns.
    addIndent(funcText, 2);
    if (!kept_cols.toString().equals("")) {
      funcText.append(
          argumentDfName + " = " + argumentDfName + ".loc[:, [" + kept_cols.toString() + "]]\n");
    } else {
      // In the case that kept_cols is none, we have to do the slicing with iloc, as numba has a
      // typing error with the
      // empty column list
      funcText.append(argumentDfName + " = " + argumentDfName + ".iloc[:, :0]\n");
    }
  }

  /**
   * Takes an input DataFrame, performs a sort on it (if needed), and stores it to the specified
   * output variable.
   *
   * @param input_df_name The name of the DataFrame to sort.
   * @param output_sorted_df_name The variable name where the output DataFrame will be stored.
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
    // Currently we are appending a column (ORIG_COL_POSITION) that keeps track of
    // the original positions. Then, in the groupby, if we sort each of the partitioned
    // DataFrames on each rank, before returning the sorted DataFrame it might be
    // faster to do one sort on the entire data, instead of a sort on each of
    // the partitioned DataFrames.
    StringBuilder sortText = new StringBuilder();
    if (!sortByCols.equals("")) {
      addIndent(sortText, 2);
      sortText
          .append(output_sorted_df_name + " = " + input_df_name + ".sort_values(by=")
          .append(sortByCols)
          .append(", ascending=")
          .append(ascendingList)
          .append(", na_position=")
          .append(NAPositionList)
          .append(")\n");
    } else {
      addIndent(sortText, 2);
      sortText.append(output_sorted_df_name + " = " + input_df_name + "\n");
    }

    return sortText.toString();
  }

  /**
   * Determine if this group of window operations can be computed in the Bodo-Engine optimized C++
   * kernel via groupby.window. Only supported if a partition is used and all the functions in the
   * set are from a limited collection that are supported in groupby.window.
   *
   * @param aggOperations The list of RexOver operations that are being computed.
   * @return Can these operations take the optimized C++ kernel path.
   */
  public static boolean usesGroupbyWindow(List<RexOver> aggOperations) {
    for (RexOver agg : aggOperations) {
      String fnName = agg.getAggOperator().getName();
      if (agg.getWindow().partitionKeys.size() == 0
          || !optimizedEngineKernelFunctions.contains(fnName)) {
        return false;
      }
    }
    return true;
  }

  /**
   * Generates the codegen for the Bodo-Engine optimized C++ kernel via groupby.window on a
   * collection of window functions using the same partition/orderby terms. The codegen will produce
   * a DataFrame whose columns contain the answers. The funciton returns the names of these columns.
   *
   * @param inputVar Name of the DataFrame containing the input data.
   * @param outputVar Name of the output DataFrame where the answers are to be stored.
   * @param aggOperations List of the window functions that are being computed. Importantly, these
   *     are all expected to have the same partition/orderby terms since this funciton should be
   *     called after window fusion.
   * @param childExprs List of expressions for any child columns that need to be projected to
   *     calculate the partition/orderby terms
   * @param childExprTypes ExprType for each childExpr being projected.
   * @param numColumnArgs Number of column arguments that the window aggregation functions take in.
   * @param builder The builder for appending generated code.
   * @param translator A ctx object containing the Hashset of columns used that need null handling,
   *     the List of precomputed column variables that need to be added to the dataframe before an
   *     apply, and the list of named parameters that need to be passed to an apply function as
   *     arguments.
   * @return A list of the names of the columsn of each of the answer within the output DataFrame.
   */
  public static List<String> generateGroupbyWindow(
      String inputVar,
      Variable outputVar,
      List<RexOver> aggOperations,
      List<Expr> childExprs,
      List<BodoSQLExprType.ExprType> childExprTypes,
      int numColumnArgs,
      Module.Builder builder,
      RexToPandasTranslator translator) {
    RexOver windowFunc = aggOperations.get(0);
    RexWindow window = windowFunc.getWindow();
    List<String> childColNames = new ArrayList<>();
    List<Expr> groupByColNames = new ArrayList<>();

    // Generate intermediate names for each partition keys
    int col_id_var = 0;
    for (int i = 0; i < window.partitionKeys.size(); i++) {
      String colName = "GRPBY_COL_" + col_id_var++;
      groupByColNames.add(new Expr.StringLiteral(colName));
      childColNames.add(colName);
    }
    ImmutableList<RexFieldCollation> orderbyKeys = window.orderKeys;
    int numOrderByKeys = orderbyKeys.size();
    List<Expr.StringLiteral> orderByLiteralList = new ArrayList<>(numOrderByKeys);
    List<Expr.BooleanLiteral> ascendingList = new ArrayList<>(numOrderByKeys);
    List<Expr.StringLiteral> NAPositionList = new ArrayList<>(numOrderByKeys);

    for (int i = 0; i < numOrderByKeys; i++) {
      final String orderByColName;
      RelFieldCollation.Direction dir = window.orderKeys.get(i).getDirection();
      RelFieldCollation.NullDirection nullDir = window.orderKeys.get(i).getNullDirection();
      if (dir == RelFieldCollation.Direction.ASCENDING) {
        orderByColName = "ASC_COL_" + col_id_var;
      } else {
        assert dir == RelFieldCollation.Direction.DESCENDING;
        orderByColName = "DEC_COL_" + col_id_var;
      }
      childColNames.add(orderByColName);
      orderByLiteralList.add(new Expr.StringLiteral(orderByColName));
      ascendingList.add(getAscendingExpr(dir));
      NAPositionList.add(getNAPositionStringLiteral(nullDir));
      col_id_var++;
    }

    // Give each window function column argument its own name
    int argColOffset = childColNames.size();
    for (int i = 0; i < numColumnArgs; i++) {
      String argColName = "ARG_COL_" + i;
      childColNames.add(argColName);
    }

    // Generate a projection in case we needed to compute anything for the orderby/partition by
    // columns
    String projection =
        generateProjectedDataframe(inputVar, childColNames, childExprs, childExprTypes);

    Expr.Raw projectionExpr = new Expr.Raw(projection);
    // Generate the groupby
    Expr.List groupbyKeys = new Expr.List(groupByColNames);
    Expr.Groupby groupby = new Expr.Groupby(projectionExpr, groupbyKeys, false, false);
    List<Expr> functionNamesAndArgs = new ArrayList<Expr>();
    for (RexOver agg : aggOperations) {
      List<Expr> tupleArgs = new ArrayList<Expr>();
      Expr nameExpr =
          new Expr.StringLiteral(agg.getAggOperator().getName().toLowerCase(Locale.ROOT));
      tupleArgs.add(nameExpr);
      switch (agg.getOperator().getName()) {
        case "CONDITIONAL_TRUE_EVENT":
        case "CONDITIONAL_CHANGE_EVENT":
          {
            // Functions where the only argument is 1 column argument
            tupleArgs.add(new Expr.StringLiteral(childColNames.get(argColOffset)));
            argColOffset++;
            break;
          }
        default:
          {
            // Default behavior: loop through all arguments and add them to the tuple
            // by emitting their literal string
            for (RexNode operand : agg.getOperands()) {
              Expr curInfo = operand.accept(translator);
              WindowedAggregationArgument arg =
                  WindowedAggregationArgument.fromLiteralExpr(curInfo.emit());
              tupleArgs.add(new Expr.StringLiteral(arg.getExprString()));
            }
            break;
          }
      }
      functionNamesAndArgs.add(new Expr.Tuple(tupleArgs));
    }
    // Generate the Window call
    Expr.Method windowExpr =
        new Expr.Method(
            groupby,
            "window",
            List.of(
                new Expr.Tuple(functionNamesAndArgs),
                new Expr.Tuple(orderByLiteralList),
                new Expr.Tuple(ascendingList),
                new Expr.Tuple(NAPositionList)),
            List.of());

    // Add the window function to the codegen
    Op.Assign assignment = new Op.Assign(outputVar, windowExpr);
    builder.addToMainFunction(assignment);
    List<String> outputs = new ArrayList<String>();
    for (int i = 0; i < aggOperations.size(); i++) {
      String outputColName = "AGG_OUTPUT_" + String.valueOf(i);
      outputs.add(outputColName);
    }
    return outputs;
  }
}
