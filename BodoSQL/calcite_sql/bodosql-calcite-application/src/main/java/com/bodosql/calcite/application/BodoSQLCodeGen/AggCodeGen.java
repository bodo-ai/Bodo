package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.Utils.AggHelpers.generateGroupByCall;
import static com.bodosql.calcite.application.Utils.AggHelpers.getCountCall;
import static com.bodosql.calcite.application.Utils.AggHelpers.getDummyColName;
import static com.bodosql.calcite.application.Utils.Utils.assertWithErrMsg;
import static com.bodosql.calcite.application.Utils.Utils.getBodoIndent;
import static com.bodosql.calcite.application.Utils.Utils.getInputColumn;
import static com.bodosql.calcite.application.Utils.Utils.isValidPythonIdentifier;
import static com.bodosql.calcite.application.Utils.Utils.makeQuoted;
import static com.bodosql.calcite.application.Utils.Utils.renameColumns;

import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.Pair;

/**
 * Class that returns the generated code for an Agg expression after all inputs have been visited.
 */
public class AggCodeGen {

  /* Hashmap of aggregation functions for which there is a one to one mapping between the SQL Function,
    and a pandas method call in the form of Col_expr.Ag_fn(), and df.agg(col_name = pd.NamedAgg(column='B', aggfunc="ag_fn"))
  */
  static HashMap<SqlKind, String> equivalentPandasMethodMap;

  static HashMap<SqlKind, String> equivalentNumpyFuncMap;

  static HashMap<String, String> equivalentPandasNameMethodMap;

  static HashMap<String, String> equivalentNumpyFuncNameMap;

  static HashMap<String, String> equivalentHelperFnMap;

  static {
    equivalentPandasMethodMap = new HashMap<>();
    equivalentNumpyFuncMap = new HashMap<>();
    equivalentPandasNameMethodMap = new HashMap<>();
    equivalentNumpyFuncNameMap = new HashMap<>();
    equivalentHelperFnMap = new HashMap<>();

    equivalentPandasMethodMap.put(SqlKind.SUM, "sum");
    equivalentPandasMethodMap.put(SqlKind.SUM0, "sum");
    equivalentPandasMethodMap.put(SqlKind.MIN, "min");
    equivalentPandasMethodMap.put(SqlKind.MAX, "max");
    equivalentPandasMethodMap.put(SqlKind.AVG, "mean");
    equivalentPandasMethodMap.put(SqlKind.MEDIAN, "median");
    equivalentPandasMethodMap.put(SqlKind.STDDEV_SAMP, "std");
    equivalentPandasMethodMap.put(SqlKind.VAR_SAMP, "var");
    equivalentPandasMethodMap.put(SqlKind.ANY_VALUE, "iloc");

    equivalentNumpyFuncMap.put(SqlKind.BIT_AND, "np.bitwise_and.reduce");
    equivalentNumpyFuncMap.put(SqlKind.BIT_XOR, "np.bitwise_xor.reduce");
    equivalentNumpyFuncMap.put(SqlKind.VAR_POP, "np.var");
    equivalentNumpyFuncMap.put(SqlKind.STDDEV_POP, "np.std");

    equivalentPandasNameMethodMap.put("COUNT_IF", "count_if");
    equivalentPandasNameMethodMap.put("KURTOSIS", "kurtosis");
    equivalentPandasNameMethodMap.put("SKEW", "skew");
    equivalentPandasNameMethodMap.put("VARIANCE_SAMP", "var");

    equivalentNumpyFuncNameMap.put("VARIANCE_POP", "np.var");

    equivalentPandasNameMethodMap.put("KURTOSIS", "kurtosis");
    equivalentPandasNameMethodMap.put("SKEW", "skew");
    equivalentHelperFnMap.put("BOOLOR_AGG", "boolor_agg");
    equivalentHelperFnMap.put("BOOLAND_AGG", "booland_agg");
    equivalentHelperFnMap.put("BOOLXOR_AGG", "boolxor_agg");
    equivalentHelperFnMap.put("BITOR_AGG", "bitor_agg");
    // Calcite's SINGLE_VALUE returns input if it has only one value, otherwise raises an error
    // https://github.com/apache/calcite/blob/f14cf4c32b9079984a988bbad40230aa6a59b127/core/src/main/java/org/apache/calcite/sql/fun/SqlSingleValueAggFunction.java#L36
    equivalentHelperFnMap.put(
        "SINGLE_VALUE", "bodo.libs.bodosql_array_kernels.ensure_single_value");
    equivalentHelperFnMap.put("APPROX_PERCENTILE", "approx_percentile");
  }

  /**
   * Function that generates the code for an aggregate expression that does not require a Group By.
   *
   * @param inVar The input variable.
   * @param inputColumnNames The names of the columns of the input var.
   * @param aggCallList The list of aggregations to be performed.
   * @param aggCallNames The list of column names to be used for the output of the aggregations
   * @param distOutput Is the output single row DataFrame distributed or replicated. When no group
   *     is used as 1 step in an aggregation (e.g. group by cube), then the output is distributed.
   *     When it is the only aggregation group across the entire output (e.g. select SUM(A) from
   *     table1), then it is replicated.
   * @return The code generated for the aggregation.
   */
  public static Expr generateAggCodeNoGroupBy(
      Variable inVar,
      List<String> inputColumnNames,
      List<AggregateCall> aggCallList,
      List<String> aggCallNames,
      boolean distOutput) {
    // Generates code like: pd.DataFrame({"sum(A)": [test_df1["A"].sum()], "mean(B)":
    // [test_df1["A"].mean()]})
    // Generate any filters. This is done on a separate line for simpler
    // code in case the series is empty.

    StringBuilder aggString = new StringBuilder();

    aggString.append("pd.DataFrame({");
    for (int i = 0; i < aggCallList.size(); i++) {
      // Generate the filter. This is done on a separate line for simpler
      // code in case the series is empty.
      // TODO: Avoid conflict with other Relnodes? Not necessary for correctness
      // because these values are use once
      AggregateCall a = aggCallList.get(i);
      String outColName;
      // Determine if we need to apply a filter
      String filterCol = "";
      if (a.filterArg != -1) {
        filterCol = inputColumnNames.get(a.filterArg);
      }

      StringBuilder seriesBuilder = new StringBuilder();
      seriesBuilder.append(inVar.emit());
      if (!(a.getAggregation().getKind() == SqlKind.COUNT && a.getArgList().isEmpty())) {
        // If we are performing a COUNT(*) then we avoid selecting a single series since
        // we may be able to compute a length without a particular column.
        // Get the input column.
        String aggCol = getInputColumn(inputColumnNames, a, new ArrayList());

        // First, construct the filtered series
        // TODO: Refactor the series to only produce unique column + filter pairs
        seriesBuilder.append("[").append(makeQuoted(aggCol)).append("]");
      }
      if (filterCol.length() > 0) {
        seriesBuilder
            .append("[")
            .append(inVar.emit())
            .append("[")
            .append(makeQuoted(filterCol))
            .append("]]");
      }

      // Get the output column name
      outColName = aggCallNames.get(i);
      aggString.append(makeQuoted(outColName)).append(": ");

      Pair<String, Boolean> funcInfo = getAggFuncInfo(a, false);
      String aggFunc = funcInfo.getKey();
      boolean isMethod = funcInfo.getValue();

      // We need an optional type in case the series is empty.
      if (a.getAggregation().getKind() != SqlKind.COUNT
          && a.getAggregation().getKind() != SqlKind.SUM0
          && a.getAggregation().getKind() != SqlKind.SINGLE_VALUE
          && !aggFunc.equals("count_if")) {
        aggString.append("bodosql.libs.null_handling.null_if_not_flag(");
      }

      // If the aggregation function is ANY_VALUE, manually alter syntax
      // to use brackets
      if (aggFunc.equals("iloc")) {
        aggString.append(seriesBuilder).append(".iloc[0]");
      } else if (aggFunc.equals("np.var")) {
        aggString.append(seriesBuilder).append(".var(ddof=0)");
      } else if (aggFunc.equals("np.std")) {
        aggString.append(seriesBuilder).append(".std(ddof=0)");
      } else if (aggFunc.equals("count_if")) {
        aggString.append(seriesBuilder).append(".sum()");
      } else if (aggFunc.equals("boolor_agg")
          || aggFunc.equals("booland_agg")
          || aggFunc.equals("boolxor_agg")
          || aggFunc.equals("bitor_agg")) {
        aggString
            .append("bodo.libs.array_kernels.")
            .append(aggFunc)
            .append("(")
            .append(seriesBuilder)
            .append(".values)");
      } else if (aggFunc.equals("approx_percentile")) {
        assertWithErrMsg(a.getArgList().size() == 2, "APPROX_PERCENTILE requires two arguments");
        // Currently, the scalar float argument is converted into a column. To
        // access the quantile value, extract the first row.
        String quantile_str =
            inVar.emit()
                + "["
                + makeQuoted(inputColumnNames.get(a.getArgList().get(1)))
                + "].iloc[0]";
        // TODO: confirm that the second argument is a float
        assertWithErrMsg(true, "The second argument to APPROX_PERCENTILE must be a scalar float");
        // TODO: confirm that the second argument is between zero and one
        assertWithErrMsg(
            true, "The second argument to APPROX_PERCENTILE must be between 0.0 and 1.0");
        aggString
            .append("bodo.libs.array_kernels.approx_percentile")
            .append("(")
            .append(seriesBuilder)
            .append(".values")
            .append(", ")
            .append(quantile_str)
            .append(")");

      } else {
        if (!isMethod) {
          // If we have a function surround the column
          aggString.append(aggFunc).append("(");
        }
        // append the column and if necessary filter
        aggString.append(seriesBuilder);
        if (isMethod) {
          // If we have a method do the method call instead.
          // We currently don't support any extra arguments
          aggString.append(".").append(aggFunc).append("(");
        }
        // Both func and method need a closing )
        aggString.append(")");
      }

      if (a.getAggregation().getKind() != SqlKind.COUNT
          && a.getAggregation().getKind() != SqlKind.SUM0
          && a.getAggregation().getKind() != SqlKind.SINGLE_VALUE
          && !aggFunc.equals("count_if")) {
        // We need an optional type in case the series is empty
        aggString.append(", len(").append(seriesBuilder).append(") > 0)");
      }

      aggString.append(", ");
    }
    // Aggregation without groupby should always have one element.
    // To force this value to be replicated (the correct output),
    // we use coerce to array.
    aggString.append("}, index=");
    if (distOutput) {
      aggString.append("bodo.hiframes.pd_index_ext.init_range_index(0, 1, 1, None)");
    } else {
      aggString.append(
          "bodo.hiframes.pd_index_ext.init_numeric_index(bodo.utils.conversion.coerce_to_array([0]))");
    }
    aggString.append(")");
    return new Expr.Raw(aggString.toString());
  }

  /**
   * Function that generates the code for an aggregate expression that does not include any
   * aggregations. This is equivalent to Select Distinct, on the current grouped columns IE,
   * dropping all the duplicates. All other columns present in the total grouping set are set to
   * null.
   *
   * @param inVar The input variable.
   * @param inputColumnNames The columns present in the input columns
   * @param group This list of column indices by which we are grouping
   * @return The code generated for the aggregation expression.
   */
  public static Expr generateAggCodeNoAgg(
      Variable inVar, List<String> inputColumnNames, List<Integer> group) {
    StringBuilder aggString = new StringBuilder();

    // Need to select a subset of columns to drop duplicates from.

    if (group.size() > 0) {
      StringBuilder neededColsIxd = new StringBuilder("[");

      for (int i = 0; i < group.size(); i++) {
        Integer idx = group.get(i);
        neededColsIxd.append(idx).append(", ");
      }
      neededColsIxd.append("]");

      aggString.append(inVar.emit());

      // First, prune unneeded columns, if they exist. This ensures that columns not being grouped
      // will be filled with null
      // when doing the concatenation
      if (group.size() < inputColumnNames.size()) {
        aggString.append(".iloc[:, ").append(neededColsIxd).append("]");
      }
      aggString.append(".drop_duplicates()");
    } else {
      // If we're grouping by no columns with no aggregations, the expected
      // output for this group is one row of all NULL's. In order to match this behavior, we create
      // a dataframe
      // with a length of one, with no columns. When doing the concat, the rows present in
      // the other dataframes will be populated with NULL values.
      aggString.append(
          "pd.DataFrame(index=bodo.hiframes.pd_index_ext.init_range_index(0, 1, 1, None))");
    }

    return new Expr.Raw(aggString.toString());
  }

  /**
   * Function that generates the code for an aggregate expression that requires a Group By. This
   * code has a side effect of filling outputColumnNames with the column names generated for outVar.
   *
   * @param inVar The input variable.
   * @param group Indices of the columns to group by for the current aggregation.
   * @param inputColumnNames The names of the columns of the input var.
   * @param aggCallList The list of aggregations to be performed.
   * @param aggCallNames The list of column names in which to store the outputs of the aggregation
   * @return The code generated for the aggregation.
   */
  public static Expr generateAggCodeWithGroupBy(
      Variable inVar,
      List<String> inputColumnNames,
      List<Integer> group,
      List<AggregateCall> aggCallList,
      final List<String> aggCallNames) {
    StringBuilder aggString = new StringBuilder();
    aggString.append(inVar.emit());

    // Generate the Group By section
    aggString.append(generateGroupByCall(inputColumnNames, group));

    /*
     * create the corresponding aggregation string using named aggregate syntax with tuples.
     * e.g. .agg(out1=pd.NamedAgg(column="in1", aggfunc="sum"), out2=pd.NamedAgg(column="in2", aggfunc="sum"),
     * out3=pd.NamedAgg(column="in1", aggfunc="mean"))
     */
    aggString.append(".agg(");
    HashMap<String, String> renamedAggColumns = new HashMap<>();
    for (int i = 0; i < aggCallList.size(); i++) {
      AggregateCall a = aggCallList.get(i);
      String aggCol = getInputColumn(inputColumnNames, a, group);
      String outputCol = aggCallNames.get(i);
      // Generate a dummy column to prevent syntax issues with names that aren't
      // supported by Pandas NamedAgg. If the name is a valid Python identifier
      // we don't need a rename
      String tempName = outputCol;
      if (!isValidPythonIdentifier(outputCol)) {
        tempName = getDummyColName(i);
        renamedAggColumns.put(tempName, outputCol);
      }

      Pair<String, Boolean> funcInfo = getAggFuncInfo(a, true);
      String aggFunc = funcInfo.getKey();
      // When inside of a Group By, use .iloc[0] instead of .head(1)[0]
      if (aggFunc == "iloc") {
        aggFunc = "first";
      }
      if (!(aggFunc.equals("np.var") || aggFunc.equals("np.std"))) {
        aggFunc = makeQuoted(aggFunc);
      }
      if (aggFunc.equals("approx_percentile")) {
        throw new BodoSQLCodegenException("APPROX_PERCENTILE not supported with Group By yet");
      }

      aggString
          .append(tempName)
          .append("=pd.NamedAgg(column=")
          .append(makeQuoted(aggCol))
          .append(", aggfunc=")
          .append(aggFunc)
          .append("),");
    }
    aggString.append(")");
    if (renamedAggColumns.size() > 0) {
      aggString.append(".rename(columns=");
      aggString.append(renameColumns(renamedAggColumns));
      aggString.append(", copy=False)");
    }
    return new Expr.Raw(aggString.toString());
  }

  /**
   * Function that generates the code for a Group By aggregation expression that requires a group by
   * apply. Returns a pair of Strings. The first is the group by apply aggregation expression, the
   * second is a function definition, which must be appended to the generated code prior to the
   * group by apply.
   *
   * @param inVar The input variable.
   * @param inputColumnNames The names of the columns of the input var.
   * @param group Indices of the columns to group by for the current aggregation.
   * @param aggCallList The list of aggregations to be performed.
   * @param aggCallNames The column names into which to store the outputs of the aggregation
   * @param funcVar Variable for the function generated for df.apply.
   * @return A pair of the code expression generated for the aggregation, and the function
   *     definition that is used in the groupby apply.
   */
  public static Pair<Expr, Op> generateApplyCodeWithGroupBy(
      Variable inVar,
      List<String> inputColumnNames,
      List<Integer> group,
      List<AggregateCall> aggCallList,
      List<String> aggCallNames,
      Variable funcVar) {
    StringBuilder fnString = new StringBuilder();

    final String indent = getBodoIndent();
    final String funcIndent = indent + indent;

    /*
     * First we generate the closure that will be used in the apply. This
     * will compute each operation on a line and return a series, using
     * pd.Index to name the output.
     */
    fnString.append(indent).append("def ").append(funcVar.getName()).append("(df):\n");

    ArrayList<String> seriesArgs = new ArrayList<>();
    ArrayList<String> indexNames = new ArrayList<>();
    for (int i = 0; i < aggCallList.size(); i++) {
      AggregateCall a = aggCallList.get(i);
      // Get the input column
      String aggCol = getInputColumn(inputColumnNames, a, group);

      // Determine the filter column if necessary
      String filterCol = "";
      if (a.filterArg != -1) {
        filterCol = inputColumnNames.get(a.filterArg);
      }

      // Generate the output column names
      String outputCol = aggCallNames.get(i);
      indexNames.add(outputCol);
      // Generate the new line
      String seriesVar = String.format("S%d", i);
      String newVar = String.format("var%d", i);

      // Append the var name to the series
      seriesArgs.add(newVar);

      Pair<String, Boolean> funcInfo = getAggFuncInfo(a, false);
      String aggFunc = funcInfo.getKey();
      boolean isMethod = funcInfo.getValue();

      // Generate the filter. This is done on a separate line for simpler
      // code in case the series is empty.
      fnString.append(funcIndent).append(seriesVar).append(" = ");
      // append the column and if necessary filter
      // TODO: Refactor the series to only produce unique column + filter pairs
      fnString.append("df[").append(makeQuoted(aggCol)).append("]");
      if (filterCol.length() > 0) {
        fnString.append("[df[").append(makeQuoted(filterCol)).append("]]");
      }
      fnString.append("\n");
      // Generate the call
      fnString.append(funcIndent).append(newVar).append(" = ");
      if (filterCol.length() > 0
          && a.getAggregation().getKind() != SqlKind.COUNT
          && a.getAggregation().getKind() != SqlKind.SUM0
          && a.getAggregation().getKind() != SqlKind.SINGLE_VALUE
          && !aggFunc.equals("count_if")) {
        // We need an optional type in case the series is empty. Since this is
        // groupby we must have a filter for this to occur.
        fnString.append("bodosql.libs.null_handling.null_if_not_flag(");
      }
      // If the aggregation function is ANY_VALUE, manually alter syntax
      // to use brackets
      if (aggFunc.equals("iloc")) {
        fnString.append(seriesVar);
        fnString.append(".iloc[0]");
      } else if (aggFunc.equals("var_pop")) {
        fnString.append(seriesVar);
        fnString.append(".var(ddof=0)");
      } else if (aggFunc.equals("count_if")) {
        fnString.append(seriesVar);
        fnString.append(".sum()");
      } else {
        if (!isMethod) {
          // If we have a function surround the column
          fnString.append(aggFunc).append("(");
        }
        // append the column var and if necessary filter
        fnString.append(seriesVar);
        if (isMethod) {
          // If we have a method do the method call instead.
          // We currently don't support any extra arguments
          fnString.append(".").append(aggFunc).append("(");
        }
        // Both func and method need a closing )
        fnString.append(")");
      }
      if (filterCol.length() > 0
          && a.getAggregation().getKind() != SqlKind.COUNT
          && a.getAggregation().getKind() != SqlKind.SUM0
          && a.getAggregation().getKind() != SqlKind.SINGLE_VALUE
          && !aggFunc.equals("count_if")) {
        // We need an optional type in case the series is empty. Since this is
        // groupby we must have a filter for this to occur.
        fnString.append(", len(").append(seriesVar).append(") > 0)");
      }
      // Both func and method need a closing )
      fnString.append("\n");
    }
    // Generate the output Series
    fnString.append(funcIndent).append("return pd.Series((");
    for (String arg : seriesArgs) {
      fnString.append(arg).append(", ");
    }
    fnString.append("), index=pd.Index((");
    for (String name : indexNames) {
      fnString.append(makeQuoted(name)).append(", ");
    }
    fnString.append(")))\n");

    StringBuilder applyString = new StringBuilder();

    // Generate the actual group call
    applyString.append(inVar.emit());
    // Generate the Group By section
    applyString.append(generateGroupByCall(inputColumnNames, group));
    // Add the columns from the apply
    // Generate the apply call
    applyString.append(".apply(").append(funcVar.getName()).append(")");
    return new Pair<>(new Expr.Raw(applyString.toString()), new Op.Code(fnString.toString()));
  }

  /**
   * Helper function to determine the name of the function and whether or not it is a method.
   *
   * @param a Aggregation call that needs a function.
   * @param isGroupbyCall Is the being directly used inside a groupby agg?
   * @return Pair with the name of the call and whether or not it is a method.
   */
  private static Pair<String, Boolean> getAggFuncInfo(AggregateCall a, boolean isGroupbyCall) {
    SqlKind kind = a.getAggregation().getKind();
    String name = a.getAggregation().getName();
    if (kind == SqlKind.COUNT) {
      return getCountCall(a, isGroupbyCall);
    } else if (equivalentPandasMethodMap.containsKey(kind)) {
      return new Pair<>(equivalentPandasMethodMap.get(kind), true);
    } else if (equivalentNumpyFuncMap.containsKey(kind)) {
      return new Pair<>(equivalentNumpyFuncMap.get(kind), false);
    } else if (equivalentPandasNameMethodMap.containsKey(name)) {
      return new Pair<>(equivalentPandasNameMethodMap.get(name), true);
    } else if (equivalentNumpyFuncNameMap.containsKey(name)) {
      return new Pair<>(equivalentNumpyFuncNameMap.get(name), false);
    } else if (equivalentHelperFnMap.containsKey(name)) {
      return new Pair<>(equivalentHelperFnMap.get(name), false);
    } else {
      throw new BodoSQLCodegenException(
          "Unsupported Aggregate Function, "
              + a.getAggregation().toString()
              + " specified in query.");
    }
  }

  public static Expr concatDataFrames(List<String> dfNames) {
    StringBuilder concatString = new StringBuilder("pd.concat([");
    for (int i = 0; i < dfNames.size(); i++) {
      concatString.append(dfNames.get(i)).append(", ");
    }

    // We put ignore_index as True, since we don't care about the index in BodoSQL, and this results
    // in
    // faster runtime performance.
    return new Expr.Raw(concatString.append("], ignore_index=True)").toString());
  }
}
