package com.bodosql.calcite.application.BodoSQLCodeGen;

import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.getAscendingExpr;
import static com.bodosql.calcite.application.BodoSQLCodeGen.SortCodeGen.getNAPositionStringLiteral;
import static com.bodosql.calcite.application.utils.AggHelpers.generateGroupByCall;
import static com.bodosql.calcite.application.utils.AggHelpers.getCountCall;
import static com.bodosql.calcite.application.utils.AggHelpers.getDummyColName;

import com.bodosql.calcite.application.BodoCodeGenVisitor;
import com.bodosql.calcite.application.BodoSQLCodegenException;
import com.bodosql.calcite.application.utils.Utils;
import com.bodosql.calcite.ir.Expr;
import com.bodosql.calcite.ir.Op;
import com.bodosql.calcite.ir.Variable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.apache.calcite.rel.RelFieldCollation;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.util.ImmutableBitSet;
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

  // Functions which are supported exclusively through
  // bodo.utils.utils.ExtendedNamedAgg
  // For context,
  // Most aggregations are handled via pd.NamedAgg, which allows for specifying one input
  // column.
  // However, several aggregations require additional arguments beyond just a single operating
  // column.
  // For these aggregations, we use bodo.utils.utils.ExtendedNamedAgg, which takes an additional
  // tuple
  // of arguments
  static HashMap<String, String> equivalentExtendedNamedAggAggregates;

  // Maps a given SqlKind to its name in the supportedAggFuncs list
  static HashMap<SqlKind, String> kindToSupportAggFuncNameMap;
  // Maps a given SqlKind to its name in the supportedAggFuncs list
  static HashMap<String, String> nameToSupportAggFuncNameMap;

  static {
    equivalentPandasMethodMap = new HashMap<>();
    equivalentNumpyFuncMap = new HashMap<>();
    equivalentPandasNameMethodMap = new HashMap<>();
    equivalentNumpyFuncNameMap = new HashMap<>();
    equivalentHelperFnMap = new HashMap<>();
    equivalentExtendedNamedAggAggregates = new HashMap<>();
    kindToSupportAggFuncNameMap = new HashMap<>();
    nameToSupportAggFuncNameMap = new HashMap<>();

    kindToSupportAggFuncNameMap.put(SqlKind.STDDEV_POP, "std_pop");
    kindToSupportAggFuncNameMap.put(SqlKind.VAR_POP, "var_pop");
    kindToSupportAggFuncNameMap.put(SqlKind.ANY_VALUE, "first");
    kindToSupportAggFuncNameMap.put(SqlKind.SINGLE_VALUE, "first");
    nameToSupportAggFuncNameMap.put("VARIANCE_POP", "var_pop");

    equivalentPandasMethodMap.put(SqlKind.SUM, "sum");
    equivalentPandasMethodMap.put(SqlKind.SUM0, "sum");
    equivalentPandasMethodMap.put(SqlKind.MIN, "min");
    equivalentPandasMethodMap.put(SqlKind.MAX, "max");
    equivalentPandasMethodMap.put(SqlKind.AVG, "mean");
    equivalentPandasMethodMap.put(SqlKind.MEDIAN, "median");
    equivalentPandasMethodMap.put(SqlKind.STDDEV_SAMP, "std");
    equivalentPandasMethodMap.put(SqlKind.VAR_SAMP, "var");

    equivalentNumpyFuncMap.put(SqlKind.VAR_POP, "np.var");
    equivalentNumpyFuncMap.put(SqlKind.STDDEV_POP, "np.std");

    equivalentPandasNameMethodMap.put("COUNT_IF", "count_if");
    equivalentPandasNameMethodMap.put("KURTOSIS", "kurtosis");
    equivalentPandasNameMethodMap.put("SKEW", "skew");
    equivalentPandasNameMethodMap.put("VARIANCE_SAMP", "var");

    equivalentNumpyFuncNameMap.put("VARIANCE_POP", "np.var");

    equivalentPandasNameMethodMap.put("MODE", "mode");
    equivalentPandasNameMethodMap.put("KURTOSIS", "kurtosis");
    equivalentPandasNameMethodMap.put("SKEW", "skew");
    equivalentHelperFnMap.put("BOOLOR_AGG", "boolor_agg");
    equivalentHelperFnMap.put("BOOLAND_AGG", "booland_agg");
    equivalentHelperFnMap.put("BOOLXOR_AGG", "boolxor_agg");
    equivalentHelperFnMap.put("BITOR_AGG", "bitor_agg");
    equivalentHelperFnMap.put("BITAND_AGG", "bitand_agg");
    equivalentHelperFnMap.put("BITXOR_AGG", "bitxor_agg");
    equivalentHelperFnMap.put("ANY_VALUE", "anyvalue_agg");

    // Calcite's SINGLE_VALUE returns input if it has only one value, otherwise raises an error
    // https://github.com/apache/calcite/blob/f14cf4c32b9079984a988bbad40230aa6a59b127/core/src/main/java/org/apache/calcite/sql/fun/SqlSingleValueAggFunction.java#L36
    equivalentHelperFnMap.put(
        "SINGLE_VALUE", "bodo.libs.bodosql_array_kernels.ensure_single_value");
    equivalentHelperFnMap.put("APPROX_PERCENTILE", "approx_percentile");

    equivalentExtendedNamedAggAggregates.put("LISTAGG", "listagg");
    equivalentExtendedNamedAggAggregates.put("ARRAY_AGG", "array_agg");
    equivalentExtendedNamedAggAggregates.put("ARRAY_UNIQUE_AGG", "array_unique_agg");
    equivalentExtendedNamedAggAggregates.put("PERCENTILE_CONT", "percentile_cont");
    equivalentExtendedNamedAggAggregates.put("PERCENTILE_DISC", "percentile_disc");
    equivalentExtendedNamedAggAggregates.put("OBJECT_AGG", "object_agg");
  }

  /**
   * Helper function that handles code generation for aggregateCall non grouping call to listagg.
   *
   * @param inVar Input dataframe
   * @param inputColumnNames Column names of the input dataframe
   * @param aggregateCall The listagg aggregate call
   * @return
   */
  private static Expr genNonGroupedListaggCall(
      Variable inVar, List<String> inputColumnNames, AggregateCall aggregateCall) {
    assert aggregateCall.getAggregation().getKind() == SqlKind.LISTAGG
        : "Internal error in genNonGroupedListaggCall: input aggregation is not listagg";

    List<Expr.StringLiteral> orderbyList = new ArrayList<>();
    List<Expr.BooleanLiteral> ascendingList = new ArrayList<>();
    List<Expr.StringLiteral> nullDirList = new ArrayList<>();
    if (aggregateCall.collation != null) {
      // The collation is where the WITHIN GROUP orderby clause information gets stored.
      // If the collation exists, populate the relevant fields

      for (int j = 0; j < aggregateCall.collation.getFieldCollations().size(); j++) {
        RelFieldCollation curCollation = aggregateCall.collation.getFieldCollations().get(j);
        orderbyList.add(new Expr.StringLiteral(inputColumnNames.get(curCollation.getFieldIndex())));
        ascendingList.add(getAscendingExpr(curCollation.direction));
        nullDirList.add(getNAPositionStringLiteral(curCollation.nullDirection));
      }
    }

    assert (aggregateCall.getArgList().size() == 2)
        : "Internal error in generateAggCodeNoGroupBy: listagg must have 2 arguments";

    // The separator is always coerced to aggregateCall column by calcite. Since the function
    // expects aggregateCall scalar,
    // We convert it back to scalar by simply taking the 0'th element of the array.
    Expr sepExpr =
        new Expr.Index(
            new Expr.Call(
                "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
                inVar,
                new Expr.IntegerLiteral(aggregateCall.getArgList().get(1))),
            new Expr.IntegerLiteral(0));

    String aggCol = inputColumnNames.get(aggregateCall.getArgList().get(0));

    List<Expr> argsList =
        List.of(
            inVar,
            new Expr.StringLiteral(aggCol),
            new Expr.Tuple(orderbyList),
            new Expr.Tuple(ascendingList),
            new Expr.Tuple(nullDirList),
            sepExpr);

    String fn_name = "bodo.libs.bodosql_listagg.bodosql_listagg";
    return new Expr.Call(fn_name, argsList);
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
   * @param pdVisitorClass The PandasCodeGenVisitor used to lower globals.
   * @return The code generated for the aggregation.
   */
  public static Expr generateAggCodeNoGroupBy(
      Variable inVar,
      List<String> inputColumnNames,
      List<AggregateCall> aggCallList,
      List<String> aggCallNames,
      boolean distOutput,
      BodoCodeGenVisitor pdVisitorClass) {
    // Generates code like: pd.DataFrame({"sum(A)": [test_df1["A"].sum()], "mean(B)":
    // [test_df1["A"].mean()]})
    // Generate any filters. This is done on a separate line for simpler
    // code in case the series is empty.

    List<Expr> aggVars = new ArrayList<Expr>();

    for (int i = 0; i < aggCallList.size(); i++) {

      AggregateCall a = aggCallList.get(i);

      Expr aggExpr = inVar;

      // If doing an aggregation besides COUNT(*), extract the desired aggregation column
      if (!(a.getAggregation().getKind() == SqlKind.COUNT && a.getArgList().isEmpty())) {
        Expr aggCol = new Expr.IntegerLiteral(a.getArgList().get(0));
        aggExpr =
            new Expr.Call(
                "bodo.hiframes.pd_dataframe_ext.get_dataframe_data", List.of(aggExpr, aggCol));
      }

      // If necessary, apply a filter to the input
      if (a.filterArg != -1) {
        Expr filterCol =
            new Expr.Call(
                "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
                List.of(inVar, new Expr.IntegerLiteral(a.filterArg)));
        aggExpr = new Expr.Index(aggExpr, filterCol);
      }

      // Keep a copy of the expression at this point to use for length checking
      Expr filteredExpr = aggExpr;

      Pair<String, Boolean> funcInfo = getAggFuncInfo(a, false);
      String aggFunc = funcInfo.getKey();
      boolean isMethod = funcInfo.getValue();

      // If the aggregation function is ANY_VALUE, manually alter syntax
      // to use brackets
      if (aggFunc.equals("anyvalue_agg")) {
        aggExpr = new Expr.Call("bodo.libs.array_kernels.anyvalue_agg", aggExpr);
      } else if (aggFunc.equals("np.var")) {
        kotlin.Pair<String, Expr> ddofKwarg = new kotlin.Pair("ddof", Expr.Companion.getZero());
        aggExpr = new Expr.Call("pd.Series", aggExpr);
        aggExpr = new Expr.Method(aggExpr, "var", List.of(), List.of(ddofKwarg));
      } else if (aggFunc.equals("np.std")) {
        kotlin.Pair<String, Expr> ddofKwarg = new kotlin.Pair("ddof", Expr.Companion.getZero());
        aggExpr = new Expr.Call("pd.Series", aggExpr);
        aggExpr = new Expr.Method(aggExpr, "std", List.of(), List.of(ddofKwarg));
      } else if (aggFunc.equals("count_if")) {
        aggExpr = new Expr.Call("pd.Series", aggExpr);
        aggExpr = new Expr.Method(aggExpr, "sum", List.of(), List.of());
      } else if (aggFunc.equals("boolor_agg")
          || aggFunc.equals("booland_agg")
          || aggFunc.equals("boolxor_agg")
          || aggFunc.equals("bitor_agg")
          || aggFunc.equals("bitand_agg")
          || aggFunc.equals("bitxor_agg")) {
        aggExpr = new Expr.Call("bodo.libs.array_kernels." + aggFunc, aggExpr);
      } else if (aggFunc.equals("array_agg")
          || aggFunc.equals("array_unique_agg")
          || aggFunc.equals("object_agg")) {
        throw new BodoSQLCodegenException(aggFunc + " not supported without a GROUP BY clause");
      } else if (aggFunc.equals("percentile_cont") || aggFunc.equals("percentile_disc")) {
        if (a.collation == null) {
          throw new BodoSQLCodegenException(a.getName() + " requires a WITHIN GROUP term");
        }
        if (a.collation.getFieldCollations().size() > 1) {
          throw new BodoSQLCodegenException(
              a.getName() + " requires the terms to be ordered by a single column");
        }
        Expr quantileScalar = new Expr.Index(aggExpr, Expr.Companion.getZero());
        RelFieldCollation curCollation = a.collation.getFieldCollations().get(0);
        aggExpr =
            new Expr.Call(
                "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
                List.of(inVar, new Expr.IntegerLiteral(curCollation.getFieldIndex())));
        aggExpr = new Expr.Call("bodo.libs.array_kernels." + aggFunc, aggExpr, quantileScalar);

      } else if (aggFunc.equals("approx_percentile")) {
        Utils.assertWithErrMsg(
            a.getArgList().size() == 2, "APPROX_PERCENTILE requires two arguments");
        // Currently, the scalar float argument is converted into a column. To
        // access the quantile value, extract the first row.
        Expr quantileColumn =
            new Expr.Call(
                "bodo.hiframes.pd_dataframe_ext.get_dataframe_data",
                List.of(inVar, new Expr.IntegerLiteral(a.getArgList().get(1))));
        Expr quantileScalar = new Expr.Index(quantileColumn, Expr.Companion.getZero());
        // TODO: confirm that the second argument is a float
        Utils.assertWithErrMsg(
            true, "The second argument to APPROX_PERCENTILE must be a scalar float");
        // TODO: confirm that the second argument is between zero and one
        Utils.assertWithErrMsg(
            true, "The second argument to APPROX_PERCENTILE must be between 0.0 and 1.0");
        aggExpr =
            new Expr.Call("bodo.libs.array_kernels.approx_percentile", aggExpr, quantileScalar);
      } else if (isMethod) {
        aggExpr = new Expr.Call("pd.Series", aggExpr);
        aggExpr = new Expr.Method(aggExpr, aggFunc, List.of(), List.of());
      } else if (aggFunc.equals("listagg")) {
        aggExpr = genNonGroupedListaggCall(inVar, inputColumnNames, a);
      } else {
        aggExpr = new Expr.Call(aggFunc, aggExpr);
      }

      // If doing an aggregation besides one of the count families, coerce the output to null
      // if the input is empty.
      // We need an optional type in case the series is empty.
      if (a.getAggregation().getKind() != SqlKind.COUNT
          && a.getAggregation().getKind() != SqlKind.SUM0
          && a.getAggregation().getKind() != SqlKind.SINGLE_VALUE
          && !aggFunc.equals("count_if")) {
        Expr lengthFlag =
            new Expr.Binary(">", new Expr.Call("len", filteredExpr), Expr.Companion.getZero());
        aggExpr = new Expr.Call("bodosql.libs.null_handling.null_if_not_flag", aggExpr, lengthFlag);
      }

      // Force the result to be an array that is replicated unless the output can be distributed
      // (e.g. if it is inside a grouping set). Each path will always take exactly one of the
      // routes.
      if (distOutput) {
        List<kotlin.Pair<String, Expr>> namedParams =
            List.of(new kotlin.Pair("scalar_to_arr_len", new Expr.IntegerLiteral(1)));
        aggExpr =
            new Expr.Call("bodo.utils.conversion.coerce_to_array", List.of(aggExpr), namedParams);
      } else {
        aggExpr =
            new Expr.Call(
                "bodo.utils.conversion.make_replicated_array", aggExpr, Expr.Companion.getOne());
      }

      // Store the result in a variable
      Variable arrayVar = pdVisitorClass.storeAsArrayVariable(aggExpr);
      aggVars.add(arrayVar);
    }

    Expr.Tuple valuesTuple = new Expr.Tuple(aggVars);

    // Generate the index
    Expr.Call indexCall;
    if (distOutput) {
      indexCall =
          new Expr.Call(
              "bodo.hiframes.pd_index_ext.init_range_index",
              List.of(
                  Expr.Companion.getZero(),
                  Expr.Companion.getOne(),
                  Expr.Companion.getOne(),
                  Expr.None.INSTANCE));
    } else {
      // Aggregation without groupby should always have one element.
      // To force this value to be replicated (the correct output),
      // we use coerce to array.
      indexCall =
          new Expr.Call(
              "bodo.hiframes.pd_index_ext.init_numeric_index",
              new Expr.Call(
                  "bodo.utils.conversion.coerce_to_array",
                  new Expr.List(Expr.Companion.getZero())));
    }
    // Generate the column names global
    List<Expr.StringLiteral> colNamesLiteral = Utils.stringsToStringLiterals(aggCallNames);
    Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
    Variable colNamesMeta = pdVisitorClass.lowerAsColNamesMetaType(colNamesTuple);
    Expr dfExpr =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(valuesTuple, indexCall, colNamesMeta));
    return dfExpr;
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
      aggString.append(".drop_duplicates(ignore_index=True)");
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

    // Generate the Group By section
    aggString.append(generateGroupByCall(inVar, inputColumnNames, group).emit());

    /*
     * create the corresponding aggregation string using named aggregate syntax with tuples.
     * e.g. .agg(out1=pd.NamedAgg(column="in1", aggfunc="sum"), out2=pd.NamedAgg(column="in2", aggfunc="sum"),
     * out3=pd.NamedAgg(column="in1", aggfunc="mean"))
     */
    aggString.append(".agg(");
    HashMap<String, String> renamedAggColumns = new HashMap<>();
    for (int i = 0; i < aggCallList.size(); i++) {
      AggregateCall a = aggCallList.get(i);
      String aggCol = Utils.getInputColumn(inputColumnNames, a, group);
      String outputCol = aggCallNames.get(i);
      // Generate a dummy column to prevent syntax issues with names that aren't
      // supported by Pandas NamedAgg. If the name is a valid Python identifier
      // we don't need a rename
      String tempName = outputCol;
      if (!Utils.isValidPythonIdentifier(outputCol)) {
        tempName = getDummyColName(i);
        renamedAggColumns.put(tempName, outputCol);
      }

      Pair<String, Boolean> funcInfo = getAggFuncInfo(a, true);
      String aggFunc = funcInfo.getKey();
      // When inside of a Group By, use .iloc[0] instead of .head(1)[0]
      if (aggFunc == "anyvalue_agg") {
        aggFunc = "first";
      }
      if (!(aggFunc.equals("np.var") || aggFunc.equals("np.std"))) {
        aggFunc = Utils.makeQuoted(aggFunc);
      }
      if (aggFunc.equals("approx_percentile")) {
        throw new BodoSQLCodegenException("APPROX_PERCENTILE not supported with Group By yet");
      }

      // Most aggregations are handled via pd.NamedAgg, some are handled by ExtendedNamedAgg.
      // See the comment for equivalentExtendedNamedAggAggregates for more info.
      if (!equivalentExtendedNamedAggAggregates.containsKey(a.getAggregation().getName())) {
        aggString
            .append(tempName)
            .append("=pd.NamedAgg(column=")
            .append(Utils.makeQuoted(aggCol))
            .append(", aggfunc=")
            .append(aggFunc)
            .append("),");
      } else if (aggFunc.equals("\"array_unique_agg\"")) {
        List<Expr> additionalArgsList = new ArrayList<>();
        List<Expr.StringLiteral> orderbyList = new ArrayList<>();
        List<Expr.BooleanLiteral> ascendingList = new ArrayList<>();
        List<Expr.StringLiteral> nullDirList = new ArrayList<>();
        orderbyList.add(new Expr.StringLiteral(inputColumnNames.get(a.getArgList().get(0))));
        ascendingList.add(getAscendingExpr(RelFieldCollation.Direction.ASCENDING));
        nullDirList.add(getNAPositionStringLiteral(RelFieldCollation.NullDirection.LAST));
        additionalArgsList.add(new Expr.Tuple(orderbyList));
        additionalArgsList.add(new Expr.Tuple(ascendingList));
        additionalArgsList.add(new Expr.Tuple(nullDirList));
        aggString
            .append(tempName)
            .append("=bodo.utils.utils.ExtendedNamedAgg(column=")
            .append(Utils.makeQuoted(aggCol))
            .append(", aggfunc=")
            .append("\"array_agg_distinct\"")
            .append(", additional_args=")
            .append((new Expr.Tuple(additionalArgsList)).emit())
            .append("),");
      } else {
        Expr.Tuple additionalArgs = getAdditionalArgs(a, inputColumnNames);
        // Calcite will think that the percentile amount is the aggregation column and
        // that the aggregation column is the extra argument, so we need to switch them.
        if (aggFunc.equals("\"percentile_cont\"") || aggFunc.equals("\"percentile_disc\"")) {
          Expr percentileCol = additionalArgs.getArgs().get(0);
          additionalArgs = new Expr.Tuple(new Expr.StringLiteral(aggCol));
          aggCol = percentileCol.emit();
        }
        // ARRAY_AGG(DISTINCT xxx) has its own ftype
        if (aggFunc.equals("\"array_agg\"") && a.isDistinct()) {
          aggFunc = "\"array_agg_distinct\"";
        }
        // Calcite will switch the locations of the key and value columns, so we need to switch
        // them.
        if (aggFunc.equals("\"object_agg\"")) {
          Expr valueCol = additionalArgs.getArgs().get(0);
          additionalArgs = new Expr.Tuple(new Expr.StringLiteral(aggCol));
          aggCol = valueCol.emit();
        }
        aggString
            .append(tempName)
            .append("=bodo.utils.utils.ExtendedNamedAgg(column=")
            .append(Utils.makeQuoted(aggCol))
            .append(", aggfunc=")
            .append(aggFunc)
            .append(", additional_args=")
            .append(additionalArgs.emit())
            .append("),");
      }
    }
    aggString.append(")");
    if (renamedAggColumns.size() > 0) {
      aggString.append(".rename(columns=");
      aggString.append(Utils.renameColumns(renamedAggColumns));
      aggString.append(", copy=False)");
    }
    return new Expr.Raw(aggString.toString());
  }

  /**
   * Helper function that handles creating the additional_args tuple to pass to
   * bodo.utils.utils.ExtendedNamedAgg.
   *
   * @param agg Aggregate call in question (must be in equivalentExtendedNamedAggAggregates)
   * @param inputColumnNames The column names to the input dataframe.
   * @return An expr to be used for the additional_args keyword argument of
   *     bodo.utils.utils.ExtendedNamedAgg.
   */
  static Expr.Tuple getAdditionalArgs(AggregateCall agg, List<String> inputColumnNames) {
    SqlKind kind = agg.getAggregation().getKind();
    assert equivalentExtendedNamedAggAggregates.containsKey(agg.getAggregation().getName());
    List<Integer> argsList = agg.getArgList();
    List<Expr> additionalArgsList = new ArrayList<>();
    RelFieldCollation curCollation;
    switch (kind) {
      case PERCENTILE_DISC:
      case PERCENTILE_CONT:
        if (agg.collation == null) {
          throw new BodoSQLCodegenException(agg.getName() + " requires a WITHIN GROUP term");
        }
        if (agg.collation.getFieldCollations().size() > 1) {
          throw new BodoSQLCodegenException(
              agg.getName() + " requires the terms to be ordered by a single column");
        }
        curCollation = agg.collation.getFieldCollations().get(0);
        additionalArgsList.add(
            new Expr.StringLiteral(inputColumnNames.get(curCollation.getFieldIndex())));
        return new Expr.Tuple(additionalArgsList);
        // TODO: try to fuse logic with LISTAGG
      case ARRAY_AGG:
        assert argsList.size() == 1;
        if (agg.collation != null) {
          // If the collation exists, populate the relevant fields

          List<Expr.StringLiteral> orderbyList = new ArrayList<>();
          List<Expr.BooleanLiteral> ascendingList = new ArrayList<>();
          List<Expr.StringLiteral> nullDirList = new ArrayList<>();

          // If DISTINCT is provided, then only a single ordering column can be provided,
          // and it must be the same as the data column.
          if (agg.isDistinct()) {
            // If there is no ordering, insert one that is the same as the input column.
            if (agg.collation.getFieldCollations().size() == 0) {
              orderbyList.add(
                  new Expr.StringLiteral(inputColumnNames.get(agg.getArgList().get(0))));
              ascendingList.add(getAscendingExpr(RelFieldCollation.Direction.ASCENDING));
              nullDirList.add(getNAPositionStringLiteral(RelFieldCollation.NullDirection.LAST));
            }
            // Otherwise, verify that it matches the input column.
            else if ((agg.collation.getFieldCollations().size() > 1)
                || (agg.collation.getFieldCollations().get(0).getFieldIndex()
                    != agg.getArgList().get(0))) {
              throw new BodoSQLCodegenException(
                  "ARRAY_AGG with DISTINCT keyword requires the WITHIN GROUP clause (if provided)"
                      + " to be the same as the aggregated data.");
            }
          }

          for (int i = 0; i < agg.collation.getFieldCollations().size(); i++) {
            curCollation = agg.collation.getFieldCollations().get(i);
            orderbyList.add(
                new Expr.StringLiteral(inputColumnNames.get(curCollation.getFieldIndex())));
            ascendingList.add(getAscendingExpr(curCollation.direction));
            nullDirList.add(getNAPositionStringLiteral(curCollation.nullDirection));
          }
          additionalArgsList.add(new Expr.Tuple(orderbyList));
          additionalArgsList.add(new Expr.Tuple(ascendingList));
          additionalArgsList.add(new Expr.Tuple(nullDirList));
        } else {
          // Otherwise, just add empty tuples
          for (int i = 0; i < 3; i++) {
            additionalArgsList.add(new Expr.Tuple());
          }
        }

        return new Expr.Tuple(additionalArgsList);
      case LISTAGG:
        if (argsList.size() == 1) {
          throw new BodoSQLCodegenException(
              "Internal error in getAdditionalArgs: Listagg should be unconditionally converted to"
                  + " two argument form in ListAggOptionalReplaceRule.java");
        } else {
          assert argsList.size() == 2;
          Expr.StringLiteral colName =
              new Expr.StringLiteral(inputColumnNames.get(argsList.get(1)));
          additionalArgsList.add(colName);
        }

        if (agg.collation != null) {
          // If the collation exists, populate the relevant fields

          List<Expr.StringLiteral> orderbyList = new ArrayList<>();
          List<Expr.BooleanLiteral> ascendingList = new ArrayList<>();
          List<Expr.StringLiteral> nullDirList = new ArrayList<>();

          for (int i = 0; i < agg.collation.getFieldCollations().size(); i++) {
            curCollation = agg.collation.getFieldCollations().get(i);
            orderbyList.add(
                new Expr.StringLiteral(inputColumnNames.get(curCollation.getFieldIndex())));
            ascendingList.add(getAscendingExpr(curCollation.direction));
            nullDirList.add(getNAPositionStringLiteral(curCollation.nullDirection));
          }
          additionalArgsList.add(new Expr.Tuple(orderbyList));
          additionalArgsList.add(new Expr.Tuple(ascendingList));
          additionalArgsList.add(new Expr.Tuple(nullDirList));
        } else {
          // Otherwise, just add empty tuples
          for (int i = 0; i < 3; i++) {
            additionalArgsList.add(new Expr.Tuple());
          }
        }

        return new Expr.Tuple(additionalArgsList);
      case OTHER_FUNCTION:
        String name = agg.getAggregation().getName();
        switch (name) {
          case "OBJECT_AGG":
            Integer keyIdx = argsList.get(1);
            additionalArgsList.add(new Expr.StringLiteral(inputColumnNames.get(keyIdx)));
            return new Expr.Tuple(additionalArgsList);
          default:
            throw new BodoSQLCodegenException(
                "Internal error in getAdditionalArgs: " + (name) + " not handled");
        }
      default:
        throw new BodoSQLCodegenException(
            "Internal error in getAdditionalArgs: " + kind.name() + " not handled");
    }
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
   * @param pdVisitorClass The PandasCodeGenVisitor used to lower globals.
   * @return A pair of the code expression generated for the aggregation, and the function
   *     definition that is used in the groupby apply.
   */
  public static Pair<Expr, Op> generateApplyCodeWithGroupBy(
      Variable inVar,
      List<String> inputColumnNames,
      List<Integer> group,
      List<AggregateCall> aggCallList,
      List<String> aggCallNames,
      Variable funcVar,
      BodoCodeGenVisitor pdVisitorClass) {
    StringBuilder fnString = new StringBuilder();

    final String indent = Utils.getBodoIndent();
    final String funcIndent = indent + indent;

    /*
     * First we generate the closure that will be used in the apply.
     */
    fnString.append(indent).append("def ").append(funcVar.getName()).append("(df):\n");

    ArrayList<String> seriesArgs = new ArrayList<>();
    ArrayList<String> indexNames = new ArrayList<>();

    for (int i : group) {
      Expr columnName = new Expr.StringLiteral(inputColumnNames.get(i));
      Expr groupCol = new Expr.Index(new Expr.Raw("df"), columnName);
      Expr groupValScalar =
          new Expr.Index(new Expr.Attribute(groupCol, "iloc"), Expr.Companion.getZero());
      seriesArgs.add(groupValScalar.emit());
    }

    for (int i = 0; i < aggCallList.size(); i++) {
      AggregateCall a = aggCallList.get(i);
      // Get the input column
      String aggCol = Utils.getInputColumn(inputColumnNames, a, group);

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
      fnString.append("df[").append(Utils.makeQuoted(aggCol)).append("]");
      if (filterCol.length() > 0) {
        fnString.append("[df[").append(Utils.makeQuoted(filterCol)).append("]]");
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
      if (aggFunc.equals("anyvalue_agg")) {
        // If the aggregation function is ANY_VALUE, use the bodo kernel
        fnString.append("bodo.libs.array_kernels.anyvalue_agg(");
        fnString.append(seriesVar);
        fnString.append(")");
      } else if (aggFunc.equals("var_pop")) {
        fnString.append(seriesVar);
        fnString.append(".var(ddof=0)");
      } else if (aggFunc.equals("count_if")) {
        fnString.append(seriesVar);
        fnString.append(".sum()");
      } else if (aggFunc.equals("listagg")) {
        Variable dfVar = new Variable("df");
        fnString.append(genNonGroupedListaggCall(dfVar, inputColumnNames, a).emit());
      } else if (aggFunc.equals("percentile_cont")
          || aggFunc.equals("percentile_disc")
          || aggFunc.equals("array_agg")) {
        throw new BodoSQLCodegenException(aggFunc + " not supported with a filter clause");
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
    // Generate the output

    List<Expr> aggVars = new ArrayList<Expr>();
    for (String arg : seriesArgs) {
      Expr varSingletonList = new Expr.List(new Expr.Raw(arg));
      aggVars.add(new Expr.Call("bodo.utils.conversion.coerce_to_array", varSingletonList));
    }

    Expr.Tuple valuesTuple = new Expr.Tuple(aggVars);

    // Generate the index
    Expr.Call indexCall =
        new Expr.Call(
            "bodo.hiframes.pd_index_ext.init_range_index",
            List.of(
                Expr.Companion.getZero(),
                Expr.Companion.getOne(),
                Expr.Companion.getOne(),
                Expr.None.INSTANCE));
    // Generate the column names global
    List<Expr> colNamesLiteral = new ArrayList<Expr>();
    for (int i : group) {
      String columnName = inputColumnNames.get(i);
      colNamesLiteral.add(new Expr.StringLiteral(columnName));
    }
    for (String aggCallName : aggCallNames) {
      colNamesLiteral.add(new Expr.StringLiteral(aggCallName));
    }
    Expr.Tuple colNamesTuple = new Expr.Tuple(colNamesLiteral);
    Variable colNamesMeta = pdVisitorClass.lowerAsColNamesMetaType(colNamesTuple);
    Expr dfExpr =
        new Expr.Call(
            "bodo.hiframes.pd_dataframe_ext.init_dataframe",
            List.of(valuesTuple, indexCall, colNamesMeta));
    fnString.append(funcIndent).append("return ").append(dfExpr.emit()).append("\n");

    StringBuilder applyString = new StringBuilder();

    // Generate the actual group call
    applyString.append(generateGroupByCall(inVar, inputColumnNames, group).emit());
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
    } else if (equivalentExtendedNamedAggAggregates.containsKey(name)) {
      return new Pair<>(equivalentExtendedNamedAggAggregates.get(name), false);
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

  /**
   * Generate a list that contains the indices of the key columns used for aggregate.
   *
   * @param groupSet The join equality information.
   * @return A List of Expr.IntegerLiteral that contains the selected indices.
   */
  public static List<Expr.IntegerLiteral> getStreamingGroupByKeyIndices(ImmutableBitSet groupSet) {
    List<Expr.IntegerLiteral> indices = new ArrayList<>();
    for (int i = 0; i < groupSet.size(); i++) {
      if (groupSet.get(i)) {
        indices.add(new Expr.IntegerLiteral(i));
      }
    }
    return indices;
  }

  /**
   * Generate two lists of integers, the first one is the offset list of each aggregate call, the
   * second one is the list of indices of all aggregate calls.
   *
   * @param aggCalls The list of aggregate calls.
   * @param visitor The visitor used for lowering global variables.
   * @param firstKeyColumnIndex An expr that evaluates to the index of the first key column (Used
   *     for COUNT(*))
   * @return A pair of variables the contains these two lists.
   */
  public static Pair<Variable, Variable> getStreamingGroupByOffsetAndCols(
      List<AggregateCall> aggCalls,
      BodoCodeGenVisitor visitor,
      Expr.IntegerLiteral firstKeyColumnIndex) {
    List<Expr.IntegerLiteral> offsets = new ArrayList<>();
    offsets.add(new Expr.IntegerLiteral(0));
    List<Expr.IntegerLiteral> cols = new ArrayList<>();
    int length = 0;
    for (int i = 0; i < aggCalls.size(); i++) {
      List<Integer> argList = aggCalls.get(i).getArgList();
      if (argList.size() > 0) {
        for (int j = 0; j < argList.size(); j++) {
          cols.add(new Expr.IntegerLiteral(argList.get(j)));
        }
        length += argList.size();
      } else {
        // Count(*) case. In this case, the group by code still expects one input column.
        // Therefore, we elect to pass the index of the first key column
        if (aggCalls.get(i).getAggregation().getKind() != SqlKind.COUNT) {
          throw new RuntimeException(
              "Internal error in getStreamingGroupByOffsetAndCols: Expect 'COUNT' aggregate call."
                  + " Found "
                  + aggCalls.get(i).getAggregation().getKind().toString()
                  + " instead.");
        }
        cols.add(firstKeyColumnIndex);
        length += 1;
      }
      offsets.add(new Expr.IntegerLiteral(length));
    }
    Variable offsetVar = visitor.lowerAsMetaType(new Expr.Tuple(offsets));
    Variable colsVar = visitor.lowerAsMetaType(new Expr.Tuple(cols));
    return new Pair<>(offsetVar, colsVar);
  }

  /**
   * Generate a list of integer that represents the aggregate function based on the order in
   * Bodo_FTypes::FTypeEnum in _groupby_ftypes.h
   *
   * @param aggCalls The list of aggregate calls.
   * @param visitor The visitor used for lowering global variables.
   * @return A variable that contains this integer list
   */
  public static Variable getStreamingGroupbyFnames(
      List<AggregateCall> aggCalls, BodoCodeGenVisitor visitor) {
    List<Expr.StringLiteral> fnames = new ArrayList<>();
    for (int i = 0; i < aggCalls.size(); i++) {
      AggregateCall curAggCall = aggCalls.get(i);
      SqlKind kind = curAggCall.getAggregation().getKind();
      String name = curAggCall.getAggregation().getName();

      if (kind == SqlKind.COUNT) {
        name = getCountCall(curAggCall, true).left;
      } else if (kindToSupportAggFuncNameMap.containsKey(kind)) {
        name = kindToSupportAggFuncNameMap.get(kind);
      } else if (nameToSupportAggFuncNameMap.containsKey(name)) {
        name = nameToSupportAggFuncNameMap.get(name);
      } else if (equivalentPandasMethodMap.containsKey(kind)) {
        name = equivalentPandasMethodMap.get(kind);
      } else if (equivalentNumpyFuncMap.containsKey(kind)) {
        name = equivalentNumpyFuncMap.get(kind);
      } else if (equivalentPandasNameMethodMap.containsKey(name)) {
        name = equivalentPandasNameMethodMap.get(name);
      } else if (equivalentNumpyFuncNameMap.containsKey(name)) {
        name = equivalentNumpyFuncNameMap.get(name);
      } else if (equivalentHelperFnMap.containsKey(name)) {
        name = equivalentHelperFnMap.get(name);
      } else if (equivalentExtendedNamedAggAggregates.containsKey(name)) {
        name = equivalentExtendedNamedAggAggregates.get(name);
      } else {
        name = name.toLowerCase();
      }
      // TODO: [BSE-714] Support ANY_VALUE and SINGLE_VALUE in streaming
      fnames.add(new Expr.StringLiteral(name));
    }
    return visitor.lowerAsMetaType(new Expr.Tuple(fnames));
  }
}
